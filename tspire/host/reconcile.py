"""Confidence-gated merge of a vision read with a rule-based prediction.

The vision read is fresh truth but noisy; the prediction (tspire.host.predict) is derived
from rules and is right whenever its inputs were right. :func:`reconcile` keeps the vision
value when it's plausible and replaces it with the prediction when it is implausible or
diverges far from what the rules expect (the classic OCR/LLM failure: a dropped or extra
digit). For player stats backed by a fixed screen region, a hard conflict can be broken by
re-reading just that region with the LLM arbiter.

Only numeric combat fields are reconciled (player hp/block/energy, monster hp). Names,
intents, powers and the hand are left to the vision read — the predictor can't reconstruct
them.
"""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Protocol

from tspire.common.schema import GameState, Monster

log = logging.getLogger("tspire.host.reconcile")

# A vision value within TOL of the prediction is treated as agreement (kept as-is).
# A gap beyond DIVERGE is a hard conflict (override / arbitrate).
TOL = 2
DIVERGE = 8


class Arbiter(Protocol):
    """Re-reads a single fixed region with a high-zoom LLM crop. Methods return the
    (current, max) pair, or None when the re-read failed / Ollama is unavailable."""

    def reread_player_hp(self) -> tuple[int, int] | None: ...
    def reread_energy(self) -> tuple[int, int] | None: ...


def reconcile(
    vision: GameState,
    predicted: GameState | None,
    before: GameState | None,
    arbiter: Arbiter | None = None,
    *,
    tol: int = TOL,
    diverge: int = DIVERGE,
    allow_monster_overrides: bool = True,
) -> GameState:
    """Return ``vision`` with implausible combat values corrected toward ``predicted``.

    Mutates and returns the ``vision`` GameState (it's freshly built each read).
    """
    if predicted is None or vision.combat_state is None or predicted.combat_state is None:
        return vision

    vcombat = vision.combat_state
    pcombat = predicted.combat_state
    bcombat = before.combat_state if before is not None else None

    # ---- player hp (the headline case) ----------------------------------
    before_hp = bcombat.player.current_hp if bcombat else 0
    resolved = _resolve_hp(
        vision=vcombat.player.current_hp,
        predicted=pcombat.player.current_hp,
        before=before_hp,
        max_hp=vcombat.player.max_hp,
        tol=tol,
        diverge=diverge,
        arbiter_reread=arbiter.reread_player_hp if arbiter else None,
        label="player hp",
    )
    vcombat.player.current_hp = resolved
    vision.current_hp = resolved

    # ---- player energy --------------------------------------------------
    v_energy = vcombat.player.energy
    p_energy = pcombat.player.energy
    if _implausible_energy(v_energy) or abs(v_energy - p_energy) > diverge:
        reread = arbiter.reread_energy() if arbiter else None
        if reread and reread[0] >= 0:
            vcombat.player.energy = _nearest(reread[0], v_energy, p_energy)
        elif _implausible_energy(v_energy):
            log.debug("reconcile energy: vision=%s implausible -> predicted=%s", v_energy, p_energy)
            vcombat.player.energy = p_energy

    # ---- player block (resets / accrues deterministically) --------------
    if vcombat.player.block < 0 or abs(vcombat.player.block - pcombat.player.block) > diverge:
        log.debug(
            "reconcile block: vision=%s -> predicted=%s",
            vcombat.player.block,
            pcombat.player.block,
        )
        vcombat.player.block = max(0, pcombat.player.block)

    # ---- monster hp (no fixed region -> rule-only, no arbiter) ----------
    before_by_index = {m.index: m for m in bcombat.monsters} if bcombat else {}
    matched_predicted: set[int] = set()
    for monster in vcombat.monsters:
        pred = _match_monster(monster, pcombat.monsters, matched_predicted)
        if pred is None:
            continue
        matched_predicted.add(id(pred))
        monster.index = pred.index
        if not monster.name:
            monster.name = pred.name
        if not monster.monster_id:
            monster.monster_id = pred.monster_id
        if monster.max_hp <= 0:
            monster.max_hp = pred.max_hp
        prev = before_by_index.get(pred.index) or before_by_index.get(monster.index)
        prev_hp = prev.current_hp if prev else pred.current_hp
        increased = prev_hp > 0 and monster.current_hp > prev_hp
        diverged = abs(monster.current_hp - pred.current_hp) > diverge
        if allow_monster_overrides and (increased or diverged):
            log.debug(
                "reconcile monster[%d] hp: vision=%s -> predicted=%s",
                monster.index,
                monster.current_hp,
                pred.current_hp,
            )
            monster.current_hp = pred.current_hp

    for monster in pcombat.monsters:
        if id(monster) in matched_predicted or not _alive(monster):
            continue
        log.debug("reconcile monster[%d]: carrying predicted live monster missing from vision", monster.index)
        vcombat.monsters.append(deepcopy(monster))

    return vision


def _resolve_hp(
    *,
    vision: int,
    predicted: int,
    before: int,
    max_hp: int,
    tol: int,
    diverge: int,
    arbiter_reread,
    label: str,
) -> int:
    # Missing read -> trust the prediction outright.
    if vision <= 0 and predicted > 0:
        return predicted
    # Plausibility bounds: hp can't exceed max, and can't rise (no in-combat heal modeled).
    impossible = (max_hp > 0 and vision > max_hp) or (before > 0 and vision > before)
    if abs(vision - predicted) <= tol and not impossible:
        return vision
    if impossible or abs(vision - predicted) > diverge:
        reread = arbiter_reread() if arbiter_reread else None
        if reread and reread[0] > 0:
            chosen = _nearest(reread[0], vision, predicted)
            log.debug("reconcile %s: arbiter=%s -> %s (vision=%s pred=%s)",
                      label, reread[0], chosen, vision, predicted)
            return chosen
        log.debug("reconcile %s: vision=%s -> predicted=%s", label, vision, predicted)
        return predicted
    # Gray zone (between tol and diverge, still plausible): keep the fresh read.
    return vision


def _implausible_energy(energy: int) -> bool:
    return energy < 0 or energy > 12


def _nearest(reference: int, *candidates: int) -> int:
    return min(candidates, key=lambda c: abs(c - reference))


def _alive(monster: Monster) -> bool:
    if monster.is_gone or monster.half_dead:
        return False
    return monster.max_hp > 0 or monster.current_hp > 0


def _match_monster(
    vision: Monster,
    predicted: list[Monster],
    used: set[int],
) -> Monster | None:
    candidates = [m for m in predicted if id(m) not in used]
    if not candidates:
        return None
    scored = [(_monster_match_score(vision, candidate), candidate) for candidate in candidates]
    score, match = max(scored, key=lambda item: item[0])
    return match if score > 0 else None


def _monster_match_score(vision: Monster, predicted: Monster) -> int:
    score = 0
    if vision.index == predicted.index:
        score += 3
    v_name = (vision.monster_id or vision.name).strip().lower()
    p_name = (predicted.monster_id or predicted.name).strip().lower()
    if v_name and p_name and v_name == p_name:
        score += 8
    if vision.max_hp > 0 and vision.max_hp == predicted.max_hp:
        score += 6
    if vision.current_hp > 0 and predicted.current_hp > 0:
        score += max(0, 5 - min(5, abs(vision.current_hp - predicted.current_hp)))
    return score
