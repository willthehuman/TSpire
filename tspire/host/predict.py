"""Deterministic next-state predictor.

Given the combat state *before* an action and the action itself, compute the state we
*expect* to see next, using Slay the Spire's rules. The reconciler (tspire.host.reconcile)
then uses this prediction to validate / correct the noisy vision read.

This is intentionally conservative: it only models effects it can compute with confidence
(end-of-turn enemy damage; the hp/block/energy effect of a known played card). Anything it
cannot predict — re-rolled enemy intents, unknown cards, card-draw RNG — is left untouched
so the vision read stays authoritative there. When the action isn't predictable at all it
returns ``None``.

Pure function, no I/O: trivially unit-testable.
"""

from __future__ import annotations

from copy import deepcopy

from tspire.common import protocol
from tspire.common.schema import GameState, Monster, PlayerCombat, ScreenType
from tspire.host import cards

# Base energy a turn starts with. Energy relics (e.g. extra-energy bosses) aren't modeled;
# the reconciler treats energy leniently so a wrong base only ever defers to the vision read.
BASE_ENERGY = 3


def predict(before: GameState | None, command: protocol.Command) -> GameState | None:
    """Return the expected next GameState, or None when the action isn't predictable."""
    if before is None or before.screen_type != ScreenType.COMBAT or before.combat_state is None:
        return None
    if command.verb == protocol.Verb.END:
        return _predict_end(before)
    if command.verb == protocol.Verb.PLAY:
        return _predict_play(before, command.args)
    return None


def _predict_end(before: GameState) -> GameState:
    predicted = deepcopy(before)
    combat = predicted.combat_state
    assert combat is not None
    player = combat.player

    incoming = sum(
        m.intent_damage * max(1, m.intent_hits)
        for m in combat.monsters
        if _alive(m) and m.intent.is_attack and m.intent_damage > 0
    )
    dealt = max(0, incoming - player.block)
    _set_player_hp(predicted, player.current_hp - dealt)

    # Start-of-next-turn resets. Enemy intents re-roll (unknowable) -> left to vision.
    player.block = 0
    player.energy = BASE_ENERGY
    combat.turn += 1
    return predicted


def _predict_play(before: GameState, args: list[str]) -> GameState | None:
    combat_before = before.combat_state
    assert combat_before is not None
    card_index = _int_or_none(args[0]) if args else None
    if card_index is None or not (0 <= card_index < len(combat_before.hand)):
        return None
    card = combat_before.hand[card_index]
    data = cards.lookup(card.name or card.card_id)
    if data is None:
        return None

    target_index = _int_or_none(args[1]) if len(args) > 1 else None
    upgrades = card.upgrades or (1 if "+" in card.name else 0)

    predicted = deepcopy(before)
    combat = predicted.combat_state
    assert combat is not None
    player = combat.player

    if card.cost >= 0:
        player.energy = max(0, player.energy - card.cost)

    block_gain = data.block_for(upgrades)
    if block_gain:
        player.block += block_gain + _power(player, "dexterity")

    damage = data.damage_for(upgrades)
    if damage:
        per_hit = damage + _power(player, "strength")
        targets = _attack_targets(combat.monsters, data.aoe, target_index)
        for monster in targets:
            hit = int(per_hit * 1.5) if _power(monster, "vulnerable") > 0 else per_hit
            damage_left = hit * max(1, data.hits)
            if monster.block > 0:
                blocked = min(monster.block, damage_left)
                monster.block -= blocked
                damage_left -= blocked
            monster.current_hp = max(0, monster.current_hp - damage_left)
            if monster.current_hp == 0:
                monster.is_gone = True

    # The card leaves hand for the discard pile.
    combat.hand.pop(card_index)
    for i, remaining in enumerate(combat.hand):
        remaining.index = i
    combat.discard_pile_count += 1
    return predicted


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _attack_targets(
    monsters: list[Monster], aoe: bool, target_index: int | None
) -> list[Monster]:
    living = [m for m in monsters if _alive(m)]
    if aoe:
        return living
    if target_index is not None:
        return [m for m in living if m.index == target_index]
    # Single-target card with no explicit target: only unambiguous when one enemy remains.
    return living if len(living) == 1 else []


def _alive(monster: Monster) -> bool:
    if monster.is_gone or monster.half_dead:
        return False
    return monster.max_hp > 0 or monster.current_hp > 0


def _power(owner: PlayerCombat | Monster, needle: str) -> int:
    for power in owner.powers:
        if needle in (power.power_id or power.name).lower():
            return power.amount
    return 0


def _set_player_hp(state: GameState, value: int) -> None:
    player = state.combat_state.player  # type: ignore[union-attr]
    clamped = max(0, value)
    if player.max_hp > 0:
        clamped = min(clamped, player.max_hp)
    player.current_hp = clamped
    state.current_hp = clamped  # GameState mirrors the player's hp


def _int_or_none(token: str) -> int | None:
    token = token.strip()
    if token.lstrip("-").isdigit():
        return int(token)
    return None
