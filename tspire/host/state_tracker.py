"""State reduction for noisy screen reads.

Parsers produce observations. The tracker decides whether an observation is fresh enough to
become commandable state, whether missing fields should be carried as explicitly stale, and
when a pending input prediction can be safely reconciled.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json

from tspire.common import protocol
from tspire.common.schema import GameState, ScreenType
from tspire.host.predict import predict
from tspire.host.reconcile import reconcile
from tspire.host.vision.combat import ParseResult

READ_FRESH = "fresh"
READ_STALE = "stale"
READ_UNCERTAIN = "uncertain"

_MIN_COMBAT_CONFIDENCE = 0.67
_UNKNOWN_TRACKED_FIELDS = (
    "gold",
    "floor",
    "deck_count",
    "draw_pile_count",
    "discard_pile_count",
    "current_hp",
    "max_hp",
    "energy",
)


@dataclass(frozen=True)
class ObservedValue:
    value: int
    present: bool
    confidence: float = 1.0
    source: str = "vision"


@dataclass(frozen=True)
class FrameObservation:
    state: GameState
    fields: dict[str, ObservedValue]
    confidence: float

    @classmethod
    def from_parse_result(cls, state: GameState, result: ParseResult) -> "FrameObservation":
        p = result.combat.player
        observed = result.observed

        def field(name: str, value: int) -> ObservedValue:
            return ObservedValue(
                value=value,
                present=bool(observed.get(name, True)),
                confidence=result.confidence,
            )

        return cls(
            state=state,
            confidence=result.confidence,
            fields={
                "gold": field("gold", state.gold),
                "floor": field("floor", state.floor),
                "deck_count": field("deck_count", state.deck_count),
                "current_hp": field("current_hp", state.current_hp),
                "max_hp": field("max_hp", state.max_hp),
                "energy": field("energy", p.energy),
                "block": field("block", p.block),
                "draw_pile_count": field("draw_pile_count", result.combat.draw_pile_count),
                "discard_pile_count": field("discard_pile_count", result.combat.discard_pile_count),
            },
        )

    def present(self, name: str) -> bool:
        value = self.fields.get(name)
        return bool(value and value.present)


@dataclass
class PendingAction:
    command: protocol.Command
    before: GameState


class StateTracker:
    def __init__(self, config) -> None:
        self.config = config
        self.last_state: GameState | None = None
        self.accepted_state: GameState | None = None
        self.pending: PendingAction | None = None
        self.seq = 0

    def note_action(self, command: protocol.Command, before_state: GameState | None) -> None:
        if not _is_fresh_combat(before_state):
            self.pending = None
            return
        self.pending = PendingAction(command=deepcopy(command), before=deepcopy(before_state))

    def clear_pending(self) -> None:
        self.pending = None

    def reduce_noncombat(self, state: GameState) -> GameState:
        self.pending = None
        self.accepted_state = None
        return self._publish(state, READ_FRESH, [])

    def reduce_combat(self, raw: GameState, result: ParseResult, arbiter=None) -> GameState:
        observation = FrameObservation.from_parse_result(raw, result)
        previous = self.accepted_state if _is_combat(self.accepted_state) else None
        notes: list[str] = []
        unknown_fields = self._unknown_fields(observation, previous)

        if not self._acceptable_combat(observation):
            notes.append(f"combat read confidence {observation.confidence:.0%} below acceptance threshold")
            if previous is None:
                self._note_unknown_fields(unknown_fields, notes)
            state = self._carry_previous(previous, raw, notes)
            return self._publish(
                state,
                READ_UNCERTAIN if previous is None else READ_STALE,
                notes,
                unknown_fields=unknown_fields if previous is None else None,
            )

        state = deepcopy(raw)
        status = READ_FRESH
        if previous is not None:
            critical_missing = self._carry_missing_values(state, observation, previous, notes)
            if self.pending is None:
                self._carry_occluded_combat_entities(state, result, previous, notes)
            if critical_missing:
                status = READ_STALE
        else:
            self._note_unknown_fields(unknown_fields, notes)

        if self.pending is not None:
            if status == READ_FRESH:
                if not _gameplay_changed(state, self.pending.before):
                    notes.append("pending action not observed yet")
                    state = self._carry_previous(previous, raw, notes)
                    return self._publish(
                        state,
                        READ_STALE,
                        notes,
                        accept=False,
                        unknown_fields=unknown_fields,
                    )
                state = self._reconcile_pending(state, arbiter)
            else:
                notes.append("pending action kept for next fresh combat read")

        return self._publish(state, status, notes, accept=status == READ_FRESH, unknown_fields=unknown_fields)

    def _acceptable_combat(self, observation: FrameObservation) -> bool:
        if observation.confidence < _MIN_COMBAT_CONFIDENCE:
            return False
        return (
            observation.present("current_hp")
            and observation.present("max_hp")
            and observation.present("energy")
        )

    def _carry_previous(
        self,
        previous: GameState | None,
        raw: GameState,
        notes: list[str],
    ) -> GameState:
        if previous is None:
            return deepcopy(raw)
        notes.append("showing last accepted combat snapshot")
        state = deepcopy(previous)
        state.parse_confidence = raw.parse_confidence
        return state

    def _carry_missing_values(
        self,
        state: GameState,
        observation: FrameObservation,
        previous: GameState,
        notes: list[str],
    ) -> bool:
        critical_missing = False
        for name in ("floor", "deck_count"):
            if not observation.present(name):
                setattr(state, name, getattr(previous, name))
                notes.append(_carry_note(name, previous))

        if not observation.present("gold"):
            state.gold = previous.gold
            notes.append(_carry_note("gold", previous))

        if state.combat_state is None or previous.combat_state is None:
            return True

        p = state.combat_state.player
        prev_p = previous.combat_state.player
        for name in ("current_hp", "max_hp"):
            if observation.present(name):
                continue
            value = getattr(prev_p, name)
            setattr(p, name, value)
            setattr(state, name, value)
            notes.append(_carry_note(f"player {name}", previous, name))
            critical_missing = True

        if not observation.present("energy"):
            p.energy = prev_p.energy
            notes.append(_carry_note("energy", previous))
            critical_missing = True

        if not observation.present("draw_pile_count"):
            state.combat_state.draw_pile_count = previous.combat_state.draw_pile_count
            notes.append(_carry_note("draw pile", previous, "draw_pile_count"))
        if not observation.present("discard_pile_count"):
            state.combat_state.discard_pile_count = previous.combat_state.discard_pile_count
            notes.append(_carry_note("discard pile", previous, "discard_pile_count"))

        return critical_missing

    def _carry_occluded_combat_entities(
        self,
        state: GameState,
        result: ParseResult,
        previous: GameState,
        notes: list[str],
    ) -> None:
        if state.combat_state is None or previous.combat_state is None:
            return
        if not result.observed.get("hand", True) and previous.combat_state.hand:
            state.combat_state.hand = deepcopy(previous.combat_state.hand)
            notes.append("hand carried from previous read")
        self._stabilize_monsters(state, previous, notes)

    def _stabilize_monsters(
        self,
        state: GameState,
        previous: GameState,
        notes: list[str],
    ) -> None:
        if state.combat_state is None or previous.combat_state is None:
            return
        current = state.combat_state.monsters
        previous_monsters = previous.combat_state.monsters
        if not previous_monsters:
            return

        used_previous: set[int] = set()
        for monster in current:
            match_index = _best_monster_match(monster, previous_monsters, used_previous)
            if match_index is None:
                continue
            used_previous.add(match_index)
            prior = previous_monsters[match_index]
            monster.index = prior.index
            if not monster.name:
                monster.name = prior.name
            if not monster.monster_id:
                monster.monster_id = prior.monster_id
            if monster.max_hp <= 0:
                monster.max_hp = prior.max_hp

        carried = 0
        for i, prior in enumerate(previous_monsters):
            if i in used_previous or not _monster_alive(prior):
                continue
            current.append(deepcopy(prior))
            carried += 1
            notes.append(f"enemy {prior.index} carried from previous read")
        if carried:
            current.sort(key=lambda m: m.index if m.index >= 0 else 999)

    def _unknown_fields(self, observation: FrameObservation, previous: GameState | None) -> list[str]:
        previous_unknown = set(previous.unknown_fields) if previous is not None else set()
        unknown: list[str] = []
        for name in _UNKNOWN_TRACKED_FIELDS:
            if observation.present(name):
                continue
            if previous is None or name in previous_unknown:
                unknown.append(name)
        return unknown

    def _note_unknown_fields(self, fields: list[str], notes: list[str]) -> None:
        for name in fields:
            notes.append(f"{name} not read")

    def _reconcile_pending(self, state: GameState, arbiter) -> GameState:
        pending = self.pending
        self.pending = None
        if pending is None or not self.config.predict_enabled:
            return state
        predicted = predict(pending.before, pending.command)
        if predicted is None:
            return state
        allow_monster_overrides = _is_fresh_combat(pending.before)
        return reconcile(
            state,
            predicted,
            pending.before,
            arbiter,
            allow_monster_overrides=allow_monster_overrides,
        )

    def _publish(
        self,
        state: GameState,
        status: str,
        notes: list[str],
        *,
        accept: bool = True,
        unknown_fields: list[str] | None = None,
    ) -> GameState:
        self.seq += 1
        state.state_seq = self.seq
        state.read_status = status
        state.state_notes = list(notes)
        if unknown_fields is not None:
            state.unknown_fields = sorted(set(unknown_fields))
        state.available_commands = _commands_for_state(state.screen_type, status)
        if state.floor:
            state.act = _act_for_floor(state.floor)
        self.last_state = state
        if accept and status == READ_FRESH and _is_combat(state):
            self.accepted_state = deepcopy(state)
        return state


def _commands_for_state(screen_type: ScreenType, status: str) -> list[str]:
    commands = protocol.commands_for_screen(screen_type.value)
    if screen_type == ScreenType.COMBAT and status != READ_FRESH:
        return [c for c in commands if c not in {protocol.Verb.PLAY, protocol.Verb.END}]
    return commands


def _is_combat(state: GameState | None) -> bool:
    return state is not None and state.screen_type == ScreenType.COMBAT and state.combat_state is not None


def _is_fresh_combat(state: GameState | None) -> bool:
    return _is_combat(state) and state.read_status == READ_FRESH


def _carry_note(label: str, previous: GameState, field: str | None = None) -> str:
    field_name = field or label
    if field_name in previous.unknown_fields:
        return f"{label} still not read"
    return f"{label} carried from previous read"


def _best_monster_match(monster, candidates, used: set[int]) -> int | None:
    scored = [
        (_monster_match_score(monster, candidate), i)
        for i, candidate in enumerate(candidates)
        if i not in used
    ]
    if not scored:
        return None
    score, index = max(scored, key=lambda item: item[0])
    return index if score > 0 else None


def _monster_match_score(vision, previous) -> int:
    score = 0
    if vision.index == previous.index:
        score += 3
    v_name = (vision.monster_id or vision.name).strip().lower()
    p_name = (previous.monster_id or previous.name).strip().lower()
    if v_name and p_name and v_name == p_name:
        score += 8
    if vision.max_hp > 0 and vision.max_hp == previous.max_hp:
        score += 6
    if vision.current_hp > 0 and previous.current_hp > 0:
        score += max(0, 5 - min(5, abs(vision.current_hp - previous.current_hp)))
    return score


def _monster_alive(monster) -> bool:
    if monster.is_gone or monster.half_dead:
        return False
    return monster.max_hp > 0 or monster.current_hp > 0


def _gameplay_changed(current: GameState, before: GameState) -> bool:
    return _gameplay_signature(current) != _gameplay_signature(before)


def _gameplay_signature(state: GameState) -> str:
    data = state.to_dict()
    for key in (
        "available_commands",
        "parse_confidence",
        "read_status",
        "screen_message",
        "state_notes",
        "state_seq",
        "unknown_fields",
    ):
        data.pop(key, None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _act_for_floor(floor: int) -> int:
    if floor <= 0:
        return 0
    if floor <= 16:
        return 1
    if floor <= 33:
        return 2
    if floor <= 50:
        return 3
    return 4
