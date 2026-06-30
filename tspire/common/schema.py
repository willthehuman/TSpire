"""Game-state schema.

Mirrors the shape of CommunicationMod's JSON output (the de-facto standard used by the
`spirecomm` client) so the client UX is a proven design and the real mod could later be
slotted in as an alternate state backend. Fields the host cannot yet observe from the
screen are left at their defaults rather than omitted, so the client can render
consistently.

Everything here is plain dataclasses + dict (de)serialization — no host-only deps.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from typing import Any, get_args, get_origin, get_type_hints


class ScreenType(str, Enum):
    """What kind of screen the game is currently showing.

    Subset/rename of CommunicationMod's screen types. v1 only fully parses COMBAT;
    others are detected enough to report and to drive `proceed`/`choose`.
    """

    NONE = "NONE"
    COMBAT = "COMBAT"
    MAP = "MAP"
    EVENT = "EVENT"
    CARD_REWARD = "CARD_REWARD"
    COMBAT_REWARD = "COMBAT_REWARD"
    SHOP_ROOM = "SHOP_ROOM"
    SHOP_SCREEN = "SHOP_SCREEN"
    REST = "REST"
    GRID = "GRID"
    HAND_SELECT = "HAND_SELECT"
    CHEST = "CHEST"
    BOSS_REWARD = "BOSS_REWARD"
    MAIN_MENU = "MAIN_MENU"
    GAME_OVER = "GAME_OVER"
    UNKNOWN = "UNKNOWN"


class Intent(str, Enum):
    """Enemy intent categories, matching CommunicationMod's `Intent` enum names."""

    ATTACK = "ATTACK"
    ATTACK_BUFF = "ATTACK_BUFF"
    ATTACK_DEBUFF = "ATTACK_DEBUFF"
    ATTACK_DEFEND = "ATTACK_DEFEND"
    BUFF = "BUFF"
    DEBUFF = "DEBUFF"
    STRONG_DEBUFF = "STRONG_DEBUFF"
    DEBUG = "DEBUG"
    DEFEND = "DEFEND"
    DEFEND_DEBUFF = "DEFEND_DEBUFF"
    DEFEND_BUFF = "DEFEND_BUFF"
    ESCAPE = "ESCAPE"
    MAGIC = "MAGIC"
    NONE = "NONE"
    SLEEP = "SLEEP"
    STUN = "STUN"
    UNKNOWN = "UNKNOWN"

    @property
    def is_attack(self) -> bool:
        return self in {
            Intent.ATTACK,
            Intent.ATTACK_BUFF,
            Intent.ATTACK_DEBUFF,
            Intent.ATTACK_DEFEND,
        }


@dataclass
class Power:
    """A buff/debuff on the player or a monster (e.g. Strength, Vulnerable, Weak)."""

    power_id: str = ""
    name: str = ""
    amount: int = 0


@dataclass
class Card:
    """A card in hand (or a pile). `index` is its left-to-right position in hand."""

    name: str = ""
    card_id: str = ""
    cost: int = -1  # -1 = unknown/X cost; matches CommunicationMod "X" convention loosely
    upgrades: int = 0
    type: str = ""  # ATTACK | SKILL | POWER | STATUS | CURSE
    rarity: str = ""
    has_target: bool = False
    is_playable: bool = False
    exhausts: bool = False
    description: str = ""
    index: int = -1


@dataclass
class Monster:
    """A single enemy in combat. `index` is its left-to-right position."""

    name: str = ""
    monster_id: str = ""
    current_hp: int = 0
    max_hp: int = 0
    block: int = 0
    intent: Intent = Intent.UNKNOWN
    intent_damage: int = 0  # adjusted damage of a single hit, 0 if not attacking/unknown
    intent_hits: int = 1
    is_gone: bool = False  # dead or fled
    half_dead: bool = False
    powers: list[Power] = field(default_factory=list)
    index: int = -1


@dataclass
class Potion:
    name: str = ""
    potion_id: str = ""
    can_use: bool = False
    can_discard: bool = False
    requires_target: bool = False
    index: int = -1


@dataclass
class Relic:
    name: str = ""
    relic_id: str = ""
    counter: int = -1
    index: int = -1


@dataclass
class PlayerCombat:
    """The player's in-combat status."""

    current_hp: int = 0
    max_hp: int = 0
    block: int = 0
    energy: int = 0
    powers: list[Power] = field(default_factory=list)
    orbs: list[str] = field(default_factory=list)  # Defect; post-v1


@dataclass
class CombatState:
    """Everything specific to an active combat."""

    player: PlayerCombat = field(default_factory=PlayerCombat)
    monsters: list[Monster] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)
    draw_pile_count: int = 0
    discard_pile_count: int = 0
    exhaust_pile_count: int = 0
    turn: int = 0


@dataclass
class GameState:
    """Top-level game state pushed from host to client.

    `available_commands` lists the command verbs valid on the current screen (see
    tspire.common.protocol), letting the client gray out impossible actions.
    `parse_confidence` is a 0..1 hint about how trustworthy the screen read was.
    """

    screen_type: ScreenType = ScreenType.NONE
    in_combat: bool = False
    floor: int = 0
    act: int = 0
    current_hp: int = 0
    max_hp: int = 0
    gold: int = 0
    deck_count: int = 0
    relics: list[Relic] = field(default_factory=list)
    potions: list[Potion] = field(default_factory=list)
    combat_state: CombatState | None = None
    available_commands: list[str] = field(default_factory=list)
    screen_message: str = ""  # e.g. "screen not yet supported", event text
    parse_confidence: float = 0.0
    state_seq: int = 0
    read_status: str = "fresh"  # fresh | stale | uncertain
    state_notes: list[str] = field(default_factory=list)
    unknown_fields: list[str] = field(default_factory=list)

    # ---- serialization -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameState:
        return _from_dict(cls, data)


# --------------------------------------------------------------------------- #
# Generic dataclass <-> dict conversion that understands Enums and nested
# dataclass lists/optionals. Kept here so both ends serialize identically.
# --------------------------------------------------------------------------- #
def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _from_dict(tp: Any, data: Any) -> Any:
    if data is None:
        return None
    origin = get_origin(tp)
    if origin is list:
        (item_tp,) = get_args(tp) or (Any,)
        return [_from_dict(item_tp, v) for v in data]
    # Optional[X] / Union -> pick the first non-None arg.
    if origin is not None and get_args(tp):
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if non_none:
            return _from_dict(non_none[0], data)
    if isinstance(tp, type) and issubclass(tp, Enum):
        try:
            return tp(data)
        except ValueError:
            return tp.UNKNOWN if hasattr(tp, "UNKNOWN") else data
    if is_dataclass(tp) and isinstance(data, dict):
        hints = _resolved_hints(tp)  # f.type is a string under PEP 563; resolve it
        kwargs = {}
        for f in fields(tp):
            if f.name in data:
                kwargs[f.name] = _from_dict(hints.get(f.name, f.type), data[f.name])
        return tp(**kwargs)
    return data


@lru_cache(maxsize=None)
def _resolved_hints(tp: type) -> dict[str, Any]:
    return get_type_hints(tp, globalns=vars(sys.modules[tp.__module__]))
