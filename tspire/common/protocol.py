"""Wire protocol between host and client.

Two message directions, both JSON over a WebSocket text frame:

  client -> host : {"type": "command", "verb": "...", "args": [...], "id": "..."}
  host -> client : {"type": "state",  "state": {...GameState...}}
                   {"type": "ack",    "id": "...", "ok": true, "error": null}
                   {"type": "log",    "level": "info", "message": "..."}

The command vocabulary mirrors CommunicationMod / spirecomm so the model is proven.
Indices are 0-based and refer to the corresponding lists in the most recent GameState
(hand, monsters, potions, choices). The client hides indices from the user where it can
(spirecomm style) and resolves friendly input to these verbs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tspire.common.schema import GameState

PROTOCOL_VERSION = 1


# --------------------------------------------------------------------------- #
# Command vocabulary
# --------------------------------------------------------------------------- #
class Verb:
    PLAY = "play"  # play <hand_index> [target_monster_index]
    END = "end"  # end turn
    CHOOSE = "choose"  # choose <index> on a list-style screen
    POTION = "potion"  # potion <use|discard> <index> [target_monster_index]
    PROCEED = "proceed"  # advance / confirm (Y on controller)
    RETURN = "return"  # back / cancel (B on controller)
    STATE = "state"  # re-read and push current state (no game input)
    RAW = "raw"  # raw passthrough: args are low-level gamepad tokens (debug)


_READ_ONLY_VERBS = {Verb.STATE}


def is_state_altering(verb: str) -> bool:
    """True when a command may change the visible game state."""
    return verb not in _READ_ONLY_VERBS


# Verbs valid on each screen. The host fills GameState.available_commands from this so
# the client can disable impossible actions. STATE is always allowed (read-only).
COMMANDS_BY_SCREEN: dict[str, list[str]] = {
    "COMBAT": [Verb.PLAY, Verb.END, Verb.PROCEED, Verb.RETURN, Verb.STATE],
    "NONE": [Verb.STATE, Verb.PROCEED],
    "UNKNOWN": [Verb.STATE, Verb.PROCEED, Verb.RETURN],
}
# Post-v1 screens (map/event/reward/shop/...) get CHOOSE/PROCEED/RETURN once parsed.
_DEFAULT_COMMANDS = [Verb.CHOOSE, Verb.PROCEED, Verb.RETURN, Verb.STATE]


def commands_for_screen(screen_type: str) -> list[str]:
    return COMMANDS_BY_SCREEN.get(screen_type, _DEFAULT_COMMANDS)


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
@dataclass
class Command:
    verb: str
    args: list[str] = field(default_factory=list)
    id: str = ""  # client-generated; echoed in the matching ack

    def to_message(self) -> str:
        return json.dumps({"type": "command", "verb": self.verb, "args": self.args, "id": self.id})


def state_message(state: GameState) -> str:
    return json.dumps({"type": "state", "state": state.to_dict()})


def ack_message(command_id: str, ok: bool, error: str | None = None) -> str:
    return json.dumps({"type": "ack", "id": command_id, "ok": ok, "error": error})


def log_message(message: str, level: str = "info") -> str:
    return json.dumps({"type": "log", "level": level, "message": message})


def parse_message(raw: str) -> dict[str, Any]:
    """Decode a wire frame into a plain dict. Raises ValueError on malformed input."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(data, dict) or "type" not in data:
        raise ValueError("message missing 'type'")
    return data


def command_from_message(data: dict[str, Any]) -> Command:
    return Command(
        verb=str(data.get("verb", "")),
        args=[str(a) for a in data.get("args", [])],
        id=str(data.get("id", "")),
    )
