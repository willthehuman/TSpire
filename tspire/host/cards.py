"""Curated card-value table for the state predictor.

The predictor (tspire.host.predict) needs the base numeric effect of a played card to
estimate the next combat state. Slay the Spire's base values live in the game's Java
source, not in any data file we can read at runtime, so this table is a hand-curated
subset: the starter cards plus the common attacks/blocks seen early in a run.

It is intentionally PARTIAL. An unknown card returns ``None`` from :func:`lookup`, and the
predictor then leaves that card's effect to the vision read rather than fabricating one.
Extend ``CARD_DB`` as more cards need predicting.

Cards are matched by their display name (what the vision model reads), normalized to bare
lowercase alphanumerics so "Pommel Strike", "pommel_strike" and "PommelStrike" all hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CardData:
    """Base (and upgraded) numeric effect of a card.

    Only the fields the predictor uses are modeled. ``damage``/``block`` are the
    unupgraded values; ``*_up`` override them for the upgraded card when set.
    ``aoe`` marks attacks that hit every enemy (Cleave, Thunderclap, ...).
    """

    card_id: str
    damage: int = 0
    block: int = 0
    hits: int = 1
    target: bool = False
    aoe: bool = False
    exhausts: bool = False
    damage_up: int | None = None
    block_up: int | None = None

    def damage_for(self, upgrades: int) -> int:
        if upgrades > 0 and self.damage_up is not None:
            return self.damage_up
        return self.damage

    def block_for(self, upgrades: int) -> int:
        if upgrades > 0 and self.block_up is not None:
            return self.block_up
        return self.block


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Keyed by display name; built into a normalized lookup map below.
_CARDS: list[tuple[str, CardData]] = [
    # ---- starters (all classes share Strike / Defend) -------------------
    ("Strike", CardData("Strike", damage=6, target=True, damage_up=9)),
    ("Defend", CardData("Defend", block=5, block_up=8)),
    ("Bash", CardData("Bash", damage=8, target=True, damage_up=10)),  # Ironclad
    ("Survivor", CardData("Survivor", block=8, block_up=11)),  # Silent
    ("Neutralize", CardData("Neutralize", damage=3, target=True, damage_up=4)),  # Silent
    ("Zap", CardData("Zap")),  # Defect: channels Lightning, no hp effect to predict
    ("Dualcast", CardData("Dualcast")),  # Defect
    ("Eruption", CardData("Eruption", damage=9, target=True)),  # Watcher
    ("Vigilance", CardData("Vigilance", block=8, block_up=12)),  # Watcher
    # ---- common Ironclad attacks / blocks -------------------------------
    ("Pommel Strike", CardData("Pommel Strike", damage=9, target=True, damage_up=10)),
    ("Twin Strike", CardData("Twin Strike", damage=5, hits=2, target=True, damage_up=7)),
    ("Clothesline", CardData("Clothesline", damage=12, target=True, damage_up=14)),
    ("Cleave", CardData("Cleave", damage=8, aoe=True, damage_up=11)),
    ("Thunderclap", CardData("Thunderclap", damage=4, aoe=True, damage_up=7)),
    ("Iron Wave", CardData("Iron Wave", damage=5, block=5, target=True, damage_up=7, block_up=7)),
    ("Sword Boomerang", CardData("Sword Boomerang", damage=3, hits=3, aoe=False)),
    ("Anger", CardData("Anger", damage=6, target=True, damage_up=8)),
    ("Headbutt", CardData("Headbutt", damage=9, target=True, damage_up=12)),
    ("Heavy Blade", CardData("Heavy Blade", damage=14, target=True, damage_up=14)),
    ("Shrug It Off", CardData("Shrug It Off", block=8, block_up=11)),
    # ---- common Silent attacks / blocks ---------------------------------
    ("Dagger Throw", CardData("Dagger Throw", damage=9, target=True, damage_up=12)),
    ("Quick Slash", CardData("Quick Slash", damage=8, target=True, damage_up=12)),
    ("Slice", CardData("Slice", damage=6, target=True, damage_up=9)),
    ("Sucker Punch", CardData("Sucker Punch", damage=7, target=True, damage_up=9)),
    ("Dash", CardData("Dash", damage=10, block=10, target=True, damage_up=13, block_up=13)),
    ("Backflip", CardData("Backflip", block=5, block_up=8)),
    # ---- common Defect / Watcher hp-affecting cards ---------------------
    ("Strike Defect", CardData("Strike", damage=6, target=True, damage_up=9)),
    ("Strike Watcher", CardData("Strike", damage=6, target=True, damage_up=9)),
    ("Cut Through Fate", CardData("Cut Through Fate", damage=7, target=True, damage_up=9)),
]

CARD_DB: dict[str, CardData] = {_norm(name): data for name, data in _CARDS}


def lookup(name_or_id: str) -> CardData | None:
    """Return the card's base data, or None if it isn't in the curated table.

    Accepts a display name or id, with or without an upgrade marker ("Strike+").
    """
    if not name_or_id:
        return None
    key = _norm(name_or_id.split("+")[0])
    return CARD_DB.get(key)
