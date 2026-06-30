"""A fake VisionBackend for tests.

Drives the combat parser and classifier with scripted data instead of real pixels, so the
assembly logic (indexing, intent mapping, confidence, state shape) is testable without
OpenCV/Tesseract or the game running. Reads are resolved by comparing the requested Rect
against known region fields, or by nearest-center-x lookup into the configured monster /
card lists (mirroring how the parser derives sub-regions from detected boxes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tspire.host.vision.backend import BBox
from tspire.host.vision.regions import Rect, RegionMap


class FakeFrame:
    """Stands in for a numpy frame: only `.shape` is used by the parser."""

    def __init__(self, width: int = 1920, height: int = 1080) -> None:
        self.shape = (height, width, 3)


@dataclass
class FakeMonster:
    left: int
    hp: int
    hp_max: int
    intent_id: str = "attack"
    intent_score: float = 0.9
    dmg: int = 0
    name: str = "JawWorm"
    name_score: float = 0.9

    @property
    def cx(self) -> int:
        return self.left + 60  # bar width 120 -> center


@dataclass
class FakeCard:
    left: int
    cost: int
    name: str

    @property
    def cx(self) -> int:
        return self.left + 50  # card width 100 -> center


@dataclass
class FakeVisionBackend:
    regions: RegionMap
    width: int = 1920
    height: int = 1080
    player_hp: tuple[int, int] = (70, 80)
    top_hp: tuple[int, int] = (70, 80)
    energy: tuple[int, int] = (3, 3)
    block: int = 0
    block_filled: bool = False
    gold: int = 99
    floor: int = 1
    deck_count: int = 10
    draw: int = 5
    discard: int = 2
    energy_filled: bool = True
    end_turn_filled: bool = True
    draw_pile_filled: bool = False
    discard_pile_filled: bool = False
    monsters: list[FakeMonster] = field(default_factory=list)
    cards: list[FakeCard] = field(default_factory=list)

    # --- helpers ----------------------------------------------------------
    def _center_x(self, rect: Rect) -> float:
        return (rect.x + rect.w / 2) * self.width

    def _nearest_monster(self, x: float) -> FakeMonster | None:
        return min(self.monsters, key=lambda m: abs(m.cx - x), default=None)

    def _nearest_card(self, x: float) -> FakeCard | None:
        return min(self.cards, key=lambda c: abs(c.cx - x), default=None)

    # --- VisionBackend interface -----------------------------------------
    def ocr_text(self, frame, rect: Rect, *, digits: bool = False) -> str:
        card = self._nearest_card(self._center_x(rect))
        return card.name if card else ""

    def ocr_int(self, frame, rect: Rect, *, default: int = 0) -> int:
        if rect == self.regions.player_block:
            return self.block
        if rect == self.regions.draw_pile:
            return self.draw
        if rect == self.regions.discard_pile:
            return self.discard
        if rect == self.regions.gold:
            return self.gold
        if rect == self.regions.floor:
            return self.floor
        if rect == self.regions.deck_count:
            return self.deck_count
        if rect.y < 0.5:  # intent damage (upper area)
            m = self._nearest_monster(self._center_x(rect))
            return m.dmg if m else default
        card = self._nearest_card(self._center_x(rect))  # card cost (lower area)
        return card.cost if card else default

    def ocr_int_pair(self, frame, rect: Rect) -> tuple[int, int]:
        if rect == self.regions.player_hp:
            return self.player_hp
        if rect == self.regions.top_hp:
            return self.top_hp
        if rect == self.regions.energy:
            return self.energy
        m = self._nearest_monster(self._center_x(rect))
        return (m.hp, m.hp_max) if m else (0, 0)

    def find_red_bars(self, frame, search: Rect) -> list[BBox]:
        return [BBox(left=m.left, top=200, width=120, height=12) for m in self.monsters]

    def find_cards(self, frame, search: Rect) -> list[BBox]:
        return [BBox(left=c.left, top=800, width=100, height=140) for c in self.cards]

    def crop_px(self, frame, box: BBox):
        return None

    def classify_box(self, frame, box: BBox, category: str) -> tuple[str, float]:
        m = self._nearest_monster(box.cx)
        if m is None:
            return "", 0.0
        if category == "monsters":
            return m.name, m.name_score
        if category == "intents":
            return m.intent_id, m.intent_score
        return "", 0.0

    def region_filled(self, frame, rect: Rect, *, min_std: float = 12.0) -> bool:
        if rect == self.regions.energy:
            return self.energy_filled
        if rect == self.regions.end_turn:
            return self.end_turn_filled
        if rect == self.regions.draw_pile:
            return self.draw_pile_filled
        if rect == self.regions.discard_pile:
            return self.discard_pile_filled
        if rect == self.regions.player_block:
            return self.block_filled
        return False


@dataclass
class FakeArbiter:
    """Scripted Arbiter for the reconciler: returns fixed re-read pairs (or None)."""

    player_hp: tuple[int, int] | None = None
    energy: tuple[int, int] | None = None

    def reread_player_hp(self) -> tuple[int, int] | None:
        return self.player_hp

    def reread_energy(self) -> tuple[int, int] | None:
        return self.energy
