"""Named screen regions, expressed as fractions of the frame so they scale across
resolutions. The fractional values below are **initial estimates** for the standard
1920x1080 Slay the Spire combat layout and MUST be calibrated against real captures
using ``python -m tspire.host.calibrate`` (which overlays these boxes on a live frame).

Coordinate convention: (x, y, w, h) as fractions in [0, 1], origin top-left.
Fixed single-value regions (energy, gold, player HP/block) are good fits for fractional
boxes. Dynamic, variable-count things (monsters, hand cards) are *not* fixed boxes — they
are located at runtime by detection within a broad search region (see *_search fields).
"""

from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Rect:
    """A rectangle in fractional [0,1] coordinates."""

    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Return (left, top, width, height) in integer pixels, clamped to the frame."""
        left = max(0, min(frame_w, round(self.x * frame_w)))
        top = max(0, min(frame_h, round(self.y * frame_h)))
        width = max(0, min(frame_w - left, round(self.w * frame_w)))
        height = max(0, min(frame_h - top, round(self.h * frame_h)))
        return left, top, width, height


@dataclass(frozen=True)
class RegionMap:
    """All regions for a resolution. Single-value boxes + broad search areas.

    NOTE: every value here is a calibration estimate. Treat the calibrate overlay as the
    source of truth and tune these per real screenshots.
    """

    # --- top panel (always visible) ---
    gold: Rect = Rect(0.045, 0.018, 0.060, 0.035)
    floor: Rect = Rect(0.470, 0.930, 0.060, 0.040)  # floor number, bottom-center-ish
    relics_search: Rect = Rect(0.020, 0.060, 0.500, 0.060)  # left-to-right relic row
    potions_search: Rect = Rect(0.880, 0.020, 0.110, 0.060)  # potion belt, top-right

    # --- player, in combat (bottom-left) ---
    player_hp: Rect = Rect(0.020, 0.905, 0.150, 0.045)  # red HP bar + "cur/max" text
    player_block: Rect = Rect(0.060, 0.860, 0.060, 0.045)  # shield badge, when block > 0
    energy: Rect = Rect(0.050, 0.815, 0.060, 0.060)  # energy orb "cur/max"

    # --- end-turn button (bottom-right): used by the classifier to detect combat ---
    end_turn: Rect = Rect(0.855, 0.470, 0.130, 0.090)

    # --- pile counters (bottom corners) ---
    draw_pile: Rect = Rect(0.020, 0.930, 0.050, 0.055)  # draw count, bottom-left
    discard_pile: Rect = Rect(0.930, 0.930, 0.050, 0.055)  # discard count, bottom-right

    # --- dynamic search regions (count not known a priori) ---
    # Monsters occupy the upper-middle band; we locate each by its HP bar.
    monster_search: Rect = Rect(0.250, 0.150, 0.620, 0.550)
    # Hand fans across the bottom-center; cards located by their cost gem / frame.
    hand_search: Rect = Rect(0.180, 0.720, 0.640, 0.280)

    def all_regions(self) -> dict[str, Rect]:
        """Name -> Rect for every region (used by the calibration overlay)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


# Resolution -> RegionMap. v1 ships 1920x1080; add calibrated maps for other
# resolutions here as needed.
_REGION_MAPS: dict[tuple[int, int], RegionMap] = {
    (1920, 1080): RegionMap(),
}


def region_map_for(width: int, height: int) -> RegionMap:
    """Return the region map for a resolution.

    Falls back to the 16:9 default (fractional coords scale reasonably to other 16:9
    sizes). Non-16:9 resolutions will need their own calibrated entry.
    """
    return _REGION_MAPS.get((width, height), RegionMap())
