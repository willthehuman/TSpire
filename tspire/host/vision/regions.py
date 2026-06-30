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
    gold: Rect = Rect(0.245, 0.010, 0.060, 0.045)
    top_hp: Rect = Rect(0.145, 0.015, 0.100, 0.060)  # top-bar heart + "cur/max" text
    floor: Rect = Rect(0.470, 0.010, 0.060, 0.050)  # floor number, top-center banner
    deck_count: Rect = Rect(0.885, 0.010, 0.080, 0.080)  # master deck count, top-right
    relics_search: Rect = Rect(0.015, 0.075, 0.430, 0.085)  # left-to-right relic row
    potions_search: Rect = Rect(0.295, 0.010, 0.110, 0.075)  # potion belt (next to gold)

    # --- player, in combat ---
    player_hp: Rect = Rect(0.185, 0.675, 0.135, 0.065)  # red HP bar + "cur/max" text
    player_powers_search: Rect = Rect(0.175, 0.715, 0.170, 0.070)  # player buffs/debuffs
    player_block: Rect = Rect(0.160, 0.640, 0.060, 0.045)  # shield badge, when block > 0
    energy: Rect = Rect(0.065, 0.760, 0.080, 0.120)  # energy orb "cur/max", bottom-left

    # --- end-turn button (bottom-right): used by the classifier to detect combat ---
    end_turn: Rect = Rect(0.790, 0.775, 0.150, 0.105)

    # --- pile counters (bottom corners) ---
    draw_pile: Rect = Rect(0.020, 0.895, 0.075, 0.100)  # draw count, bottom-left
    discard_pile: Rect = Rect(0.915, 0.895, 0.075, 0.100)  # discard count, bottom-right

    # --- dynamic search regions (count not known a priori) ---
    # Monster HP bars sit on the ground line; restricting to that band (rather than the
    # whole enemy area) rejects most background-red false positives. We locate each enemy
    # by its HP bar, then read intent/sprite *above* it in the full frame.
    monster_search: Rect = Rect(0.450, 0.600, 0.420, 0.180)
    # Hand fans across the bottom-center; cards located by their bright frames.
    hand_search: Rect = Rect(0.230, 0.760, 0.545, 0.235)

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
