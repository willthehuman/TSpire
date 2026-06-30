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
    # NOTE: gold/floor/top_hp X positions are laid out by TopPanel RELATIVE to the rendered
    # width of the preceding text (character name, HP value, ...), so they shift run-to-run
    # and have no fixed fraction -- hence these stay broad estimates and rely on the OCR->LLM
    # fallback. (Verified against the decompiled TopPanel: nameX -> titleX -> hpIconX ->
    # goldIconX -> floorX chain.)
    gold: Rect = Rect(0.245, 0.010, 0.060, 0.045)
    top_hp: Rect = Rect(0.145, 0.015, 0.100, 0.060)  # top-bar heart + "cur/max" text
    floor: Rect = Rect(0.470, 0.010, 0.060, 0.050)  # floor number, top-center banner
    # Deck icon is fixed at the top-right: DECK_X = WIDTH - (ICON_W(64)+PAD(10))*2, centre
    # at WIDTH - 116*scale, ICON_Y = HEIGHT - 32*scale -> (0.940, 0.030). (decompiled TopPanel)
    deck_count: Rect = Rect(0.900, 0.008, 0.080, 0.075)  # master deck count, top-right
    relics_search: Rect = Rect(0.015, 0.075, 0.430, 0.085)  # left-to-right relic row
    potions_search: Rect = Rect(0.295, 0.010, 0.110, 0.075)  # potion belt (next to gold)

    # --- player, in combat (exact, derived from the decompiled game) ---
    # Player draws at drawX = WIDTH*0.25, floorY = 340*yScale. The HP "cur/max" text sits on
    # the health bar at centre (0.25*W - 4*scale, HEIGHT-(floorY+hb_y-barH/2)) ~ (0.248, 0.709),
    # bar width = hitbox width 220*scale (~0.115). Box tightened to the number for cleaner OCR.
    player_hp: Rect = Rect(0.186, 0.688, 0.124, 0.044)  # red HP bar + "cur/max" text
    player_powers_search: Rect = Rect(0.175, 0.715, 0.170, 0.070)  # player buffs/debuffs
    player_block: Rect = Rect(0.160, 0.640, 0.060, 0.045)  # shield badge, when block > 0
    # Energy number is rendered centred at (198*xScale, 190*yScale) from the bottom-left ->
    # (0.103, 0.824); box tightened to the digits (excludes the orb art below). (EnergyPanel)
    energy: Rect = Rect(0.063, 0.788, 0.080, 0.074)  # energy orb "cur/max", bottom-left

    # --- end-turn button (bottom-right): used by the classifier to detect combat ---
    end_turn: Rect = Rect(0.790, 0.775, 0.150, 0.105)

    # --- pile counters (bottom corners) ---
    draw_pile: Rect = Rect(0.020, 0.895, 0.075, 0.100)  # draw count, bottom-left
    discard_pile: Rect = Rect(0.915, 0.895, 0.075, 0.100)  # discard count, bottom-right

    # --- dynamic search regions (count not known a priori) ---
    # Monster HP bars sit on the ground line; restricting to that band (rather than the
    # whole enemy area) rejects most background-red false positives. We locate each enemy
    # by its HP bar, then read intent/sprite *above* it in the full frame.
    # Enemies sit on the right half (the player is fixed at x=0.25) and spread from just right
    # of centre toward the edge, so keep X wide. Keep Y on the ground-line band where HP bars
    # live -- too tall a band catches red intent/status icons above the sprites as false bars.
    monster_search: Rect = Rect(0.420, 0.600, 0.575, 0.220)
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
