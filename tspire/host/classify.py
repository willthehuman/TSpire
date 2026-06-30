"""Screen-type classification.

Decides which parser to run. v1 only distinguishes COMBAT from everything else (UNKNOWN);
post-v1 milestones add map/event/reward/shop/rest detection here. Combat is recognized by
a small vote of combat-only visual signals. The energy orb + End Turn button are the
fast path, with hand/monster/pile cues as fallbacks when a fixed box misses.
"""

from __future__ import annotations

from tspire.host.vision.backend import VisionBackend
from tspire.host.vision.regions import RegionMap

try:  # numpy is a host dep; keep this module importable without it for typing
    import numpy as np

    NDArray = np.ndarray
except Exception:  # pragma: no cover
    NDArray = object  # type: ignore[assignment]

from tspire.common.schema import ScreenType


def classify_screen(frame: "NDArray", regions: RegionMap, backend: VisionBackend) -> ScreenType:
    if _looks_like_combat(frame, regions, backend):
        return ScreenType.COMBAT
    return ScreenType.UNKNOWN


def _looks_like_combat(frame: "NDArray", regions: RegionMap, backend: VisionBackend) -> bool:
    # Fast path: the two fixed widgets that should both exist during the player's turn.
    energy = _filled(frame, backend, regions.energy, min_std=10.0)
    end_turn = _filled(frame, backend, regions.end_turn, min_std=10.0)
    if energy and end_turn:
        return True

    # Fallbacks: a capture can be slightly off, dimmed, or in a transition just as we read.
    # Require either one fixed combat widget plus one dynamic combat cue, or two dynamic cues.
    hand = _has_cards(frame, backend, regions)
    monsters = _has_monster_bars(frame, backend, regions)
    piles = _filled(frame, backend, regions.draw_pile, min_std=24.0) and _filled(
        frame, backend, regions.discard_pile, min_std=24.0
    )
    dynamic_count = sum(1 for signal in (hand, monsters, piles) if signal)
    return ((energy or end_turn) and dynamic_count >= 1) or dynamic_count >= 2


def _filled(frame: "NDArray", backend: VisionBackend, rect, *, min_std: float) -> bool:
    try:
        return backend.region_filled(frame, rect, min_std=min_std)
    except Exception:
        return False


def _has_cards(frame: "NDArray", backend: VisionBackend, regions: RegionMap) -> bool:
    try:
        return bool(backend.find_cards(frame, regions.hand_search))
    except Exception:
        return False


def _has_monster_bars(frame: "NDArray", backend: VisionBackend, regions: RegionMap) -> bool:
    try:
        return bool(backend.find_red_bars(frame, regions.monster_search))
    except Exception:
        return False
