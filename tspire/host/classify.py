"""Screen-type classification.

Decides which parser to run. v1 only distinguishes COMBAT from everything else (UNKNOWN);
post-v1 milestones add map/event/reward/shop/rest detection here. Combat is recognized by
a cheap structural signature: the energy orb is showing, the end-turn button is present,
and at least one monster HP bar is visible.
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
    energy_present = backend.region_filled(frame, regions.energy)
    end_turn_present = backend.region_filled(frame, regions.end_turn)
    has_monster = bool(backend.find_red_bars(frame, regions.monster_search))
    # Require the player-side signals plus at least one enemy bar. Tunable once we have
    # real captures (the calibrate tool reports each signal).
    return energy_present and end_turn_present and has_monster
