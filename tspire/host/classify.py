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
    # Cheap structural signal (no LLM, no flaky red-bar detection): the energy orb and the
    # End-Turn button are both present only during combat. This gates the expensive parse.
    return backend.region_filled(frame, regions.energy) and backend.region_filled(
        frame, regions.end_turn
    )
