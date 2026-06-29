"""Vision: turn a captured frame into game state.

Pipeline: capture (BGR frame) -> classify screen type -> per-screen parser -> GameState.
v1 implements the COMBAT parser. Everything is built around:

  * regions.py   - resolution-scaled named rectangles (fractional, so they scale)
  * backend.py   - low-level primitives: crop, OCR ints/text, color-bar detection,
                   template matching. The OpenCV/Tesseract impl + a fake for tests.
  * templates.py - the static-art template database (built by tools/extract_assets.py)
  * combat.py    - composes the above into a CombatState
"""

from tspire.host.vision.regions import Rect, RegionMap, region_map_for

__all__ = ["Rect", "RegionMap", "region_map_for"]
