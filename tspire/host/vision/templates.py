"""Static-art template database for relic / intent identity.

Slay the Spire's art is fixed, so identity recognition is best done by matching a captured
icon against the game's own images. We read those images **directly from the installed
game's ``desktop-1.0.jar``** at runtime (it's a zip) — no game assets are bundled with this
project, and recognition only works when the game is installed (see
``tspire.host.game_assets.find_game_jar``). A plain directory of PNGs also works (handy for
tests).

Matching uses **alpha-masked HSV colour-histogram correlation**, validated against a real
screen capture: grayscale-shape cosine failed (true relic not even top-5), while the colour
histogram identified Burning Blood at 0.99 vs ~0.76 for the next candidate, robust to crop
framing. Templates are 128x128 RGBA; we composite over black (matching the dark in-game HUD)
and mask by alpha so transparent padding doesn't pollute the histogram.

NOTE on potions: the jar has no per-potion image — potions are a *shape* sprite
(images/potion/<shape>) tinted with a per-potion colour at runtime, so they need shape-match
+ a colour->potion table, not a single-image template (handled elsewhere, not here).

Caveat: colour histograms separate distinctly-coloured relics well but can be ambiguous for
relics that share a palette; augment with a shape/NCC pass for those (TODO in classify()).
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

_H_BINS, _S_BINS = 30, 32
_ALPHA_MIN = 30        # min template alpha to count a pixel as "art"
_CROP_V_MIN = 50       # min crop brightness to count a pixel (drops dark HUD background)

# category -> path prefix inside the jar. Only direct children of the prefix are used
# (so images/relics/outline/* and per-shape potion layers are excluded).
_JAR_PREFIX: dict[str, str] = {
    "relics": "images/relics/",
    "intents": "images/ui/intent/",
}
# relic ids in the jar that aren't real relics
_SKIP_IDS = {f"test{i}" for i in range(1, 9)} | {"dummy", "cantUseRelic", "outline"}


class TemplateDB:
    """Histogram template DB sourced from a jar (zip) or a directory of PNGs."""

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)
        self.is_jar = self.source.is_file() and self.source.suffix.lower() in {".jar", ".zip"}
        # category -> list of (id, hs_histogram)
        self._cache: dict[str, list[tuple[str, "np.ndarray"]]] = {}

    def available(self, category: str) -> bool:
        if self.is_jar:
            return bool(next(self._iter_raw(category), None))
        return (self.source / category).is_dir() and any((self.source / category).glob("*.png"))

    # --- raw image iteration (jar or dir) --------------------------------
    def _iter_raw(self, category: str) -> Iterator[tuple[str, bytes]]:
        if self.is_jar:
            prefix = _JAR_PREFIX.get(category)
            if not prefix:
                return
            with zipfile.ZipFile(self.source) as jar:
                for name in jar.namelist():
                    if not (name.startswith(prefix) and name.lower().endswith(".png")):
                        continue
                    rel = name[len(prefix):]
                    if "/" in rel:  # skip nested dirs (e.g. relics/outline/*)
                        continue
                    stem = rel[:-4]
                    if stem in _SKIP_IDS:
                        continue
                    yield stem, jar.read(name)
        else:
            cat_dir = self.source / category
            if cat_dir.is_dir():
                for png in sorted(cat_dir.glob("*.png")):
                    if png.stem not in _SKIP_IDS:
                        yield png.stem, png.read_bytes()

    # --- histogram helpers ------------------------------------------------
    @staticmethod
    def _composite_on_black(rgba: "np.ndarray") -> "np.ndarray":
        import numpy as np

        if rgba.ndim == 3 and rgba.shape[2] == 4:
            a = rgba[:, :, 3:4].astype(np.float32) / 255.0
            return (rgba[:, :, :3].astype(np.float32) * a).astype(np.uint8)
        return rgba

    @staticmethod
    def _hs_hist(bgr: "np.ndarray", mask: "np.ndarray | None") -> "np.ndarray":
        import cv2

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [_H_BINS, _S_BINS], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _load_category(self, category: str) -> list[tuple[str, "np.ndarray"]]:
        if category in self._cache:
            return self._cache[category]
        import cv2
        import numpy as np

        entries: list[tuple[str, "np.ndarray"]] = []
        for stem, data in self._iter_raw(category):
            rgba = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                continue
            bgr = self._composite_on_black(rgba)
            mask = None
            if rgba.ndim == 3 and rgba.shape[2] == 4:
                mask = (rgba[:, :, 3] > _ALPHA_MIN).astype(np.uint8) * 255
            entries.append((stem, self._hs_hist(bgr, mask)))
        self._cache[category] = entries
        return entries

    def classify(self, crop: "np.ndarray", category: str) -> tuple[str, float]:
        """Return (best_id, correlation in [-1, 1]) for `crop` against `category`.

        TODO: for palette-ambiguous categories, take the top-k by histogram then
        disambiguate with a shape/NCC pass on alpha-aligned, scale-normalised icons.
        """
        import cv2

        entries = self._load_category(category)
        if not entries or crop is None or crop.size == 0:
            return "", 0.0
        bgr = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        value = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        mask = (value > _CROP_V_MIN).astype("uint8") * 255
        probe = self._hs_hist(bgr, mask)
        best_id, best_score = "", -1.0
        for tid, hist in entries:
            score = float(cv2.compareHist(probe, hist, cv2.HISTCMP_CORREL))
            if score > best_score:
                best_id, best_score = tid, score
        return best_id, best_score
