"""Static-art template database.

Slay the Spire's art is fixed, so identity recognition (which relic, which monster, which
intent icon) is best done by matching against the game's own images rather than OCR. This
loads a directory tree of reference PNGs:

    <templates_dir>/<category>/<id>.png      e.g. relics/BurningBlood.png

and classifies a crop by best normalized correlation. Build the tree with
``python -m tools.extract_assets``. The DB is optional: if a category is empty,
``classify`` returns ("", 0.0) and callers fall back to OCR / leave fields blank.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

# Match size: templates and crops are resized to this for correlation. Small = fast and
# tolerant of minor scale differences; large enough to keep distinguishing detail.
_MATCH_SIZE = (48, 48)


class TemplateDB:
    def __init__(self, templates_dir: str | Path) -> None:
        self.root = Path(templates_dir)
        # category -> list of (id, normalized_gray_vector)
        self._cache: dict[str, list[tuple[str, "np.ndarray"]]] = {}

    def available(self, category: str) -> bool:
        return (self.root / category).is_dir() and any((self.root / category).glob("*.png"))

    def _load_category(self, category: str) -> list[tuple[str, "np.ndarray"]]:
        if category in self._cache:
            return self._cache[category]
        import cv2

        entries: list[tuple[str, "np.ndarray"]] = []
        cat_dir = self.root / category
        if cat_dir.is_dir():
            for png in sorted(cat_dir.glob("*.png")):
                img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    entries.append((png.stem, self._normalize(img)))
        self._cache[category] = entries
        return entries

    @staticmethod
    def _normalize(gray: "np.ndarray") -> "np.ndarray":
        import cv2
        import numpy as np

        resized = cv2.resize(gray, _MATCH_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
        resized -= resized.mean()
        norm = np.linalg.norm(resized)
        return resized / norm if norm else resized

    def classify(self, crop: "np.ndarray", category: str) -> tuple[str, float]:
        """Return (best_id, score in [-1,1]) for `crop` against `category` templates."""
        import cv2

        entries = self._load_category(category)
        if not entries or crop is None or crop.size == 0:
            return "", 0.0
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        probe = self._normalize(gray)
        best_id, best_score = "", -1.0
        for tid, vec in entries:
            score = float((probe * vec).sum())  # cosine similarity (both unit-norm)
            if score > best_score:
                best_id, best_score = tid, score
        return best_id, best_score
