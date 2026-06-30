"""Low-level vision primitives.

`VisionBackend` is the interface the parsers use; `CvVisionBackend` is the real
OpenCV + Tesseract implementation. Keeping parsers behind this interface lets the combat
assembly logic be unit-tested with a fake backend (no native deps, no live game).

Heavy deps (cv2, numpy, pytesseract) are imported lazily inside the concrete backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tspire.host.vision.regions import Rect

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

_INT_RE = re.compile(r"-?\d+")
_PAIR_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


@dataclass(frozen=True)
class BBox:
    """A pixel-space bounding box (left, top, width, height)."""

    left: int
    top: int
    width: int
    height: int

    @property
    def cx(self) -> int:
        return self.left + self.width // 2

    @property
    def cy(self) -> int:
        return self.top + self.height // 2


@runtime_checkable
class VisionBackend(Protocol):
    def ocr_text(self, frame: "np.ndarray", rect: Rect, *, digits: bool = False) -> str: ...

    def ocr_int(self, frame: "np.ndarray", rect: Rect, *, default: int = 0) -> int: ...

    def ocr_int_pair(self, frame: "np.ndarray", rect: Rect) -> tuple[int, int]: ...

    def find_red_bars(self, frame: "np.ndarray", search: Rect) -> list[BBox]: ...

    def find_cards(self, frame: "np.ndarray", search: Rect) -> list[BBox]: ...

    def crop_px(self, frame: "np.ndarray", box: BBox) -> "np.ndarray": ...

    def classify_box(
        self, frame: "np.ndarray", box: BBox, category: str
    ) -> tuple[str, float]: ...

    def region_filled(self, frame: "np.ndarray", rect: Rect, *, min_std: float = 12.0) -> bool: ...


class CvVisionBackend:
    """OpenCV + Tesseract implementation."""

    def __init__(self, tesseract_cmd: str = "", templates=None) -> None:
        # pytesseract is imported lazily (in OCR methods) so the LLM vision mode, and the
        # calibrate/classify paths, run on OpenCV alone without requiring Tesseract.
        self.tesseract_cmd = tesseract_cmd
        self._tesseract_ready = False
        self.templates = templates  # optional TemplateDB for classify_box

    def _ensure_tesseract(self) -> None:
        if self._tesseract_ready:
            return
        import pytesseract

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        self._tesseract_ready = True

    # --- cropping ---------------------------------------------------------
    @staticmethod
    def _crop(frame: "np.ndarray", rect: Rect) -> "np.ndarray":
        h, w = frame.shape[:2]
        left, top, width, height = rect.to_pixels(w, h)
        return frame[top : top + height, left : left + width]

    # --- OCR --------------------------------------------------------------
    def _preprocess(self, img: "np.ndarray") -> "np.ndarray":
        """Upscale + binarize for crisper OCR of StS's light-on-dark text."""
        import cv2

        if img.size == 0:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # StS text is light on a darker background; Otsu then ensure dark-on-light for OCR.
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th.mean() < 127:  # mostly dark -> invert so text is dark on light
            th = cv2.bitwise_not(th)
        return th

    def ocr_text(self, frame: "np.ndarray", rect: Rect, *, digits: bool = False) -> str:
        import pytesseract

        self._ensure_tesseract()
        img = self._preprocess(self._crop(frame, rect))
        if img.size == 0:
            return ""
        config = "--psm 7"
        if digits:
            config += " -c tessedit_char_whitelist=0123456789/"
        return pytesseract.image_to_string(img, config=config).strip()

    def ocr_int(self, frame: "np.ndarray", rect: Rect, *, default: int = 0) -> int:
        text = self.ocr_text(frame, rect, digits=True)
        m = _INT_RE.search(text)
        return int(m.group()) if m else default

    def ocr_int_pair(self, frame: "np.ndarray", rect: Rect) -> tuple[int, int]:
        """Parse a 'current/max' field (HP, energy). Falls back to (n, n) or (0, 0)."""
        text = self.ocr_text(frame, rect, digits=True)
        m = _PAIR_RE.search(text)
        if m:
            return int(m.group(1)), int(m.group(2))
        nums = _INT_RE.findall(text)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        if nums:
            return int(nums[0]), int(nums[0])
        return 0, 0

    # --- color / content detection ---------------------------------------
    def find_red_bars(self, frame: "np.ndarray", search: Rect) -> list[BBox]:
        """Find monster HP bars (red horizontal bars) within `search`.

        Returns boxes in frame-space (search offset added back), sorted left-to-right.
        Thresholds/min-size are calibration estimates; tune with real captures.
        """
        import cv2
        import numpy as np

        roi = self._crop(frame, search)
        if roi.size == 0:
            return []
        h, w = frame.shape[:2]
        off_left, off_top, _, _ = search.to_pixels(w, h)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Pure, saturated red only. Narrow hue at both wrap-around ends excludes the
        # orange torch flames (hue ~10-25) that the old wider range caught.
        mask = cv2.inRange(hsv, (0, 130, 70), (8, 255, 255)) | cv2.inRange(
            hsv, (172, 130, 70), (180, 255, 255)
        )
        # Close horizontally to bridge the bright/depleted split within one HP bar.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 31), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bars: list[BBox] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # HP bars are wide, THIN (~20px), horizontal strips on the ground line. Size
            # floors are relative to the whole FRAME (not the search region) so a wider
            # monster_search doesn't drop smaller enemies' shorter bars. The thin max-height
            # rejects block/status blobs that morphology-merge into a fat red region.
            if cw < 0.045 * w or not (0.010 * h <= ch <= 0.035 * h):
                continue
            if cw / max(ch, 1) < 5.0:
                continue
            bars.append(BBox(left=off_left + x, top=off_top + y, width=cw, height=ch))
        bars.sort(key=lambda b: b.left)
        return bars

    def find_cards(self, frame: "np.ndarray", search: Rect) -> list[BBox]:
        """Locate hand cards within `search`, returning boxes left-to-right.

        Cards are bright, portrait-oriented rectangles fanned along the bottom. We
        threshold for bright regions and keep card-shaped contours. Aspect/size limits
        are calibration estimates; the fan's overlap and rotation mean this needs tuning
        against real captures (consider switching to cost-gem detection if overlap is bad).
        """
        import cv2
        import numpy as np

        roi = self._crop(frame, search)
        if roi.size == 0:
            return []
        h, w = frame.shape[:2]
        off_left, off_top, _, _ = search.to_pixels(w, h)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        # Open to drop speckle, then close vertically to fuse each card's interior.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_h, roi_w = roi.shape[:2]
        cards: list[BBox] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Cards are the tall bright regions in the hand band. Overlapping fans can
            # make them wider/narrower than ideal, so keep the aspect window loose.
            if ch < 0.25 * roi_h or cw < 0.05 * roi_w:
                continue
            if cw / max(ch, 1) > 1.3:  # reject wide blobs; allow portrait + slim slivers
                continue
            cards.append(BBox(left=off_left + x, top=off_top + y, width=cw, height=ch))
        cards.sort(key=lambda b: b.left)
        return cards

    def crop_px(self, frame: "np.ndarray", box: BBox) -> "np.ndarray":
        return frame[box.top : box.top + box.height, box.left : box.left + box.width]

    def classify_box(self, frame: "np.ndarray", box: BBox, category: str) -> tuple[str, float]:
        if self.templates is None:
            return "", 0.0
        return self.templates.classify(self.crop_px(frame, box), category)

    def region_filled(self, frame: "np.ndarray", rect: Rect, *, min_std: float = 12.0) -> bool:
        """Heuristic: is there meaningful (non-flat) content in this region?

        Used by the classifier (e.g. is the end-turn button present) and to decide whether
        a block badge is showing. Flat/empty regions have low pixel variance.
        """
        img = self._crop(frame, rect)
        if img.size == 0:
            return False
        return float(img.std()) >= min_std
