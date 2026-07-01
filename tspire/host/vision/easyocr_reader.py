"""Optional EasyOCR reader for Slay the Spire's *stylised* text.

Tesseract reads the flat UI text (HP bars, gold, floor) fine, but chokes on the game-art text:
card titles on their textured banner, the digits inside the energy orb, and the deck counter.
EasyOCR (deep-learning) reads those reliably and runs on CPU in well under a second per crop.

This module lazily builds a single shared reader and degrades to ``None`` when EasyOCR is not
installed, so the CV parser can fall back to Tesseract without a hard dependency. Heavy imports
happen only on first use.
"""

from __future__ import annotations

import logging

log = logging.getLogger("tspire.host.vision.easyocr")

_reader = None
_tried = False


def get_reader():
    """Return the shared EasyOCR reader, or None if EasyOCR isn't available."""
    global _reader, _tried
    if _tried:
        return _reader
    _tried = True
    try:
        import easyocr  # noqa: F401

        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        log.info("EasyOCR reader initialised (stylised-text OCR enabled)")
    except Exception:
        log.info("EasyOCR not available; falling back to Tesseract for stylised text")
        _reader = None
    return _reader


def available() -> bool:
    return get_reader() is not None


def read_boxes(image) -> list[tuple[float, float, str, float]]:
    """Detect text in ``image``; return ``[(cx, cy, text, conf)]`` in image-pixel coords.

    ``cx/cy`` are the centre of each detected word box. Returns ``[]`` when EasyOCR is
    unavailable or the crop is empty.
    """
    reader = get_reader()
    if reader is None or getattr(image, "size", 1) == 0:
        return []
    try:
        out: list[tuple[float, float, str, float]] = []
        for box, text, conf in reader.readtext(image, detail=1):
            cx = (box[0][0] + box[2][0]) / 2.0
            cy = (box[0][1] + box[2][1]) / 2.0
            out.append((float(cx), float(cy), str(text), float(conf)))
        return out
    except Exception:
        log.debug("EasyOCR read failed", exc_info=True)
        return []


def read_int(image) -> int:
    """Best single integer found in ``image`` (e.g. energy '3', deck '10'), or -1."""
    import re

    best = -1
    for _cx, _cy, text, _conf in read_boxes(image):
        m = re.search(r"\d+", text)
        if m:
            best = int(m.group())
            break
    return best
