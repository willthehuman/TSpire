"""Focus observation for closed-loop gamepad navigation.

The executor can be tested with a fake observer, but the host uses this best-effort CV
observer to verify which hand card or monster is currently focused. Unknown focus is a
valid result; callers should fail safely when verification is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("tspire.host.input.focus")


@dataclass(frozen=True)
class FocusState:
    hand_index: int | None = None
    target_index: int | None = None


class FocusObserver(Protocol):
    def observe(
        self,
        *,
        hand_count: int | None = None,
        target_count: int | None = None,
    ) -> FocusState: ...


class NullFocusObserver:
    def observe(
        self,
        *,
        hand_count: int | None = None,
        target_count: int | None = None,
    ) -> FocusState:
        return FocusState()


class ScreenFocusObserver:
    """Observe StS controller focus from the current captured frame.

    This intentionally reuses the existing ScreenStateProvider collaborators when they
    are present. The heavy OpenCV/numpy stack is already required for host vision, so no
    additional import is needed here.
    """

    def __init__(self, state_provider) -> None:
        self.state_provider = state_provider

    def observe(
        self,
        *,
        hand_count: int | None = None,
        target_count: int | None = None,
    ) -> FocusState:
        capture = getattr(self.state_provider, "capture", None)
        regions = getattr(self.state_provider, "regions", None)
        get_backend = getattr(self.state_provider, "_get_backend", None)
        if capture is None or regions is None or get_backend is None:
            return FocusState()
        try:
            frame = capture.grab()
            backend = get_backend()
            cards = backend.find_cards(frame, regions.hand_search)
            monsters = backend.find_red_bars(frame, regions.monster_search)
            return FocusState(
                hand_index=_focused_card_index(frame, cards, regions, hand_count=hand_count),
                target_index=_focused_monster_index(frame, monsters, target_count=target_count),
            )
        except Exception:
            log.debug("focus observation failed", exc_info=True)
            return FocusState()


def _focused_card_index(frame, boxes, regions=None, *, hand_count: int | None = None) -> int | None:
    if hand_count and hand_count > 0 and regions is not None:
        focused = _focused_card_index_by_slots(frame, regions, hand_count)
        if focused is not None:
            return focused
    if not boxes:
        return None
    scores = []
    h, w = frame.shape[:2]
    for box in boxes:
        crop = _crop(frame, _expand_box(box, w, h, pad_x=0.18, pad_y=0.10))
        scores.append(_cyan_score(crop))
    return _winner(scores, min_score=0.012, min_margin=1.25)


def _focused_card_index_by_slots(frame, regions, hand_count: int) -> int | None:
    h, w = frame.shape[:2]
    left, _, width, _ = regions.hand_search.to_pixels(w, h)
    # The focused card lifts far above the normal hand band, so score from mid-screen
    # through the bottom, centered on each expected hand slot.
    slot_w = width / max(hand_count, 1)
    y_top = round(0.56 * h)
    y_bottom = h
    scores: list[float] = []
    for i in range(hand_count):
        cx = left + (i + 0.5) * slot_w
        box = _SimpleBox(
            left=round(cx - slot_w * 0.55),
            top=y_top,
            width=round(slot_w * 1.10),
            height=y_bottom - y_top,
        )
        crop = _crop(frame, _clamp_box(box, w, h))
        scores.append(_cyan_score(crop))
    return _winner(scores, min_score=0.060, min_margin=1.12)


def _focused_monster_index(frame, bars, *, target_count: int | None = None) -> int | None:
    if not bars:
        return None
    scores = []
    h, w = frame.shape[:2]
    for bar in bars:
        # The target cursor/focus glow appears around the enemy sprite above the HP bar.
        box = _SimpleBox(
            left=bar.left - int(bar.width * 0.4),
            top=max(0, bar.top - int(bar.width * 1.9)),
            width=int(bar.width * 1.8),
            height=int(bar.width * 1.8),
        )
        crop = _crop(frame, _clamp_box(box, w, h))
        scores.append(max(_cyan_score(crop), _green_yellow_score(crop)))
    return _winner(scores, min_score=0.010, min_margin=1.20)


@dataclass(frozen=True)
class _SimpleBox:
    left: int
    top: int
    width: int
    height: int


def _expand_box(box, w: int, h: int, *, pad_x: float, pad_y: float) -> _SimpleBox:
    dx = int(box.width * pad_x)
    dy = int(box.height * pad_y)
    return _clamp_box(
        _SimpleBox(box.left - dx, box.top - dy, box.width + 2 * dx, box.height + 2 * dy),
        w,
        h,
    )


def _clamp_box(box, w: int, h: int) -> _SimpleBox:
    left = max(0, min(w, int(box.left)))
    top = max(0, min(h, int(box.top)))
    right = max(left, min(w, int(box.left + box.width)))
    bottom = max(top, min(h, int(box.top + box.height)))
    return _SimpleBox(left, top, right - left, bottom - top)


def _crop(frame, box: _SimpleBox):
    if box.width <= 0 or box.height <= 0:
        return None
    return frame[box.top : box.top + box.height, box.left : box.left + box.width]


def _cyan_score(crop) -> float:
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    b = crop[:, :, 0].astype("int16")
    g = crop[:, :, 1].astype("int16")
    r = crop[:, :, 2].astype("int16")
    mask = (b > 120) & (g > 120) & (r < 180) & ((b + g - 2 * r) > 80)
    return float(mask.mean())


def _green_yellow_score(crop) -> float:
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    b = crop[:, :, 0].astype("int16")
    g = crop[:, :, 1].astype("int16")
    r = crop[:, :, 2].astype("int16")
    mask = (g > 130) & (r > 70) & (b < 170) & ((g - b) > 45)
    return float(mask.mean())


def _winner(scores: list[float], *, min_score: float, min_margin: float) -> int | None:
    if not scores:
        return None
    best = max(range(len(scores)), key=scores.__getitem__)
    best_score = scores[best]
    if best_score < min_score:
        return None
    others = [s for i, s in enumerate(scores) if i != best]
    if others and best_score < max(others) * min_margin:
        return None
    return best
