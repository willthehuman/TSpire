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


# Gap-detection tunables (measured against real controller-mode frames). In StS controller
# mode the focused card lifts OUT of the hand row to a preview position, leaving its bottom
# slot nearly empty while every other slot still shows a card. So the focused slot is the one
# whose hand-row band is empty. Deliberately module constants for easy calibration.
_HAND_SLOT_HALF = 0.45  # per-slot column half-width as a fraction of slot width
# Measure only the LOWER strip of the hand band. The lifted/focused card floats up to a
# preview spot and can overlap the upper part of its own (or a neighbour's) slot, but it
# clears the very bottom of the row -- so the genuine gap shows there even at the hand edges.
_HAND_ROW_LOWER = 0.55  # start measuring this fraction of the way down the hand band
_HAND_BRIGHT_V = 110  # a pixel counts as card content when its max channel exceeds this
_HAND_GAP_MAX_PRESENCE = 0.20  # focused slot's bright fraction must be below this (near-empty)
_HAND_GAP_RATIO = 0.5  # ...and at most this fraction of the next-emptiest slot's presence
_HAND_MIN_CARD_PRESENCE = 0.20  # the other slots must actually hold cards (reject empty frames)


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
    """Pick the focused hand slot by which one is *missing* from the hand row.

    The focused card lifts to a preview position, so its hand-row slot is nearly empty while
    the others still hold cards. We measure card presence (bright fraction) per slot in the
    hand band and return the clear gap, or None when no slot is distinctly empty.
    """
    h, w = frame.shape[:2]
    left, top, width, height = regions.hand_search.to_pixels(w, h)
    slot_w = width / max(hand_count, 1)
    band_top = top + round(height * _HAND_ROW_LOWER)
    band_height = max(1, top + height - band_top)
    presence: list[float] = []
    for i in range(hand_count):
        cx = left + (i + 0.5) * slot_w
        box = _SimpleBox(
            left=round(cx - slot_w * _HAND_SLOT_HALF),
            top=band_top,
            width=round(slot_w * 2 * _HAND_SLOT_HALF),
            height=band_height,
        )
        crop = _crop(frame, _clamp_box(box, w, h))
        presence.append(_slot_presence(crop))
    return _gap_index(presence)


def _slot_presence(crop) -> float:
    """Fraction of bright (card) pixels in a hand slot; ~0 when the card has lifted away."""
    if crop is None or getattr(crop, "size", 0) == 0:
        return 0.0
    return float((crop.max(axis=2) > _HAND_BRIGHT_V).mean())


def _gap_index(presence: list[float]) -> int | None:
    if len(presence) < 2:
        return None
    order = sorted(range(len(presence)), key=presence.__getitem__)
    emptiest, runner_up = order[0], order[1]
    if presence[runner_up] < _HAND_MIN_CARD_PRESENCE:
        return None  # the other slots are empty too -> not a real hand (e.g. blank frame)
    if presence[emptiest] >= _HAND_GAP_MAX_PRESENCE:
        return None  # no slot is empty -> nothing focused (or detection failed)
    if presence[emptiest] > presence[runner_up] * _HAND_GAP_RATIO:
        return None  # gap not distinct enough
    return emptiest


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
