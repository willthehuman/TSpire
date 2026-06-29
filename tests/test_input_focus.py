import numpy as np

from tspire.host.input.focus import _focused_card_index_by_slots, _gap_index, _slot_presence
from tspire.host.vision.regions import RegionMap


def _hand_frame(focused_slot: int | None):
    """A synthetic 5-card hand in the hand-row band. The focused card has lifted away to a
    preview position, so its slot is empty; every other slot still shows a (bright) card."""
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    regions = RegionMap()
    left, top, width, height = regions.hand_search.to_pixels(1920, 1080)
    slot_w = width / 5
    for i in range(5):
        if i == focused_slot:
            continue  # focused card lifted out of the hand row -> empty slot
        cx = round(left + (i + 0.5) * slot_w)
        frame[top : top + height, cx - 50 : cx + 50] = (220, 220, 40)  # card present
    return frame, regions


def test_focused_card_index_by_slots_picks_the_empty_gap_slot():
    frame, regions = _hand_frame(focused_slot=2)
    assert _focused_card_index_by_slots(frame, regions, 5) == 2


def test_focused_card_index_by_slots_none_when_all_slots_full():
    frame, regions = _hand_frame(focused_slot=None)
    assert _focused_card_index_by_slots(frame, regions, 5) is None


def test_focused_card_index_by_slots_none_on_empty_frame():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert _focused_card_index_by_slots(frame, RegionMap(), 5) is None


def test_slot_presence_bright_vs_dark():
    assert _slot_presence(np.full((100, 100, 3), 220, np.uint8)) == 1.0
    assert _slot_presence(np.zeros((100, 100, 3), np.uint8)) == 0.0


def test_gap_index_requires_a_distinct_empty_slot():
    assert _gap_index([0.6, 0.6, 0.05, 0.6, 0.6]) == 2  # clear gap
    assert _gap_index([0.6, 0.5, 0.55, 0.6, 0.5]) is None  # no slot empty
    assert _gap_index([0.30, 0.5, 0.55, 0.6, 0.5]) is None  # emptiest not distinct enough
