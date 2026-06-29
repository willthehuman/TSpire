import numpy as np

from tspire.host.input.focus import _focused_card_index_by_slots
from tspire.host.vision.regions import RegionMap


def test_focused_card_index_by_slots_detects_cyan_glow():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    regions = RegionMap()
    left, _, width, _ = regions.hand_search.to_pixels(1920, 1080)
    slot_w = width / 5
    cx = round(left + (4 + 0.5) * slot_w)
    frame[650:1040, cx - 45 : cx + 45] = (220, 220, 40)  # BGR cyan-ish glow

    assert _focused_card_index_by_slots(frame, regions, 5) == 4


def test_focused_card_index_by_slots_returns_none_without_glow():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert _focused_card_index_by_slots(frame, RegionMap(), 5) is None
