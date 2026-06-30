import numpy as np

from tspire.host.calibrate import _normalize_image_frame
from tspire.host.capture import (
    WindowCapture,
    WindowRect,
    normalize_frame_to_client,
    _title_matches,
    _window_rank,
)
from tspire.host.config import HostConfig


class _Window:
    def __init__(self, title, width=100, height=100, minimized=False):
        self.title = title
        self.width = width
        self.height = height
        self.isMinimized = minimized
        self.left = 10
        self.top = 20


def test_title_matches_exact_game_and_rejects_explorer():
    assert _title_matches("Slay the Spire", "Slay the Spire")
    assert not _title_matches("Slay the Spire - File Explorer", "Slay the Spire")


def test_window_rank_prefers_exact_large_visible_window():
    game = _Window("Slay the Spire", width=1920, height=1080)
    folder = _Window("Slay the Spire notes", width=2000, height=1200)
    minimized = _Window("Slay the Spire", width=1920, height=1080, minimized=True)

    assert sorted([folder, minimized, game], key=_window_rank)[0] is game


def test_client_rect_falls_back_to_full_window_without_handle():
    # A window object without a Win32 handle (_hWnd) -> use its full rect, no crash.
    cap = WindowCapture("Slay the Spire")
    win = _Window("Slay the Spire", width=1920, height=1080)
    assert cap._client_rect(win) == WindowRect(left=10, top=20, width=1920, height=1080)


def test_ensure_foreground_early_out():
    cap = WindowCapture("Slay the Spire")
    win = _Window("Slay the Spire", width=1920, height=1080)

    find_called = False
    is_fg_called = False
    activate_called = False

    def mock_find_window():
        nonlocal find_called
        find_called = True
        return win

    def mock_is_foreground(w):
        nonlocal is_fg_called
        is_fg_called = True
        return True

    def mock_activate(w):
        nonlocal activate_called
        activate_called = True
        return True

    cap._find_window = mock_find_window
    cap.is_foreground = mock_is_foreground
    cap._activate = mock_activate

    # When already foreground, it should return True immediately without calling _activate
    assert cap.ensure_foreground() is True
    assert find_called
    assert is_fg_called
    assert not activate_called


def test_normalize_image_frame_crops_framed_screenshot():
    frame = np.zeros((1107, 1922, 3), dtype=np.uint8)
    frame[27, 1] = (1, 2, 3)
    messages = []

    out = _normalize_image_frame(frame, HostConfig(width=1920, height=1080), report=messages.append)

    assert out.shape[:2] == (1080, 1920)
    assert tuple(out[0, 0]) == (1, 2, 3)
    assert "cropped framed image" in messages[0]


def test_normalize_frame_to_client_crops_like_live_capture():
    frame = np.zeros((1107, 1922, 3), dtype=np.uint8)
    frame[27, 1] = (4, 5, 6)

    out = normalize_frame_to_client(frame, 1920, 1080)

    assert out.shape[:2] == (1080, 1920)
    assert tuple(out[0, 0]) == (4, 5, 6)


def test_normalize_image_frame_keeps_exact_client_screenshot():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    messages = []

    out = _normalize_image_frame(frame, HostConfig(width=1920, height=1080), report=messages.append)

    assert out is frame
    assert messages == []


def test_normalize_image_frame_warns_on_non_matching_size():
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    messages = []

    out = _normalize_image_frame(frame, HostConfig(width=1920, height=1080), report=messages.append)

    assert out is frame
    assert "warning: image is 1600x900" in messages[0]
