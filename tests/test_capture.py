from tspire.host.capture import WindowCapture, WindowRect, _title_matches, _window_rank


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
