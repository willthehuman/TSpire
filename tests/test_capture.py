from tspire.host.capture import _title_matches, _window_rank


class _Window:
    def __init__(self, title, width=100, height=100, minimized=False):
        self.title = title
        self.width = width
        self.height = height
        self.isMinimized = minimized


def test_title_matches_exact_game_and_rejects_explorer():
    assert _title_matches("Slay the Spire", "Slay the Spire")
    assert not _title_matches("Slay the Spire - File Explorer", "Slay the Spire")


def test_window_rank_prefers_exact_large_visible_window():
    game = _Window("Slay the Spire", width=1920, height=1080)
    folder = _Window("Slay the Spire notes", width=2000, height=1200)
    minimized = _Window("Slay the Spire", width=1920, height=1080, minimized=True)

    assert sorted([folder, minimized, game], key=_window_rank)[0] is game
