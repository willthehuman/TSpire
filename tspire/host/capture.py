"""Screen capture: locate the Slay the Spire window and grab frames.

Uses `pygetwindow` to find the window rectangle and `mss` to grab pixels. Returns frames
as BGR numpy arrays (OpenCV's convention) so the vision code can use them directly.

Heavy deps (mss, numpy, pygetwindow) are imported lazily so this module can be imported
on machines without the host extras (e.g. for unit tests that feed in saved frames).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np


@dataclass(frozen=True)
class WindowRect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class WindowNotFoundError(RuntimeError):
    pass


class WindowCapture:
    """Finds and captures the game window by title substring."""

    def __init__(self, title_substring: str = "Slay the Spire", *, focus_before_capture: bool = False) -> None:
        self.title_substring = title_substring
        self.focus_before_capture = focus_before_capture
        self._sct = None  # lazily created mss instance

    def find_window(self) -> WindowRect:
        w = self._find_window()
        return WindowRect(left=w.left, top=w.top, width=w.width, height=w.height)

    def focus_window(self) -> WindowRect:
        """Bring the target window to the foreground and return its current rectangle."""
        w = self._find_window()
        try:
            if getattr(w, "isMinimized", False):
                w.restore()
                time.sleep(0.15)
            w.activate()
            time.sleep(0.35)
        except Exception:
            # Some Windows focus-stealing protections can reject activation; still return
            # the rect so callers can capture/log the current state.
            pass
        return WindowRect(left=w.left, top=w.top, width=w.width, height=w.height)

    def _find_window(self):
        import pygetwindow as gw

        matches = [
            w
            for w in gw.getAllWindows()
            if _title_matches(w.title or "", self.title_substring)
        ]
        # Prefer a visible, non-minimized window with a real size.
        matches = [w for w in matches if w.width > 0 and w.height > 0]
        if not matches:
            raise WindowNotFoundError(
                f"no window whose title contains {self.title_substring!r}"
            )
        matches.sort(key=_window_rank)
        return matches[0]

    def grab(self, rect: WindowRect | None = None) -> "np.ndarray":
        """Capture the window (or a given rect) as a BGR numpy array."""
        import mss
        import numpy as np

        if rect is None:
            rect = self.focus_window() if self.focus_before_capture else self.find_window()
        if self._sct is None:
            self._sct = mss.mss()
        monitor = {
            "left": rect.left,
            "top": rect.top,
            "width": rect.width,
            "height": rect.height,
        }
        shot = self._sct.grab(monitor)
        # mss gives BGRA; drop alpha -> BGR for OpenCV.
        frame = np.asarray(shot)[:, :, :3]
        return frame

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None


def _title_matches(title: str, wanted: str) -> bool:
    title_l = title.lower()
    wanted_l = wanted.lower()
    if title_l == wanted_l:
        return True
    if wanted_l not in title_l:
        return False
    # Avoid common non-game windows like Explorer folders named after the game.
    rejected = ("file explorer", "codex", "browser", "chrome", "edge")
    return not any(word in title_l for word in rejected)


def _window_rank(window) -> tuple[int, int, int]:
    title = (window.title or "").lower()
    exact = 0 if title == "slay the spire" else 1
    minimized = 1 if getattr(window, "isMinimized", False) else 0
    area = max(window.width, 0) * max(window.height, 0)
    return exact, minimized, -area
