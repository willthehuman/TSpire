"""Screen capture: locate the Slay the Spire window and grab frames.

Uses `pygetwindow` to find the window rectangle and `mss` to grab pixels. Returns frames
as BGR numpy arrays (OpenCV's convention) so the vision code can use them directly.

Heavy deps (mss, numpy, pygetwindow) are imported lazily so this module can be imported
on machines without the host extras (e.g. for unit tests that feed in saved frames).
"""

from __future__ import annotations

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

    def __init__(self, title_substring: str = "Slay the Spire") -> None:
        self.title_substring = title_substring
        self._sct = None  # lazily created mss instance

    def find_window(self) -> WindowRect:
        import pygetwindow as gw

        matches = [
            w
            for w in gw.getAllWindows()
            if self.title_substring.lower() in (w.title or "").lower()
        ]
        # Prefer a visible, non-minimized window with a real size.
        matches = [w for w in matches if w.width > 0 and w.height > 0]
        if not matches:
            raise WindowNotFoundError(
                f"no window whose title contains {self.title_substring!r}"
            )
        w = matches[0]
        return WindowRect(left=w.left, top=w.top, width=w.width, height=w.height)

    def grab(self, rect: WindowRect | None = None) -> "np.ndarray":
        """Capture the window (or a given rect) as a BGR numpy array."""
        import mss
        import numpy as np

        if rect is None:
            rect = self.find_window()
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
