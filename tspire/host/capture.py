"""Screen capture: locate the Slay the Spire window and grab frames.

Uses `pygetwindow` to find the window rectangle and `mss` to grab pixels. Returns frames
as BGR numpy arrays (OpenCV's convention) so the vision code can use them directly.

Heavy deps (mss, numpy, pygetwindow) are imported lazily so this module can be imported
on machines without the host extras (e.g. for unit tests that feed in saved frames).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

log = logging.getLogger("tspire.host.capture")


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
        return self._rect_of(self._find_window())

    def focus_window(self) -> WindowRect:
        """Bring the target window to the foreground and return its current rectangle."""
        w = self._find_window()
        self._activate(w)
        return self._rect_of(w)

    def ensure_foreground(self) -> bool:
        """Bring the game window to the foreground; return whether it actually became it.

        When the window successfully becomes foreground, a safe-zone click is sent
        to force Windows to fully transfer input ownership. Without a real input
        event, Unity games (StS included) may ignore virtual gamepad input even
        though GetForegroundWindow() returns the game's hwnd.
        """
        try:
            w = self._find_window()
        except Exception:
            return False
        if self.is_foreground(w) and not getattr(w, "isMinimized", False):
            return True
        self._activate(w)
        if not self.is_foreground(w):
            return False
        hwnd = getattr(w, "_hWnd", None)
        if hwnd:
            self._click_safe_zone(hwnd)
        return True

    def _activate(self, w) -> bool:
        # StS only accepts controller input while it is the true foreground window, and
        # plain pygetwindow .activate() loses to Windows' foreground-lock when called from a
        # background process. Use the AttachThreadInput + SetForegroundWindow workaround
        # first, falling back to pygetwindow.
        hwnd = getattr(w, "_hWnd", None)
        if hwnd and self._force_foreground(hwnd):
            time.sleep(0.1)
            return True
        try:
            if getattr(w, "isMinimized", False):
                w.restore()
                time.sleep(0.15)
            w.activate()
            time.sleep(0.35)
        except Exception:
            # Some Windows focus-stealing protections can reject activation; callers still
            # get a usable rect to capture/log the current state.
            pass
        return self.is_foreground(w)

    @staticmethod
    def is_foreground(w) -> bool:
        """True if the given window is currently the OS foreground window."""
        hwnd = getattr(w, "_hWnd", None)
        if not hwnd:
            return False
        try:
            import ctypes
            import ctypes.wintypes

            user32 = ctypes.windll.user32
            fg = user32.GetForegroundWindow()
            if fg == hwnd:
                return True
            # Check if the foreground window is a child/descendant of our window (GA_ROOT = 2)
            if user32.GetAncestor(fg, 2) == hwnd:
                return True
            # Check if it belongs to the same process
            fg_pid = ctypes.wintypes.DWORD()
            tgt_pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(fg, ctypes.byref(fg_pid))
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(tgt_pid))
            return fg_pid.value == tgt_pid.value and fg_pid.value != 0
        except Exception:
            return False

    @staticmethod
    def _click_safe_zone(hwnd: int) -> None:
        """Send a real left-click to a non-interactive corner of the window.

        Windows only fully transfers input focus when the target window receives a
        real input event.  Unity games (StS included) may ignore virtual gamepad
        input until this happens, even when GetForegroundWindow() returns their
        hwnd.

        The safe zone is the top-left corner of the client area, offset by a few
        pixels -- a region that does not overlap any interactive UI element in StS
        in any game state (combat, map, shop, rest, event).

        The cursor position is saved before the click and restored afterwards so
        the user does not notice the move.
        """
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            rc = wintypes.RECT()
            if not user32.GetClientRect(hwnd, ctypes.byref(rc)):
                return
            origin = wintypes.POINT(0, 0)
            if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
                return

            # Safe zone: small offset from client-area origin (non-interactive border).
            safe_x = origin.x + 2
            safe_y = origin.y + 2

            # Save current cursor position so we can restore it.
            cursor = wintypes.POINT()
            user32.GetCursorPos(ctypes.byref(cursor))

            try:
                user32.SetCursorPos(safe_x, safe_y)
                # Give OS/game time to register cursor position
                time.sleep(0.02)
                user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                time.sleep(0.01)
                user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
                # Let game process the click event at safe zone before we restore cursor
                time.sleep(0.03)
            finally:
                user32.SetCursorPos(cursor.x, cursor.y)
        except Exception:
            log.debug("safe-zone click failed", exc_info=True)

    @staticmethod
    def _force_foreground(hwnd) -> bool:
        """Force a window to the true foreground. StS only reads the controller while it is
        the foreground, and Windows' foreground-lock resists a background process, so this
        combines the AttachThreadInput trick, an ALT-key nudge, a z-order raise, and retries,
        verifying GetForegroundWindow after each attempt."""
        try:
            import ctypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
        except Exception:
            return False
        _SW_RESTORE, _SW_SHOW = 9, 5
        _HWND_TOP = 0
        _SWP_NOSIZE, _SWP_NOMOVE, _SWP_SHOWWINDOW = 0x0001, 0x0002, 0x0040
        _KEYEVENTF_KEYUP, _VK_MENU = 0x0002, 0x12
        try:
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, _SW_RESTORE)
                time.sleep(0.15)  # Let the window restore before attempting to focus
            for attempt in range(3):
                if user32.GetForegroundWindow() == hwnd:
                    return True
                cur_tid = kernel32.GetCurrentThreadId()
                fg = user32.GetForegroundWindow()
                fg_tid = user32.GetWindowThreadProcessId(fg, None)
                tgt_tid = user32.GetWindowThreadProcessId(hwnd, None)
                attached_fg = bool(fg_tid and fg_tid != cur_tid and user32.AttachThreadInput(cur_tid, fg_tid, True))
                attached_tgt = bool(tgt_tid and tgt_tid != cur_tid and user32.AttachThreadInput(cur_tid, tgt_tid, True))
                try:
                    # An input event lifts Windows' foreground-change lock so
                    # SetForegroundWindow takes effect; the z-order raise + active/focus
                    # calls make it stick.
                    user32.keybd_event(_VK_MENU, 0, 0, 0)
                    user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
                    user32.ShowWindow(hwnd, _SW_SHOW)
                    user32.BringWindowToTop(hwnd)
                    user32.SetWindowPos(hwnd, _HWND_TOP, 0, 0, 0, 0, _SWP_NOSIZE | _SWP_NOMOVE | _SWP_SHOWWINDOW)
                    user32.SetForegroundWindow(hwnd)
                    user32.SetActiveWindow(hwnd)
                    user32.SetFocus(hwnd)
                finally:
                    if attached_fg:
                        user32.AttachThreadInput(cur_tid, fg_tid, False)
                    if attached_tgt:
                        user32.AttachThreadInput(cur_tid, tgt_tid, False)
                if user32.GetForegroundWindow() == hwnd:
                    return True
                time.sleep(0.05)
            return user32.GetForegroundWindow() == hwnd
        except Exception:
            log.debug("force-foreground failed", exc_info=True)
            return False

    @staticmethod
    def _rect_of(w) -> WindowRect:
        return WindowRect(left=w.left, top=w.top, width=w.width, height=w.height)

    def _client_rect(self, w) -> WindowRect:
        """The window's client area (no title bar/borders) in screen coords.

        StS in windowed mode has a title bar; capturing the whole window rect would offset
        every fractional vision region. Falls back to the full window rect when the native
        handle or Win32 calls are unavailable (e.g. non-Windows, tests).
        """
        hwnd = getattr(w, "_hWnd", None)
        if hwnd:
            try:
                import ctypes
                from ctypes import wintypes

                user32 = ctypes.windll.user32
                rc = wintypes.RECT()
                if user32.GetClientRect(hwnd, ctypes.byref(rc)):
                    origin = wintypes.POINT(0, 0)
                    user32.ClientToScreen(hwnd, ctypes.byref(origin))
                    width = rc.right - rc.left
                    height = rc.bottom - rc.top
                    if width > 0 and height > 0:
                        return WindowRect(left=origin.x, top=origin.y, width=width, height=height)
            except Exception:
                log.debug("client-rect lookup failed; using full window rect", exc_info=True)
        return self._rect_of(w)

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
            w = self._find_window()
            if self.focus_before_capture:
                self._activate(w)
            rect = self._client_rect(w)
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
