import sys, os
sys.stdout.reconfigure(encoding="utf-8")

path = r"C:\Users\itsmo\TSpire\tspire\host\capture.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update ensure_foreground to call _click_safe_zone
old_ensure = """    def ensure_foreground(self) -> bool:
        \"\"\"Bring the game window to the foreground; return whether it actually became it.\"\"\"
        try:
            w = self._find_window()
        except Exception:
            return False
        self._activate(w)
        return self.is_foreground(w)"""

new_ensure = """    def ensure_foreground(self) -> bool:
        \"\"\"Bring the game window to the foreground; return whether it actually became it.

        When the window successfully becomes foreground, a safe-zone click is sent
        to force Windows to fully transfer input ownership. Without a real input
        event, Unity games (StS included) may ignore virtual gamepad input even
        though GetForegroundWindow() returns the game's hwnd.
        \"\"\"
        try:
            w = self._find_window()
        except Exception:
            return False
        self._activate(w)
        if not self.is_foreground(w):
            return False
        hwnd = getattr(w, "_hWnd", None)
        if hwnd:
            self._click_safe_zone(hwnd)
        return True"""

if old_ensure not in content:
    print("ERROR: Could not find old ensure_foreground")
    sys.exit(1)

content = content.replace(old_ensure, new_ensure, 1)

# 2. Insert _click_safe_zone between is_foreground and _force_foreground
old_boundary = """            return False

    @staticmethod
    def _force_foreground(hwnd) -> bool:"""

new_boundary = """            return False

    @staticmethod
    def _click_safe_zone(hwnd: int) -> None:
        \"\"\"Send a real left-click to a non-interactive corner of the window.

        Windows only fully transfers input focus when the target window receives a
        real input event.  Unity games (StS included) may ignore virtual gamepad
        input until this happens, even when GetForegroundWindow() returns their
        hwnd.

        The safe zone is the top-left corner of the client area, offset by a few
        pixels -- a region that does not overlap any interactive UI element in StS
        in any game state (combat, map, shop, rest, event).

        The cursor position is saved before the click and restored afterwards so
        the user does not notice the move.
        \"\"\"
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
                user32.mouse_event(0x0002, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTDOWN
                user32.mouse_event(0x0004, 0, 0, 0, 0)  # MOUSEEVENTF_LEFTUP
                log.debug("safe-zone click at (%d, %d)", safe_x, safe_y)
            finally:
                user32.SetCursorPos(cursor.x, cursor.y)
        except Exception:
            log.debug("safe-zone click failed", exc_info=True)

    @staticmethod
    def _force_foreground(hwnd) -> bool:"""

if old_boundary not in content:
    print("ERROR: Could not find boundary")
    sys.exit(1)

content = content.replace(old_boundary, new_boundary, 1)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("OK: capture.py updated")
