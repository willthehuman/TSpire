"""Throwaway: verify WindowCapture.ensure_foreground() reliably reclaims the StS window."""
import ctypes

from tspire.host.capture import WindowCapture
from tspire.host.config import HostConfig

cfg = HostConfig.load()
cap = WindowCapture(cfg.window_title, focus_before_capture=cfg.focus_before_capture)
user32 = ctypes.windll.user32
w = cap._find_window()
print("StS hwnd:", w._hWnd)
for i in range(3):
    before = user32.GetForegroundWindow()
    ok = cap.ensure_foreground()
    print(f"run {i}: fg_before={before} ok={ok} is_foreground_now={cap.is_foreground(w)}")
