"""Decompile-grounded controller setup diagnostic for the gamepad backend.

With Steam Input OFF, Slay the Spire enumerates DirectInput controllers exactly ONCE at
startup (``CInputHelper.initializeIfAble``) and binds ``controllers.first()`` -- it never
re-scans, and it name-matches "360"/"Xbox One". So the pad must be present before launch, be
the first controller, and not be hidden/remapped by Steam Input. This module checks those
conditions from the host and turns the silent "no controller detected" failure into an
explicit report.

``enumerate_controllers`` is Windows-only (winmm joystick API + the registry friendly name);
``analyze_controllers`` is a pure function so the logic is unit-testable without a device.
"""

from __future__ import annotations

import sys


def enumerate_controllers() -> list[dict]:
    """Best-effort list of connected game controllers as ``[{"id", "name"}]`` (first-to-last).

    Uses the legacy winmm joystick API, which surfaces XInput/ViGEmBus pads too, plus the
    registry OEM name for a friendly label. Returns ``[]`` off Windows or on any failure.
    """
    if sys.platform != "win32":  # pragma: no cover - host is Windows
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class JOYCAPSW(ctypes.Structure):
            _fields_ = [
                ("wMid", wintypes.WORD),
                ("wPid", wintypes.WORD),
                ("szPname", wintypes.WCHAR * 32),
                ("wXmin", wintypes.UINT), ("wXmax", wintypes.UINT),
                ("wYmin", wintypes.UINT), ("wYmax", wintypes.UINT),
                ("wZmin", wintypes.UINT), ("wZmax", wintypes.UINT),
                ("wNumButtons", wintypes.UINT),
                ("wPeriodMin", wintypes.UINT), ("wPeriodMax", wintypes.UINT),
                ("wRmin", wintypes.UINT), ("wRmax", wintypes.UINT),
                ("wUmin", wintypes.UINT), ("wUmax", wintypes.UINT),
                ("wVmin", wintypes.UINT), ("wVmax", wintypes.UINT),
                ("wCaps", wintypes.UINT),
                ("wMaxAxes", wintypes.UINT), ("wNumAxes", wintypes.UINT), ("wMaxButtons", wintypes.UINT),
                ("szRegKey", wintypes.WCHAR * 32),
                ("szOEMVxD", wintypes.WCHAR * 260),
            ]

        winmm = ctypes.windll.winmm
        num = int(winmm.joyGetNumDevs())
        found: list[dict] = []
        for i in range(num):
            caps = JOYCAPSW()
            if winmm.joyGetDevCapsW(i, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:  # JOYERR_NOERROR
                name = _friendly_name(caps.szRegKey) or caps.szPname or f"controller {i}"
                found.append({"id": i, "name": name})
        return found
    except Exception:  # pragma: no cover - platform/driver dependent
        return []


def _friendly_name(reg_key: str) -> str | None:
    if not reg_key:
        return None
    try:
        import winreg

        path = (
            r"System\CurrentControlSet\Control\MediaProperties\PrivateProperties"
            r"\Joystick\OEM\%s" % reg_key
        )
        for root in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            try:
                with winreg.OpenKey(root, path) as key:
                    value, _ = winreg.QueryValueEx(key, "OEMName")
                    if value:
                        return str(value)
            except OSError:
                continue
    except Exception:  # pragma: no cover
        return None
    return None


def _is_xbox(name: str) -> bool:
    n = name.lower()
    return "360" in n or "xbox" in n


def analyze_controllers(controllers: list[dict]) -> list[str]:
    """Turn a controller list into human-readable setup warnings (empty controllers-list too).

    Pure function so it can be unit-tested. Each message explains a way StS's one-shot,
    first-only, name-matched DirectInput binding can silently fail.
    """
    if not controllers:
        return [
            "No game controllers are visible to the OS. For the gamepad backend, confirm "
            "vgamepad/ViGEmBus created the virtual pad and that Steam Input is not hiding it."
        ]

    names = [c["name"] for c in controllers]
    msgs: list[str] = []

    steamish = [c for c in controllers if "steam" in c["name"].lower()]
    if steamish:
        msgs.append(
            f"A Steam Input controller is present ({steamish[0]['name']!r}). Steam Input is "
            "likely intercepting the pad -- StS then reads via Steam's API instead of "
            "DirectInput and the vgamepad buttons won't map as expected. Disable Steam Input "
            "for Slay the Spire (right-click the game -> Properties -> Controller)."
        )

    xbox = [c for c in controllers if _is_xbox(c["name"])]
    if not xbox:
        msgs.append(
            f"No Xbox/360-style controller found (saw {names}). StS name-matches '360'/'Xbox "
            "One' and vgamepad emulates an Xbox 360 pad, so the virtual pad does not appear "
            "to be present."
        )
    else:
        if not _is_xbox(controllers[0]["name"]):
            msgs.append(
                f"The FIRST controller is {controllers[0]['name']!r}, not the Xbox pad. StS "
                "binds controllers.first(), so it may pick the wrong device -- unplug other "
                "controllers so the virtual pad is first."
            )
        if len(controllers) > 1:
            msgs.append(
                f"{len(controllers)} controllers present ({names}); StS binds only the first. "
                "Remove extras so the virtual pad is the one that gets bound."
            )

    if not msgs:
        msgs.append(
            f"Controller setup looks OK: {names[0]!r} is the first (and only Xbox-style) "
            "controller, and no Steam Input pad was seen."
        )
    return msgs


def collect_controller_warnings() -> list[str]:
    """Enumerate + analyze in one call (used by preflight)."""
    return analyze_controllers(enumerate_controllers())
