"""Gamepad driver abstraction.

The rest of the host talks in stable semantic tokens ("select", "left", "proceed")
instead of vgamepad constants. Real input is lazy-imported so non-host environments can
still import and test the executor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("tspire.host.input")


class InputUnavailable(RuntimeError):
    """Raised when real gamepad input cannot be sent."""


@dataclass(frozen=True)
class InputTiming:
    press_seconds: float = 0.06
    step_delay: float = 0.08
    settle_seconds: float = 0.25
    command_timeout: float = 30.0

    @classmethod
    def from_config(cls, config) -> "InputTiming":
        return cls(
            press_seconds=max(0.0, float(config.input_press_seconds)),
            step_delay=max(0.0, float(config.input_step_delay)),
            settle_seconds=max(0.0, float(config.input_settle_seconds)),
            command_timeout=max(0.1, float(config.input_command_timeout)),
        )


class GamepadDriver(Protocol):
    """Small interface used by the command executor."""

    available: bool
    diagnostic: str | None

    def press(self, token: str, duration: float | None = None) -> None: ...

    def close(self) -> None: ...


_ALIASES = {
    "a": "select",
    "select": "select",
    "confirm": "select",
    "b": "cancel",
    "back": "cancel",
    "cancel": "cancel",
    "return": "cancel",
    "x": "view",
    "view": "view",
    "y": "proceed",
    "proceed": "proceed",
    "end": "proceed",
    "end_turn": "proceed",
    "lb": "page_left",
    "left_bumper": "page_left",
    "page_left": "page_left",
    "rb": "page_right",
    "right_bumper": "page_right",
    "page_right": "page_right",
    "map": "map",
    "settings": "settings",
    "start": "settings",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "ls_left": "left",
    "ls_right": "right",
    "ls_up": "up",
    "ls_down": "down",
    "dpad_left": "dpad_left",
    "dpad_right": "dpad_right",
    "dpad_up": "dpad_up",
    "dpad_down": "dpad_down",
    "rs_left": "rs_left",
    "rs_right": "rs_right",
    "rs_up": "rs_up",
    "rs_down": "rs_down",
}

CANONICAL_TOKENS = tuple(sorted(set(_ALIASES.values())))

_BUTTON_MEMBERS = {
    "select": "XUSB_GAMEPAD_A",
    "cancel": "XUSB_GAMEPAD_B",
    "view": "XUSB_GAMEPAD_X",
    "proceed": "XUSB_GAMEPAD_Y",
    "page_left": "XUSB_GAMEPAD_LEFT_SHOULDER",
    "page_right": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "map": "XUSB_GAMEPAD_BACK",
    "settings": "XUSB_GAMEPAD_START",
    "dpad_up": "XUSB_GAMEPAD_DPAD_UP",
    "dpad_down": "XUSB_GAMEPAD_DPAD_DOWN",
    "dpad_left": "XUSB_GAMEPAD_DPAD_LEFT",
    "dpad_right": "XUSB_GAMEPAD_DPAD_RIGHT",
}

_AXES = {
    "left": ("left", -32768, 0),
    "right": ("left", 32767, 0),
    "up": ("left", 0, 32767),
    "down": ("left", 0, -32768),
    "rs_left": ("right", -32768, 0),
    "rs_right": ("right", 32767, 0),
    "rs_up": ("right", 0, 32767),
    "rs_down": ("right", 0, -32768),
}


def normalize_token(token: str) -> str:
    key = token.strip().lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(CANONICAL_TOKENS)
        raise ValueError(f"unknown gamepad token {token!r}; allowed: {allowed}") from exc


class DryRunDriver:
    """Records normalized tokens without touching a real controller."""

    available = True
    diagnostic = "input dry-run enabled"

    def __init__(self, timing: InputTiming | None = None) -> None:
        self.timing = timing or InputTiming()
        self.presses: list[str] = []

    def press(self, token: str, duration: float | None = None) -> None:
        normalized = normalize_token(token)
        self.presses.append(normalized)
        log.info("dry-run gamepad press: %s", normalized)

    def close(self) -> None:
        pass


class DisabledDriver:
    """Driver used when real input is requested but unavailable."""

    available = False

    def __init__(self, diagnostic: str) -> None:
        self.diagnostic = diagnostic

    def press(self, token: str, duration: float | None = None) -> None:
        raise InputUnavailable(self.diagnostic)

    def close(self) -> None:
        pass


class VGamepadDriver:
    """vgamepad-backed virtual Xbox 360 controller."""

    available = True
    diagnostic = None

    def __init__(self, timing: InputTiming | None = None) -> None:
        self.timing = timing or InputTiming()
        try:
            import vgamepad as vg
        except Exception as exc:  # pragma: no cover - depends on host install
            raise InputUnavailable(
                "vgamepad is not installed; install host extras and ViGEmBus"
            ) from exc
        try:
            self._vg = vg
            self._gamepad = vg.VX360Gamepad()
            self._gamepad.update()
        except Exception as exc:  # pragma: no cover - depends on host driver
            raise InputUnavailable(
                "could not create virtual Xbox controller; verify ViGEmBus is installed"
            ) from exc

    def press(self, token: str, duration: float | None = None) -> None:
        normalized = normalize_token(token)
        hold = self.timing.press_seconds if duration is None else max(0.0, duration)
        if normalized in _BUTTON_MEMBERS:
            self._press_button(_BUTTON_MEMBERS[normalized], hold)
        elif normalized in _AXES:
            self._press_axis(_AXES[normalized], hold)
        else:  # defensive; normalize_token should prevent this.
            raise ValueError(f"unsupported gamepad token {normalized!r}")
        if self.timing.step_delay:
            time.sleep(self.timing.step_delay)

    def close(self) -> None:
        try:
            self._gamepad.reset()
            self._gamepad.update()
        except Exception:
            pass

    def _press_button(self, member: str, duration: float) -> None:
        button = getattr(self._vg.XUSB_BUTTON, member)
        self._gamepad.press_button(button=button)
        self._gamepad.update()
        if duration:
            time.sleep(duration)
        self._gamepad.release_button(button=button)
        self._gamepad.update()

    def _press_axis(self, axis: tuple[str, int, int], duration: float) -> None:
        stick, x, y = axis
        if stick == "left":
            self._gamepad.left_joystick(x_value=x, y_value=y)
        else:
            self._gamepad.right_joystick(x_value=x, y_value=y)
        self._gamepad.update()
        if duration:
            time.sleep(duration)
        if stick == "left":
            self._gamepad.left_joystick(x_value=0, y_value=0)
        else:
            self._gamepad.right_joystick(x_value=0, y_value=0)
        self._gamepad.update()


def build_driver(config, timing: InputTiming) -> GamepadDriver:
    if config.input_dry_run:
        return DryRunDriver(timing)
    try:
        return VGamepadDriver(timing)
    except InputUnavailable as exc:
        return DisabledDriver(str(exc))
