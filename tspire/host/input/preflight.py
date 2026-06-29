"""Read-only diagnostics for the input executor."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from tspire.host.capture import WindowCapture, WindowNotFoundError
from tspire.host.game_assets import find_game_jar


def collect_preflight_warnings(config, state_provider=None) -> list[str]:
    warnings: list[str] = []
    if not config.input_dry_run and importlib.util.find_spec("vgamepad") is None:
        warnings.append("vgamepad is not installed; real controller input will be unavailable")

    window_warning = _window_warning(config, state_provider)
    if window_warning:
        warnings.append(window_warning)

    prefs_warning = _controller_pref_warning(config)
    if prefs_warning:
        warnings.append(prefs_warning)

    return warnings


def _window_warning(config, state_provider) -> str | None:
    capture = getattr(state_provider, "capture", None)
    if capture is None:
        capture = WindowCapture(
            config.window_title,
            focus_before_capture=config.focus_before_capture,
        )
    try:
        capture.find_window()
    except WindowNotFoundError:
        return f"Slay the Spire window {config.window_title!r} was not found"
    except ModuleNotFoundError as exc:
        return f"cannot check Slay the Spire window because {exc.name} is unavailable"
    except Exception as exc:
        return f"cannot check Slay the Spire window: {exc}"
    return None


def _controller_pref_warning(config) -> str | None:
    try:
        jar = find_game_jar(config.jar_path)
    except Exception:
        return None
    if jar is None:
        return None
    prefs = Path(jar).parent / "preferences" / "STSGameplaySettings"
    if not prefs.is_file():
        return None
    try:
        data = json.loads(prefs.read_text(encoding="utf-8"))
    except Exception:
        return None
    enabled = str(data.get("Controller Enabled", "")).lower()
    if enabled == "false":
        return (
            "Slay the Spire preference 'Controller Enabled' is false; enable controller "
            "support in-game before sending real input"
        )
    return None
