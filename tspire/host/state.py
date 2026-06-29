"""ScreenStateProvider: capture -> classify -> parse -> GameState.

This is the real StateProvider wired into the server (replacing M0's stub). It owns the
window capture, the resolution's region map, the vision backend, and the template DB.
Reads are defensive: capture/parse failures degrade to an UNKNOWN state with a message
rather than raising, so the server loop keeps running.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tspire.common import protocol
from tspire.common.schema import GameState, ScreenType
from tspire.host.capture import WindowCapture, WindowNotFoundError
from tspire.host.config import HostConfig
from tspire.host.vision import region_map_for
from tspire.host.vision.combat import parse_combat

log = logging.getLogger("tspire.host.state")

# Default location of the extracted template DB (see tools/extract_assets.py).
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "assets" / "templates"


class ScreenStateProvider:
    def __init__(self, config: HostConfig) -> None:
        self.config = config
        self.capture = WindowCapture(config.window_title)
        self.regions = region_map_for(config.width, config.height)
        self._backend = None  # built lazily so import/tests don't require cv2/tesseract

    def _get_backend(self):
        if self._backend is None:
            from tspire.host.vision.backend import CvVisionBackend
            from tspire.host.vision.templates import TemplateDB

            templates = TemplateDB(_TEMPLATES_DIR)
            self._backend = CvVisionBackend(self.config.tesseract_cmd, templates=templates)
        return self._backend

    def read(self) -> GameState:
        try:
            frame = self.capture.grab()
        except WindowNotFoundError:
            return GameState(
                screen_type=ScreenType.NONE,
                screen_message=f"game window {self.config.window_title!r} not found",
                available_commands=protocol.commands_for_screen(ScreenType.NONE.value),
            )

        backend = self._get_backend()
        from tspire.host.classify import classify_screen

        screen = classify_screen(frame, self.regions, backend)
        if screen == ScreenType.COMBAT:
            return self._build_combat_state(frame, backend)

        return GameState(
            screen_type=screen,
            screen_message="screen not yet supported by parser (v1 = combat only)",
            available_commands=protocol.commands_for_screen(screen.value),
            gold=backend.ocr_int(frame, self.regions.gold),
        )

    def _build_combat_state(self, frame, backend) -> GameState:
        result = parse_combat(frame, self.regions, backend)
        player = result.combat.player
        return GameState(
            screen_type=ScreenType.COMBAT,
            in_combat=True,
            current_hp=player.current_hp,
            max_hp=player.max_hp,
            gold=backend.ocr_int(frame, self.regions.gold),
            combat_state=result.combat,
            available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
            parse_confidence=round(result.confidence, 2),
        )
