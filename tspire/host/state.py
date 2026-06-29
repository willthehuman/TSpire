"""ScreenStateProvider: capture -> classify -> parse -> GameState.

The real StateProvider wired into the server. Two parsing backends, chosen by
config.vision_mode:

  * "llm" (default) - OllamaVisionParser reads monsters + hand with a local vision model;
    fixed player stats come from upscaled-crop calls. Robust on the busy scene. Marked
    `expensive` so the server only reads on demand (connect / after a command / state
    request), never on the idle poll timer.
  * "cv" - the OpenCV + Tesseract parser (fast, but needs calibration + Tesseract).

Either way a cheap OpenCV check (energy + End-Turn regions filled) gates whether we are in
combat, so the expensive parse never runs off a non-combat screen.

Reads are defensive: capture/parse failures degrade to an UNKNOWN state with a message
rather than raising, so the server loop keeps running.
"""

from __future__ import annotations

import logging

from tspire.common import protocol
from tspire.common.schema import GameState, ScreenType
from tspire.host.capture import WindowCapture, WindowNotFoundError
from tspire.host.config import HostConfig
from tspire.host.game_assets import find_game_jar
from tspire.host.vision import region_map_for

log = logging.getLogger("tspire.host.state")


class ScreenStateProvider:
    def __init__(self, config: HostConfig) -> None:
        self.config = config
        self.capture = WindowCapture(
            config.window_title,
            focus_before_capture=config.focus_before_capture,
        )
        self.regions = region_map_for(config.width, config.height)
        # LLM parsing is slow (~seconds) -> the server must not poll it on a timer.
        self.expensive = config.vision_mode == "llm"
        self._backend = None  # OpenCV backend (classify, and cv-mode parsing)
        self._llm = None  # OllamaVisionParser (llm mode)

    def _get_backend(self):
        if self._backend is None:
            from tspire.host.vision.backend import CvVisionBackend
            from tspire.host.vision.templates import TemplateDB

            jar = find_game_jar(self.config.jar_path)
            if jar is None:
                log.warning("desktop-1.0.jar not found; relic/intent art unavailable. "
                            "Set jar_path in config or install the game.")
            templates = TemplateDB(jar) if jar else None
            self._backend = CvVisionBackend(self.config.tesseract_cmd, templates=templates)
        return self._backend

    def _get_llm(self):
        if self._llm is None:
            from tspire.host.vision.llm import OllamaVisionParser

            self._llm = OllamaVisionParser(
                model=self.config.ollama_model,
                url=self.config.ollama_url,
                regions=self.regions,
                image_width=self.config.llm_image_width,
            )
        return self._llm

    def read(self) -> GameState:
        try:
            frame = self.capture.grab()
        except WindowNotFoundError:
            return GameState(
                screen_type=ScreenType.NONE,
                screen_message=f"game window {self.config.window_title!r} not found",
                available_commands=protocol.commands_for_screen(ScreenType.NONE.value),
            )

        from tspire.host.classify import classify_screen

        backend = self._get_backend()
        screen = classify_screen(frame, self.regions, backend)
        if screen == ScreenType.COMBAT:
            return self._build_combat_state(frame, backend)

        return GameState(
            screen_type=screen,
            screen_message="screen not yet supported by parser (v1 = combat only)",
            available_commands=protocol.commands_for_screen(screen.value),
        )

    def _build_combat_state(self, frame, backend) -> GameState:
        if self.config.vision_mode == "llm":
            # Only pay for a block-reading call when a block badge is actually visible.
            read_block = backend.region_filled(frame, self.regions.player_block)
            result = self._get_llm().parse_combat(frame, read_block=read_block)
        else:
            from tspire.host.vision.combat import parse_combat

            result = parse_combat(frame, self.regions, backend)

        player = result.combat.player
        return GameState(
            screen_type=ScreenType.COMBAT,
            in_combat=True,
            current_hp=player.current_hp,
            max_hp=player.max_hp,
            combat_state=result.combat,
            available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
            parse_confidence=round(result.confidence, 2),
        )
