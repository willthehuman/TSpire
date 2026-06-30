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
from tspire.host.capture import WindowCapture, WindowNotFoundError, normalize_frame_to_client
from tspire.host.config import HostConfig
from tspire.host.game_assets import find_game_jar
from tspire.host.predict import predict
from tspire.host.reconcile import reconcile
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
        self._last_state: GameState | None = None
        # (command, before_state) recorded by the input executor after a play/end-turn, so
        # the next read can predict-and-reconcile against it. One-shot: cleared once consumed.
        self._pending: tuple[protocol.Command, GameState] | None = None

    def note_action(self, command: protocol.Command, before_state: GameState | None) -> None:
        """Record a state-altering combat command so the next read can reconcile against a
        rule-based prediction. No-op unless the prior state was combat (nothing to predict)."""
        if before_state is None or before_state.screen_type != ScreenType.COMBAT:
            self._pending = None
            return
        self._pending = (command, before_state)

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
        frame = normalize_frame_to_client(
            frame,
            int(self.config.width),
            int(self.config.height),
            report=log.info,
        )

        from tspire.host.classify import classify_screen

        backend = self._get_backend()
        screen = classify_screen(frame, self.regions, backend)
        if screen == ScreenType.COMBAT:
            return self._remember(self._build_combat_state(frame, backend))

        return self._remember(GameState(
            screen_type=screen,
            screen_message="screen not yet supported by parser (v1 = combat only)",
            available_commands=protocol.commands_for_screen(screen.value),
        ))

    def _build_combat_state(self, frame, backend) -> GameState:
        if self.config.vision_mode == "llm":
            # Only pay for a block-reading call when a block badge is actually visible.
            read_block = backend.region_filled(frame, self.regions.player_block)
            result = self._get_llm().parse_combat(frame, read_block=read_block)
        else:
            from tspire.host.vision.combat import parse_combat

            result = parse_combat(frame, self.regions, backend)

        player = result.combat.player
        previous = self._last_state
        gold = _prefer_read_value(result.gold, previous.gold if previous else 0)
        floor = _prefer_read_value(result.floor, previous.floor if previous else 0)
        deck_count = _prefer_read_value(result.deck_count, previous.deck_count if previous else 0)
        current_hp = _prefer_read_value(player.current_hp, previous.current_hp if previous else 0)
        max_hp = _prefer_read_value(player.max_hp, previous.max_hp if previous else 0)
        if current_hp:
            player.current_hp = current_hp
        if max_hp:
            player.max_hp = max_hp
        state = GameState(
            screen_type=ScreenType.COMBAT,
            in_combat=True,
            floor=floor,
            act=_act_for_floor(floor),
            current_hp=current_hp,
            max_hp=max_hp,
            gold=gold,
            deck_count=deck_count,
            combat_state=result.combat,
            available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
            parse_confidence=round(result.confidence, 2),
        )
        return self._reconcile_with_prediction(state, frame)

    def _reconcile_with_prediction(self, state: GameState, frame) -> GameState:
        """Correct implausible combat reads against a rule-based prediction of the action
        that was just executed. The pending action is one-shot (cleared whether or not it
        applied) so it never affects more than the first read after the command."""
        pending = self._pending
        self._pending = None
        if not self.config.predict_enabled or pending is None:
            return state
        command, before = pending
        predicted = predict(before, command)
        if predicted is None:
            return state
        arbiter = self._arbiter(frame) if self.config.predict_arbiter else None
        return reconcile(state, predicted, before, arbiter)

    def _arbiter(self, frame):
        """Bind the LLM re-read calls to the current frame for the reconciler. Returns None
        when the LLM backend can't be constructed; the reconciler then degrades to rules."""
        try:
            llm = self._get_llm()
        except Exception:
            log.debug("arbiter unavailable", exc_info=True)
            return None
        return _LlmArbiter(llm, frame)

    def _remember(self, state: GameState) -> GameState:
        self._last_state = state
        return state


class _LlmArbiter:
    """Adapts OllamaVisionParser to the reconciler's Arbiter protocol by binding the frame
    and turning failed/zero re-reads into None."""

    def __init__(self, llm, frame) -> None:
        self._llm = llm
        self._frame = frame

    def reread_player_hp(self) -> tuple[int, int] | None:
        return self._reread(self._llm.reread_player_hp)

    def reread_energy(self) -> tuple[int, int] | None:
        return self._reread(self._llm.reread_energy)

    def _reread(self, fn) -> tuple[int, int] | None:
        try:
            current, maximum = fn(self._frame)
        except Exception:
            log.debug("arbiter re-read failed", exc_info=True)
            return None
        return (current, maximum) if current > 0 else None


def _prefer_read_value(value: int, fallback: int) -> int:
    return value if value > 0 else fallback


def _act_for_floor(floor: int) -> int:
    if floor <= 0:
        return 0
    if floor <= 16:
        return 1
    if floor <= 33:
        return 2
    if floor <= 50:
        return 3
    return 4
