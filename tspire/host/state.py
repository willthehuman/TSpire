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
from tspire.host.state_tracker import StateTracker
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
        self._tracker = StateTracker(config)

    def note_action(self, command: protocol.Command, before_state: GameState | None) -> None:
        """Record a state-altering combat command so the next read can reconcile against a
        rule-based prediction. No-op unless the prior state was combat (nothing to predict)."""
        self._state_tracker().note_action(command, before_state)
        self._sync_legacy_state()

    def clear_pending_action(self) -> None:
        self._state_tracker().clear_pending()
        self._sync_legacy_state()

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

            log.info(
                "Ollama vision parser using model %s at %s",
                self.config.ollama_model,
                self.config.ollama_url,
            )
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
            return self._remember(self._state_tracker().reduce_noncombat(GameState(
                screen_type=ScreenType.NONE,
                screen_message=f"game window {self.config.window_title!r} not found",
                available_commands=protocol.commands_for_screen(ScreenType.NONE.value),
            )))
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
            return self._build_combat_state(frame, backend)

        return self._remember(self._state_tracker().reduce_noncombat(GameState(
            screen_type=screen,
            screen_message="screen not yet supported by parser (v1 = combat only)",
            available_commands=protocol.commands_for_screen(screen.value),
        )))

    def _build_combat_state(self, frame, backend) -> GameState:
        if self.config.vision_mode == "llm":
            # Only pay for a block-reading call when a block badge is actually visible.
            read_block = backend.region_filled(frame, self.regions.player_block)
            result = self._get_llm().parse_combat(frame, read_block=read_block)
        else:
            from tspire.host.vision.combat import parse_combat

            result = parse_combat(frame, self.regions, backend)

        player = result.combat.player
        state = GameState(
            screen_type=ScreenType.COMBAT,
            in_combat=True,
            floor=result.floor,
            act=_act_for_floor(result.floor),
            current_hp=player.current_hp,
            max_hp=player.max_hp,
            gold=result.gold,
            deck_count=result.deck_count,
            combat_state=result.combat,
            available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
            parse_confidence=round(result.confidence, 2),
        )
        arbiter = self._arbiter(frame) if self.config.predict_arbiter else None
        return self._remember(self._state_tracker().reduce_combat(state, result, arbiter))

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
        self._sync_legacy_state()
        return state

    def _state_tracker(self) -> StateTracker:
        tracker = getattr(self, "_tracker", None)
        if tracker is None:
            tracker = StateTracker(self.config)
            if self._last_state is not None:
                tracker.last_state = self._last_state
                if (
                    self._last_state.screen_type == ScreenType.COMBAT
                    and self._last_state.combat_state is not None
                    and self._last_state.read_status == "fresh"
                ):
                    tracker.accepted_state = self._last_state
            self._tracker = tracker
        return tracker

    def _sync_legacy_state(self) -> None:
        tracker = self._state_tracker()
        self._pending = (
            (tracker.pending.command, tracker.pending.before)
            if tracker.pending is not None
            else None
        )


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
        return (current, maximum) if maximum > 0 or current > 0 else None


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
