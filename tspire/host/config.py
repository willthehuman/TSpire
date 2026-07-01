"""Host configuration.

v1 locks the game to a known resolution in windowed/borderless mode; all vision region
maps are keyed on this resolution (see tspire.host.vision.regions). Config loads from
``tspire_host.json`` next to the working dir if present, else uses defaults, and can be
overridden by environment variables (TSPIRE_*) for quick tweaks.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_FILENAME = "tspire_host.json"


@dataclass
class HostConfig:
    # WebSocket server bind address.
    host: str = "0.0.0.0"
    port: int = 8765

    # Expected game resolution. Region maps must exist for this (see regions.py).
    width: int = 1920
    height: int = 1080

    # Substring used to find the Slay the Spire window (case-insensitive).
    window_title: str = "Slay the Spire"
    # Bring the game to the foreground before screen capture/input. mss captures screen
    # pixels, so an occluding window would otherwise be captured over the game rectangle.
    focus_before_capture: bool = True

    # Path to the game's desktop-1.0.jar (relic/intent art is read from it at runtime).
    # Empty -> auto-detect (project dir, then Steam libraries). See game_assets.find_game_jar.
    jar_path: str = ""

    # Path to the tesseract executable; empty -> rely on PATH. (Only used by the "cv"
    # vision mode; the default "llm" mode needs no Tesseract.)
    tesseract_cmd: str = ""

    # Vision mode: "llm" (Ollama vision model, robust) or "cv" (OpenCV+Tesseract).
    vision_mode: str = "llm"
    # In "cv" mode, use EasyOCR (deep-learning OCR) for the game's stylised text that Tesseract
    # can't read -- card titles, the energy orb, the deck counter. Falls back to Tesseract when
    # EasyOCR isn't installed. Card titles are then fuzzy-matched to real card names.
    use_easyocr: bool = True
    # Ollama connection + model for vision_mode == "llm".
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:31b-cloud"
    # Width the full frame is downscaled to before sending to the model (px).
    llm_image_width: int = 1024
    # In "llm" mode, read the fixed HUD numbers (energy, HP, block, gold, floor, deck) with
    # local OpenCV+Tesseract OCR instead of a separate LLM call per number. This collapses
    # ~6 model calls into one scene call + sub-second OCR. Falls back to the LLM crop per
    # field when OCR yields nothing (e.g. Tesseract not installed), so it never regresses.
    ocr_hud_numbers: bool = True

    # Polling interval (seconds) for the capture loop when idle. Ignored for expensive
    # (LLM) providers, which only read on connect / after commands / on a state request.
    poll_interval: float = 0.5

    # Predicted-state reconciliation: after a play/end-turn, estimate the next combat state
    # from game rules and use it to correct implausible vision reads. predict_arbiter lets a
    # hard conflict be broken by re-reading the region with the LLM (needs Ollama). Set
    # predict_enabled false to push raw vision reads (the pre-prediction behavior).
    predict_enabled: bool = True
    predict_arbiter: bool = True

    # If true, the input executor logs button sequences instead of sending them.
    input_dry_run: bool = False
    # Input backend for real input: "mouse" (default), "keyboard", or "gamepad".
    # "mouse" plays cards by click-dragging at detected coordinates (deterministic, no
    # controller focus/hot-plug concerns); the others navigate with simulated buttons.
    input_backend: str = "mouse"
    # If true, the RAW protocol command accepts low-level input tokens.
    input_raw_enabled: bool = False
    # Gamepad/keyboard executor timings, in seconds.
    input_press_seconds: float = 0.06
    input_step_delay: float = 0.08
    input_settle_seconds: float = 0.25
    input_command_timeout: float = 30.0
    # After a controller play, press UP to release the card the game keeps lifted/focused (it
    # otherwise obscures the next screen read). Set false if it interferes with a setup.
    gamepad_clear_focus_after_play: bool = True
    # Deterministic controller navigation: instead of grabbing a frame and re-checking focus
    # after every d-pad press (slow), anchor the cursor at card 0 (UP then DOWN sets the hand
    # index to 0 in-game) and step right a known number of times. Far faster; relies on input
    # landing reliably. Set false to use the slower closed-loop CV navigation.
    gamepad_deterministic_nav: bool = True
    # Delay between deterministic nav presses (on top of input_step_delay) so each d-pad press
    # registers as a distinct frame. Raise if steps are dropped and the wrong card is played.
    gamepad_nav_delay: float = 0.06

    # --- mouse backend tunables ---
    # A card play is a click-drag from the card to the target (monster) or, for
    # non-targeted cards, to the play zone. The drag interpolates this many cursor steps
    # over this many seconds so libGDX registers it as a drag, not a teleport+click.
    mouse_drag_steps: int = 16
    mouse_drag_seconds: float = 0.18
    # Hold the button down on the card (before moving) so StS registers the grab, and hold
    # at the drop point before releasing so it registers the card as held in the play/target
    # zone. Non-targeted cards especially need these beats or the release reads as "card
    # returned to hand" and nothing plays.
    mouse_pickup_hold_seconds: float = 0.10
    mouse_drop_dwell_seconds: float = 0.10
    # Centre of the "play this card" drop zone, as fractions of the client area.
    mouse_play_zone_x: float = 0.5
    mouse_play_zone_y: float = 0.40
    # Vertical click row for hand cards (fraction from the top). Card X comes from StS's exact
    # hand-layout math; this Y picks where on the (tall) card faces to click. Tune with the
    # `--mouse` probe overlay if the dots sit above/below your hand.
    mouse_hand_row_y: float = 0.88
    # For monster targeting, click this fraction of the frame height ABOVE the detected HP bar
    # (the hover hitbox is the sprite body, which sits above the bar). Raise for tall enemies.
    mouse_target_above_bar: float = 0.05
    # Cheap frame-change verification (replaces slow LLM re-reads on the input path):
    # poll a downscaled-frame signature until it differs, or give up after the timeout.
    mouse_verify_timeout: float = 2.0
    mouse_verify_poll: float = 0.12
    mouse_change_threshold: float = 2.5  # mean abs gray delta on the 32x32 band signature
    # The change signature covers the frame from this height fraction down (hand + HUD band),
    # so small plays like block cards are not averaged out by the static upper screen.
    mouse_change_region_top: float = 0.55
    # Restore the user's cursor position after each click/drag.
    mouse_restore_cursor: bool = True

    # --- keyboard backend (StS number-key hotkeys) tunables ---
    key_tap_seconds: float = 0.04  # key hold duration
    key_gap_seconds: float = 0.12  # delay between consecutive key taps
    # Extra pause after the auto-target confirm and after each target step, so StS's
    # single-target mode engages before the next key (else `right`/Enter is misrouted to
    # hand navigation). Raise this if targeted plays pick the wrong enemy or don't fire.
    key_target_settle_seconds: float = 0.22
    # Ollama thinking mode. Kept false for vision calls so thinking models return only JSON.
    ollama_think: bool = False

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "HostConfig":
        cfg = cls()
        cfg_path = Path(path) if path else Path.cwd() / CONFIG_FILENAME
        if cfg_path.is_file():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        env = os.environ
        if "TSPIRE_HOST" in env:
            self.host = env["TSPIRE_HOST"]
        if "TSPIRE_PORT" in env:
            self.port = int(env["TSPIRE_PORT"])
        if "TSPIRE_WINDOW_TITLE" in env:
            self.window_title = env["TSPIRE_WINDOW_TITLE"]
        if "TSPIRE_FOCUS_BEFORE_CAPTURE" in env:
            self.focus_before_capture = env["TSPIRE_FOCUS_BEFORE_CAPTURE"].lower() in {"1", "true", "yes"}
        if "TSPIRE_JAR_PATH" in env:
            self.jar_path = env["TSPIRE_JAR_PATH"]
        if "TSPIRE_TESSERACT_CMD" in env:
            self.tesseract_cmd = env["TSPIRE_TESSERACT_CMD"]
        if "TSPIRE_VISION_MODE" in env:
            self.vision_mode = env["TSPIRE_VISION_MODE"]
        if "TSPIRE_USE_EASYOCR" in env:
            self.use_easyocr = env["TSPIRE_USE_EASYOCR"].lower() in {"1", "true", "yes"}
        if "TSPIRE_OLLAMA_URL" in env:
            self.ollama_url = env["TSPIRE_OLLAMA_URL"]
        if "TSPIRE_OLLAMA_MODEL" in env:
            self.ollama_model = env["TSPIRE_OLLAMA_MODEL"]
        if "TSPIRE_LLM_IMAGE_WIDTH" in env:
            self.llm_image_width = int(env["TSPIRE_LLM_IMAGE_WIDTH"])
        if "TSPIRE_OCR_HUD_NUMBERS" in env:
            self.ocr_hud_numbers = env["TSPIRE_OCR_HUD_NUMBERS"].lower() in {"1", "true", "yes"}
        if "TSPIRE_PREDICT_ENABLED" in env:
            self.predict_enabled = env["TSPIRE_PREDICT_ENABLED"].lower() in {"1", "true", "yes"}
        if "TSPIRE_PREDICT_ARBITER" in env:
            self.predict_arbiter = env["TSPIRE_PREDICT_ARBITER"].lower() in {"1", "true", "yes"}
        if "TSPIRE_INPUT_DRY_RUN" in env:
            self.input_dry_run = env["TSPIRE_INPUT_DRY_RUN"].lower() in {"1", "true", "yes"}
        if "TSPIRE_INPUT_BACKEND" in env:
            self.input_backend = env["TSPIRE_INPUT_BACKEND"].lower()
        if "TSPIRE_INPUT_RAW" in env:
            self.input_raw_enabled = env["TSPIRE_INPUT_RAW"].lower() in {"1", "true", "yes"}
        if "TSPIRE_INPUT_PRESS_SECONDS" in env:
            self.input_press_seconds = float(env["TSPIRE_INPUT_PRESS_SECONDS"])
        if "TSPIRE_INPUT_STEP_DELAY" in env:
            self.input_step_delay = float(env["TSPIRE_INPUT_STEP_DELAY"])
        if "TSPIRE_INPUT_SETTLE_SECONDS" in env:
            self.input_settle_seconds = float(env["TSPIRE_INPUT_SETTLE_SECONDS"])
        if "TSPIRE_INPUT_COMMAND_TIMEOUT" in env:
            self.input_command_timeout = float(env["TSPIRE_INPUT_COMMAND_TIMEOUT"])
        if "TSPIRE_GAMEPAD_CLEAR_FOCUS_AFTER_PLAY" in env:
            self.gamepad_clear_focus_after_play = (
                env["TSPIRE_GAMEPAD_CLEAR_FOCUS_AFTER_PLAY"].lower() in {"1", "true", "yes"}
            )
        if "TSPIRE_GAMEPAD_DETERMINISTIC_NAV" in env:
            self.gamepad_deterministic_nav = (
                env["TSPIRE_GAMEPAD_DETERMINISTIC_NAV"].lower() in {"1", "true", "yes"}
            )
        if "TSPIRE_GAMEPAD_NAV_DELAY" in env:
            self.gamepad_nav_delay = float(env["TSPIRE_GAMEPAD_NAV_DELAY"])
        if "TSPIRE_MOUSE_DRAG_SECONDS" in env:
            self.mouse_drag_seconds = float(env["TSPIRE_MOUSE_DRAG_SECONDS"])
        if "TSPIRE_MOUSE_VERIFY_TIMEOUT" in env:
            self.mouse_verify_timeout = float(env["TSPIRE_MOUSE_VERIFY_TIMEOUT"])
        if "TSPIRE_MOUSE_CHANGE_THRESHOLD" in env:
            self.mouse_change_threshold = float(env["TSPIRE_MOUSE_CHANGE_THRESHOLD"])
        if "TSPIRE_MOUSE_CHANGE_REGION_TOP" in env:
            self.mouse_change_region_top = float(env["TSPIRE_MOUSE_CHANGE_REGION_TOP"])
        if "TSPIRE_MOUSE_DROP_DWELL_SECONDS" in env:
            self.mouse_drop_dwell_seconds = float(env["TSPIRE_MOUSE_DROP_DWELL_SECONDS"])
        if "TSPIRE_KEY_GAP_SECONDS" in env:
            self.key_gap_seconds = float(env["TSPIRE_KEY_GAP_SECONDS"])
        if "TSPIRE_KEY_TARGET_SETTLE_SECONDS" in env:
            self.key_target_settle_seconds = float(env["TSPIRE_KEY_TARGET_SETTLE_SECONDS"])
        if "TSPIRE_MOUSE_PLAY_ZONE_X" in env:
            self.mouse_play_zone_x = float(env["TSPIRE_MOUSE_PLAY_ZONE_X"])
        if "TSPIRE_MOUSE_PLAY_ZONE_Y" in env:
            self.mouse_play_zone_y = float(env["TSPIRE_MOUSE_PLAY_ZONE_Y"])
        if "TSPIRE_MOUSE_HAND_ROW_Y" in env:
            self.mouse_hand_row_y = float(env["TSPIRE_MOUSE_HAND_ROW_Y"])
        if "TSPIRE_MOUSE_RESTORE_CURSOR" in env:
            self.mouse_restore_cursor = env["TSPIRE_MOUSE_RESTORE_CURSOR"].lower() in {"1", "true", "yes"}
        if "TSPIRE_OLLAMA_THINK" in env:
            self.ollama_think = env["TSPIRE_OLLAMA_THINK"].lower() in {"1", "true", "yes"}

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        cfg_path = Path(path) if path else Path.cwd() / CONFIG_FILENAME
        cfg_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
