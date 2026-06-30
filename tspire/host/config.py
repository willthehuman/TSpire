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
    # Ollama connection + model for vision_mode == "llm".
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e4b-it-qat"
    # Width the full frame is downscaled to before sending to the model (px).
    llm_image_width: int = 1024

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
    # If true, the RAW protocol command accepts low-level input tokens.
    input_raw_enabled: bool = False
    # Gamepad executor timings, in seconds.
    input_press_seconds: float = 0.06
    input_step_delay: float = 0.08
    input_settle_seconds: float = 0.25
    input_command_timeout: float = 30.0

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
        if "TSPIRE_OLLAMA_URL" in env:
            self.ollama_url = env["TSPIRE_OLLAMA_URL"]
        if "TSPIRE_OLLAMA_MODEL" in env:
            self.ollama_model = env["TSPIRE_OLLAMA_MODEL"]
        if "TSPIRE_LLM_IMAGE_WIDTH" in env:
            self.llm_image_width = int(env["TSPIRE_LLM_IMAGE_WIDTH"])
        if "TSPIRE_PREDICT_ENABLED" in env:
            self.predict_enabled = env["TSPIRE_PREDICT_ENABLED"].lower() in {"1", "true", "yes"}
        if "TSPIRE_PREDICT_ARBITER" in env:
            self.predict_arbiter = env["TSPIRE_PREDICT_ARBITER"].lower() in {"1", "true", "yes"}
        if "TSPIRE_INPUT_DRY_RUN" in env:
            self.input_dry_run = env["TSPIRE_INPUT_DRY_RUN"].lower() in {"1", "true", "yes"}
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

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        cfg_path = Path(path) if path else Path.cwd() / CONFIG_FILENAME
        cfg_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
