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

    # If true, the input executor logs button sequences instead of sending them.
    input_dry_run: bool = False

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
        if "TSPIRE_TESSERACT_CMD" in env:
            self.tesseract_cmd = env["TSPIRE_TESSERACT_CMD"]
        if "TSPIRE_INPUT_DRY_RUN" in env:
            self.input_dry_run = env["TSPIRE_INPUT_DRY_RUN"].lower() in {"1", "true", "yes"}

    def save(self, path: str | os.PathLike[str] | None = None) -> None:
        cfg_path = Path(path) if path else Path.cwd() / CONFIG_FILENAME
        cfg_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
