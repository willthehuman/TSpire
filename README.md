# TSpire

Play **Slay the Spire** (the first game) from a remote terminal — **without a mod**, so
achievements stay enabled.

The game runs normally on a gaming PC. A **host** process there reads the screen and acts as
a **virtual Xbox360 controller** (via `vgamepad` / ViGEmBus) to send input. A **client**
running in any terminal renders the state and relays your commands over a WebSocket. You make
every decision; the host is just "eyes + hands".

Screen reading has two modes (config `vision_mode`):

- **`llm`** (default) — a local **Ollama** vision model (e.g. `gemma4:e4b-it-qat`) reads the
  busy combat scene (all enemies + the overlapping hand) robustly. Fixed numbers
  (energy/HP/block) are read from upscaled region crops, one per call (this model loses
  accuracy with multiple images per call). ~15–20s per combat read; runs only during combat
  and on demand (connect / after a command), never on an idle timer. No Tesseract needed.
- **`cv`** — OpenCV template matching + Tesseract OCR. Fast, but fragile on the cluttered
  scene and needs calibration + a Tesseract install.

Why not keyboard or mouse automation? StS keyboard control is incomplete (you can pick cards
with `1..9` but still need the mouse to target/play), and mouse automation is brittle. The
game is, however, 100% playable on a gamepad via a focus cursor — so we drive that.

## Status

Milestone-based build (see `~/.claude/plans/this-is-an-empty-binary-spark.md`):

- **M0 Scaffold** — schema, config, capture, WebSocket host↔client loop. ✓
- **M1 State read** — combat classifier + dual parser (local LLM vision / OpenCV), region
  calibration overlay. ✓ *Validated end-to-end on a real combat screenshot with
  `gemma4:e4b-it-qat`: both enemies, full hand (names+costs), HP/energy/block all correct.*
- **M2** — Textual combat dashboard client. ← next
- **M3** — gamepad input executor with closed-loop focus navigation.
- **M4** — command validation, push-on-change, recovery.

## Game art (relics / intents)

Relic and intent identity is recognized by matching against the game's **own art, read
directly from the installed `desktop-1.0.jar` at runtime** — no assets are bundled, and this
only works when the game is installed. The jar is auto-detected (project dir, then Steam
libraries via the registry + `libraryfolders.vdf`); override with `jar_path` in config or
`TSPIRE_JAR_PATH`. Matching uses alpha-masked HSV colour histograms (validated: Burning Blood
@ 0.99). *Potions* have no per-potion image in the jar (they're a shape sprite tinted at
runtime), so they need shape-match + a colour→potion table — not yet implemented.

## Calibration (one-time, on the gaming PC)

Region coordinates and detection thresholds ship as **estimates for 1920×1080** and must be
tuned to your real screen. Overlay them on a live or saved combat frame and adjust
`tspire/host/vision/regions.py`:

```bash
python -m tspire.host.calibrate                   # live capture
python -m tspire.host.calibrate --image shot.png  # or a saved screenshot
```

## Layout

| Path | Role |
|------|------|
| `tspire/common/` | Shared state schema + wire protocol (no host-only deps) |
| `tspire/host/` | Screen capture, vision parsing, gamepad input, WebSocket server |
| `tspire/client/` | Terminal UI and command parsing |
| `tools/` | One-off tooling (e.g. extracting card/relic art from the game jar) |
| `tests/` | Parser/schema tests + screenshot fixtures |

## Install

The client is dependency-light; the host needs native vision/input extras (Windows + the
ViGEmBus driver, installed automatically by `vgamepad` on first run).

```bash
# Client machine (can be remote):
pip install -e ".[client]"

# Host machine (the gaming PC):
pip install -e ".[host]"
# also install Tesseract OCR and ensure it is on PATH (or set tesseract_cmd in config)
```

## Run (M0)

```bash
# On the gaming PC:
python -m tspire.host.server -v

# In a terminal (here or remote):
python -m tspire.client.app --host <gaming-pc-ip>
```

In M0 the host serves a stub state and echoes commands, so you can confirm the loop works
before the parser and input executor land.

## Configuration

Host config loads from `tspire_host.json` in the working directory (see
`tspire/host/config.py`), overridable via `TSPIRE_*` env vars. v1 expects the game in
windowed/borderless mode at a fixed resolution (default 1920×1080); vision region maps are
keyed on it.

## Develop

```bash
pip install -e ".[dev]"
pytest
```
