# TSpire

Play **Slay the Spire** (the first game) from a remote terminal — **without a mod**, so
achievements stay enabled.

The game runs normally on a gaming PC. A **host** process there reads the screen and sends
input. A **client** running in any terminal renders the state and relays your commands over a
WebSocket. You make every decision; the host is just "eyes + hands".

Screen reading has two modes (config `vision_mode`):

- **`llm`** (default) — a local **Ollama** vision model (`gemma4:31b-cloud`) reads the
  busy combat scene (all enemies + the overlapping hand) robustly. The fixed HUD numbers
  (energy/HP/block/gold/floor/deck) are read with local **OpenCV+Tesseract OCR** when
  available (`ocr_hud_numbers`, default on), collapsing what used to be ~6 model calls into
  one scene call + sub-second OCR; if OCR reads nothing (e.g. no Tesseract), each number
  falls back to its own model crop, so it never regresses. Runs only during combat and on
  demand (connect / after a command), never on an idle timer.
- **`cv`** — OpenCV template matching + Tesseract OCR for everything. Fast, but fragile on
  the cluttered scene and needs calibration + a Tesseract install.

There are three input backends (`input_backend`, or `TSPIRE_INPUT_BACKEND`):

- **`mouse`** (default) — plays a card by click-**dragging** it to the target enemy (or, for a
  non-targeted card, to the play zone) and ends the turn by clicking the end-turn button. A
  real mouse event foregrounds the game on its own, and **card positions come from Slay the
  Spire's own deterministic hand-layout math** (read out of the game files), so they're exact
  and resolution-independent — no fragile CV card detection. Monster targets use HP-bar
  detection. Input success is confirmed with a cheap frame-change check, not a slow re-read.
- **`keyboard`** (experimental) — plays a card with StS's **number-key hotkeys** (1-9,0 select
  hand cards 1-10; requires *Settings → "Show Card keys" on*). `E` ends the turn. Coordinate-
  free in principle, but StS's keyboard input is a stateful multi-mode system (a persistent
  `keyboardCardIndex`, mouse/keyboard-mode toggling, frame-delayed targeting), so driving it
  reliably open-loop is unreliable — multi-enemy targeting in particular can grab the wrong
  card or mis-target. Prefer `mouse`; revisiting this would need closed-loop screen feedback.
- **`gamepad`** — the original virtual Xbox360 pad with arrow-key focus navigation (see the
  controller-detection notes below). Kept for completeness.

## Status

Milestone-based build (see `~/.claude/plans/this-is-an-empty-binary-spark.md`):

- **M0 Scaffold** — schema, config, capture, WebSocket host↔client loop. ✓
- **M1 State read** — combat classifier + dual parser (local LLM vision / OpenCV), region
  calibration overlay. ✓ *Validated end-to-end on a real combat screenshot with
  `gemma4:31b-cloud`: both enemies, full hand (names+costs), HP/energy/block all correct.*
- **M2 Client render** — Textual combat dashboard (top bar, enemies+intents, player, hand),
  friendly command parser, keybindings, reconnecting WebSocket client. ✓
- **M3** — input executors: mouse-drag backend (default), plus keyboard / gamepad fallbacks
  with closed-loop combat focus navigation. ✓
- **M4** — command validation, push-on-change, recovery.

## Game art (relics / intents)

Relic and intent identity is recognized by matching against the game's **own art, read
directly from the installed `desktop-1.0.jar` at runtime** — no assets are bundled, and this
only works when the game is installed. The jar is auto-detected (project dir, then Steam
libraries via the registry + `libraryfolders.vdf`); override with `jar_path` in config or
`TSPIRE_JAR_PATH`. Matching uses alpha-masked HSV colour histograms (validated: Burning Blood
@ 0.99).

**Potions** ship a jar-derived **metadata table** (every potion's name, flask shape, and
colour category, read from the class constant pools — dump it with
`python -m tspire.host.vision.potions`). *Identifying* a potion from its tiny belt icon is
not reliable by CV/vision (the flask shapes are too similar at ~25px), so reliable potion ID
will use the focus-cursor tooltip in a later input pass: focusing a potion renders its name
as text. Relic/potion identity isn't needed for turn-to-turn combat.

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
| `tspire/host/` | Screen capture, vision parsing, keyboard/gamepad input, WebSocket server |
| `tspire/client/` | Terminal UI and command parsing |
| `tools/` | One-off tooling (e.g. extracting card/relic art from the game jar) |
| `tests/` | Parser/schema tests + screenshot fixtures |

## Install

The client is dependency-light; the host needs native vision extras on Windows. The optional
gamepad fallback also needs the ViGEmBus driver, installed automatically by `vgamepad` on
first run.

```bash
# Client machine (can be remote):
pip install -e ".[client]"

# Host machine (the gaming PC):
pip install -e ".[host]"
# also install Tesseract OCR and ensure it is on PATH (or set tesseract_cmd in config)
```

## Run (M2 dashboard)

```bash
# On the gaming PC (StS running, Ollama up):
python -m tspire.host.server -v

# In any terminal (here or remote):
python -m tspire.client.app --host <gaming-pc-ip>
```

The client shows a combat dashboard (enemies with intents + damage, your block/energy/powers,
your hand with costs) and takes typed commands. **Commands** (indices from the dashboard):
`play <i> [t]` (`p`), `end` (`e`), `proceed`, `back`, `state` (`r`), `?` help. Keys:
`q` quit, `r` refresh, `?` help. Chain combat commands with semicolons, e.g.
`p 0 0; p 0 0; e`; the host predicts intermediate combat state and refreshes once at the
end. Potion use/discard is deferred until focus-tooltip reading is reliable.

The default backend sends **mouse** click-drags — nothing to enable in-game. Start with
`TSPIRE_INPUT_DRY_RUN=true` to log planned clicks/drags without touching the game. Before
trusting it, calibrate the click points on a live combat:

```bash
python -m tspire.host.input_probe --mouse                 # overlay click points -> mouse_points.png
python -m tspire.host.input_probe --mouse --play-card 0   # also play hand card 0 (end-to-end)
python -m tspire.host.input_probe --mouse --play-card 0 --target 0   # ...onto monster 0
```

The overlay writes `mouse_points.png` with a marker on every card / monster / the play zone /
the end-turn button so you can confirm they land; tune `tspire/host/vision/regions.py`
(`hand_search`, `monster_search`, `end_turn`) and the `mouse_play_zone_*` config if a marker
is off. To use the virtual Xbox360 controller fallback instead, set
`TSPIRE_INPUT_BACKEND=gamepad`; then Slay the Spire's controller support must be enabled
in-game, and the detection notes below apply.

**Controller backend detection (only for `input_backend = "gamepad"`):**

- **Start the host *before* launching Slay the Spire.** The game does **not** hot-plug
  controllers — it only detects pads present at startup. Run the host (which creates the
  virtual pad) first, then launch the game, so it sees the controller. A pad created while
  the game is already running is ignored.
- **Disable Steam Input for Slay the Spire** (Steam → right-click the game → Properties →
  Controller → *Disable Steam Input*). Otherwise Steam captures the virtual XInput pad
  (you'll see Steam's "Xbox 360 controller connected/disconnected" toasts) and the game
  receives nothing.
- Confirm **Controller Enabled** is on in the in-game settings.

When it's working you'll see the game switch to controller mode: a card lifts/focuses with
its tooltip and on-screen button prompts (X / Y / LT / RT) appear.

### Verify controller input

Before trusting any combat command, confirm StS actually detects the virtual pad. With the
game in a combat (several cards in hand, controller support enabled):

```bash
python -m tspire.host.input_probe --sequence left,left,right,right
python -m tspire.host.input_probe --dry-run --hand-count 5   # offline wiring check
```

The probe foregrounds the window, sends the sequence, and prints the observed card-focus
index after each press. **PASS** means the focus moved (the game sees the pad);
**FAIL**/**INCONCLUSIVE** point at the likely cause (window focus, the in-game `Controller
Enabled` setting, ViGEmBus/`vgamepad`, or the focus-glow thresholds). Use `--hand-count N`
to skip the (slow) state read, plus `--press-seconds`/`--settle` to tune timing.

## Configuration

Host config loads from `tspire_host.json` in the working directory (see
`tspire/host/config.py`), overridable via `TSPIRE_*` env vars. v1 expects the game in
windowed/borderless mode at a fixed resolution (default 1920×1080); vision region maps are
keyed on it.

Useful env overrides: `TSPIRE_VISION_MODE`, `TSPIRE_OLLAMA_URL`,
`TSPIRE_OLLAMA_MODEL`, `TSPIRE_LLM_IMAGE_WIDTH`, `TSPIRE_OCR_HUD_NUMBERS`,
`TSPIRE_TESSERACT_CMD`, `TSPIRE_OLLAMA_THINK`, `TSPIRE_INPUT_BACKEND`,
`TSPIRE_INPUT_DRY_RUN`, `TSPIRE_INPUT_RAW`, `TSPIRE_INPUT_PRESS_SECONDS`,
`TSPIRE_INPUT_STEP_DELAY`, `TSPIRE_INPUT_SETTLE_SECONDS`,
`TSPIRE_INPUT_COMMAND_TIMEOUT`, and the mouse-backend tunables
`TSPIRE_MOUSE_DRAG_SECONDS`, `TSPIRE_MOUSE_VERIFY_TIMEOUT`,
`TSPIRE_MOUSE_CHANGE_THRESHOLD`, `TSPIRE_MOUSE_PLAY_ZONE_X` / `_Y`,
`TSPIRE_MOUSE_RESTORE_CURSOR`.

## Develop

```bash
pip install -e ".[dev]"
pytest
```
