"""Mouse input backend: play cards by click-dragging at detected coordinates.

This is the default and most reliable "hands" for the host. Unlike the controller/keyboard
backends, it does not navigate a wrapping focus cursor or depend on libGDX accepting a
virtual pad -- it issues real OS mouse events at the on-screen position of each card and
target, which is deterministic and also foregrounds the game as a side effect.

Pieces:
  * ``MouseDriver``        -- Win32 click / click-drag at absolute screen coordinates.
  * ``CardTargetLocator``  -- turns detected card/monster boxes + the window's client rect
                              into screen-space click points (with a geometric fan fallback).
  * ``FrameChangeDetector``-- cheap "did anything change" check that replaces the slow
                              LLM re-read on the input verification path.
  * ``MouseCommandHandler``-- the command executor wired into the server for this backend.

Heavy deps (ctypes is Windows-only; cv2/numpy for the change signature) are imported lazily
so this module imports cleanly for unit tests on any platform.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from tspire.common import protocol
from tspire.common.schema import GameState, Monster, ScreenType
from tspire.host.input.driver import InputUnavailable
from tspire.host.input.executor import (
    CommandError,
    _int_arg,
    _monster_alive,
)

log = logging.getLogger("tspire.host.input.mouse")

Point = tuple[int, int]


# --------------------------------------------------------------------------- #
# Driver: real OS mouse events
# --------------------------------------------------------------------------- #
class MouseDriver:
    """Win32 mouse driver using SendInput with absolute moves.

    StS reads the mouse through LWJGL, which tracks raw/absolute input events. ``SetCursorPos``
    teleports the cursor but does not always emit input events a game's raw-input reader sees,
    so a SetCursorPos "drag" can fail to pick up or play a card. ``SendInput`` with
    ``MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE`` emits genuine move events (over the whole
    virtual desktop), which the game honors as a real drag.
    """

    available = True
    diagnostic = None

    # SendInput / event flags.
    _INPUT_MOUSE = 0
    _MOVE = 0x0001
    _LEFTDOWN = 0x0002
    _LEFTUP = 0x0004
    _ABSOLUTE = 0x8000
    _VIRTUALDESK = 0x4000
    # GetSystemMetrics indices for the virtual screen.
    _SM_XVIRTUALSCREEN = 76
    _SM_YVIRTUALSCREEN = 77
    _SM_CXVIRTUALSCREEN = 78
    _SM_CYVIRTUALSCREEN = 79

    def __init__(self, config) -> None:
        self.steps = max(2, int(getattr(config, "mouse_drag_steps", 16)))
        self.drag_seconds = max(0.0, float(getattr(config, "mouse_drag_seconds", 0.18)))
        self.drop_dwell = max(0.0, float(getattr(config, "mouse_drop_dwell_seconds", 0.10)))
        self.pickup_hold = max(0.0, float(getattr(config, "mouse_pickup_hold_seconds", 0.10)))
        self.restore_cursor = bool(getattr(config, "mouse_restore_cursor", True))
        import sys

        if sys.platform != "win32":  # pragma: no cover - host is Windows
            raise InputUnavailable("mouse input is only available on Windows")
        try:
            import ctypes
            from ctypes import wintypes

            self._ctypes = ctypes
            self._user32 = ctypes.windll.user32
            self._POINT = wintypes.POINT
            self._build_input_structs(ctypes, wintypes)
        except Exception as exc:  # pragma: no cover - platform dependent
            raise InputUnavailable("could not initialize Win32 mouse input") from exc

    def _build_input_structs(self, ctypes, wintypes) -> None:
        ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("u", _U)]

        self._MOUSEINPUT = MOUSEINPUT
        self._INPUT = INPUT

    def _metric(self, index: int) -> int:
        return int(self._user32.GetSystemMetrics(index))

    def _to_absolute(self, x: int, y: int) -> tuple[int, int]:
        vx = self._metric(self._SM_XVIRTUALSCREEN)
        vy = self._metric(self._SM_YVIRTUALSCREEN)
        vw = max(1, self._metric(self._SM_CXVIRTUALSCREEN) - 1)
        vh = max(1, self._metric(self._SM_CYVIRTUALSCREEN) - 1)
        nx = round((int(x) - vx) * 65535 / vw)
        ny = round((int(y) - vy) * 65535 / vh)
        return nx, ny

    def _send(self, flags: int, x: int = 0, y: int = 0) -> None:
        nx, ny = (self._to_absolute(x, y) if flags & self._ABSOLUTE else (0, 0))
        mi = self._MOUSEINPUT(nx, ny, 0, flags, 0, 0)
        inp = self._INPUT()
        inp.type = self._INPUT_MOUSE
        inp.u.mi = mi
        self._user32.SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(inp))

    def _cursor(self) -> Point:
        pt = self._POINT()
        self._user32.GetCursorPos(self._ctypes.byref(pt))
        return (pt.x, pt.y)

    def _move(self, x: int, y: int) -> None:
        self._send(self._MOVE | self._ABSOLUTE | self._VIRTUALDESK, x, y)

    def _restore(self, saved: Point | None) -> None:
        if saved is not None:
            self._user32.SetCursorPos(int(saved[0]), int(saved[1]))

    def click(self, x: int, y: int) -> None:
        saved = self._cursor() if self.restore_cursor else None
        try:
            self._move(x, y)
            time.sleep(0.03)
            self._send(self._LEFTDOWN)
            time.sleep(0.04)
            self._send(self._LEFTUP)
            time.sleep(0.03)
        finally:
            self._restore(saved)

    def drag(self, start: Point, end: Point) -> None:
        """Press at ``start``, glide to ``end`` over several steps, release.

        The interpolation + grab hold matter: StS must register the button-down on the card
        (so it picks the card up) before the move begins, and a single jump reads as a click,
        not a drag.
        """
        saved = self._cursor() if self.restore_cursor else None
        x0, y0 = start
        x1, y1 = end
        per_step = self.drag_seconds / self.steps if self.steps else 0.0
        try:
            self._move(x0, y0)
            time.sleep(0.04)
            self._send(self._LEFTDOWN)
            # Hold on the card so StS registers the grab before we start moving.
            time.sleep(self.pickup_hold)
            for i in range(1, self.steps + 1):
                t = i / self.steps
                self._move(round(x0 + (x1 - x0) * t), round(y0 + (y1 - y0) * t))
                if per_step:
                    time.sleep(per_step)
            # Dwell at the drop point so StS registers the card as held in the play/target
            # zone before release, or the release reads as "returned to hand".
            self._move(x1, y1)
            time.sleep(self.drop_dwell)
            self._send(self._LEFTUP)
            time.sleep(0.05)
        finally:
            self._restore(saved)

    def close(self) -> None:
        pass


class DryRunMouseDriver:
    """Records click/drag intents without touching the OS (dry-run + tests)."""

    available = True
    diagnostic = "mouse dry-run enabled"

    def __init__(self, config=None) -> None:
        self.clicks: list[Point] = []
        self.drags: list[tuple[Point, Point]] = []

    def click(self, x: int, y: int) -> None:
        self.clicks.append((int(x), int(y)))
        log.info("dry-run mouse click: (%d, %d)", x, y)

    def drag(self, start: Point, end: Point) -> None:
        self.drags.append((start, end))
        log.info("dry-run mouse drag: %s -> %s", start, end)

    def close(self) -> None:
        pass


class DisabledMouseDriver:
    available = False

    def __init__(self, diagnostic: str) -> None:
        self.diagnostic = diagnostic

    def click(self, x: int, y: int) -> None:
        raise InputUnavailable(self.diagnostic)

    def drag(self, start: Point, end: Point) -> None:
        raise InputUnavailable(self.diagnostic)

    def close(self) -> None:
        pass


def build_mouse_driver(config):
    if getattr(config, "input_dry_run", False):
        return DryRunMouseDriver(config)
    try:
        return MouseDriver(config)
    except InputUnavailable as exc:
        return DisabledMouseDriver(str(exc))


# --------------------------------------------------------------------------- #
# Coordinate location
# --------------------------------------------------------------------------- #
# Slay the Spire's exact hand layout, lifted from the game's CardGroup.refreshHandLayout.
# Each entry is the per-card X offset (left-to-right) in units of AbstractCard.IMG_WIDTH_S,
# centred on screen-centre X. The hand is always centred, and the spacing depends only on the
# card count -- never the resolution.
_HAND_X_OFFSETS: dict[int, tuple[float, ...]] = {
    1: (0.0,),
    2: (-0.47, 0.53),
    3: (-0.9, 0.0, 0.9),
    4: (-1.36, -0.46, 0.46, 1.36),
    5: (-1.7, -0.9, 0.0, 0.9, 1.7),
    6: (-2.1, -1.3, -0.43, 0.43, 1.3, 2.1),
    7: (-2.4, -1.7, -0.9, 0.0, 0.9, 1.7, 2.4),
    8: (-2.5, -1.82, -1.1, -0.38, 0.38, 1.1, 1.77, 2.5),
    9: (-2.8, -2.2, -1.53, -0.8, 0.0, 0.8, 1.53, 2.2, 2.8),
    10: (-2.9, -2.4, -1.8, -1.1, -0.4, 0.4, 1.1, 1.8, 2.4, 2.9),
}
# IMG_WIDTH_S = 300 * scale * 0.7 = 210 * scale; Settings.WIDTH = 1920 * scale. The ratio is
# therefore a resolution-independent constant.
_IMG_WIDTH_S_RATIO = 210.0 / 1920.0  # = 0.109375
# StS's per-card sink puts the hitbox CENTRES very low (~0.93-1.0 of height, near the screen
# edge). The cards are tall, so a single row a little higher sits on every card's visible face
# and inside its hitbox for typical hands -- safer than clicking the bottom edge.
_DEFAULT_HAND_ROW_Y = 0.88


@dataclass
class FrameLayout:
    """Screen-space click points for the current frame."""

    cards: list[Point] = field(default_factory=list)
    monsters: dict[int, Point] = field(default_factory=dict)
    card_source: str = "cv"  # "cv" or "geometric"


class CardTargetLocator:
    """Map detected card/monster boxes to absolute screen click points.

    Reuses the existing ``WindowCapture`` + region map + CV backend that the state provider
    already owns (the same collaborators ``ScreenFocusObserver`` uses).
    """

    def __init__(self, state_provider, config) -> None:
        self.state_provider = state_provider
        self.config = config

    # -- public ---------------------------------------------------------------
    def locate(self, *, expected_hand: int, monsters: list[Monster]) -> FrameLayout:
        capture = getattr(self.state_provider, "capture", None)
        regions = getattr(self.state_provider, "regions", None)
        get_backend = getattr(self.state_provider, "_get_backend", None)
        if capture is None or regions is None or get_backend is None:
            raise CommandError("mouse backend requires the screen state provider")

        frame = capture.grab()
        cr = capture.client_rect()
        fh, fw = frame.shape[:2]
        backend = get_backend()

        monster_points = self._monster_points(frame, fw, fh, cr, backend, regions, monsters)

        # Card positions come from Slay the Spire's OWN deterministic hand layout
        # (CardGroup.refreshHandLayout): the hand is centred on screen-centre X with a
        # hardcoded per-size offset table, in units of IMG_WIDTH_S = 0.109375 * width. This
        # is exact and resolution-independent, so we don't depend on CV card detection (which
        # is fragile on the overlapping fan and was returning 0 boxes). CV/geometric are only
        # a fallback for unusual hands larger than the table (10+ via relics).
        if expected_hand in _HAND_X_OFFSETS:
            return FrameLayout(
                cards=self._sts_hand_points(cr, expected_hand),
                monsters=monster_points,
                card_source="sts-layout",
            )
        cards = self._card_points(frame, fw, fh, cr, backend, regions, expected_hand)
        source = "cv" if len(cards) == expected_hand and expected_hand > 0 else "geometric"
        if source == "geometric":
            cards = self._geometric_hand(frame, fw, fh, cr, regions, expected_hand)
        return FrameLayout(cards=cards, monsters=monster_points, card_source=source)

    def _sts_hand_points(self, cr, hand_size: int) -> list[Point]:
        """Exact hand-card click points from StS's CardGroup.refreshHandLayout.

        x = WIDTH/2 + offset * IMG_WIDTH_S, where IMG_WIDTH_S = 0.109375 * WIDTH, so the
        x-fraction is ``0.5 + offset * 0.109375`` (resolution-independent). y is a single
        tunable hand-row fraction (``mouse_hand_row_y``) -- the cards are tall and a fixed
        row sits on every card's face for typical hands, which is steadier than the layout's
        very-low hitbox centres.
        """
        offsets = _HAND_X_OFFSETS[hand_size]
        y_frac = float(getattr(self.config, "mouse_hand_row_y", _DEFAULT_HAND_ROW_Y))
        y = cr.top + round(y_frac * cr.height)
        points: list[Point] = []
        for off in offsets:
            x_frac = 0.5 + off * _IMG_WIDTH_S_RATIO
            points.append((cr.left + round(x_frac * cr.width), y))
        return points

    def play_zone_point(self) -> Point:
        cr = self._client_rect()
        return (
            cr.left + round(self.config.mouse_play_zone_x * cr.width),
            cr.top + round(self.config.mouse_play_zone_y * cr.height),
        )

    def end_turn_point(self) -> Point:
        cr = self._client_rect()
        regions = getattr(self.state_provider, "regions", None)
        if regions is None:
            raise CommandError("mouse backend requires the screen state provider")
        left, top, w, h = regions.end_turn.to_pixels(cr.width, cr.height)
        return (cr.left + left + w // 2, cr.top + top + h // 2)

    # -- internals ------------------------------------------------------------
    def _client_rect(self):
        capture = getattr(self.state_provider, "capture", None)
        if capture is None:
            raise CommandError("mouse backend requires the screen state provider")
        return capture.client_rect()

    def _to_screen(self, x: float, y: float, fw: int, fh: int, cr) -> Point:
        # Frames are grabbed at exactly the client rect, but scale defensively against any
        # capture/logical-pixel mismatch rather than assuming 1:1.
        sx = cr.left + (x / fw) * cr.width if fw else cr.left
        sy = cr.top + (y / fh) * cr.height if fh else cr.top
        return (round(sx), round(sy))

    def _card_points(self, frame, fw, fh, cr, backend, regions, expected_hand) -> list[Point]:
        try:
            boxes = backend.find_cards(frame, regions.hand_search)
        except Exception:
            log.debug("find_cards failed", exc_info=True)
            return []
        points: list[Point] = []
        for box in boxes:
            # Click the lower-centre strip: on a fanned hand the next card overlaps the top,
            # but the bottom of each card is clear.
            cx = box.left + box.width / 2
            cy = box.top + box.height * 0.82
            points.append(self._to_screen(cx, cy, fw, fh, cr))
        return points

    def _geometric_hand(self, frame, fw, fh, cr, regions, expected_hand) -> list[Point]:
        """Fallback when CV card detection disagrees with the known hand size.

        The hand is centred and its width depends on card count, so spreading points across
        the *whole* hand-search region (which is sized for a full hand) overshoots a small
        hand. Instead, measure where the card pixels actually are (the bright bounding box
        within the search region) and distribute ``expected_hand`` points across that real
        span -- so 3 centred cards get 3 centred points. Falls back to the full region only
        when no card pixels are found.
        """
        if expected_hand <= 0:
            return []
        left, top, w, h = regions.hand_search.to_pixels(fw, fh)
        extent = self._hand_extent(frame, left, top, w, h)
        if extent is None:
            x0, x1, cy = left, left + w, top + h * 0.55
        else:
            x0, x1, y0, y1 = extent
            cy = y0 + (y1 - y0) * 0.55  # card centres sit a bit below the top of the mass
        span = max(1, x1 - x0)
        # Inset slightly so the end cards aren't clipped at the very edge of the mass.
        inset = span * 0.5 / expected_hand
        usable = max(1.0, span - 2 * inset)
        points: list[Point] = []
        for i in range(expected_hand):
            frac = (i + 0.5) / expected_hand
            cx = x0 + inset + frac * usable
            points.append(self._to_screen(cx, cy, fw, fh, cr))
        return points

    def _hand_extent(self, frame, left, top, w, h):
        """Bounding box (x0, x1, y0, y1) of card (bright) pixels inside the hand region.

        Reuses the same brightness cue as ``find_cards`` (gray > 70) but only needs the
        overall extent, not clean per-card contours -- so it survives the fan overlap that
        defeats contour detection.
        """
        try:
            import cv2
            import numpy as np

            roi = frame[top : top + h, left : left + w]
            if getattr(roi, "size", 0) == 0:
                return None
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mask = gray > 70
            cols = mask.any(axis=0)
            rows = mask.any(axis=1)
            if not cols.any() or not rows.any():
                return None
            xs = np.where(cols)[0]
            ys = np.where(rows)[0]
            return (left + int(xs[0]), left + int(xs[-1]), top + int(ys[0]), top + int(ys[-1]))
        except Exception:  # pragma: no cover - missing native deps or a fake frame
            log.debug("hand extent measurement failed", exc_info=True)
            return None

    def _monster_points(self, frame, fw, fh, cr, backend, regions, monsters) -> dict[int, Point]:
        living = [m for m in monsters if _monster_alive(m)]
        try:
            bars = backend.find_red_bars(frame, regions.monster_search)
        except Exception:
            log.debug("find_red_bars failed", exc_info=True)
            bars = []
        points: dict[int, Point] = {}
        chosen = _select_hp_bars(bars, len(living), fh)
        if living and len(chosen) == len(living):
            chosen.sort(key=lambda b: b.left)
            offset = float(getattr(self.config, "mouse_target_above_bar", 0.05)) * fh
            for monster, bar in zip(living, chosen):
                cx = bar.left + bar.width / 2  # HP bar is centred under the sprite
                cy = bar.top - offset  # hover hitbox is the sprite BODY, above the bar
                points[monster.index] = self._to_screen(cx, max(cy, 0), fw, fh, cr)
            return points
        # Fewer bars than enemies (or none detected): last-resort geometric guess kept to the
        # centre-right where StS places enemies (player is fixed at x=0.25), not the full wide
        # search band (which pushed the ends off-screen).
        n = max(len(living), 1)
        span_left, span_right = 0.52, 0.86
        for ordinal, monster in enumerate(living):
            frac = (ordinal + 0.5) / n
            x_frac = span_left + frac * (span_right - span_left)
            points[monster.index] = (cr.left + round(x_frac * cr.width), cr.top + round(0.55 * cr.height))
        return points


# --------------------------------------------------------------------------- #
# Cheap input verification (no LLM)
# --------------------------------------------------------------------------- #
class FrameChangeDetector:
    """Detect that *something* changed on screen, cheaply, by comparing a downscaled
    grayscale signature of successive frames. Replaces the slow LLM re-read loop on the
    input verification path.

    The signature covers only the **bottom band** of the frame (hand + energy + block +
    player area). Every successful card play changes the hand (the played card leaves) and
    usually energy, so this band always moves substantially -- including for non-targeted
    block cards, whose change is tiny relative to the *whole* frame and used to slip under a
    full-frame threshold. End turn discards the whole hand, so it shows up strongly too.
    """

    def __init__(self, capture, config) -> None:
        self.capture = capture
        self.threshold = float(getattr(config, "mouse_change_threshold", 2.5))
        self.timeout = float(getattr(config, "mouse_verify_timeout", 2.0))
        self.poll = max(0.0, float(getattr(config, "mouse_verify_poll", 0.12)))
        # Measure from this fraction of the frame height down to the bottom.
        self.band_top = min(0.95, max(0.0, float(getattr(config, "mouse_change_region_top", 0.55))))

    def signature(self):
        import cv2

        frame = self.capture.grab()
        h = frame.shape[0]
        top = int(h * self.band_top)
        band = frame[top:, :]
        if band.size == 0:
            band = frame
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype("float32")

    def _changed(self, before, after) -> bool:
        import numpy as np

        return float(np.abs(after - before).mean()) >= self.threshold

    def wait_for_change(self, before) -> bool:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() <= deadline:
            if self.poll:
                time.sleep(self.poll)
            try:
                if self._changed(before, self.signature()):
                    return True
            except Exception:
                log.debug("frame-change check failed", exc_info=True)
                return True  # fail open: don't block on a flaky read
        return False


# --------------------------------------------------------------------------- #
# Command handler
# --------------------------------------------------------------------------- #
class MouseCommandHandler:
    """Execute client commands with real mouse click-drags.

    Combat-first, mirroring ``GamepadCommandHandler``'s contract so the server can use
    either interchangeably. PROCEED/RETURN fall back to keyboard Enter/Esc (trivial and
    reliable); POTION/CHOOSE stay deferred as in the gamepad path.
    """

    def __init__(
        self,
        config,
        state_provider,
        *,
        driver=None,
        locator: CardTargetLocator | None = None,
        detector: FrameChangeDetector | None = None,
        key_driver=None,
    ) -> None:
        self.config = config
        self.state_provider = state_provider
        self.driver = driver or build_mouse_driver(config)
        self.locator = locator or CardTargetLocator(state_provider, config)
        self._key_driver = key_driver  # built lazily for proceed/return
        capture = getattr(state_provider, "capture", None)
        self.detector = detector or (FrameChangeDetector(capture, config) if capture else None)
        self._foreground_failures = 0
        if getattr(self.driver, "diagnostic", None):
            level = logging.INFO if self.driver.available else logging.WARNING
            log.log(level, "mouse input driver: %s", self.driver.diagnostic)

    # -- public contract ------------------------------------------------------
    def execute(
        self,
        command: protocol.Command,
        state_hint: GameState | None = None,
        *,
        verify_state_change: bool = True,
        note_action: bool = True,
    ) -> tuple[bool, str | None]:
        try:
            return self._execute(
                command,
                state_hint,
                verify_state_change=verify_state_change,
                note_action=note_action,
            )
        except (CommandError, ValueError, InputUnavailable) as exc:
            return False, str(exc)
        except Exception:
            log.exception("mouse input command failed")
            return False, "input command failed (see host log)"

    def _execute(
        self,
        command: protocol.Command,
        state_hint: GameState | None,
        *,
        verify_state_change: bool,
        note_action: bool,
    ) -> tuple[bool, str | None]:
        if command.verb == protocol.Verb.STATE:
            return True, None
        if not self.driver.available:
            return False, getattr(self.driver, "diagnostic", None) or "mouse input unavailable"
        if not self._ensure_foreground():
            return False, "could not bring Slay the Spire to the foreground"
        if command.verb == protocol.Verb.POTION:
            return False, "potion execution is deferred until potion focus/tooltips are reliable"
        if command.verb == protocol.Verb.RAW:
            return False, "raw input requires the gamepad backend"
        if command.verb == protocol.Verb.PROCEED:
            self._key("proceed")
            return True, None
        if command.verb == protocol.Verb.RETURN:
            self._key("cancel")
            return True, None
        if command.verb == protocol.Verb.END:
            before = self._require_combat(state_hint)
            if note_action:
                self._note_action(command, before)
            ok, error = self._end_turn(before, verify_state_change=verify_state_change)
            if not ok:
                self._clear_pending_action()
            return ok, error
        if command.verb == protocol.Verb.PLAY:
            before = self._require_combat(state_hint)
            if note_action:
                self._note_action(command, before)
            ok, error = self._play(command.args, before, verify_state_change=verify_state_change)
            if not ok:
                self._clear_pending_action()
            return ok, error
        return False, f"unsupported command {command.verb!r}"

    # -- verbs ----------------------------------------------------------------
    def _play(
        self,
        args: list[str],
        before: GameState,
        *,
        verify_state_change: bool,
    ) -> tuple[bool, str | None]:
        combat = before.combat_state
        assert combat is not None
        card_index = _int_arg(args, 0, "card")
        target_index = _int_arg(args, 1, "target", optional=True)
        if card_index < 0 or card_index >= len(combat.hand):
            return False, f"card index {card_index} is out of range"
        if target_index is not None and not _has_living_monster(combat.monsters, target_index):
            return False, f"target index {target_index} is out of range"

        before_sig = self._signature() if verify_state_change else None
        layout = self.locator.locate(expected_hand=len(combat.hand), monsters=combat.monsters)
        if card_index >= len(layout.cards):
            return False, f"could not locate hand card {card_index} on screen"
        card_point = layout.cards[card_index]

        play_zone = self.locator.play_zone_point()
        living = [m for m in combat.monsters if _monster_alive(m)]

        if target_index is not None:
            # With a single living enemy, StS auto-targets it, so dropping the card in the
            # play zone plays it -- no enemy coordinate needed (monster positions are
            # per-encounter and only come from fragile HP-bar detection).
            if len(living) <= 1:
                self.driver.drag(card_point, play_zone)
            else:
                target_point = layout.monsters.get(target_index)
                if target_point is None:
                    return False, f"could not locate target {target_index} on screen"
                self.driver.drag(card_point, target_point)
            if not verify_state_change:
                return True, None
            if self._wait_for_change(before_sig):
                return True, None
            # No change: the drop likely missed the enemy and the card is now stuck in
            # targeting mode. Cancel so the game doesn't stay stuck, then (for a card that may
            # actually be non-targeted) try the play zone once.
            self._key("cancel")
            self.driver.drag(card_point, play_zone)
            if self._wait_for_change(before_sig):
                return True, None
            self._key("cancel")
            return False, (
                "targeted play did not resolve (the drop likely missed the enemy). For "
                "multi-enemy targeting, try the keyboard backend "
                "(TSPIRE_INPUT_BACKEND=keyboard), which picks targets with arrow keys and "
                "needs no enemy coordinates."
            )

        self.driver.drag(card_point, play_zone)
        if not verify_state_change:
            return True, None
        if self._wait_for_change(before_sig):
            return True, None
        return False, (
            "play input sent, but no combat state change was observed; "
            "if this card needs a target, provide a target index"
        )

    def _end_turn(self, before: GameState, *, verify_state_change: bool) -> tuple[bool, str | None]:
        before_sig = self._signature() if verify_state_change else None
        self.driver.click(*self.locator.end_turn_point())
        if not verify_state_change:
            return True, None
        if self._wait_for_change(before_sig):
            return True, None
        return False, "end turn input sent, but no combat state change was observed"

    # -- helpers --------------------------------------------------------------
    def _require_combat(self, state_hint: GameState | None) -> GameState:
        # Mouse play reads live CV coordinates from a fresh frame, so it does not need a
        # fresh *LLM* state -- a known combat screen (even slightly stale) is enough. Only
        # fall back to a (slow) read when we don't have a usable combat hint at all.
        state = state_hint if _is_usable_combat(state_hint) else self._read_state()
        if state.screen_type != ScreenType.COMBAT or state.combat_state is None:
            raise CommandError(
                f"combat input is only available on COMBAT, got {state.screen_type.value}"
            )
        return state

    def _read_state(self) -> GameState:
        try:
            return self.state_provider.read()
        except Exception as exc:
            raise CommandError(f"could not read game state: {exc}") from exc

    def _signature(self):
        if self.detector is None:
            return None
        try:
            return self.detector.signature()
        except Exception:
            log.debug("baseline signature failed", exc_info=True)
            return None

    def _wait_for_change(self, before_sig) -> bool:
        if self.detector is None or before_sig is None:
            return True  # can't verify cheaply -> trust the input (authoritative read follows)
        return self.detector.wait_for_change(before_sig)

    def _ensure_foreground(self) -> bool:
        capture = getattr(self.state_provider, "capture", None)
        ensure = getattr(capture, "ensure_foreground", None)
        if ensure is None:
            return True
        try:
            try:
                ok = bool(ensure(click_safe_zone=False))
            except TypeError:
                ok = bool(ensure())
        except Exception:
            log.debug("could not foreground game window", exc_info=True)
            ok = False
        if ok:
            self._foreground_failures = 0
        else:
            self._foreground_failures += 1
            if self._foreground_failures == 1 or self._foreground_failures % 10 == 0:
                log.warning(
                    "could not bring Slay the Spire to the foreground; the mouse click may "
                    "land on another window. Make sure the game isn't minimized."
                )
        return ok

    def _key(self, token: str) -> None:
        if self._key_driver is None:
            from tspire.host.input.driver import InputTiming, KeyboardDriver

            self._key_driver = KeyboardDriver(InputTiming.from_config(self.config))
        self._key_driver.press(token)

    def _note_action(self, command: protocol.Command, before: GameState | None) -> None:
        note = getattr(self.state_provider, "note_action", None)
        if note is None:
            return
        try:
            note(command, before)
        except Exception:
            log.debug("note_action failed", exc_info=True)

    def _clear_pending_action(self) -> None:
        clear = getattr(self.state_provider, "clear_pending_action", None)
        if clear is None:
            return
        try:
            clear()
        except Exception:
            log.debug("clear_pending_action failed", exc_info=True)


def _select_hp_bars(bars, count: int, frame_h: int) -> list:
    """Pick the ``count`` bars most likely to be the real monster HP bars.

    Detection returns extra red blobs (intent/status icons, and block status that morphology-
    merges into a fat red region), and enemies can sit at different heights, so neither
    "widest N" nor "same Y row" is reliable. The robust cue: real StS HP bars are uniformly
    THIN (~20px). Keep the ``count`` bars whose height is closest to the median bar height
    (rejecting the thick/odd blobs), preferring wider ones on ties, then order left-to-right.
    """
    if count <= 0 or not bars:
        return []
    if len(bars) <= count:
        return list(bars)
    heights = sorted(b.height for b in bars)
    median_h = heights[len(heights) // 2]
    chosen = sorted(bars, key=lambda b: (abs(b.height - median_h), -b.width))[:count]
    chosen.sort(key=lambda b: b.left)
    return chosen


def _has_living_monster(monsters: list[Monster], index: int) -> bool:
    return any(m.index == index and _monster_alive(m) for m in monsters)


def _is_usable_combat(state: GameState | None) -> bool:
    return (
        state is not None
        and state.screen_type == ScreenType.COMBAT
        and state.combat_state is not None
    )


# Re-export for callers that want the fresh-combat predicate name symmetry.
__all__ = [
    "MouseDriver",
    "DryRunMouseDriver",
    "build_mouse_driver",
    "CardTargetLocator",
    "FrameChangeDetector",
    "MouseCommandHandler",
]
