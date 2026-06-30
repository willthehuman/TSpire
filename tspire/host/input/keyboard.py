"""Keyboard input backend: play cards with Slay the Spire's number-key hotkeys.

Discovered by decompiling the game: with "Show Card keys" enabled, number keys 1-9,0 select
hand cards 1-10 (``InputHelper.getCardSelectedByHotkey``), END_TURN is ``E``, CONFIRM is
``Enter``, CANCEL is ``Esc``. This is coordinate-free for card *selection* -- no fragile CV
or hand-layout math needed to pick the card.

The one subtlety (also from the decompile): pressing the number key only *grabs* the card
(``manuallySelectCard``); it then follows the cursor, and a non-target card only plays when
CONFIRM fires while the cursor is in the drop zone (``isHoveringDropZone``). So we park the
cursor at the play zone first, then tap the number, then CONFIRM. Targeted cards: CONFIRM
auto-targets the first (leftmost) enemy, then LEFT/RIGHT walks to the requested target and a
second CONFIRM plays it.

Win32-only (keystrokes + a single cursor move). Heavy bits are lazy-imported so the module
imports for unit tests anywhere.
"""

from __future__ import annotations

import logging
import time

from tspire.common import protocol
from tspire.common.schema import GameState, Monster, ScreenType
from tspire.host.input.driver import InputUnavailable
from tspire.host.input.executor import CommandError, _int_arg, _monster_alive
from tspire.host.input.mouse import FrameChangeDetector, Point

log = logging.getLogger("tspire.host.input.keyboard")

# Win32 virtual-key codes for the keys we send.
_VK = {
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    "enter": 0x0D,
    "escape": 0x1B,
    "end_turn": 0x45,  # E
    "left": 0x25,
    "right": 0x27,
}


def _card_key(index: int) -> str | None:
    """Hand index -> number-key token. Card 0 -> '1' ... card 8 -> '9', card 9 -> '0'."""
    if 0 <= index <= 8:
        return str(index + 1)
    if index == 9:
        return "0"
    return None  # 11th+ card has no hotkey


class KeyTapDriver:
    """Sends discrete key taps via Win32 keybd_event."""

    available = True
    diagnostic = None

    def __init__(self, config) -> None:
        self.tap_seconds = max(0.0, float(getattr(config, "key_tap_seconds", 0.04)))
        self.gap_seconds = max(0.0, float(getattr(config, "key_gap_seconds", 0.09)))
        import sys

        if sys.platform != "win32":  # pragma: no cover - host is Windows
            raise InputUnavailable("keyboard input is only available on Windows")
        try:
            import ctypes

            self._user32 = ctypes.windll.user32
            self._ctypes = ctypes
            from ctypes import wintypes

            self._POINT = wintypes.POINT
        except Exception as exc:  # pragma: no cover - platform dependent
            raise InputUnavailable("could not initialize Win32 keyboard input") from exc

    def tap(self, token: str) -> None:
        vk = _VK.get(token)
        if vk is None:
            raise ValueError(f"unknown key token {token!r}")
        self._user32.keybd_event(vk, 0, 0, 0)
        if self.tap_seconds:
            time.sleep(self.tap_seconds)
        self._user32.keybd_event(vk, 0, 0x0002, 0)  # KEYEVENTF_KEYUP
        if self.gap_seconds:
            time.sleep(self.gap_seconds)

    def move_cursor(self, point: Point) -> None:
        self._user32.SetCursorPos(int(point[0]), int(point[1]))

    def close(self) -> None:
        pass


class DryRunKeyDriver:
    available = True
    diagnostic = "keyboard dry-run enabled"

    def __init__(self, config=None) -> None:
        self.taps: list[str] = []
        self.cursor: list[Point] = []

    def tap(self, token: str) -> None:
        if token not in _VK:
            raise ValueError(f"unknown key token {token!r}")
        self.taps.append(token)
        log.info("dry-run key tap: %s", token)

    def move_cursor(self, point: Point) -> None:
        self.cursor.append(point)

    def close(self) -> None:
        pass


class DisabledKeyDriver:
    available = False

    def __init__(self, diagnostic: str) -> None:
        self.diagnostic = diagnostic

    def tap(self, token: str) -> None:
        raise InputUnavailable(self.diagnostic)

    def move_cursor(self, point: Point) -> None:
        raise InputUnavailable(self.diagnostic)

    def close(self) -> None:
        pass


def build_key_driver(config):
    if getattr(config, "input_dry_run", False):
        return DryRunKeyDriver(config)
    try:
        return KeyTapDriver(config)
    except InputUnavailable as exc:
        return DisabledKeyDriver(str(exc))


class KeyboardCommandHandler:
    """Execute client commands with StS number-key hotkeys (+ a parked cursor for the drop
    zone). Same contract as the mouse/gamepad handlers so the server can use any of them."""

    def __init__(
        self,
        config,
        state_provider,
        *,
        key_driver=None,
        detector: FrameChangeDetector | None = None,
    ) -> None:
        self.config = config
        self.state_provider = state_provider
        self.key = key_driver or build_key_driver(config)
        self._target_settle = max(0.0, float(getattr(config, "key_target_settle_seconds", 0.22)))
        capture = getattr(state_provider, "capture", None)
        self.detector = detector or (FrameChangeDetector(capture, config) if capture else None)
        self._foreground_failures = 0
        if getattr(self.key, "diagnostic", None):
            level = logging.INFO if self.key.available else logging.WARNING
            log.log(level, "keyboard input driver: %s", self.key.diagnostic)

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
                command, state_hint, verify_state_change=verify_state_change, note_action=note_action
            )
        except (CommandError, ValueError, InputUnavailable) as exc:
            return False, str(exc)
        except Exception:
            log.exception("keyboard input command failed")
            return False, "input command failed (see host log)"

    def _execute(self, command, state_hint, *, verify_state_change, note_action):
        if command.verb == protocol.Verb.STATE:
            return True, None
        if not self.key.available:
            return False, getattr(self.key, "diagnostic", None) or "keyboard input unavailable"
        if not self._ensure_foreground():
            return False, "could not bring Slay the Spire to the foreground"
        if command.verb == protocol.Verb.POTION:
            return False, "potion execution is deferred until potion focus/tooltips are reliable"
        if command.verb == protocol.Verb.RAW:
            return False, "raw input requires the gamepad backend"
        if command.verb == protocol.Verb.PROCEED:
            self.key.tap("enter")
            return True, None
        if command.verb == protocol.Verb.RETURN:
            self.key.tap("escape")
            return True, None
        if command.verb == protocol.Verb.END:
            before = self._require_combat(state_hint)
            if note_action:
                self._note_action(command, before)
            before_sig = self._signature() if verify_state_change else None
            self.key.tap("end_turn")
            return self._verify(before_sig, verify_state_change, "end turn")
        if command.verb == protocol.Verb.PLAY:
            before = self._require_combat(state_hint)
            if note_action:
                self._note_action(command, before)
            ok, error = self._play(command.args, before, verify_state_change=verify_state_change)
            if not ok:
                self._clear_pending_action()
            return ok, error
        return False, f"unsupported command {command.verb!r}"

    def _play(self, args, before, *, verify_state_change):
        combat = before.combat_state
        assert combat is not None
        card_index = _int_arg(args, 0, "card")
        target_index = _int_arg(args, 1, "target", optional=True)
        if card_index < 0 or card_index >= len(combat.hand):
            return False, f"card index {card_index} is out of range"
        key = _card_key(card_index)
        if key is None:
            return False, f"hand card {card_index} has no keyboard hotkey (only 1-10 are bound)"
        living = [m for m in combat.monsters if _monster_alive(m)]
        if target_index is not None and not any(m.index == target_index for m in living):
            return False, f"target index {target_index} is out of range"

        before_sig = self._signature() if verify_state_change else None
        # IMPORTANT: do NOT move the mouse. StS leaves keyboard mode the moment the cursor
        # moves, which disables the arrow-key targeting. Pure key sequence:
        #   number  -> grab the card (getCardSelectedByHotkey)
        #   Enter   -> enters keyboard mode AND confirms: a targeted card auto-targets the
        #              first (leftmost) enemy and opens keyboard target mode; a non-targeted
        #              card moves to the drop zone ready to play.
        # Each Enter only takes effect one FRAME before single-target mode engages, so a
        # `right`/confirm sent too soon is routed to hand navigation instead of targeting --
        # hence the settle pauses at every state transition.
        self.key.tap(key)
        self.key.tap("enter")
        self._settle()  # let single-target mode (or the drop-zone hover) engage
        if target_index is None:
            self.key.tap("enter")  # second confirm plays the non-targeted card
            return self._verify(before_sig, verify_state_change, "play")
        # Targeted: walk from the leftmost enemy to the requested target, then confirm.
        for _ in range(self._target_steps(living, target_index)):
            self.key.tap("right")
            self._settle()  # let the target highlight move before the next key
        self.key.tap("enter")  # confirm on the selected target
        return self._verify(before_sig, verify_state_change, "play")

    def _settle(self) -> None:
        if self._target_settle > 0:
            time.sleep(self._target_settle)

    @staticmethod
    def _target_steps(living: list[Monster], target_index: int) -> int:
        """Right-presses from the leftmost living enemy to the requested target.

        StS auto-targets the leftmost enemy and RIGHT walks left-to-right; our monster list is
        already in board order, so the ordinal of the target among living monsters is the
        number of steps.
        """
        order = [m.index for m in living]
        try:
            return max(0, order.index(target_index))
        except ValueError:
            return 0

    # -- shared helpers -------------------------------------------------------
    def _verify(self, before_sig, verify_state_change, label) -> tuple[bool, str | None]:
        if not verify_state_change:
            return True, None
        if self._wait_for_change(before_sig):
            return True, None
        return False, f"{label} input sent, but no combat state change was observed"

    def _require_combat(self, state_hint):
        state = state_hint if _is_usable_combat(state_hint) else self._read_state()
        if state.screen_type != ScreenType.COMBAT or state.combat_state is None:
            raise CommandError(
                f"combat input is only available on COMBAT, got {state.screen_type.value}"
            )
        return state

    def _read_state(self):
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
            return True
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
        if not ok:
            self._foreground_failures += 1
            if self._foreground_failures == 1 or self._foreground_failures % 10 == 0:
                log.warning("could not bring Slay the Spire to the foreground; key input may be lost.")
        else:
            self._foreground_failures = 0
        return ok

    def _note_action(self, command, before):
        note = getattr(self.state_provider, "note_action", None)
        if note is None:
            return
        try:
            note(command, before)
        except Exception:
            log.debug("note_action failed", exc_info=True)

    def _clear_pending_action(self):
        clear = getattr(self.state_provider, "clear_pending_action", None)
        if clear is None:
            return
        try:
            clear()
        except Exception:
            log.debug("clear_pending_action failed", exc_info=True)


def _is_usable_combat(state: GameState | None) -> bool:
    return (
        state is not None
        and state.screen_type == ScreenType.COMBAT
        and state.combat_state is not None
    )
