"""Protocol command -> gamepad input executor."""

from __future__ import annotations

import json
import logging
import time
from typing import Protocol

from tspire.common import protocol
from tspire.common.schema import GameState, Monster, ScreenType
from tspire.host.input.driver import (
    DryRunDriver,
    GamepadDriver,
    InputTiming,
    InputUnavailable,
    build_driver,
    normalize_token,
)
from tspire.host.input.focus import FocusObserver, FocusState, NullFocusObserver, ScreenFocusObserver
from tspire.host.input.preflight import collect_preflight_warnings

log = logging.getLogger("tspire.host.input")


class StateProvider(Protocol):
    def read(self) -> GameState: ...


class CommandError(RuntimeError):
    pass


class GamepadCommandHandler:
    """Execute client commands with a virtual Xbox controller.

    M3 intentionally supports combat-first verbs. Unsupported surfaces fail explicitly
    rather than sending speculative input.
    """

    def __init__(
        self,
        config,
        state_provider: StateProvider,
        *,
        driver: GamepadDriver | None = None,
        observer: FocusObserver | None = None,
    ) -> None:
        self.config = config
        self.state_provider = state_provider
        self.timing = InputTiming.from_config(config)
        self.driver = driver or build_driver(config, self.timing)
        self._observe_hand_count: int | None = None
        self._observe_target_count: int | None = None
        if observer is not None:
            self.observer = observer
        elif isinstance(self.driver, DryRunDriver):
            self.observer = NullFocusObserver()
        else:
            self.observer = ScreenFocusObserver(state_provider)

        for warning in collect_preflight_warnings(config, state_provider):
            log.warning("input preflight: %s", warning)
        if self.driver.diagnostic:
            level = logging.INFO if self.driver.available else logging.WARNING
            log.log(level, "input driver: %s", self.driver.diagnostic)

    def execute(self, command: protocol.Command) -> tuple[bool, str | None]:
        try:
            return self._execute(command)
        except (CommandError, ValueError, InputUnavailable) as exc:
            return False, str(exc)
        except Exception:
            log.exception("input command failed")
            return False, "input command failed (see host log)"

    def _execute(self, command: protocol.Command) -> tuple[bool, str | None]:
        if command.verb == protocol.Verb.STATE:
            return True, None
        if not self.driver.available:
            return False, self.driver.diagnostic or "gamepad input unavailable"
        # StS only reads the virtual pad while it is the foreground window. _play/_end_turn
        # foreground it as a side-effect of reading state, but RAW does not; do it here so
        # every input-producing verb is foregrounded before any button is pressed.
        self._ensure_foreground()
        if command.verb == protocol.Verb.POTION:
            return False, "potion execution is deferred until potion focus/tooltips are reliable"
        if command.verb == protocol.Verb.RAW:
            return self._raw(command.args)
        if command.verb == protocol.Verb.PROCEED:
            self._press("proceed")
            return True, None
        if command.verb == protocol.Verb.RETURN:
            self._press("cancel")
            return True, None
        if command.verb == protocol.Verb.END:
            return self._end_turn()
        if command.verb == protocol.Verb.PLAY:
            return self._play(command.args)
        return False, f"unsupported command {command.verb!r}"

    def _raw(self, args: list[str]) -> tuple[bool, str | None]:
        if not self.config.input_raw_enabled:
            return False, "raw input is disabled; set TSPIRE_INPUT_RAW=1 to enable it"
        if not args:
            return False, "raw input needs at least one token"
        tokens = [normalize_token(arg) for arg in args]
        for token in tokens:
            self._press(token)
        return True, None

    def _end_turn(self) -> tuple[bool, str | None]:
        before = self._require_combat()
        before_sig = _state_signature(before)
        self._press("proceed")
        if not self._wait_for_change(before_sig):
            return False, "end turn input sent, but no combat state change was observed"
        return True, None

    def _play(self, args: list[str]) -> tuple[bool, str | None]:
        before = self._require_combat()
        combat = before.combat_state
        assert combat is not None  # for type checkers; _require_combat guarantees it.
        card_index = _int_arg(args, 0, "card")
        target_index = _int_arg(args, 1, "target", optional=True)
        if card_index < 0 or card_index >= len(combat.hand):
            return False, f"card index {card_index} is out of range"
        if target_index is not None and not _has_monster_index(combat.monsters, target_index):
            return False, f"target index {target_index} is out of range"

        before_sig = _state_signature(before)
        if not self._focus_hand_card(card_index, len(combat.hand)):
            return False, f"could not verify focus on hand card {card_index}"
        self._press("select")

        if target_index is not None:
            if not self._focus_target(target_index, combat.monsters):
                self._press("cancel")
                return False, f"could not verify focus on target {target_index}"
            self._press("select")

        if not self._wait_for_change(before_sig):
            self._press("cancel")
            if target_index is None:
                return False, "card did not resolve; provide a target index if it requires one"
            return False, "play input sent, but no combat state change was observed"
        return True, None

    def _require_combat(self) -> GameState:
        state = self._read_state()
        if state.screen_type != ScreenType.COMBAT or state.combat_state is None:
            raise CommandError(f"combat input is only available on COMBAT, got {state.screen_type.value}")
        return state

    def _read_state(self) -> GameState:
        try:
            return self.state_provider.read()
        except Exception as exc:
            raise CommandError(f"could not read game state: {exc}") from exc

    def _focus_hand_card(self, target: int, hand_count: int) -> bool:
        self._observe_hand_count = hand_count
        if self._verification_bypassed:
            self._press("down")
            for _ in range(hand_count + 2):
                self._press("left")
            for _ in range(target):
                self._press("right")
            return True

        for _ in range(2):
            self._press("down")
            for _ in range(hand_count + 2):
                self._press("left")
            if not self._wait_for_focus(hand_index=0):
                continue
            moved = True
            for i in range(1, target + 1):
                self._press("right")
                if not self._wait_for_focus(hand_index=i):
                    moved = False
                    break
            if moved:
                return True
        return False

    def _focus_target(self, target: int, monsters: list[Monster]) -> bool:
        self._observe_target_count = len(monsters)
        living = [m for m in monsters if _monster_alive(m)]
        if len(living) == 1 and living[0].index == target:
            return True
        if self._verification_bypassed:
            for _ in range(len(monsters) + 2):
                self._press("left")
            for _ in range(target):
                self._press("right")
            return True

        focused = self._observe().target_index
        if focused is None:
            for _ in range(len(monsters) + 2):
                self._press("left")
            if not self._wait_for_focus(target_index=0):
                return False
            focused = 0

        direction = "right" if target > focused else "left"
        for expected in _range_exclusive(focused, target):
            self._press(direction)
            if not self._wait_for_focus(target_index=expected):
                return False
        return self._observe().target_index == target

    def _wait_for_focus(
        self,
        *,
        hand_index: int | None = None,
        target_index: int | None = None,
    ) -> bool:
        deadline = time.monotonic() + self.timing.command_timeout
        while time.monotonic() <= deadline:
            focus = self._observe()
            if hand_index is not None and focus.hand_index == hand_index:
                return True
            if target_index is not None and focus.target_index == target_index:
                return True
            self._sleep(self.timing.settle_seconds)
        return False

    def _wait_for_change(self, before_sig: str) -> bool:
        if self._verification_bypassed:
            return True
        deadline = time.monotonic() + self.timing.command_timeout
        while time.monotonic() <= deadline:
            self._sleep(self.timing.settle_seconds)
            if _state_signature(self._read_state()) != before_sig:
                return True
        return False

    def _ensure_foreground(self) -> None:
        capture = getattr(self.state_provider, "capture", None)
        focus = getattr(capture, "focus_window", None)
        if focus is not None and not self._verification_bypassed:
            try:
                focus()
            except Exception:
                log.debug("could not foreground game window", exc_info=True)

    def _observe(self) -> FocusState:
        return self.observer.observe(
            hand_count=self._observe_hand_count,
            target_count=self._observe_target_count,
        )

    def _press(self, token: str) -> None:
        self.driver.press(token, self.timing.press_seconds)

    @staticmethod
    def _sleep(seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    @property
    def _verification_bypassed(self) -> bool:
        return isinstance(self.driver, DryRunDriver)


def _int_arg(args: list[str], pos: int, name: str, *, optional: bool = False) -> int | None:
    if pos >= len(args):
        if optional:
            return None
        raise CommandError(f"missing {name} index")
    token = args[pos]
    if not token.lstrip("-").isdigit():
        raise CommandError(f"{name} index must be a number, got {token!r}")
    return int(token)


def _has_monster_index(monsters: list[Monster], index: int) -> bool:
    return any(m.index == index and _monster_alive(m) for m in monsters)


def _monster_alive(monster: Monster) -> bool:
    if monster.is_gone or monster.half_dead:
        return False
    return monster.max_hp > 0 or monster.current_hp > 0


def _range_exclusive(start: int, stop: int):
    if start < stop:
        return range(start + 1, stop + 1)
    return range(start - 1, stop - 1, -1)


def _state_signature(state: GameState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
