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
        self._foreground_failures = 0
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

    def execute(
        self,
        command: protocol.Command,
        state_hint: GameState | None = None,
    ) -> tuple[bool, str | None]:
        try:
            return self._execute(command, state_hint)
        except (CommandError, ValueError, InputUnavailable) as exc:
            return False, str(exc)
        except Exception:
            log.exception("input command failed")
            return False, "input command failed (see host log)"

    def _execute(
        self,
        command: protocol.Command,
        state_hint: GameState | None = None,
    ) -> tuple[bool, str | None]:
        if command.verb == protocol.Verb.STATE:
            return True, None
        if not self.driver.available:
            return False, self.driver.diagnostic or "gamepad input unavailable"
        # StS only reads the virtual pad while it is the foreground window. _play/_end_turn
        # foreground it as a side-effect of reading state, but RAW does not; do it here so
        # every input-producing verb is foregrounded before any button is pressed.
        if not self._ensure_foreground():
            return False, "could not bring Slay the Spire to the foreground"
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
            ok, error = self._end_turn()
            if ok:
                self._note_action(command, state_hint)
            return ok, error
        if command.verb == protocol.Verb.PLAY:
            ok, error = self._play(command.args, state_hint)
            if ok:
                self._note_action(command, state_hint)
            return ok, error
        return False, f"unsupported command {command.verb!r}"

    def _note_action(self, command: protocol.Command, before: GameState | None) -> None:
        """Hand the executed action + pre-action state to the provider so its next read can
        predict-and-reconcile. Optional on the provider; guarded for stub/fake providers."""
        note = getattr(self.state_provider, "note_action", None)
        if note is None:
            return
        try:
            note(command, before)
        except Exception:
            log.debug("note_action failed", exc_info=True)

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

    def _play(
        self,
        args: list[str],
        state_hint: GameState | None = None,
    ) -> tuple[bool, str | None]:
        before = self._require_combat(state_hint)
        combat = before.combat_state
        assert combat is not None  # for type checkers; _require_combat guarantees it.
        card_index = _int_arg(args, 0, "card")
        target_index = _int_arg(args, 1, "target", optional=True)
        if card_index < 0 or card_index >= len(combat.hand):
            return False, f"card index {card_index} is out of range"
        if target_index is not None and not _has_monster_index(combat.monsters, target_index):
            return False, f"target index {target_index} is out of range"

        if not self._focus_hand_card(card_index, len(combat.hand)):
            return False, f"could not verify focus on hand card {card_index}"
        self._press("select")

        if target_index is not None:
            if not self._focus_target(target_index, combat.monsters):
                self._press("cancel")
                return False, f"could not verify focus on target {target_index}"
            self._press("select")
            return True, None

        if self._target_focus_appeared(len(combat.monsters)):
            self._press("cancel")
            return False, "card did not resolve; provide a target index if it requires one"
        return True, None

    def _require_combat(self, state_hint: GameState | None = None) -> GameState:
        state = state_hint if _is_combat_state(state_hint) else self._read_state()
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

        # StS's hand cursor WRAPS and moves exactly one card per press (verified live). The
        # focus observer is reliable mid-hand but can't see the edge cards (index 0 / last),
        # whose lifted preview leaves no detectable gap. So anchor on a CONFIRMED index and
        # step the computed wrap-aware distance; re-confirm when possible, and accept an
        # unconfirmable edge as the deterministic destination. Self-corrects from any
        # confirmed position if a press is dropped.
        self._press("down")  # ensure the cursor is in the hand
        anchor = self._establish_anchor(hand_count)
        if anchor is None:
            return False
        for _ in range(4):
            current = self._observe().hand_index
            if current == target:
                return True
            ref = current if current is not None else anchor
            steps = _hand_steps(ref, target, hand_count)
            direction = "right" if steps > 0 else "left"
            for _ in range(abs(steps)):
                self._press(direction)
                self._sleep(self.timing.settle_seconds)
            anchor = target  # deterministic single-card-per-press landed us here
            check = self._observe().hand_index
            if check == target:
                return True
            if check is None and target in (0, hand_count - 1):
                return True  # edge card: not CV-confirmable, but the step count is exact
        return False

    def _establish_anchor(self, hand_count: int) -> int | None:
        """Return a CONFIRMED hand index to navigate from.

        The cursor may start on an edge card (which the observer can't read) or off the hand
        entirely, so nudge it right one card at a time until a readable index appears.
        """
        deadline = time.monotonic() + self.timing.command_timeout
        nudges = 0
        while time.monotonic() <= deadline:
            idx = self._observe().hand_index
            if idx is not None:
                return idx
            if nudges >= hand_count + 1:
                return None  # never found a readable slot (e.g. a degenerate tiny hand)
            self._press("right")
            nudges += 1
            self._sleep(self.timing.settle_seconds)
        return None

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

    def _target_focus_appeared(self, target_count: int) -> bool:
        """Briefly detect whether selecting a card opened target selection."""
        if self._verification_bypassed or target_count <= 0:
            return False
        self._observe_target_count = target_count
        probe_seconds = min(
            self.timing.command_timeout,
            max(0.15, self.timing.settle_seconds * 2),
        )
        deadline = time.monotonic() + probe_seconds
        step = self.timing.settle_seconds if self.timing.settle_seconds > 0 else 0.01
        while time.monotonic() <= deadline:
            if self._observe().target_index is not None:
                return True
            self._sleep(min(step, 0.05))
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

    def _ensure_foreground(self) -> bool:
        capture = getattr(self.state_provider, "capture", None)
        ensure = getattr(capture, "ensure_foreground", None)
        if ensure is None or self._verification_bypassed:
            return True
        try:
            ok = bool(ensure())
        except Exception:
            log.debug("could not foreground game window", exc_info=True)
            ok = False
        if ok:
            self._foreground_failures = 0
        else:
            self._foreground_failures += 1
            # StS ignores controller input unless it is the foreground window, so a press
            # sent now will be dropped. Warn the user (throttled) so they can act.
            if self._foreground_failures == 1 or self._foreground_failures % 10 == 0:
                log.warning(
                    "could not bring Slay the Spire to the foreground; controller input may "
                    "be ignored. Make sure the game window isn't minimized and nothing is "
                    "forcing itself on top."
                )
        return ok

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


def _is_combat_state(state: GameState | None) -> bool:
    return state is not None and state.screen_type == ScreenType.COMBAT and state.combat_state is not None


def _hand_steps(current: int, target: int, hand_count: int) -> int:
    """Signed shortest step count around the wrapping hand cursor (+right / -left)."""
    right_steps = (target - current) % hand_count
    left_steps = (current - target) % hand_count
    return right_steps if right_steps <= left_steps else -left_steps


def _hand_direction(current: int, target: int, hand_count: int) -> str:
    """Shortest press direction around the wrapping hand cursor."""
    return "right" if _hand_steps(current, target, hand_count) >= 0 else "left"


def _range_exclusive(start: int, stop: int):
    if start < stop:
        return range(start + 1, stop + 1)
    return range(start - 1, stop - 1, -1)


def _state_signature(state: GameState) -> str:
    return json.dumps(state.to_dict(), sort_keys=True, separators=(",", ":"))
