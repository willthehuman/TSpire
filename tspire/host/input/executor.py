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
    KeyboardDriver,
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
        # Fast image-based change detector (same one the mouse/keyboard backends use) so a
        # play/end is confirmed by a cheap frame diff, not a slow LLM re-read. None when there
        # is no capture (tests/fakes) -> the state-signature loop is used instead.
        _capture = getattr(state_provider, "capture", None)
        if _capture is not None:
            from tspire.host.input.mouse import FrameChangeDetector

            self.detector = FrameChangeDetector(_capture, config)
        else:
            self.detector = None
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
            log.exception("input command failed")
            return False, "input command failed (see host log)"

    def _execute(
        self,
        command: protocol.Command,
        state_hint: GameState | None = None,
        *,
        verify_state_change: bool = True,
        note_action: bool = True,
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
            ok, error = self._play(
                command.args,
                before,
                verify_state_change=verify_state_change,
            )
            if not ok:
                self._clear_pending_action()
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

    def _clear_pending_action(self) -> None:
        clear = getattr(self.state_provider, "clear_pending_action", None)
        if clear is None:
            return
        try:
            clear()
        except Exception:
            log.debug("clear_pending_action failed", exc_info=True)

    def _raw(self, args: list[str]) -> tuple[bool, str | None]:
        if not self.config.input_raw_enabled:
            return False, "raw input is disabled; set TSPIRE_INPUT_RAW=1 to enable it"
        if not args:
            return False, "raw input needs at least one token"
        tokens = [normalize_token(arg) for arg in args]
        for token in tokens:
            self._press(token)
        return True, None

    def _end_turn(self, before: GameState, *, verify_state_change: bool = True) -> tuple[bool, str | None]:
        before_sig = self._change_baseline(before)
        self._press("end_turn")
        if not verify_state_change:
            return True, None
        if not self._wait_for_change(before_sig):
            return False, "end turn input sent, but no combat state change was observed"
        return True, None

    def _play(
        self,
        args: list[str],
        before: GameState,
        *,
        verify_state_change: bool = True,
    ) -> tuple[bool, str | None]:
        combat = before.combat_state
        assert combat is not None  # for type checkers; _require_combat guarantees it.
        card_index = _int_arg(args, 0, "card")
        target_index = _int_arg(args, 1, "target", optional=True)
        if card_index < 0 or card_index >= len(combat.hand):
            return False, f"card index {card_index} is out of range"
        if target_index is not None and not _has_monster_index(combat.monsters, target_index):
            return False, f"target index {target_index} is out of range"

        if self._deterministic_nav:
            return self._play_deterministic(
                card_index, target_index, combat, before, verify_state_change
            )

        before_sig = self._change_baseline(before)
        if not self._focus_hand_card(card_index, len(combat.hand)):
            return False, f"could not verify focus on hand card {card_index}"
        self._press("select")

        if target_index is not None:
            if not self._focus_target(target_index, combat.monsters):
                self._press("cancel")
                return False, f"could not verify focus on target {target_index}"
            self._press("select")
            if not verify_state_change:
                return True, None
            if not self._wait_for_change(before_sig):
                return False, "play input sent, but no combat state change was observed"
            return self._played_ok()

        if self._target_focus_appeared(len(combat.monsters)):
            self._press("cancel")
            return False, "card did not resolve; provide a target index if it requires one"
        if not verify_state_change:
            return True, None
        if not self._wait_for_change(before_sig):
            return False, "play input sent, but no combat state change was observed"
        return self._played_ok()

    @property
    def _deterministic_nav(self) -> bool:
        # Dry-run can't observe focus, so it is always deterministic; otherwise honour config.
        return self._verification_bypassed or bool(
            getattr(self.config, "gamepad_deterministic_nav", True)
        )

    def _nav_delay(self) -> None:
        self._sleep(max(0.0, float(getattr(self.config, "gamepad_nav_delay", 0.06))))

    def _play_deterministic(
        self,
        card_index: int,
        target_index: int | None,
        combat,
        before: GameState,
        verify_state_change: bool,
    ) -> tuple[bool, str | None]:
        """Play a card with no closed-loop CV: anchor the hand cursor at card 0 and step.

        StS controller navigation is deterministic (one card per d-pad press, wrap-safe), and
        pressing DOWN out of inspect mode sets the hand index to 0 (decompiled
        ``AbstractPlayer``). So UP (enter inspect) then DOWN lands on card 0, then RIGHT steps
        to the target card. SELECT grabs it; a targeted card auto-targets the leftmost enemy,
        then RIGHT walks to the requested target and SELECT confirms.
        """
        living = [m for m in combat.monsters if _monster_alive(m)]

        self._press("up")
        self._nav_delay()
        self._press("down")  # -> hand card 0 (keyboardCardIndex = 0)
        self._nav_delay()
        for _ in range(card_index):
            self._press("right")
            self._nav_delay()
        self._press("select")  # grab; a targeted card auto-targets the leftmost enemy
        self._nav_delay()

        if target_index is not None:
            for _ in range(_target_ordinal(living, target_index)):
                self._press("right")
                self._nav_delay()

        # Snapshot the change baseline HERE -- after all navigation, with the card grabbed and
        # targeted but not yet played. The d-pad navigation moves/lifts cards and the reticle,
        # so a baseline taken earlier would read the navigation itself as a "change" and report
        # a false success. Let the grab/target settle first so only the final SELECT (the play)
        # registers as the change.
        self._sleep(self.timing.settle_seconds)
        baseline = self._change_baseline(before)

        if target_index is not None:
            self._press("select")  # confirm on the selected target
        else:
            self._press("select")  # second confirm plays a non-targeted card

        # NOTE: no clear-focus press here. After a play we are back in the hand with a card
        # lifted, and the NEXT play's UP+DOWN anchor both un-lifts it and re-anchors on card 0.
        # Pressing UP here would leave us in inspect mode, and the next anchor's UP would then
        # jump to the relics/top panel instead of the hand -- breaking the deterministic count.
        if not verify_state_change:
            return True, None
        if self._wait_for_change(baseline):
            return True, None
        return False, "play input sent, but no combat state change was observed"

    def _played_ok(self) -> tuple[bool, str | None]:
        """Return success, first releasing the card the game keeps lifted after a play.

        In controller mode a card is always focused/lifted; leaving it up obscures the next
        screen read. UP triggers the game's releaseCard(), returning the hand to a neutral fan.
        """
        if getattr(self.config, "gamepad_clear_focus_after_play", True):
            try:
                self._press("up")
            except Exception:
                log.debug("clear-focus press failed", exc_info=True)
        return True, None

    def _require_combat(self, state_hint: GameState | None = None) -> GameState:
        state = state_hint if _is_fresh_combat_state(state_hint) else self._read_state()
        if state.screen_type != ScreenType.COMBAT or state.combat_state is None:
            raise CommandError(f"combat input is only available on COMBAT, got {state.screen_type.value}")
        if state.read_status != "fresh":
            raise CommandError(f"fresh combat state is required, got {state.read_status}")
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
        if target == 0 and self._focus_first_hand_card(hand_count):
            return True
        anchor = self._establish_anchor(hand_count)
        if anchor is None:
            return self._focus_hand_card_unverified(target, hand_count)
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
            if target in (0, hand_count - 1) and check != target:
                log.warning(
                    "hand focus observer did not confirm edge card %s after deterministic steps",
                    target,
                )
                return True
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
                return self._focus_target_unverified(target, len(monsters))
            focused = 0

        direction = "right" if target > focused else "left"
        for expected in _range_exclusive(focused, target):
            self._press(direction)
            if not self._wait_for_focus(target_index=expected):
                return self._focus_target_unverified(target, len(monsters))
        if self._observe().target_index == target:
            return True
        return self._focus_target_unverified(target, len(monsters))

    def _focus_hand_card_unverified(self, target: int, hand_count: int) -> bool:
        """Fallback when the visual hand-focus observer cannot find an anchor.

        In controller mode, moving down from the battlefield commonly lands on the leftmost
        hand card, which is also the edge card our observer is worst at seeing. Use this only
        for card 0; other targets need a confirmed anchor because the hand cursor wraps.
        """
        if target != 0 or hand_count <= 0:
            return False
        log.warning("hand focus observer could not anchor; assuming hand card 0 after down")
        return True

    def _focus_first_hand_card(self, hand_count: int) -> bool:
        """Best-effort reset to the first hand card for edge-card plays.

        The hand cursor wraps, so repeated left/right cannot create a stable left edge.
        Leaving the hand and pressing down re-enters it at the leftmost card in controller
        mode. The observer often cannot confirm card 0, so accept the deterministic entry
        even when the follow-up read is missing or stale.
        """
        if hand_count <= 0:
            return False
        self._press("up")
        self._sleep(self.timing.settle_seconds)
        self._press("down")
        self._sleep(self.timing.settle_seconds)
        focus = self._observe().hand_index
        if focus == 0 or focus is None:
            return True
        log.warning("hand focus observer reported %s after first-card reset; trusting controller reset", focus)
        return True

    def _focus_target_unverified(self, target: int, target_count: int) -> bool:
        """Fallback after a left sweep when target-focus detection cannot confirm.

        Enemy targeting does not have the edge-card gap problem, but the glow detector can
        still miss on dark scenes. The existing left sweep is intended to land on target 0;
        from there, step right deterministically to the requested target.
        """
        if target < 0 or target >= target_count:
            return False
        log.warning("target focus observer could not confirm; using deterministic target steps")
        for _ in range(target_count + 2):
            self._press("left")
            self._sleep(self.timing.settle_seconds)
        for _ in range(target):
            self._press("right")
            self._sleep(self.timing.settle_seconds)
        return True

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

    def _change_baseline(self, before: GameState):
        """Snapshot a 'before' signature for verification: a cheap image signature when a
        detector is available, else the state signature (tests / no capture)."""
        if self.detector is not None:
            try:
                return ("frame", self.detector.signature())
            except Exception:
                log.debug("frame baseline failed; using state signature", exc_info=True)
        return ("state", _state_signature(before))

    def _wait_for_change(self, baseline) -> bool:
        if self._verification_bypassed:
            return True
        kind, sig = baseline
        if kind == "frame" and self.detector is not None:
            # Cheap frame diff -- no slow LLM re-reads on the input path.
            return self.detector.wait_for_change(sig)
        deadline = time.monotonic() + self.timing.command_timeout
        while time.monotonic() <= deadline:
            self._sleep(self.timing.settle_seconds)
            if _state_signature(self._read_state()) != sig:
                return True
        return False

    def _ensure_foreground(self) -> bool:
        capture = getattr(self.state_provider, "capture", None)
        ensure = getattr(capture, "ensure_foreground", None)
        if ensure is None or self._verification_bypassed:
            return True
        try:
            click_safe_zone = not isinstance(self.driver, KeyboardDriver)
            try:
                ok = bool(ensure(click_safe_zone=click_safe_zone))
            except TypeError:
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


def _target_ordinal(living: list[Monster], target_index: int) -> int:
    """Right-presses from the auto-targeted leftmost enemy to the requested target.

    StS auto-targets the leftmost living enemy and RIGHT walks left-to-right; our monster list
    is in board order, so the target's position among living monsters is the number of steps.
    """
    order = [m.index for m in living]
    try:
        return max(0, order.index(target_index))
    except ValueError:
        return 0


def _monster_alive(monster: Monster) -> bool:
    if monster.is_gone or monster.half_dead:
        return False
    return monster.max_hp > 0 or monster.current_hp > 0


def _is_combat_state(state: GameState | None) -> bool:
    return state is not None and state.screen_type == ScreenType.COMBAT and state.combat_state is not None


def _is_fresh_combat_state(state: GameState | None) -> bool:
    return _is_combat_state(state) and state.read_status == "fresh"


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
    data = state.to_dict()
    for key in (
        "available_commands",
        "parse_confidence",
        "read_status",
        "screen_message",
        "state_notes",
        "state_seq",
        "unknown_fields",
    ):
        data.pop(key, None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))
