"""Diagnostic: confirm Slay the Spire detects the virtual controller.

Sends a small navigation sequence (default ``left,left,right,right``) and reports the
card-focus index observed after each press. If the focus index moves, StS is receiving the
virtual pad; if it never moves (or is never seen), the printed verdict points at the likely
cause (window focus, in-game controller setting, ViGEmBus, or CV thresholds).

Run on the gaming PC with StS in a combat that has several cards in hand and controller
support enabled in-game::

    python -m tspire.host.input_probe --sequence left,left,right,right
    python -m tspire.host.input_probe --dry-run --hand-count 5   # offline wiring check
"""

from __future__ import annotations

import argparse
import logging
import time

from tspire.common.schema import ScreenType
from tspire.host.config import HostConfig
from tspire.host.input.driver import (
    DryRunDriver,
    InputTiming,
    build_driver,
    normalize_token,
)
from tspire.host.input.focus import NullFocusObserver, ScreenFocusObserver
from tspire.host.input.preflight import collect_preflight_warnings

log = logging.getLogger("tspire.host.input_probe")

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


def parse_sequence(text: str) -> list[str]:
    """Split a comma/space separated sequence into normalized gamepad tokens."""
    raw = [part for part in text.replace(",", " ").split() if part]
    if not raw:
        raise ValueError("sequence is empty")
    return [normalize_token(part) for part in raw]


def classify_result(observed: list[int | None]) -> tuple[str, str]:
    """Turn the per-step observed focus indices into a verdict + guidance message."""
    seen = [idx for idx in observed if idx is not None]
    distinct = set(seen)
    if len(distinct) >= 2:
        return PASS, "card focus moved in response to input; the game sees the virtual pad."
    if not seen:
        return (
            FAIL,
            "no card focus was ever detected. Most likely the game did not receive the "
            "input. Slay the Spire does NOT hot-plug controllers: the virtual pad must "
            "already exist when the game launches, so start the TSpire host (which creates "
            "the pad) BEFORE launching Slay the Spire. Also disable Steam Input for the "
            "game (Steam -> right-click Slay the Spire -> Properties -> Controller -> "
            "'Disable Steam Input'), since Steam otherwise captures the pad. Then confirm "
            "the game is the foreground window and 'Controller Enabled' is on in-game. "
            "If the highlight DOES move on screen but this still reports None, the "
            "focus-detection thresholds in tspire/host/input/focus.py need tuning instead.",
        )
    return (
        INCONCLUSIVE,
        f"focus was detected but never changed (stayed on index {seen[0]}). The hand may "
        "have a single card, or presses are not registering; try more cards or a longer "
        "--sequence.",
    )


def _resolve_hand_count(args, state_provider) -> int:
    if args.hand_count is not None:
        return args.hand_count
    state = state_provider.read()
    if state.screen_type != ScreenType.COMBAT or state.combat_state is None:
        raise SystemExit(
            f"not in combat (screen: {state.screen_type.value}); enter a combat or pass "
            "--hand-count N"
        )
    return len(state.combat_state.hand)


def run(args) -> int:
    config = HostConfig.load()
    if args.dry_run:
        config.input_dry_run = True
    timing = InputTiming(
        press_seconds=args.press_seconds if args.press_seconds is not None else config.input_press_seconds,
        step_delay=config.input_step_delay,
        settle_seconds=args.settle if args.settle is not None else config.input_settle_seconds,
        command_timeout=config.input_command_timeout,
    )

    sequence = parse_sequence(args.sequence)

    from tspire.host.state import ScreenStateProvider

    state_provider = ScreenStateProvider(config)

    for warning in collect_preflight_warnings(config, state_provider):
        print(f"preflight: {warning}")

    driver = build_driver(config, timing)
    if not driver.available:
        print(f"gamepad unavailable: {driver.diagnostic}")
        return 1
    if driver.diagnostic:
        print(f"driver: {driver.diagnostic}")

    if args.prelaunch_wait > 0:
        # StS only detects controllers present at launch, so give the user time to (re)launch
        # the game now that the virtual pad exists, then enter a combat.
        print(f"pad is up. (Re)launch Slay the Spire and enter a combat now; "
              f"waiting {args.prelaunch_wait}s...")
        time.sleep(args.prelaunch_wait)

    observer = NullFocusObserver() if isinstance(driver, DryRunDriver) else ScreenFocusObserver(state_provider)

    hand_count = _resolve_hand_count(args, state_provider)
    print(f"hand_count={hand_count}  sequence={sequence}")

    def observe() -> int | None:
        return observer.observe(hand_count=hand_count).hand_index

    capture = getattr(state_provider, "capture", None)

    frame_dir = None
    if args.save_frames:
        from pathlib import Path

        frame_dir = Path(args.save_frames)
        frame_dir.mkdir(parents=True, exist_ok=True)

    def save_frame(label: str) -> None:
        if frame_dir is None or capture is None:
            return
        try:
            import cv2

            cv2.imwrite(str(frame_dir / f"{label}.png"), capture.grab())
        except Exception:
            log.debug("could not save frame", exc_info=True)

    if capture is not None:
        try:
            capture.focus_window()
        except Exception:
            log.debug("could not foreground game window", exc_info=True)

    observed: list[int | None] = []
    baseline = observe()
    print(f"baseline: hand_index={baseline}")
    observed.append(baseline)
    save_frame("baseline")

    for step, token in enumerate(sequence, start=1):
        if capture is not None:
            try:
                capture.focus_window()
            except Exception:
                log.debug("could not foreground game window", exc_info=True)
        driver.press(token, timing.press_seconds)
        if timing.settle_seconds > 0:
            time.sleep(timing.settle_seconds)
        idx = observe()
        observed.append(idx)
        print(f"step {step}: pressed {token} -> hand_index={idx}")
        save_frame(f"step{step}_{token}_idx{idx}")

    if args.hold > 0:
        # Keep the virtual controller connected so a human can eyeball the lifted card;
        # without this the pad disconnects the instant the process exits.
        print(f"holding controller connected for {args.hold}s...")
        time.sleep(args.hold)

    driver.close()

    verdict, message = classify_result(observed)
    print(f"\n{verdict}: {message}")
    return 0 if verdict == PASS else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe whether StS detects the virtual controller.")
    parser.add_argument(
        "--sequence",
        default="left,left,right,right",
        help="comma/space separated gamepad tokens to send (default: left,left,right,right)",
    )
    parser.add_argument("--hand-count", type=int, default=None, help="cards in hand (else read from game state)")
    parser.add_argument("--press-seconds", type=float, default=None, help="override button press duration")
    parser.add_argument("--settle", type=float, default=None, help="override settle delay after each press")
    parser.add_argument(
        "--hold",
        type=float,
        default=0.0,
        help="keep the controller connected this many seconds after the sequence (for eyeballing)",
    )
    parser.add_argument(
        "--prelaunch-wait",
        type=float,
        default=0.0,
        dest="prelaunch_wait",
        help="create the pad, then wait this many seconds so you can (re)launch StS after it "
        "exists (StS only detects controllers present at launch)",
    )
    parser.add_argument(
        "--save-frames",
        default=None,
        dest="save_frames",
        metavar="DIR",
        help="write the captured frame at baseline and after each press into DIR (for tuning)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use the dry-run driver (offline wiring check; no real input, no focus detection)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
