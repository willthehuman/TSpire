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


def run_mouse(args) -> int:
    """Calibration/diagnostic for the mouse backend.

    Grabs a live combat frame, computes the click points the mouse backend would use for
    every hand card, monster, the play zone and the end-turn button, overlays them on the
    frame for eyeballing, and optionally performs one real play so you can confirm the
    coordinates land. Run with StS in a combat (controller support irrelevant here).
    """
    config = HostConfig.load()
    if args.dry_run:
        config.input_dry_run = True

    from tspire.host.input.mouse import CardTargetLocator, build_mouse_driver
    from tspire.host.state import ScreenStateProvider

    state_provider = ScreenStateProvider(config)
    capture = state_provider.capture
    try:
        capture.ensure_foreground(click_safe_zone=False)
    except Exception:
        log.debug("could not foreground game window", exc_info=True)

    state = state_provider.read()
    if state.screen_type != ScreenType.COMBAT or state.combat_state is None:
        print(f"not in combat (screen: {state.screen_type.value}); enter a combat first")
        return 1
    combat = state.combat_state
    print(f"hand={len(combat.hand)} monsters={[m.index for m in combat.monsters]}")

    # CV diagnostics (informational only): card positions now come from StS's exact hand
    # layout, so CV card detection no longer matters for hands up to 10. Bars still drive
    # monster targeting.
    found_bars = []
    try:
        backend = state_provider._get_backend()
        frame = capture.grab()
        found_cards = backend.find_cards(frame, state_provider.regions.hand_search)
        found_bars = backend.find_red_bars(frame, state_provider.regions.monster_search)
        print(f"CV find_cards: {len(found_cards)} box(es) [informational]; "
              f"find_red_bars: {len(found_bars)} bar(s) of {len(combat.monsters)} monster(s)")
        for i, b in enumerate(found_bars):
            print(f"    bar {i}: left={b.left} top={b.top} w={b.width} h={b.height} "
                  f"-> centre=({b.left + b.width // 2}, {b.top + b.height // 2})")
    except Exception:
        log.debug("CV detection diagnostics failed", exc_info=True)

    locator = CardTargetLocator(state_provider, config)
    layout = locator.locate(expected_hand=len(combat.hand), monsters=combat.monsters)
    play_zone = locator.play_zone_point()
    end_turn = locator.end_turn_point()

    print(f"card source: {layout.card_source}")
    for i, point in enumerate(layout.cards):
        print(f"  card {i}: {point}")
    for index, point in sorted(layout.monsters.items()):
        print(f"  monster {index}: {point}")
    print(f"  play zone: {play_zone}")
    print(f"  end turn:  {end_turn}")

    _overlay_mouse_points(capture, layout, play_zone, end_turn, args.save_frames, found_bars)

    if args.play_card is not None:
        from tspire.host.input.mouse import MouseCommandHandler
        from tspire.common import protocol

        handler = MouseCommandHandler(config, state_provider, locator=locator)
        play_args = [str(args.play_card)]
        if args.target is not None:
            play_args.append(str(args.target))
        print(f"playing card {play_args} ...")
        ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, play_args), state_hint=state)
        print(f"{'PASS' if ok else 'FAIL'}: {error or 'state changed'}")
        return 0 if ok else 1
    return 0


def _overlay_mouse_points(capture, layout, play_zone, end_turn, save_dir, bars=()) -> None:
    try:
        import cv2

        frame = capture.grab().copy()
        cr = capture.client_rect()
    except Exception:
        log.debug("could not capture frame for overlay", exc_info=True)
        return

    def to_frame(point):
        return (int(point[0] - cr.left), int(point[1] - cr.top))

    # Draw the raw detected HP bars (frame-space already) in white so we can see what the
    # detector found vs. where the chosen targets landed.
    for b in bars:
        cv2.rectangle(frame, (b.left, b.top), (b.left + b.width, b.top + b.height), (255, 255, 255), 2)

    # Draw the search regions so it's obvious whether they actually cover the cards/enemies.
    fh, fw = frame.shape[:2]
    regions = getattr(capture, "regions", None)
    try:
        from tspire.host.vision import region_map_for

        rmap = region_map_for(fw, fh)
        for name, color in (("hand_search", (0, 255, 255)), ("monster_search", (255, 255, 0))):
            rect = getattr(rmap, name)
            x, y, w, h = rect.to_pixels(fw, fh)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    except Exception:
        log.debug("could not draw search regions", exc_info=True)

    for i, point in enumerate(layout.cards):
        fx, fy = to_frame(point)
        cv2.circle(frame, (fx, fy), 10, (0, 255, 0), 2)
        cv2.putText(frame, f"c{i}", (fx + 8, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    for index, point in layout.monsters.items():
        fx, fy = to_frame(point)
        cv2.circle(frame, (fx, fy), 12, (0, 0, 255), 2)
        cv2.putText(frame, f"m{index}", (fx + 8, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for label, point, color in (("play", play_zone, (255, 200, 0)), ("end", end_turn, (255, 0, 255))):
        fx, fy = to_frame(point)
        cv2.drawMarker(frame, (fx, fy), color, cv2.MARKER_CROSS, 24, 2)
        cv2.putText(frame, label, (fx + 8, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    from pathlib import Path

    out_dir = Path(save_dir) if save_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mouse_points.png"
    cv2.imwrite(str(out_path), frame)
    print(f"overlay written to {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe whether StS detects the virtual controller.")
    parser.add_argument(
        "--mouse",
        action="store_true",
        help="mouse-backend calibration: overlay the computed card/monster/play/end click "
        "points on a live combat frame (and optionally play a card with --play-card)",
    )
    parser.add_argument("--play-card", type=int, default=None, dest="play_card",
                        help="(mouse mode) actually play this hand index, for an end-to-end check")
    parser.add_argument("--target", type=int, default=None,
                        help="(mouse mode) target monster index for --play-card")
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
    parser.add_argument(
        "--controller-check",
        action="store_true",
        dest="controller_check",
        help="gamepad setup diagnostic: list OS controllers and flag Steam Input / wrong-order "
        "/ missing virtual pad (no game or input required)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    if args.controller_check:
        return run_controller_check()
    if args.mouse:
        return run_mouse(args)
    return run(args)


def run_controller_check() -> int:
    """List the controllers the OS exposes and report gamepad-backend setup problems."""
    from tspire.host.input.controller_check import analyze_controllers, enumerate_controllers

    controllers = enumerate_controllers()
    print(f"OS controllers ({len(controllers)}):")
    for c in controllers:
        print(f"  [{c['id']}] {c['name']}")
    print("\nsetup diagnosis:")
    problems = analyze_controllers(controllers)
    for msg in problems:
        print(f"  - {msg}")
    ok = len(problems) == 1 and problems[0].startswith("Controller setup looks OK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
