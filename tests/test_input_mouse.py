from tspire.common import protocol
from tspire.common.schema import Card, CombatState, GameState, Monster, PlayerCombat, ScreenType
from tspire.host.capture import WindowRect
from tspire.host.config import HostConfig
from tspire.host.input.mouse import (
    CardTargetLocator,
    DryRunMouseDriver,
    FrameLayout,
    MouseCommandHandler,
)
from tspire.host.vision.backend import BBox
from tspire.host.vision.regions import RegionMap


def _cfg(**overrides):
    data = {"input_backend": "mouse", "mouse_verify_timeout": 0.02, "mouse_verify_poll": 0.0}
    data.update(overrides)
    return HostConfig(**data)


def _combat(*, hand=2, monsters=1, read_status="fresh"):
    state = GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        combat_state=CombatState(
            player=PlayerCombat(energy=3, current_hp=70, max_hp=80),
            hand=[Card(name=f"Card {i}", cost=1, index=i) for i in range(hand)],
            monsters=[
                Monster(name=f"Enemy {i}", current_hp=10, max_hp=10, index=i)
                for i in range(monsters)
            ],
        ),
        available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
    )
    state.read_status = read_status
    return state


class FakeStateProvider:
    def __init__(self, states):
        self.states = list(states)
        self.reads = 0

    def read(self):
        idx = min(self.reads, len(self.states) - 1)
        self.reads += 1
        return self.states[idx]


class FakeMouseDriver:
    available = True
    diagnostic = None

    def __init__(self):
        self.clicks = []
        self.drags = []

    def click(self, x, y):
        self.clicks.append((x, y))

    def drag(self, start, end):
        self.drags.append((start, end))

    def close(self):
        pass


class FakeLocator:
    def __init__(self, *, cards, monsters, play=(960, 430), end=(1300, 850)):
        self._layout = FrameLayout(cards=list(cards), monsters=dict(monsters))
        self._play = play
        self._end = end
        self.calls = 0

    def locate(self, *, expected_hand, monsters):
        self.calls += 1
        return self._layout

    def play_zone_point(self):
        return self._play

    def end_turn_point(self):
        return self._end


class FakeDetector:
    def __init__(self, changed=True):
        # changed may be a bool (constant) or a list consumed one result per wait.
        self._changed = changed
        self.waits = 0

    def signature(self):
        return object()

    def wait_for_change(self, before):
        self.waits += 1
        if isinstance(self._changed, list):
            return self._changed.pop(0) if self._changed else False
        return self._changed


class FakeKeyDriver:
    def __init__(self):
        self.presses = []

    def press(self, token, duration=None):
        self.presses.append(token)


class FakeKeyboardFallback:
    def __init__(self, result=(True, None)):
        self.result = result
        self.calls = []

    def execute(self, command, state_hint=None, *, verify_state_change=True, note_action=True):
        self.calls.append(
            {
                "command": command,
                "state_hint": state_hint,
                "verify_state_change": verify_state_change,
                "note_action": note_action,
            }
        )
        return self.result


def _handler(
    state,
    *,
    config=None,
    driver=None,
    locator=None,
    detector=None,
    provider=None,
    key=None,
    keyboard_fallback=None,
):
    provider = provider or FakeStateProvider([state])
    return MouseCommandHandler(
        config or _cfg(),
        provider,
        driver=driver or FakeMouseDriver(),
        locator=locator,
        detector=detector or FakeDetector(changed=True),
        key_driver=key or FakeKeyDriver(),
        keyboard_fallback=keyboard_fallback,
    )


# --------------------------------------------------------------------------- #
# Command handler
# --------------------------------------------------------------------------- #
def test_play_with_target_drags_card_onto_monster():
    state = _combat(hand=3, monsters=2)
    driver = FakeMouseDriver()
    locator = FakeLocator(
        cards=[(100, 900), (200, 900), (300, 900)],
        monsters={0: (700, 400), 1: (900, 400)},
    )
    handler = _handler(state, driver=driver, locator=locator)

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["1", "1"]), state_hint=state)

    assert ok and error is None
    assert driver.drags == [((200, 900), (900, 400))]
    assert driver.clicks == []


def test_multi_enemy_target_miss_cancels_then_retries_play_zone():
    # 2 enemies: the monster drag changes nothing (missed), so the handler cancels (Esc) to
    # un-stick targeting, then retries the play zone (handles a non-targeted card given a
    # target). Succeeds on the retry.
    state = _combat(hand=3, monsters=2)
    driver = FakeMouseDriver()
    key = FakeKeyDriver()
    locator = FakeLocator(
        cards=[(100, 900), (200, 900), (300, 900)],
        monsters={0: (700, 400), 1: (900, 400)},
        play=(960, 430),
    )
    handler = _handler(state, driver=driver, locator=locator, detector=FakeDetector([False, True]), key=key)

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["2", "0"]), state_hint=state)

    assert ok and error is None
    assert driver.drags == [((300, 900), (700, 400)), ((300, 900), (960, 430))]
    assert key.presses == ["cancel"]  # un-stuck targeting before the retry


def test_multi_enemy_target_miss_falls_back_to_keyboard():
    state = _combat(hand=3, monsters=2)
    driver = FakeMouseDriver()
    key = FakeKeyDriver()
    fallback = FakeKeyboardFallback()
    locator = FakeLocator(
        cards=[(100, 900), (200, 900), (300, 900)],
        monsters={0: (700, 400), 1: (900, 400)},
        play=(960, 430),
    )
    handler = _handler(
        state,
        driver=driver,
        locator=locator,
        detector=FakeDetector([False, False]),
        key=key,
        keyboard_fallback=fallback,
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["2", "1"]), state_hint=state)

    assert ok and error is None
    assert driver.drags == [((300, 900), (900, 400)), ((300, 900), (960, 430))]
    assert key.presses == ["cancel", "cancel"]
    assert len(fallback.calls) == 1
    call = fallback.calls[0]
    assert call["command"].verb == protocol.Verb.PLAY
    assert call["command"].args == ["2", "1"]
    assert call["state_hint"] is state
    assert call["verify_state_change"] is True
    assert call["note_action"] is False


def test_multi_enemy_target_miss_reports_keyboard_fallback_failure():
    state = _combat(hand=2, monsters=2)
    fallback = FakeKeyboardFallback((False, "keyboard input unavailable"))
    locator = FakeLocator(
        cards=[(100, 900), (200, 900)],
        monsters={0: (700, 400), 1: (900, 400)},
        play=(960, 430),
    )
    handler = _handler(
        state,
        locator=locator,
        detector=FakeDetector([False, False]),
        keyboard_fallback=fallback,
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["1", "1"]), state_hint=state)

    assert not ok
    assert "keyboard fallback also failed" in error
    assert "keyboard input unavailable" in error


def test_multi_enemy_target_miss_can_disable_keyboard_fallback():
    state = _combat(hand=2, monsters=2)
    fallback = FakeKeyboardFallback()
    locator = FakeLocator(
        cards=[(100, 900), (200, 900)],
        monsters={0: (700, 400), 1: (900, 400)},
        play=(960, 430),
    )
    handler = _handler(
        state,
        config=_cfg(mouse_keyboard_fallback=False),
        locator=locator,
        detector=FakeDetector([False, False]),
        keyboard_fallback=fallback,
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["1", "1"]), state_hint=state)

    assert not ok
    assert "targeted play did not resolve" in error
    assert fallback.calls == []


def test_single_enemy_targeted_play_uses_play_zone_auto_target():
    # With one living enemy StS auto-targets, so a targeted card is dropped in the play zone
    # (no enemy coordinate needed) rather than dragged onto a detected bar.
    state = _combat(hand=2, monsters=1)
    driver = FakeMouseDriver()
    locator = FakeLocator(cards=[(100, 900), (200, 900)], monsters={0: (700, 400)}, play=(960, 430))
    handler = _handler(state, driver=driver, locator=locator)

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=state)

    assert ok and error is None
    assert driver.drags == [((100, 900), (960, 430))]  # play zone, not the monster point


def test_play_without_target_drags_card_to_play_zone():
    state = _combat(hand=2, monsters=1)
    driver = FakeMouseDriver()
    locator = FakeLocator(cards=[(100, 900), (200, 900)], monsters={0: (700, 400)}, play=(960, 430))
    handler = _handler(state, driver=driver, locator=locator)

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]), state_hint=state)

    assert ok and error is None
    assert driver.drags == [((100, 900), (960, 430))]


def test_end_turn_clicks_end_turn_point():
    state = _combat()
    driver = FakeMouseDriver()
    locator = FakeLocator(cards=[(100, 900)], monsters={0: (700, 400)}, end=(1300, 850))
    handler = _handler(state, driver=driver, locator=locator)

    ok, error = handler.execute(protocol.Command(protocol.Verb.END), state_hint=state)

    assert ok and error is None
    assert driver.clicks == [(1300, 850)]


def test_play_fails_when_no_state_change_observed():
    state = _combat(hand=2, monsters=1)
    locator = FakeLocator(cards=[(100, 900), (200, 900)], monsters={0: (700, 400)})
    handler = _handler(state, locator=locator, detector=FakeDetector(changed=False))

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]), state_hint=state)

    assert not ok
    assert "no combat state change" in error
    assert "provide a target" in error  # non-targeted hint


def test_play_rejects_out_of_range_card():
    state = _combat(hand=2, monsters=1)
    handler = _handler(state, locator=FakeLocator(cards=[(1, 1), (2, 2)], monsters={0: (3, 3)}))
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["5"]), state_hint=state)
    assert not ok
    assert "out of range" in error


def test_play_rejects_out_of_range_target():
    state = _combat(hand=2, monsters=1)
    handler = _handler(state, locator=FakeLocator(cards=[(1, 1), (2, 2)], monsters={0: (3, 3)}))
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "4"]), state_hint=state)
    assert not ok
    assert "out of range" in error


def test_accepts_stale_combat_hint_without_reading_state():
    # Mouse play uses live CV coordinates, so a slightly stale combat hint is fine: it must
    # NOT trigger a (slow) authoritative read just to satisfy a freshness gate.
    state = _combat(hand=2, monsters=1, read_status="stale")
    provider = FakeStateProvider([state])
    locator = FakeLocator(cards=[(100, 900), (200, 900)], monsters={0: (700, 400)})
    handler = _handler(state, locator=locator, provider=provider)

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=state)

    assert ok and error is None
    assert provider.reads == 0


def test_state_command_is_ok():
    handler = _handler(_combat())
    ok, error = handler.execute(protocol.Command(protocol.Verb.STATE))
    assert ok and error is None


def test_proceed_and_return_use_key_driver():
    class FakeKeyDriver:
        def __init__(self):
            self.presses = []

        def press(self, token, duration=None):
            self.presses.append(token)

    key = FakeKeyDriver()
    handler = MouseCommandHandler(
        _cfg(),
        FakeStateProvider([_combat()]),
        driver=FakeMouseDriver(),
        locator=FakeLocator(cards=[(1, 1)], monsters={0: (2, 2)}),
        detector=FakeDetector(),
        key_driver=key,
    )

    assert handler.execute(protocol.Command(protocol.Verb.PROCEED)) == (True, None)
    assert handler.execute(protocol.Command(protocol.Verb.RETURN)) == (True, None)
    assert key.presses == ["proceed", "cancel"]


def test_potion_is_deferred():
    handler = _handler(_combat())
    ok, error = handler.execute(protocol.Command(protocol.Verb.POTION, ["use", "0"]))
    assert not ok and "deferred" in error


def test_foreground_failure_aborts_before_any_input():
    class Capture:
        def ensure_foreground(self, *, click_safe_zone=True):
            return False

    class Provider(FakeStateProvider):
        def __init__(self, states):
            super().__init__(states)
            self.capture = Capture()

    driver = FakeMouseDriver()
    handler = MouseCommandHandler(
        _cfg(),
        Provider([_combat()]),
        driver=driver,
        locator=FakeLocator(cards=[(1, 1)], monsters={0: (2, 2)}),
        detector=FakeDetector(),
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]), state_hint=_combat())

    assert not ok
    assert "foreground" in error
    assert driver.drags == [] and driver.clicks == []


def test_dry_run_driver_records_drag():
    driver = DryRunMouseDriver()
    driver.drag((1, 2), (3, 4))
    driver.click(5, 6)
    assert driver.drags == [((1, 2), (3, 4))]
    assert driver.clicks == [(5, 6)]


# --------------------------------------------------------------------------- #
# Coordinate locator
# --------------------------------------------------------------------------- #
class FakeFrame:
    shape = (1080, 1920, 3)


class FakeBackend:
    def __init__(self, *, cards, bars):
        self._cards = cards
        self._bars = bars

    def find_cards(self, frame, search):
        return list(self._cards)

    def find_red_bars(self, frame, search):
        return list(self._bars)


class FakeCapture:
    def __init__(self, rect):
        self._rect = rect

    def grab(self):
        return FakeFrame()

    def client_rect(self):
        return self._rect


class LocatorProvider:
    def __init__(self, backend, rect):
        self.capture = FakeCapture(rect)
        self.regions = RegionMap()
        self._backend = backend

    def _get_backend(self):
        return self._backend


def test_locator_uses_exact_sts_hand_layout():
    # Cards come from StS's deterministic hand layout (centred, table-driven spacing), not CV.
    rect = WindowRect(left=50, top=20, width=1920, height=1080)
    backend = FakeBackend(cards=[], bars=[BBox(left=900, top=600, width=160, height=20)])
    locator = CardTargetLocator(LocatorProvider(backend, rect), _cfg())

    monsters = [Monster(name="E", current_hp=10, max_hp=10, index=0)]
    layout = locator.locate(expected_hand=3, monsters=monsters)

    assert layout.card_source == "sts-layout"
    xs = [x for x, _ in layout.cards]
    centre = 50 + 960  # client centre x
    # 3-card offsets are -0.9, 0, +0.9 * (210/1920) of width -> symmetric about centre.
    assert xs[1] == centre
    assert centre - xs[0] == xs[2] - centre  # symmetric
    assert xs[0] == 50 + round((0.5 - 0.9 * (210 / 1920)) * 1920)  # == 821
    # monster point still comes from the HP bar centre.
    assert layout.monsters[0][0] == 50 + (900 + 80)


def test_locator_card_x_is_resolution_independent_fraction():
    # The same hand maps to the same x-fractions regardless of resolution/offset.
    a = CardTargetLocator(
        LocatorProvider(FakeBackend(cards=[], bars=[]), WindowRect(0, 0, 1920, 1080)), _cfg()
    ).locate(expected_hand=5, monsters=[])
    b = CardTargetLocator(
        LocatorProvider(FakeBackend(cards=[], bars=[]), WindowRect(100, 200, 1280, 720)), _cfg()
    ).locate(expected_hand=5, monsters=[])
    fa = [x / 1920 for x, _ in a.cards]
    fb = [(x - 100) / 1280 for x, _ in b.cards]
    for ea, eb in zip(fa, fb):
        assert abs(ea - eb) < 0.002


def test_locator_falls_back_for_oversized_hand():
    # Hands larger than the layout table (11+ via relics) fall back to CV/geometric.
    rect = WindowRect(left=0, top=0, width=1920, height=1080)
    backend = FakeBackend(cards=[BBox(left=100, top=800, width=120, height=200)], bars=[])
    locator = CardTargetLocator(LocatorProvider(backend, rect), _cfg())

    layout = locator.locate(expected_hand=11, monsters=[])

    assert layout.card_source in ("cv", "geometric")
    assert len(layout.cards) == 11
    xs = [x for x, _ in layout.cards]
    assert xs == sorted(xs)


class MutableCapture:
    def __init__(self, frame):
        self.frame = frame

    def grab(self):
        return self.frame


def _np_cv2():
    try:
        import cv2  # noqa: F401
        import numpy as np

        return np, cv2
    except Exception:  # pragma: no cover - native deps absent
        return None, None


def test_change_detector_catches_small_bottom_band_change():
    np, _ = _np_cv2()
    if np is None:
        return
    from tspire.host.input.mouse import FrameChangeDetector

    h, w = 200, 320
    base = np.full((h, w, 3), 120, dtype=np.uint8)
    cap = MutableCapture(base)
    detector = FrameChangeDetector(cap, _cfg(mouse_verify_timeout=0.05, mouse_verify_poll=0.0))

    before = detector.signature()
    # Change only a small patch low in the frame (like a block badge / energy / a card slot).
    changed = base.copy()
    changed[150:175, 40:90, :] = 255
    # This patch is a tiny fraction of the WHOLE frame, but a meaningful part of the band.
    cap.frame = changed
    assert detector.wait_for_change(before) is True


def test_change_detector_reports_no_change_for_identical_frames():
    np, _ = _np_cv2()
    if np is None:
        return
    from tspire.host.input.mouse import FrameChangeDetector

    base = np.full((200, 320, 3), 120, dtype=np.uint8)
    cap = MutableCapture(base)
    detector = FrameChangeDetector(cap, _cfg(mouse_verify_timeout=0.04, mouse_verify_poll=0.0))
    before = detector.signature()
    assert detector.wait_for_change(before) is False


def test_locator_end_turn_and_play_zone_points_from_regions_and_config():
    rect = WindowRect(left=10, top=10, width=1920, height=1080)
    locator = CardTargetLocator(LocatorProvider(FakeBackend(cards=[], bars=[]), rect), _cfg())

    px, py = locator.play_zone_point()
    assert px == 10 + round(0.5 * 1920)
    assert py == 10 + round(0.40 * 1080)
    # end-turn point is the centre of the end_turn region.
    ex, ey = locator.end_turn_point()
    assert 10 < ex < 10 + 1920 and 10 < ey < 10 + 1080
