from tspire.common import protocol
from tspire.common.schema import Card, CombatState, GameState, Monster, PlayerCombat, ScreenType
from tspire.host.config import HostConfig
from tspire.host.input.driver import normalize_token
from tspire.host.input.executor import GamepadCommandHandler, _hand_direction, _hand_steps
from tspire.host.input.focus import FocusState


def test_hand_direction_takes_shortest_wrapping_path():
    # 5-card hand; cursor wraps, so direction is the shorter way around.
    assert _hand_direction(0, 2, 5) == "right"
    assert _hand_direction(2, 0, 5) == "left"
    assert _hand_direction(0, 4, 5) == "left"  # wrap: 0 -> 4 is one step left
    assert _hand_direction(4, 0, 5) == "right"  # wrap: 4 -> 0 is one step right
    assert _hand_direction(3, 3, 5) == "right"  # already there (either is fine)


def test_hand_steps_signed_shortest_count():
    assert _hand_steps(0, 2, 5) == 2  # two steps right
    assert _hand_steps(2, 0, 5) == -2  # two steps left
    assert _hand_steps(0, 4, 5) == -1  # wrap: one step left
    assert _hand_steps(4, 0, 5) == 1  # wrap: one step right
    assert _hand_steps(3, 3, 5) == 0


class FakeStateProvider:
    def __init__(self, states):
        self.states = list(states)
        self.reads = 0

    def read(self):
        idx = min(self.reads, len(self.states) - 1)
        self.reads += 1
        return self.states[idx]


class FakeDriver:
    available = True
    diagnostic = None

    def __init__(self):
        self.presses = []

    def press(self, token, duration=None):
        self.presses.append(normalize_token(token))

    def close(self):
        pass


class FakeObserver:
    def __init__(self, states):
        self.states = list(states)
        self.last = self.states[-1] if self.states else FocusState()

    def observe(self, *, hand_count=None, target_count=None):
        if self.states:
            self.last = self.states.pop(0)
        return self.last


def _cfg(**overrides):
    data = {
        "input_dry_run": False,
        "input_press_seconds": 0.0,
        "input_step_delay": 0.0,
        "input_settle_seconds": 0.0,
        "input_command_timeout": 0.02,
    }
    data.update(overrides)
    return HostConfig(**data)


def _combat(*, energy=3, hand=2, monsters=1):
    return GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        combat_state=CombatState(
            player=PlayerCombat(energy=energy, current_hp=70, max_hp=80),
            hand=[Card(name=f"Card {i}", cost=1, index=i) for i in range(hand)],
            monsters=[
                Monster(name=f"Enemy {i}", current_hp=10, max_hp=10, index=i)
                for i in range(monsters)
            ],
        ),
        available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
    )


def test_potion_command_is_explicitly_deferred():
    handler = GamepadCommandHandler(_cfg(), FakeStateProvider([_combat()]), driver=FakeDriver())
    ok, error = handler.execute(protocol.Command(protocol.Verb.POTION, ["use", "0"]))
    assert not ok
    assert "deferred" in error


def test_raw_command_requires_debug_flag():
    driver = FakeDriver()
    handler = GamepadCommandHandler(_cfg(), FakeStateProvider([_combat()]), driver=driver)
    ok, error = handler.execute(protocol.Command(protocol.Verb.RAW, ["a"]))
    assert not ok
    assert "disabled" in error
    assert driver.presses == []


def test_raw_command_sends_normalized_tokens_when_enabled():
    driver = FakeDriver()
    cfg = _cfg(input_raw_enabled=True)
    handler = GamepadCommandHandler(cfg, FakeStateProvider([_combat()]), driver=driver)
    ok, error = handler.execute(protocol.Command(protocol.Verb.RAW, ["a", "right"]))
    assert ok and error is None
    assert driver.presses == ["select", "right"]


def test_play_in_dry_run_records_hand_and_target_sequence():
    cfg = _cfg(input_dry_run=True)
    provider = FakeStateProvider([_combat(hand=3, monsters=1)])
    handler = GamepadCommandHandler(cfg, provider)
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["1", "0"]))
    assert ok and error is None
    assert handler.driver.presses == [
        "down",
        "left",
        "left",
        "left",
        "left",
        "left",
        "right",
        "select",
        "select",
    ]


def test_play_without_target_detects_target_mode_and_cancels():
    state = _combat(hand=1, monsters=1)
    driver = FakeDriver()
    observer = FakeObserver([
        FocusState(hand_index=0),
        FocusState(hand_index=0),
        FocusState(target_index=0),
    ])
    handler = GamepadCommandHandler(
        _cfg(),
        FakeStateProvider([state, state, state]),
        driver=driver,
        observer=observer,
    )
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]))
    assert not ok
    assert "provide a target" in error
    assert driver.presses[-1] == "cancel"


def test_play_with_observed_target_succeeds_after_state_change():
    before = _combat(hand=2, monsters=2, energy=3)
    after = _combat(hand=1, monsters=2, energy=2)
    driver = FakeDriver()
    observer = FakeObserver(
        [
            FocusState(hand_index=0),  # _wait_for_any_focus: cursor starts on card 0
            FocusState(hand_index=0),  # loop sees 0, steps right toward target card 1
            FocusState(hand_index=1),  # reached card 1
            FocusState(target_index=0),  # targeting: cursor on monster 0
            FocusState(target_index=1),  # after stepping right: on monster 1
            FocusState(target_index=1),  # confirm
        ]
    )
    handler = GamepadCommandHandler(
        _cfg(),
        FakeStateProvider([after]),
        driver=driver,
        observer=observer,
    )
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["1", "1"]), state_hint=before)
    assert ok and error is None
    # Closed-loop navigation: exact press count varies, but the card and target are both
    # selected, navigation starts by entering the hand, and the last action is a select.
    assert driver.presses.count("select") == 2
    assert driver.presses[0] == "down"
    assert driver.presses[-1] == "select"
    assert "right" in driver.presses


def test_play_uses_cached_state_hint_without_full_state_read():
    state = _combat(hand=2, monsters=1)
    provider = FakeStateProvider([_combat(hand=1, monsters=1)])
    driver = FakeDriver()
    observer = FakeObserver([FocusState(hand_index=0), FocusState(hand_index=0)])
    handler = GamepadCommandHandler(
        _cfg(),
        provider,
        driver=driver,
        observer=observer,
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=state)

    assert ok and error is None
    assert provider.reads == 1
    assert driver.presses == ["down", "up", "down", "select", "select"]


def test_play_card_zero_falls_back_when_focus_observer_cannot_anchor():
    state = _combat(hand=2, monsters=2)
    driver = FakeDriver()
    handler = GamepadCommandHandler(
        _cfg(),
        FakeStateProvider([_combat(hand=1, monsters=2)]),
        driver=driver,
        observer=FakeObserver([FocusState()] * 20),
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=state)

    assert ok and error is None
    assert driver.presses[0] == "down"
    assert driver.presses.count("select") == 2


def test_play_card_zero_uses_first_card_reset_when_observer_is_stale():
    state = _combat(hand=4, monsters=2)
    driver = FakeDriver()
    observer = FakeObserver(
        [
            FocusState(hand_index=2),  # stale or wrong read after reset
            FocusState(target_index=0),
        ]
    )
    handler = GamepadCommandHandler(
        _cfg(),
        FakeStateProvider([_combat(hand=3, monsters=2)]),
        driver=driver,
        observer=observer,
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=state)

    assert ok and error is None
    assert driver.presses[:3] == ["down", "up", "down"]
    assert driver.presses.count("select") == 2


def test_play_target_zero_falls_back_when_confirmation_fails():
    state = _combat(hand=2, monsters=2)
    driver = FakeDriver()
    observer = FakeObserver(
        [
            FocusState(hand_index=0),
            FocusState(hand_index=0),
            FocusState(target_index=1),
            FocusState(),
            FocusState(),
            FocusState(),
        ]
    )
    handler = GamepadCommandHandler(
        _cfg(),
        FakeStateProvider([_combat(hand=1, monsters=2)]),
        driver=driver,
        observer=observer,
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=state)

    assert ok and error is None
    assert driver.presses.count("select") == 2
    assert driver.presses[-1] == "select"


def test_play_reports_failure_when_inputs_do_not_change_combat_state():
    state = _combat(hand=2, monsters=1)
    driver = FakeDriver()
    handler = GamepadCommandHandler(
        _cfg(),
        FakeStateProvider([state, state]),
        driver=driver,
        observer=FakeObserver([FocusState(hand_index=0), FocusState(target_index=0)]),
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=state)

    assert not ok
    assert "no combat state change" in error
    assert driver.presses.count("select") == 2


def test_input_aborts_when_foregrounding_fails():
    class Capture:
        def find_window(self):
            return object()

        def ensure_foreground(self):
            return False

    class Provider(FakeStateProvider):
        capture = Capture()

    driver = FakeDriver()
    handler = GamepadCommandHandler(
        _cfg(),
        Provider([_combat()]),
        driver=driver,
        observer=FakeObserver([FocusState(hand_index=0)]),
    )

    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]))

    assert not ok
    assert "foreground" in error
    assert driver.presses == []


def test_end_turn_requires_combat_state_change():
    state = _combat()
    driver = FakeDriver()
    handler = GamepadCommandHandler(_cfg(), FakeStateProvider([state, state]), driver=driver)
    ok, error = handler.execute(protocol.Command(protocol.Verb.END))
    assert not ok
    assert "no combat state change" in error
    assert driver.presses == ["proceed"]


def test_combat_input_rejects_non_combat_screen():
    state = GameState(screen_type=ScreenType.MAP)
    handler = GamepadCommandHandler(_cfg(input_dry_run=True), FakeStateProvider([state]))
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]))
    assert not ok
    assert "only available on COMBAT" in error


def test_combat_input_rejects_stale_combat_state():
    state = _combat()
    state.read_status = "stale"
    handler = GamepadCommandHandler(_cfg(input_dry_run=True), FakeStateProvider([state]))
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]))
    assert not ok
    assert "fresh combat state" in error


def test_build_session_wires_gamepad_handler():
    try:
        import websockets  # noqa: F401
    except ModuleNotFoundError:
        try:
            import pytest
        except ModuleNotFoundError:
            return
        pytest.skip("websockets is not installed")

    from tspire.host import server

    session = server.build_session(_cfg(input_dry_run=True))
    assert isinstance(session.command_handler, GamepadCommandHandler)
