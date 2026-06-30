from tspire.common import protocol
from tspire.common.schema import Card, CombatState, GameState, Monster, PlayerCombat, ScreenType
from tspire.host.config import HostConfig
from tspire.host.input.keyboard import KeyboardCommandHandler, _card_key, build_key_driver, DryRunKeyDriver


def _cfg(**overrides):
    data = {"input_backend": "keyboard", "key_target_settle_seconds": 0.0}
    data.update(overrides)
    return HostConfig(**data)


def _combat(*, hand=3, monsters=2, read_status="fresh"):
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


class FakeKeyDriver:
    available = True
    diagnostic = None

    def __init__(self):
        self.taps = []
        self.cursor = []

    def tap(self, token):
        self.taps.append(token)

    def move_cursor(self, point):
        self.cursor.append(point)

    def close(self):
        pass


class FakeDetector:
    def __init__(self, changed=True):
        self._changed = changed

    def signature(self):
        return object()

    def wait_for_change(self, before):
        return self._changed


def _handler(state, *, key=None, detector=None, provider=None):
    return KeyboardCommandHandler(
        _cfg(),
        provider or FakeStateProvider([state]),
        key_driver=key or FakeKeyDriver(),
        detector=detector or FakeDetector(changed=True),
    )


def test_card_key_mapping():
    assert [_card_key(i) for i in range(10)] == ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
    assert _card_key(10) is None


def test_non_target_play_number_then_two_confirms_no_mouse():
    key = FakeKeyDriver()
    handler = _handler(_combat(hand=3), key=key)
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["1"]), state_hint=_combat(hand=3))
    assert ok and error is None
    assert key.cursor == []  # never moves the mouse (would drop out of keyboard mode)
    # card index 1 -> key "2"; Enter enters keyboard mode + grabs to drop zone; Enter plays.
    assert key.taps == ["2", "enter", "enter"]


def test_target_play_walks_right_to_target_then_confirms():
    key = FakeKeyDriver()
    handler = _handler(_combat(hand=3, monsters=3), key=key)
    # target monster index 2 -> from leftmost (auto-targeted), two RIGHT presses.
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "2"]), state_hint=_combat(hand=3, monsters=3))
    assert ok and error is None
    assert key.taps == ["1", "enter", "right", "right", "enter"]


def test_target_play_on_first_enemy_needs_no_walk():
    key = FakeKeyDriver()
    handler = _handler(_combat(hand=2, monsters=2), key=key)
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0", "0"]), state_hint=_combat(hand=2, monsters=2))
    assert ok and error is None
    assert key.taps == ["1", "enter", "enter"]  # confirm auto-targets first, confirm plays


def test_card_index_10_uses_zero_key():
    key = FakeKeyDriver()
    state = _combat(hand=10, monsters=1)
    handler = _handler(state, key=key)
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["9"]), state_hint=state)
    assert ok and error is None
    assert key.taps == ["0", "enter", "enter"]  # non-targeted: number + two confirms


def test_end_turn_taps_e():
    key = FakeKeyDriver()
    handler = _handler(_combat(), key=key)
    ok, error = handler.execute(protocol.Command(protocol.Verb.END), state_hint=_combat())
    assert ok and error is None
    assert key.taps == ["end_turn"]


def test_proceed_and_return():
    key = FakeKeyDriver()
    handler = _handler(_combat(), key=key)
    assert handler.execute(protocol.Command(protocol.Verb.PROCEED)) == (True, None)
    assert handler.execute(protocol.Command(protocol.Verb.RETURN)) == (True, None)
    assert key.taps == ["enter", "escape"]


def test_play_rejects_out_of_range_card():
    handler = _handler(_combat(hand=2))
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["5"]), state_hint=_combat(hand=2))
    assert not ok and "out of range" in error


def test_play_reports_no_change():
    handler = _handler(_combat(hand=2), detector=FakeDetector(changed=False))
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]), state_hint=_combat(hand=2))
    assert not ok and "no combat state change" in error


def test_accepts_stale_combat_without_reading():
    provider = FakeStateProvider([_combat()])
    state = _combat(read_status="stale")
    handler = KeyboardCommandHandler(
        _cfg(), provider, key_driver=FakeKeyDriver(), detector=FakeDetector(changed=True),
    )
    ok, error = handler.execute(protocol.Command(protocol.Verb.PLAY, ["0"]), state_hint=state)
    assert ok and error is None
    assert provider.reads == 0


def test_dry_run_key_driver_records():
    d = DryRunKeyDriver()
    d.tap("2")
    d.move_cursor((1, 2))
    assert d.taps == ["2"] and d.cursor == [(1, 2)]


def test_build_session_wires_keyboard_handler():
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
    assert isinstance(session.command_handler, KeyboardCommandHandler)
