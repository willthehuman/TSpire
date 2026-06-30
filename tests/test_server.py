import asyncio
import time

import pytest

pytest.importorskip("websockets")

from tspire.common import protocol  # noqa: E402
from tspire.common.schema import Card, CombatState, GameState, Monster, PlayerCombat, ScreenType  # noqa: E402
from tspire.host import server  # noqa: E402
from tspire.host.config import HostConfig  # noqa: E402


class FakeWs:
    def __init__(self):
        self.sent = []

    async def send(self, message):
        self.sent.append(protocol.parse_message(message))


class FakeProvider:
    def __init__(self, states):
        self.states = list(states)
        self.reads = 0

    def read(self):
        idx = min(self.reads, len(self.states) - 1)
        self.reads += 1
        return self.states[idx]


class RecordingHandler:
    def __init__(self):
        self.calls = []

    def execute(self, command, state_hint=None, *, verify_state_change=True, note_action=True):
        self.calls.append((command, state_hint, {
            "verify_state_change": verify_state_change,
            "note_action": note_action,
        }))
        return True, None


def _cfg(**overrides):
    data = {"input_settle_seconds": 0.0}
    data.update(overrides)
    return HostConfig(**data)


def _state(screen=ScreenType.COMBAT):
    return GameState(
        screen_type=screen,
        available_commands=protocol.commands_for_screen(screen.value),
    )


def _combat_state(*, hand=None):
    hand = hand if hand is not None else [
        Card(name="Strike", cost=1, index=0),
        Card(name="Strike", cost=1, index=1),
    ]
    return GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        current_hp=70,
        max_hp=80,
        combat_state=CombatState(
            player=PlayerCombat(current_hp=70, max_hp=80, energy=3),
            monsters=[Monster(name="Jaw Worm", current_hp=40, max_hp=44, index=0)],
            hand=hand,
        ),
        available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
    )


@pytest.mark.asyncio
async def test_command_receives_last_state_hint_and_pushes_fresh_state():
    before = _state(ScreenType.COMBAT)
    after = _state(ScreenType.MAP)
    handler = RecordingHandler()
    session = server.GameSession(_cfg(), state_provider=FakeProvider([after]), command_handler=handler)
    session.last_state = before
    host = server.HostServer(session)
    ws = FakeWs()
    host.clients.add(ws)

    command = protocol.Command(protocol.Verb.PLAY, ["0"], id="abc")
    await host._on_message(ws, command.to_message())

    assert handler.calls[0][1] is before
    assert [msg["type"] for msg in ws.sent] == ["ack", "state"]
    assert ws.sent[0]["id"] == "abc"
    assert ws.sent[1]["state"]["screen_type"] == ScreenType.MAP.value


@pytest.mark.asyncio
async def test_state_altering_command_settles_before_refresh(monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    host = server.HostServer(
        server.GameSession(
            _cfg(input_settle_seconds=0.12),
            state_provider=FakeProvider([_state()]),
            command_handler=RecordingHandler(),
        )
    )
    ws = FakeWs()
    host.clients.add(ws)

    await host._on_message(ws, protocol.Command(protocol.Verb.PLAY, ["0"], id="1").to_message())
    assert slept == [0.12]


@pytest.mark.asyncio
async def test_state_command_refreshes_without_settle(monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(server.asyncio, "sleep", fake_sleep)
    host = server.HostServer(
        server.GameSession(
            _cfg(input_settle_seconds=0.12),
            state_provider=FakeProvider([_state()]),
            command_handler=RecordingHandler(),
        )
    )
    ws = FakeWs()
    host.clients.add(ws)

    await host._on_message(ws, protocol.Command(protocol.Verb.STATE, id="1").to_message())
    assert slept == []
    assert [msg["type"] for msg in ws.sent] == ["ack", "state"]


@pytest.mark.asyncio
async def test_command_execution_is_serialized():
    class SlowHandler:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        def execute(self, command, state_hint=None):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            self.active -= 1
            return True, None

    handler = SlowHandler()
    host = server.HostServer(
        server.GameSession(_cfg(), state_provider=FakeProvider([_state()]), command_handler=handler)
    )
    ws1 = FakeWs()
    ws2 = FakeWs()
    host.clients.update({ws1, ws2})

    await asyncio.gather(
        host._on_message(ws1, protocol.Command(protocol.Verb.STATE, id="1").to_message()),
        host._on_message(ws2, protocol.Command(protocol.Verb.STATE, id="2").to_message()),
    )

    assert handler.max_active == 1


@pytest.mark.asyncio
async def test_chain_pushes_one_final_state_and_uses_predicted_state_hints():
    before = _combat_state()
    after = _state(ScreenType.MAP)
    handler = RecordingHandler()
    provider = FakeProvider([after])
    session = server.GameSession(_cfg(), state_provider=provider, command_handler=handler)
    session.last_state = before
    host = server.HostServer(session)
    ws = FakeWs()
    host.clients.add(ws)

    commands = [
        protocol.Command(protocol.Verb.PLAY, ["0", "0"]),
        protocol.Command(protocol.Verb.PLAY, ["0", "0"]),
    ]
    await host._on_message(ws, protocol.chain_message("chain-1", commands))

    assert provider.reads == 1
    assert [msg["type"] for msg in ws.sent] == ["ack", "state"]
    assert ws.sent[0]["id"] == "chain-1"
    assert ws.sent[0]["ok"] is True
    assert len(ws.sent[0]["results"]) == 2
    assert handler.calls[0][1] is before
    assert handler.calls[1][1].combat_state.hand[0].name == "Strike"
    assert len(handler.calls[1][1].combat_state.hand) == 1
    assert handler.calls[0][2]["verify_state_change"] is False
    assert handler.calls[0][2]["note_action"] is False


@pytest.mark.asyncio
async def test_chain_stops_on_first_failure_and_reports_partial_results():
    class FailingSecondHandler(RecordingHandler):
        def execute(self, command, state_hint=None, *, verify_state_change=True, note_action=True):
            self.calls.append((command, state_hint, {
                "verify_state_change": verify_state_change,
                "note_action": note_action,
            }))
            if len(self.calls) == 2:
                return False, "nope"
            return True, None

    before = _combat_state(hand=[
        Card(name="Strike", cost=1, index=0),
        Card(name="Strike", cost=1, index=1),
        Card(name="Strike", cost=1, index=2),
    ])
    handler = FailingSecondHandler()
    session = server.GameSession(_cfg(), state_provider=FakeProvider([_state()]), command_handler=handler)
    session.last_state = before
    host = server.HostServer(session)
    ws = FakeWs()
    host.clients.add(ws)

    commands = [
        protocol.Command(protocol.Verb.PLAY, ["0", "0"]),
        protocol.Command(protocol.Verb.PLAY, ["0", "0"]),
        protocol.Command(protocol.Verb.END),
    ]
    await host._on_message(ws, protocol.chain_message("chain-2", commands))

    assert len(handler.calls) == 2
    assert ws.sent[0]["ok"] is False
    assert ws.sent[0]["error"] == "nope"
    assert [r["ok"] for r in ws.sent[0]["results"]] == [True, False]


@pytest.mark.asyncio
async def test_chain_rejects_unpredictable_non_final_command_before_input():
    before = _combat_state(hand=[
        Card(name="Unknown Rare", cost=1, index=0),
        Card(name="Strike", cost=1, index=1),
    ])
    handler = RecordingHandler()
    provider = FakeProvider([before])
    session = server.GameSession(_cfg(), state_provider=provider, command_handler=handler)
    session.last_state = before
    host = server.HostServer(session)
    ws = FakeWs()
    host.clients.add(ws)

    commands = [
        protocol.Command(protocol.Verb.PLAY, ["0", "0"]),
        protocol.Command(protocol.Verb.PLAY, ["0", "0"]),
    ]
    await host._on_message(ws, protocol.chain_message("chain-3", commands))

    assert handler.calls == []
    assert ws.sent[0]["ok"] is False
    assert "unpredictable" in ws.sent[0]["error"]
    assert [msg["type"] for msg in ws.sent] == ["ack", "state"]
