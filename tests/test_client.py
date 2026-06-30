"""Tests for the command parser, dashboard rendering, and the Textual app wiring.

textual/rich are client deps; skip if absent.
"""
import asyncio

import pytest

textual = pytest.importorskip("textual")
from rich.console import Console  # noqa: E402

from tspire.client.commands import parse_line  # noqa: E402
from tspire.client.views import combat_panel, intent_label, render_state  # noqa: E402
from tspire.common import protocol  # noqa: E402
from tspire.common.schema import GameState, Intent  # noqa: E402

import tests.test_schema as ts  # reuse _sample_state  # noqa: E402


# --- command parser --------------------------------------------------------
COMBAT = protocol.commands_for_screen("COMBAT")
POTION_AVAILABLE = [*COMBAT, protocol.Verb.POTION]


def test_parse_play():
    r = parse_line("play 2 1", COMBAT)
    assert r.command.verb == protocol.Verb.PLAY and r.command.args == ["2", "1"]


def test_parse_play_alias_no_target():
    r = parse_line("p 0", COMBAT)
    assert r.command.args == ["0"]


def test_parse_end_alias():
    assert parse_line("e", COMBAT).command.verb == protocol.Verb.END
    assert parse_line("end", COMBAT).command.verb == protocol.Verb.END


def test_parse_potion_use_and_discard():
    assert parse_line("potion use 0 1", POTION_AVAILABLE).command.args == ["use", "0", "1"]
    assert parse_line("pot discard 2", POTION_AVAILABLE).command.args == ["discard", "2"]


def test_combat_commands_defer_potions_for_m3():
    assert protocol.Verb.POTION not in COMBAT
    assert protocol.Verb.PLAY in COMBAT
    assert protocol.Verb.END in COMBAT
    assert protocol.Verb.PROCEED in COMBAT
    assert protocol.Verb.RETURN in COMBAT


def test_parse_empty_refreshes():
    r = parse_line("", COMBAT)
    assert r.command.verb == protocol.Verb.STATE and r.note


def test_parse_raw_debug_command():
    r = parse_line("raw a right", COMBAT)
    assert r.command.verb == protocol.Verb.RAW
    assert r.command.args == ["a", "right"]


def test_parse_unknown_returns_error():
    r = parse_line("frobnicate", COMBAT)
    assert r.command is None and r.error


def test_parse_help_is_client_only():
    r = parse_line("?", COMBAT)
    assert r.command is None and r.note


def test_parse_rejects_unavailable_verb():
    # 'end' not available on a non-combat screen
    r = parse_line("end", [protocol.Verb.STATE, protocol.Verb.PROCEED])
    assert r.command is None and r.error and "available" in r.error


def test_parse_missing_index():
    r = parse_line("play", COMBAT)
    assert r.command is None and r.error and "card" in r.error


def test_parse_bad_index():
    r = parse_line("play foo", COMBAT)
    assert r.command is None and "number" in r.error


# --- dashboard rendering (pure functions) ----------------------------------
def _render(state: GameState) -> str:
    from io import StringIO

    buf = StringIO()
    Console(file=buf, width=100, force_terminal=False, color_system=None).print(combat_panel(state))
    return buf.getvalue()


def test_combat_panel_mentions_key_fields():
    out = _render(ts._sample_state())
    for needle in ["Strike", "Defend", "Jaw Worm", "Attack", "FLOOR", "Deck", "Piles", "discard 0", "[ATK]"]:
        assert needle in out, f"missing {needle!r}"


def test_intent_attack_shows_damage_and_total():
    label = intent_label(Intent.ATTACK, 6, 2).plain
    assert "Attack" in label and "6x2" in label and "(12)" in label


def test_intent_non_attack_has_no_damage():
    assert "12" not in intent_label(Intent.BUFF, 0).plain


def test_render_state_non_combat():
    from io import StringIO

    buf = StringIO()
    Console(file=buf, width=80, force_terminal=False, color_system=None).print(render_state(GameState()))
    assert "Slay the Spire" in buf.getvalue()  # panel title present


# --- Textual app wiring ----------------------------------------------------
class _FakeConn:
    """A fake HostConnection backed by a queue of incoming frames."""

    def __init__(self, *frames: dict) -> None:
        self.frames = asyncio.Queue()
        for frame in frames:
            self.frames.put_nowait(frame)
        self.sent: list[tuple[str, list[str]]] = []
        self.sent_ids: list[str] = []
        self._next_id = 1
        self._ws = True  # truthy so action_refresh sends

    async def send_command(self, verb, args=None):
        command_id = str(self._next_id)
        self._next_id += 1
        self.sent.append((verb, args or []))
        self.sent_ids.append(command_id)
        return command_id

    async def messages(self):
        while True:
            yield await self.frames.get()

    async def push(self, frame: dict) -> None:
        await self.frames.put(frame)


@pytest.mark.asyncio
async def test_app_mounts_and_renders_state():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    frame = {"type": "state", "state": state.to_dict()}
    app = TSpireApp(connection=_FakeConn(frame))
    async with app.run_test() as pilot:
        await pilot.pause()
        # state should have arrived via the reader loop
        assert app.state is not None
        assert app.state.combat_state.hand[0].name == "Strike"
        # dashboard widget renders without raising
        dash = app.query_one("#dashboard")
        assert dash.render() is not None


@pytest.mark.asyncio
async def test_app_sends_command_on_submit():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    frame = {"type": "state", "state": state.to_dict()}
    conn = _FakeConn(frame)
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        cmd = app.query_one("#cmd")
        cmd.value = "p 0 0"
        await pilot.press("enter")
        assert ("play", ["0", "0"]) in conn.sent


def _plain(widget) -> str:
    renderable = widget.render()
    return getattr(renderable, "plain", str(renderable))


@pytest.mark.asyncio
async def test_app_disables_input_until_success_ack_is_followed_by_state():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    frame = {"type": "state", "state": state.to_dict()}
    conn = _FakeConn(frame)
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        cmd = app.query_one("#cmd")
        cmd.value = "p 0 0"
        await pilot.press("enter")

        command_id = conn.sent_ids[-1]
        assert app._pending_command_id == command_id
        assert cmd.disabled
        assert "running play 0 0" in _plain(app.query_one("#status"))

        await conn.push({"type": "ack", "id": command_id, "ok": True})
        await pilot.pause()
        assert app._pending_command_id == command_id
        assert cmd.disabled
        assert "refreshing state" in _plain(app.query_one("#status"))

        await conn.push({"type": "state", "state": state.to_dict()})
        await pilot.pause()
        assert app._pending_command_id is None
        assert not cmd.disabled


@pytest.mark.asyncio
async def test_app_clears_pending_on_failed_ack():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    conn = _FakeConn({"type": "state", "state": state.to_dict()})
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        cmd = app.query_one("#cmd")
        cmd.value = "p 0 0"
        await pilot.press("enter")

        await conn.push({"type": "ack", "id": conn.sent_ids[-1], "ok": False, "error": "nope"})
        await pilot.pause()
        assert app._pending_command_id is None
        assert not cmd.disabled


@pytest.mark.asyncio
async def test_app_ignores_stale_ack_while_command_is_pending():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    conn = _FakeConn({"type": "state", "state": state.to_dict()})
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        cmd = app.query_one("#cmd")
        cmd.value = "p 0 0"
        await pilot.press("enter")

        command_id = conn.sent_ids[-1]
        await conn.push({"type": "ack", "id": "stale", "ok": False, "error": "old"})
        await pilot.pause()
        assert app._pending_command_id == command_id
        assert cmd.disabled

        await conn.push({"type": "ack", "id": command_id, "ok": False, "error": "current"})
        await pilot.pause()
        assert app._pending_command_id is None
        assert not cmd.disabled


@pytest.mark.asyncio
async def test_refresh_key_sets_pending_and_is_blocked_while_pending():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    conn = _FakeConn({"type": "state", "state": state.to_dict()})
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_refresh()
        await pilot.pause()
        assert conn.sent == [("state", [])]
        assert app._pending_command_id == conn.sent_ids[-1]

        app.action_refresh()
        await pilot.pause()
        assert conn.sent == [("state", [])]


@pytest.mark.asyncio
async def test_pending_status_spinner_uses_requested_sequence():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    conn = _FakeConn({"type": "state", "state": state.to_dict()})
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._spinner_index = 0
        app._set_pending("1", "play 0")
        assert _plain(app.query_one("#status")).startswith("/ running play 0")
        app._tick_status()
        assert _plain(app.query_one("#status")).startswith("| running play 0")
        app._tick_status()
        assert _plain(app.query_one("#status")).startswith("\\ running play 0")
        app._tick_status()
        assert _plain(app.query_one("#status")).startswith("- running play 0")
        app._pending_ack_ok = True
        app._tick_status()
        assert "refreshing state" in _plain(app.query_one("#status"))
        app._clear_pending()


@pytest.mark.asyncio
async def test_command_history_uses_up_and_down_arrows():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    conn = _FakeConn({"type": "state", "state": state.to_dict()})
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        cmd = app.query_one("#cmd")
        app._command_history = ["play 0", "end"]
        cmd.value = "draft"

        await pilot.press("up")
        assert cmd.value == "end"
        await pilot.press("up")
        assert cmd.value == "play 0"
        await pilot.press("down")
        assert cmd.value == "end"
        await pilot.press("down")
        assert cmd.value == "draft"


@pytest.mark.asyncio
async def test_tab_accepts_contextual_command_completion():
    from tspire.client.app import TSpireApp

    state = ts._sample_state()
    conn = _FakeConn({"type": "state", "state": state.to_dict()})
    app = TSpireApp(connection=conn)
    async with app.run_test() as pilot:
        await pilot.pause()
        cmd = app.query_one("#cmd")
        cmd.value = "pla"

        await pilot.press("tab")
        assert cmd.value == "play 0"

        await pilot.press("tab")
        assert cmd.value == "play 0 0"
