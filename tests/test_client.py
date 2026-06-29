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
COMBAT = [protocol.Verb.PLAY, protocol.Verb.END, protocol.Verb.POTION, protocol.Verb.STATE]


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
    assert parse_line("potion use 0 1", COMBAT).command.args == ["use", "0", "1"]
    assert parse_line("pot discard 2", COMBAT).command.args == ["discard", "2"]


def test_parse_empty_refreshes():
    r = parse_line("", COMBAT)
    assert r.command.verb == protocol.Verb.STATE and r.note


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
    for needle in ["Strike", "Defend", "Jaw Worm", "Attack", "FLOOR", "[ATK]"]:
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
    """A fake HostConnection that yields one canned state then idles."""

    def __init__(self, frame: dict) -> None:
        self.frame = frame
        self.sent: list[tuple[str, list[str]]] = []
        self._ws = True  # truthy so action_refresh sends

    async def send_command(self, verb, args=None):
        self.sent.append((verb, args or []))

    async def messages(self):
        yield self.frame
        await asyncio.sleep(0.2)  # then stay quiet so the test can end


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
