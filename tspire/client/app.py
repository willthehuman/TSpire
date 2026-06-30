"""Textual dashboard client.

Connects to the host over WebSocket, renders the game state as a combat dashboard, and
turns typed commands into host commands. Reuses ``HostConnection`` and the command parser.

Run with: ``python -m tspire.client.app`` (or the ``tspire-client`` script).

A non-combat scene falls back to a compact info view; a missing textual install at runtime
is reported clearly (the host works without it).
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.events import Key
from textual.widgets import Header, Input, RichLog, Static
from textual.widget import Widget
from rich.text import Text

from tspire.client.commands import HELP_TEXT, parse_chain, parse_line
from tspire.client.connection import HostConnection
from tspire.client.views import render_state
from tspire.common import protocol
from tspire.common.schema import GameState


_SPINNER_FRAMES = ("/", "|", "\\", "-")
_DIVIDER = "-" * 160


class Dashboard(Widget):
    """Renders the current game state; re-renders whenever app.state changes."""

    def render(self):
        state: Optional[GameState] = self.app.state  # type: ignore[attr-defined]
        if state is None:
            from rich.text import Text

            return Text("Connecting to host and waiting for first state...", style="dim italic")
        return render_state(state)


class TSpireApp(App):
    """Slay the Spire remote-control dashboard."""

    CSS = """
    Screen { layout: vertical; }
    #dashboard { border: round $primary; height: 1fr; padding: 0 1; }
    #command_panel { height: 13; border: round $accent; padding: 0 1; }
    #status { height: 1; color: $text-muted; }
    #log { height: 6; border: none; }
    #input_divider { height: 1; color: $text-muted; }
    #suggestion { height: 1; color: $text-muted; }
    #cmd { height: 3; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("question_mark", "help", "Help", key_display="?", show=True),
    ]

    state: reactive[Optional[GameState]] = reactive(None)

    def __init__(self, url: str = "", connection: "HostConnection | None" = None) -> None:
        super().__init__()
        self.url = url
        self.conn = connection or HostConnection(url)
        self._reader_task: Optional[asyncio.Task] = None
        self._pending_command_id: str | None = None
        self._pending_command_label = ""
        self._pending_ack_ok = False
        self._spinner_index = 0
        self._command_history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self._setting_input = False
        self._programmatic_input_value: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Dashboard(id="dashboard")
        yield Container(
            Static(Text("ready", style="dim"), id="status"),
            RichLog(id="log", highlight=False, markup=True),
            Static(Text(_DIVIDER, style="dim"), id="input_divider"),
            Static(Text("", style="dim"), id="suggestion"),
            Input(id="cmd", placeholder="play 0 1   |   end   |   ?help"),
            id="command_panel",
        )

    # --- lifecycle ---------------------------------------------------------
    def on_mount(self) -> None:
        self._reader_task = asyncio.create_task(self._read_loop())
        self.set_interval(0.2, self._tick_status)
        self.query_one("#cmd", Input).focus()
        self._update_suggestion()

    def on_unmount(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()

    async def _read_loop(self) -> None:
        log = self.query_one("#log", RichLog)
        async for msg in self.conn.messages():
            kind = msg.get("type")
            if kind == "state":
                self.state = GameState.from_dict(msg["state"])
                self.query_one(Dashboard).refresh()
                self._update_suggestion()
                if self._pending_command_id and self._pending_ack_ok:
                    self._clear_pending()
            elif kind == "ack":
                if self._is_stale_ack(msg):
                    continue
                if msg.get("ok"):
                    self._pending_ack_ok = bool(self._pending_command_id)
                    if self._pending_command_id:
                        self._tick_status()
                else:
                    log.write(f"[red]failed:[/red] {msg.get('error')}")
                    self._clear_pending()
            elif kind == "log":
                lvl = msg.get("level", "info")
                style = "yellow" if lvl == "warning" else "red" if lvl == "error" else "dim"
                log.write(f"[{style}]{msg.get('message')}[/]")

    # --- input -------------------------------------------------------------
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""
        self._history_index = None
        self._history_draft = ""
        self._update_suggestion()
        log = self.query_one("#log", RichLog)
        if not line:
            return
        if self._pending_command_id:
            log.write("[yellow]busy:[/yellow] command already running")
            return
        available = self.state.available_commands if self.state else []
        result = parse_chain(line, available) if ";" in line else parse_line(line, available)
        if result.error:
            log.write(f"[red]error:[/red] {result.error}")
            return
        if result.note == HELP_TEXT:
            for ln in HELP_TEXT.splitlines():
                log.write(ln)
            return
        if result.note:
            log.write(f"[dim]{result.note}[/]")
        if result.command is None:
            if not result.commands:
                return
        self._remember_command(line)
        try:
            if result.commands:
                command_id = await self.conn.send_chain(result.commands)
                label = _chain_label(result.commands)
            else:
                assert result.command is not None
                command_id = await self.conn.send_command(result.command.verb, result.command.args)
                label = f"{result.command.verb} {' '.join(result.command.args)}".rstrip()
            self._set_pending(str(command_id), label)
            log.write(f"[cyan]>[/] {label}")
        except ConnectionError:
            self._clear_pending()
            log.write("[yellow]not connected yet; retrying in the background...[/]")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "cmd":
            return
        if event.value == self._programmatic_input_value:
            self._programmatic_input_value = None
        elif not self._setting_input:
            self._history_index = None
            self._history_draft = event.value
        self._update_suggestion()

    def on_key(self, event: Key) -> None:
        cmd = self.query_one("#cmd", Input)
        if not getattr(cmd, "has_focus", False) or cmd.disabled:
            return
        if event.key == "up":
            self._history_previous()
        elif event.key == "down":
            self._history_next()
        elif event.key == "tab":
            self._accept_suggestion()
        else:
            return
        event.prevent_default()
        event.stop()

    # --- keybindings -------------------------------------------------------
    def action_refresh(self) -> None:
        if self._pending_command_id:
            self.query_one("#log", RichLog).write("[yellow]busy:[/yellow] command already running")
            return
        if self.conn._ws is not None:
            asyncio.create_task(self._safe_send("state", []))

    def action_help(self) -> None:
        log = self.query_one("#log", RichLog)
        for ln in HELP_TEXT.splitlines():
            log.write(ln)

    async def _safe_send(self, verb: str, args: list[str]) -> None:
        try:
            command_id = await self.conn.send_command(verb, args)
            label = f"{verb} {' '.join(args)}".rstrip()
            self._set_pending(str(command_id), label)
        except ConnectionError:
            self.query_one("#log", RichLog).write("[yellow]not connected[/]")
            self._clear_pending()

    def _remember_command(self, line: str) -> None:
        if not line:
            return
        if not self._command_history or self._command_history[-1] != line:
            self._command_history.append(line)
        if len(self._command_history) > 100:
            del self._command_history[:-100]
        self._history_index = None
        self._history_draft = ""

    def _history_previous(self) -> None:
        if not self._command_history:
            return
        cmd = self.query_one("#cmd", Input)
        if self._history_index is None:
            self._history_draft = cmd.value
            self._history_index = len(self._command_history) - 1
        else:
            self._history_index = max(0, self._history_index - 1)
        self._set_command_input(self._command_history[self._history_index])

    def _history_next(self) -> None:
        if self._history_index is None:
            return
        if self._history_index < len(self._command_history) - 1:
            self._history_index += 1
            value = self._command_history[self._history_index]
        else:
            self._history_index = None
            value = self._history_draft
            self._history_draft = ""
        self._set_command_input(value)

    def _accept_suggestion(self) -> None:
        suggestion = self._current_suggestion()
        if suggestion:
            self._set_command_input(suggestion)

    def _set_command_input(self, value: str) -> None:
        cmd = self.query_one("#cmd", Input)
        self._setting_input = True
        self._programmatic_input_value = value
        try:
            cmd.value = value
            cmd.cursor_position = len(value)
        finally:
            self._setting_input = False
        self._update_suggestion()

    def _update_suggestion(self) -> None:
        if not self.is_mounted:
            return
        suggestion = self._current_suggestion()
        try:
            widget = self.query_one("#suggestion", Static)
        except NoMatches:
            return
        if suggestion:
            widget.update(Text.assemble(("completion ", "dim"), (suggestion, "cyan")))
        else:
            widget.update(Text(""))

    def _current_suggestion(self) -> str:
        if self._pending_command_id:
            return ""
        try:
            value = self.query_one("#cmd", Input).value
        except NoMatches:
            return ""
        needle = value.lower()
        for candidate in self._completion_candidates():
            if candidate.lower().startswith(needle) and candidate != value:
                return candidate
        return ""

    def _completion_candidates(self) -> list[str]:
        state = self.state
        available = set(state.available_commands if state else [])
        candidates: list[str] = []
        seen: set[str] = set()

        def add(value: str) -> None:
            if value not in seen:
                candidates.append(value)
                seen.add(value)

        cs = state.combat_state if state else None
        if protocol.Verb.PLAY in available:
            hand = sorted(cs.hand, key=lambda c: c.index) if cs else []
            monsters = [m.index for m in sorted(cs.monsters, key=lambda m: m.index) if not m.is_gone] if cs else []
            if hand:
                for card in hand:
                    idx = card.index
                    add(f"play {idx}")
                    add(f"p {idx}")
                    for target in monsters:
                        add(f"play {idx} {target}")
                        add(f"p {idx} {target}")
            else:
                add("play ")
                add("p ")
        if protocol.Verb.END in available:
            add("end")
            add("e")
        if protocol.Verb.POTION in available:
            potions = state.potions if state else []
            if potions:
                for potion in potions:
                    idx = potion.index if potion.index >= 0 else potions.index(potion)
                    add(f"potion use {idx}")
                    add(f"potion discard {idx}")
            else:
                add("potion use ")
        if protocol.Verb.PROCEED in available:
            add("proceed")
        if protocol.Verb.RETURN in available:
            add("back")
            add("return")
        if protocol.Verb.CHOOSE in available:
            add("choose ")
        add("state")
        add("refresh")
        add("help")
        add("?")
        add("raw ")
        return candidates

    def _set_pending(self, command_id: str, label: str) -> None:
        self._pending_command_id = command_id
        self._pending_command_label = label
        self._pending_ack_ok = False
        self.query_one("#cmd", Input).disabled = True
        self._update_suggestion()
        self._tick_status()

    def _clear_pending(self) -> None:
        self._pending_command_id = None
        self._pending_command_label = ""
        self._pending_ack_ok = False
        cmd = self.query_one("#cmd", Input)
        cmd.disabled = False
        cmd.focus()
        self._set_status(Text("ready", style="dim"))
        self._update_suggestion()

    def _tick_status(self) -> None:
        if not self._pending_command_id:
            return
        spinner = _SPINNER_FRAMES[self._spinner_index % len(_SPINNER_FRAMES)]
        self._spinner_index += 1
        if self._pending_ack_ok:
            label = "refreshing state"
        else:
            label = f"running {self._pending_command_label}"
        self._set_status(Text(f"{spinner} {label}", style="cyan"))

    def _set_status(self, text: Text) -> None:
        if not self.is_mounted:
            return
        try:
            self.query_one("#status", Static).update(text)
        except NoMatches:
            return

    def _is_stale_ack(self, msg: dict) -> bool:
        if not self._pending_command_id:
            return False
        return str(msg.get("id", "")) != self._pending_command_id


def main() -> None:
    parser = argparse.ArgumentParser(description="TSpire Textual dashboard client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    url = f"ws://{args.host}:{args.port}"
    TSpireApp(url).run()


def _chain_label(commands: list[protocol.Command]) -> str:
    return "; ".join(f"{command.verb} {' '.join(command.args)}".rstrip() for command in commands)


if __name__ == "__main__":
    main()
