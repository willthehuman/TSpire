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
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static
from textual.widget import Widget

from tspire.client.commands import HELP_TEXT, parse_line
from tspire.client.connection import HostConnection
from tspire.client.views import render_state
from tspire.common.schema import GameState


class Dashboard(Widget):
    """Renders the current game state; re-renders whenever app.state changes."""

    def render(self):
        state: Optional[GameState] = self.app.state  # type: ignore[attr-defined]
        if state is None:
            from rich.text import Text

            return Text("Connecting to host and waiting for first state…", style="dim italic")
        return render_state(state)


class TSpireApp(App):
    """Slay the Spire remote-control dashboard."""

    CSS = """
    Screen { layout: vertical; }
    #dashboard { border: round $primary; height: 1fr; padding: 0 1; }
    #log { height: 8; border: round $accent; }
    #cmd { dock: bottom; }
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

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Dashboard(id="dashboard")
        yield RichLog(id="log", highlight=False, markup=True)
        yield Input(id="cmd", placeholder="play 0 1   |   end   |   ?help")

    # --- lifecycle ---------------------------------------------------------
    def on_mount(self) -> None:
        self._reader_task = asyncio.create_task(self._read_loop())
        self.query_one("#cmd", Input).focus()

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
            elif kind == "ack":
                if msg.get("ok"):
                    pass  # success is implicit (state follows)
                else:
                    log.write(f"[red]failed:[/red] {msg.get('error')}")
            elif kind == "log":
                lvl = msg.get("level", "info")
                style = "yellow" if lvl == "warning" else "red" if lvl == "error" else "dim"
                log.write(f"[{style}]{msg.get('message')}[/]")

    # --- input -------------------------------------------------------------
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        line = event.value.strip()
        event.input.value = ""
        log = self.query_one("#log", RichLog)
        if not line:
            return
        available = self.state.available_commands if self.state else []
        result = parse_line(line, available)
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
            return
        try:
            await self.conn.send_command(result.command.verb, result.command.args)
            log.write(f"[cyan]>[/] {result.command.verb} {' '.join(result.command.args)}".rstrip())
        except ConnectionError:
            log.write("[yellow]not connected yet; retrying in the background…[/]")

    # --- keybindings -------------------------------------------------------
    def action_refresh(self) -> None:
        if self.conn._ws is not None:
            asyncio.create_task(self._safe_send("state", []))

    def action_help(self) -> None:
        log = self.query_one("#log", RichLog)
        for ln in HELP_TEXT.splitlines():
            log.write(ln)

    async def _safe_send(self, verb: str, args: list[str]) -> None:
        try:
            await self.conn.send_command(verb, args)
        except ConnectionError:
            self.query_one("#log", RichLog).write("[yellow]not connected[/]")


def main() -> None:
    parser = argparse.ArgumentParser(description="TSpire Textual dashboard client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    url = f"ws://{args.host}:{args.port}"
    TSpireApp(url).run()


if __name__ == "__main__":
    main()
