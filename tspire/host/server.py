"""WebSocket host server.

Owns the connection to clients and two pluggable collaborators:

  * a StateProvider  - reads the screen and returns a GameState
  * a CommandHandler - executes a Command via the virtual gamepad

M0 ships trivial defaults (a stub state provider and an echo command handler) so the
host<->client loop works end to end before the parser (M1) and input executor (M3) land.
Wire the real ones into ``GameSession`` as they are built.

Run with: ``python -m tspire.host.server`` (or the ``tspire-host`` script).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Awaitable, Callable, Protocol

import websockets
from websockets.asyncio.server import ServerConnection, serve

from tspire.common import protocol
from tspire.common.schema import GameState, ScreenType
from tspire.host.config import HostConfig

log = logging.getLogger("tspire.host")


class StateProvider(Protocol):
    """Reads the current game screen and returns a GameState."""

    def read(self) -> GameState: ...


class CommandHandler(Protocol):
    """Executes a command. Returns (ok, error_message)."""

    def execute(self, command: protocol.Command) -> tuple[bool, str | None]: ...


# --------------------------------------------------------------------------- #
# M0 default collaborators (replaced by real ones in M1 / M3)
# --------------------------------------------------------------------------- #
class StubStateProvider:
    """Placeholder used until the vision parser (M1) is wired in."""

    def read(self) -> GameState:
        return GameState(
            screen_type=ScreenType.NONE,
            screen_message="vision parser not wired yet (M0 scaffold)",
            available_commands=protocol.commands_for_screen(ScreenType.NONE.value),
        )


class EchoCommandHandler:
    """Placeholder used until the gamepad executor (M3) is wired in."""

    def execute(self, command: protocol.Command) -> tuple[bool, str | None]:
        log.info("echo command: verb=%s args=%s", command.verb, command.args)
        if command.verb == protocol.Verb.STATE:
            return True, None
        return False, "input executor not wired yet (M0 scaffold)"


class GameSession:
    """Holds collaborators and tracks the last state pushed to clients."""

    def __init__(
        self,
        config: HostConfig,
        state_provider: StateProvider | None = None,
        command_handler: CommandHandler | None = None,
    ) -> None:
        self.config = config
        self.state_provider = state_provider or StubStateProvider()
        self.command_handler = command_handler or EchoCommandHandler()
        self.last_state: GameState | None = None

    def read_state(self) -> GameState:
        try:
            state = self.state_provider.read()
        except Exception:  # screen read must never kill the server loop
            log.exception("state read failed")
            state = GameState(
                screen_type=ScreenType.UNKNOWN,
                screen_message="state read failed (see host log)",
            )
        self.last_state = state
        return state


class HostServer:
    def __init__(self, session: GameSession) -> None:
        self.session = session
        self.clients: set[ServerConnection] = set()
        self._state_dirty = asyncio.Event()

    async def _send(self, ws: ServerConnection, message: str) -> None:
        try:
            await ws.send(message)
        except websockets.ConnectionClosed:
            pass

    async def broadcast(self, message: str) -> None:
        if self.clients:
            await asyncio.gather(*(self._send(ws, message) for ws in self.clients))

    async def push_state(self) -> None:
        state = await asyncio.to_thread(self.session.read_state)
        await self.broadcast(protocol.state_message(state))

    async def handle_client(self, ws: ServerConnection) -> None:
        self.clients.add(ws)
        log.info("client connected (%d total)", len(self.clients))
        try:
            # Send a fresh snapshot to the newcomer.
            state = await asyncio.to_thread(self.session.read_state)
            await self._send(ws, protocol.state_message(state))
            async for raw in ws:
                await self._on_message(ws, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(ws)
            log.info("client disconnected (%d total)", len(self.clients))

    async def _on_message(self, ws: ServerConnection, raw: str) -> None:
        try:
            data = protocol.parse_message(raw)
        except ValueError as exc:
            await self._send(ws, protocol.log_message(str(exc), level="error"))
            return
        if data.get("type") != "command":
            return
        command = protocol.command_from_message(data)
        ok, error = await asyncio.to_thread(self.session.command_handler.execute, command)
        await self._send(ws, protocol.ack_message(command.id, ok, error))
        # After any command, re-read and push the (possibly changed) state.
        await self.push_state()

    async def poll_loop(self) -> None:
        """Periodically refresh state so clients see passive changes.

        Skipped for expensive (LLM) providers, which would be hammered by a timer; those
        push on connect and after each command instead.
        """
        if getattr(self.session.state_provider, "expensive", False):
            log.info("expensive state provider: idle polling disabled (reads on demand)")
            await asyncio.Future()  # idle forever; reads happen on connect / after commands
        while True:
            await asyncio.sleep(self.session.config.poll_interval)
            if self.clients:
                await self.push_state()

    async def serve(self) -> None:
        cfg = self.session.config
        log.info("host listening on ws://%s:%d", cfg.host, cfg.port)
        async with serve(self.handle_client, cfg.host, cfg.port):
            await self.poll_loop()


def build_session(config: HostConfig) -> GameSession:
    """Wire collaborators. As milestones land, swap the defaults here.

    M1: state_provider = ScreenStateProvider(config)  (done)
    M3: command_handler = GamepadCommandHandler(config)
    """
    state_provider: StateProvider
    missing = _missing_host_deps(config.vision_mode)
    if missing:
        # Host vision extras not installed -> fall back to stub so the loop still runs
        # (useful on a dev box without the full host stack).
        log.warning("vision provider unavailable (missing: %s); using stub. "
                    "Install host extras: pip install -e \".[host]\"", ", ".join(missing))
        state_provider = StubStateProvider()
    else:
        from tspire.host.state import ScreenStateProvider

        state_provider = ScreenStateProvider(config)
        log.info("vision state provider active")
    from tspire.host.input.executor import GamepadCommandHandler

    command_handler = GamepadCommandHandler(config, state_provider)
    log.info("gamepad command handler active")
    return GameSession(config, state_provider=state_provider, command_handler=command_handler)


def _missing_host_deps(vision_mode: str) -> list[str]:
    import importlib.util

    # LLM mode talks to Ollama over HTTP (stdlib) and only uses OpenCV for crops/classify;
    # CV mode additionally needs Tesseract via pytesseract.
    required = ["mss", "cv2", "numpy"]
    if vision_mode == "cv":
        required.append("pytesseract")
    return [m for m in required if importlib.util.find_spec(m) is None]


def main() -> None:
    parser = argparse.ArgumentParser(description="TSpire host server")
    parser.add_argument("--config", help="path to tspire_host.json", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = HostConfig.load(args.config)
    if args.port is not None:
        config.port = args.port

    session = build_session(config)
    server = HostServer(session)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
