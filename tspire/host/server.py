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
import inspect
import logging
from dataclasses import dataclass
from typing import Protocol

import websockets
from websockets.asyncio.server import ServerConnection, serve

from tspire.common import protocol
from tspire.common.schema import GameState, ScreenType
from tspire.host.config import HostConfig
from tspire.host.predict import predict

log = logging.getLogger("tspire.host")


@dataclass
class _ChainStep:
    command: protocol.Command
    state_hint: GameState | None
    predicted_after: GameState | None


class StateProvider(Protocol):
    """Reads the current game screen and returns a GameState."""

    def read(self) -> GameState: ...


class CommandHandler(Protocol):
    """Executes a command. Returns (ok, error_message)."""

    def execute(
        self,
        command: protocol.Command,
        state_hint: GameState | None = None,
        *,
        verify_state_change: bool = True,
        note_action: bool = True,
    ) -> tuple[bool, str | None]: ...


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

    def execute(
        self,
        command: protocol.Command,
        state_hint: GameState | None = None,
    ) -> tuple[bool, str | None]:
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
        self._command_lock = asyncio.Lock()

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
        kind = data.get("type")
        if kind == "command":
            command = protocol.command_from_message(data)
            async with self._command_lock:
                await self._execute_one(ws, command)
            return
        if kind == "chain":
            try:
                commands = protocol.commands_from_message(data)
            except ValueError as exc:
                await self._send(ws, protocol.log_message(str(exc), level="error"))
                return
            command_id = str(data.get("id", ""))
            async with self._command_lock:
                await self._execute_chain(ws, command_id, commands)

    async def _execute_one(self, ws: ServerConnection, command: protocol.Command) -> None:
        state_hint = self.session.last_state
        ok, error = await asyncio.to_thread(
            self.session.command_handler.execute,
            command,
            state_hint,
        )
        await self._send(ws, protocol.ack_message(command.id, ok, error))
        if protocol.is_state_altering(command.verb):
            await asyncio.sleep(max(0.0, float(self.session.config.input_settle_seconds)))
        # After any command, re-read and push the authoritative state.
        await self.push_state()

    async def _execute_chain(
        self,
        ws: ServerConnection,
        command_id: str,
        commands: list[protocol.Command],
    ) -> None:
        before_state = self.session.last_state
        plan, error = _plan_chain(commands, before_state)
        results: list[dict] = []
        successful: list[protocol.Command] = []
        predicted_after_success: GameState | None = None

        if error is not None:
            await self._send(ws, protocol.ack_message(command_id, False, error, results=results))
            await self.push_state()
            return

        assert plan is not None
        for step in plan:
            ok, step_error = await asyncio.to_thread(
                _execute_command,
                self.session.command_handler,
                step.command,
                step.state_hint,
                False,
                False,
            )
            results.append(_step_result(step.command, ok, step_error))
            if not ok:
                break
            successful.append(step.command)
            if step.predicted_after is not None:
                predicted_after_success = step.predicted_after

        if successful and predicted_after_success is not None:
            _note_prediction(
                self.session.state_provider,
                successful,
                before_state,
                predicted_after_success,
            )

        ok = len(results) == len(commands) and all(result["ok"] for result in results)
        first_error = next((str(result["error"]) for result in results if result.get("error")), None)
        await self._send(ws, protocol.ack_message(command_id, ok, first_error, results=results))
        if any(protocol.is_state_altering(command.verb) for command in successful):
            await asyncio.sleep(max(0.0, float(self.session.config.input_settle_seconds)))
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
    backend = str(getattr(config, "input_backend", "") or "").lower()
    if backend == "mouse":
        from tspire.host.input.mouse import MouseCommandHandler

        command_handler = MouseCommandHandler(config, state_provider)
        log.info("mouse command handler active")
    elif backend == "keyboard":
        from tspire.host.input.keyboard import KeyboardCommandHandler

        command_handler = KeyboardCommandHandler(config, state_provider)
        log.info("keyboard (number-key) command handler active")
    else:
        from tspire.host.input.executor import GamepadCommandHandler

        command_handler = GamepadCommandHandler(config, state_provider)
        log.info("%s command handler active", backend or "gamepad")
    return GameSession(config, state_provider=state_provider, command_handler=command_handler)


def _missing_host_deps(vision_mode: str) -> list[str]:
    import importlib.util

    # LLM mode talks to Ollama over HTTP (stdlib) and only uses OpenCV for crops/classify;
    # CV mode additionally needs Tesseract via pytesseract.
    required = ["mss", "cv2", "numpy"]
    if vision_mode == "cv":
        required.append("pytesseract")
    return [m for m in required if importlib.util.find_spec(m) is None]


def _plan_chain(
    commands: list[protocol.Command],
    before_state: GameState | None,
) -> tuple[list[_ChainStep] | None, str | None]:
    if not commands:
        return None, "chain needs at least one command"
    error = _validate_chain_shape(commands)
    if error:
        return None, error

    plan: list[_ChainStep] = []
    state_hint = before_state
    for i, command in enumerate(commands):
        is_last = i == len(commands) - 1
        predicted = predict(state_hint, command) if command.verb in {protocol.Verb.PLAY, protocol.Verb.END} else None
        if not is_last and predicted is None:
            return None, f"cannot chain after unpredictable command {command.verb} {' '.join(command.args)}".rstrip()
        plan.append(_ChainStep(command=command, state_hint=state_hint, predicted_after=predicted))
        if predicted is not None:
            state_hint = predicted
    return plan, None


def _validate_chain_shape(commands: list[protocol.Command]) -> str | None:
    terminal = {protocol.Verb.END, protocol.Verb.PROCEED, protocol.Verb.RETURN}
    for i, command in enumerate(commands):
        is_last = i == len(commands) - 1
        if command.verb in {protocol.Verb.STATE, protocol.Verb.RAW}:
            return f"'{command.verb}' is not allowed inside a command chain"
        if command.verb == protocol.Verb.PLAY:
            continue
        if command.verb in terminal and is_last:
            continue
        if command.verb in terminal:
            return f"'{command.verb}' must be the last command in a chain"
        return f"'{command.verb}' is not supported in command chains"
    return None


def _execute_command(
    handler: CommandHandler,
    command: protocol.Command,
    state_hint: GameState | None,
    verify_state_change: bool,
    note_action: bool,
) -> tuple[bool, str | None]:
    execute = handler.execute
    params = inspect.signature(execute).parameters
    kwargs = {}
    if "verify_state_change" in params:
        kwargs["verify_state_change"] = verify_state_change
    if "note_action" in params:
        kwargs["note_action"] = note_action
    return execute(command, state_hint, **kwargs)


def _step_result(command: protocol.Command, ok: bool, error: str | None) -> dict:
    return {
        "verb": command.verb,
        "args": list(command.args),
        "ok": ok,
        "error": error,
    }


def _note_prediction(
    state_provider: StateProvider,
    commands: list[protocol.Command],
    before_state: GameState | None,
    predicted_state: GameState | None,
) -> None:
    note = getattr(state_provider, "note_prediction", None)
    if note is None:
        return
    try:
        note(commands, before_state, predicted_state)
    except Exception:
        log.debug("note_prediction failed", exc_info=True)


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
