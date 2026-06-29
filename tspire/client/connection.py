"""Client-side WebSocket connection helper.

Wraps connect/reconnect and frame decoding so both the simple M0 client and the M2
Textual app share one implementation. Callers consume an async iterator of decoded
messages and call ``send_command`` to act.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import AsyncIterator

import websockets
from websockets.asyncio.client import ClientConnection, connect

from tspire.common import protocol

log = logging.getLogger("tspire.client")


class HostConnection:
    def __init__(self, url: str, *, reconnect: bool = True) -> None:
        self.url = url
        self.reconnect = reconnect
        self._ws: ClientConnection | None = None
        self._id_counter = itertools.count(1)

    async def send_command(self, verb: str, args: list[str] | None = None) -> str:
        """Send a command; returns the generated command id."""
        if self._ws is None:
            raise ConnectionError("not connected")
        command = protocol.Command(verb=verb, args=args or [], id=str(next(self._id_counter)))
        await self._ws.send(command.to_message())
        return command.id

    async def messages(self) -> AsyncIterator[dict]:
        """Yield decoded messages, reconnecting on drop if enabled."""
        backoff = 1.0
        while True:
            try:
                async with connect(self.url) as ws:
                    self._ws = ws
                    backoff = 1.0
                    log.info("connected to %s", self.url)
                    async for raw in ws:
                        try:
                            yield protocol.parse_message(raw)
                        except ValueError:
                            log.warning("dropping malformed frame")
            except (OSError, websockets.WebSocketException) as exc:
                self._ws = None
                if not self.reconnect:
                    raise
                log.warning("connection lost (%s); retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
            finally:
                self._ws = None
