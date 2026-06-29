"""Terminal client entry point.

M0: a minimal line-based client that prints incoming state and forwards typed commands,
used to validate the host<->client loop. M2 replaces the rendering with a Textual
dashboard (see tspire.client.views) while reusing HostConnection and the command parser.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from tspire.client.connection import HostConnection
from tspire.common.schema import GameState


def _print_state(state: GameState) -> None:
    cs = state.combat_state
    print(f"\n[{state.screen_type.value}] HP {state.current_hp}/{state.max_hp}  gold {state.gold}")
    if state.screen_message:
        print(f"  ! {state.screen_message}")
    if cs is not None:
        p = cs.player
        print(f"  you: block {p.block}  energy {p.energy}")
        for m in cs.monsters:
            intent = m.intent.value
            dmg = f" {m.intent_damage}x{m.intent_hits}" if m.intent_damage else ""
            print(f"  enemy {m.index}: {m.name} {m.current_hp}/{m.max_hp} blk {m.block} [{intent}{dmg}]")
        for c in cs.hand:
            tgt = " (needs target)" if c.has_target else ""
            print(f"  card {c.index}: ({c.cost}) {c.name}{tgt}")
    if state.available_commands:
        print(f"  commands: {', '.join(state.available_commands)}")


async def _reader(conn: HostConnection) -> None:
    async for msg in conn.messages():
        kind = msg.get("type")
        if kind == "state":
            _print_state(GameState.from_dict(msg["state"]))
        elif kind == "ack":
            if not msg.get("ok"):
                print(f"  <ack {msg.get('id')}> ERROR: {msg.get('error')}")
        elif kind == "log":
            print(f"  <{msg.get('level')}> {msg.get('message')}")
        print("> ", end="", flush=True)


async def _writer(conn: HostConnection) -> None:
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            return
        parts = line.split()
        if not parts:
            continue
        verb, args = parts[0], parts[1:]
        try:
            await conn.send_command(verb, args)
        except ConnectionError:
            print("  not connected yet; try again in a moment")


async def _run(url: str) -> None:
    conn = HostConnection(url)
    print(f"Connecting to {url} ... type commands like 'play 0 1', 'end', 'state'. Ctrl+C to quit.")
    await asyncio.gather(_reader(conn), _writer(conn))


def main() -> None:
    parser = argparse.ArgumentParser(description="TSpire terminal client (M0 line mode)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    url = f"ws://{args.host}:{args.port}"
    try:
        asyncio.run(_run(url))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
