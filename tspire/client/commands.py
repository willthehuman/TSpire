"""Parse a line of human input into a protocol Command.

Friendly aliases on top of the wire verbs (see tspire.common.protocol). Indices are
0-based and refer to the lists shown in the dashboard. Unknown input returns an error the
caller can display (it is never sent to the host).
"""

from __future__ import annotations

from dataclasses import dataclass

from tspire.common import protocol

# Single-token aliases -> canonical verb. Keep these unambiguous (no overlap).
_ALIASES = {
    "p": protocol.Verb.PLAY,
    "play": protocol.Verb.PLAY,
    "e": protocol.Verb.END,
    "end": protocol.Verb.END,
    "turn": protocol.Verb.END,
    "po": protocol.Verb.POTION,
    "potion": protocol.Verb.POTION,
    "pot": protocol.Verb.POTION,
    "space": protocol.Verb.PROCEED,
    "enter": protocol.Verb.PROCEED,
    "proceed": protocol.Verb.PROCEED,
    "confirm": protocol.Verb.PROCEED,
    "b": protocol.Verb.RETURN,
    "back": protocol.Verb.RETURN,
    "return": protocol.Verb.RETURN,
    "cancel": protocol.Verb.RETURN,
    "r": protocol.Verb.STATE,
    "refresh": protocol.Verb.STATE,
    "state": protocol.Verb.STATE,
    "raw": protocol.Verb.RAW,
}


@dataclass
class ParseResult:
    command: protocol.Command | None  # None means client-only (help/empty) or an error
    error: str | None = None
    note: str | None = None  # client-only feedback (e.g. "help shown", "refreshed")
    commands: list[protocol.Command] | None = None  # command chain


def parse_line(line: str, available_commands: list[str]) -> ParseResult:
    """Parse a typed line. `available_commands` gates which verbs are valid right now."""
    parts = line.split()
    if not parts:
        return ParseResult(command=protocol.Command(verb=protocol.Verb.STATE), note="refreshed")

    head = parts[0].lower()
    if head in {"?", "help", "h"}:
        return ParseResult(command=None, note=HELP_TEXT)

    verb = _ALIASES.get(head)
    if verb is None:
        return ParseResult(command=None, error=f"unknown command {head!r}. Type ? for help.")

    if verb not in available_commands and verb not in {protocol.Verb.STATE, protocol.Verb.RAW}:
        return ParseResult(
            command=None,
            error=f"'{head}' isn't available on this screen (valid: {', '.join(available_commands)}).",
        )

    rest = parts[1:]
    try:
        command = _build(verb, rest)
    except ValueError as exc:
        return ParseResult(command=None, error=str(exc))
    return ParseResult(command=command)


def parse_chain(line: str, available_commands: list[str]) -> ParseResult:
    """Parse a semicolon-separated command chain."""
    parts = line.split(";")
    if len(parts) <= 1:
        return parse_line(line, available_commands)
    if any(not part.strip() for part in parts):
        return ParseResult(command=None, error="chain contains an empty command segment.")

    commands: list[protocol.Command] = []
    for raw in parts:
        result = parse_line(raw.strip(), available_commands)
        if result.error:
            return result
        if result.note == HELP_TEXT or result.command is None:
            return ParseResult(command=None, error="help is not allowed inside a command chain.")
        if result.command.verb in {protocol.Verb.STATE, protocol.Verb.RAW}:
            return ParseResult(
                command=None,
                error=f"'{result.command.verb}' is not allowed inside a command chain.",
            )
        commands.append(result.command)

    terminal = {protocol.Verb.END, protocol.Verb.PROCEED, protocol.Verb.RETURN}
    for i, command in enumerate(commands):
        is_last = i == len(commands) - 1
        if command.verb == protocol.Verb.PLAY:
            continue
        if command.verb in terminal and is_last:
            continue
        if command.verb in terminal:
            return ParseResult(command=None, error=f"'{command.verb}' must be the last command in a chain.")
        return ParseResult(command=None, error=f"'{command.verb}' is not supported in command chains.")

    return ParseResult(command=None, commands=commands)


def _build(verb: str, rest: list[str]) -> protocol.Command:
    if verb == protocol.Verb.PLAY:
        idx = _index(rest, 0, "card")
        target = _index(rest, 1, "target", optional=True)
        args = [str(idx)] + ([str(target)] if target is not None else [])
        return protocol.Command(verb=verb, args=args)
    if verb == protocol.Verb.POTION:
        if not rest:
            raise ValueError("potion needs an action: 'potion use <i>' or 'potion discard <i>'.")
        action = rest[0].lower()
        if action not in {"use", "discard"}:
            raise ValueError("potion action must be 'use' or 'discard'.")
        idx = _index(rest, 1, "potion")
        target = _index(rest, 2, "target", optional=True)
        args = [action, str(idx)] + ([str(target)] if target is not None else [])
        return protocol.Command(verb=verb, args=args)
    if verb in (protocol.Verb.CHOOSE,):
        idx = _index(rest, 0, "choice")
        return protocol.Command(verb=verb, args=[str(idx)])
    if verb == protocol.Verb.RAW:
        if not rest:
            raise ValueError("raw needs at least one gamepad token.")
        return protocol.Command(verb=verb, args=rest)
    # end / proceed / return / state take no args.
    return protocol.Command(verb=verb, args=[])


def _index(rest: list[str], pos: int, what: str, *, optional: bool = False) -> int:
    if pos >= len(rest):
        if optional:
            return None  # type: ignore[return-value]
        raise ValueError(f"missing {what} index.")
    tok = rest[pos]
    if not tok.lstrip("-").isdigit():
        raise ValueError(f"{what} index must be a number, got {tok!r}.")
    return int(tok)


HELP_TEXT = (
    "Commands:\n"
    "  play <i> [t]   play card i (optionally on target t)   alias: p\n"
    "  end            end your turn                          alias: e\n"
    "  proceed        confirm / advance                      alias: space\n"
    "  back           cancel / return                        alias: b\n"
    "  state          re-read the screen                     alias: r\n"
    "  raw <tokens>   debug-only gamepad tokens (host gated)\n"
    "  ?              this help\n"
    "Indices are the numbers shown beside each card / enemy / potion."
)
