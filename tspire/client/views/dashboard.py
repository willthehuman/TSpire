"""Render the combat state as Rich panels for the Textual dashboard.

Pure functions: GameState -> Rich renderable. No I/O, so they're trivially testable
(``combat_panel(sample_state)`` must not raise and should mention key fields).
"""

from __future__ import annotations

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tspire.common.schema import GameState, Intent

# Intent -> (icon, label, rich style). ASCII-safe to avoid mojibake in narrow terminals.
_INTENT: dict[Intent, tuple[str, str, str]] = {
    Intent.ATTACK: ("[ATK]", "Attack", "bold red"),
    Intent.ATTACK_BUFF: ("[ATK]", "Attack+Buffs", "bold red"),
    Intent.ATTACK_DEBUFF: ("[ATK]", "Attack+Debuffs", "bold red"),
    Intent.ATTACK_DEFEND: ("[ATK]", "Attack+Block", "bold red"),
    Intent.DEFEND: ("[BLK]", "Defend", "bold cyan"),
    Intent.DEFEND_BUFF: ("[BLK]", "Defend+Buffs", "bold cyan"),
    Intent.DEFEND_DEBUFF: ("[BLK]", "Defend+Debuffs", "bold cyan"),
    Intent.BUFF: ("[BUF]", "Buffing", "bold green"),
    Intent.DEBUFF: ("[DBF]", "Debuffing you", "bold magenta"),
    Intent.STRONG_DEBUFF: ("[DBF]", "Strong debuff", "bold magenta"),
    Intent.SLEEP: ("[zzz]", "Sleeping", "dim"),
    Intent.STUN: ("[STN]", "Stunned", "yellow"),
    Intent.ESCAPE: ("[ESC]", "Escaping", "yellow"),
    Intent.MAGIC: ("[MAG]", "Unknown", "yellow"),
    Intent.NONE: ("[---]", "Waiting", "dim"),
    Intent.UNKNOWN: ("[???]", "Unknown", "yellow"),
    Intent.DEBUG: ("[???]", "Unknown", "yellow"),
}


def intent_label(intent: Intent, damage: int, hits: int = 1) -> Text:
    icon, label, style = _INTENT.get(intent, _INTENT[Intent.UNKNOWN])
    text = Text()
    text.append(f"{icon} ", style=style)
    text.append(label)
    if damage:
        total = damage * max(hits, 1)
        text.append(f"  {damage}" + (f"x{hits}" if hits > 1 else "") + f" ({total})",
                    style="bold red")
    return text


def _hp_text(current: int, maximum: int) -> Text:
    pct = current / maximum if maximum else 0.0
    style = "bold red" if pct < 0.3 else "bold yellow" if pct < 0.6 else "bold green"
    return Text(f"{current}/{maximum}", style=style)


def _top_bar(state: GameState) -> Panel:
    cs = state.combat_state
    player_hp = _hp_text(state.current_hp, state.max_hp)
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("HP", player_hp)
    t.add_row("Gold", Text(str(state.gold), style="yellow"))
    if cs is not None:
        energy = cs.player.energy
        t.add_row("Energy", Text(str(energy), style="bold cyan"))
        if cs.player.block:
            t.add_row("Block", Text(str(cs.player.block), style="bold blue"))
        if cs.draw_pile_count or cs.discard_pile_count:
            t.add_row("Piles", Text(f"draw {cs.draw_pile_count}  discard {cs.discard_pile_count}", style="dim"))
    meta = Text.assemble(
        ("FLOOR ", "dim"), (str(state.floor) or "?", "bold"),
        ("   ACT ", "dim"), (str(state.act) or "?", "bold"),
        ("   ", ""),
        (f"[{state.screen_type.value}]", "bold yellow"),
    )
    if state.parse_confidence and state.parse_confidence < 0.9:
        meta.append(f"   conf {state.parse_confidence:.0%}", style="dim red")
    return Panel(Group(t, Text(""), meta), title="Slay the Spire", border_style="blue")


def _enemies(state: GameState) -> Panel:
    cs = state.combat_state
    t = Table.grid(padding=(0, 1))
    t.add_column(width=2, style="dim")   # index
    t.add_column()                        # name
    t.add_column()                        # hp
    t.add_column()                        # block
    t.add_column()                        # intent
    if cs is None or not cs.monsters:
        t.add_row("", Text("(no enemies — not in combat)", style="dim"))
    else:
        for m in cs.monsters:
            name = m.name or f"enemy"
            t.add_row(
                str(m.index),
                Text(name, style="bold"),
                _hp_text(m.current_hp, m.max_hp),
                Text(f"blk {m.block}", style="blue") if m.block else Text(""),
                intent_label(m.intent, m.intent_damage, m.intent_hits),
            )
    return Panel(t, title="Enemies", border_style="red")


def _player(state: GameState) -> Panel:
    cs = state.combat_state
    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim")
    t.add_column()
    if cs is None:
        t.add_row("", Text("(no combat data)", style="dim"))
    else:
        p = cs.player
        t.add_row("HP", _hp_text(p.current_hp or state.current_hp, p.max_hp or state.max_hp))
        t.add_row("Block", Text(str(p.block), style="bold blue") if p.block else Text("0", style="dim"))
        t.add_row("Energy", Text(str(p.energy), style="bold cyan"))
        t.add_row("Turn", Text(str(cs.turn), style="dim"))
        if p.powers:
            t.add_row("Powers", Text(", ".join(_power_str(po) for po in p.powers), style="magenta"))
    return Panel(t, title="You", border_style="green")


def _power_str(power) -> str:
    return f"{power.name or power.power_id}{(' ' + str(power.amount)) if power.amount else ''}".strip()


def _hand(state: GameState) -> Panel:
    cs = state.combat_state
    t = Table.grid(padding=(0, 1))
    t.add_column(width=3, style="dim bold")
    t.add_column(width=5, justify="right")
    t.add_column()
    t.add_column()
    if cs is None or not cs.hand:
        t.add_row("", "", Text("(empty hand)", style="dim"), "")
    else:
        for c in cs.hand:
            cost = Text(f"[{c.cost}]", style="bold yellow") if c.cost >= 0 else Text("[?]", style="dim")
            target = Text(" -> target", style="cyan") if c.has_target else Text("")
            name = Text(c.name or "card", style="bold")
            if c.type == "ATTACK":
                name.stylize("red")
            elif c.type == "SKILL":
                name.stylize("green")
            elif c.type == "POWER":
                name.stylize("magenta")
            t.add_row(str(c.index), cost, name, target)
    return Panel(t, title=f"Hand ({len(cs.hand) if cs else 0})", border_style="cyan")


def _potions(state: GameState) -> Panel | None:
    if not state.potions:
        return None
    t = Table.grid(padding=(0, 1))
    t.add_column(width=2, style="dim")
    t.add_column()
    for i, p in enumerate(state.potions):
        t.add_row(str(i), Text(p.name or "potion", style="magenta"))
    return Panel(t, title="Potions", border_style="magenta")


def combat_panel(state: GameState):
    """Compose the full combat dashboard. Returns a Rich renderable (never raises)."""
    if state.screen_message:
        banner = Text(state.screen_message, style="yellow")
    else:
        banner = Text("")

    enemies = _enemies(state)
    player = _player(state)
    # side-by-side via a borderless grid
    side = Table.grid(expand=True)
    side.add_column(ratio=3)
    side.add_column(ratio=2)
    side.add_row(enemies, player)

    hand = _hand(state)
    potions = _potions(state)

    lower = [hand]
    if potions is not None:
        lower.insert(0, potions)

    return Group(
        _top_bar(state),
        Text(""),
        side,
        Text(""),
        *lower,
        Text(""),
        banner,
    )


def render_state(state: GameState):
    """Top-level renderer: full dashboard in combat, a compact view otherwise."""
    if state.combat_state is not None or state.in_combat:
        return combat_panel(state)
    cmds = ", ".join(state.available_commands) if state.available_commands else "(none)"
    info = Table.grid(padding=(0, 1))
    info.add_column(style="dim")
    info.add_column()
    info.add_row("Screen", Text(state.screen_type.value, style="bold yellow"))
    info.add_row("HP", _hp_text(state.current_hp, state.max_hp))
    info.add_row("Gold", Text(str(state.gold), style="yellow"))
    info.add_row("Commands", Text(cmds, style="cyan"))
    body = [Panel(info, title="Slay the Spire", border_style="blue")]
    if state.screen_message:
        body.append(Text(state.screen_message, style="yellow"))
    body.append(Text("Type a command below. '?' for help.", style="dim"))
    return Group(*body)
