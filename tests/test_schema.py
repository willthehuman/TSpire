"""Schema + protocol round-trip tests (no host deps required)."""

from tspire.common import protocol
from tspire.common.schema import (
    Card,
    CombatState,
    GameState,
    Intent,
    Monster,
    PlayerCombat,
    Power,
    ScreenType,
)


def _sample_state() -> GameState:
    return GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        floor=3,
        act=1,
        current_hp=68,
        max_hp=80,
        gold=99,
        combat_state=CombatState(
            player=PlayerCombat(
                current_hp=68,
                max_hp=80,
                block=5,
                energy=3,
                powers=[Power(power_id="Strength", name="Strength", amount=2)],
            ),
            monsters=[
                Monster(
                    name="Jaw Worm",
                    monster_id="JawWorm",
                    current_hp=40,
                    max_hp=44,
                    block=0,
                    intent=Intent.ATTACK,
                    intent_damage=11,
                    intent_hits=1,
                    index=0,
                )
            ],
            hand=[
                Card(name="Strike", cost=1, type="ATTACK", has_target=True, is_playable=True, index=0),
                Card(name="Defend", cost=1, type="SKILL", is_playable=True, index=1),
            ],
            draw_pile_count=5,
            discard_pile_count=0,
            turn=2,
        ),
        available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
    )


def test_state_round_trip_preserves_everything():
    state = _sample_state()
    restored = GameState.from_dict(state.to_dict())
    assert restored == state


def test_enums_serialize_as_strings():
    d = _sample_state().to_dict()
    assert d["screen_type"] == "COMBAT"
    assert d["combat_state"]["monsters"][0]["intent"] == "ATTACK"


def test_from_dict_tolerates_unknown_enum_value():
    d = _sample_state().to_dict()
    d["combat_state"]["monsters"][0]["intent"] = "SOMETHING_NEW"
    restored = GameState.from_dict(d)
    assert restored.combat_state.monsters[0].intent == Intent.UNKNOWN


def test_from_dict_ignores_extra_keys():
    d = _sample_state().to_dict()
    d["totally_unexpected"] = 123
    restored = GameState.from_dict(d)
    assert restored.floor == 3


def test_command_message_round_trip():
    cmd = protocol.Command(verb=protocol.Verb.PLAY, args=["0", "1"], id="42")
    data = protocol.parse_message(cmd.to_message())
    assert data["type"] == "command"
    restored = protocol.command_from_message(data)
    assert restored == cmd


def test_state_message_is_parseable():
    msg = protocol.state_message(_sample_state())
    data = protocol.parse_message(msg)
    assert data["type"] == "state"
    GameState.from_dict(data["state"])  # should not raise


def test_intent_is_attack_helper():
    assert Intent.ATTACK_DEBUFF.is_attack
    assert not Intent.BUFF.is_attack
