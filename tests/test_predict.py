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
from tspire.host import cards
from tspire.host.predict import BASE_ENERGY, predict


def _combat(
    *,
    hp=80,
    max_hp=80,
    block=0,
    energy=3,
    player_powers=None,
    monsters=None,
    hand=None,
    turn=1,
):
    return GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        current_hp=hp,
        max_hp=max_hp,
        combat_state=CombatState(
            player=PlayerCombat(
                current_hp=hp,
                max_hp=max_hp,
                block=block,
                energy=energy,
                powers=player_powers or [],
            ),
            monsters=monsters or [],
            hand=hand or [],
            turn=turn,
        ),
    )


def _end():
    return protocol.Command(protocol.Verb.END)


def _play(card_index, target=None):
    args = [str(card_index)] + ([str(target)] if target is not None else [])
    return protocol.Command(protocol.Verb.PLAY, args)


# --- end turn -------------------------------------------------------------
def test_end_turn_enemy_attack_minus_block():
    # The user's example: 80 hp, an enemy hitting for 7, 5 block -> 78.
    before = _combat(
        hp=80,
        block=5,
        monsters=[Monster(current_hp=20, max_hp=20, intent=Intent.ATTACK, intent_damage=7, index=0)],
    )
    state = predict(before, _end())
    assert state.current_hp == 78
    assert state.combat_state.player.current_hp == 78
    assert state.combat_state.player.block == 0  # resets next turn
    assert state.combat_state.player.energy == BASE_ENERGY
    assert state.combat_state.turn == 2


def test_end_turn_block_fully_absorbs():
    before = _combat(
        hp=50,
        block=10,
        monsters=[Monster(current_hp=20, max_hp=20, intent=Intent.ATTACK, intent_damage=6, index=0)],
    )
    assert predict(before, _end()).current_hp == 50


def test_end_turn_sums_multi_hit_and_multi_monster():
    before = _combat(
        hp=80,
        block=3,
        monsters=[
            Monster(current_hp=20, max_hp=20, intent=Intent.ATTACK, intent_damage=5, intent_hits=2, index=0),
            Monster(current_hp=10, max_hp=10, intent=Intent.ATTACK, intent_damage=4, index=1),
        ],
    )
    # incoming = 5*2 + 4 = 14; minus 3 block = 11
    assert predict(before, _end()).current_hp == 69


def test_end_turn_ignores_non_attack_and_dead_enemies():
    before = _combat(
        hp=80,
        monsters=[
            Monster(current_hp=20, max_hp=20, intent=Intent.DEFEND, intent_damage=0, index=0),
            Monster(current_hp=0, max_hp=20, intent=Intent.ATTACK, intent_damage=9, is_gone=True, index=1),
        ],
    )
    assert predict(before, _end()).current_hp == 80


def test_end_turn_clamps_hp_at_zero():
    before = _combat(
        hp=4,
        monsters=[Monster(current_hp=20, max_hp=20, intent=Intent.ATTACK, intent_damage=99, index=0)],
    )
    assert predict(before, _end()).current_hp == 0


# --- play ----------------------------------------------------------------
def test_play_attack_reduces_target_hp_and_spends_energy():
    before = _combat(
        energy=3,
        monsters=[Monster(current_hp=20, max_hp=20, index=0)],
        hand=[Card(name="Strike", cost=1, index=0)],
    )
    state = predict(before, _play(0, 0))
    assert state.combat_state.monsters[0].current_hp == 14  # 20 - 6
    assert state.combat_state.player.energy == 2
    assert state.combat_state.hand == []  # card left hand
    assert state.combat_state.discard_pile_count == 1


def test_play_attack_consumes_monster_block_before_hp():
    before = _combat(
        energy=3,
        monsters=[Monster(current_hp=20, max_hp=20, block=4, index=0)],
        hand=[Card(name="Strike", cost=1, index=0)],
    )
    monster = predict(before, _play(0, 0)).combat_state.monsters[0]
    assert monster.block == 0
    assert monster.current_hp == 18


def test_play_attack_adds_strength_and_vulnerable():
    before = _combat(
        player_powers=[Power(power_id="Strength", name="Strength", amount=3)],
        monsters=[
            Monster(
                current_hp=40,
                max_hp=40,
                index=0,
                powers=[Power(power_id="Vulnerable", name="Vulnerable", amount=1)],
            )
        ],
        hand=[Card(name="Strike", cost=1, index=0)],
    )
    # (6 + 3 strength) = 9, x1.5 vulnerable = 13 -> 40 - 13 = 27
    assert predict(before, _play(0, 0)).combat_state.monsters[0].current_hp == 27


def test_play_block_card_adds_block():
    before = _combat(block=0, hand=[Card(name="Defend", cost=1, index=0)])
    assert predict(before, _play(0)).combat_state.player.block == 5


def test_play_single_target_card_targets_lone_enemy_without_index():
    before = _combat(
        monsters=[Monster(current_hp=20, max_hp=20, index=0)],
        hand=[Card(name="Strike", cost=1, index=0)],
    )
    assert predict(before, _play(0)).combat_state.monsters[0].current_hp == 14


def test_play_unknown_card_is_unpredictable():
    before = _combat(hand=[Card(name="Some Unknown Rare", cost=2, index=0)])
    assert predict(before, _play(0)) is None


def test_play_runtime_derived_card(monkeypatch):
    monkeypatch.setattr(
        cards,
        "_runtime_card_db",
        lambda: {
            cards._norm("Runtime Strike"): cards.CardData(
                "Runtime Strike",
                name="Runtime Strike",
                damage=7,
                target=True,
                damage_up=10,
                source="jar",
            )
        },
    )
    before = _combat(
        energy=3,
        monsters=[Monster(current_hp=20, max_hp=20, index=0)],
        hand=[Card(name="Runtime Strike", cost=1, index=0)],
    )

    state = predict(before, _play(0, 0))

    assert state.combat_state.monsters[0].current_hp == 13
    assert state.combat_state.player.energy == 2


def test_play_unpredictable_runtime_card_stays_unpredictable(monkeypatch):
    monkeypatch.setattr(
        cards,
        "_runtime_card_db",
        lambda: {
            cards._norm("Runtime Draw"): cards.CardData(
                "Runtime Draw",
                name="Runtime Draw",
                damage=7,
                target=True,
                predictable=False,
                source="jar",
            )
        },
    )
    before = _combat(
        monsters=[Monster(current_hp=20, max_hp=20, index=0)],
        hand=[Card(name="Runtime Draw", cost=1, index=0)],
    )

    assert predict(before, _play(0, 0)) is None


def test_play_upgraded_card_uses_upgraded_value():
    before = _combat(
        monsters=[Monster(current_hp=20, max_hp=20, index=0)],
        hand=[Card(name="Strike+", cost=1, index=0)],
    )
    assert predict(before, _play(0, 0)).combat_state.monsters[0].current_hp == 11  # 20 - 9


# --- guards ---------------------------------------------------------------
def test_predict_returns_none_off_combat():
    assert predict(GameState(screen_type=ScreenType.MAP), _end()) is None


def test_predict_returns_none_for_unmodeled_verb():
    assert predict(_combat(), protocol.Command(protocol.Verb.PROCEED)) is None


def test_predict_does_not_mutate_before():
    before = _combat(
        hp=80,
        block=5,
        monsters=[Monster(current_hp=20, max_hp=20, intent=Intent.ATTACK, intent_damage=7, index=0)],
    )
    predict(before, _end())
    assert before.current_hp == 80
    assert before.combat_state.player.block == 5
