from tspire.common import protocol
from tspire.common.schema import (
    CombatState,
    GameState,
    Intent,
    Monster,
    PlayerCombat,
    ScreenType,
)
from tspire.host.config import HostConfig
from tspire.host.state import ScreenStateProvider, _act_for_floor
from tspire.host.vision.regions import RegionMap

from tests.fakes import FakeCard, FakeFrame, FakeMonster, FakeVisionBackend


def _provider(*, vision_mode="cv", predict_enabled=False):
    provider = ScreenStateProvider.__new__(ScreenStateProvider)
    provider.config = HostConfig(vision_mode=vision_mode, predict_enabled=predict_enabled)
    provider.regions = RegionMap()
    provider._backend = None
    provider._llm = None
    provider._last_state = None
    provider._pending = None
    return provider


def test_build_combat_state_includes_run_stats_and_act():
    provider = _provider()
    backend = FakeVisionBackend(
        regions=provider.regions,
        gold=99,
        floor=1,
        player_hp=(80, 80),
        energy=(3, 3),
        monsters=[FakeMonster(left=900, hp=17, hp_max=17, dmg=7)],
        cards=[FakeCard(left=300, cost=1, name="Strike")],
    )

    state = provider._build_combat_state(FakeFrame(), backend)

    assert state.gold == 99
    assert state.floor == 1
    assert state.act == 1
    assert state.deck_count == 10
    assert state.current_hp == 80
    assert state.max_hp == 80
    assert state.combat_state.player.energy == 3


def test_build_combat_state_preserves_previous_nonzero_run_stats_when_read_fails():
    provider = _provider()
    provider._last_state = GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        floor=18,
        act=2,
        current_hp=70,
        max_hp=80,
        gold=99,
        deck_count=10,
        combat_state=CombatState(player=PlayerCombat(current_hp=70, max_hp=80, energy=2)),
        available_commands=protocol.commands_for_screen(ScreenType.COMBAT.value),
    )
    backend = FakeVisionBackend(
        regions=provider.regions,
        gold=0,
        floor=0,
        deck_count=0,
        player_hp=(0, 0),
        energy=(0, 0),
    )

    state = provider._build_combat_state(FakeFrame(), backend)

    assert state.gold == 99
    assert state.floor == 18
    assert state.act == 2
    assert state.deck_count == 10
    assert state.current_hp == 70
    assert state.max_hp == 80
    assert state.combat_state.player.current_hp == 70
    assert state.combat_state.player.max_hp == 80


def _attacking_combat(hp, *, block=0, intent_damage=7):
    return GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        current_hp=hp,
        max_hp=80,
        combat_state=CombatState(
            player=PlayerCombat(current_hp=hp, max_hp=80, block=block, energy=3),
            monsters=[Monster(current_hp=20, max_hp=20, intent=Intent.ATTACK, intent_damage=intent_damage, index=0)],
        ),
    )


def _bad_hp_backend(provider):
    # Vision misreads HP as 8 (dropped a digit); everything else benign.
    return FakeVisionBackend(regions=provider.regions, player_hp=(8, 80), energy=(3, 3))


def test_pending_end_turn_corrects_digit_dropped_hp():
    provider = _provider(predict_enabled=True)
    provider.config.predict_arbiter = False  # rules only; no Ollama
    before = _attacking_combat(80)  # 80 hp, enemy hits for 7, no block -> predict 73
    provider._last_state = before
    provider.note_action(protocol.Command(protocol.Verb.END), before)

    state = provider._build_combat_state(FakeFrame(), _bad_hp_backend(provider))

    assert state.current_hp == 73
    assert state.combat_state.player.current_hp == 73


def test_pending_action_is_one_shot():
    provider = _provider(predict_enabled=True)
    provider.config.predict_arbiter = False
    before = _attacking_combat(80)
    provider._last_state = before
    provider.note_action(protocol.Command(protocol.Verb.END), before)

    # First read consumes the pending action and corrects.
    provider._build_combat_state(FakeFrame(), _bad_hp_backend(provider))
    # Second read has nothing pending -> raw vision (8) passes through.
    second = provider._build_combat_state(FakeFrame(), _bad_hp_backend(provider))
    assert second.current_hp == 8


def test_predict_disabled_passes_raw_vision():
    provider = _provider(predict_enabled=False)
    before = _attacking_combat(80)
    provider._last_state = before
    provider.note_action(protocol.Command(protocol.Verb.END), before)

    state = provider._build_combat_state(FakeFrame(), _bad_hp_backend(provider))
    assert state.current_hp == 8  # reconciliation off


def test_note_action_ignores_non_combat_before():
    provider = _provider(predict_enabled=True)
    provider.note_action(protocol.Command(protocol.Verb.END), GameState(screen_type=ScreenType.MAP))
    assert provider._pending is None


def test_act_for_floor_ranges():
    assert _act_for_floor(0) == 0
    assert _act_for_floor(1) == 1
    assert _act_for_floor(16) == 1
    assert _act_for_floor(17) == 2
    assert _act_for_floor(34) == 3
    assert _act_for_floor(51) == 4
