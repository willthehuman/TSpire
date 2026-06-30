from tspire.common.schema import (
    CombatState,
    GameState,
    Monster,
    PlayerCombat,
    ScreenType,
)
from tspire.host.reconcile import reconcile

from tests.fakes import FakeArbiter


def _state(*, hp=80, max_hp=80, block=0, energy=3, monsters=None):
    return GameState(
        screen_type=ScreenType.COMBAT,
        in_combat=True,
        current_hp=hp,
        max_hp=max_hp,
        combat_state=CombatState(
            player=PlayerCombat(current_hp=hp, max_hp=max_hp, block=block, energy=energy),
            monsters=monsters or [],
        ),
    )


def test_no_prediction_passes_vision_through():
    vision = _state(hp=42)
    assert reconcile(vision, None, None) is vision
    assert vision.current_hp == 42


def test_vision_within_tolerance_is_kept():
    vision = _state(hp=79)
    predicted = _state(hp=78)
    before = _state(hp=80)
    out = reconcile(vision, predicted, before)
    assert out.current_hp == 79  # close enough -> trust fresh read


def test_digit_drop_hp_is_overridden_by_prediction():
    # Parser dropped a digit: read 8 instead of 78. Implausible drop -> use prediction.
    vision = _state(hp=8)
    predicted = _state(hp=78)
    before = _state(hp=80)
    out = reconcile(vision, predicted, before)
    assert out.current_hp == 78
    assert out.combat_state.player.current_hp == 78


def test_hp_increase_is_rejected():
    vision = _state(hp=95)  # can't rise above the 80 we had
    predicted = _state(hp=78)
    before = _state(hp=80)
    assert reconcile(vision, predicted, before).current_hp == 78


def test_conflict_uses_arbiter_to_pick_nearest_candidate():
    vision = _state(hp=8)
    predicted = _state(hp=78)
    before = _state(hp=80)
    # Arbiter re-reads 9 -> vision (8) is closer than prediction (78).
    arbiter = FakeArbiter(player_hp=(9, 80))
    assert reconcile(vision, predicted, before, arbiter).current_hp == 8


def test_arbiter_failure_falls_back_to_prediction():
    vision = _state(hp=8)
    predicted = _state(hp=78)
    before = _state(hp=80)
    arbiter = FakeArbiter(player_hp=None)
    assert reconcile(vision, predicted, before, arbiter).current_hp == 78


def test_monster_hp_increase_rejected_rule_only():
    vision = _state(monsters=[Monster(current_hp=30, max_hp=40, index=0)])
    predicted = _state(monsters=[Monster(current_hp=14, max_hp=40, index=0)])
    before = _state(monsters=[Monster(current_hp=20, max_hp=40, index=0)])
    out = reconcile(vision, predicted, before)
    assert out.combat_state.monsters[0].current_hp == 14


def test_monster_hp_plausible_decrease_is_kept():
    vision = _state(monsters=[Monster(current_hp=15, max_hp=40, index=0)])
    predicted = _state(monsters=[Monster(current_hp=14, max_hp=40, index=0)])
    before = _state(monsters=[Monster(current_hp=20, max_hp=40, index=0)])
    out = reconcile(vision, predicted, before)
    assert out.combat_state.monsters[0].current_hp == 15


def test_monster_reconcile_matches_by_identity_not_vision_index():
    before = _state(
        monsters=[
            Monster(name="A", current_hp=20, max_hp=20, index=0),
            Monster(name="B", current_hp=30, max_hp=30, index=1),
        ]
    )
    predicted = _state(
        monsters=[
            Monster(name="A", current_hp=14, max_hp=20, index=0),
            Monster(name="B", current_hp=30, max_hp=30, index=1),
        ]
    )
    # Vision returned display order reversed and dropped a digit on A. B must not inherit
    # A's predicted HP just because it arrived at index 0.
    vision = _state(
        monsters=[
            Monster(name="B", current_hp=30, max_hp=30, index=0),
            Monster(name="A", current_hp=4, max_hp=20, index=1),
        ]
    )

    out = reconcile(vision, predicted, before)

    assert [(m.name, m.current_hp) for m in out.combat_state.monsters] == [("B", 30), ("A", 14)]


def test_reconcile_carries_predicted_live_monster_missing_from_vision():
    before = _state(
        monsters=[
            Monster(name="A", current_hp=20, max_hp=20, index=0),
            Monster(name="B", current_hp=30, max_hp=30, index=1),
        ]
    )
    predicted = _state(
        monsters=[
            Monster(name="A", current_hp=14, max_hp=20, index=0),
            Monster(name="B", current_hp=30, max_hp=30, index=1),
        ]
    )
    vision = _state(monsters=[Monster(name="A", current_hp=14, max_hp=20, index=0)])

    out = reconcile(vision, predicted, before)

    assert [(m.index, m.name, m.current_hp) for m in out.combat_state.monsters] == [
        (0, "A", 14),
        (1, "B", 30),
    ]


def test_implausible_energy_falls_back_to_prediction():
    vision = _state(energy=88)
    predicted = _state(energy=2)
    before = _state(energy=3)
    assert reconcile(vision, predicted, before).combat_state.player.energy == 2
