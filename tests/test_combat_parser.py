"""Combat parser + classifier + region-math tests (no native deps, no game)."""

from tspire.common.schema import Intent, ScreenType
from tspire.host.classify import classify_screen
from tspire.host.vision.combat import parse_combat
from tspire.host.vision.regions import Rect, RegionMap

from tests.fakes import FakeCard, FakeFrame, FakeMonster, FakeVisionBackend


def _backend(**kw) -> FakeVisionBackend:
    regions = RegionMap()
    return FakeVisionBackend(regions=regions, **kw)


# --- region math -----------------------------------------------------------
def test_rect_to_pixels_basic():
    left, top, w, h = Rect(0.5, 0.5, 0.25, 0.1).to_pixels(1920, 1080)
    assert (left, top, w, h) == (960, 540, 480, 108)


def test_rect_to_pixels_clamps_to_frame():
    left, top, w, h = Rect(0.9, 0.9, 0.5, 0.5).to_pixels(1000, 1000)
    assert left == 900 and top == 900
    assert left + w <= 1000 and top + h <= 1000


def test_region_map_lists_all_named_regions():
    names = RegionMap().all_regions()
    assert {"energy", "player_hp", "monster_search", "hand_search", "draw_pile"} <= set(names)


# --- classifier ------------------------------------------------------------
def test_classify_detects_combat():
    backend = _backend(monsters=[FakeMonster(left=600, hp=40, hp_max=44)])
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_combat_independent_of_monster_detection():
    # The cheap classifier keys on energy + End-Turn only (the LLM finds monsters), so it
    # detects combat even when red-bar detection sees nothing.
    backend = _backend(monsters=[])
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_unknown_without_energy():
    backend = _backend(monsters=[FakeMonster(left=600, hp=40, hp_max=44)], energy_filled=False)
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.UNKNOWN


def test_classify_unknown_without_end_turn():
    backend = _backend(monsters=[FakeMonster(left=600, hp=40, hp_max=44)], end_turn_filled=False)
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.UNKNOWN


# --- combat parsing --------------------------------------------------------
def test_parses_player_and_piles():
    backend = _backend(
        player_hp=(68, 80), energy=(2, 3), block=5, block_filled=True, draw=7, discard=1,
        monsters=[FakeMonster(left=600, hp=40, hp_max=44)],
    )
    result = parse_combat(FakeFrame(), backend.regions, backend)
    p = result.combat.player
    assert (p.current_hp, p.max_hp, p.block, p.energy) == (68, 80, 5, 2)
    assert result.combat.draw_pile_count == 7
    assert result.combat.discard_pile_count == 1


def test_parses_multiple_monsters_left_to_right_with_intents():
    backend = _backend(
        monsters=[
            FakeMonster(left=400, hp=10, hp_max=20, intent_id="defend", dmg=0, name="Cultist"),
            FakeMonster(left=900, hp=30, hp_max=30, intent_id="attack", dmg=12, name="JawWorm"),
        ]
    )
    monsters = parse_combat(FakeFrame(), backend.regions, backend).combat.monsters
    assert [m.index for m in monsters] == [0, 1]
    assert monsters[0].name == "Cultist" and monsters[0].intent == Intent.DEFEND
    assert monsters[1].intent == Intent.ATTACK and monsters[1].intent_damage == 12
    assert monsters[1].current_hp == 30 and monsters[1].max_hp == 30


def test_low_confidence_match_drops_monster_name_and_intent():
    backend = _backend(
        monsters=[FakeMonster(left=600, hp=40, hp_max=44, intent_score=0.1, name_score=0.1)]
    )
    m = parse_combat(FakeFrame(), backend.regions, backend).combat.monsters[0]
    assert m.name == "" and m.intent == Intent.UNKNOWN


def test_parses_hand_with_costs_and_names():
    backend = _backend(
        monsters=[FakeMonster(left=600, hp=40, hp_max=44)],
        cards=[FakeCard(left=300, cost=1, name="Strike"), FakeCard(left=600, cost=2, name="Bash")],
    )
    hand = parse_combat(FakeFrame(), backend.regions, backend).combat.hand
    assert [(c.index, c.cost, c.name) for c in hand] == [(0, 1, "Strike"), (1, 2, "Bash")]


def test_confidence_high_when_everything_reads():
    backend = _backend(
        player_hp=(70, 80), energy=(3, 3),
        monsters=[FakeMonster(left=600, hp=40, hp_max=44)],
        cards=[FakeCard(left=300, cost=1, name="Strike")],
    )
    result = parse_combat(FakeFrame(), backend.regions, backend)
    assert result.confidence == 1.0


def test_confidence_low_when_reads_fail():
    backend = _backend(
        player_hp=(0, 0), energy=(0, 0),
        monsters=[FakeMonster(left=600, hp=0, hp_max=0)],
    )
    result = parse_combat(FakeFrame(), backend.regions, backend)
    assert result.confidence < 0.5
