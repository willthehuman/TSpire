"""Combat parser + classifier + region-math tests (no native deps, no game)."""

import pytest

from tspire.common.schema import Intent, ScreenType
from tspire.host.classify import classify_screen
from tspire.host.vision.combat import parse_combat
from tspire.host.vision.card_names import CardNameIndex
from tspire.host.vision.regions import Rect, RegionMap

from tests.fakes import FakeCard, FakeFrame, FakeMonster, FakeVisionBackend


@pytest.fixture(autouse=True)
def _easyocr_unavailable(monkeypatch):
    monkeypatch.setattr("tspire.host.vision.combat._easyocr_on", lambda: False)


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
    assert {
        "energy",
        "player_hp",
        "top_hp",
        "deck_count",
        "monster_search",
        "hand_search",
        "relics_search",
        "potions_search",
        "player_powers_search",
        "draw_pile",
        "discard_pile",
    } <= set(names)


def test_default_regions_align_with_1920x1080_client_layout():
    regions = RegionMap()
    # energy / player_hp / deck_count are derived from the decompiled game (EnergyPanel,
    # AbstractCreature health bar, TopPanel deck icon) and tightened to the digits.
    assert regions.energy.to_pixels(1920, 1080) == (121, 851, 154, 80)
    assert regions.top_hp.to_pixels(1920, 1080) == (278, 16, 192, 65)
    assert regions.player_hp.to_pixels(1920, 1080) == (357, 743, 238, 48)
    assert regions.player_powers_search.to_pixels(1920, 1080) == (336, 772, 326, 76)
    assert regions.floor.to_pixels(1920, 1080) == (902, 11, 115, 54)
    assert regions.deck_count.to_pixels(1920, 1080) == (1728, 9, 154, 81)
    assert regions.relics_search.to_pixels(1920, 1080) == (29, 81, 826, 92)
    assert regions.potions_search.to_pixels(1920, 1080) == (566, 11, 211, 81)
    assert regions.draw_pile.to_pixels(1920, 1080) == (38, 967, 144, 108)
    assert regions.discard_pile.to_pixels(1920, 1080) == (1757, 967, 144, 108)
    assert regions.hand_search.to_pixels(1920, 1080) == (442, 821, 1046, 254)


# --- classifier ------------------------------------------------------------
def test_classify_detects_combat():
    backend = _backend(monsters=[FakeMonster(left=600, hp=40, hp_max=44)])
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_combat_independent_of_monster_detection():
    # The cheap classifier keys on energy + End-Turn only (the LLM finds monsters), so it
    # detects combat even when red-bar detection sees nothing.
    backend = _backend(monsters=[])
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_detects_combat_without_energy_when_dynamic_cue_present():
    backend = _backend(monsters=[FakeMonster(left=600, hp=40, hp_max=44)], energy_filled=False)
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_detects_combat_without_end_turn_when_dynamic_cue_present():
    backend = _backend(monsters=[FakeMonster(left=600, hp=40, hp_max=44)], end_turn_filled=False)
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_detects_combat_from_piles_when_end_turn_misses():
    backend = _backend(
        monsters=[],
        end_turn_filled=False,
        draw_pile_filled=True,
        discard_pile_filled=True,
    )
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_detects_combat_from_cards_and_monsters_if_fixed_regions_miss():
    backend = _backend(
        energy_filled=False,
        end_turn_filled=False,
        monsters=[FakeMonster(left=600, hp=40, hp_max=44)],
        cards=[FakeCard(left=300, cost=1, name="Strike")],
    )
    assert classify_screen(FakeFrame(), backend.regions, backend) == ScreenType.COMBAT


def test_classify_unknown_without_fixed_or_enough_dynamic_signals():
    backend = _backend(
        energy_filled=False,
        end_turn_filled=False,
        monsters=[FakeMonster(left=600, hp=40, hp_max=44)],
        cards=[],
    )
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
    assert (result.gold, result.floor, result.deck_count) == (99, 1, 10)


def test_player_hp_falls_back_to_top_bar_region():
    backend = _backend(
        player_hp=(0, 0),
        top_hp=(80, 80),
        energy=(3, 3),
        monsters=[FakeMonster(left=600, hp=40, hp_max=44)],
    )

    result = parse_combat(FakeFrame(), backend.regions, backend)

    assert (result.combat.player.current_hp, result.combat.player.max_hp) == (80, 80)


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


def test_easyocr_flag_disables_availability_probe(monkeypatch):
    backend = _backend(
        monsters=[FakeMonster(left=600, hp=40, hp_max=44)],
        cards=[FakeCard(left=300, cost=1, name="Strike")],
    )

    def fail_if_called():
        raise AssertionError("EasyOCR availability should not be checked when disabled")

    monkeypatch.setattr("tspire.host.vision.combat._easyocr_on", fail_if_called)

    result = parse_combat(FakeFrame(), backend.regions, backend, use_easyocr=False)

    assert result.combat.player.energy == 3
    assert result.deck_count == 10


def test_easyocr_reads_stylized_fields_and_hand(monkeypatch):
    class SliceableFrame(FakeFrame):
        size = 1

        def __getitem__(self, _key):
            return self

    backend = _backend(monsters=[FakeMonster(left=600, hp=40, hp_max=44)])
    reads = iter([4, 20])

    monkeypatch.setattr("tspire.host.vision.combat._easyocr_on", lambda: True)
    monkeypatch.setattr("tspire.host.vision.easyocr_reader.read_int", lambda _crop: next(reads))
    monkeypatch.setattr(
        "tspire.host.vision.easyocr_reader.read_boxes",
        lambda _crop: [
            (20.0, 10.0, "1", 0.95),
            (95.0, 10.0, "Strike", 0.95),
            (220.0, 10.0, "2", 0.95),
            (305.0, 10.0, "Bash", 0.95),
        ],
    )
    monkeypatch.setattr(
        "tspire.host.vision.card_names.default_card_index",
        lambda: CardNameIndex(["Strike", "Bash"]),
    )

    result = parse_combat(SliceableFrame(), backend.regions, backend)

    assert result.combat.player.energy == 4
    assert result.deck_count == 20
    assert [(c.index, c.cost, c.name) for c in result.combat.hand] == [
        (0, 1, "Strike"),
        (1, 2, "Bash"),
    ]


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
