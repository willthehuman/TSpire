"""OllamaVisionParser mapping + assembly tests (model calls stubbed, no network)."""

from tspire.common.schema import Intent
from tspire.host.vision.llm import (
    _ENERGY_PROMPT,
    _loads_json_object,
    _pair_values,
    OllamaVisionParser,
)
from tspire.host.vision.regions import RegionMap


def _parser() -> OllamaVisionParser:
    return OllamaVisionParser(model="m", url="http://x", regions=RegionMap())


def test_to_monster_maps_intent_and_blanks_placeholder_name():
    m = OllamaVisionParser._to_monster(
        {"name": "None", "current_hp": 11, "max_hp": 17, "intent": "Attack", "intent_value": 7},
        index=2,
    )
    assert m.index == 2 and m.current_hp == 11 and m.max_hp == 17
    assert m.intent == Intent.ATTACK and m.intent_damage == 7
    assert m.name == ""  # "None" is treated as no name


def test_to_monster_unknown_intent_falls_back():
    m = OllamaVisionParser._to_monster({"current_hp": 5, "max_hp": 5, "intent": "wat"}, 0)
    assert m.intent == Intent.UNKNOWN


def test_to_monster_accepts_hp_string_alias():
    m = OllamaVisionParser._to_monster({"hp": "17/23", "intent": "attack", "intent_value": "7"}, 0)
    assert (m.current_hp, m.max_hp, m.intent_damage) == (17, 23, 7)


def test_to_card_reads_name_and_cost():
    c = OllamaVisionParser._to_card({"name": "Bash", "cost": 2}, index=1)
    assert (c.index, c.name, c.cost, c.is_playable) == (1, "Bash", 2, True)


def test_loads_json_object_accepts_markdown_fence():
    assert _loads_json_object("```json\n{\"enemies\": []}\n```") == {"enemies": []}


def test_loads_json_object_accepts_key_value_response():
    assert _loads_json_object("current: 80, max: 80") == {"current": 80, "max": 80}


def test_pair_values_accepts_hp_string():
    assert _pair_values({"hp": "80/80"}) == (80, 80)


def test_parse_combat_assembles_state_without_network(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {
        "gold": 99,
        "floor": 1,
        "deck_count": 10,
        "draw_pile_count": 5,
        "discard_pile_count": 0,
        "player_powers": [{"name": "Vulnerable", "amount": 2}],
        "monsters": [
            {
                "name": "Jaw Worm",
                "current_hp": 40,
                "max_hp": 44,
                "intent": "attack",
                "intent_value": 11,
                "powers": [{"name": "Strength", "amount": 3}],
            },
            {"name": "None", "current_hp": 10, "max_hp": 10, "intent": "defend"},
        ],
        "hand": [{"name": "Strike", "cost": 1}, {"name": "Defend", "cost": 1}],
    })
    monkeypatch.setattr(parser, "_pair",
                        lambda frame, rect, prompt: (3, 3) if prompt is _ENERGY_PROMPT else (68, 80))
    monkeypatch.setattr(parser, "_block", lambda frame: 5)

    result = parser.parse_combat(object(), read_block=True)
    p = result.combat.player
    assert (p.energy, p.current_hp, p.max_hp, p.block) == (3, 68, 80, 5)
    assert [m.intent for m in result.combat.monsters] == [Intent.ATTACK, Intent.DEFEND]
    assert result.combat.monsters[0].intent_damage == 11
    assert result.combat.monsters[0].powers[0].name == "Strength"
    assert [c.name for c in result.combat.hand] == ["Strike", "Defend"]
    assert [po.name for po in p.powers] == ["Vulnerable"]
    assert (result.combat.draw_pile_count, result.combat.discard_pile_count) == (5, 0)
    assert (result.gold, result.floor, result.deck_count) == (99, 1, 10)
    assert result.confidence == 1.0


def test_parse_combat_preserves_zero_energy(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {"monsters": [], "hand": []})
    monkeypatch.setattr(parser, "_pair",
                        lambda frame, rect, prompt: (0, 3) if prompt is _ENERGY_PROMPT else (80, 80))

    result = parser.parse_combat(object())

    assert result.combat.player.energy == 0


def test_parse_combat_uses_scene_stats_when_fixed_crops_fail(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {
        "gold": 99,
        "floor": 1,
        "deck_count": 10,
        "current_hp": 80,
        "max_hp": 80,
        "energy": 3,
        "monsters": [],
        "hand": [],
    })
    monkeypatch.setattr(parser, "_pair", lambda frame, rect, prompt: (0, 0))

    result = parser.parse_combat(object())

    p = result.combat.player
    assert (p.current_hp, p.max_hp, p.energy) == (80, 80, 3)
    assert (result.gold, result.floor, result.deck_count) == (99, 1, 10)


def test_parse_combat_uses_top_hp_when_combat_hp_and_scene_fail(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {"monsters": [], "hand": []})

    def _pair(frame, rect, prompt):
        if rect == parser.regions.top_hp:
            return 80, 80
        return 0, 0

    monkeypatch.setattr(parser, "_pair", _pair)

    result = parser.parse_combat(object())

    assert (result.combat.player.current_hp, result.combat.player.max_hp) == (80, 80)


def test_parse_combat_skips_block_call_when_not_requested(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {"monsters": [], "hand": []})
    monkeypatch.setattr(parser, "_pair", lambda frame, rect, prompt: (0, 0))

    def _fail(frame):
        raise AssertionError("block call should not happen when read_block=False")

    monkeypatch.setattr(parser, "_block", _fail)
    result = parser.parse_combat(object(), read_block=False)
    assert result.combat.player.block == 0
    assert result.confidence == 0.0  # nothing read
