"""OllamaVisionParser mapping + assembly tests (model calls stubbed, no network)."""

from tspire.common.schema import Intent
from tspire.host.vision.llm import (
    _ENERGY_PROMPT,
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


def test_to_card_reads_name_and_cost():
    c = OllamaVisionParser._to_card({"name": "Bash", "cost": 2}, index=1)
    assert (c.index, c.name, c.cost, c.is_playable) == (1, "Bash", 2, True)


def test_parse_combat_assembles_state_without_network(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {
        "monsters": [
            {"name": "Jaw Worm", "current_hp": 40, "max_hp": 44, "intent": "attack", "intent_value": 11},
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
    assert [c.name for c in result.combat.hand] == ["Strike", "Defend"]
    assert result.confidence == 1.0


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
