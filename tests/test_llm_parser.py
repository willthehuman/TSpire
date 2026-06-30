"""OllamaVisionParser mapping + assembly tests (model calls stubbed, no network)."""

import json

from tspire.common.schema import Intent
from tspire.host.vision.llm import (
    _ENERGY_PROMPT,
    _SCENE_SCHEMA,
    _VALUE_SCHEMA,
    _loads_json_object,
    _pair_values,
    _to_intent,
    _normalize_schema_response,
    _validate_json_schema,
    OllamaVisionParser,
)
from tspire.host.vision.regions import RegionMap


def _parser() -> OllamaVisionParser:
    return OllamaVisionParser(model="m", url="http://x", regions=RegionMap())


class _FakeOcr:
    """Stand-in for the CV backend's OCR methods, recording the regions it was asked for."""

    def __init__(self, pairs, ints):
        self._pairs = pairs
        self._ints = ints
        self.pair_calls = []
        self.int_calls = []

    def ocr_int_pair(self, frame, rect):
        self.pair_calls.append(rect)
        return self._pairs.get(rect, (0, 0))

    def ocr_int(self, frame, rect, *, default=0):
        self.int_calls.append(rect)
        return self._ints.get(rect, default)


def test_parse_combat_reads_hud_numbers_via_ocr_without_model_crops(monkeypatch):
    parser = _parser()
    r = parser.regions
    monkeypatch.setattr(parser, "_scene", lambda frame: {
        "monsters": [{"name": "Jaw Worm", "current_hp": 40, "max_hp": 44, "intent": "attack"}],
        "hand": [{"name": "Strike", "cost": 1}],
    })

    def _no_model_pair(*a, **k):
        raise AssertionError("model _pair must not be called when OCR succeeds")

    def _no_model_value(*a, **k):
        raise AssertionError("model _value must not be called when OCR succeeds")

    monkeypatch.setattr(parser, "_pair", _no_model_pair)
    monkeypatch.setattr(parser, "_value", _no_model_value)

    ocr = _FakeOcr(
        pairs={r.energy: (3, 3), r.player_hp: (68, 80)},
        ints={r.gold: 99, r.floor: 4, r.deck_count: 11},
    )
    result = parser.parse_combat(object(), ocr=ocr)

    p = result.combat.player
    assert (p.energy, p.current_hp, p.max_hp) == (3, 68, 80)
    assert (result.gold, result.floor, result.deck_count) == (99, 4, 11)


def test_parse_combat_falls_back_to_model_when_ocr_blank(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {"monsters": [], "hand": []})
    monkeypatch.setattr(parser, "_pair",
                        lambda frame, rect, prompt: (3, 3) if prompt is _ENERGY_PROMPT else (80, 80))
    monkeypatch.setattr(parser, "_value", lambda frame, rect, prompt, allow_zero=True: (7, True))

    ocr = _FakeOcr(pairs={}, ints={})  # OCR reads nothing -> model crops used
    result = parser.parse_combat(object(), ocr=ocr)

    p = result.combat.player
    assert (p.energy, p.current_hp, p.max_hp) == (3, 80, 80)
    assert result.gold == 7


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


def test_to_monster_accepts_intent_damage_alias():
    m = OllamaVisionParser._to_monster({"hp": "17/23", "intent": "attack 7", "intent_damage": 7}, 0)
    assert m.intent == Intent.ATTACK
    assert m.intent_damage == 7


def test_to_intent_understands_free_text():
    assert _to_intent("attack + defend") == Intent.ATTACK_DEFEND
    assert _to_intent("block") == Intent.DEFEND


def test_to_card_reads_name_and_cost():
    c = OllamaVisionParser._to_card({"name": "Bash", "cost": 2}, index=1)
    assert (c.index, c.name, c.cost, c.is_playable) == (1, "Bash", 2, True)


def test_loads_json_object_accepts_markdown_fence():
    assert _loads_json_object("```json\n{\"enemies\": []}\n```") == {"enemies": []}


def test_loads_json_object_accepts_key_value_response():
    assert _loads_json_object("current: 80, max: 80") == {"current": 80, "max": 80}


def test_pair_values_accepts_hp_string():
    assert _pair_values({"hp": "80/80"}) == (80, 80)


def test_scene_response_normalizes_cloud_model_nested_shape():
    raw = {
        "run": {
            "gold": 10,
            "floor": 1,
            "master_deck_count": 1,
            "hp": {"current": 80, "max": 80},
            "energy": 3,
        },
        "piles": {"draw_pile": 5, "discard_pile": 0},
        "monsters": [
            {
                "name": "Jaw Worm",
                "hp": {"current": 17, "max": 17},
                "block": None,
                "intent": "attack",
                "intent_damage": "7",
            }
        ],
        "cards": [{"card": "Strike", "cost": "1"}],
    }

    data = _normalize_schema_response(raw, _SCENE_SCHEMA)

    assert _validate_json_schema(data, _SCENE_SCHEMA) == []
    assert data["gold"] == 10
    assert data["deck_count"] == 1
    assert data["draw_pile_count"] == 5
    assert data["current_hp"] == 80
    assert data["monsters"][0]["current_hp"] == 17
    assert data["monsters"][0]["max_hp"] == 17
    assert data["monsters"][0]["block"] == 0
    assert data["monsters"][0]["intent_value"] == 7
    assert data["hand"][0] == {"name": "Strike", "cost": 1}


def test_validate_json_schema_rejects_missing_required_field():
    assert _validate_json_schema({"monsters": []}, _SCENE_SCHEMA) == ["$.hand missing"]


def test_generate_retries_until_response_matches_schema(monkeypatch):
    parser = _parser()
    prompts = []

    def _generate_text(prompt, images, schema):
        prompts.append(prompt)
        if len(prompts) == 1:
            return '{"monsters": []}'
        return '{"monsters": [], "hand": []}'

    monkeypatch.setattr(parser, "_generate_text", _generate_text)

    assert parser._generate("read scene", ["img"], _SCENE_SCHEMA) == {"monsters": [], "hand": []}
    assert len(prompts) == 2
    assert "previous answer was invalid" in prompts[1]


def test_generate_text_sends_configured_model_and_schema(monkeypatch):
    parser = OllamaVisionParser(model="gemma4:31b-cloud", url="http://x", regions=RegionMap())
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "{\"value\": 7}"}).encode()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode())
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert parser._generate_text("read value", ["png"], _VALUE_SCHEMA) == '{"value": 7}'
    assert captured["url"] == "http://x/api/generate"
    assert captured["payload"]["model"] == "gemma4:31b-cloud"
    assert captured["payload"]["format"] == _VALUE_SCHEMA
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"]["temperature"] == 0


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


def test_parse_combat_prefers_focused_run_stat_crops(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {
        "gold": 0,
        "floor": 0,
        "deck_count": 0,
        "monsters": [{"name": "Jaw Worm", "current_hp": 40, "max_hp": 44, "intent": "attack"}],
        "hand": [{"name": "Strike", "cost": 1}],
    })
    monkeypatch.setattr(parser, "_pair",
                        lambda frame, rect, prompt: (3, 3) if prompt is _ENERGY_PROMPT else (80, 80))
    monkeypatch.setattr(parser, "_value", lambda frame, rect, prompt, allow_zero=True: {
        parser.regions.gold: (123, True),
        parser.regions.floor: (4, True),
        parser.regions.deck_count: (11, True),
    }[rect])

    result = parser.parse_combat(object())

    assert (result.gold, result.floor, result.deck_count) == (123, 4, 11)


def test_parse_combat_treats_zero_run_stats_as_unread(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {
        "gold": 0,
        "floor": 0,
        "deck_count": 0,
        "monsters": [{"name": "Jaw Worm", "current_hp": 40, "max_hp": 44, "intent": "attack"}],
        "hand": [{"name": "Strike", "cost": 1}],
    })
    monkeypatch.setattr(parser, "_pair",
                        lambda frame, rect, prompt: (3, 3) if prompt is _ENERGY_PROMPT else (80, 80))
    monkeypatch.setattr(parser, "_value", lambda frame, rect, prompt, allow_zero=True: (0, False))

    result = parser.parse_combat(object())

    assert result.observed["gold"] is False
    assert result.observed["floor"] is False
    assert result.observed["deck_count"] is False


def test_parse_combat_uses_details_fallback_when_scene_misses_monsters_and_hand(monkeypatch):
    parser = _parser()
    monkeypatch.setattr(parser, "_scene", lambda frame: {
        "gold": 99,
        "floor": 1,
        "deck_count": 10,
        "monsters": [],
        "hand": [],
    })
    monkeypatch.setattr(parser, "_details", lambda frame: {
        "monsters": [{"name": "Jaw Worm", "hp": "40/44", "intent": "attack 11", "damage": 11}],
        "hand": [{"name": "Strike", "cost": 1}],
    })
    monkeypatch.setattr(parser, "_pair",
                        lambda frame, rect, prompt: (3, 3) if prompt is _ENERGY_PROMPT else (80, 80))

    result = parser.parse_combat(object())

    assert result.confidence == 1.0
    assert result.combat.monsters[0].name == "Jaw Worm"
    assert result.combat.monsters[0].intent_damage == 11
    assert result.combat.hand[0].name == "Strike"


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
