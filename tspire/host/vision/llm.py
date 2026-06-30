"""Local vision-model combat parser (Ollama).

Parses the combat scene with a multimodal model instead of hand-tuned CV. Validated with
gemma4:31b-cloud: it reads the busy battlefield (all enemies + the overlapping hand fan)
far more robustly than color/contour heuristics. Two focused calls per read, because
cramming many images into one call degrades accuracy:

  1. SCENE call - the full (downscaled) frame -> monsters[] and hand[].
  2. STATS call - a few upscaled crops of fixed regions -> energy / player HP / block.

Fixed-region crops are used for the small numeric fields the model misreads at full-frame
scale (the energy orb etc.). Coordinates come from the calibrated region map, so accuracy
depends on calibration just like the CV path.

Uses only the stdlib (urllib) to talk to Ollama -> no extra runtime dependency.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request

from tspire.common.schema import Card, CombatState, Intent, Monster, PlayerCombat, Power
from tspire.host.vision.combat import ParseResult
from tspire.host.vision.regions import Rect, RegionMap

log = logging.getLogger("tspire.host.vision.llm")

# Free-text intent words (from the model) -> Intent enum.
_INTENT_WORDS: dict[str, Intent] = {
    "attack": Intent.ATTACK,
    "defend": Intent.DEFEND,
    "block": Intent.DEFEND,
    "buff": Intent.BUFF,
    "debuff": Intent.DEBUFF,
    "attack_defend": Intent.ATTACK_DEFEND,
    "attack_buff": Intent.ATTACK_BUFF,
    "attack_debuff": Intent.ATTACK_DEBUFF,
    "sleep": Intent.SLEEP,
    "stun": Intent.STUN,
    "escape": Intent.ESCAPE,
    "unknown": Intent.UNKNOWN,
    "none": Intent.NONE,
}

_SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "gold": {"type": "integer"},
        "floor": {"type": "integer"},
        "deck_count": {"type": "integer"},
        "draw_pile_count": {"type": "integer"},
        "discard_pile_count": {"type": "integer"},
        "current_hp": {"type": "integer"},
        "max_hp": {"type": "integer"},
        "energy": {"type": "integer"},
        "player_powers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "amount": {"type": "integer"},
                },
                "required": ["name"],
            },
        },
        "monsters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current_hp": {"type": "integer"},
                    "max_hp": {"type": "integer"},
                    "block": {"type": "integer"},
                    "intent": {"type": "string"},
                    "intent_value": {"type": "integer"},
                    "powers": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "amount": {"type": "integer"},
                            },
                            "required": ["name"],
                        },
                    },
                },
                "required": ["current_hp", "max_hp", "intent"],
            },
        },
        "hand": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "cost": {"type": "integer"}},
                "required": ["name", "cost"],
            },
        },
    },
    "required": ["monsters", "hand"],
}

_SCENE_PROMPT = (
    "Read this Slay the Spire combat screenshot and return JSON.\n"
    "RUN: read the player's gold, current floor, master deck count at the top-right, "
    "current/max HP, and current energy if visible.\n"
    "PILES: read the draw pile count in the bottom-left and discard pile count in the "
    "bottom-right.\n"
    "PLAYER POWERS: list visible player buffs/debuffs under the player HP bar with their "
    "stack amounts when a number is visible.\n"
    "ENEMIES (monsters): there may be 1 to 5. Look carefully across the WHOLE battlefield; "
    "some are small. Each enemy has a red HP bar showing current/max HP, optional block "
    "(a number on a shield), and an intent icon above it. If the intent shows a number it "
    "is an attack and that number is intent_value (per-hit damage); otherwise intent is one "
    "of defend, buff, debuff, sleep, unknown. List visible enemy buffs/debuffs as powers.\n"
    "HAND: list every card along the bottom, left to right, with its name and the energy "
    "cost shown in the circular gem at the card's top-left."
)

_DETAILS_PROMPT = (
    "This is a zoomed crop of the Slay the Spire combat area. Return JSON with monsters "
    "and hand. MONSTERS: list every visible enemy left to right; use the HP bar text for "
    "current_hp/max_hp, include block if a shield number is visible, and describe the "
    "intent above the enemy. If the intent has an attack number, put that number in "
    "intent_value. HAND: list every visible card along the bottom left to right with name "
    "and cost. Only return an empty array when that section is truly not visible."
)

_DETAILS_RECT = Rect(0.030, 0.120, 0.940, 0.860)

# One crop per numeric field: even two images per call makes this model bleed one value
# into the other, so each fixed stat is read in its own single-image call.
_PAIR_SCHEMA = {
    "type": "object",
    "properties": {"current": {"type": "integer"}, "max": {"type": "integer"}},
    "required": ["current", "max"],
}
_ENERGY_PROMPT = (
    "This is the ENERGY orb from a Slay the Spire combat screen, showing current/max energy "
    "(for example 3/3). Return current and max."
)
_HP_PROMPT = (
    "This is the player's HP bar from a Slay the Spire combat screen, showing current/max HP "
    "(for example 80/80). Return current and max."
)

_BLOCK_SCHEMA = {
    "type": "object",
    "properties": {"block": {"type": "integer"}},
    "required": ["block"],
}
_BLOCK_PROMPT = (
    "This is a zoomed crop of the player's block badge in Slay the Spire (a number on a "
    "shield). Return the block number; if no shield/number is visible, return 0."
)

_VALUE_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "integer"}},
    "required": ["value"],
}
_GOLD_PROMPT = (
    "This is the player's gold counter from the Slay the Spire top bar. Return only the "
    "gold amount as value. If no readable number is visible, return -1."
)
_FLOOR_PROMPT = (
    "This is the current floor number from the Slay the Spire top-center banner. Return "
    "only the floor as value. If no readable number is visible, return -1."
)
_DECK_PROMPT = (
    "This is the master deck count from the Slay the Spire top-right deck icon. Return "
    "only the card count as value. If no readable number is visible, return -1."
)


class OllamaError(RuntimeError):
    pass


class OllamaVisionParser:
    def __init__(
        self,
        model: str,
        url: str,
        regions: RegionMap,
        image_width: int = 1024,
        *,
        think: bool = False,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.regions = regions
        self.image_width = image_width
        self.think = bool(think)

    # --- public ----------------------------------------------------------
    def parse_combat(self, frame, *, read_block: bool = False, ocr=None) -> ParseResult:
        """Parse the combat frame.

        The busy scene (monsters + hand) always goes to the model. The fixed HUD numbers
        (energy, HP, block, gold, floor, deck) are read with local OCR first when an ``ocr``
        backend is supplied, falling back to a per-field model call only when OCR yields
        nothing -- collapsing the common case from ~6 model calls to one. ``read_block``
        triggers a block read only when a block badge was detected (usually absent)."""
        scene = self._scene(frame)
        energy, energy_max = self._read_pair(ocr, frame, self.regions.energy, _ENERGY_PROMPT)
        hp, hp_max = self._read_pair(ocr, frame, self.regions.player_hp, _HP_PROMPT)
        scene_energy = _intish(scene.get("energy", 0), 0)
        scene_hp = _intish(scene.get("current_hp", scene.get("hp", 0)), 0)
        scene_hp_max = _intish(scene.get("max_hp", 0), 0)
        gold, gold_seen = self._read_value(ocr, frame, self.regions.gold, _GOLD_PROMPT)
        floor, floor_seen = self._read_value(ocr, frame, self.regions.floor, _FLOOR_PROMPT)
        deck_count, deck_seen = self._read_value(ocr, frame, self.regions.deck_count, _DECK_PROMPT)
        if energy_max <= 0 and "energy" in scene:
            energy = scene_energy
        if hp <= 0 and hp_max <= 0 and scene_hp > 0:
            hp = scene_hp
        if hp_max <= 0 and scene_hp_max > 0:
            hp_max = scene_hp_max
        if hp_max <= 0:
            hp, hp_max = self._read_pair(ocr, frame, self.regions.top_hp, _HP_PROMPT)
        block = self._read_block(ocr, frame) if read_block else 0

        player = PlayerCombat(
            current_hp=hp,
            max_hp=hp_max,
            block=block,
            energy=energy,
            powers=self._to_powers(scene.get("player_powers", scene.get("powers", []))),
        )
        monsters_data = scene.get("monsters", scene.get("enemies", []))
        hand_data = scene.get("hand", scene.get("cards", []))
        if not monsters_data or not hand_data:
            details = self._details(frame)
            if not monsters_data:
                monsters_data = details.get("monsters", details.get("enemies", []))
            if not hand_data:
                hand_data = details.get("hand", details.get("cards", []))
        monsters = [self._to_monster(m, i) for i, m in enumerate(monsters_data)]
        hand = [self._to_card(c, i) for i, c in enumerate(hand_data)]
        combat = CombatState(
            player=player,
            monsters=monsters,
            hand=hand,
            draw_pile_count=_intish(scene.get("draw_pile_count", scene.get("draw_pile", 0)), 0),
            discard_pile_count=_intish(scene.get("discard_pile_count", scene.get("discard_pile", 0)), 0),
        )

        # Confidence: did we get the basics? (hp read + at least one monster + a hand)
        signals = [player.max_hp > 0, bool(monsters), bool(hand)]
        confidence = round(sum(1 for s in signals if s) / len(signals), 2)
        observed = {
            "current_hp": hp_max > 0 or hp > 0,
            "max_hp": hp_max > 0,
            "energy": energy_max > 0 or "energy" in scene,
            "block": True,
            "gold": gold_seen or ("gold" in scene and _intish(scene.get("gold", 0), 0) > 0),
            "floor": floor_seen or ("floor" in scene and _intish(scene.get("floor", 0), 0) > 0),
            "deck_count": deck_seen
            or (("deck_count" in scene or "deck" in scene)
                and _intish(scene.get("deck_count", scene.get("deck", 0)), 0) > 0),
            "draw_pile_count": "draw_pile_count" in scene or "draw_pile" in scene,
            "discard_pile_count": "discard_pile_count" in scene or "discard_pile" in scene,
            "monsters": bool(monsters),
            "hand": bool(hand),
        }
        return ParseResult(
            combat=combat,
            confidence=confidence,
            gold=gold if gold_seen else max(0, _intish(scene.get("gold", 0), 0)),
            floor=floor if floor_seen else _intish(scene.get("floor", 0), 0),
            deck_count=deck_count if deck_seen else _intish(scene.get("deck_count", scene.get("deck", 0)), 0),
            observed=observed,
        )

    # --- arbiter re-reads -------------------------------------------------
    # The reconciler calls these to break a hard vision/prediction conflict by re-reading
    # one fixed region at high zoom. Same upscaled-crop path as the main stats calls.
    def reread_player_hp(self, frame) -> tuple[int, int]:
        return self._pair(frame, self.regions.player_hp, _HP_PROMPT)

    def reread_energy(self, frame) -> tuple[int, int]:
        return self._pair(frame, self.regions.energy, _ENERGY_PROMPT)

    # --- OCR-first HUD reads ---------------------------------------------
    # Try local OCR for a fixed-region number; fall back to the (slow) model crop only when
    # OCR is unavailable or returns nothing. Keeps the fast path off the model entirely.
    def _read_pair(self, ocr, frame, rect: Rect, prompt: str) -> tuple[int, int]:
        pair = _ocr_pair(ocr, frame, rect)
        if pair is not None:
            return pair
        return self._pair(frame, rect, prompt)

    def _read_value(self, ocr, frame, rect: Rect, prompt: str) -> tuple[int, bool]:
        value = _ocr_int(ocr, frame, rect)
        if value is not None and value > 0:
            return value, True
        return self._value(frame, rect, prompt, allow_zero=False)

    def _read_block(self, ocr, frame) -> int:
        value = _ocr_int(ocr, frame, self.regions.player_block)
        if value is not None:
            return max(0, value)
        return self._block(frame)

    # --- calls ------------------------------------------------------------
    def _scene(self, frame) -> dict:
        full = self._encode(self._resize_to_width(frame, self.image_width))
        return self._generate(_SCENE_PROMPT, [full], _SCENE_SCHEMA)

    def _details(self, frame) -> dict:
        try:
            crop = self._encode(self._resize_to_width(self._crop_rect(frame, _DETAILS_RECT), self.image_width))
            return self._generate(_DETAILS_PROMPT, [crop], _SCENE_SCHEMA)
        except Exception:
            log.debug("combat details fallback failed", exc_info=True)
            return {}

    def _pair(self, frame, rect: Rect, prompt: str) -> tuple[int, int]:
        crop = self._encode(self._crop_upscaled(frame, rect))
        try:
            data = self._generate(prompt, [crop], _PAIR_SCHEMA)
            return _pair_values(data)
        except OllamaError:
            return 0, 0

    def _block(self, frame) -> int:
        crop = self._encode(self._crop_upscaled(frame, self.regions.player_block))
        try:
            return int(self._generate(_BLOCK_PROMPT, [crop], _BLOCK_SCHEMA).get("block", 0))
        except OllamaError:
            return 0

    def _value(self, frame, rect: Rect, prompt: str, *, allow_zero: bool = True) -> tuple[int, bool]:
        try:
            crop = self._encode(self._crop_upscaled(frame, rect))
            value = _intish(self._generate(prompt, [crop], _VALUE_SCHEMA).get("value", -1), -1)
        except Exception:
            return 0, False
        if value == 0 and not allow_zero:
            return 0, False
        return (max(0, value), True) if value >= 0 else (0, False)

    def _generate(self, prompt: str, images: list[str], schema: dict) -> dict:
        errors: list[str] = []
        text = ""
        for attempt in range(2):
            request_prompt = prompt
            if attempt:
                request_prompt = (
                    f"{prompt}\n\nYour previous answer was invalid: "
                    f"{'; '.join(errors[:3])}. Return only JSON that matches the schema."
                )
            text = self._generate_text(request_prompt, images, schema)
            try:
                data = _normalize_schema_response(_loads_json_object(text), schema)
            except json.JSONDecodeError as exc:
                errors = [f"invalid JSON: {exc.msg}"]
                continue
            errors = _validate_json_schema(data, schema)
            if not errors:
                return data
            log.warning("Ollama response failed schema validation: %s", "; ".join(errors[:3]))
        detail = "; ".join(errors[:3]) if errors else "empty response"
        raise OllamaError(f"model did not return valid schema JSON ({detail}): {text[:200]}")

    def _generate_text(self, prompt: str, images: list[str], schema: dict) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "format": schema,
            "think": self.think,
            "keep_alive": "10m",
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(
            f"{self.url}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                body = json.load(resp)
        except Exception as exc:  # network / timeout / decode
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        text = body.get("response")
        if not isinstance(text, str):
            raise OllamaError("Ollama response did not include a text response")
        return text.strip()

    # --- image helpers ----------------------------------------------------
    @staticmethod
    def _resize_to_width(frame, width: int):
        import cv2

        h, w = frame.shape[:2]
        if w == width:
            return frame
        return cv2.resize(frame, (width, max(1, round(h * width / w))))

    def _crop_upscaled(self, frame, rect: Rect, scale: int = 5):
        import cv2

        h, w = frame.shape[:2]
        left, top, cw, ch = rect.to_pixels(w, h)
        crop = frame[top : top + ch, left : left + cw]
        if crop.size == 0:
            return crop
        return cv2.resize(crop, (max(1, cw * scale), max(1, ch * scale)), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _crop_rect(frame, rect: Rect):
        h, w = frame.shape[:2]
        left, top, cw, ch = rect.to_pixels(w, h)
        return frame[top : top + ch, left : left + cw]

    @staticmethod
    def _encode(image) -> str:
        import cv2

        return base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode()

    # --- mapping ----------------------------------------------------------
    @staticmethod
    def _to_monster(data: dict, index: int) -> Monster:
        intent = _to_intent(data.get("intent", ""))
        name = str(data.get("name") or "").strip()
        if name.lower() in {"none", "unknown", "enemy"}:
            name = ""
        current_hp, max_hp = _hp_values(data)
        return Monster(
            name=name,
            current_hp=current_hp,
            max_hp=max_hp,
            block=_intish(data.get("block", 0), 0),
            intent=intent,
            intent_damage=_intish(
                data.get("intent_value", data.get("intent_damage", data.get("damage", 0))), 0
            ),
            powers=OllamaVisionParser._to_powers(data.get("powers", [])),
            index=index,
        )

    @staticmethod
    def _to_card(data: dict, index: int) -> Card:
        return Card(
            name=str(data.get("name") or data.get("card") or "").strip(),
            cost=_intish(data.get("cost", -1), -1),
            is_playable=True,
            index=index,
        )

    @staticmethod
    def _to_powers(items) -> list[Power]:
        powers: list[Power] = []
        if not isinstance(items, list):
            return powers
        for item in items:
            if isinstance(item, str):
                name = item.strip()
                amount = 0
            elif isinstance(item, dict):
                name = str(item.get("name") or item.get("power") or item.get("id") or "").strip()
                amount = _intish(item.get("amount", item.get("stacks", item.get("stack", 0))), 0)
            else:
                continue
            if not name or name.lower() in {"none", "unknown", "no powers"}:
                continue
            powers.append(Power(power_id=name, name=name, amount=amount))
        return powers


def _ocr_pair(ocr, frame, rect: Rect) -> tuple[int, int] | None:
    """Read a 'current/max' field via local OCR. None when unavailable or unreadable."""
    if ocr is None:
        return None
    try:
        current, maximum = ocr.ocr_int_pair(frame, rect)
    except Exception:
        log.debug("OCR pair read failed; falling back to model", exc_info=True)
        return None
    if maximum > 0 or current > 0:
        return current, maximum
    return None


def _ocr_int(ocr, frame, rect: Rect) -> int | None:
    """Read a single number via local OCR. None when unavailable or unreadable."""
    if ocr is None:
        return None
    try:
        value = ocr.ocr_int(frame, rect, default=-1)
    except Exception:
        log.debug("OCR int read failed; falling back to model", exc_info=True)
        return None
    return value if value >= 0 else None


def _loads_json_object(text: str) -> dict:
    """Load JSON from strict output, fenced markdown, or text with one JSON object."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(stripped[start : end + 1])
        else:
            data = _loads_key_value_object(stripped)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("expected JSON object", stripped, 0)
    return data


def _loads_key_value_object(text: str) -> dict:
    pairs = re.findall(r"([A-Za-z_][\w ]*)\s*:\s*([^,\n]+)", text)
    if not pairs:
        raise json.JSONDecodeError("expected JSON object", text, 0)
    data = {}
    for key, value in pairs:
        clean_key = key.strip().lower().replace(" ", "_")
        clean_value = value.strip().strip('"').strip("'")
        data[clean_key] = _intish(clean_value, clean_value)
    return data


def _normalize_schema_response(data: dict, schema: dict) -> dict:
    if schema is _SCENE_SCHEMA:
        data = _normalize_scene_response(data)
    elif schema is _PAIR_SCHEMA:
        data = _normalize_pair_response(data)
    elif schema is _BLOCK_SCHEMA:
        data = {"block": _intish(data.get("block", data.get("value", 0)), 0)}
    elif schema is _VALUE_SCHEMA:
        data = {"value": _intish(data.get("value", data.get("amount", -1)), -1)}
    return _coerce_schema_types(data, schema)


def _normalize_scene_response(data: dict) -> dict:
    out = dict(data)
    run = _as_dict(data.get("run"))
    piles = _as_dict(data.get("piles"))
    player = _as_dict(data.get("player"))

    _copy_first(out, "gold", run, "gold")
    _copy_first(out, "floor", run, "floor", "current_floor")
    _copy_first(out, "deck_count", run, "deck_count", "master_deck_count", "deck", "card_count")
    _copy_first(out, "energy", run, "energy", "current_energy")
    _copy_first(out, "energy", player, "energy", "current_energy")
    _copy_first(out, "draw_pile_count", piles, "draw_pile_count", "draw_pile", "draw")
    _copy_first(out, "discard_pile_count", piles, "discard_pile_count", "discard_pile", "discard")

    hp_sources = (
        _as_dict(run.get("hp")),
        _as_dict(player.get("hp")),
        _as_dict(data.get("hp")),
        run,
        player,
    )
    for hp in hp_sources:
        _copy_first(out, "current_hp", hp, "current_hp", "current", "hp_current")
        _copy_first(out, "max_hp", hp, "max_hp", "max", "maximum", "hp_max")

    if "player_powers" not in out:
        _copy_first(out, "player_powers", player, "player_powers", "powers")

    monsters = _first_present(data, "monsters", "enemies")
    if monsters is not None:
        out["monsters"] = [_normalize_monster_response(m) for m in _as_list(monsters)]

    hand = _first_present(data, "hand", "cards", "hand_cards")
    if hand is not None:
        out["hand"] = [_normalize_card_response(c) for c in _as_list(hand)]

    return out


def _normalize_monster_response(item) -> dict:
    monster = dict(item) if isinstance(item, dict) else {}
    if "name" not in monster:
        _copy_first(monster, "name", monster, "monster", "enemy", "id")
    current_hp, max_hp = _hp_values(monster)
    if _has_hp_value(monster):
        monster.setdefault("current_hp", current_hp)
        monster.setdefault("max_hp", max_hp)
    if "block" in monster:
        block = monster["block"]
        if block is None or str(block).strip().lower() in {"", "none", "no block"}:
            monster["block"] = 0
    if "intent" not in monster:
        _copy_first(monster, "intent", monster, "action", "move", "intent_type")
    monster.setdefault("intent", "unknown")
    if "intent_value" not in monster:
        _copy_first(monster, "intent_value", monster, "intent_damage", "damage", "attack_damage")
    if "powers" in monster:
        monster["powers"] = [_normalize_power_response(p) for p in _as_list(monster["powers"])]
    return monster


def _normalize_card_response(item) -> dict:
    card = dict(item) if isinstance(item, dict) else {}
    if "name" not in card:
        _copy_first(card, "name", card, "card", "title")
    card.setdefault("name", "")
    if "cost" not in card:
        _copy_first(card, "cost", card, "energy", "energy_cost")
    card.setdefault("cost", -1)
    return card


def _normalize_power_response(item) -> dict:
    if isinstance(item, str):
        return {"name": item, "amount": 0}
    power = dict(item) if isinstance(item, dict) else {}
    if "name" not in power:
        _copy_first(power, "name", power, "power", "id")
    power.setdefault("name", "")
    if "amount" not in power:
        _copy_first(power, "amount", power, "stacks", "stack")
    power.setdefault("amount", 0)
    return power


def _normalize_pair_response(data: dict) -> dict:
    current, maximum = _pair_values(data)
    return {"current": current, "max": maximum}


def _coerce_schema_types(value, schema: dict):
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return value
        coerced = {}
        for name, child_schema in schema.get("properties", {}).items():
            if name in value:
                coerced[name] = _coerce_schema_types(value[name], child_schema)
        return coerced
    if expected == "array":
        if not isinstance(value, list):
            return value
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return value
        return [_coerce_schema_types(item, item_schema) for item in value]
    if expected == "integer":
        return _intish(value, 0)
    if expected == "string":
        return "" if value is None else str(value)
    return value


def _copy_first(target: dict, target_key: str, source: dict, *source_keys: str) -> None:
    if target_key in target or not source:
        return
    value = _first_present(source, *source_keys)
    if value is not None:
        target[target_key] = value


def _first_present(source: dict, *keys: str):
    for key in keys:
        if key in source:
            return source[key]
    return None


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _has_hp_value(data: dict) -> bool:
    return any(key in data for key in ("current_hp", "max_hp", "current", "max", "hp", "health"))


def _validate_json_schema(value, schema: dict, path: str = "$") -> list[str]:
    """Validate the small JSON Schema subset sent to Ollama.

    This intentionally stays dependency-free and covers the types used in this file:
    object, array, integer, string, required, properties, and items.
    """
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path} expected object"]
        errors: list[str] = []
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}.{name} missing")
        properties = schema.get("properties", {})
        for name, child_schema in properties.items():
            if name in value:
                errors.extend(_validate_json_schema(value[name], child_schema, f"{path}.{name}"))
        return errors
    if expected == "array":
        if not isinstance(value, list):
            return [f"{path} expected array"]
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return []
        errors: list[str] = []
        for i, item in enumerate(value):
            errors.extend(_validate_json_schema(item, item_schema, f"{path}[{i}]"))
        return errors
    if expected == "integer":
        return [] if isinstance(value, int) and not isinstance(value, bool) else [f"{path} expected integer"]
    if expected == "string":
        return [] if isinstance(value, str) else [f"{path} expected string"]
    return []


def _hp_values(data: dict) -> tuple[int, int]:
    current = _intish(data.get("current_hp", data.get("current", 0)), 0)
    maximum = _intish(data.get("max_hp", data.get("max", 0)), 0)
    if (current, maximum) != (0, 0):
        return current, maximum
    hp = data.get("hp", data.get("health", ""))
    if isinstance(hp, dict):
        current = _intish(hp.get("current", hp.get("current_hp", 0)), 0)
        maximum = _intish(hp.get("max", hp.get("max_hp", hp.get("maximum", 0))), 0)
        return current, maximum
    if isinstance(hp, str):
        nums = [int(n) for n in re.findall(r"\d+", hp)]
        if len(nums) >= 2:
            return nums[0], nums[1]
        if len(nums) == 1:
            return nums[0], nums[0]
    return 0, 0


def _pair_values(data: dict) -> tuple[int, int]:
    current = _intish(data.get("current", data.get("current_hp", 0)), 0)
    maximum = _intish(data.get("max", data.get("max_hp", 0)), 0)
    if (current, maximum) != (0, 0):
        return current, maximum
    for key in ("hp", "health", "energy", "value"):
        value = data.get(key)
        if isinstance(value, dict):
            current = _intish(value.get("current", value.get("current_hp", 0)), 0)
            maximum = _intish(value.get("max", value.get("max_hp", value.get("maximum", 0))), 0)
            return current, maximum
        if isinstance(value, str):
            nums = [int(n) for n in re.findall(r"\d+", value)]
            if len(nums) >= 2:
                return nums[0], nums[1]
            if len(nums) == 1:
                return nums[0], nums[0]
    return 0, 0


def _to_intent(value) -> Intent:
    text = str(value or "").lower().strip()
    normalized = re.sub(r"[^a-z_]+", "_", text).strip("_")
    if normalized in _INTENT_WORDS:
        return _INTENT_WORDS[normalized]
    if "attack" in normalized:
        if "defend" in normalized or "block" in normalized:
            return Intent.ATTACK_DEFEND
        if "buff" in normalized:
            return Intent.ATTACK_BUFF
        if "debuff" in normalized:
            return Intent.ATTACK_DEBUFF
        return Intent.ATTACK
    if "defend" in normalized or "block" in normalized:
        return Intent.DEFEND
    if "debuff" in normalized:
        return Intent.DEBUFF
    if "buff" in normalized:
        return Intent.BUFF
    if "sleep" in normalized:
        return Intent.SLEEP
    if "stun" in normalized:
        return Intent.STUN
    if "escape" in normalized:
        return Intent.ESCAPE
    return Intent.UNKNOWN


def _intish(value, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value)
        if m:
            return int(m.group())
    return default
