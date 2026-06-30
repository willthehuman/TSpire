"""Local vision-model combat parser (Ollama).

Parses the combat scene with a multimodal model instead of hand-tuned CV. Validated with
gemma4:e4b-it-qat: it reads the busy battlefield (all enemies + the overlapping hand fan)
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


class OllamaError(RuntimeError):
    pass


class OllamaVisionParser:
    def __init__(self, model: str, url: str, regions: RegionMap, image_width: int = 1024) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.regions = regions
        self.image_width = image_width

    # --- public ----------------------------------------------------------
    def parse_combat(self, frame, *, read_block: bool = False) -> ParseResult:
        """Parse the combat frame. `read_block` triggers an extra block-reading call;
        the caller sets it only when a block badge is detected (it is usually absent),
        keeping the common case to two model calls."""
        scene = self._scene(frame)
        energy, _ = self._pair(frame, self.regions.energy, _ENERGY_PROMPT)
        hp, hp_max = self._pair(frame, self.regions.player_hp, _HP_PROMPT)
        scene_energy = _intish(scene.get("energy", 0), 0)
        scene_hp = _intish(scene.get("current_hp", scene.get("hp", 0)), 0)
        scene_hp_max = _intish(scene.get("max_hp", 0), 0)
        if energy <= 0 and scene_energy > 0:
            energy = scene_energy
        if hp <= 0 and scene_hp > 0:
            hp = scene_hp
        if hp_max <= 0 and scene_hp_max > 0:
            hp_max = scene_hp_max
        if hp_max <= 0:
            hp, hp_max = self._pair(frame, self.regions.top_hp, _HP_PROMPT)
        block = self._block(frame) if read_block else 0

        player = PlayerCombat(
            current_hp=hp,
            max_hp=hp_max,
            block=block,
            energy=energy,
            powers=self._to_powers(scene.get("player_powers", scene.get("powers", []))),
        )
        monsters_data = scene.get("monsters", scene.get("enemies", []))
        hand_data = scene.get("hand", scene.get("cards", []))
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
        return ParseResult(
            combat=combat,
            confidence=confidence,
            gold=_intish(scene.get("gold", 0), 0),
            floor=_intish(scene.get("floor", 0), 0),
            deck_count=_intish(scene.get("deck_count", scene.get("deck", 0)), 0),
        )

    # --- arbiter re-reads -------------------------------------------------
    # The reconciler calls these to break a hard vision/prediction conflict by re-reading
    # one fixed region at high zoom. Same upscaled-crop path as the main stats calls.
    def reread_player_hp(self, frame) -> tuple[int, int]:
        return self._pair(frame, self.regions.player_hp, _HP_PROMPT)

    def reread_energy(self, frame) -> tuple[int, int]:
        return self._pair(frame, self.regions.energy, _ENERGY_PROMPT)

    # --- calls ------------------------------------------------------------
    def _scene(self, frame) -> dict:
        full = self._encode(self._resize_to_width(frame, self.image_width))
        return self._generate(_SCENE_PROMPT, [full], _SCENE_SCHEMA)

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

    def _generate(self, prompt: str, images: list[str], schema: dict) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "stream": False,
            "format": schema,
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
        text = body.get("response", "").strip()
        try:
            return _loads_json_object(text)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"model did not return valid JSON: {text[:200]}") from exc

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
    def _encode(image) -> str:
        import cv2

        return base64.b64encode(cv2.imencode(".png", image)[1].tobytes()).decode()

    # --- mapping ----------------------------------------------------------
    @staticmethod
    def _to_monster(data: dict, index: int) -> Monster:
        intent = _INTENT_WORDS.get(str(data.get("intent", "")).lower().strip(), Intent.UNKNOWN)
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
            intent_damage=_intish(data.get("intent_value", 0), 0),
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


def _hp_values(data: dict) -> tuple[int, int]:
    current = _intish(data.get("current_hp", data.get("current", 0)), 0)
    maximum = _intish(data.get("max_hp", data.get("max", 0)), 0)
    if (current, maximum) != (0, 0):
        return current, maximum
    hp = data.get("hp", data.get("health", ""))
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
        if isinstance(value, str):
            nums = [int(n) for n in re.findall(r"\d+", value)]
            if len(nums) >= 2:
                return nums[0], nums[1]
            if len(nums) == 1:
                return nums[0], nums[0]
    return 0, 0


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
