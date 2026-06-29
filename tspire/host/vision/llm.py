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
import urllib.request

from tspire.common.schema import Card, CombatState, Intent, Monster, PlayerCombat
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
    "ENEMIES (monsters): there may be 1 to 5. Look carefully across the WHOLE battlefield; "
    "some are small. Each enemy has a red HP bar showing current/max HP, optional block "
    "(a number on a shield), and an intent icon above it. If the intent shows a number it "
    "is an attack and that number is intent_value (per-hit damage); otherwise intent is one "
    "of defend, buff, debuff, sleep, unknown.\n"
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
        block = self._block(frame) if read_block else 0

        player = PlayerCombat(
            current_hp=hp, max_hp=hp_max, block=block, energy=energy
        )
        monsters = [self._to_monster(m, i) for i, m in enumerate(scene.get("monsters", []))]
        hand = [self._to_card(c, i) for i, c in enumerate(scene.get("hand", []))]
        combat = CombatState(player=player, monsters=monsters, hand=hand)

        # Confidence: did we get the basics? (hp read + at least one monster + a hand)
        signals = [player.max_hp > 0, bool(monsters), bool(hand)]
        confidence = round(sum(1 for s in signals if s) / len(signals), 2)
        return ParseResult(combat=combat, confidence=confidence)

    # --- calls ------------------------------------------------------------
    def _scene(self, frame) -> dict:
        full = self._encode(self._resize_to_width(frame, self.image_width))
        return self._generate(_SCENE_PROMPT, [full], _SCENE_SCHEMA)

    def _pair(self, frame, rect: Rect, prompt: str) -> tuple[int, int]:
        crop = self._encode(self._crop_upscaled(frame, rect))
        try:
            data = self._generate(prompt, [crop], _PAIR_SCHEMA)
            return int(data.get("current", 0)), int(data.get("max", 0))
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
            return json.loads(text)
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
        return Monster(
            name=name,
            current_hp=int(data.get("current_hp", 0)),
            max_hp=int(data.get("max_hp", 0)),
            block=int(data.get("block", 0)),
            intent=intent,
            intent_damage=int(data.get("intent_value", 0)),
            index=index,
        )

    @staticmethod
    def _to_card(data: dict, index: int) -> Card:
        return Card(
            name=str(data.get("name") or "").strip(),
            cost=int(data.get("cost", -1)),
            is_playable=True,
            index=index,
        )
