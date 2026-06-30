"""Combat screen parser: frame -> CombatState.

Composes the vision primitives (OCR, HP-bar detection, card detection, template
classification) into the structured combat state. All the pixel offsets used to derive
sub-regions from a detected monster/card box are **calibration estimates** (marked
CALIBRATE) and expected to be tuned against real captures via the calibrate overlay.

The function is intentionally tolerant: any sub-read that fails leaves a sensible default
rather than raising, and an overall confidence score reflects how much was read cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

from tspire.common.schema import Card, CombatState, Intent, Monster, PlayerCombat
from tspire.host.vision.backend import BBox, VisionBackend
from tspire.host.vision.regions import Rect, RegionMap

# Map template-id stems (from templates/intents/*.png) to the Intent enum. Extend as the
# intent template set grows; unknown ids fall through to Intent.UNKNOWN.
_INTENT_ALIASES: dict[str, Intent] = {
    "attack": Intent.ATTACK,
    "aggressive": Intent.ATTACK,
    "attack_buff": Intent.ATTACK_BUFF,
    "attack_debuff": Intent.ATTACK_DEBUFF,
    "attack_defend": Intent.ATTACK_DEFEND,
    "defend": Intent.DEFEND,
    "defend_buff": Intent.DEFEND_BUFF,
    "defend_debuff": Intent.DEFEND_DEBUFF,
    "buff": Intent.BUFF,
    "debuff": Intent.DEBUFF,
    "strong_debuff": Intent.STRONG_DEBUFF,
    "escape": Intent.ESCAPE,
    "sleep": Intent.SLEEP,
    "stun": Intent.STUN,
    "unknown": Intent.UNKNOWN,
}

# Confidence below which a template match is treated as "no match".
_MATCH_THRESHOLD = 0.55


@dataclass
class ParseResult:
    combat: CombatState
    confidence: float
    gold: int = 0
    floor: int = 0
    deck_count: int = 0


def parse_combat(frame, regions: RegionMap, backend: VisionBackend) -> ParseResult:
    h, w = frame.shape[:2]
    signals: list[bool] = []

    player = _parse_player(frame, regions, backend, signals)
    monsters = _parse_monsters(frame, regions, backend, w, h, signals)
    hand = _parse_hand(frame, regions, backend, w, h, signals)

    combat = CombatState(
        player=player,
        monsters=monsters,
        hand=hand,
        draw_pile_count=backend.ocr_int(frame, regions.draw_pile),
        discard_pile_count=backend.ocr_int(frame, regions.discard_pile),
    )
    confidence = (sum(signals) / len(signals)) if signals else 0.0
    return ParseResult(
        combat=combat,
        confidence=confidence,
        gold=backend.ocr_int(frame, regions.gold),
        floor=backend.ocr_int(frame, regions.floor),
        deck_count=backend.ocr_int(frame, regions.deck_count),
    )


def _parse_player(frame, regions: RegionMap, backend: VisionBackend, signals: list[bool]) -> PlayerCombat:
    hp, hp_max = backend.ocr_int_pair(frame, regions.player_hp)
    if hp_max <= 0:
        hp, hp_max = backend.ocr_int_pair(frame, regions.top_hp)
    energy, energy_max = backend.ocr_int_pair(frame, regions.energy)
    block = 0
    if backend.region_filled(frame, regions.player_block):
        block = backend.ocr_int(frame, regions.player_block)
    signals.append(hp_max > 0)
    signals.append(energy_max > 0 or energy > 0)
    return PlayerCombat(
        current_hp=hp, max_hp=hp_max, block=block, energy=energy
    )


def _parse_monsters(frame, regions: RegionMap, backend: VisionBackend, w: int, h: int, signals: list[bool]) -> list[Monster]:
    bars = backend.find_red_bars(frame, regions.monster_search)
    monsters: list[Monster] = []
    for i, bar in enumerate(bars):
        hp, hp_max = backend.ocr_int_pair(frame, _hp_text_rect(bar, w, h))
        intent, dmg = _parse_intent(frame, bar, backend, w, h)
        name, score = backend.classify_box(frame, _sprite_box(bar), "monsters")
        monsters.append(
            Monster(
                name=name if score >= _MATCH_THRESHOLD else "",
                monster_id=name if score >= _MATCH_THRESHOLD else "",
                current_hp=hp,
                max_hp=hp_max,
                intent=intent,
                intent_damage=dmg,
                index=i,
            )
        )
        signals.append(hp_max > 0)
    return monsters


def _parse_hand(frame, regions: RegionMap, backend: VisionBackend, w: int, h: int, signals: list[bool]) -> list[Card]:
    boxes = backend.find_cards(frame, regions.hand_search)
    hand: list[Card] = []
    for i, box in enumerate(boxes):
        cost = backend.ocr_int(frame, _cost_gem_rect(box, w, h), default=-1)
        name = backend.ocr_text(frame, _card_title_rect(box, w, h))
        hand.append(
            Card(name=name, cost=cost, is_playable=True, index=i)
        )
        signals.append(bool(name) or cost >= 0)
    return hand


# --------------------------------------------------------------------------- #
# Sub-region geometry derived from a detected box. CALIBRATE: all ratios below.
# --------------------------------------------------------------------------- #
def _box_to_rect(box: BBox, w: int, h: int) -> Rect:
    return Rect(box.left / w, box.top / h, box.width / w, box.height / h)


def _hp_text_rect(bar: BBox, w: int, h: int) -> Rect:
    # The "cur/max" text sits on/just below the HP bar.
    box = BBox(left=bar.left, top=bar.top, width=bar.width, height=max(bar.height * 3, 18))
    return _box_to_rect(box, w, h)


def _parse_intent(frame, bar: BBox, backend: VisionBackend, w: int, h: int) -> tuple[Intent, int]:
    # Intent icon floats above the sprite, well above the HP bar. CALIBRATE offsets.
    size = max(int(bar.width * 0.6), 24)
    box = BBox(
        left=bar.cx - size // 2,
        top=max(0, bar.top - int(bar.width * 1.4)),
        width=size,
        height=size,
    )
    intent_id, score = backend.classify_box(frame, box, "intents")
    intent = _INTENT_ALIASES.get(intent_id.lower(), Intent.UNKNOWN) if score >= _MATCH_THRESHOLD else Intent.UNKNOWN
    dmg = 0
    if intent.is_attack:
        # Damage number is rendered under the intent icon.
        num_box = BBox(left=box.left, top=box.top + size, width=size, height=size // 2)
        dmg = backend.ocr_int(frame, _box_to_rect(num_box, w, h), default=0)
    return intent, dmg


def _sprite_box(bar: BBox) -> BBox:
    # The monster sprite occupies the space above its HP bar. CALIBRATE.
    height = int(bar.width * 1.2)
    return BBox(left=bar.left, top=max(0, bar.top - height), width=bar.width, height=height)


def _cost_gem_rect(card: BBox, w: int, h: int) -> Rect:
    side = int(card.width * 0.24)
    box = BBox(left=card.left, top=card.top, width=side, height=side)
    return _box_to_rect(box, w, h)


def _card_title_rect(card: BBox, w: int, h: int) -> Rect:
    box = BBox(
        left=card.left + int(card.width * 0.12),
        top=card.top + int(card.height * 0.05),
        width=int(card.width * 0.76),
        height=int(card.height * 0.14),
    )
    return _box_to_rect(box, w, h)
