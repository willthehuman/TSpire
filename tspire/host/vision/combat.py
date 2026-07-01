"""Combat screen parser: frame -> CombatState.

Composes the vision primitives (OCR, HP-bar detection, card detection, template
classification) into the structured combat state. All the pixel offsets used to derive
sub-regions from a detected monster/card box are **calibration estimates** (marked
CALIBRATE) and expected to be tuned against real captures via the calibrate overlay.

The function is intentionally tolerant: any sub-read that fails leaves a sensible default
rather than raising, and an overall confidence score reflects how much was read cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    observed: dict[str, bool] = field(default_factory=dict)


def parse_combat(frame, regions: RegionMap, backend: VisionBackend, *, use_easyocr: bool = True) -> ParseResult:
    h, w = frame.shape[:2]
    signals: list[bool] = []

    # EasyOCR reads the game's stylised text (card titles, energy orb, deck) that Tesseract
    # can't; enabled only when requested AND installed. Tesseract still does the flat text.
    use_eo = bool(use_easyocr) and _easyocr_on()

    player, observed = _parse_player(frame, regions, backend, signals, use_eo)
    monsters = _parse_monsters(frame, regions, backend, w, h, signals)
    hand = _parse_hand(frame, regions, backend, w, h, signals, use_eo)
    signals.append(bool(monsters))
    signals.append(bool(hand))

    draw = backend.ocr_int(frame, regions.draw_pile, default=-1)
    discard = backend.ocr_int(frame, regions.discard_pile, default=-1)
    gold = backend.ocr_int(frame, regions.gold, default=-1)
    floor = backend.ocr_int(frame, regions.floor, default=-1)
    deck_count = _eo_int(frame, _DECK_EO, w, h) if use_eo else -1
    if deck_count < 0:
        deck_count = backend.ocr_int(frame, regions.deck_count, default=-1)
    observed.update(
        {
            "draw_pile_count": draw >= 0,
            "discard_pile_count": discard >= 0,
            "gold": gold >= 0,
            "floor": floor > 0,
            "deck_count": deck_count > 0,
            "monsters": bool(monsters),
            "hand": bool(hand),
        }
    )

    combat = CombatState(
        player=player,
        monsters=monsters,
        hand=hand,
        draw_pile_count=max(0, draw),
        discard_pile_count=max(0, discard),
    )
    confidence = (sum(signals) / len(signals)) if signals else 0.0
    return ParseResult(
        combat=combat,
        confidence=confidence,
        gold=max(0, gold),
        floor=max(0, floor),
        deck_count=max(0, deck_count),
        observed=observed,
    )


def _parse_player(
    frame, regions: RegionMap, backend: VisionBackend, signals: list[bool], use_eo: bool = False
) -> tuple[PlayerCombat, dict[str, bool]]:
    hp, hp_max = backend.ocr_int_pair(frame, regions.player_hp)
    if hp_max <= 0:
        hp, hp_max = backend.ocr_int_pair(frame, regions.top_hp)
    # The energy orb's stylised digits defeat Tesseract; EasyOCR reads them. The orb shows
    # cur/max but current is what matters for playability, so a single value is fine.
    energy = _eo_int(frame, _ENERGY_EO, frame.shape[1], frame.shape[0]) if use_eo else -1
    if energy >= 0:
        energy_max = energy
    else:
        energy, energy_max = backend.ocr_int_pair(frame, regions.energy)
    block = 0
    if backend.region_filled(frame, regions.player_block):
        block = backend.ocr_int(frame, regions.player_block)
    signals.append(hp_max > 0)
    signals.append(energy_max > 0 or energy > 0)
    return (
        PlayerCombat(current_hp=hp, max_hp=hp_max, block=block, energy=energy),
        {
            "current_hp": hp_max > 0 or hp > 0,
            "max_hp": hp_max > 0,
            "energy": energy_max > 0 or energy > 0,
            "block": True,
        },
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


def _parse_hand(frame, regions: RegionMap, backend: VisionBackend, w: int, h: int, signals: list[bool], use_eo: bool = False) -> list[Card]:
    from tspire.host.vision.card_names import default_card_index

    index = default_card_index()
    if use_eo:
        eo_hand = _parse_hand_easyocr(frame, w, h, index)
        if eo_hand is not None:
            for _ in eo_hand:
                signals.append(True)
            return eo_hand

    boxes = backend.find_cards(frame, regions.hand_search)
    hand: list[Card] = []
    for i, box in enumerate(boxes):
        cost = backend.ocr_int(frame, _cost_gem_rect(box, w, h), default=-1)
        raw = backend.ocr_text(frame, _card_title_rect(box, w, h))
        # Snap the (stylised, error-prone) OCR title to the nearest real card name; keep the
        # raw text only if nothing matches, so a bad read degrades instead of asserting junk.
        resolved, _score = index.resolve(raw)
        hand.append(
            Card(name=resolved or raw, cost=cost, is_playable=True, index=i)
        )
        signals.append(bool(resolved) or cost >= 0)
    return hand


# EasyOCR crop regions (fractions) for the stylised fields Tesseract can't read.
_ENERGY_EO = Rect(0.020, 0.795, 0.085, 0.105)   # energy orb, bottom-left
_DECK_EO = Rect(0.895, 0.010, 0.095, 0.060)     # deck counter, top-right
# The hand-title band spans the fanned hand; kept tall enough to catch the SUNK edge-card
# titles, and starting just below the "card key" numbers so those aren't read as costs.
_HAND_BAND_EO = Rect(0.245, 0.800, 0.510, 0.085)


def _easyocr_on() -> bool:
    from tspire.host.vision import easyocr_reader

    return easyocr_reader.available()


def _eo_int(frame, rect: Rect, w: int, h: int) -> int:
    from tspire.host.vision import easyocr_reader

    left, top, cw, ch = rect.to_pixels(w, h)
    crop = frame[top : top + ch, left : left + cw]
    return easyocr_reader.read_int(crop)


def _parse_hand_easyocr(frame, w: int, h: int, index) -> list[Card] | None:
    """Detect the hand with EasyOCR: one pass over the title band yields count + names +
    positions. Words are fuzzy-matched to real card names; digit tokens are treated as cost
    gems and mapped to the nearest card on their right. Returns None (fall back to Tesseract)
    when nothing is detected."""
    from tspire.host.vision import easyocr_reader

    left, top, bw, bh = _HAND_BAND_EO.to_pixels(w, h)
    dets = easyocr_reader.read_boxes(frame[top : top + bh, left : left + bw])
    if not dets:
        return None
    names: list[tuple[float, str]] = []
    costs: list[tuple[float, int]] = []
    for cx, _cy, text, _conf in dets:
        gx = left + cx
        token = text.strip()
        if token.isdigit():
            costs.append((gx, int(token)))
            continue
        name, score = index.resolve(text)
        if name and score >= 0.7:
            names.append((gx, name))
    if not names:
        return None
    names.sort()
    hand: list[Card] = []
    for i, (gx, name) in enumerate(names):
        # The cost gem sits just left of the title; take the nearest digit within that gap.
        gap = [(gx - nx, val) for nx, val in costs if 15 <= (gx - nx) <= 170 and val <= 9]
        cost = min(gap)[1] if gap else -1
        hand.append(Card(name=name, cost=cost, is_playable=True, index=i))
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
    """Read a monster's intent from the icon/number floating above it.

    A *pure* attack intent has no dedicated icon in the game's art -- it is just the damage
    number -- so we detect attacks by OCR'ing that number, and template-match the icon only for
    the non-attack / combo intents (defend, buff, debuff, stun, attackDefend, ...). Enemies vary
    in height, so both reads scan a band of vertical offsets (in bar-widths above the HP bar)
    rather than a single calibrated point.
    """
    # 1) attack damage number
    dmg = 0
    for up in (0.7, 0.85, 1.0, 1.15, 1.3):
        rect = Rect(
            (bar.cx - 0.32 * bar.width) / w,
            (bar.top - up * bar.width) / h,
            0.64 * bar.width / w,
            0.34 * bar.width / h,
        )
        val = backend.ocr_int(frame, rect, default=-1)
        if val > 0:
            dmg = val
            break

    # 2) icon classification (for defend/buff/debuff/stun/escape/sleep and attack-combos)
    intent_id, best = "", 0.0
    size = max(int(bar.width * 0.7), 32)
    for up in (0.85, 1.1, 1.4, 1.7):
        box = BBox(left=bar.cx - size // 2, top=max(0, bar.top - int(bar.width * up)), width=size, height=size)
        iid, score = backend.classify_box(frame, box, "intents")
        if score > best:
            intent_id, best = iid, score
    icon_intent = _INTENT_ALIASES.get(intent_id.lower(), Intent.UNKNOWN) if best >= _MATCH_THRESHOLD else Intent.UNKNOWN

    if dmg > 0:
        # A number means an attack; keep the combo type if the icon identified one.
        if icon_intent in (Intent.ATTACK_DEFEND, Intent.ATTACK_BUFF, Intent.ATTACK_DEBUFF):
            return icon_intent, dmg
        return Intent.ATTACK, dmg
    return icon_intent, 0


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
