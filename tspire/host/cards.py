"""Card metadata for the state predictor.

The predictor needs the base numeric effect of a played card to estimate the next combat
state. We derive simple card stats from the user's installed Slay the Spire jar at runtime
(class bytecode + localization), and keep a small curated fallback/override table for
semantic cases that cannot be proven from plain fields alone.

Nothing extracted from the game is bundled or written to the repo.
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

from tspire.host.game_assets import find_game_jar
from tspire.host.vision import classparse

log = logging.getLogger("tspire.host.cards")

_CARD_PKG = "com/megacrit/cardcrawl/cards/"
_LOC = "localization/{lang}/cards.json"
_ABSTRACT_CARD = "com/megacrit/cardcrawl/cards/AbstractCard"


@dataclass(frozen=True)
class CardData:
    """Base (and upgraded) numeric effect of a card.

    Existing predictor-facing fields are kept compatible. Additional jar-derived metadata is
    present for future use, but prediction still stays conservative via ``predictable``.
    """

    card_id: str
    damage: int = 0
    block: int = 0
    hits: int = 1
    target: bool = False
    aoe: bool = False
    exhausts: bool = False
    damage_up: int | None = None
    block_up: int | None = None
    name: str = ""
    cost: int | None = None
    type: str = ""
    color: str = ""
    rarity: str = ""
    target_type: str = ""
    magic: int = 0
    magic_up: int | None = None
    predictable: bool = True
    source: str = "curated"

    def damage_for(self, upgrades: int) -> int:
        if upgrades > 0 and self.damage_up is not None:
            return self.damage_up
        return self.damage

    def block_for(self, upgrades: int) -> int:
        if upgrades > 0 and self.block_up is not None:
            return self.block_up
        return self.block


ClassReader = Callable[[str], classparse.ClassFile]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def lookup(name_or_id: str) -> CardData | None:
    """Return predictable card data, or None when the card should defer to vision.

    Accepts a display name or id, with or without an upgrade marker ("Strike+").
    """
    if not name_or_id:
        return None
    key = _norm(name_or_id.split("+")[0])

    runtime = _runtime_card_db()
    data = runtime.get(key)
    override = CARD_DB.get(key)
    if data is not None and data.predictable:
        # Curated entries intentionally override jar-derived basics for semantic details
        # like multi-hit counts that are not safe to infer from base fields alone.
        return override or data
    return override


def build_card_table(
    jar: Path | str | None = None,
    *,
    loc_lang: str = "eng",
    reader: ClassReader | None = None,
    loc: dict | None = None,
    classes: list[str] | None = None,
) -> dict[str, CardData]:
    """Build a normalized lookup table from jar-derived card metadata.

    ``reader``, ``loc`` and ``classes`` are injectable for tests, so CI does not need a
    Slay the Spire install. Returned entries include both display-name and card-id aliases.
    """
    if reader is not None:
        return _build_card_table(reader, loc or {}, classes or [])
    if jar is None:
        return {}
    try:
        with zipfile.ZipFile(jar) as zf:
            loc = loc if loc is not None else _load_card_localization(zf, loc_lang)
            classes = classes if classes is not None else _list_card_classes(zf.namelist())

            def read(resource: str) -> classparse.ClassFile:
                return classparse.parse_class_file(zf.read(resource))

            return _build_card_table(read, loc, classes)
    except (FileNotFoundError, zipfile.BadZipFile, OSError):
        log.debug("could not build card table from jar", exc_info=True)
        return {}


@lru_cache(maxsize=1)
def _runtime_card_db() -> dict[str, CardData]:
    jar = find_game_jar("")
    if jar is None:
        return {}
    return build_card_table(jar)


def _build_card_table(reader: ClassReader, loc: dict, classes: list[str]) -> dict[str, CardData]:
    entries: list[tuple[str, CardData]] = []
    for resource in classes:
        if "$" in resource:
            continue
        try:
            class_file = reader(resource)
            data = _extract_card(resource, class_file, loc)
        except Exception:
            log.debug("could not extract card metadata from %s", resource, exc_info=True)
            continue
        if data is not None:
            entries.append((data.name or data.card_id, data))
    return _index_card_entries(entries)


def _extract_card(resource: str, class_file: classparse.ClassFile, loc: dict) -> CardData | None:
    if "/deprecated/" in resource.lower():
        return None
    init = _method(class_file, "<init>", "()V")
    if init is None or init.code is None:
        return None

    instructions = list(classparse.iter_instructions(class_file.pool, init.code))
    init_at = _abstract_card_init_index(instructions)
    if init_at is None:
        return None

    before_init = instructions[:init_at]
    card_id = _first_string(before_init)
    if not card_id:
        return None

    localized = loc.get(card_id, {}) if isinstance(loc, dict) else {}
    if not isinstance(localized, dict) or not localized.get("NAME"):
        return None
    display_name = str(localized.get("NAME") or card_id).strip()
    description = str(localized.get("DESCRIPTION") or "")

    cost_values = [ins.value for ins in before_init if _is_int_push(ins)]
    cost = cost_values[-1] if cost_values else None
    card_type = _enum_member(before_init, "AbstractCard$CardType")
    color = _enum_member(before_init, "AbstractCard$CardColor")
    rarity = _enum_member(before_init, "AbstractCard$CardRarity")
    target_type = _enum_member(before_init, "AbstractCard$CardTarget")

    fields = _field_assignments(instructions[init_at + 1 :])
    damage = max(0, fields.get("baseDamage", 0))
    block = max(0, fields.get("baseBlock", 0))
    magic = max(0, fields.get("baseMagicNumber", 0))
    exhausts = bool(fields.get("exhaust", 0))
    aoe = target_type == "ALL_ENEMY" or bool(fields.get("isMultiDamage", 0))
    target = target_type in {"ENEMY", "SELF_AND_ENEMY"}
    deltas = _upgrade_deltas(class_file)

    data = CardData(
        card_id=card_id,
        name=display_name,
        cost=cost,
        type=card_type,
        color=color,
        rarity=rarity,
        target_type=target_type,
        damage=damage,
        block=block,
        magic=magic,
        exhausts=exhausts,
        aoe=aoe,
        target=target,
        damage_up=damage + deltas["damage"] if damage and deltas["damage"] is not None else None,
        block_up=block + deltas["block"] if block and deltas["block"] is not None else None,
        magic_up=magic + deltas["magic"] if magic and deltas["magic"] is not None else None,
        predictable=False,
        source="jar",
    )
    return replace(data, predictable=_simple_numeric_effect(data, description))


def _abstract_card_init_index(instructions: list[classparse.Instruction]) -> int | None:
    for i, ins in enumerate(instructions):
        if ins.opname != "invokespecial" or not isinstance(ins.value, tuple):
            continue
        cls, name, desc = ins.value
        if cls == _ABSTRACT_CARD and name == "<init>" and desc.endswith("CardTarget;)V"):
            return i
    return None


def _method(class_file: classparse.ClassFile, name: str, descriptor: str | None = None):
    for method in class_file.methods:
        if method.name == name and (descriptor is None or method.descriptor == descriptor):
            return method
    return None


def _first_string(instructions: list[classparse.Instruction]) -> str:
    for ins in instructions:
        if ins.opname in {"ldc", "ldc_w"} and isinstance(ins.value, str):
            return ins.value
    return ""


def _enum_member(instructions: list[classparse.Instruction], owner_suffix: str) -> str:
    for ins in instructions:
        if ins.opname != "getstatic" or not isinstance(ins.value, tuple):
            continue
        cls, name, _desc = ins.value
        if cls and cls.split("/")[-1].endswith(owner_suffix):
            return name
    return ""


def _field_assignments(instructions: Iterable[classparse.Instruction]) -> dict[str, int]:
    fields: dict[str, int] = {}
    pending_int: int | None = None
    wanted = {"baseDamage", "baseBlock", "baseMagicNumber", "exhaust", "isMultiDamage"}
    for ins in instructions:
        if _is_int_push(ins):
            pending_int = int(ins.value)
            continue
        if ins.opname == "putfield" and isinstance(ins.value, tuple):
            _cls, name, _desc = ins.value
            if name in wanted and pending_int is not None:
                fields[name] = pending_int
            pending_int = None
            continue
        if ins.opname not in {"aload_0", "aload_1", "dup"}:
            pending_int = None
    return fields


def _upgrade_deltas(class_file: classparse.ClassFile) -> dict[str, int | None]:
    method = _method(class_file, "upgrade", "()V")
    deltas: dict[str, int | None] = {"damage": None, "block": None, "magic": None}
    if method is None or method.code is None:
        return deltas

    pending_int: int | None = None
    for ins in classparse.iter_instructions(class_file.pool, method.code):
        if _is_int_push(ins):
            pending_int = int(ins.value)
            continue
        if ins.opname == "invokevirtual" and isinstance(ins.value, tuple):
            _cls, name, _desc = ins.value
            if pending_int is not None:
                if name == "upgradeDamage":
                    deltas["damage"] = pending_int
                elif name == "upgradeBlock":
                    deltas["block"] = pending_int
                elif name == "upgradeMagicNumber":
                    deltas["magic"] = pending_int
            pending_int = None
            continue
        if ins.opname not in {"aload_0", "aload_1", "dup"}:
            pending_int = None
    return deltas


def _is_int_push(ins: classparse.Instruction) -> bool:
    return isinstance(ins.value, int) and ins.opname in {
        "iconst_m1",
        "iconst_0",
        "iconst_1",
        "iconst_2",
        "iconst_3",
        "iconst_4",
        "iconst_5",
        "bipush",
        "sipush",
        "ldc",
        "ldc_w",
    }


_SIMPLE_DAMAGE_RE = re.compile(r"^Deal !D! damage(?: to ALL enemies)?$", re.IGNORECASE)
_SIMPLE_BLOCK_RE = re.compile(r"^Gain !B! Block$", re.IGNORECASE)


def _simple_numeric_effect(data: CardData, description: str) -> bool:
    if data.cost == -1:  # X-cost cards depend on chosen energy.
        return False
    if data.damage <= 0 and data.block <= 0:
        return False
    if not description.strip():
        return True

    clauses = [
        part.strip().rstrip(".")
        for part in re.split(r"\s+NL\s+|(?<=\.)\s+", description.strip())
        if part.strip().rstrip(".")
    ]
    has_damage = False
    has_block = False
    for clause in clauses:
        if _SIMPLE_DAMAGE_RE.match(clause):
            has_damage = True
        elif _SIMPLE_BLOCK_RE.match(clause):
            has_block = True
        else:
            return False
    return (data.damage <= 0 or has_damage) and (data.block <= 0 or has_block)


def _load_card_localization(zf: zipfile.ZipFile, lang: str) -> dict:
    try:
        return json.loads(zf.read(_LOC.format(lang=lang)).decode("utf-8-sig"))
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _list_card_classes(names: Iterable[str]) -> list[str]:
    return [
        name
        for name in names
        if name.startswith(_CARD_PKG) and name.endswith(".class") and "$" not in name
        and "/deprecated/" not in name.lower()
    ]


def _index_card_entries(entries: Iterable[tuple[str, CardData]]) -> dict[str, CardData]:
    out: dict[str, CardData] = {}
    for display_name, data in entries:
        named = replace(data, name=data.name or display_name)
        for alias in {display_name, named.name, named.card_id}:
            key = _norm(alias)
            if key:
                out.setdefault(key, named)
    return out


# Keyed by display name; built into a normalized lookup map below.
_CURATED_CARDS: list[tuple[str, CardData]] = [
    # ---- starters (all classes share Strike / Defend) -------------------
    ("Strike", CardData("Strike", damage=6, target=True, damage_up=9)),
    ("Defend", CardData("Defend", block=5, block_up=8)),
    ("Bash", CardData("Bash", damage=8, target=True, damage_up=10)),  # Ironclad
    ("Survivor", CardData("Survivor", block=8, block_up=11)),  # Silent
    ("Neutralize", CardData("Neutralize", damage=3, target=True, damage_up=4)),  # Silent
    ("Zap", CardData("Zap")),  # Defect: channels Lightning, no hp effect to predict
    ("Dualcast", CardData("Dualcast")),  # Defect
    ("Eruption", CardData("Eruption", damage=9, target=True)),  # Watcher
    ("Vigilance", CardData("Vigilance", block=8, block_up=12)),  # Watcher
    # ---- common Ironclad attacks / blocks -------------------------------
    ("Pommel Strike", CardData("Pommel Strike", damage=9, target=True, damage_up=10)),
    ("Twin Strike", CardData("Twin Strike", damage=5, hits=2, target=True, damage_up=7)),
    ("Clothesline", CardData("Clothesline", damage=12, target=True, damage_up=14)),
    ("Cleave", CardData("Cleave", damage=8, aoe=True, damage_up=11)),
    ("Thunderclap", CardData("Thunderclap", damage=4, aoe=True, damage_up=7)),
    ("Iron Wave", CardData("Iron Wave", damage=5, block=5, target=True, damage_up=7, block_up=7)),
    ("Sword Boomerang", CardData("Sword Boomerang", damage=3, hits=3, aoe=False)),
    ("Anger", CardData("Anger", damage=6, target=True, damage_up=8)),
    ("Headbutt", CardData("Headbutt", damage=9, target=True, damage_up=12)),
    ("Heavy Blade", CardData("Heavy Blade", damage=14, target=True, damage_up=14)),
    ("Shrug It Off", CardData("Shrug It Off", block=8, block_up=11)),
    # ---- common Silent attacks / blocks ---------------------------------
    ("Dagger Throw", CardData("Dagger Throw", damage=9, target=True, damage_up=12)),
    ("Quick Slash", CardData("Quick Slash", damage=8, target=True, damage_up=12)),
    ("Slice", CardData("Slice", damage=6, target=True, damage_up=9)),
    ("Sucker Punch", CardData("Sucker Punch", damage=7, target=True, damage_up=9)),
    ("Dash", CardData("Dash", damage=10, block=10, target=True, damage_up=13, block_up=13)),
    ("Backflip", CardData("Backflip", block=5, block_up=8)),
    # ---- common Defect / Watcher hp-affecting cards ---------------------
    ("Strike Defect", CardData("Strike", damage=6, target=True, damage_up=9)),
    ("Strike Watcher", CardData("Strike", damage=6, target=True, damage_up=9)),
    ("Cut Through Fate", CardData("Cut Through Fate", damage=7, target=True, damage_up=9)),
]

CARD_DB: dict[str, CardData] = _index_card_entries(
    (name, replace(data, name=data.name or name, source="curated")) for name, data in _CURATED_CARDS
)
