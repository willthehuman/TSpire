"""Potion metadata, read authoritatively from the installed game's jar.

Each potion's flask SHAPE and liquid COLOR category are encoded in its compiled class as
``getstatic AbstractPotion$PotionSize.X`` / ``AbstractPotion$PotionColor.Y`` references —
we read those straight from the class constant pools (see classparse.py), no decompilation.
Names come from the game's localization JSON. Nothing is bundled.

Why this table and not a from-scratch identifier: belt-scale potion *identification* from a
~25px icon is not reliably achievable by CV (the flask shapes are genuinely similar and the
icons tiny) or by a vision model (it confuses the game's shape vocabulary, e.g. "bottle" vs
"heart"). The robust identification path is the focus-cursor tooltip (M3): focusing a potion
makes the game render its NAME as text, which is trivially readable. This table then resolves
that name/id to its shape, colour category and (via the rest of the DB) effects.

The table is still directly useful: client UI display, command targeting, and as the lookup a
best-effort matcher scores against (see PotionSuggester, documented as low-confidence).
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from tspire.host.vision import classparse

_POTIONS_PKG = "com/megacrit/cardcrawl/potions/"
_LOC = "localization/eng/potions.json"
# Classes in the potions package that aren't real potions.
_SKIP_CLASSES = {"AbstractPotion", "PotionHelper", "PotionSlot"}

# PotionColor enum members seen in this jar (for reference / validation).
COLOR_CATEGORIES = (
    "POISON", "BLUE", "FIRE", "GREEN", "EXPLOSIVE", "WEAK", "FEAR", "STRENGTH",
    "WHITE", "FAIRY", "ANCIENT", "ELIXIR", "NONE", "ENERGY", "SWIFT", "FRUIT",
    "SNECKO", "SMOKE", "STEROID", "SKILL", "ATTACK", "POWER",
)
# The flask shape folders shipped in the jar (PotionSize also has legacy M/S/T/H sizes,
# which are NOT shapes). Used to validate a PotionSize member is a real flask folder.
FLASK_SHAPES = frozenset({
    "anvil", "bolt", "bottle", "card", "eye", "fairy", "ghost", "heart",
    "jar", "moon", "snecko", "sphere", "spiky",
})


@dataclass
class PotionMeta:
    id: str               # canonical id (class stem), e.g. "FruitJuice"
    potion_id: str = ""   # the game's POTION_ID string (sometimes has spaces), e.g. "Fruit Juice"
    name: str = ""        # display name from localization
    shape: str = ""       # flask shape folder, e.g. "heart"
    color: str = ""       # PotionColor category, e.g. "FRUIT" (empty for new-API potions)
    rarity: str = ""


ClassReader = Callable[[str], list]
"""Reads a class resource path from a jar -> its constant pool."""


def _default_reader(jar: Path | str) -> ClassReader:
    zf = zipfile.ZipFile(jar)

    def read(resource: str) -> list:
        return classparse.parse_class(zf.read(resource))

    return read


def build_potion_table(
    jar: Path | str | None = None,
    *,
    loc_lang: str = "eng",
    reader: ClassReader | None = None,
    loc: dict | None = None,
    classes: list[str] | None = None,
) -> dict[str, PotionMeta]:
    """Build {canonical_id: PotionMeta} from the jar.

    `reader`, `loc`, `classes` are injectable for tests (supply canned data, no jar needed).
    When None, they are derived from `jar`; if `jar` is also None, returns {}.
    """
    if reader is None:
        if jar is None:
            return {}
        reader = _default_reader(jar)
    if loc is None:
        loc = _load_localization(jar, loc_lang) if jar else {}
    if classes is None:
        classes = _list_classes(jar) if jar else []
    # Validate flask shapes against the jar's folders when available, else the known set.
    valid_shapes = _shape_dirs_zip(jar) if jar else set(FLASK_SHAPES)

    table: dict[str, PotionMeta] = {}
    for resource in classes:
        stem = resource.split("/")[-1][:-6]
        if stem in _SKIP_CLASSES or "$" in resource:
            continue
        pool = reader(resource)
        shape_member = _one(classparse.enum_field_args(pool, "$PotionSize"))
        color = _one(classparse.enum_field_args(pool, "$PotionColor"))
        rarity = _one(classparse.enum_field_args(pool, "$PotionRarity"))
        potion_id = _find_potion_id(pool, loc)
        # PotionSize mixes flask shapes (HEART, ANVIL...) with legacy sizes (M/S/T/H).
        # Only keep it as a shape if a matching image folder exists.
        shape = shape_member.lower() if shape_member and shape_member.lower() in valid_shapes else ""
        meta = PotionMeta(
            id=stem,
            potion_id=potion_id,
            name=loc.get(potion_id, {}).get("NAME", "") if potion_id else "",
            shape=shape,
            color=color,
            rarity=rarity,
        )
        table[stem] = meta
    return table


def _shape_dirs_zip(jar) -> set[str]:
    out = set()
    with zipfile.ZipFile(jar) as zf:
        for n in zf.namelist():
            parts = n.split("/")
            if len(parts) == 4 and parts[0] == "images" and parts[1] == "potion":
                out.add(parts[2])
    return out


def _one(items: set[str]) -> str:
    items = {i for i in items if i != "$VALUES"}
    return next(iter(items), "")


def _find_potion_id(pool, loc: dict) -> str:
    """The POTION_ID is the class's String constant that is also a localization key."""
    loc_keys = set(loc)
    for e in pool:
        if e and e[0] == "Utf8" and e[1] in loc_keys:
            return e[1]
    return ""


def _load_localization(jar, lang: str) -> dict:
    try:
        with zipfile.ZipFile(jar) as zf:
            return json.loads(zf.read(f"localization/{lang}/potions.json"))
    except (FileNotFoundError, KeyError, zipfile.BadZipfile, json.JSONDecodeError):
        return {}


def _list_classes(jar) -> list[str]:
    with zipfile.ZipFile(jar) as zf:
        return _jar_classes_from_iter(zf.namelist())


def _jar_classes_from_reader(jar) -> list[str]:
    with zipfile.ZipFile(jar) as zf:
        return _jar_classes_from_iter(zf.namelist())


def _jar_classes_from_iter(names) -> list[str]:
    return [n for n in names if n.startswith(_POTIONS_PKG) and n.endswith(".class")]


# --------------------------------------------------------------------------- #
# Best-effort matcher (LOW CONFIDENCE — see module docstring)
# --------------------------------------------------------------------------- #
@dataclass
class PotionSuggester:
    """Scores a belt crop against the table by shape (Hu contour distance) + hue.

    WARNING: unreliable at belt scale (~25px) — flask shapes are too similar and icons too
    small for confident matching. Treat results as rough hints only; authoritative ID uses
    the M3 focus-tooltip. Provided so a UI can show a low-confidence guess.
    """

    table: dict[str, PotionMeta]
    jar: Path | str | None = None
    _shape_contours: dict = field(default_factory=dict, init=False, repr=False)

    def _shapes(self):
        if not self._shape_contours and self.jar:
            self._shape_contours = _load_shape_contours(self.jar)
        return self._shape_contours

    def suggest(self, crop, k: int = 3) -> list[tuple[str, float]]:
        """Return [(potion_id, score), ...] ranked; scores are relative, not calibrated."""
        shapes = self._shapes()
        if not shapes:
            return []
        crop_cnt = _crop_contour(crop)
        if crop_cnt is None:
            return []
        scored = []
        for pid, meta in self.table.items():
            shape_d = float(cv2.matchShapes(crop_cnt, shapes[meta.shape],
                                             cv2.CONTOURS_MATCH_I2, 0)) if meta.shape in shapes else 9.0
            scored.append((pid, -shape_d))  # closer shape => higher score
        return sorted(scored, key=lambda s: s[1], reverse=True)[:k]


def _crop_contour(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 40) | (hsv[:, :, 2] > 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea) if cnts else None


def _load_shape_contours(jar) -> dict:
    out = {}
    with zipfile.ZipFile(jar) as zf:
        for shape in _shape_dirs(zf):
            try:
                data = zf.read(f"images/potion/{shape}/liquid.png")
            except KeyError:
                continue
            a = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            src = a[:, :, 3] if a.ndim == 3 and a.shape[2] == 4 and a[:, :, 3].max() > 0 \
                else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            binary = (src > 20).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                out[shape] = max(cnts, key=cv2.contourArea)
    return out


def _shape_dirs(zf) -> list[str]:
    shapes = set()
    for n in zf.namelist():
        parts = n.split("/")
        if len(parts) == 4 and parts[0] == "images" and parts[1] == "potion":
            shapes.add(parts[2])
    return sorted(shapes)


if __name__ == "__main__":
    import argparse
    import sys

    from tspire.host.game_assets import find_game_jar

    ap = argparse.ArgumentParser(description="Dump the jar-derived potion metadata table")
    ap.add_argument("--jar", help="path to desktop-1.0.jar", default=None)
    args = ap.parse_args()
    jar = find_game_jar(args.jar)
    if not jar:
        sys.exit("desktop-1.0.jar not found")
    table = build_potion_table(jar)
    print(f"# {len(table)} potions from {jar}\n")
    for mid in sorted(table, key=lambda k: (table[k].shape, k)):
        m = table[mid]
        print(f"{m.id:22s} shape={m.shape or '-':7s} color={m.color or '-':8s} "
              f"name={m.name!r}")
