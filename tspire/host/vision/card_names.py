"""Resolve OCR'd card titles to real card names using the game's own name list.

Slay the Spire ships the complete card name list in ``localization/eng/cards.json`` inside
``desktop-1.0.jar``. We load it once and fuzzy-match OCR output against it, so an imperfect
title read (``"Acrobatlcs"``) still resolves to the real card (``"Acrobatics"``). This turns
"read the stylised title exactly" into "read it approximately and snap to the nearest known
card" -- robust, fast (a string-distance lookup), and needs no card-art templates.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache

log = logging.getLogger("tspire.host.vision.card_names")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


class CardNameIndex:
    """A set of known card names with fuzzy lookup from noisy OCR text."""

    def __init__(self, names: list[str]) -> None:
        # De-duplicate but keep display form; map normalized -> display.
        self._by_norm: dict[str, str] = {}
        for name in names:
            norm = _normalize(name)
            if norm:
                self._by_norm.setdefault(norm, name)

    def __len__(self) -> int:
        return len(self._by_norm)

    @property
    def names(self) -> list[str]:
        return list(self._by_norm.values())

    def resolve(self, text: str, *, min_score: float = 0.62) -> tuple[str, float]:
        """Best matching card name for OCR ``text`` and its similarity score.

        Returns ``("", score)`` when nothing clears ``min_score`` (so the caller can keep the
        raw OCR text or mark the card unknown rather than assert a bad guess).
        """
        query = _normalize(text)
        if not query or not self._by_norm:
            return "", 0.0
        # Exact normalized hit wins immediately.
        if query in self._by_norm:
            return self._by_norm[query], 1.0
        best_name, best_score = "", 0.0
        for norm, display in self._by_norm.items():
            score = SequenceMatcher(None, query, norm).ratio()
            if score > best_score:
                best_name, best_score = display, score
        return (best_name, best_score) if best_score >= min_score else ("", best_score)


def load_card_names(jar_path: str | None) -> list[str]:
    """Read the English card names from the game jar. Returns [] if unavailable."""
    if not jar_path:
        return []
    try:
        import zipfile

        with zipfile.ZipFile(jar_path) as z:
            raw = z.read("localization/eng/cards.json").decode("utf-8-sig")
        data = json.loads(raw)
    except Exception:
        log.debug("could not load card names from jar", exc_info=True)
        return []
    names: list[str] = []
    for entry in data.values():
        if isinstance(entry, dict):
            name = entry.get("NAME")
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


@lru_cache(maxsize=1)
def default_card_index() -> CardNameIndex:
    """Card name index built from the auto-detected game jar (cached)."""
    from tspire.host.game_assets import find_game_jar

    try:
        jar = find_game_jar("")
    except Exception:
        jar = None
    names = load_card_names(jar) if jar else []
    if not names:
        log.warning("no card names loaded; card-name resolution will pass raw OCR text through")
    return CardNameIndex(names)
