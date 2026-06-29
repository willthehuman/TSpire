"""Build the template database from Slay the Spire's own art.

The game ships its art inside ``desktop-1.0.jar`` (a zip). This copies the static PNGs we
match against into ``tspire/assets/templates/<category>/<id>.png``.

Usage:
    python -m tools.extract_assets --jar "C:/path/to/SlayTheSpire/desktop-1.0.jar"
    python -m tools.extract_assets --jar <jar> --out tspire/assets/templates

Categories and the jar paths they map from (the jar layout is stable but verify with
``--list`` if a category comes up empty):

  intents  <- images/ui/intents/*.png        (attack/defend/buff/... icons)
  relics   <- images/relics/*.png            (relic icons)
  potions  <- images/potions/*.png           (potion icons)
  cards    <- images/1024Portraits/**/*.png  (card art; large, will be downscaled at match)

NOTE: monsters use skeletal-animation atlases, not single PNGs, so there is no clean
per-monster image to extract. Monster templates are best captured from real screenshots
later (drop cropped PNGs into templates/monsters/). This tool skips them.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

# category -> list of jar path prefixes that belong to it.
_CATEGORY_PREFIXES: dict[str, tuple[str, ...]] = {
    "intents": ("images/ui/intents/",),
    "relics": ("images/relics/",),
    "potions": ("images/potions/",),
    "cards": ("images/1024Portraits/",),
}

_DEFAULT_OUT = Path(__file__).resolve().parent.parent / "tspire" / "assets" / "templates"


def _category_for(name: str) -> str | None:
    lower = name.lower()
    if not lower.endswith(".png"):
        return None
    for category, prefixes in _CATEGORY_PREFIXES.items():
        if any(lower.startswith(p) for p in prefixes):
            return category
    return None


def extract(jar_path: Path, out_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {c: 0 for c in _CATEGORY_PREFIXES}
    with zipfile.ZipFile(jar_path) as jar:
        for name in jar.namelist():
            category = _category_for(name)
            if category is None:
                continue
            target_dir = out_dir / category
            target_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(name).stem
            (target_dir / f"{stem}.png").write_bytes(jar.read(name))
            counts[category] += 1
    return counts


def list_png_prefixes(jar_path: Path, limit: int = 40) -> list[str]:
    """Print distinct top-level image/ directories (to help verify prefixes)."""
    seen: dict[str, int] = {}
    with zipfile.ZipFile(jar_path) as jar:
        for name in jar.namelist():
            if name.lower().endswith(".png") and name.startswith("images/"):
                key = "/".join(name.split("/")[:3])
                seen[key] = seen.get(key, 0) + 1
    return [f"{k}  ({v})" for k, v in sorted(seen.items())][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract StS art into the template DB")
    parser.add_argument("--jar", required=True, type=Path, help="path to desktop-1.0.jar")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT, help="template DB output dir")
    parser.add_argument("--list", action="store_true", help="list png dirs in the jar and exit")
    args = parser.parse_args()

    if not args.jar.is_file():
        raise SystemExit(f"jar not found: {args.jar}")
    if args.list:
        for line in list_png_prefixes(args.jar):
            print(line)
        return

    counts = extract(args.jar, args.out)
    total = sum(counts.values())
    print(f"extracted {total} images to {args.out}")
    for category, n in counts.items():
        print(f"  {category}: {n}")
    if total == 0:
        print("nothing extracted — run with --list to inspect the jar layout")


if __name__ == "__main__":
    main()
