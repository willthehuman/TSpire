"""Locate the installed Slay the Spire ``desktop-1.0.jar``.

We read game art from the jar at runtime rather than bundling it, so the project ships no
copyrighted assets and only works when the game is installed. Resolution order:

  1. an explicit path (config ``jar_path``);
  2. ``desktop-1.0.jar`` in the current dir / project root (developer convenience);
  3. every Steam library (parsed from ``libraryfolders.vdf``), plus common install roots.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_REL = Path("steamapps") / "common" / "SlayTheSpire" / "desktop-1.0.jar"


def find_game_jar(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    # 1) explicit
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p

    # 2) dev copy next to the project / cwd
    for p in (Path.cwd() / "desktop-1.0.jar", Path(__file__).resolve().parents[2] / "desktop-1.0.jar"):
        if p.is_file():
            return p

    # 3) Steam libraries
    for lib in _steam_libraries():
        candidate = lib / _REL
        if candidate.is_file():
            return candidate
    return None


def _steam_roots() -> list[Path]:
    roots: list[Path] = []
    # Registry (most reliable on Windows)
    try:
        import winreg

        for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                          (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
            try:
                with winreg.OpenKey(hive, key) as k:
                    val = winreg.QueryValueEx(k, "SteamPath" if hive == winreg.HKEY_CURRENT_USER else "InstallPath")[0]
                    roots.append(Path(val))
            except OSError:
                pass
    except ImportError:
        pass
    # Common locations
    roots += [
        Path(r"C:/Program Files (x86)/Steam"),
        Path(r"C:/Program Files/Steam"),
        Path.home() / ".steam" / "steam",
        Path.home() / ".local" / "share" / "Steam",
    ]
    # De-dupe, keep existing
    seen, out = set(), []
    for r in roots:
        if r and r.exists() and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _steam_libraries() -> list[Path]:
    libs: list[Path] = []
    for root in _steam_roots():
        libs.append(root)  # the root itself is a library
        for vdf in (root / "steamapps" / "libraryfolders.vdf",
                    root / "config" / "libraryfolders.vdf"):
            if vdf.is_file():
                try:
                    text = vdf.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                # entries look like:  "path"   "D:\\SteamLibrary"
                for m in re.finditer(r'"path"\s*"([^"]+)"', text):
                    libs.append(Path(m.group(1).replace("\\\\", "\\")))
    # De-dupe
    seen, out = set(), []
    for lib in libs:
        if lib not in seen:
            seen.add(lib)
            out.append(lib)
    return out
