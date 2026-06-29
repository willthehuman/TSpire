"""Minimal Java .class constant-pool reader (no decompiler needed).

We use this to read enum field references straight from the game's compiled classes —
e.g. each potion's ``PotionSize`` (flask shape) and ``PotionColor`` are encoded as
``getstatic AbstractPotion$PotionSize.X`` references, which appear in the class constant
pool as field references. That gives us authoritative, version-synced metadata without
decompiling or bundling anything.

Only the constant pool is parsed (UTF-8 strings, Integer/Float/Long/Double constants, and
Class/String/Fieldref/Methodref/NameAndType references). Enough to read enum members and
inline int/float constants from a class.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path


def _read_cp(data: bytes) -> list:
    if data[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("not a .class file (bad magic)")
    idx = 8  # skip magic(4) + minor(2) + major(2)
    cp_count = struct.unpack_from(">H", data, idx)[0]
    idx += 2
    pool: list = [None] * cp_count  # constant pool is 1-indexed
    i = 1
    while i < cp_count:
        tag = data[idx]
        idx += 1
        if tag == 1:  # Utf8
            ln = struct.unpack_from(">H", data, idx)[0]
            idx += 2
            pool[i] = ("Utf8", data[idx:idx + ln].decode("utf-8", "replace"))
            idx += ln
        elif tag == 3:  # Integer
            pool[i] = ("Integer", struct.unpack_from(">i", data, idx)[0])
            idx += 4
        elif tag == 4:  # Float
            pool[i] = ("Float", struct.unpack_from(">f", data, idx)[0])
            idx += 4
        elif tag in (5, 6):  # Long / Double — occupy two slots
            pool[i] = ("Long" if tag == 5 else "Double",
                       struct.unpack_from(">q" if tag == 5 else ">d", data, idx)[0])
            idx += 8
            i += 1
        elif tag == 7:  # Class
            pool[i] = ("Class", struct.unpack_from(">H", data, idx)[0])
            idx += 2
        elif tag == 8:  # String
            pool[i] = ("String", struct.unpack_from(">H", data, idx)[0])
            idx += 2
        elif tag in (9, 10, 11, 12):  # Fieldref/Methodref/IfaceMethodref/NameAndType
            pool[i] = (tag, struct.unpack_from(">HH", data, idx))
            idx += 4
        elif tag == 15:  # MethodHandle
            idx += 3
        elif tag == 16:  # MethodType
            idx += 2
        elif tag == 18:  # InvokeDynamic
            idx += 4
        else:
            raise ValueError(f"unknown constant-pool tag {tag} at slot {i}")
        i += 1
    return pool


def _utf(pool, i):
    return pool[i][1] if i and pool[i] and pool[i][0] == "Utf8" else None


def _class_name(pool, i):
    return _utf(pool, pool[i][1]) if i and pool[i] and pool[i][0] == "Class" else None


def field_refs(pool) -> list[tuple[str, str, str]]:
    """All Fieldref/Methodref/IfaceMethodref in the pool as (class, name, descriptor)."""
    out = []
    for e in pool:
        if e and e[0] in (9, 10, 11):
            cls = _class_name(pool, e[1][0])
            nt = pool[e[1][1]]
            if nt and nt[0] == 12:
                out.append((cls, _utf(pool, nt[1][0]), _utf(pool, nt[1][1])))
    return out


def parse_class(data: bytes) -> list:
    """Parse a .class file's bytes; return its constant pool."""
    return _read_cp(data)


def enum_field_args(pool, enum_owner_suffix: str) -> set[str]:
    """Field-reference names whose owning class endswith `enum_owner_suffix`.

    E.g. enum_field_args(pool, 'AbstractPotion$PotionColor') -> {'FRUIT','FIRE',...}
    (the color/size enum members referenced by the class being inspected).
    """
    return {
        name
        for cls, name, _desc in field_refs(pool)
        if cls and cls.split("/")[-1].endswith(enum_owner_suffix)
    }


def read_class(jar: Path | str, class_resource: str) -> list:
    """Read a class from a jar/zip by its resource path (e.g. 'com/x/Foo.class')."""
    with zipfile.ZipFile(jar) as zf:
        return parse_class(zf.read(class_resource))
