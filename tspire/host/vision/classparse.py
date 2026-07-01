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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class MethodInfo:
    name: str
    descriptor: str
    code: bytes | None = None


@dataclass(frozen=True)
class ClassFile:
    pool: list
    methods: list[MethodInfo]


@dataclass(frozen=True)
class Instruction:
    offset: int
    opcode: int
    opname: str
    operand: int | tuple[int, int] | None = None
    value: Any = None


def _read_cp(data: bytes) -> list:
    pool, _idx = _read_cp_with_offset(data)
    return pool


def _read_cp_with_offset(data: bytes) -> tuple[list, int]:
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
    return pool, idx


def _utf(pool, i):
    return pool[i][1] if i and pool[i] and pool[i][0] == "Utf8" else None


def _class_name(pool, i):
    return _utf(pool, pool[i][1]) if i and pool[i] and pool[i][0] == "Class" else None


def constant_value(pool, i):
    """Resolve a constant-pool slot used by ldc/ldc_w into a Python value."""
    if not i or i >= len(pool):
        return None
    entry = pool[i]
    if entry is None:
        return None
    tag = entry[0]
    if tag in {"Integer", "Float", "Long", "Double", "Utf8"}:
        return entry[1]
    if tag == "String":
        return _utf(pool, entry[1])
    if tag == "Class":
        return _class_name(pool, i)
    return entry


def member_ref(pool, i) -> tuple[str, str, str] | None:
    """Resolve a Fieldref/Methodref/IfaceMethodref slot."""
    if not i or i >= len(pool):
        return None
    entry = pool[i]
    if not entry or entry[0] not in (9, 10, 11):
        return None
    cls = _class_name(pool, entry[1][0])
    nt = pool[entry[1][1]]
    if nt and nt[0] == 12:
        return (cls, _utf(pool, nt[1][0]), _utf(pool, nt[1][1]))
    return None


def field_refs(pool) -> list[tuple[str, str, str]]:
    """All Fieldref/Methodref/IfaceMethodref in the pool as (class, name, descriptor)."""
    out = []
    for i, e in enumerate(pool):
        if e and e[0] in (9, 10, 11):
            ref = member_ref(pool, i)
            if ref is not None:
                out.append(ref)
    return out


def parse_class(data: bytes) -> list:
    """Parse a .class file's bytes; return its constant pool."""
    return _read_cp(data)


def parse_class_file(data: bytes) -> ClassFile:
    """Parse the constant pool plus method Code attributes from a .class file.

    This is intentionally not a decompiler. It extracts only method names/descriptors and
    raw bytecode so callers can scan simple constructor/upgrade patterns.
    """
    pool, idx = _read_cp_with_offset(data)
    idx = _skip_class_header(data, idx)
    idx = _skip_members(data, idx)  # fields
    methods, idx = _read_methods(data, idx, pool)
    return ClassFile(pool=pool, methods=methods)


def _skip_class_header(data: bytes, idx: int) -> int:
    # access_flags, this_class, super_class
    idx += 6
    interfaces_count = struct.unpack_from(">H", data, idx)[0]
    return idx + 2 + interfaces_count * 2


def _skip_members(data: bytes, idx: int) -> int:
    count = struct.unpack_from(">H", data, idx)[0]
    idx += 2
    for _ in range(count):
        idx += 6  # access_flags, name_index, descriptor_index
        attr_count = struct.unpack_from(">H", data, idx)[0]
        idx += 2
        for _ in range(attr_count):
            _name_index = struct.unpack_from(">H", data, idx)[0]
            length = struct.unpack_from(">I", data, idx + 2)[0]
            idx += 6 + length
    return idx


def _read_methods(data: bytes, idx: int, pool: list) -> tuple[list[MethodInfo], int]:
    count = struct.unpack_from(">H", data, idx)[0]
    idx += 2
    methods: list[MethodInfo] = []
    for _ in range(count):
        _access, name_i, desc_i, attr_count = struct.unpack_from(">HHHH", data, idx)
        idx += 8
        code: bytes | None = None
        for _ in range(attr_count):
            attr_name_i = struct.unpack_from(">H", data, idx)[0]
            length = struct.unpack_from(">I", data, idx + 2)[0]
            idx += 6
            if _utf(pool, attr_name_i) == "Code":
                code_len = struct.unpack_from(">I", data, idx + 4)[0]
                code = data[idx + 8 : idx + 8 + code_len]
            idx += length
        methods.append(MethodInfo(name=_utf(pool, name_i) or "", descriptor=_utf(pool, desc_i) or "", code=code))
    return methods, idx


_OPNAME: dict[int, str] = {
    0x00: "nop",
    0x01: "aconst_null",
    0x02: "iconst_m1",
    0x03: "iconst_0",
    0x04: "iconst_1",
    0x05: "iconst_2",
    0x06: "iconst_3",
    0x07: "iconst_4",
    0x08: "iconst_5",
    0x10: "bipush",
    0x11: "sipush",
    0x12: "ldc",
    0x13: "ldc_w",
    0x14: "ldc2_w",
    0x2A: "aload_0",
    0x2B: "aload_1",
    0x59: "dup",
    0x99: "ifeq",
    0x9A: "ifne",
    0xA7: "goto",
    0xAC: "ireturn",
    0xB0: "areturn",
    0xB1: "return",
    0xB2: "getstatic",
    0xB3: "putstatic",
    0xB4: "getfield",
    0xB5: "putfield",
    0xB6: "invokevirtual",
    0xB7: "invokespecial",
    0xB8: "invokestatic",
    0xB9: "invokeinterface",
    0xBA: "invokedynamic",
    0xBB: "new",
}

_FIXED_LENGTHS: dict[int, int] = {
    **{op: 1 for op in range(0x00, 0x10)},
    0x10: 2,
    0x11: 3,
    0x12: 2,
    0x13: 3,
    0x14: 3,
    **{op: 2 for op in range(0x15, 0x1A)},
    **{op: 1 for op in range(0x1A, 0x36)},
    **{op: 2 for op in range(0x36, 0x3B)},
    **{op: 1 for op in range(0x3B, 0x84)},
    0x84: 3,
    **{op: 1 for op in range(0x85, 0x99)},
    **{op: 3 for op in range(0x99, 0xA9)},
    0xA9: 2,
    **{op: 1 for op in range(0xAC, 0xB2)},
    **{op: 3 for op in range(0xB2, 0xB9)},
    0xB9: 5,
    0xBA: 5,
    0xBB: 3,
    0xBC: 2,
    0xBD: 3,
    0xBE: 1,
    0xBF: 1,
    0xC0: 3,
    0xC1: 3,
    0xC2: 1,
    0xC3: 1,
    0xC5: 4,
    0xC6: 3,
    0xC7: 3,
    0xC8: 5,
    0xC9: 5,
}


def iter_instructions(pool: list, code: bytes) -> Iterator[Instruction]:
    """Yield bytecode instructions with simple constants/member refs resolved."""
    pc = 0
    while pc < len(code):
        opcode = code[pc]
        opname = _OPNAME.get(opcode, f"op_{opcode:02x}")
        operand: int | tuple[int, int] | None = None
        value: Any = None
        length = _instruction_length(code, pc)

        if opcode == 0x10:  # bipush
            value = struct.unpack_from(">b", code, pc + 1)[0]
        elif opcode == 0x11:  # sipush
            value = struct.unpack_from(">h", code, pc + 1)[0]
        elif 0x02 <= opcode <= 0x08:
            value = opcode - 0x03
        elif opcode == 0x12:
            operand = code[pc + 1]
            value = constant_value(pool, operand)
        elif opcode in (0x13, 0x14):
            operand = struct.unpack_from(">H", code, pc + 1)[0]
            value = constant_value(pool, operand)
        elif opcode in (0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8):
            operand = struct.unpack_from(">H", code, pc + 1)[0]
            value = member_ref(pool, operand)
        elif opcode in (0xBB, 0xBD, 0xC0, 0xC1):
            operand = struct.unpack_from(">H", code, pc + 1)[0]
            value = _class_name(pool, operand)
        elif length == 3:
            operand = struct.unpack_from(">h", code, pc + 1)[0]

        yield Instruction(pc, opcode, opname, operand, value)
        pc += length


def _instruction_length(code: bytes, pc: int) -> int:
    opcode = code[pc]
    if opcode == 0xAA:  # tableswitch
        idx = pc + 1
        while (idx - pc) % 4:
            idx += 1
        low = struct.unpack_from(">i", code, idx + 4)[0]
        high = struct.unpack_from(">i", code, idx + 8)[0]
        return (idx - pc) + 12 + max(0, high - low + 1) * 4
    if opcode == 0xAB:  # lookupswitch
        idx = pc + 1
        while (idx - pc) % 4:
            idx += 1
        pairs = struct.unpack_from(">i", code, idx + 4)[0]
        return (idx - pc) + 8 + max(0, pairs) * 8
    if opcode == 0xC4:  # wide
        if pc + 1 >= len(code):
            raise ValueError("truncated wide instruction")
        return 6 if code[pc + 1] == 0x84 else 4
    try:
        return _FIXED_LENGTHS[opcode]
    except KeyError as exc:
        raise ValueError(f"unsupported bytecode opcode 0x{opcode:02x} at {pc}") from exc


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


def read_class_file(jar: Path | str, class_resource: str) -> ClassFile:
    """Read a class file from a jar/zip, including method bytecode."""
    with zipfile.ZipFile(jar) as zf:
        return parse_class_file(zf.read(class_resource))
