import struct

from tspire.host.vision import classparse


def _cp_entry(tag, value):
    out = bytearray([tag])
    if tag == 1:
        raw = value.encode()
        out += struct.pack(">H", len(raw)) + raw
    elif tag == 7:
        out += struct.pack(">H", value)
    else:
        raise AssertionError(f"unsupported test cp tag {tag}")
    return out


def _minimal_class_with_method(code: bytes, *, name: str = "upgrade", desc: str = "()V") -> bytes:
    pool = [
        (1, "Test"),  # 1
        (7, 1),       # 2 this class
        (1, "java/lang/Object"),  # 3
        (7, 3),       # 4 super
        (1, name),    # 5 method name
        (1, desc),    # 6 method desc
        (1, "Code"),  # 7
    ]
    out = bytearray(b"\xca\xfe\xba\xbe")
    out += struct.pack(">HHH", 0, 52, len(pool) + 1)
    for entry in pool:
        out += _cp_entry(*entry)
    out += struct.pack(">HHH", 0x0021, 2, 4)  # access, this_class, super_class
    out += struct.pack(">H", 0)  # interfaces
    out += struct.pack(">H", 0)  # fields
    out += struct.pack(">H", 1)  # methods
    out += struct.pack(">HHHH", 0x0001, 5, 6, 1)
    code_attr = struct.pack(">HHI", 2, 1, len(code)) + code + struct.pack(">HH", 0, 0)
    out += struct.pack(">HI", 7, len(code_attr)) + code_attr
    out += struct.pack(">H", 0)  # class attrs
    return bytes(out)


def test_parse_class_file_reads_method_code():
    raw = _minimal_class_with_method(bytes([0x04, 0xB1]))  # iconst_1; return

    parsed = classparse.parse_class_file(raw)

    assert len(parsed.methods) == 1
    assert parsed.methods[0].name == "upgrade"
    assert parsed.methods[0].code == bytes([0x04, 0xB1])


def test_iter_instructions_decodes_integer_pushes():
    parsed = classparse.parse_class_file(_minimal_class_with_method(bytes([0x02, 0x10, 0x7F, 0xB1])))

    instructions = list(classparse.iter_instructions(parsed.pool, parsed.methods[0].code))

    assert [(ins.opname, ins.value) for ins in instructions[:2]] == [
        ("iconst_m1", -1),
        ("bipush", 127),
    ]
