"""Tests for the Java constant-pool reader and the jar-derived potion metadata table.

cv2/numpy are host deps; skip if absent.
"""
import struct

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from tspire.host.vision import classparse
from tspire.host.vision.potions import PotionMeta, build_potion_table


# --- constant-pool builder (builds a real minimal .class) ------------------
def _class_bytes(pool_entries):
    """Pack a .class with the given constant pool entries (list of (tag, value))."""
    body = bytearray()
    body += b"\xca\xfe\xba\xbe"      # magic
    body += struct.pack(">HH", 0, 52)  # minor, major (Java 8)
    n = len(pool_entries) + 1
    body += struct.pack(">H", n)
    for tag, val in pool_entries:
        body += bytes([tag])
        if tag == 1:  # Utf8
            b = val.encode()
            body += struct.pack(">H", len(b)) + b
        elif tag == 3:  # Integer
            body += struct.pack(">i", val)
        elif tag == 7:  # Class
            body += struct.pack(">H", val)
        elif tag == 12:  # NameAndType
            body += struct.pack(">HH", *val)
        elif tag in (9, 10, 11):  # Fieldref/Methodref/IfaceMethodref
            body += struct.pack(">HH", *val)
    # pad with a no-op so the file is well-formed enough for our parser
    return bytes(body)


def test_parse_class_reads_utf8_and_integer():
    pool = [
        (1, "hello"),       # slot 1
        (3, 42),            # slot 2
    ]
    parsed = classparse.parse_class(_class_bytes(pool))
    assert parsed[1] == ("Utf8", "hello")
    assert parsed[2] == ("Integer", 42)


def test_enum_field_args_extracts_enum_member():
    # pool: [Utf8 "AbstractPotion$PotionSize", Class->1, Utf8 "HEART",
    #        NameAndType->(3,"Lx;"), Fieldref->(2,4)]
    pool = [
        (1, "com/megacrit/cardcrawl/potions/AbstractPotion$PotionSize"),  # 1
        (7, 1),       # 2 Class
        (1, "HEART"),  # 3
        (1, "L...;"),  # 4
        (12, (3, 4)),  # 5 NameAndType(name=3, desc=4)
        (9, (2, 5)),   # 6 Fieldref(class=2, nt=5)
    ]
    parsed = classparse.parse_class(_class_bytes(pool))
    assert "HEART" in classparse.enum_field_args(parsed, "$PotionSize")
    assert classparse.enum_field_args(parsed, "$PotionColor") == set()


# --- potion table builder (injected, no jar) -------------------------------
def _fake_pool(shape_member, color_member, potion_id):
    """A parsed-pool list mimicking classparse output for one potion class."""
    # indices chosen so references resolve
    return [
        None,
        ("Utf8", "com/megacrit/cardcrawl/potions/AbstractPotion$PotionSize"),  # 1
        ("Class", 1),                                                            # 2
        ("Utf8", shape_member),                                                  # 3
        ("Utf8", "Ldesc;"),                                                      # 4
        ("NameAndType", (3, 4)),                                                 # 5  -> wait, format
    ]


def _parsed_pool(shape_member, color_member, potion_id):
    """Build a pool list in the *parsed* (tuple) format classparse returns."""
    return [
        None,
        ("Utf8", "com/megacrit/cardcrawl/potions/AbstractPotion$PotionSize"),  # 1
        ("Class", 1),                                                          # 2
        ("Utf8", shape_member),                                               # 3
        ("Utf8", "Lx;"),                                                      # 4
        (12, (3, 4)),        # NameAndType                                    # 5
        (9, (2, 5)),         # Fieldref PotionSize.<member>                  # 6
        ("Utf8", "com/megacrit/cardcrawl/potions/AbstractPotion$PotionColor"),# 7
        ("Class", 7),                                                          # 8
        ("Utf8", color_member),                                               # 9
        (12, (9, 4)),        # NameAndType                                    # 10
        (9, (8, 10)),        # Fieldref PotionColor.<color>                   # 11
        ("Utf8", potion_id), # the localization-key string                    # 12
    ]


def test_build_table_injected():
    loc = {"Fruit Juice": {"NAME": "Fruit Juice"}, "Fire Potion": {"NAME": "Fire Potion"}}
    classes = ["com/megacrit/cardcrawl/potions/FruitJuice.class",
               "com/megacrit/cardcrawl/potions/FirePotion.class"]
    pools = {
        "FruitJuice": _parsed_pool("HEART", "FRUIT", "Fruit Juice"),
        "FirePotion": _parsed_pool("M", "FIRE", "Fire Potion"),
    }
    reader = lambda res: pools[res.split("/")[-1][:-6]]
    table = build_potion_table(reader=reader, loc=loc, classes=classes)

    fj = table["FruitJuice"]
    assert fj.shape == "heart"           # HEART maps to a real folder
    assert fj.color == "FRUIT"
    assert fj.name == "Fruit Juice"
    assert fj.potion_id == "Fruit Juice"

    fp = table["FirePotion"]
    assert fp.shape == ""                # M is a generic size, not a flask folder
    assert fp.color == "FIRE"


def test_build_table_skips_abstract_and_inner():
    classes = ["com/megacrit/cardcrawl/potions/AbstractPotion.class",
               "com/megacrit/cardcrawl/potions/Foo$1.class",
               "com/megacrit/cardcrawl/potions/Foo.class"]
    table = build_potion_table(
        reader=lambda r: _parsed_pool("BOTTLE", "BLUE", "Foo"),
        loc={"Foo": {"NAME": "Foo"}}, classes=classes,
    )
    assert set(table) == {"Foo"}


def test_build_table_returns_empty_without_jar_or_reader():
    assert build_potion_table(None) == {}


def test_potion_meta_round_trips():
    # sanity: the dataclass is plain and constructible
    m = PotionMeta(id="X", shape="jar", color="BLUE", name="X Potion")
    assert (m.id, m.shape, m.color) == ("X", "jar", "BLUE")
