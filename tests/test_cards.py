from tspire.host import cards
from tspire.host.vision import classparse


def _u2(value: int) -> bytes:
    return bytes([(value >> 8) & 0xFF, value & 0xFF])


def _push_int(value: int) -> bytes:
    if -1 <= value <= 5:
        return bytes([0x03 + value])
    if -128 <= value <= 127:
        return bytes([0x10, value & 0xFF])
    return bytes([0x11, (value >> 8) & 0xFF, value & 0xFF])


class _Pool:
    def __init__(self):
        self.items = [None]

    def add(self, item):
        self.items.append(item)
        return len(self.items) - 1

    def utf(self, value: str) -> int:
        return self.add(("Utf8", value))

    def cls(self, name: str) -> int:
        return self.add(("Class", self.utf(name)))

    def string(self, value: str) -> int:
        return self.add(("String", self.utf(value)))

    def name_type(self, name: str, desc: str) -> int:
        return self.add((12, (self.utf(name), self.utf(desc))))

    def field(self, owner: str, name: str, desc: str = "I") -> int:
        return self.add((9, (self.cls(owner), self.name_type(name, desc))))

    def method(self, owner: str, name: str, desc: str) -> int:
        return self.add((10, (self.cls(owner), self.name_type(name, desc))))

    def enum(self, owner: str, member: str) -> int:
        return self.field(owner, member, "Lx;")


def _fake_card_class(
    *,
    card_id: str = "Runtime Strike",
    cost: int = 1,
    card_type: str = "ATTACK",
    target: str = "ENEMY",
    base_damage: int = 7,
    base_block: int = 0,
    base_magic: int = 2,
    exhaust: bool = True,
    is_multi_damage: bool = False,
    upgrade_damage: int | None = 3,
    upgrade_block: int | None = None,
    upgrade_magic: int | None = 1,
) -> classparse.ClassFile:
    pool = _Pool()
    owner = "com/megacrit/cardcrawl/cards/blue/RuntimeStrike"
    abstract = "com/megacrit/cardcrawl/cards/AbstractCard"
    init_ref = pool.method(
        abstract,
        "<init>",
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;ILjava/lang/String;"
        "Lcom/megacrit/cardcrawl/cards/AbstractCard$CardType;"
        "Lcom/megacrit/cardcrawl/cards/AbstractCard$CardColor;"
        "Lcom/megacrit/cardcrawl/cards/AbstractCard$CardRarity;"
        "Lcom/megacrit/cardcrawl/cards/AbstractCard$CardTarget;)V",
    )
    type_ref = pool.enum("com/megacrit/cardcrawl/cards/AbstractCard$CardType", card_type)
    color_ref = pool.enum("com/megacrit/cardcrawl/cards/AbstractCard$CardColor", "BLUE")
    rarity_ref = pool.enum("com/megacrit/cardcrawl/cards/AbstractCard$CardRarity", "COMMON")
    target_ref = pool.enum("com/megacrit/cardcrawl/cards/AbstractCard$CardTarget", target)
    id_ref = pool.string(card_id)
    fields = {
        "baseDamage": pool.field(owner, "baseDamage"),
        "baseBlock": pool.field(owner, "baseBlock"),
        "baseMagicNumber": pool.field(owner, "baseMagicNumber"),
        "exhaust": pool.field(owner, "exhaust", "Z"),
        "isMultiDamage": pool.field(owner, "isMultiDamage", "Z"),
    }

    init = bytearray([0x2A, 0x12, id_ref])
    init += _push_int(cost)
    for ref in (type_ref, color_ref, rarity_ref, target_ref):
        init += bytes([0xB2]) + _u2(ref)
    init += bytes([0xB7]) + _u2(init_ref)
    for name, value in (
        ("baseDamage", base_damage),
        ("baseBlock", base_block),
        ("baseMagicNumber", base_magic),
        ("exhaust", int(exhaust)),
        ("isMultiDamage", int(is_multi_damage)),
    ):
        if value:
            init += bytes([0x2A]) + _push_int(value) + bytes([0xB5]) + _u2(fields[name])
    init += bytes([0xB1])

    upgrade = bytearray()
    for method_name, value in (
        ("upgradeDamage", upgrade_damage),
        ("upgradeBlock", upgrade_block),
        ("upgradeMagicNumber", upgrade_magic),
    ):
        if value is None:
            continue
        upgrade += bytes([0x2A]) + _push_int(value)
        upgrade += bytes([0xB6]) + _u2(pool.method(owner, method_name, "(I)V"))
    upgrade += bytes([0xB1])

    return classparse.ClassFile(
        pool=pool.items,
        methods=[
            classparse.MethodInfo("<init>", "()V", bytes(init)),
            classparse.MethodInfo("upgrade", "()V", bytes(upgrade)),
        ],
    )


def test_build_card_table_extracts_simple_runtime_card():
    loc = {"Runtime Strike": {"NAME": "Runtime Strike", "DESCRIPTION": "Deal !D! damage."}}
    table = cards.build_card_table(
        reader=lambda _resource: _fake_card_class(),
        loc=loc,
        classes=["com/megacrit/cardcrawl/cards/blue/RuntimeStrike.class"],
    )

    data = table[cards._norm("Runtime Strike")]

    assert data.card_id == "Runtime Strike"
    assert data.cost == 1
    assert data.type == "ATTACK"
    assert data.color == "BLUE"
    assert data.rarity == "COMMON"
    assert data.target_type == "ENEMY"
    assert data.damage == 7
    assert data.damage_up == 10
    assert data.magic == 2
    assert data.magic_up == 3
    assert data.exhausts is True
    assert data.target is True
    assert data.predictable is True


def test_build_card_table_extracts_block_and_aoe_flags():
    loc = {"Runtime Guard": {"NAME": "Runtime Guard", "DESCRIPTION": "Gain !B! Block."}}
    table = cards.build_card_table(
        reader=lambda _resource: _fake_card_class(
            card_id="Runtime Guard",
            card_type="SKILL",
            target="ALL_ENEMY",
            base_damage=0,
            base_block=8,
            base_magic=0,
            exhaust=False,
            is_multi_damage=True,
            upgrade_damage=None,
            upgrade_block=3,
            upgrade_magic=None,
        ),
        loc=loc,
        classes=["com/megacrit/cardcrawl/cards/blue/RuntimeGuard.class"],
    )

    data = table[cards._norm("Runtime Guard")]

    assert data.block == 8
    assert data.block_up == 11
    assert data.aoe is True
    assert data.exhausts is False
    assert data.predictable is True


def test_build_card_table_keeps_dynamic_cards_unpredictable():
    loc = {"Runtime Draw": {"NAME": "Runtime Draw", "DESCRIPTION": "Deal !D! damage. NL Draw 1 card."}}
    table = cards.build_card_table(
        reader=lambda _resource: _fake_card_class(card_id="Runtime Draw"),
        loc=loc,
        classes=["com/megacrit/cardcrawl/cards/blue/RuntimeDraw.class"],
    )

    assert table[cards._norm("Runtime Draw")].predictable is False


def test_lookup_uses_runtime_table_and_skips_unpredictable(monkeypatch):
    runtime = {
        cards._norm("Runtime Strike"): cards.CardData(
            "Runtime Strike", name="Runtime Strike", damage=7, target=True, predictable=True, source="jar"
        ),
        cards._norm("Runtime Draw"): cards.CardData(
            "Runtime Draw", name="Runtime Draw", damage=7, target=True, predictable=False, source="jar"
        ),
    }
    monkeypatch.setattr(cards, "_runtime_card_db", lambda: runtime)

    assert cards.lookup("Runtime Strike").damage == 7
    assert cards.lookup("Runtime Draw") is None
