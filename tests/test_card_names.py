from tspire.host.vision.card_names import CardNameIndex, _normalize


def _idx():
    return CardNameIndex(["Strike", "Defend", "Bash", "Pommel Strike", "A Thousand Cuts", "Iron Wave"])


def test_exact_and_case_insensitive_match():
    idx = _idx()
    assert idx.resolve("Strike")[0] == "Strike"
    assert idx.resolve("strike")[0] == "Strike"
    assert idx.resolve("STRIKE")[1] == 1.0


def test_fuzzy_corrects_ocr_noise():
    idx = _idx()
    assert idx.resolve("Strke")[0] == "Strike"          # dropped letter
    assert idx.resolve("PommelStrike")[0] == "Pommel Strike"  # lost space
    assert idx.resolve("A Thousnd Cuts")[0] == "A Thousand Cuts"


def test_rejects_unrelated_text():
    idx = _idx()
    name, score = idx.resolve("xyzzy garbage")
    assert name == ""      # below threshold -> no false assertion
    assert score < 0.62


def test_empty_and_normalize():
    idx = _idx()
    assert idx.resolve("")[0] == ""
    assert idx.resolve("   ")[0] == ""
    assert _normalize("Pommel Strike!") == "pommelstrike"


def test_empty_index_passes_nothing():
    idx = CardNameIndex([])
    assert len(idx) == 0
    assert idx.resolve("Strike") == ("", 0.0)
