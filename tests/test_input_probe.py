import pytest

from tspire.host.input_probe import (
    FAIL,
    INCONCLUSIVE,
    PASS,
    classify_result,
    parse_sequence,
)


def test_parse_sequence_normalizes_aliases():
    assert parse_sequence("a, ls_left right end-turn") == ["select", "left", "right", "end_turn"]


def test_parse_sequence_rejects_unknown_token():
    with pytest.raises(ValueError):
        parse_sequence("left nope")


def test_parse_sequence_rejects_empty():
    with pytest.raises(ValueError):
        parse_sequence("   ")


def test_classify_result_pass_when_focus_moves():
    verdict, _ = classify_result([0, 0, 1, 0])
    assert verdict == PASS


def test_classify_result_fail_when_focus_never_seen():
    verdict, message = classify_result([None, None, None])
    assert verdict == FAIL
    assert "Controller Enabled" in message


def test_classify_result_inconclusive_when_focus_static():
    verdict, message = classify_result([2, 2, 2])
    assert verdict == INCONCLUSIVE
    assert "index 2" in message
