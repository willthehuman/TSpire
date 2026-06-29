from tspire.host.input.driver import DryRunDriver, InputTiming, _AXES, normalize_token


def test_normalize_token_aliases():
    assert normalize_token("A") == "select"
    assert normalize_token("back") == "cancel"
    assert normalize_token("end-turn") == "proceed"
    assert normalize_token("ls_left") == "left"


def test_normalize_token_rejects_unknown():
    try:
        normalize_token("nope")
    except ValueError as exc:
        assert "unknown gamepad token" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_dry_run_driver_records_normalized_tokens():
    driver = DryRunDriver(InputTiming())
    driver.press("A")
    driver.press("right")
    driver.press("end_turn")
    assert driver.presses == ["select", "right", "proceed"]


def test_left_stick_axis_map_uses_xinput_y_convention():
    assert _AXES["up"] == ("left", 0, 32767)
    assert _AXES["down"] == ("left", 0, -32768)
