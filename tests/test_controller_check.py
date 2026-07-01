from tspire.host.input.controller_check import analyze_controllers


def _c(name, i=0):
    return {"id": i, "name": name}


def test_no_controllers_reports_missing_pad():
    msgs = analyze_controllers([])
    assert len(msgs) == 1
    assert "No game controllers" in msgs[0]


def test_single_xbox_pad_is_ok():
    msgs = analyze_controllers([_c("Xbox 360 Controller for Windows")])
    assert len(msgs) == 1
    assert msgs[0].startswith("Controller setup looks OK")


def test_steam_input_controller_is_flagged():
    msgs = analyze_controllers([_c("Steam Virtual Gamepad")])
    text = " ".join(msgs)
    assert "Steam Input" in text
    assert "Disable Steam Input" in text


def test_non_xbox_only_is_flagged_as_missing_virtual_pad():
    msgs = analyze_controllers([_c("Logitech Extreme 3D")])
    assert any("No Xbox/360-style controller" in m for m in msgs)


def test_xbox_pad_not_first_is_flagged():
    msgs = analyze_controllers([
        _c("Logitech Extreme 3D", 0),
        _c("Controller (XBOX 360 For Windows)", 1),
    ])
    text = " ".join(msgs)
    assert "FIRST controller" in text
    assert "2 controllers present" in text


def test_two_xbox_pads_flags_multiple():
    msgs = analyze_controllers([
        _c("Xbox 360 Controller for Windows", 0),
        _c("Xbox 360 Controller for Windows", 1),
    ])
    assert any("2 controllers present" in m for m in msgs)
