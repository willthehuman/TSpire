from tspire.host.config import HostConfig
from tspire.host.input.driver import (
    DryRunDriver,
    InputTiming,
    KeyboardDriver,
    _AXES,
    build_driver,
    normalize_token,
)


def test_normalize_token_aliases():
    assert normalize_token("A") == "select"
    assert normalize_token("back") == "cancel"
    assert normalize_token("end-turn") == "end_turn"
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
    assert driver.presses == ["select", "right", "end_turn"]


def test_keyboard_driver_maps_semantic_tokens_to_keys():
    class FakeUser32:
        def __init__(self):
            self.events = []

        def keybd_event(self, vk, scan, flags, extra):
            self.events.append((vk, flags))

    fake = FakeUser32()
    driver = object.__new__(KeyboardDriver)
    driver.timing = InputTiming(press_seconds=0.0, step_delay=0.0)
    driver._user32 = fake

    driver.press("end_turn")
    driver.press("select")
    driver.press("cancel")
    driver.press("left")

    assert fake.events == [
        (0x45, 0),
        (0x45, 0x0002),
        (0x0D, 0),
        (0x0D, 0x0002),
        (0x1B, 0),
        (0x1B, 0x0002),
        (0x25, 0),
        (0x25, 0x0002),
    ]


def test_dry_run_overrides_real_backend_choice():
    cfg = HostConfig(input_dry_run=True, input_backend="gamepad")
    assert isinstance(build_driver(cfg, InputTiming()), DryRunDriver)


def test_left_stick_axis_map_uses_xinput_y_convention():
    assert _AXES["up"] == ("left", 0, 32767)
    assert _AXES["down"] == ("left", 0, -32768)
