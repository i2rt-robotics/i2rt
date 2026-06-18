"""Unit tests for the USB-GPIO driver and the RPi.GPIO-compatible backend.

No hardware is touched: the ``fake_serial`` fixture (conftest) replaces
``serial.Serial`` with an in-memory fake. Edge/debounce behavior is driven via
direct ``_poll_once()`` calls so there is no dependence on the background
polling thread's timing.
"""

import logging
from typing import Any

import pytest

from i2rt.utils import usb_gpio_driver
from i2rt.utils.usb_gpio_driver import PullMode, UsbGpioBackend, UsbToGpio, UsbToGpioError

# Channel map mirroring linear_rail_controller: GPIO5->1, GPIO6->2, GPIO12->3.
UPPER, LOWER, BRAKE = 5, 6, 12
PIN_MAP = {UPPER: 1, LOWER: 2, BRAKE: 3}


def _set_writes(fake: dict[str, Any]) -> list[bytes]:
    """SET_PINS (0x3A) frames written so far across the fixture's serial instances."""
    return [w for inst in fake["instances"] for w in inst.writes if w and w[0] == 0x3A]


# -- UsbToGpio: frame encoding (the assertions the PR description promised) -----


def test_set_pins_frame(fake_serial: dict[str, Any]) -> None:
    dev = UsbToGpio("/dev/ttyFAKE")
    dev.set_pins({3: True})
    assert dev._serial.writes[-1] == bytes.fromhex("3a 03 01")
    dev.set_pins({3: False})
    assert dev._serial.writes[-1] == bytes.fromhex("3a 03 00")


def test_read_inputs_frame_and_parse(fake_serial: dict[str, Any]) -> None:
    fake_serial["levels"][1] = False
    dev = UsbToGpio("/dev/ttyFAKE")
    result = dev.read_inputs(1, 1, PullMode.PULL_UP)
    assert dev._serial.writes[-1] == bytes.fromhex("5b 01 01 00")
    assert result == {1: False}


def test_read_pin_frame_and_parse(fake_serial: dict[str, Any]) -> None:
    fake_serial["levels"][1] = True
    dev = UsbToGpio("/dev/ttyFAKE")
    assert dev.read_pin(1) is True
    assert dev._serial.writes[-1] == bytes.fromhex("3f 01")


# -- UsbToGpio._request: boot-frame resend and error paths ---------------------


def test_request_drops_boot_frame_and_resends(fake_serial: dict[str, Any]) -> None:
    seen = {"boot": False}

    def responder(data: bytes) -> bytes:
        if not seen["boot"]:
            seen["boot"] = True
            return bytes(usb_gpio_driver.BOOT_FRAME)  # arrives in place of the first reply
        return bytes((0x2F, data[1], 1))  # real READ_PIN reply, level high

    fake_serial["responder"] = responder
    dev = UsbToGpio("/dev/ttyFAKE")
    assert dev.read_pin(1) is True
    # The request was sent twice: once that read the boot frame, once for the real reply.
    assert dev._serial.writes == [bytes.fromhex("3f 01"), bytes.fromhex("3f 01")]


def test_request_short_read_raises(fake_serial: dict[str, Any]) -> None:
    fake_serial["responder"] = lambda data: b""  # nothing comes back -> timeout
    dev = UsbToGpio("/dev/ttyFAKE")
    with pytest.raises(UsbToGpioError, match="timed out"):
        dev.read_pin(1)


def test_request_wrong_head_raises(fake_serial: dict[str, Any]) -> None:
    fake_serial["responder"] = lambda data: bytes((0x99, data[1], 0))  # wrong reply head
    dev = UsbToGpio("/dev/ttyFAKE")
    with pytest.raises(UsbToGpioError, match="reply head"):
        dev.read_pin(1)


# -- UsbToGpio.set_pins: 0x2A echo verification --------------------------------


def test_set_pins_raises_on_echo_mismatch(fake_serial: dict[str, Any]) -> None:
    # The board echoes 0x2A + the sent (channel, level) pairs; a mismatching echo
    # (here the level bit is flipped) means the write was not applied as sent.
    fake_serial["responder"] = lambda data: bytes((0x2A, data[1], data[2] ^ 1))
    dev = UsbToGpio("/dev/ttyFAKE")
    with pytest.raises(UsbToGpioError, match="echo"):
        dev.set_pins({3: True})


# -- UsbGpioBackend.output: verify the 0x2A echo, resend once, then warn --------


def test_output_verifies_echo(fake_serial: dict[str, Any]) -> None:
    # A matching 0x2A echo confirms the write on the first try -- exactly one SET_PINS,
    # no resend, and output() never reads back via 0x3F.
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(BRAKE, backend.OUT)
    backend.output(BRAKE, backend.HIGH)
    assert bytes.fromhex("3a 03 01") in backend._dev._serial.writes
    assert len(_set_writes(fake_serial)) == 1  # confirmed first try, no resend
    assert all(w[0] != 0x3F for inst in fake_serial["instances"] for w in inst.writes)


def test_output_resends_then_warns_on_missing_echo(
    fake_serial: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    # No echo comes back: output() resends the set once, then logs a warning and
    # does NOT raise, so brake control survives a flaky/silent converter.
    fake_serial["responder"] = lambda data: b""  # board never echoes
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(BRAKE, backend.OUT)
    with caplog.at_level(logging.WARNING):
        backend.output(BRAKE, backend.HIGH)  # must not raise
    assert len(_set_writes(fake_serial)) == 2  # original + one resend
    assert any("not confirmed by echo" in record.message for record in caplog.records)


def test_output_resends_then_warns_on_wrong_echo(
    fake_serial: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    # A mismatching echo (wrong level) is treated like a missing one: resend once,
    # then warn, never raise.
    fake_serial["responder"] = lambda data: bytes((0x2A, data[1], data[2] ^ 1))
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(BRAKE, backend.OUT)
    with caplog.at_level(logging.WARNING):
        backend.output(BRAKE, backend.HIGH)  # must not raise
    assert len(_set_writes(fake_serial)) == 2  # original + one resend
    assert any("not confirmed by echo" in record.message for record in caplog.records)


def test_cleanup_keeps_brake_latched_low_not_released(fake_serial: dict[str, Any]) -> None:
    # The converter latches its last-driven level across a port close, so cleanup()
    # must NOT drive outputs HIGH (released): a brake engaged LOW stays engaged after
    # shutdown, holding the rail. Regression guard for brake-released-on-shutdown.
    fake_serial["levels"][3] = False
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(BRAKE, backend.OUT)
    backend.output(BRAKE, backend.LOW)  # engage (drive ch3 low)
    serial_obj = backend._dev._serial
    backend.cleanup()
    sets = _set_writes(fake_serial)
    assert sets[-1] == bytes.fromhex("3a 03 00")  # last brake write stayed LOW (engaged)
    assert bytes.fromhex("3a 03 01") not in sets  # cleanup never released ch3 (HIGH)
    assert serial_obj.is_open is False


def test_cleanup_accepts_pin_tuple_and_tears_down(fake_serial: dict[str, Any]) -> None:
    # LinearRailController.cleanup() calls GPIO.cleanup((UPPER, LOWER)); on x86 the backend
    # must accept and ignore the tuple arg and still fully tear down the shared port.
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(BRAKE, backend.OUT)
    backend.output(BRAKE, backend.LOW)
    serial_obj = backend._dev._serial
    backend.cleanup((UPPER, LOWER))  # tuple arg, as the controller passes
    assert serial_obj.is_open is False
    assert backend._dev is None


# -- UsbGpioBackend.input ------------------------------------------------------


def test_input_reflects_level(fake_serial: dict[str, Any]) -> None:
    levels = fake_serial["levels"]
    levels[1] = True
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(UPPER, backend.IN, pull_up_down=backend.PUD_UP)
    assert backend.input(UPPER) == backend.HIGH
    assert bytes.fromhex("5b 01 01 00") in backend._dev._serial.writes  # pull-up byte = 0
    levels[1] = False
    assert backend.input(UPPER) == backend.LOW


def test_input_pull_down_frame(fake_serial: dict[str, Any]) -> None:
    fake_serial["levels"][1] = False
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(UPPER, backend.IN, pull_up_down=backend.PUD_DOWN)
    backend.input(UPPER)
    assert bytes.fromhex("5b 01 01 01") in backend._dev._serial.writes  # pull-down byte = 1


# -- add_event_detect: edge filtering and debounce (driven via _poll_once) -----


def _watched_backend(
    fake_serial: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    pin: int,
    edge: int,
    bouncetime: int = 0,
    start_level: bool = False,
) -> tuple[UsbGpioBackend, list[int]]:
    """A backend with one watched input, the poll thread suppressed so the test
    drives _poll_once() itself."""
    fake_serial["levels"][PIN_MAP[pin]] = start_level
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(pin, backend.IN, pull_up_down=backend.PUD_UP)
    monkeypatch.setattr(backend, "_start_poll_thread", lambda: None)
    fired: list[int] = []
    backend.add_event_detect(pin, edge, fired.append, bouncetime=bouncetime)
    return backend, fired


def test_seed_read_fires_no_spurious_edge(fake_serial: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    backend, fired = _watched_backend(fake_serial, monkeypatch, UPPER, UsbGpioBackend.BOTH, start_level=True)
    backend._poll_once()  # level unchanged from the seed
    assert fired == []


def test_event_detect_both_fires_on_each_change(fake_serial: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    backend, fired = _watched_backend(fake_serial, monkeypatch, UPPER, UsbGpioBackend.BOTH, start_level=False)
    fake_serial["levels"][1] = True
    backend._poll_once()
    fake_serial["levels"][1] = False
    backend._poll_once()
    assert fired == [UPPER, UPPER]


def test_event_detect_rising_only(fake_serial: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    backend, fired = _watched_backend(fake_serial, monkeypatch, UPPER, UsbGpioBackend.RISING, start_level=False)
    fake_serial["levels"][1] = True  # LOW -> HIGH
    backend._poll_once()
    fake_serial["levels"][1] = False  # HIGH -> LOW (must be ignored)
    backend._poll_once()
    assert fired == [UPPER]


def test_event_detect_falling_only(fake_serial: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    backend, fired = _watched_backend(fake_serial, monkeypatch, UPPER, UsbGpioBackend.FALLING, start_level=True)
    fake_serial["levels"][1] = False  # HIGH -> LOW
    backend._poll_once()
    fake_serial["levels"][1] = True  # LOW -> HIGH (must be ignored)
    backend._poll_once()
    assert fired == [UPPER]


def test_event_detect_debounce_suppresses_then_fires(
    fake_serial: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"t": 100.0}
    monkeypatch.setattr(usb_gpio_driver.time, "monotonic", lambda: clock["t"])
    backend, fired = _watched_backend(
        fake_serial, monkeypatch, UPPER, UsbGpioBackend.BOTH, bouncetime=50, start_level=False
    )  # 50 ms debounce
    fake_serial["levels"][1] = True
    clock["t"] = 100.0
    backend._poll_once()  # first edge fires (>= bouncetime since seed ts 0.0)
    fake_serial["levels"][1] = False
    clock["t"] = 100.02  # 20 ms < 50 ms -> suppressed
    backend._poll_once()
    assert fired == [UPPER]
    clock["t"] = 100.10  # now 100 ms since the last fire -> the pending change fires
    backend._poll_once()
    assert fired == [UPPER, UPPER]


# -- poll-read failure: rate-limited logging (#5) ------------------------------


def test_poll_read_failure_is_rate_limited(
    fake_serial: dict[str, Any], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    backend, _ = _watched_backend(fake_serial, monkeypatch, UPPER, UsbGpioBackend.BOTH, start_level=False)
    backend._dev._serial.responder = lambda data: b""  # every read now times out
    with caplog.at_level(logging.WARNING, logger="i2rt.utils.usb_gpio_driver"):
        waits = [backend._poll_once() for _ in range(5)]
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "poll read failed" in r.message]
    assert len(warnings) == 1  # logged once, not five times
    assert waits[0] < waits[-1]  # backed off as failures accumulated
    assert backend._poll_fail_count == 5


# -- cleanup -------------------------------------------------------------------


def test_cleanup_stops_thread_and_closes_serial(fake_serial: dict[str, Any]) -> None:
    fake_serial["levels"][1] = False
    backend = UsbGpioBackend("/dev/ttyFAKE", pin_map=PIN_MAP)
    backend.setmode(backend.BCM)
    backend.setup(UPPER, backend.IN, pull_up_down=backend.PUD_UP)
    backend.add_event_detect(UPPER, backend.BOTH, lambda p: None, bouncetime=0)  # starts the real poll thread
    serial_obj = backend._dev._serial
    backend.cleanup()
    assert serial_obj.is_open is False
    assert backend._dev is None
    assert backend._watch == {}
    assert backend._poll_thread is None
