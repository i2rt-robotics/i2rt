"""Exercise real receive timing against a deterministic bus, without hardware."""

from collections import deque

import can
import pytest

from i2rt.motor_drivers import can_interface
from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.utils import ReceiveMode


class Clock:
    def __init__(self) -> None:
        self.ns = 0

    def time(self) -> float:
        return self.ns / 1_000_000_000

    def sleep(self, seconds: float) -> None:
        self.ns += round(seconds * 1_000_000_000)


class Bus:
    """Schedule replies relative to each send; recv consumes its timeout."""

    state = can.BusState.ACTIVE
    channel_info = "deterministic test bus"

    def __init__(self, clock: Clock, attempts: list[list[tuple[float, can.Message]] | can.CanError]) -> None:
        self.clock = clock
        self.attempts = deque(attempts)
        self.pending: list[tuple[int, can.Message]] = []
        self.sent: list[can.Message] = []

    def send(self, message: can.Message) -> None:
        self.sent.append(message)
        attempt = self.attempts.popleft() if self.attempts else []
        if isinstance(attempt, can.CanError):
            raise attempt
        self.pending.extend((self.clock.ns + round(delay * 1_000_000_000), reply) for delay, reply in attempt)
        self.pending.sort(key=lambda entry: entry[0])

    def recv(self, timeout: float) -> can.Message | None:
        deadline = self.clock.ns + round(timeout * 1_000_000_000)
        if self.pending and self.pending[0][0] <= deadline:
            at, message = self.pending.pop(0)
            self.clock.ns = max(self.clock.ns, at)
            return message
        self.clock.ns = deadline
        return None


def reply(arbitration_id: int) -> can.Message:
    return can.Message(arbitration_id=arbitration_id, data=bytes(8), is_extended_id=False)


def interface(monkeypatch: pytest.MonkeyPatch, bus: Bus, mode: ReceiveMode = ReceiveMode.p16) -> CanInterface:
    monkeypatch.setattr(can_interface, "time", bus.clock)
    monkeypatch.setattr(can.interface, "Bus", lambda **_: bus)
    return CanInterface(channel="test", receive_mode=mode)


@pytest.mark.parametrize("delay", [0.001, 0.012, 0.018])
@pytest.mark.parametrize("mode", list(ReceiveMode))
def test_valid_reply_does_not_retransmit(monkeypatch: pytest.MonkeyPatch, delay: float, mode: ReceiveMode) -> None:
    response = reply(mode.get_receive_id(2))
    bus = Bus(Clock(), [[(delay, response)]])
    client = interface(monkeypatch, bus, mode)

    assert client._send_message_get_response(2, 2, [0] * 8, max_retry=1) is response
    assert len(bus.sent) == 1


def test_explicit_reply_id_for_register_request(monkeypatch: pytest.MonkeyPatch) -> None:
    response = reply(0x12)
    bus = Bus(Clock(), [[(0.012, response)]])
    client = interface(monkeypatch, bus, ReceiveMode.same)

    assert client._send_message_get_response(0x7FF, 2, [2, 0, 0x33, 9, 0, 0, 0, 0], expected_id=0x12) is response
    assert len(bus.sent) == 1


def test_wrong_motor_reply_before_correct_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    response = reply(0x12)
    bus = Bus(Clock(), [[(0.001, reply(0x11)), (0.003, response)]])
    client = interface(monkeypatch, bus)

    assert client._send_message_get_response(2, 2, [0] * 8, max_retry=1) is response
    assert len(bus.sent) == 1


def test_wrong_fallback_reply_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    response = reply(0x12)
    bus = Bus(Clock(), [[(0.012, reply(0x11))], [(0.001, response)]])
    client = interface(monkeypatch, bus)

    assert client._send_message_get_response(2, 2, [0] * 8, max_retry=2) is response
    assert len(bus.sent) == 2


def test_missing_reply_keeps_retry_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = Bus(Clock(), [])
    client = interface(monkeypatch, bus)

    with pytest.raises(AssertionError, match="fail to communicate with the motor 2"):
        client._send_message_get_response(2, 2, [0] * 8, max_retry=3)
    assert len(bus.sent) == 3
    assert bus.clock.time() <= 0.063


def test_reply_after_receive_windows_does_not_extend_retry_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = Bus(Clock(), [[(0.025, reply(0x12))]])
    client = interface(monkeypatch, bus)

    with pytest.raises(AssertionError, match="fail to communicate"):
        client._send_message_get_response(2, 2, [0] * 8, max_retry=1)
    assert len(bus.sent) == 1
    assert bus.clock.time() <= 0.021


def test_send_failure_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    response = reply(0x12)
    bus = Bus(Clock(), [can.CanError("send failed"), [(0.012, response)]])
    client = interface(monkeypatch, bus)

    assert client._send_message_get_response(2, 2, [0] * 8, max_retry=2) is response
    assert len(bus.sent) == 2


def test_fallback_receives_motor_id_not_transmit_id(monkeypatch: pytest.MonkeyPatch) -> None:
    response = reply(0x12)
    bus = Bus(Clock(), [[]])
    client = interface(monkeypatch, bus)
    received: list[int] = []

    def fallback(motor_id: int) -> can.Message:
        received.append(motor_id)
        return response

    monkeypatch.setattr(client, "try_receive_message", fallback)
    assert client._send_message_get_response(0x7FF, 2, [0] * 8, max_retry=1) is response
    assert received == [2]
