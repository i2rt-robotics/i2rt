"""Tests for the yumi gripper reader, against a fake CAN bus that answers like the handle encoder.

The gripper mapping is easy to get backwards (the encoder zero is the jaws' *open* stop, and i2rt reports
1 = open), and the reader runs on a background thread whose failures are invisible unless the getter surfaces
them. These tests pin the mapping end to end through the real ``CanInterface`` and ``PassiveEncoderReader``,
and pin that a silent encoder becomes a ``TimeoutError`` naming the cause, that ``close`` is safe to repeat,
and that a failed construction does not leak the CAN socket.
"""

import collections
import math
import struct
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import can
import pytest

from i2rt.utils.encoder_manager import EncoderCanID, PassiveJointEncoder
from i2rt.yumi.gripper import DEFAULT_RANGE_RAD, GripperState, UmiGripper

# int16 counts per revolution on the wire. The full-range count is derived from the default range (5.1 rad ->
# 3325 counts, 5.1006 rad: the first count at or past it), so retuning the range cannot leave these tests stale.
_COUNTS_PER_REV = 4096
_FULL_RANGE_COUNTS = math.ceil(DEFAULT_RANGE_RAD * _COUNTS_PER_REV / (2 * math.pi))


class _FakeEncoderBus:
    """A python-can bus stand-in: answers each report request on 0x50E with a report on 0x50F."""

    def __init__(self, channel: str) -> None:
        self.channel_info = f"fake:{channel}"
        self.state = can.BusState.ACTIVE
        self.counts = 0
        self.velocity_counts = 0
        self.digital_inputs = 0
        self.replying = True
        self.shutdown_calls = 0
        self._rx: collections.deque[can.Message] = collections.deque()

    def send(self, msg: can.Message, timeout: float | None = None) -> None:
        if not self.replying or msg.arbitration_id != EncoderCanID.REQ or list(msg.data) != [0xFF, 0x02]:
            return
        data = struct.pack("!BhhB", 0x0E, self.counts, self.velocity_counts, self.digital_inputs)
        self._rx.append(can.Message(arbitration_id=EncoderCanID.REPORT, data=data, is_extended_id=False))

    def recv(self, timeout: float | None = None) -> can.Message | None:
        if self._rx:
            return self._rx.popleft()
        if timeout:
            time.sleep(min(timeout, 0.001))
        return None

    def shutdown(self) -> None:
        self.shutdown_calls += 1


@pytest.fixture
def buses(monkeypatch: pytest.MonkeyPatch) -> list[_FakeEncoderBus]:
    """Route every ``can.interface.Bus`` to a fake and skip the real encoder validation."""
    created: list[_FakeEncoderBus] = []

    def make_bus(channel: str, **_kwargs: Any) -> _FakeEncoderBus:
        bus = _FakeEncoderBus(channel)
        created.append(bus)
        return bus

    monkeypatch.setattr(can.interface, "Bus", make_bus)
    monkeypatch.setattr(PassiveJointEncoder, "validate_encoders", staticmethod(lambda _channel, _config: {}))
    return created


@pytest.fixture
def gripper(buses: list[_FakeEncoderBus]) -> Iterator[UmiGripper]:
    g = UmiGripper("can_fake")
    try:
        yield g
    finally:
        g.close()


def _wait_for(predicate: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"condition not met within {timeout_s} s")
        time.sleep(0.005)


def _settled_state(
    gripper: UmiGripper, bus: _FakeEncoderBus, counts: int, velocity_counts: int = 0, digital_inputs: int = 0
) -> GripperState:
    """Set the fake encoder's reading and return the first state the poll thread reports after it."""
    bus.counts = counts
    bus.velocity_counts = velocity_counts
    bus.digital_inputs = digital_inputs
    marker = time.time()
    _wait_for(lambda: gripper.get_state().timestamp > marker)
    # One more fresh sample, so a read that was in flight across the change cannot be the one returned.
    after = gripper.get_state().timestamp
    _wait_for(lambda: gripper.get_state().timestamp > after)
    return gripper.get_state()


def test_zero_counts_reads_fully_open(gripper: UmiGripper, buses: list[_FakeEncoderBus]) -> None:
    state = _settled_state(gripper, buses[0], 0)
    assert state.position == pytest.approx(1.0)


@pytest.mark.parametrize("sign", [1, -1])
def test_full_range_reads_closed(gripper: UmiGripper, buses: list[_FakeEncoderBus], sign: int) -> None:
    assert _settled_state(gripper, buses[0], sign * _FULL_RANGE_COUNTS).position == pytest.approx(0.0)


@pytest.mark.parametrize("counts", [_FULL_RANGE_COUNTS + 500, -(_FULL_RANGE_COUNTS + 500), 32767, -32768])
def test_beyond_range_is_clipped_to_closed(gripper: UmiGripper, buses: list[_FakeEncoderBus], counts: int) -> None:
    assert _settled_state(gripper, buses[0], counts).position == pytest.approx(0.0)


def test_sign_is_ignored_and_mapping_is_linear(gripper: UmiGripper, buses: list[_FakeEncoderBus]) -> None:
    half = _FULL_RANGE_COUNTS // 2
    positive = _settled_state(gripper, buses[0], half).position
    negative = _settled_state(gripper, buses[0], -half).position
    assert positive == pytest.approx(negative)
    # Half the full-range count is half the range, to within one count.
    assert positive == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize(
    "counts", [_FULL_RANGE_COUNTS // 2, -(_FULL_RANGE_COUNTS // 2), _FULL_RANGE_COUNTS + 500, -32768]
)
def test_raw_angle_is_signed_and_unclipped(gripper: UmiGripper, buses: list[_FakeEncoderBus], counts: int) -> None:
    # The position folds the sign and clips at the range; the raw angle keeps both, so a bad zero is visible.
    state = _settled_state(gripper, buses[0], counts)
    assert state.raw_angle == pytest.approx(counts * 2 * math.pi / _COUNTS_PER_REV)


def test_raw_velocity_is_rad_per_second(gripper: UmiGripper, buses: list[_FakeEncoderBus]) -> None:
    state = _settled_state(gripper, buses[0], 0, velocity_counts=-1024)
    assert state.raw_velocity == pytest.approx(-1024 * 2 * 3.141592653589793 / _COUNTS_PER_REV)


@pytest.mark.parametrize(
    ("digital_inputs", "buttons"), [(0, (False, False)), (1, (True, False)), (2, (False, True)), (3, (True, True))]
)
def test_buttons_are_the_digital_inputs(
    gripper: UmiGripper, buses: list[_FakeEncoderBus], digital_inputs: int, buttons: tuple[bool, bool]
) -> None:
    # Bit 0 is the first input, bit 1 the second.
    assert _settled_state(gripper, buses[0], 0, digital_inputs=digital_inputs).buttons == buttons


def test_timestamp_is_wall_clock(gripper: UmiGripper) -> None:
    assert abs(gripper.get_state().timestamp - time.time()) < 5.0


def test_poll_thread_is_daemon(gripper: UmiGripper) -> None:
    threads = [t for t in threading.enumerate() if t.name == "umi-gripper-can_fake"]
    assert threads
    assert all(t.daemon for t in threads)


def test_silent_encoder_raises_timeout_with_last_error(buses: list[_FakeEncoderBus]) -> None:
    with UmiGripper("can_fake", stale_after_s=0.2) as gripper:
        buses[0].replying = False
        messages: list[str] = []

        def raised_with_cause() -> bool:
            try:
                gripper.get_state()
            except TimeoutError as exc:
                messages.append(str(exc))
                return "AssertionError" in messages[-1]
            return False

        _wait_for(raised_with_cause, timeout_s=10.0)
        assert "stale after 0.2 s" in messages[-1]
        assert "fail to communicate" in messages[-1]

        # The thread survives the failures and recovers once the encoder answers again.
        buses[0].replying = True

        def fresh() -> bool:
            try:
                gripper.get_state()
            except TimeoutError:
                return False
            return True

        _wait_for(fresh, timeout_s=10.0)


def test_close_is_idempotent(buses: list[_FakeEncoderBus]) -> None:
    gripper = UmiGripper("can_fake")
    gripper.close()
    gripper.close()
    assert buses[0].shutdown_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        gripper.get_state()


def test_context_manager_closes(buses: list[_FakeEncoderBus]) -> None:
    with UmiGripper("can_fake") as gripper:
        gripper.get_state()
    assert buses[0].shutdown_calls == 1


def test_validation_failure_closes_can_interface(
    buses: list[_FakeEncoderBus], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_encoders(channel: str, _config: object) -> dict:
        raise RuntimeError(f"No encoders found on {channel}")

    monkeypatch.setattr(PassiveJointEncoder, "validate_encoders", staticmethod(no_encoders))
    with pytest.raises(RuntimeError, match="No encoders found"):
        UmiGripper("can_fake")
    assert len(buses) == 1
    assert buses[0].shutdown_calls == 1


def test_no_first_sample_times_out_and_closes(buses: list[_FakeEncoderBus], monkeypatch: pytest.MonkeyPatch) -> None:
    def make_silent_bus(channel: str, **_kwargs: Any) -> _FakeEncoderBus:
        bus = _FakeEncoderBus(channel)
        bus.replying = False
        buses.append(bus)
        return bus

    monkeypatch.setattr(can.interface, "Bus", make_silent_bus)
    with pytest.raises(TimeoutError, match="No gripper encoder reading"):
        UmiGripper("can_fake", first_sample_timeout_s=0.5)
    assert len(buses) == 1
    assert buses[0].shutdown_calls == 1
