"""The compiled EtherCAT-to-CAN master, run in loopback: an emulation of the board as measured on
hardware, where each frame comes back as its own reply two cycles later in the same slot, and a slot
with nothing to answer keeps its last content. That exercises the real loop thread, the slot
rotation, reply routing, stale-slot rejection, stamps and the watchdog without a NIC or a board.

Assertions are on counts, not wall-clock rates, so they hold on a loaded CI machine.
"""

import time

import numpy as np
import pytest

i2rt_ecat = pytest.importorskip("i2rt_ecat")
EcatCanBus = i2rt_ecat.EcatCanBus


def _frames(n: int, base: int = 0) -> np.ndarray:
    return np.array([[base + i + 1] * 8 for i in range(n)], dtype=np.uint8)


def _run(bus: "EcatCanBus", frames: np.ndarray, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        bus.set_frames(frames)
        time.sleep(0.01)


def test_each_frame_comes_back_to_its_own_motor() -> None:
    with EcatCanBus("lo", [(0, 1, i) for i in range(1, 8)], loopback=True) as bus:
        tx = _frames(7)
        _run(bus, tx, 0.1)
        rx, _, count = bus.read()
        assert (count > 0).all()
        np.testing.assert_array_equal(rx, tx)


def test_seven_motors_on_one_channel_get_three_sevenths_of_the_cycles() -> None:
    """Three slots per channel per cycle, rotating: 7 motors are each served 3 cycles in 7."""
    with EcatCanBus("lo", [(0, 1, i) for i in range(1, 8)], loopback=True) as bus:
        _run(bus, _frames(7), 0.3)
        _, _, count = bus.read()
        ticks = bus.stats()["ticks"]
    # Three frames a cycle, less the first two cycles' worth, which the two-cycle reply lag leaves unanswered.
    assert abs(int(count.sum()) - 3 * ticks) <= 3 * 3
    assert int(count.max()) - int(count.min()) <= 1  # the rotation is fair


def test_three_per_channel_are_served_every_cycle() -> None:
    motors = [(0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 2), (0, 2, 5)]  # ids may repeat across channels
    with EcatCanBus("lo", motors, loopback=True) as bus:
        _run(bus, _frames(6), 0.2)
        rx, _, count = bus.read()
        ticks = bus.stats()["ticks"]
    assert (np.abs(count.astype(np.int64) - ticks) <= 3).all()
    np.testing.assert_array_equal(rx, _frames(6))


@pytest.mark.parametrize("motors", [[(0, 1, 1)], [(0, 1, 1), (0, 1, 2)], [(0, 1, i) for i in (1, 2, 3)] + [(0, 2, 1)]])
def test_a_channel_with_fewer_than_three_motors_is_refused(motors: list) -> None:
    """With fewer than three, some slot goes unasked each cycle and keeps a stale reply, and a
    stopped motor's own last reply can sit there and read as fresh."""
    with pytest.raises(OSError, match="needs at least 3"):
        EcatCanBus("lo", motors, loopback=True)


def test_a_dead_motor_in_the_rotation_stops_being_heard_and_lends_nobody_its_slot() -> None:
    """The board leaves an unanswered slot holding its previous reply. Found on hardware: with a
    silent id in the rotation, the motor that last used its slot was counted twice. A reply is only
    believed from the motor the slot was sent to, so neither can happen here."""
    with EcatCanBus("lo", [(0, 1, i) for i in range(1, 8)], loopback=True) as bus:
        _run(bus, _frames(7), 0.1)
        bus.loopback_mute(4)
        _, _, before = bus.read()
        _run(bus, _frames(7), 0.2)
        _, _, after = bus.read()
    gained = after.astype(np.int64) - before.astype(np.int64)
    assert gained[3] <= 1  # joint4 is silent
    alive = np.delete(gained, 3)
    assert alive.max() - alive.min() <= 1  # and nobody else inherited its slot


def test_a_dead_motor_in_a_group_of_three_is_not_re_read_forever() -> None:
    """Three motors per channel would each own one slot, where a dead motor's last reply would sit
    unchanged and look fresh every cycle -- blinding a missing-frame watchdog. Their slots shift every
    cycle, so a stale reply is never where that motor was asked."""
    with EcatCanBus("lo", [(0, 1, 1), (0, 1, 2), (0, 1, 3)], loopback=True) as bus:
        _run(bus, _frames(3), 0.1)
        bus.loopback_mute(1)
        _, stamp_before, before = bus.read()
        _run(bus, _frames(3), 0.2)
        _, stamp_after, after = bus.read()
    assert after[0] - before[0] <= 2 and stamp_after[0] - stamp_before[0] < 0.01
    alive = after[1:].astype(np.int64) - before[1:].astype(np.int64)
    assert (alive > 50).all() and alive.max() - alive.min() <= 2  # nobody inherited the dead slot


def test_reply_stamps_are_wall_clock_and_advance() -> None:
    with EcatCanBus("lo", [(0, 1, 1), (0, 1, 2), (0, 1, 3)], loopback=True) as bus:
        t0 = time.time()
        _run(bus, _frames(3), 0.05)
        _, first, _ = bus.read()
        _run(bus, _frames(3), 0.05)
        _, second, _ = bus.read()
        t1 = time.time()
    assert t0 <= first[0] < second[0] <= t1


def test_the_watchdog_releases_every_motor_and_resumes_on_the_next_command() -> None:
    release = _frames(3, base=100)
    with EcatCanBus("lo", [(0, 1, i) for i in (1, 2, 3)], release=release, watchdog_ms=30, loopback=True) as bus:
        _run(bus, _frames(3), 0.1)
        trips_before = bus.stats()["watchdog_trips"]
        time.sleep(0.15)  # no command for five watchdog periods
        rx, _, _ = bus.read()
        np.testing.assert_array_equal(rx, release)
        assert bus.stats()["watchdog_trips"] == trips_before + 1
        _run(bus, _frames(3), 0.05)
        rx, _, _ = bus.read()
        np.testing.assert_array_equal(rx, _frames(3))


def test_raw_mode_sends_each_frame_once_and_queues_every_reply() -> None:
    with EcatCanBus("lo", loopback=True) as bus:
        bus.send_raw(0, 1, 0x7FF, bytes([1, 0, 0x33, 7]))
        bus.send_raw(0, 2, 5, b"\x01\x02")
        first, second = bus.recv_raw(0.2), bus.recv_raw(0.2)
        assert first is not None and second is not None
        assert (first.channel, first.can_id, first.data) == (1, 0x7FF, bytes([1, 0, 0x33, 7]))
        assert (second.channel, second.can_id, second.data) == (2, 5, b"\x01\x02")
        assert bus.recv_raw(0.02) is None  # sent once, not re-sent every cycle


def test_raw_frames_sharing_an_id_go_out_one_per_cycle() -> None:
    """The board answers none of several same-id frames in one cycle -- three register requests to the
    0x7FF broadcast id are the case that exposed it -- so they are spread over cycles."""
    with EcatCanBus("lo", loopback=True) as bus:
        for motor in (1, 2, 3):
            bus.send_raw(0, 1, 0x7FF, bytes([motor, 0, 0x33, 7, 0, 0, 0, 0]))
        out, _ = bus.trace(8)
        got = [bus.recv_raw(0.2) for _ in range(3)]
    per_cycle = [sum(int(img[2 + 14 * k + 5] != 0) for k in range(3)) for img in out]
    assert max(per_cycle) == 1
    assert [f.data[0] for f in got if f is not None] == [1, 2, 3]


def test_close_is_idempotent_and_a_second_bus_is_refused_while_one_is_open() -> None:
    three = [(0, 1, 1), (0, 1, 2), (0, 1, 3)]
    bus = EcatCanBus("lo", three, loopback=True)
    try:
        with pytest.raises(OSError, match="already open"):
            EcatCanBus("lo", three, loopback=True)
    finally:
        bus.close()
    bus.close()
    EcatCanBus("lo", three, loopback=True).close()  # the slot is free again


@pytest.mark.parametrize(
    "motors, match",
    [
        ([(0, 3, 1)], "out of range"),
        ([(5, 1, 1)], "out of range"),
        ([(0, 1, 0)], "out of range"),
        ([(0, 1, 4), (0, 1, 4)], "twice"),
    ],
)
def test_bad_motor_maps_are_refused(motors: list, match: str) -> None:
    with pytest.raises(OSError, match=match):
        EcatCanBus("lo", motors, loopback=True)


def test_a_missing_interface_is_named() -> None:
    with pytest.raises(OSError, match="no network interface 'no_such_nic0'"):
        EcatCanBus("no_such_nic0", [(0, 1, 1), (0, 1, 2), (0, 1, 3)])
