"""Tests for the Flow Base fault ramp-down and shutdown sequence.

Hardware-free. ``FakeChain`` stands in for ``DMChainCanInterface`` one level above the wire: it holds
the commanded velocities, integrates them into steering positions so the caster law actually closes,
and records the teardown calls in order. That is the right altitude here -- these tests pin *this*
package's shutdown ordering, not python-can's behaviour.

Two things make the ordering worth pinning at all. ``set_commands`` only swaps a list in memory, so a
neutral command reaches the motors only while the chain thread is alive; and that thread is
**non-daemon**, so leaving it running blocks interpreter shutdown before ``atexit`` ever runs. Getting
the order wrong therefore either leaves the wheels turning or hangs the process, and neither shows up
in a test that only checks the calls were made.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from i2rt.flow_base import caster_steering_check as csc
from i2rt.flow_base import flow_base_controller as fbc
from i2rt.motor_drivers.utils import MotorInfo

NUM_MOTORS = 8

RUNAWAY_STEER_VEL = 15.0
"""rad/s reported by a steering motor in ``runaway`` mode. Over ``csc.RUNAWAY_RATE`` (12.56) and under
the motor's own 30 rad/s limit, so it is a rate the hardware could really produce -- and it is over the
controller's caster-flip-brake threshold on every single cycle, which is the point of it."""


class FakeChain:
    """A ``DMChainCanInterface`` stand-in that integrates its own steering commands."""

    def __init__(self, stalled: np.ndarray | None = None, runaway: np.ndarray | None = None) -> None:
        self.motor_offset = np.zeros(NUM_MOTORS)
        self.motor_direction = np.ones(NUM_MOTORS)
        self.running = True
        self.command_lock = threading.RLock()
        self.commands: list = []
        self.calls: list[str] = []
        self.closed = False
        self._pos = np.zeros(NUM_MOTORS)
        self._vel = np.zeros(NUM_MOTORS)
        self._stalled = np.zeros(4, dtype=bool) if stalled is None else stalled
        self._runaway = np.zeros(4, dtype=bool) if runaway is None else runaway
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return NUM_MOTORS

    def read_states(self, torques: np.ndarray | None = None) -> list[MotorInfo]:
        with self._lock:
            now = time.monotonic()
            dt = now - self._last
            self._last = now
            # First-order: each motor reaches its commanded velocity immediately, except a stalled
            # steering motor, which reports zero and does not move, and a runaway one, which ignores
            # the command outright and reports RUNAWAY_STEER_VEL whatever it was told.
            vel = self._vel.copy()
            steer = np.where(self._stalled, 0.0, vel[0:8:2])
            vel[0:8:2] = np.where(self._runaway, RUNAWAY_STEER_VEL, steer)
            self._pos = self._pos + vel * dt
            return [
                MotorInfo(id=i + 1, error_code="0x1", pos=float(self._pos[i]), vel=float(vel[i]), eff=0.0)
                for i in range(NUM_MOTORS)
            ]

    def set_commands(self, torques, pos=None, vel=None, kp=None, kd=None, get_state=True):  # noqa: ANN001, ANN201
        with self._lock:
            if vel is not None:
                self._vel = np.asarray(vel, dtype=float).copy()
        self.calls.append("set_commands")
        return None if not get_state else self.read_states()

    def close(self) -> None:
        self.calls.append("close")
        self.running = False
        self.closed = True

    @property
    def commanded_steer_vel(self) -> np.ndarray:
        with self._lock:
            return self._vel[0:8:2].copy()


@pytest.fixture(autouse=True)
def _no_pid_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the single-instance lock out of the test run.

    ``Vehicle.__init__`` writes ``/tmp/base-controller.pid`` and refuses to start if another instance
    holds it -- which would make these tests fail on a machine that happens to be driving a base, and
    litter ``/tmp`` besides.
    """
    monkeypatch.setattr(fbc, "create_pid_file", lambda name: None)


def _vehicle(chain: FakeChain, **kwargs) -> fbc.Vehicle:  # noqa: ANN003
    return fbc.Vehicle(channel=chain, auto_start=False, **kwargs)


# --- close(): ordering, ownership, idempotence -----------------------------------------------------


def test_close_releases_a_chain_it_opened_in_the_right_order() -> None:
    chain = FakeChain()
    vehicle = _vehicle(chain)
    vehicle._owns_motor_chain = True  # as if we had been given a channel name
    vehicle.close()
    # Neutral must be commanded before the chain is stopped, or it never reaches the wire.
    assert chain.calls == ["set_commands", "close"]
    assert chain.running is False


def test_close_leaves_a_borrowed_chain_alone() -> None:
    # A caller who handed us a live chain still owns it and may keep using it.
    chain = FakeChain()
    vehicle = _vehicle(chain)
    assert vehicle._owns_motor_chain is False
    vehicle.close()
    assert "close" not in chain.calls
    assert chain.running is True


def test_close_is_idempotent() -> None:
    chain = FakeChain()
    vehicle = _vehicle(chain)
    vehicle._owns_motor_chain = True
    vehicle.close()
    calls = list(chain.calls)
    vehicle.close()
    assert chain.calls == calls


def test_close_stops_the_chain_even_if_neutralising_raises() -> None:
    # The one moment when leaving a live thread driving the bus is least acceptable is when we could
    # not neutralise, so the teardown lives in a finally.
    chain = FakeChain()
    vehicle = _vehicle(chain)
    vehicle._owns_motor_chain = True

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("CAN write failed")

    vehicle.caster_module_controller.set_neutral = boom
    vehicle.close()
    assert chain.running is False
    assert chain.closed is True


def test_a_bad_gpio_device_still_releases_the_chain_it_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """The window between opening the chain and binding the object whose close() would release it.

    ``DMChainCanInterface.__init__`` starts a non-daemon reader thread and discards the handle, so
    nothing can join it; and a constructor that raises never binds ``vehicle``, so no close() is ever
    called on it. Without the guard the interpreter hangs before ``atexit``, ``/tmp/base-controller.pid``
    survives, and the next launch is refused while the orphan keeps driving the bus at 250 Hz.

    The failure modelled here is the reachable one: ``initialize_brake_gpio`` re-raises anything that is
    not a ``RuntimeError``, so ``--linear-rail`` with a ``--device`` that names the wrong port on x86
    lands exactly here -- a mistake the ``--device`` help string already calls out.
    """
    from i2rt.flow_base import linear_rail_controller as lrc

    chain = FakeChain()
    monkeypatch.setattr(fbc, "DMChainCanInterface", lambda **kwargs: chain)

    def boom() -> None:
        raise OSError("could not open port /dev/ttyUSB1: No such file or directory")

    monkeypatch.setattr(lrc, "initialize_brake_gpio", boom)

    with pytest.raises(OSError, match="ttyUSB1"):
        fbc.LinearRailVehicle(channel="can_test", auto_start=False, verify_motor_config=False, enable_linear_rail=True)

    assert chain.closed is True, "the chain we opened must not outlive the constructor that opened it"
    assert chain.running is False
    # Nothing to neutralise: super().__init__() never ran, so there is no caster_module_controller.
    assert chain.calls == ["close"]


def test_a_failure_inside_the_base_constructor_also_releases_the_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rest of the same window: the base class can raise too, and it is past the GPIO work."""
    chain = FakeChain()
    monkeypatch.setattr(fbc, "DMChainCanInterface", lambda **kwargs: chain)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("the bus went away mid-construction")

    monkeypatch.setattr(fbc, "VehicleMotorController", boom)

    with pytest.raises(RuntimeError, match="mid-construction"):
        fbc.LinearRailVehicle(
            channel="can_test", auto_start=False, verify_motor_config=False, enable_linear_rail=False
        )

    assert chain.closed is True
    assert chain.running is False


def test_a_rail_failure_after_the_base_is_up_neutralises_before_releasing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past super().__init__() the unwind goes through close(), which stops the motors first.

    Homing drives the rail, so a chain stopped without neutralising leaves it moving at its last
    commanded velocity until the motor's own TIMEOUT failsafe cuts it -- and that failsafe is a
    per-station property (``set_timeout.py`` disables it unless passed ``--timeout``), so this path
    cannot rely on it.
    """
    from i2rt.flow_base import linear_rail_controller as lrc

    chain = FakeChain()
    monkeypatch.setattr(fbc, "DMChainCanInterface", lambda **kwargs: chain)
    monkeypatch.setattr(lrc, "initialize_brake_gpio", lambda: None)
    monkeypatch.setattr(lrc, "set_usb_gpio_device", lambda _device: None)
    monkeypatch.setattr(
        lrc.SingleMotorControlInterface, "from_multi_motor_chain", classmethod(lambda *a, **k: object())
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("homing never found the limit switch")

    monkeypatch.setattr(lrc, "LinearRailController", boom)

    with pytest.raises(RuntimeError, match="limit switch"):
        fbc.LinearRailVehicle(channel="can_test", auto_start=False, verify_motor_config=False, enable_linear_rail=True)

    # Neutral before close, exactly as in the normal teardown.
    assert chain.calls == ["set_commands", "close"]
    assert chain.closed is True


# --- the disable switch ----------------------------------------------------------------------------


def test_the_check_can_be_turned_off_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        vehicle = _vehicle(FakeChain(), check_caster_steering=False)
    assert vehicle._caster_monitor is None
    assert vehicle.caster_fault() is None
    assert vehicle.caster_history_csv() is None
    assert "DISABLED" in caplog.text


def test_the_check_is_on_by_default() -> None:
    assert _vehicle(FakeChain())._caster_monitor is not None


# --- the fault path, end to end through a running control loop -------------------------------------


@pytest.fixture
def _fast_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the startup hold-off so these tests cost tenths of a second, not seconds."""
    monkeypatch.setattr(csc, "STARTUP_HOLDOFF_S", 0.1)


def _command(elapsed: float) -> np.ndarray:
    """Forward, then strafe: a 90 degree change of commanded direction.

    A *constant* command would prove nothing. The casters start at zero, which is already the
    equilibrium for driving forward, so the commanded steer rate is ~zero and a stalled motor is
    genuinely indistinguishable from a healthy one -- see
    ``test_a_stalled_motor_already_at_the_right_angle_is_not_reported``. The fault only becomes
    observable once the commanded direction moves.
    """
    return np.array([0.3, 0.0, 0.0]) if elapsed < 1.0 else np.array([0.0, 0.3, 0.0])


def _drive_until_fault(vehicle: fbc.Vehicle, timeout: float = 8.0) -> csc.CasterFault | None:
    """Stream velocity commands and poll exactly as ``__main__`` does."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if vehicle.control_loop_running:
            vehicle.set_target_velocity(_command(time.monotonic() - start), frame="local")
        fault = vehicle.caster_fault()
        if fault is not None:
            return fault
        time.sleep(0.02)
    return None


def test_a_stalled_steering_motor_stops_the_base_and_publishes_a_fault(_fast_startup: None) -> None:
    chain = FakeChain(stalled=np.array([False, False, True, False]))
    vehicle = _vehicle(chain)
    vehicle._owns_motor_chain = True
    vehicle.start_control()
    try:
        fault = _drive_until_fault(vehicle)
        assert fault is not None, "a permanently stalled steering motor must be caught"
        assert fault.caster == 2
        assert f"CAN id {csc.steering_can_id(2)}" in fault.render()
        # Published only once the loop has ramped down and exited.
        vehicle.wait_for_stop()
        assert vehicle.control_loop_running is False
        np.testing.assert_allclose(chain.commanded_steer_vel, 0.0, atol=1e-9)
    finally:
        vehicle.close()


def test_a_runaway_steering_motor_stops_the_base_and_publishes_a_fault(_fast_startup: None) -> None:
    """The backstop, end to end, with the flip brake firing on every single cycle.

    A steering motor stuck at ``RUNAWAY_STEER_VEL`` is over the controller's caster-flip-brake threshold
    -- the same 12.56 rad/s the monitor uses -- every cycle, so ``hold_off("caster-flip brake")`` runs
    immediately before every ``update``. The rate and heading detectors are therefore suppressed for the
    whole run *by design*: the runaway row is the only one that can fire, which is exactly the coupling
    this test exists to pin. It is also why the fault must render from a snapshot taken inside a
    hold-off. (``_fast_startup`` is here for speed only -- the runaway row is exempt from the startup
    hold-off too.)
    """
    chain = FakeChain(runaway=np.array([False, True, False, False]))
    vehicle = _vehicle(chain)
    vehicle._owns_motor_chain = True
    vehicle.start_control()
    try:
        fault = _drive_until_fault(vehicle)
        assert fault is not None, "a steering motor spinning at 15 rad/s must be caught"
        assert fault.kind == "SUSTAINED_STEER_RATE", "the brake suppresses rate/heading; only this can fire"
        assert fault.caster == 1
        assert f"CAN id {csc.steering_can_id(1)}" in fault.render()
        vehicle.wait_for_stop()
        assert vehicle.control_loop_running is False
        np.testing.assert_allclose(chain.commanded_steer_vel, 0.0, atol=1e-9)
    finally:
        vehicle.close()


def test_the_fault_is_withheld_until_the_base_has_stopped(_fast_startup: None) -> None:
    # The load-bearing half of the two-stage latch: __main__ polls at 50 Hz, and acting on the fault
    # at detection time would tear the chain down mid-ramp, at speed.
    chain = FakeChain(stalled=np.array([True, False, False, False]))
    vehicle = _vehicle(chain)
    vehicle.start_control()
    try:
        start = time.monotonic()
        seen_ramping = False
        while time.monotonic() - start < 8.0:
            if vehicle.control_loop_running:
                vehicle.set_target_velocity(_command(time.monotonic() - start), frame="local")
            if vehicle._caster_fault is not None:
                # Detected internally. Until the ramp finishes, the public accessor stays quiet.
                if vehicle._caster_fault_ready.is_set():
                    break
                assert vehicle.caster_fault() is None
                seen_ramping = True
            time.sleep(0.005)
        assert vehicle._caster_fault is not None, "the fault should have been detected"
        assert seen_ramping, "the fault should have been withheld for at least one poll while ramping"
        vehicle.wait_for_stop()
        assert vehicle.caster_fault() is not None
    finally:
        vehicle.close()


def test_a_healthy_base_drives_without_tripping(_fast_startup: None) -> None:
    chain = FakeChain()
    vehicle = _vehicle(chain)
    vehicle.start_control()
    try:
        start = time.monotonic()
        while time.monotonic() - start < 4.0:
            # The same 90 degree direction change that trips a stalled caster.
            vehicle.set_target_velocity(_command(time.monotonic() - start), frame="local")
            assert vehicle.caster_fault() is None
            time.sleep(0.02)
        assert vehicle.control_loop_running is True
    finally:
        vehicle.close()


def test_the_history_csv_is_available_after_a_fault(_fast_startup: None) -> None:
    chain = FakeChain(stalled=np.array([False, True, False, False]))
    vehicle = _vehicle(chain)
    vehicle.start_control()
    try:
        assert _drive_until_fault(vehicle) is not None
        csv = vehicle.caster_history_csv()
        assert csv is not None
        header, *rows = csv.strip().splitlines()
        assert header.startswith("t,vx,vy,w,")
        assert len(rows) > 10
    finally:
        vehicle.close()


def test_wait_for_stop_does_not_join_itself() -> None:
    # control_loop must never join the thread it is running on; that raises RuntimeError.
    chain = FakeChain()
    vehicle = _vehicle(chain)
    vehicle.control_loop_thread = threading.current_thread()
    vehicle.wait_for_stop(timeout=0.01)  # must simply return
