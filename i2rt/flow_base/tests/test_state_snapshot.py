"""Regression tests for the per-control-cycle state snapshot (ROB record-skew fix).

The base control loop reads the motor chain once per cycle in ``update_state()`` and integrates
odometry from that sample. The recording ``state_callback`` then reads wheel state (and, with a rail,
rail state) back out. Before this fix those getters did a *fresh* ``read_states()``, which -- because
``set_velocities`` waits a CAN cycle on ``command_lock`` -- is usually the next sample: odometry came
from sample k while wheel/rail came from k+1, yet the recorder stamped all three with one timestamp.

These tests drive the real control-cycle methods against a fake chain whose ``read_states()`` returns
a new sample on every call (standing in for the CAN reader thread advancing). The guard is that the
three channels a ``state_callback`` reads all come from the single sample ``update_state`` read.
Without the fix ``get_wheel_states`` / ``get_linear_rail_state`` re-read and the equality assertions
fail.
"""

import threading

import numpy as np
import pytest

from i2rt.flow_base import flow_base_controller
from i2rt.flow_base.flow_base_controller import LinearRailVehicle, Vehicle
from i2rt.flow_base.linear_rail_controller import LinearRailController
from i2rt.motor_drivers.utils import MotorInfo


class FakeChain:
    """Minimal DMChainCanInterface stand-in: each read_states() reports a fresh, newer sample.

    pos == vel == eff == an integer sample counter, so the sample a reader got is recoverable from any
    field. The counter advances on every read, modelling the CAN thread publishing between two reads.
    """

    def __init__(self, num_motors: int) -> None:
        self.num_motors = num_motors
        self.motor_offset = np.zeros(num_motors)
        self.motor_direction = np.ones(num_motors)
        self.command_lock = threading.Lock()
        self.commands: list = []
        self.running = True
        self.sample = 0
        self.reads = 0

    def __len__(self) -> int:
        return self.num_motors

    def read_states(self, torques=None) -> list:  # noqa: ANN001
        states = [
            MotorInfo(
                id=i + 1, error_code="0x1", pos=float(self.sample), vel=float(self.sample), eff=float(self.sample)
            )
            for i in range(self.num_motors)
        ]
        self.sample += 1
        self.reads += 1
        return states


@pytest.fixture
def _no_pid_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # Vehicle.__init__ enforces a single hardware instance via /tmp/base-controller.pid; under
    # pytest-xdist that collides across workers. We never touch hardware here, so stub it out.
    monkeypatch.setattr(flow_base_controller, "create_pid_file", lambda name: None)


def _make_vehicle(chain: FakeChain) -> Vehicle:
    return Vehicle(
        channel=chain,  # a non-str channel is adopted as-is; nothing opens CAN
        auto_start=False,  # no control thread: we step the cycle by hand
        verify_motor_config=False,
        check_caster_steering=False,
    )


def test_wheel_state_matches_odometry_sample_in_callback(_no_pid_file: None) -> None:
    chain = FakeChain(num_motors=8)
    vehicle = _make_vehicle(chain)

    recorded: list[tuple[float, float]] = []

    def record() -> None:
        # Odometry channel: the sample update_state fed into the integrator is exactly self.dq
        # (motor_direction is 1, so dq == the sample's vel). Wheel channel: what the recorder reads.
        odometry_sample = vehicle.dq[1]
        wheel_sample = vehicle.get_wheel_states()["drive"]["vel"][0]
        recorded.append((odometry_sample, wheel_sample))

    vehicle._state_callback = record

    # Three hand-run control cycles: read + integrate, then fire the callback, like control_loop.
    for _ in range(3):
        vehicle.update_state(vehicle.control_period)
        vehicle._notify_state()

    # 1:1 within a cycle: odometry and wheel come from the same sample (this is the whole fix).
    for odometry_sample, wheel_sample in recorded:
        assert odometry_sample == wheel_sample

    # Ordered across cycles: each recorded frame is a distinct, strictly-increasing sample.
    odometry_samples = [o for o, _ in recorded]
    assert odometry_samples == sorted(set(odometry_samples))
    assert len(odometry_samples) == 3


def test_snapshot_cleared_outside_callback(_no_pid_file: None) -> None:
    # Outside the callback window get_wheel_states must read live, not serve a stale snapshot.
    chain = FakeChain(num_motors=8)
    vehicle = _make_vehicle(chain)
    vehicle.update_state(vehicle.control_period)
    assert vehicle._callback_snapshot is None

    before = chain.reads
    vehicle.get_wheel_states()
    assert chain.reads == before + 1  # a real live read happened


class _FakeRailInterface:
    """Stands in for SingleMotorControlInterface: a live get_state() plus the chain index."""

    def __init__(self, target_motor_idx: int) -> None:
        self.target_motor_idx = target_motor_idx
        self.live_reads = 0

    def get_state(self) -> MotorInfo:
        self.live_reads += 1
        return MotorInfo(id=self.target_motor_idx + 1, error_code="0x1", pos=-1.0, vel=-1.0, eff=-1.0)


def test_rail_state_uses_snapshot_during_callback(_no_pid_file: None) -> None:
    # The rail is the third recorded channel. Build only the rail plumbing (no GPIO / homing / CAN):
    # a LinearRailVehicle shell plus a real LinearRailController over a fake single-motor interface.
    rail_idx = 8
    iface = _FakeRailInterface(target_motor_idx=rail_idx)
    controller = LinearRailController(iface, auto_home=False)
    controller.meters_per_rad = 0.001  # mark calibrated so linear units are produced

    vehicle = LinearRailVehicle.__new__(LinearRailVehicle)
    vehicle.linear_rail = controller
    vehicle._callback_snapshot = None

    # Outside the callback: reads the rail live.
    live = vehicle.get_linear_rail_state()
    assert iface.live_reads == 1
    assert live["position"]["motor"] == -1.0

    # Inside the callback window: the cached sample's rail motor (index 8) is reported, no live read.
    snapshot = [
        MotorInfo(id=i + 1, error_code="0x1", pos=float(i), vel=float(i), eff=float(i)) for i in range(rail_idx + 1)
    ]
    vehicle._callback_snapshot = snapshot
    snap_state = vehicle.get_linear_rail_state()
    assert iface.live_reads == 1  # unchanged: served from the snapshot
    assert snap_state["position"]["motor"] == float(rail_idx)
    assert snap_state["eff"] == float(rail_idx)
