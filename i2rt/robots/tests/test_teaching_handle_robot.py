"""A teaching handle with no arm under it is a sensor, not a one-DOF gripper.

``--arm no_arm --gripper yam_teaching_handle_v1`` used to take the motorized-gripper path and
try to bring up a motor at CAN id 0x07, which the handle does not have.  It has no motor at
all: only the passive trigger encoder, the same one an arm-mounted handle carries.  These
tests pin the resulting shape -- zero commandable DOFs, no gripper index, and the trigger
published as a normalized ``gripper_pos`` -- and keep the motorized grippers on their own
path.

The real-hardware tests run against a fake CAN bus and a scripted encoder, so the suite
stays sim-only.
"""

import time
from typing import Any, List, Optional

import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import PassiveEncoderInfo
from i2rt.robots import get_robot, teaching_handle_robot
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.teaching_handle_robot import TeachingHandleRobot
from i2rt.robots.utils import ArmType, GripperType

HANDLES = [GripperType.YAM_TEACHING_HANDLE_V1, GripperType.YAM_TEACHING_HANDLE_V2]


class _FakeCanInterface:
    """Stands in for the socketcan bus: opens nothing, records that it was closed."""

    def __init__(self, channel: str, name: str = "fake", use_buffered_reader: bool = False) -> None:
        self.channel = channel
        self.name = name
        self.closed = False

    def _drain_bus(self, timeout_s: float = 0.05, idle_count: int = 10) -> int:
        return 0

    def close(self) -> None:
        self.closed = True


class _FakeEncoderChain:
    """Reports one fixed trigger position, optionally going dead after N reads."""

    def __init__(self, position: float, buttons: List[bool], fail_after: Optional[int]) -> None:
        self.position = position
        self.buttons = buttons
        self.fail_after = fail_after
        self.reads = 0

    def read_states(self) -> List[PassiveEncoderInfo]:
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise AssertionError("fail to communicate with the encoder")
        return [PassiveEncoderInfo(id=0x50E, position=self.position, velocity=0.0, io_inputs=self.buttons)]


@pytest.fixture
def fake_bus(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point the real-hardware path at a fake bus; returns a builder for the encoder."""
    monkeypatch.setattr(teaching_handle_robot, "CanInterface", _FakeCanInterface)

    def install(
        position: float = 0.0,
        buttons: Optional[List[bool]] = None,
        fail_after: Optional[int] = None,
    ) -> _FakeEncoderChain:
        chain = _FakeEncoderChain(position, [False, False] if buttons is None else buttons, fail_after)
        monkeypatch.setattr(get_robot, "get_encoder_chain", lambda _can_interface: chain)
        return chain

    return install


# ---------------------------------------------------------------------------
# Shape of the robot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gripper_type", HANDLES)
def test_real_handle_only_robot_has_no_motors(gripper_type: GripperType, fake_bus: Any) -> None:
    """The handle is passive, so nothing is commandable and no motor chain is brought up."""
    fake_bus()
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=gripper_type, sim=False)
    try:
        assert isinstance(robot, TeachingHandleRobot)
        assert robot.num_dofs() == 0
        assert len(robot.motor_chain) == 0
        assert robot.get_joint_pos().shape == (0,)
        info = robot.get_robot_info()
        assert info["gripper_index"] is None
        # No gains to show: control interfaces gate their PD panel on these keys.
        assert "kp" not in info and "kd" not in info
    finally:
        robot.close()


@pytest.mark.parametrize("gripper_type", HANDLES)
def test_sim_handle_only_robot_matches_the_real_one(gripper_type: GripperType) -> None:
    """--sim previews the same robot: no commandable DOF, trigger at rest = open."""
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=gripper_type, sim=True)
    try:
        assert robot.num_dofs() == 0
        assert robot.get_robot_info()["gripper_index"] is None
        assert robot.get_observations()["gripper_pos"][0] == pytest.approx(1.0)
    finally:
        robot.close()


@pytest.mark.parametrize("arm_type", [ArmType.YAM, ArmType.NO_ARM])
@pytest.mark.parametrize("gripper_type", HANDLES)
def test_sim_trigger_stays_at_rest_once_physics_runs(arm_type: ArmType, gripper_type: GripperType) -> None:
    """The unactuated trigger must not fall under gravity and read as squeezed."""
    robot = get_yam_robot(arm_type=arm_type, gripper_type=gripper_type, sim=True)
    try:
        robot.start_server()
        time.sleep(0.5)
        assert robot.get_observations()["gripper_pos"][0] == pytest.approx(1.0)
    finally:
        robot.close()


def test_motorized_gripper_only_robot_still_has_a_gripper_dof() -> None:
    """The handle branch must not swallow the grippers that really do have a motor."""
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.LINEAR_4310, sim=True)
    try:
        assert robot.num_dofs() == 1
        assert robot.get_robot_info()["gripper_index"] == 0
    finally:
        robot.close()


# ---------------------------------------------------------------------------
# What it reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_position,expected",
    [
        (0.0, 1.0),  # at rest      -> OPEN
        (0.5, 0.5),  # half squeeze
        (1.0, 0.0),  # full squeeze -> CLOSED
    ],
)
def test_gripper_pos_is_reported_in_the_arm_convention(raw_position: float, expected: float, fake_bus: Any) -> None:
    """A consumer must not be able to tell this handle from an arm-mounted one."""
    fake_bus(position=raw_position)
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.YAM_TEACHING_HANDLE_V1, sim=False)
    try:
        obs = robot.get_observations()
        assert obs["joint_pos"].shape == (0,)
        assert obs["gripper_pos"] == pytest.approx([expected])
    finally:
        robot.close()


def test_buttons_reach_consumers_through_the_chain(fake_bus: Any) -> None:
    """Every teaching-handle consumer reads buttons off robot.motor_chain, not the robot."""
    fake_bus(buttons=[True, False])
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.YAM_TEACHING_HANDLE_V1, sim=False)
    try:
        assert robot.motor_chain.same_bus_device_driver is not None
        states = robot.motor_chain.get_same_bus_device_states()
        assert list(states[0].io_inputs) == [True, False]
    finally:
        robot.close()


def test_empty_commands_are_silent_and_anything_else_raises(fake_bus: Any) -> None:
    """Control interfaces command every tick; with zero DOFs that must stay harmless.

    A real command, or a request to record or move, is a caller aimed at the wrong robot.
    """
    fake_bus()
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.YAM_TEACHING_HANDLE_V1, sim=False)
    try:
        robot.command_joint_pos(np.zeros(0))
        robot.command_joint_state({"pos": np.zeros(0), "vel": np.zeros(0)})
        assert robot.get_observations()["gripper_pos"] == pytest.approx([1.0])
        with pytest.raises(ValueError, match="0 DOFs"):
            robot.command_joint_pos(np.zeros(6))
        with pytest.raises(ValueError, match="0 DOFs"):
            robot.command_joint_state({"pos": np.zeros(1)})
        with pytest.raises(NotImplementedError, match="passive"):
            robot.start_mcap_recording()
        with pytest.raises(NotImplementedError, match="passive"):
            robot.move_joints(np.zeros(6))
    finally:
        robot.close()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_close_stops_the_poll_thread_and_the_bus(fake_bus: Any) -> None:
    fake_bus()
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.YAM_TEACHING_HANDLE_V1, sim=False)
    chain = robot.motor_chain
    robot.close()
    assert not chain.running
    assert not chain._thread.is_alive()
    assert chain.can_interface.closed


def test_a_failed_construction_closes_the_bus(fake_bus: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller that retries construction must not leak a socket per attempt."""
    fake_bus(fail_after=0)  # the construction read fails
    opened: List[_FakeCanInterface] = []

    def open_bus(**kwargs: Any) -> _FakeCanInterface:
        opened.append(_FakeCanInterface(**kwargs))
        return opened[-1]

    monkeypatch.setattr(teaching_handle_robot, "CanInterface", open_bus)
    with pytest.raises(AssertionError, match="fail to communicate"):
        get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.YAM_TEACHING_HANDLE_V1, sim=False)
    assert len(opened) == 1 and opened[0].closed


def test_a_dead_encoder_raises_instead_of_freezing(fake_bus: Any) -> None:
    """An unplugged handle must surface, not leave the viewer mirroring a stale snapshot."""
    fake_bus(fail_after=1)  # the construction read succeeds; the first poll does not
    robot = get_yam_robot(arm_type=ArmType.NO_ARM, gripper_type=GripperType.YAM_TEACHING_HANDLE_V1, sim=False)
    try:
        deadline = time.time() + 5.0
        while robot.motor_chain.running and time.time() < deadline:
            time.sleep(0.01)
        assert not robot.motor_chain.running, "the poll loop should stop once the encoder stops answering"
        with pytest.raises(RuntimeError, match="no longer readable"):
            robot.get_joint_pos()
    finally:
        robot.close()
