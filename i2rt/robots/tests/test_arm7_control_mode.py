"""The 7-DOF arms' ``control_mode``: what ``get_yam_robot`` builds for each, and the two things the
OSC path had to learn on this branch -- joint6's four-bar and a position-controlled gripper.

Sim-only, like the rest of the suite. The factory tests run the real control thread against
``SimMotorChain``; the four-bar test steps the loop by hand so it is deterministic.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import pytest

from i2rt.motor_drivers.utils import MotorInfo
from i2rt.robots.arm7_osc import ARM_DOF, COMMAND_TIMEOUT_S, DEFAULT_MOTOR_LIST, Arm7OSC, ArmState, arm7_xml
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.sim_motor_chain import SimMotorChain
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config

JOINT6_INDEX = 5
YAM_7DOF_JOINT6_LINKAGE = _load_arm_config(ArmType.YAM_7DOF_RIGHT).transmissions[JOINT6_INDEX]
Q_MID = np.array([0.3, 0.9, 0.4, -1.2, 0.3, -0.5, 0.2])


def _wait_until(done: "object", timeout: float = 15.0) -> None:
    """Poll ``done()``: the loop runs in wall time, which a loaded CI machine stretches."""
    deadline = time.monotonic() + timeout
    while not done():  # type: ignore[operator]
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


@pytest.mark.parametrize(
    "arm_type, gripper_type",
    [
        (ArmType.YAM, GripperType.LINEAR_4310),
        (ArmType.YAM_ULTRA_2, GripperType.LINEAR_4310),
        (ArmType.YAM_7DOF_RIGHT, GripperType.YAM_TEACHING_HANDLE_V1),
    ],
)
def test_osc_is_refused_where_it_cannot_run(arm_type: ArmType, gripper_type: GripperType) -> None:
    with pytest.raises(ValueError, match="control_mode"):
        get_yam_robot(arm_type=arm_type, gripper_type=gripper_type, sim=True, control_mode="osc")


@pytest.mark.parametrize("control_mode", ["osc", "osc_full"])
def test_osc_robot_tracks_the_arm_and_the_gripper_like_a_pd_robot(control_mode: str) -> None:
    """Same calls as a PD robot: one command engages the arm, holds without a stream, and the
    trailing entry drives the gripper.

    Starts at ``Q_MID`` because ``osc_full`` tracks the tool pose and reaches the joint target only
    through the nullspace: from the singular home pose it can settle in another IK branch.
    """
    robot = get_yam_robot(
        arm_type=ArmType.YAM_7DOF_RIGHT, gripper_type=GripperType.LINEAR_4310, sim=True, control_mode=control_mode
    )
    try:
        assert isinstance(robot, Arm7OSC)
        assert robot.num_dofs() == ARM_DOF + 1
        assert robot.state is ArmState.FLOATING
        robot.motor_chain.place(np.concatenate([Q_MID, [0.0, 0.0]]))

        target = np.append(Q_MID, 0.8)
        robot.command_joint_pos(target)
        assert robot.state is ArmState.ACTIVE
        # Tight, because the sim gripper rings: its motor gains are Nm/rad on hardware and land on
        # a finger slide in metres here, which leaves it underdamped for a couple of seconds.
        _wait_until(lambda: np.abs(robot.get_joint_pos() - target).max() < 2e-3)

        time.sleep(2 * COMMAND_TIMEOUT_S)  # no command stream: a held target must not time out
        assert robot.state is ArmState.ACTIVE, robot.fault
        obs = robot.get_observations()
        assert obs["gripper_pos"].shape == (1,)
        np.testing.assert_allclose(obs["gripper_pos"], 0.8, atol=1e-2)
    finally:
        robot.close()
    assert not robot.motor_chain.running, "close() left the chain it was built with running"


def test_osc_robot_without_a_gripper_has_seven_dofs() -> None:
    robot = get_yam_robot(
        arm_type=ArmType.YAM_7DOF_LEFT, gripper_type=GripperType.NO_GRIPPER, sim=True, control_mode="osc"
    )
    try:
        assert robot.num_dofs() == ARM_DOF
        assert "gripper_pos" not in robot.get_observations()
    finally:
        robot.close()


class _FourBarChain(SimMotorChain):
    """``SimMotorChain`` with joint6 behind the real linkage: reports the motor's angle and rate,
    and applies MIT torque and kd as the motor would through it."""

    def _ratio(self) -> float:
        motor = YAM_7DOF_JOINT6_LINKAGE.motor_from_joint(self.data.qpos[JOINT6_INDEX])
        return YAM_7DOF_JOINT6_LINKAGE.ratio(motor)

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> Optional[List[MotorInfo]]:
        ratio = self._ratio()
        torques = np.array(torques, dtype=float)
        torques[JOINT6_INDEX] /= ratio  # tau_joint = tau_motor / ratio
        if kd is not None:
            kd = np.array(kd, dtype=float)
            kd[JOINT6_INDEX] /= ratio**2  # motor damping seen at the joint
        return super().set_commands(torques, pos, vel, kp, kd, get_state)

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        infos = super().read_states(torques)
        info = infos[JOINT6_INDEX]
        motor = YAM_7DOF_JOINT6_LINKAGE.motor_from_joint(info.pos)
        info.vel /= YAM_7DOF_JOINT6_LINKAGE.ratio(motor)
        info.pos = motor
        return infos


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
def test_osc_holds_joint6_only_through_its_declared_linkage(declared: bool) -> None:
    """Behind the four-bar, ``Arm7OSC`` must hold the joint where it was told -- which it can only
    do when told about the linkage. Undeclared, it reads the motor's angle as the joint's, so it
    aims at the wrong joint6, and its joint6 loop gain comes out ~1/ratio^2 too high: either it
    holds off-target or it rings up into the velocity guard."""
    xml = arm7_xml("right")
    chain = _FourBarChain(xml, DEFAULT_MOTOR_LIST, initial_qpos=np.concatenate([Q_MID, [0.0, 0.0]]))
    arm = Arm7OSC(
        chain,
        xml,
        auto_start=False,
        command_timeout_s=None,
        transmissions={JOINT6_INDEX: YAM_7DOF_JOINT6_LINKAGE} if declared else None,
    )
    try:
        arm._step()  # first feedback
        arm.enable_arm()
        arm.command_joint_pos(Q_MID)
        arm._enable_time = 0.0  # skip the ramp
        for _ in range(1500):  # 3 s at 500 Hz
            arm._step()
        joint6 = chain.get_true_state()["pos"][JOINT6_INDEX]
        if declared:
            assert arm.state is ArmState.ACTIVE, arm.fault
            assert abs(joint6 - Q_MID[JOINT6_INDEX]) < 2e-3
            np.testing.assert_allclose(arm.get_joint_pos(), chain.get_true_state()["pos"][:ARM_DOF], atol=2e-3)
        else:
            assert arm.state is not ArmState.ACTIVE or abs(joint6 - Q_MID[JOINT6_INDEX]) > 5e-2
    finally:
        arm.close()
