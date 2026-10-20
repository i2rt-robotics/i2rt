"""The two idle regimes a viewer can put a robot in: gravity-comp, and literally zero torque.

``MotorChainRobot.update`` sends ``commands.torques + g * gravity_comp_factor``, so zeroing the
command alone is not zero torque -- gravity compensation keeps flowing. Both halves have to move
together, which is what these tests pin. The distinction matters at the motor: one regime holds the
arm up, the other lets it fall.
"""

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmType, GripperType

N_MOTORS = 7


def _bare_robot(gravity_comp_available: bool = True) -> Any:
    """A MotorChainRobot with only the attributes the idle helpers touch.

    Built with ``__new__`` rather than a real chain, following the idiom in
    ``test_motor_chain_recording.py``: the constructor opens a CAN socket.
    """
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot.motor_chain = [None] * N_MOTORS  # len() is all these helpers need
    robot._arm_type = ArmType.YAM
    robot._gripper_type = GripperType.LINEAR_4310
    robot._grav_comp_kd = np.full(N_MOTORS, 0.1)
    robot._kp = np.full(N_MOTORS, 80.0)
    robot._kd = np.full(N_MOTORS, 5.0)
    robot.use_gravity_comp = gravity_comp_available
    robot._gravity_comp_available = gravity_comp_available
    robot._command_lock = threading.Lock()
    return robot


def test_zero_torque_idle_commands_nothing() -> None:
    """Zero torque, zero PD gains, and gravity comp off -- the frame dm_driver.py's loop sends."""
    robot = _bare_robot()
    robot.enter_zero_torque_idle()

    assert robot.use_gravity_comp is False, "gravity comp still on: the motors would hold the arm"
    for field in ("torques", "pos", "vel", "kp", "kd"):
        assert np.array_equal(getattr(robot._commands, field), np.zeros(N_MOTORS)), field
    # What update() would add on top of the command, with the arm at any pose.
    assert np.array_equal(
        robot._compute_gravity_compensation(SimpleNamespace(pos=np.zeros(N_MOTORS))), np.zeros(N_MOTORS)
    )


def test_gravity_comp_idle_is_the_way_back() -> None:
    """Re-entering gravity-comp idle restores both halves, so the arm is held again."""
    robot = _bare_robot()
    robot.enter_zero_torque_idle()
    robot.enter_gravity_comp_idle()

    assert robot.use_gravity_comp is True
    assert np.array_equal(robot._commands.torques, np.zeros(N_MOTORS))
    assert np.array_equal(robot._commands.kd, robot._grav_comp_kd), "grav-comp idle keeps its damping"


def test_gravity_comp_idle_cannot_resurrect_it_without_a_model() -> None:
    """A robot built without an XML path has no KDL, so it must stay off."""
    robot = _bare_robot(gravity_comp_available=False)
    robot.enter_zero_torque_idle()
    robot.enter_gravity_comp_idle()
    assert robot.use_gravity_comp is False


@pytest.mark.parametrize("arm", [ArmType.YAM, ArmType.YAM_7DOF_RIGHT], ids=lambda a: a.value)
def test_sim_zero_torque_lets_the_arm_fall(arm: ArmType) -> None:
    """In sim the regime is visible as motion: held under gravity comp, falling without it.

    ``disable_gravity_comp`` freezes physics instead (CONTROL mode teleports), so it cannot stand in
    for this -- a frozen arm would show the operator a still picture of a limb that is really
    dropping.
    """
    window = 0.4

    def drift(robot: Any) -> float:
        """Largest joint movement over one window, in rad."""
        before = robot.get_joint_pos().copy()
        time.sleep(window)
        return float(np.max(np.abs(robot.get_joint_pos() - before)))

    robot = get_yam_robot(arm_type=arm, gripper_type=GripperType.LINEAR_4310, sim=True)
    try:
        robot.start_server()
        held = drift(robot)
        assert held < 1e-3, f"gravity comp should hold the arm still, moved {held:.2e} rad"

        # Falling from rest, so the first window only catches the start of the acceleration --
        # compared against the held drift rather than a bare constant, which keeps the margin
        # meaningful whatever the arm weighs.
        robot.enter_zero_torque_idle()
        falling = drift(robot)
        assert falling > 5e-3 and falling > 10 * held, (
            f"zero torque should let the arm fall: moved {falling:.2e} rad vs {held:.2e} held"
        )

        # The way back. Asserted as state rather than motion on purpose: the sim's gravity comp
        # cancels gravity but adds no damping term, so an arm that was already moving coasts at
        # roughly constant velocity instead of stopping. Position alone therefore cannot show the
        # switch, and a drift-based check here would be testing something the sim does not do.
        # (A real chain does settle: enter_gravity_comp_idle also sets kd = grav_comp_kd.)
        robot.enable_gravity_comp()
        assert robot._zero_torque is False, "re-enabling gravity comp should leave zero-torque mode"
    finally:
        robot.close()
