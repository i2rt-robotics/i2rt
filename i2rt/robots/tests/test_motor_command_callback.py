import threading
from types import SimpleNamespace
from typing import Any, List, Tuple, cast

import numpy as np

from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot, MotorCommandSample


def _make_robot(n_joints: int, *, kp: float, kd: float) -> Tuple[Any, np.ndarray]:
    """A MotorChainRobot wired just enough to run update() once, without a CAN bus.

    ``__new__`` skips __init__ (which would start control threads and read motors); we set only the
    attributes update() touches and stub the gravity model and _update_joint_state.
    """
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._command_lock = threading.Lock()
    robot._state_lock = threading.Lock()
    robot._commands = JointCommands.init_all_zero(n_joints)
    robot._commands.kp = np.full(n_joints, kp)
    robot._commands.kd = np.full(n_joints, kd)
    robot._coulomb_friction = np.zeros(n_joints)
    robot.use_coulomb_friction = False
    robot.gravity_comp_factor = np.ones(n_joints)
    robot._transmissions = {}
    robot._clip_motor_torque = np.inf
    robot._gripper_index = None
    robot._last_motor_torques = None

    gravity = np.arange(1.0, n_joints + 1.0)
    robot._compute_gravity_compensation = lambda _state: gravity
    robot._joint_state = SimpleNamespace(pos=np.zeros(n_joints), vel=np.zeros(n_joints), timestamp=0.0)

    def fake_update_joint_state(motor_torques: np.ndarray, joint_commands: JointCommands) -> None:
        # set_commands() would refresh self._joint_state (and its timestamp) here.
        robot._joint_state = SimpleNamespace(
            pos=np.zeros(n_joints), vel=np.zeros(n_joints), timestamp=robot._joint_state.timestamp + 1.0
        )

    robot._update_joint_state = fake_update_joint_state
    return robot, gravity


def test_callback_fires_once_per_update_with_the_sent_command() -> None:
    n = 7
    robot, gravity = _make_robot(n, kp=80.0, kd=5.0)
    samples: List[MotorCommandSample] = []
    robot._motor_command_callback = samples.append

    MotorChainRobot.update(robot)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.timestamp == 1.0  # the post-_update_joint_state stamp, shared with robot-state
    assert np.array_equal(sample.kp, np.full(n, 80.0))
    assert np.array_equal(sample.kd, np.full(n, 5.0))
    assert np.array_equal(sample.gravity_torque, gravity)  # g * gravity_comp_factor (ones)
    assert np.array_equal(sample.friction_torque, np.zeros(n))  # Coulomb friction disabled
    # No clip and no commanded torque, so the sent torque is exactly the two components.
    assert np.allclose(sample.torque, sample.gravity_torque + sample.friction_torque)


def test_callback_tracks_the_kp_kd_switch() -> None:
    # Grav-comp idle holds kp=0; command_joint_pos switches to the configured gains. The sample must
    # carry whichever is active that cycle.
    n = 7
    robot, _ = _make_robot(n, kp=0.0, kd=0.3)
    samples: List[MotorCommandSample] = []
    robot._motor_command_callback = samples.append

    MotorChainRobot.update(robot)  # idle
    robot._commands.kp = np.full(n, 80.0)
    robot._commands.kd = np.full(n, 5.0)
    MotorChainRobot.update(robot)  # link engaged

    assert len(samples) == 2
    assert np.array_equal(samples[0].kp, np.zeros(n))
    assert np.array_equal(samples[1].kp, np.full(n, 80.0))
    assert np.array_equal(samples[1].kd, np.full(n, 5.0))


def test_torque_is_post_clip_while_gravity_is_pre_clip() -> None:
    # The sent torque is clamped by the motor-side clip; gravity_torque/friction_torque are the
    # pre-clip joint-space components, so the two deliberately diverge once the clip fires. A
    # consumer must not assume torque == gravity + friction in general.
    n = 7
    robot, gravity = _make_robot(n, kp=0.0, kd=0.3)
    robot._clip_motor_torque = 2.0  # gravity is [1..7], so joints 3..7 saturate
    samples: List[MotorCommandSample] = []
    robot._motor_command_callback = samples.append

    MotorChainRobot.update(robot)

    sample = samples[0]
    assert np.array_equal(sample.gravity_torque, gravity)  # recorded unclipped
    assert np.allclose(sample.torque, np.clip(gravity, -2.0, 2.0))  # sent clipped
    assert not np.allclose(sample.torque, sample.gravity_torque)  # they diverge under clip


def test_update_is_a_noop_without_a_callback() -> None:
    robot, _ = _make_robot(7, kp=80.0, kd=5.0)
    robot._motor_command_callback = None
    MotorChainRobot.update(robot)  # must not raise
