import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import JointCommands, JointStates, MotorChainRobot
from i2rt.robots.utils import GripperType, JointMapper


def _robot() -> MotorChainRobot:
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot.motor_chain = [None, None]
    robot._command_lock = threading.Lock()
    robot._state_lock = threading.Lock()
    robot._commands = JointCommands.init_all_zero(2)
    robot._kp, robot._kd = np.ones(2), np.ones(2)
    robot._joint_limits = None
    robot.remapper = JointMapper({}, 2)
    robot._joint_state = JointStates([], np.ones(2), np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2), 0.0)
    return robot


def test_robot_commands_and_observations_own_their_arrays() -> None:
    robot = _robot()
    pos, vel, kp, kd = [np.ones(2) for _ in range(4)]
    robot.update_kp_kd(kp, kd)
    robot.command_joint_state({"pos": pos, "vel": vel})
    for value in (pos, vel, kp, kd, robot.get_joint_pos()):
        value[:] = 99.0
    for value in (robot._commands.pos, robot._commands.vel, robot._commands.kp, robot._commands.kd):
        np.testing.assert_array_equal(value, np.ones(2))
    np.testing.assert_array_equal(robot._joint_state.pos, np.ones(2))


def test_joint_state_command_is_not_replaced_before_taking_the_lock() -> None:
    robot = _robot()
    lock, attempted = threading.Lock(), threading.Event()

    class ObservedLock:
        def __enter__(self) -> None:
            attempted.set()
            lock.acquire()

        def __exit__(self, *_args: object) -> None:
            lock.release()

    robot._command_lock = ObservedLock()
    previous = robot._commands
    with ThreadPoolExecutor(1) as pool:
        with lock:
            future = pool.submit(robot.command_joint_state, {"pos": np.ones(2), "vel": np.zeros(2)})
            assert attempted.wait(1.0)
            assert robot._commands is previous
        future.result(timeout=1.0)
    np.testing.assert_array_equal(robot._commands.pos, np.ones(2))


def test_robot_readers_are_not_blocked_by_motor_io() -> None:
    robot = _robot()
    robot._compute_gravity_compensation = lambda _state: np.zeros(2)
    robot.use_coulomb_friction = False
    robot.gravity_comp_factor = np.ones(2)
    robot._clip_motor_torque = np.inf
    robot._gripper_index = None
    entered, release = threading.Event(), threading.Event()

    def motor_io(*_args: object) -> None:
        entered.set()
        assert release.wait(2.0)

    robot._update_joint_state = motor_io
    with ThreadPoolExecutor(2) as pool:
        writer = pool.submit(robot.update)
        try:
            assert entered.wait(1.0)
            np.testing.assert_array_equal(pool.submit(robot.get_joint_pos).result(timeout=0.2), np.ones(2))
        finally:
            release.set()
        writer.result(timeout=1.0)


def test_sim_joint_state_is_published_atomically(monkeypatch: pytest.MonkeyPatch) -> None:
    robot = get_yam_robot(gripper_type=GripperType.NO_GRIPPER, sim=True)
    entered, release = threading.Event(), threading.Event()
    original = robot.command_joint_pos

    def set_position(pos: np.ndarray) -> None:
        original(pos)
        entered.set()
        assert release.wait(2.0)

    monkeypatch.setattr(robot, "command_joint_pos", set_position)
    command = {"pos": np.full(6, 0.2), "vel": np.full(6, 0.3)}
    with ThreadPoolExecutor(2) as pool:
        writer = pool.submit(robot.command_joint_state, command)
        try:
            assert entered.wait(1.0)
            reader = pool.submit(robot.get_joint_state)
            with pytest.raises(TimeoutError):
                reader.result(timeout=0.1)
        finally:
            release.set()
            robot.close()
        writer.result(timeout=1.0)
        for key, value in reader.result(timeout=1.0).items():
            np.testing.assert_array_equal(value, command[key])
