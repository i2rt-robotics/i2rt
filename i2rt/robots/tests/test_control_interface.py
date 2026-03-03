"""Integration tests for MujocoControlInterface using SimRobot.

Exercises VIS mode (mirror, sync), CONTROL mode (IK, command), and
dimension handling for all arm/gripper combinations.
"""

import numpy as np
import pytest

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.sim_robot import SimRobot
from i2rt.robots.utils import ArmType, GripperType
from i2rt.utils.mujoco_control_interface import MujocoControlInterface

GRIPPER_COMBOS = [
    (GripperType.CRANK_4310, "grasp_site"),
    (GripperType.LINEAR_4310, "grasp_site"),
    (GripperType.LINEAR_3507, "grasp_site"),
    (GripperType.NO_GRIPPER, "grasp_site"),
    (GripperType.YAM_TEACHING_HANDLE, "tcp_site"),
]


def _combo_id(val: object) -> str:
    if isinstance(val, tuple):
        return val[0].value
    return str(val)


def _make_iface(gripper: GripperType, site: str) -> tuple[SimRobot, MujocoControlInterface]:
    robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=gripper, sim=True)
    assert isinstance(robot, SimRobot)
    iface = MujocoControlInterface(robot, robot.xml_path, ee_site=site)
    return robot, iface


@pytest.mark.parametrize("gripper,site", GRIPPER_COMBOS, ids=_combo_id)
def test_vis_mode_mirror(gripper: GripperType, site: str) -> None:
    """VIS mode: mirror_robot should set qpos and forward without error."""
    _robot, iface = _make_iface(gripper, site)
    iface._mirror_robot()
    iface._sync_mocap_to_ee()


@pytest.mark.parametrize("gripper,site", GRIPPER_COMBOS, ids=_combo_id)
def test_vis_mode_with_nonzero_joints(gripper: GripperType, site: str) -> None:
    """VIS mode should handle non-zero joint positions correctly."""
    robot, iface = _make_iface(gripper, site)
    q = np.zeros(robot.num_dofs())
    q[: min(3, len(q))] = [0.1, 0.5, 0.3]
    robot.command_joint_pos(q)
    iface._mirror_robot()
    iface._sync_mocap_to_ee()
    grasp_pos = iface._data.site(iface._ee_site_id).xpos.copy()
    assert np.all(np.isfinite(grasp_pos))


@pytest.mark.parametrize("gripper,site", GRIPPER_COMBOS, ids=_combo_id)
def test_control_mode_ik_dimensions(gripper: GripperType, site: str) -> None:
    """CONTROL mode: IK should run without dimension mismatch."""
    robot, iface = _make_iface(gripper, site)
    iface._mirror_robot()
    iface._sync_mocap_to_ee()

    target = iface._mocap_pose_4x4()
    init_q = iface._data.qpos[: iface._nq].copy()
    _ok, ik_q = iface._kin.ik(target, site, init_q=init_q, max_iters=10)

    assert ik_q.shape[0] == iface._nq
    cmd = robot.get_joint_pos().copy()
    cmd[: iface._n_arm] = ik_q[: iface._n_arm]
    robot.command_joint_pos(cmd)
    assert cmd.shape[0] == robot.num_dofs()


@pytest.mark.parametrize("gripper,site", GRIPPER_COMBOS, ids=_combo_id)
def test_control_mode_preserves_gripper(gripper: GripperType, site: str) -> None:
    """CONTROL mode should not overwrite the gripper joint value."""
    robot, iface = _make_iface(gripper, site)
    n_dofs = robot.num_dofs()
    if robot._gripper_index is None:
        pytest.skip("No gripper joint to test")

    q = np.zeros(n_dofs)
    q[robot._gripper_index] = -1.0
    robot.command_joint_pos(q)
    gripper_val = robot.get_joint_pos()[robot._gripper_index]

    iface._mirror_robot()
    iface._sync_mocap_to_ee()

    target = iface._mocap_pose_4x4()
    init_q = iface._data.qpos[: iface._nq].copy()
    _ok, ik_q = iface._kin.ik(target, site, init_q=init_q, max_iters=10)

    cmd = robot.get_joint_pos().copy()
    cmd[: iface._n_arm] = ik_q[: iface._n_arm]
    assert cmd[robot._gripper_index] == pytest.approx(gripper_val, abs=1e-6)
    robot.command_joint_pos(cmd)


@pytest.mark.parametrize("gripper,site", GRIPPER_COMBOS, ids=_combo_id)
def test_ik_cycle_accuracy(gripper: GripperType, site: str) -> None:
    """IK at the current EE pose should converge and return near-original joints."""
    robot, iface = _make_iface(gripper, site)
    q0 = np.zeros(robot.num_dofs())
    q0[: min(3, len(q0))] = [0.2, 0.8, 0.5]
    robot.command_joint_pos(q0)
    iface._mirror_robot()
    iface._sync_mocap_to_ee()

    target = iface._mocap_pose_4x4()
    init_q = iface._data.qpos[: iface._nq].copy()
    ok, ik_q = iface._kin.ik(target, site, init_q=init_q)

    assert ok, f"IK failed to converge for {gripper.value}"
    pose_check = iface._kin.fk(ik_q)
    np.testing.assert_allclose(pose_check[:3, 3], target[:3, 3], atol=1e-3)
