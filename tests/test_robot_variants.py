"""Simulation-mode tests for all YAM arm x gripper combinations.

Run with:
    pytest tests/test_robot_variants.py -v
"""

import mujoco
import numpy as np
import pytest

from i2rt.robots.get_robot import _ARM_JOINT_LIMITS, get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

# All YAM-family arm variants
YAM_ARMS = [
    ArmType.YAM,
    ArmType.YAM_PRO,
    ArmType.YAM_ULTRA,
    ArmType.BIG_YAM,
]

# Grippers that work with the YAM family
YAM_GRIPPERS = [
    GripperType.LINEAR_4310,
    GripperType.LINEAR_3507,
    GripperType.CRANK_4310,
    GripperType.YAM_TEACHING_HANDLE,
    GripperType.NO_GRIPPER,
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_robot(arm_type: ArmType, gripper_type: GripperType) -> MotorChainRobot:
    """Create a SimRobot for the given arm/gripper pair."""
    return get_yam_robot(arm_type=arm_type, gripper_type=gripper_type, sim=True)


# ---------------------------------------------------------------------------
# All arm x gripper combos: smoke-test loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm_type", YAM_ARMS)
@pytest.mark.parametrize("gripper_type", YAM_GRIPPERS)
def test_sim_robot_loads(arm_type: ArmType, gripper_type: GripperType) -> None:
    """Every arm/gripper combination should load without error in sim mode."""
    robot = make_robot(arm_type, gripper_type)

    pos = robot.get_joint_pos()
    assert isinstance(pos, np.ndarray)
    assert pos.ndim == 1
    assert len(pos) > 0

    obs = robot.get_observations()
    assert "joint_pos" in obs
    assert "joint_vel" in obs

    # Commanding zeros should not raise
    robot.command_joint_pos(np.zeros_like(pos))

    robot.close()


# ---------------------------------------------------------------------------
# DOF counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm_type", YAM_ARMS)
@pytest.mark.parametrize(
    "gripper_type,expected_dofs,has_gripper_obs",
    [
        (GripperType.LINEAR_4310, 7, True),
        (GripperType.LINEAR_3507, 7, True),
        (GripperType.CRANK_4310, 7, True),
        (GripperType.YAM_TEACHING_HANDLE, 6, False),
        (GripperType.NO_GRIPPER, 6, False),
    ],
)
def test_dof_count(arm_type: ArmType, gripper_type: GripperType, expected_dofs: int, has_gripper_obs: bool) -> None:
    """Robots with an active gripper motor should report 7 DOFs, arm-only 6."""
    robot = make_robot(arm_type, gripper_type)

    assert robot.num_dofs() == expected_dofs

    obs = robot.get_observations()
    if has_gripper_obs:
        assert "gripper_pos" in obs
        assert obs["gripper_pos"].shape == (1,)
    else:
        assert "gripper_pos" not in obs

    robot.close()


# ---------------------------------------------------------------------------
# Per-arm joint limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm_type", YAM_ARMS)
def test_joint_limits_match_arm_type(arm_type: ArmType) -> None:
    """Each arm variant should embed its own joint limits (XML range ± 0.15 buffer)."""
    robot = make_robot(arm_type, GripperType.NO_GRIPPER)
    info = robot.get_robot_info()
    limits = info["joint_limits"]

    expected = _ARM_JOINT_LIMITS[arm_type]
    np.testing.assert_allclose(limits[:, 0], expected[:, 0] - 0.15, atol=1e-6)
    np.testing.assert_allclose(limits[:, 1], expected[:, 1] + 0.15, atol=1e-6)

    robot.close()


# ---------------------------------------------------------------------------
# big_yam specifics
# ---------------------------------------------------------------------------


def test_big_yam_no_gripper_is_arm_only() -> None:
    """big_yam + no_gripper should give a 6-DOF arm-only robot."""
    robot = make_robot(ArmType.BIG_YAM, GripperType.NO_GRIPPER)
    info = robot.get_robot_info()

    assert robot.num_dofs() == 6
    assert info["gripper_index"] is None
    assert "gripper_pos" not in robot.get_observations()

    robot.close()


# ---------------------------------------------------------------------------
# MuJoCo XML combination sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm_type", YAM_ARMS)
@pytest.mark.parametrize("gripper_type", YAM_GRIPPERS)
def test_combined_xml_loads_in_mujoco(arm_type: ArmType, gripper_type: GripperType) -> None:
    """The combined arm+gripper XML must be loadable by MuJoCo."""
    xml_path = combine_arm_and_gripper_xml(arm_type.get_xml_path(), gripper_type.get_xml_path())
    model = mujoco.MjModel.from_xml_path(xml_path)

    assert model.nq > 0
    assert model.njnt > 0


# ---------------------------------------------------------------------------
# EE site frame convention at zero pose
# ---------------------------------------------------------------------------


def _get_ee_site_axes(arm_type: ArmType, gripper_type: GripperType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load combined model at qpos=0 and return grasp_site (x, y, z) world axes."""
    xml_path = combine_arm_and_gripper_xml(arm_type.get_xml_path(), gripper_type.get_xml_path())
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)

    # Prefer grasp_site; fall back to tcp_site (teaching handle has no grasp_site)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    if site_id < 0:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_site")
    assert site_id >= 0, f"No grasp_site or tcp_site in {arm_type.value}+{gripper_type.value}"

    xmat = data.site_xmat[site_id].reshape(3, 3)
    return xmat[:, 0], xmat[:, 1], xmat[:, 2]


@pytest.mark.parametrize("arm_type", YAM_ARMS)
@pytest.mark.parametrize("gripper_type", YAM_GRIPPERS)
def test_grasp_site_frame_at_zero_pose(arm_type: ArmType, gripper_type: GripperType) -> None:
    """At qpos=0 the EE site frame must be: X down, Y left, Z front.

    Concretely:
      - X axis ≈ world [0, 0, -1]  (pointing down)
      - Y axis is horizontal        (z-component ≈ 0)
      - Z axis is horizontal        (z-component ≈ 0, pointing front)
    """
    x_axis, y_axis, z_axis = _get_ee_site_axes(arm_type, gripper_type)
    label = f"{arm_type.value}+{gripper_type.value}"

    # X should point down (world -Z)
    np.testing.assert_allclose(
        x_axis,
        [0, 0, -1],
        atol=1e-3,
        err_msg=f"{label}: X axis should point down [0,0,-1], got {x_axis}",
    )
    # Z (front) should be horizontal
    assert abs(z_axis[2]) < 1e-3, f"{label}: Z axis should be horizontal (front), z-component={z_axis[2]:.6f}"
    # Y (left) should be horizontal
    assert abs(y_axis[2]) < 1e-3, f"{label}: Y axis should be horizontal (left), z-component={y_axis[2]:.6f}"


@pytest.mark.parametrize("arm_type", YAM_ARMS)
def test_grasp_site_consistent_across_grippers(arm_type: ArmType) -> None:
    """All grippers on the same arm must produce the same EE site orientation at zero pose."""
    reference = None
    ref_gripper = None
    for gripper_type in YAM_GRIPPERS:
        x, y, z = _get_ee_site_axes(arm_type, gripper_type)
        axes = np.column_stack([x, y, z])
        if reference is None:
            reference = axes
            ref_gripper = gripper_type
        else:
            np.testing.assert_allclose(
                axes,
                reference,
                atol=1e-3,
                err_msg=(f"{arm_type.value}: {gripper_type.value} site frame differs from {ref_gripper.value}"),
            )
