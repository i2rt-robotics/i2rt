"""Simulation-mode tests for every arm x gripper combination.

Run with:
    pytest tests/test_robot_variants.py -v
"""

from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest

from i2rt.motor_drivers.dm_driver import MotorInfo
from i2rt.robots.get_robot import get_yam_robot, wrap_around_offset
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config, combine_arm_and_gripper_xml

# Every shipped arm variant. Derived from the enum rather than listed, so a newly registered arm
# is covered here the day it lands instead of silently sitting outside this file.
YAM_ARMS = [arm for arm in ArmType if arm != ArmType.NO_ARM]


def n_arm_joints(arm_type: ArmType) -> int:
    """Motor-driven joints on this arm: six for the YAM family, seven for yam_7dof."""
    return len(_load_arm_config(arm_type).motor_list)


# Where the end-effector site's X axis points at qpos=0, which follows from how the arm is mounted.
# The YAM family stands up from the origin and hangs its tool downward; yam_7dof is plate-mounted --
# its mounting face is the world x-z plane, normal +Y, and the arm hangs along -Z -- so at home its
# tool points horizontally along +X instead.
HOME_EE_X_AXIS = dict.fromkeys((ArmType.YAM_7DOF_LEFT, ArmType.YAM_7DOF_RIGHT), np.array([1.0, 0.0, 0.0]))
DEFAULT_HOME_EE_X_AXIS = np.array([0.0, 0.0, -1.0])

# Grippers that work with the YAM family
YAM_GRIPPERS = [
    GripperType.LINEAR_4310,
    GripperType.LINEAR_3507,
    GripperType.CRANK_4310,
    GripperType.YAM_TEACHING_HANDLE_V1,
    GripperType.YAM_TEACHING_HANDLE_V2,
    GripperType.NO_GRIPPER,
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_robot(arm_type: ArmType, gripper_type: GripperType) -> Robot:
    """Create a SimRobot for the given arm/gripper pair."""
    return get_yam_robot(arm_type=arm_type, gripper_type=gripper_type, sim=True)


def test_get_yam_robot_preserves_positional_gripper_argument() -> None:
    """The legacy third positional argument remains the gripper type."""
    robot = get_yam_robot("can0", ArmType.YAM, GripperType.NO_GRIPPER, sim=True)

    assert robot.num_dofs() == 6
    robot.close()


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
    "gripper_type,gripper_dofs,has_gripper_obs",
    [
        (GripperType.LINEAR_4310, 1, True),
        (GripperType.LINEAR_3507, 1, True),
        (GripperType.CRANK_4310, 1, True),
        # The teaching handles publish gripper_pos from their passive trigger, but the
        # trigger has no motor, so it is not a commandable DOF.
        (GripperType.YAM_TEACHING_HANDLE_V1, 0, True),
        (GripperType.YAM_TEACHING_HANDLE_V2, 0, True),
        (GripperType.NO_GRIPPER, 0, False),
    ],
)
def test_dof_count(arm_type: ArmType, gripper_type: GripperType, gripper_dofs: int, has_gripper_obs: bool) -> None:
    """A robot reports one DOF per arm joint, plus one more when a gripper motor is on the chain."""
    robot = make_robot(arm_type, gripper_type)

    assert robot.num_dofs() == n_arm_joints(arm_type) + gripper_dofs

    obs = robot.get_observations()
    if has_gripper_obs:
        assert "gripper_pos" in obs
        assert obs["gripper_pos"].shape == (1,)
    else:
        assert "gripper_pos" not in obs

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
    xml_path = combine_arm_and_gripper_xml(arm_type, gripper_type)
    model = mujoco.MjModel.from_xml_path(xml_path)

    assert model.nq > 0
    assert model.njnt > 0


# ---------------------------------------------------------------------------
# EE site frame convention at zero pose
# ---------------------------------------------------------------------------


def _get_ee_site_axes(arm_type: ArmType, gripper_type: GripperType) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load combined model at qpos=0 and return grasp_site (x, y, z) world axes."""
    xml_path = combine_arm_and_gripper_xml(arm_type, gripper_type)
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
    """At qpos=0 the EE site frame is world-axis-aligned and points where the mount implies.

    For the YAM family that is X down (world -Z), Y left and Z front, both horizontal. yam_7dof is
    plate-mounted and hangs along -Z, so its tool points horizontally along +X instead. What holds
    for every arm is that all three axes land on world axes -- a stale mount block or a mis-signed
    wrist axis takes the frame off them.
    """
    x_axis, y_axis, z_axis = _get_ee_site_axes(arm_type, gripper_type)
    label = f"{arm_type.value}+{gripper_type.value}"

    expected_x = HOME_EE_X_AXIS.get(arm_type, DEFAULT_HOME_EE_X_AXIS)
    np.testing.assert_allclose(
        x_axis,
        expected_x,
        atol=1e-3,
        err_msg=f"{label}: X axis should be {expected_x}, got {x_axis}",
    )
    for name, axis in (("Y", y_axis), ("Z", z_axis)):
        dominant = int(np.argmax(np.abs(axis)))
        nearest = np.zeros(3)
        nearest[dominant] = np.sign(axis[dominant])
        np.testing.assert_allclose(
            axis,
            nearest,
            atol=1e-3,
            err_msg=f"{label}: {name} axis is not world-axis-aligned, got {axis}",
        )


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


# --------------------------------------------------------------------- wrap-around offsets
# The chain reports ``(raw - offset) * direction``, so an offset step that pulls a ``+1`` motor
# toward zero pushes a ``-1`` motor away from it. Bench case: yam_7dof_right's gripper reads raw
# -5.5526 rad with ``gripper_motor_direction: -1``, which reports as +5.5526 and so trips the
# correction. Sent the wrong way it lands at +11.84, putting most of the gripper's 5.68 rad stroke
# past the MIT position range -- where ``float_to_uint`` clamps the command and the motor stalls at
# full torque as soon as the first non-zero kp reaches it.
MIT_POSITION_MAX = 12.5  # MotorConstants.POSITION_MAX, and the motors' own PMAX register


@pytest.mark.parametrize("direction", [1, -1])
@pytest.mark.parametrize("reported", [5.5526, -5.5526, 3.2, -3.2, 8.0])
def test_wrap_around_offset_pulls_either_polarity_inside_pi(reported: float, direction: int) -> None:
    """The correction lands inside (-pi, pi] whichever way the motor is wired."""
    raw = reported / direction  # reported = (raw - 0) * direction at offset 0
    corrected = (raw - wrap_around_offset(reported, direction)) * direction
    assert abs(corrected) <= np.pi, f"{reported} rad on a {direction:+d} motor corrected to {corrected}"


def test_wrap_around_offset_keeps_a_reversed_gripper_inside_the_mit_range() -> None:
    """The bench case: the 7-DOF gripper's whole stroke must stay commandable."""
    reported, direction, stroke = 5.5526, -1, 5.678
    raw = reported / direction
    closed = (raw - wrap_around_offset(reported, direction)) * direction
    assert max(abs(closed), abs(closed + stroke)) < MIT_POSITION_MAX, (
        f"stroke [{closed:.4f}, {closed + stroke:.4f}] leaves the +-{MIT_POSITION_MAX} rad range"
    )


def test_wrap_around_offset_leaves_an_in_range_motor_alone() -> None:
    """No step for a motor already inside the band, whatever its polarity."""
    for direction in (1, -1):
        for pos in (0.0, 0.2054, -0.2054, 3.0, -3.0):
            assert wrap_around_offset(pos, direction) == 0.0


# --------------------------------------------------------------- transmissions are 7-DOF only
# joint6 on the 7-DOF arms is the one joint on any of these arms driven through a linkage. Every
# other arm is direct drive and must stay bit-identical to before that code existed.
def test_joint_to_motor_pos_is_the_identity_without_a_transmission() -> None:
    """A direct-drive arm's commands must come through the conversion unchanged.

    Not merely close: the same values, so nothing about a YAM's control path shifts.
    """
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    pos = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    out = robot._joint_to_motor_pos(pos)
    assert out is pos, "a transmission-free arm should not even copy the command"
    assert np.array_equal(out, pos)


def test_motor_state_conversion_is_untouched_without_a_transmission() -> None:
    """The read path leaves a direct-drive arm's feedback exactly as the motors reported it."""
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._gripper_index = None
    robot.remapper = SimpleNamespace(
        to_command_joint_pos_space=lambda v: v,
        to_command_joint_vel_space=lambda v: v,
    )
    motor_state = [MotorInfo(id=i + 1, error_code=1, pos=0.1 * i, vel=0.2 * i, eff=0.3 * i) for i in range(6)]
    joint_state = robot._motor_state_to_joint_state(motor_state)
    assert np.array_equal(joint_state.pos, np.array([0.1 * i for i in range(6)]))
    assert np.array_equal(joint_state.vel, np.array([0.2 * i for i in range(6)]))
    assert np.array_equal(joint_state.eff, np.array([0.3 * i for i in range(6)]))


# ---------------------------------------------------------------------------
# The two teaching-handle revisions
# ---------------------------------------------------------------------------


def test_pre_versioning_handle_names_resolve_to_v1() -> None:
    """Downstream configs written before v1/v2 existed must keep getting the v1 handle."""
    from i2rt.robot_models import GRIPPER_TEACHING_HANDLE_PATH, GRIPPER_TEACHING_HANDLE_V1_PATH

    assert GripperType.from_string_name("yam_teaching_handle") is GripperType.YAM_TEACHING_HANDLE_V1
    assert GripperType.YAM_TEACHING_HANDLE is GripperType.YAM_TEACHING_HANDLE_V1
    assert GRIPPER_TEACHING_HANDLE_PATH == GRIPPER_TEACHING_HANDLE_V1_PATH
    assert "yam_teaching_handle" not in GripperType.available_grippers()


def test_teaching_handle_revisions_are_distinct() -> None:
    """The v2 handle must be the v2 model, not a second copy of v1.

    Both handles share every convention -- the passive hinge named ``gripper``, its range, and
    the ``tcp_site`` orientation -- so a model file accidentally pointing at the wrong shell
    would pass every other test in this file. These two numbers separate them: the v2 shell is
    14.0 g lighter, and its trigger pivot sits 0.51 mm further out.
    """
    handles = {}
    for gripper_type in (GripperType.YAM_TEACHING_HANDLE_V1, GripperType.YAM_TEACHING_HANDLE_V2):
        model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, gripper_type))
        data = mujoco.MjData(model)
        data.qpos[:] = 0
        mujoco.mj_forward(model, data)

        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper")
        assert joint_id != -1, f"{gripper_type.value}: no passive joint named 'gripper'"
        np.testing.assert_allclose(model.jnt_range[joint_id], [-0.680678, 0.0], atol=1e-12)

        flange = model.body("gripper").id
        trigger = model.body("trigger").id
        handles[gripper_type] = (
            model.body_mass[flange] + model.body_mass[trigger],
            float(np.linalg.norm(data.xpos[trigger] - data.xpos[flange])),
        )

    v1_mass, v1_reach = handles[GripperType.YAM_TEACHING_HANDLE_V1]
    v2_mass, v2_reach = handles[GripperType.YAM_TEACHING_HANDLE_V2]

    assert v1_mass == pytest.approx(0.2575941, abs=1e-7)
    assert v2_mass == pytest.approx(0.24359111, abs=1e-7)
    assert v1_reach == pytest.approx(0.1014822104, abs=1e-9)
    assert v2_reach == pytest.approx(0.1019915745, abs=1e-9)


@pytest.mark.parametrize("gripper_type", [GripperType.YAM_TEACHING_HANDLE_V1, GripperType.YAM_TEACHING_HANDLE_V2])
def test_teaching_handle_tcp_is_the_shells_furthest_point_along_the_approach_axis(gripper_type: GripperType) -> None:
    """tcp_site sits where a gripper's grasp_site does: at the tool's far end along flange +Z.

    For a handle that is the shell's largest-z vertex in the flange frame, on the front of the butt. A
    re-cut or re-exported shell mesh that moves the butt must move tcp_site with it.
    """
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, gripper_type))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    flange = model.body("gripper").id
    to_flange = data.xmat[flange].reshape(3, 3)
    (shell,) = [g for g in range(model.ngeom) if model.geom_bodyid[g] == flange and model.geom_dataid[g] >= 0]
    mesh = model.geom_dataid[shell]
    verts = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
    in_flange = (verts @ data.geom_xmat[shell].reshape(3, 3).T + data.geom_xpos[shell] - data.xpos[flange]) @ to_flange

    tcp = (data.site_xpos[model.site("tcp_site").id] - data.xpos[flange]) @ to_flange
    np.testing.assert_allclose(tcp, in_flange[in_flange[:, 2].argmax()], atol=1e-6)
