"""What is specific to the 7-DOF arm: the mirror between its two sides, and its hardware config.

Since the arm moved under ``robot_models/arm/`` it is an ``ArmType`` like any other, so the
generic suites now cover most of it and this file no longer repeats them:
``test_urdf_mjcf_alignment.py`` recomputes frames, axes, ranges, masses, COMs and inertias from
both files (for **both** sides), ``test_urdf_mjcf_posed_alignment.py`` does the same at posed
configurations, ``test_assembly.py`` composes it against every gripper and
``test_arm_variant_registry`` keeps the enum and the on-disk models in sync.

What is left here is what no ``ArmType``-parametrized test can express:

  - **the mirror.** The left model is the right one reflected about the world x-z plane, generated
    by ``mirror_arm_model.py``. Every cross-side test below is run **posed**, not just at home: at
    ``q = 0`` the two arms are reflections of each other whatever their axes say, so every mirror
    sign error hides there.
  - **the specified joint directions.** ``HOME_WORLD_AXES`` pins the world directions the hardware
    was specified in, rather than restating whatever the model happens to say.
  - **seven joints and a gripper on 0x08.** The arm's own joints take CAN ``0x01``-``0x07``, so
    unlike every YAM its gripper cannot sit on ``0x07`` and the config declares its own id.
  - **the gripper mount interface.** The as-built export fused the wrist into the gripper, so the
    spigot that locates a gripper had to be grafted back from ``yam``; the check is that the two
    arms' interfaces measure the same, not that this one's mesh holds some number.
  - the gravity ceiling, the self-collision sweep and the sim wiring, all of which want the arm's own numbers rather than a family-wide tolerance.
"""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
import pytest

from i2rt.robots import get_robot
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config, combine_arm_and_gripper_xml
from i2rt.utils.mujoco_utils import MuJoCoKDL

# The two sides are separate ArmTypes, not one arm plus a side argument.
RIGHT, LEFT = ArmType.YAM_7DOF_RIGHT, ArmType.YAM_7DOF_LEFT
ARMS = (RIGHT, LEFT)
GRIPPERS = list(GripperType)
# The gripper the arm ships with, and the one the sim/hardware default builds.
DEFAULT_GRIPPER = GripperType.LINEAR_4310


# Chain order in the arm-only MJCF. joint1..joint7 drive link1..link6 then the terminal `gripper`
# mount; the fingers arrive from the gripper model at composition, renumbered to joint8/joint9.
BODIES = ["base"] + [f"link{i}" for i in range(1, 7)] + ["gripper"]
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
N_ARM_JOINTS = len(ARM_JOINTS)
# Real links, i.e. everything but the mass=1e-6 mount placeholder that composition overwrites.
ARM_LINKS = BODIES[:-1]

# Generated model, so agreement is at machine epsilon; these leave ~4 orders of margin over the
# observed worst case and are still many orders tighter than any real misalignment.
MASS_TOL = 1e-12
COM_TOL = 1e-12
INERTIA_TOL = 1e-14

MAX_GRAVITY_TORQUE = 20.0  # Nm, the same ceiling test_gravity_comp.py holds the YAM arms to

# Reflection about the world x-z plane. The two sides are true mirror images, so identical joint
# vectors put them in mirror-symmetric poses and no per-joint coordinate map is needed -- see
# test_left_is_the_mirror_of_right_when_posed, which feeds both models the same q.
MIRROR = np.diag([1.0, -1.0, 1.0])

# Each arm joint's world axis at the home pose, as specified for the hardware. Written out rather
# than derived, so this file pins the requirement instead of restating whatever the model happens to
# say. The left arm's are the reflection of these, which for a rotation axis means `-MIRROR @ a`.
HOME_WORLD_AXES = np.array(
    [
        [0.0, -1.0, 0.0],  # joint1  -y
        [-1.0, 0.0, 0.0],  # joint2  -x
        [0.0, 0.0, -1.0],  # joint3  -z
        [0.0, -1.0, 0.0],  # joint4  -y
        [0.0, 0.0, -1.0],  # joint5  -z
        [-1.0, 0.0, 0.0],  # joint6  -x
        [0.0, -1.0, 0.0],  # joint7  -y
    ]
)

# How close a nominally axis-aligned joint gets: the ONShape export writes joint rpy at %.6g, and
# that error compounds down the chain to ~5e-5 rad at the wrist. The model's own noise floor, not
# the mirror's -- the mirror is exact to machine epsilon.
AXIS_ALIGNMENT_TOL = 5e-4

# Poses inside the URDF joint limits, used for the posed sweeps.
# test_poses_are_within_the_joint_limits keeps this honest: the limits are hardware data and have
# changed once already, and a pose that drifts outside them would quietly weaken the gravity check.
POSES = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.3, -0.2, -0.6, 0.7, 0.2, -0.9],
        [-1.2, 1.4, 1.1, -1.8, -1.5, -0.6, 1.3],
        [1.5, 2.6, 1.3, 0.9, 1.2, 0.5, -1.4],
        [2.7, 0.9, 0.6, -1.1, -0.8, -0.3, 0.7],
        [-1.4, 3.2, 1.75, 0.4, 1.65, 0.75, 1.3],
    ]
)


# --------------------------------------------------------------------------- helpers
def _both_models() -> Tuple[mujoco.MjModel, mujoco.MjModel]:
    return (mujoco.MjModel.from_xml_path(RIGHT.get_xml_path()), mujoco.MjModel.from_xml_path(LEFT.get_xml_path()))


def _world_frames(model: mujoco.MjModel, q: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Each body's (world position, world rotation) at configuration ``q``."""
    data = mujoco.MjData(model)
    data.qpos[: len(q)] = q
    mujoco.mj_kinematics(model, data)
    return {name: (data.body(name).xpos.copy(), data.body(name).xmat.reshape(3, 3).copy()) for name in BODIES}


@pytest.fixture(scope="module", params=ARMS, ids=lambda a: a.value)
def arm(request: pytest.FixtureRequest) -> ArmType:
    """Both mirrored models, for every test in this file."""
    return request.param


@pytest.fixture(scope="module")
def mj_model(arm: ArmType) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(arm.get_xml_path())


# --------------------------------------------------------------------------- structure
def test_model_structure(mj_model: mujoco.MjModel) -> None:
    """Seven revolute arm joints and a terminal mount, one body per joint plus the base."""
    names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(mj_model.nbody)]
    assert names == ["world"] + BODIES
    joints = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mj_model.njnt)]
    assert joints == ARM_JOINTS
    assert all(mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE for i in range(mj_model.njnt))


def test_terminal_body_is_a_mount_placeholder(mj_model: mujoco.MjModel) -> None:
    """``gripper`` is the end-effector mount, not a link: composition overwrites its inertial.

    The arm-only MJCF gives it a token mass purely so MuJoCo compiles, exactly as every YAM arm
    does. A real inertial here would be silently discarded the moment a gripper is composed on --
    which is why the arm-side wrist section carries *geometry* on this body but no mass. Geoms are
    kept by composition; only the inertial is replaced.
    """
    assert mj_model.body("gripper").mass[0] == pytest.approx(1e-6)
    assert mj_model.body("gripper").geomnum[0] == 1, "the mount frame carries the wrist geometry"
    gid = mj_model.body("gripper").geomadr[0]
    mesh = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_MESH, mj_model.geom_dataid[gid])
    assert mesh == "wrist", f"expected the split wrist mesh on the mount, got {mesh!r}"


def _mount_interface(arm: ArmType) -> dict[str, float]:
    """Profile the arm-side/gripper-side mount interface, in the gripper's own frame.

    Measured along the gripper axis from the gripper root, which ``tcp_site`` marks, so the numbers
    are comparable between any two arms whatever frame convention each one's mount uses -- a YAM
    bolts its gripper on at identity, this arm at a 70 mm offset off the wrist-pitch axis. The axis
    is the site's own third column, signed to point from the gripper back towards the arm, so no
    assumption about either frame's handedness is needed.
    """
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(arm, DEFAULT_GRIPPER))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    site = data.site("tcp_site")
    axis, origin = site.xmat.reshape(3, 3)[:, 2], site.xpos

    clouds = {}
    for g in range(model.ngeom):
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        verts = model.mesh_vert[start : start + count].reshape(-1, 3)
        world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        offset = world - origin
        along = offset @ axis
        clouds[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)] = (
            along,
            np.linalg.norm(offset - np.outer(along, axis), axis=1),
        )

    # The arm's last mesh: the split wrist on a 7-DOF arm's mount, link5 on a YAM.
    arm_mesh = "wrist" if "wrist" in clouds else "link5"
    if clouds[arm_mesh][0].mean() < clouds["gripper"][0].mean():  # axis points into the gripper
        clouds = {k: (-a, r) for k, (a, r) in clouds.items()}
    assert {arm_mesh, "gripper"} <= set(clouds), f"{arm.value}: got {sorted(clouds)}"
    along, radial = clouds[arm_mesh]
    face = clouds["gripper"][0].max()
    spigot = along < face - 2e-3
    assert spigot.any(), (
        f"{arm.value}: no arm-side material past the gripper's mounting face -- the mount spigot is "
        "missing entirely, so the arm butts flat on the gripper's collar"
    )
    return {
        "face": face,
        # The seating face is the arm's first material wider than the gripper's bore.
        "shoulder": float(along[radial > 0.020].min() - face),
        # How far the spigot reaches into that bore.
        "engagement": float(face - along[radial < 0.0195].min()),
        "boss_diameter": float(2 * radial[spigot].max()),
    }


def test_wrist_carries_the_same_mount_spigot_as_a_yam(arm: ArmType) -> None:
    """The wrist mounts a gripper exactly the way a YAM's last link does.

    Every YAM ends in a tapered spigot that seats in the gripper's bore. This arm's as-built export
    fused wrist and gripper into one solid, erasing that interface, and the split that recovered the
    arm half cut at the mounting face -- handing the whole spigot to the gripper side and leaving the
    wrist butting flat on the gripper's collar. ``yam``'s own geometry is grafted back on, so the
    interface must now measure the same on both arms: same seating face, same engagement depth, same
    spigot diameter. Comparing against ``yam`` rather than against fixed numbers is what makes this
    the requirement rather than a restatement of whatever the mesh happens to hold.
    """
    got, want = _mount_interface(arm), _mount_interface(ArmType.YAM)
    for key, tol in (("shoulder", 5e-4), ("engagement", 5e-4), ("boss_diameter", 5e-4)):
        assert got[key] == pytest.approx(want[key], abs=tol), (
            f"{arm.value}: {key} is {got[key] * 1000:+.3f} mm, yam's is {want[key] * 1000:+.3f} mm"
        )


def test_arm_mass_excludes_the_end_effector(mj_model: mujoco.MjModel) -> None:
    """The shipped arm mass is the seven links only; the gripper's arrives with the gripper."""
    total = sum(mj_model.body(name).mass[0] for name in ARM_LINKS)
    assert total == pytest.approx(5.843760, abs=1e-6)


# --------------------------------------------------------------------------- the mirror
def test_left_is_the_mirror_of_right_when_posed() -> None:
    """Identical joint vectors put the two arms in mirror-symmetric poses.

    Posed, not just at home: at ``q = 0`` the two models are reflections of each other whatever
    their axes say, so an axis sign error is invisible there and only shows up once a joint moves.
    """
    right, left = _both_models()
    rng = np.random.default_rng(0)
    sweep = np.vstack([POSES, rng.uniform(right.jnt_range[:, 0], right.jnt_range[:, 1], size=(20, N_ARM_JOINTS))])
    bad = []
    for q in sweep:
        fr, fl = _world_frames(right, q), _world_frames(left, q)
        for name in BODIES:
            dp = float(np.max(np.abs(MIRROR @ fr[name][0] - fl[name][0])))
            dr = float(np.max(np.abs(MIRROR @ fr[name][1] @ MIRROR - fl[name][1])))
            if dp > 1e-12 or dr > 1e-12:
                bad.append(f"{name} at {np.round(q, 3)}: dpos={dp:.2e} drot={dr:.2e}")
    assert not bad, "left is not the reflection of right -> " + "; ".join(bad[:5])


def test_left_is_the_mirror_of_right_inertially() -> None:
    """Mass and principal moments survive the reflection; the COM reflects with the body."""
    right, left = _both_models()
    for name in ARM_LINKS:
        r, lft = right.body(name), left.body(name)
        assert r.mass[0] == pytest.approx(lft.mass[0], abs=MASS_TOL), f"{name}: mass"
        assert np.allclose(r.inertia, lft.inertia, atol=INERTIA_TOL), f"{name}: principal moments"
        assert np.allclose(MIRROR @ r.ipos, lft.ipos, atol=COM_TOL), f"{name}: COM"


def test_left_joint_axes_are_the_mirror_of_right() -> None:
    """A rotation axis is a pseudovector, so it reflects as ``-MIRROR @ a`` -- exactly, not nearly.

    Both models are written by the same sign-flipping script, so this is bit-exact or broken.
    """
    right, left = _both_models()
    for i in range(right.njnt):
        assert np.array_equal(-MIRROR @ right.jnt_axis[i], left.jnt_axis[i]), ARM_JOINTS[i]


def test_sides_share_joint_ranges() -> None:
    """The reflection leaves the joint coordinate alone, so no range needs re-signing."""
    right, left = _both_models()
    assert np.array_equal(right.jnt_range, left.jnt_range)
    assert np.array_equal(right.jnt_limited, left.jnt_limited)


def test_actuator_force_range_is_side_independent() -> None:
    """A joint's torque bound is a property of its motor, so the mirror must leave it alone.

    Easy to clobber and invisible if it is: nothing else in this file reads ``actuatorfrcrange``,
    and a wrong value silently re-scales what the model believes the joint can produce rather than
    failing.
    """
    right, left = _both_models()
    assert np.array_equal(right.jnt_actfrcrange, left.jnt_actfrcrange)
    assert np.array_equal(right.jnt_actfrclimited, left.jnt_actfrclimited)


# --------------------------------------------------------------------------- specified directions
def test_home_world_axes_are_the_specified_directions(arm: ArmType, mj_model: mujoco.MjModel) -> None:
    """Each joint turns about the world direction the hardware was specified in, at home.

    Stated in world terms rather than as local ``<axis>`` values, because the local axes that
    satisfy this are deliberately not uniform -- joint3 and joint5 sit on -Z, the rest on +Z -- and
    "tidying" them would move the arm.
    """
    data = mujoco.MjData(mj_model)
    mujoco.mj_kinematics(mj_model, data)
    expected = HOME_WORLD_AXES if arm == RIGHT else HOME_WORLD_AXES @ (-MIRROR).T
    for i, name in enumerate(ARM_JOINTS):
        jid = mj_model.joint(name).id
        world_axis = data.body(mj_model.jnt_bodyid[jid]).xmat.reshape(3, 3) @ mj_model.jnt_axis[jid]
        assert np.allclose(world_axis, expected[i], atol=AXIS_ALIGNMENT_TOL), (
            f"{arm.value} {name}: expected {expected[i]}, got {np.round(world_axis, 6)}"
        )


def test_poses_are_within_the_joint_limits(arm: ArmType, mj_model: mujoco.MjModel) -> None:
    """The sweep this file reuses must stay inside the arm's real travel.

    The limits are hardware data and have already changed once; a pose that drifted outside them
    would quietly weaken the gravity ceiling rather than fail.
    """
    lo, hi = mj_model.jnt_range[:, 0], mj_model.jnt_range[:, 1]
    for q in POSES:
        assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9), f"{arm.value}: {q} outside {lo}..{hi}"


# --------------------------------------------------------------------------- config <-> model
def test_sides_share_hardware_config() -> None:
    """Both sides are the same physical arm, so their configs must not drift apart.

    ``directions`` excepted: it maps motor polarity to each side's model, and on the left arm
    joint3, joint5 and joint6's motors do not reverse across the reflection the way their axes do.
    """
    right_hw = _load_arm_config(RIGHT)
    left_hw = _load_arm_config(LEFT)
    assert right_hw.motor_list == left_hw.motor_list
    assert right_hw.gripper_can_id == left_hw.gripper_can_id
    for field in ("kp", "kd", "gravity_comp_factor", "grav_comp_kd", "coulomb_friction"):
        assert np.array_equal(getattr(right_hw, field), getattr(left_hw, field)), field
    assert right_hw.motor_registers == left_hw.motor_registers
    assert right_hw.transmissions == left_hw.transmissions


def test_config_matches_model(arm: ArmType) -> None:
    """The config declares one motor per arm joint, plus the gripper on its own CAN id.

    ``0x08`` is the one thing about this chain that differs from every YAM's: seven joints take
    ``0x01``-``0x07``, so the id the YAM grippers use is already spoken for.
    """
    hw = _load_arm_config(arm)
    assert len(hw.motor_list) == N_ARM_JOINTS
    for field in (hw.directions, hw.kp, hw.kd, hw.gravity_comp_factor, hw.grav_comp_kd, hw.coulomb_friction):
        assert len(field) == N_ARM_JOINTS

    arm_ids = [can_id for can_id, _ in hw.motor_list]
    assert len(set(arm_ids)) == N_ARM_JOINTS, "duplicate CAN id among the arm motors"
    assert hw.gripper_can_id == 0x08
    assert hw.gripper_can_id not in arm_ids, "the gripper motor collides with an arm motor's CAN id"
    # The startup check holds every arm motor to its KT_Value/GREF; the gripper is not checked.
    assert set(hw.motor_registers) == set(arm_ids)
    assert all(set(regs) == {"KT_Value", "GREF"} for regs in hw.motor_registers.values())


# --------------------------------------------------------------------------- composition
@pytest.mark.parametrize("gripper", GRIPPERS, ids=lambda g: g.value)
def test_composes_with_every_gripper(gripper: GripperType, arm: ArmType) -> None:
    """Every gripper mounts on this arm, with its joints renumbered to follow the arm's seven.

    The renumbering is the part that only this arm exercises: a gripper model names its fingers
    ``joint7``/``joint8`` for a six-joint arm, which collides here and MuJoCo rejects outright.
    """
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(arm, gripper))
    joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert joints[:N_ARM_JOINTS] == ARM_JOINTS, f"{gripper.value}: arm joints were renamed"
    assert len(joints) == len(set(joints)), f"{gripper.value}: duplicate joint names"
    # Only positional `jointN` names are renumbered; the teaching handles' passive hinge is named
    # `gripper` so that it can be found by name, and must keep that name.
    fingers = [j for j in joints[N_ARM_JOINTS:] if j != "gripper"]
    expected_fingers = [f"joint{N_ARM_JOINTS + 1 + i}" for i in range(len(fingers))]
    assert fingers == expected_fingers, f"{gripper.value}: fingers not renumbered"


def test_composed_gripper_lands_where_the_mount_says(arm: ArmType) -> None:
    """The gripper's own root frame ends up at the mount offset the arm config declares.

    This arm's last joint is a wrist pitch, not the tool roll, so the gripper bolts ~70 mm off the
    joint axis -- an offset the mount pose cannot carry (moving it would move joint7's axis line)
    and which ``gripper_pos``/``gripper_quat`` supply instead. If that were dropped the gripper
    would collapse onto the wrist.
    """
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(arm, DEFAULT_GRIPPER))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mount = data.body("gripper")
    tcp = data.site("grasp_site").xpos
    offset_in_mount = mount.xmat.reshape(3, 3).T @ (tcp - mount.xpos)
    assert np.linalg.norm(offset_in_mount) > 0.15, (
        f"{arm.value}: the tool centre sits {np.linalg.norm(offset_in_mount) * 1000:.1f} mm from the "
        "mount -- the per-arm gripper offset looks to have been dropped"
    )


def test_gravity_torques_within_limits(arm: ArmType) -> None:
    """Gravity-compensation torques stay under the ceiling MotorChainRobot enforces.

    Run on the composed model, gripper included: that is the robot the control loop drives, and the
    end-effector is most of the wrist's load.
    """
    kdl = MuJoCoKDL(combine_arm_and_gripper_xml(arm, DEFAULT_GRIPPER))
    rng = np.random.default_rng(0)
    lo, hi = kdl.joint_limits[:N_ARM_JOINTS, 0], kdl.joint_limits[:N_ARM_JOINTS, 1]
    for q_arm in np.vstack([POSES, rng.uniform(lo, hi, size=(20, N_ARM_JOINTS))]):
        q = np.zeros(kdl.model.nq)
        q[:N_ARM_JOINTS] = q_arm
        tau = kdl.compute_inverse_dynamics(q, np.zeros_like(q), np.zeros_like(q))[:N_ARM_JOINTS]
        assert np.max(np.abs(tau)) < MAX_GRAVITY_TORQUE, f"{np.max(np.abs(tau)):.2f} Nm at {q_arm}"


@pytest.mark.parametrize("gripper", GRIPPERS, ids=lambda g: g.value)
def test_no_structural_self_collision(gripper: GripperType, arm: ArmType) -> None:
    """No non-parent-child body pair may penetrate anywhere in the joint space.

    MuJoCo collides meshes by their convex hulls and only auto-excludes direct parent-child pairs,
    so a permanently-overlapping pair would make any self-collision gate block every command. The
    arm ships a ``link5``/``gripper`` exclude because the wrist section on the mount nests around
    the short ``link6`` segment and their hulls overlap by up to 38 mm at every reachable pose. That is
    the arm's own geometry, so it holds for every gripper including ``no_gripper``. This test is
    what keeps the exclude sufficient -- and no wider than it needs to be.
    """
    model = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(arm, gripper))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    sweep = np.vstack(
        [
            np.zeros(model.nq),
            model.jnt_range[:, 0],
            model.jnt_range[:, 1],
            rng.uniform(model.jnt_range[:, 0], model.jnt_range[:, 1], size=(200, model.nq)),
        ]
    )
    offenders = set()
    for q in sweep:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            b1, b2 = model.geom_bodyid[data.contact[c].geom1], model.geom_bodyid[data.contact[c].geom2]
            if model.body_parentid[b1] == b2 or model.body_parentid[b2] == b1:
                continue
            name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)  # noqa: E731
            offenders.add(tuple(sorted((name(b1), name(b2)))))
    assert not offenders, f"{arm.value}+{gripper.value}: bodies penetrate -> {sorted(offenders)}"


# --------------------------------------------------------------------------- sim wiring
@pytest.fixture
def sim_robot(arm: ArmType) -> Robot:
    robot = get_yam_robot(arm_type=arm, gripper_type=DEFAULT_GRIPPER, sim=True)
    yield robot
    robot.close()


def test_sim_robot_dofs(sim_robot: Robot) -> None:
    """Seven arm joints plus the gripper, with the gripper last as MotorChainRobot requires."""
    assert sim_robot.num_dofs() == N_ARM_JOINTS + 1
    assert len(sim_robot.get_joint_pos()) == N_ARM_JOINTS + 1
    assert sim_robot.get_robot_info()["gripper_index"] == N_ARM_JOINTS


def test_sim_robot_round_trips_a_command(sim_robot: Robot) -> None:
    """A commanded arm pose is reached, and the gripper command stays in its normalized range."""
    target = np.array([0.3, 0.4, -0.3, -0.5, 0.6, 0.2, -0.4, 0.0])
    for _ in range(200):
        sim_robot.command_joint_pos(target)
    reached = sim_robot.get_joint_pos()
    assert np.allclose(reached[:N_ARM_JOINTS], target[:N_ARM_JOINTS], atol=0.05), reached
    assert 0.0 <= reached[N_ARM_JOINTS] <= 1.0


def test_sim_joint_limits_are_the_model_limits_plus_buffer(arm: ArmType, sim_robot: Robot) -> None:
    """The arm limits are widened by the same 0.15 rad buffer every YAM arm gets."""
    model_limits = mujoco.MjModel.from_xml_path(arm.get_xml_path()).jnt_range[:N_ARM_JOINTS]
    limits = np.asarray(sim_robot.get_robot_info()["joint_limits"])[:N_ARM_JOINTS]
    assert np.allclose(limits[:, 0], model_limits[:, 0] - 0.15)
    assert np.allclose(limits[:, 1], model_limits[:, 1] + 0.15)


# --------------------------------------------------------------------------- the joint6 four-bar
# joint6's motor turns an input crank, not the joint, and there is no encoder on the joint -- so the
# joint angle is computed from the motor's. These pin the map against the bench: driving joint6 to
# both hard stops put the motor at -77.91 deg and +66.22 deg over five identical runs, and the
# linkage carries those onto the URDF's own joint limits.
J6_MOTOR_AT_STOPS = (-77.91, 66.22)  # deg
J6_JOINT_AT_STOPS = (-55.0, 45.0)  # deg, the URDF's joint6 range
J6_STOP_TOL = 0.2  # deg
JOINT6 = 5  # joint index
# Both sides declare the same linkage (test_sides_share_hardware_config), so the right's stands in.
LINKAGE = _load_arm_config(RIGHT).transmissions[JOINT6]


def test_joint6_linkage_maps_the_measured_stops_onto_the_urdf_limits() -> None:
    """The bench measurement and the model agree once the linkage is applied.

    This is the evidence the URDF's joint6 range is right: the motor travels 144.13 deg where the
    joint travels 100, because the four-bar is a reduction averaging about 0.69.
    """
    linkage = LINKAGE
    for motor_deg, joint_deg in zip(J6_MOTOR_AT_STOPS, J6_JOINT_AT_STOPS, strict=True):
        got = np.degrees(linkage.joint_from_motor(np.radians(motor_deg)))
        assert abs(got - joint_deg) < J6_STOP_TOL, f"motor {motor_deg} -> {got}, expected {joint_deg}"


def test_joint6_zero_maps_to_zero() -> None:
    """The motor is zeroed with joint6 at zero, so the two origins coincide by construction."""
    assert abs(LINKAGE.joint_from_motor(0.0)) < 1e-12


def test_joint6_linkage_inverse_round_trips() -> None:
    """motor_from_joint undoes joint_from_motor across the whole travel.

    The inverse is what the command path relies on, and the map is nonlinear enough that a
    small-angle approximation would not do.
    """
    linkage = LINKAGE
    lo, hi = linkage.motor_range
    for motor in np.linspace(lo, hi, 200):
        assert abs(linkage.motor_from_joint(linkage.joint_from_motor(motor)) - motor) < 1e-9


def test_joint6_linkage_is_a_reduction_and_never_reverses() -> None:
    """The ratio stays positive, so +command gives +joint everywhere in the travel.

    A dead point inside the travel would make the joint reverse while the motor kept turning, and
    would leave the inverse ambiguous; a linkage that does not close gives NaN and fails too. This is
    the guard motor_from_joint's bracketed bisection relies on.
    """
    linkage = LINKAGE
    lo, hi = linkage.motor_range
    ratios = np.array([linkage.ratio(m) for m in np.linspace(lo, hi, 300)])
    assert np.all(ratios > 0), "joint6 reverses inside its travel"
    assert np.all(ratios < 1.0), "the four-bar should reduce, not multiply"


# --------------------------------------------------------------------------- real-hardware wiring
# get_yam_robot's real path against a stand-in for the CAN chain. Everything above the chain is the
# real code -- model composition, KDL, MotorChainRobot and its joint6 conversion -- which is where
# a linkage joint can be silently run as direct drive.
class _FakeChain:
    """A motionless chain: every motor reads zero except joint6's, which reads ``joint6_motor``."""

    def __init__(
        self,
        motor_list: List[Any],
        offsets: List[float],
        directions: List[int],
        *_: Any,
        joint6_motor: float,
        **__: Any,
    ) -> None:
        self.motor_direction = np.array(directions)
        self.motor_offset = np.array(offsets, dtype=float)
        self.running = True
        self._states = [
            SimpleNamespace(
                pos=joint6_motor if i == JOINT6 else 0.0, vel=0.0, eff=0.0, temp_mos=0.0, temp_rotor=0.0, timestamp=0.0
            )
            for i in range(len(motor_list))
        ]

    def __len__(self) -> int:
        return len(self._states)

    def read_states(self) -> List[SimpleNamespace]:
        return self._states

    def set_commands(self, *_: Any, **__: Any) -> List[SimpleNamespace]:
        return self._states

    def start_thread(self) -> None:
        pass

    def close(self) -> None:
        self.running = False


def _real_robot(
    monkeypatch: pytest.MonkeyPatch,
    arm: ArmType,
    gripper: GripperType = GripperType.NO_GRIPPER,
    joint6_motor_deg: float = 0.0,
    **kwargs: Any,
) -> MotorChainRobot:
    monkeypatch.setattr(
        get_robot, "DMChainCanInterface", partial(_FakeChain, joint6_motor=np.radians(joint6_motor_deg))
    )
    return get_yam_robot(arm_type=arm, gripper_type=gripper, **kwargs)


@pytest.mark.parametrize(
    "gripper",
    [GripperType.NO_GRIPPER, GripperType.YAM_TEACHING_HANDLE_V1, GripperType.LINEAR_4310],
    ids=lambda g: g.value,
)
def test_real_robot_keeps_the_joint6_linkage(
    arm: ArmType, gripper: GripperType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every build carries joint6's linkage, with a gripper or without.

    The gripperless builds once fell back to ``ArmType.YAM`` and ran joint6 as direct drive: its
    state was the raw motor angle, gravity comp ran at the wrong q, and its torque was 1.15-7x off.
    """
    kwargs = {"gripper_limits_override": np.array([0.0, 1.0])} if gripper == GripperType.LINEAR_4310 else {}
    robot = _real_robot(monkeypatch, arm, gripper, **kwargs)
    try:
        assert robot._transmissions == _load_arm_config(arm).transmissions
        assert JOINT6 in robot._transmissions
        assert robot.get_robot_info()["arm_type"] == arm
    finally:
        robot.close()


def test_startup_hold_targets_the_motor_angle(arm: ArmType, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without zero-gravity mode the arm holds its startup pose -- in motor angles, as sent.

    The joint state is joint space. Sent unconverted, joint6 at its lower stop would be pulled
    ~16 deg off where it stands until the caller's first command.
    """
    motor_deg = J6_MOTOR_AT_STOPS[0]
    robot = _real_robot(monkeypatch, arm, joint6_motor_deg=motor_deg, zero_gravity_mode=False)
    try:
        assert np.degrees(robot._commands.pos[JOINT6]) == pytest.approx(motor_deg, abs=1e-6)
    finally:
        robot.close()


@pytest.mark.parametrize("motor_deg", [-90.0, -85.0, 75.0, 77.91])
def test_out_of_range_joint6_reading_refuses(arm: ArmType, motor_deg: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """A joint6 motor reading past its travel refuses startup instead of converting.

    Each one gets past the joint-limit check otherwise. -90 converts to NaN, which fails every
    comparison. -85 gives -61.6 deg, inside the buffered limits. +75 is past the dead point, where the
    joint folds back to 45.3 deg and the ratio turns negative. And +77.91 is the lower stop read with
    a flipped ``directions`` sign.
    """
    with pytest.raises(RuntimeError, match="outside its linkage's travel"):
        _real_robot(monkeypatch, arm, joint6_motor_deg=motor_deg).close()  # if it builds, stop its thread


@pytest.mark.parametrize("motor_deg", [J6_MOTOR_AT_STOPS[0], 0.0, J6_MOTOR_AT_STOPS[1]])
def test_in_range_joint6_reading_builds(arm: ArmType, motor_deg: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both measured hard stops and zero are inside the travel."""
    _real_robot(monkeypatch, arm, joint6_motor_deg=motor_deg).close()
