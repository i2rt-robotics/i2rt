"""Posed URDF <-> MJCF alignment: command each arm through a set of joint configurations in
simulation and confirm link/joint frames stay consistent.

The sibling ``test_urdf_mjcf_alignment.py`` checks only the *home* pose, comparing the two files
by static parsing. This module complements it by exercising *non-home* configurations. Each arm
is loaded through the same simulation model that ships -- arm + ``linear_4310`` gripper, built by
``combine_arm_and_gripper_xml`` -- commanded to every configuration in ``YAM_ARM_MOTIONS`` (the
MuJoCo forward kinematics MuJoCo/``SimRobot`` would compute), and compared against an independent
analytical URDF forward-kinematics pass at the same joint angles.

Two properties are asserted:
  1. within-arm  -- for each arm, every arm-link world frame and every joint world axis agree
     between the MJCF (MuJoCo FK) and the URDF (matrix-product FK) at each pose.
  2. cross-arm   -- at each pose, the joint *axis* directions in world are identical across the
     yam-family arms (yam / yam_pro / yam_ultra). Full link-frame orientation is deliberately not
     compared cross-arm: the arms use different (but kinematically equivalent) frame conventions,
     so only the joint axis -- the frame-invariant "orientation" of a revolute joint -- matches.

big_yam scope notes:
  - big_yam mounts every gripper with a wrist-frame convention that differs from its URDF: all
    gripper configs give it ``quat "0 0.707107 0 -0.707107"`` versus the native ``"-0.5 0.5 0.5
    -0.5"``. ``combine_arm_and_gripper_xml`` therefore rewrites big_yam's link6 (the mount)
    orientation and joint6 axis, so those two frames are intentionally not compared to the URDF;
    its five physical arm links/joints (and the link6 mount *origin*) still are.
  - big_yam's URDF/MJCF use axis signs opposite to the yam family, so the same command vector
    reaches a different physical pose; it is excluded from the cross-arm axis check.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from functools import lru_cache

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

# Joint configurations to sweep. Each row is [joint1..joint6, gripper]; the 7th (gripper) value
# is part of the sim command but does not move any arm link/joint frame, so it is not applied.
YAM_ARM_MOTIONS = [
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    np.array([2.93, 0.05, 0.60, -0.08, 0.15, 2.07, 0.4]),
    np.array([3.06, 1.68, 3.14, 1.58, 1.57, -2.07, 0]),
    np.array([0.06, 0.68, 1.14, 0.58, 0.57, 0.07, 0.75]),
    np.array([-2.60, 0.0, 0.0, 1.58, -1.57, 1.40, 1.0]),
    np.array([-2.61, 0.0, 3.14, -1.59, -1.57, -2.06, 0.6]),
    np.array([0.12, 0.0, 0.74, 1.58, -1.57, 2.07, 0]),
]

ARMS = [ArmType.YAM, ArmType.YAM_PRO, ArmType.YAM_ULTRA, ArmType.BIG_YAM]
# Arms whose gripper mount preserves the native (URDF) wrist frame, so link6/joint6 stay
# URDF-comparable in the combined model. big_yam mounts grippers rotated (see module docstring).
MOUNT_ALIGNED_ARMS = {ArmType.YAM, ArmType.YAM_PRO, ArmType.YAM_ULTRA}
# Arms that share a joint-axis convention, so the same command reaches the same physical pose.
CROSS_ARM_FAMILY = [ArmType.YAM, ArmType.YAM_PRO, ArmType.YAM_ULTRA]

N_ARM_JOINTS = 6
# Every gripper is mounted with linear_4310, matching the default sim/hardware end-effector.
SIM_GRIPPER = GripperType.LINEAR_4310

# Same tolerances as the home-pose test: the yam family agrees to ~1e-15, big_yam to ~1e-6.
POS_TOL = 1e-5
ROT_TOL = 1e-4
AXIS_TOL = 1e-4

ARM_JOINT_RE = re.compile(r"^(?:dof_)?joint(\d+)$")


# --------------------------------------------------------------------------- math
def _triplet(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return R.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()


def _wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """MuJoCo stores quaternions as (w, x, y, z); scipy wants (x, y, z, w)."""
    return R.from_quat(np.roll(quat, -1)).as_matrix()


def _transform(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about the (unit-normalized) ``axis`` -- a revolute joint's motion."""
    axis = axis / np.linalg.norm(axis)
    return R.from_rotvec(axis * angle).as_matrix()


# ------------------------------------------------------------------- URDF forward kinematics
def _urdf_posed_frames(
    urdf_path: str, q_arm: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Analytical URDF FK at joint configuration ``q_arm`` (6 arm joints).

    Walks the revolute chain joint1..joint6, composing each joint's fixed origin transform with
    its rotation by the commanded angle. Returns, keyed by joint index n=1..6:
      - the world 4x4 transform of joint n's child link (link n)
      - the joint's rotation axis in world coordinates
    """
    robot = ET.parse(urdf_path).getroot()
    arm_joints = [
        j
        for _, j in sorted(
            (int(ARM_JOINT_RE.match(j.get("name")).group(1)), j)
            for j in robot.findall("joint")
            if ARM_JOINT_RE.match(j.get("name")) and j.get("type") == "revolute"
        )
    ]
    base = arm_joints[0].find("parent").get("link")

    world: dict[str, np.ndarray] = {base: np.eye(4)}
    link_world: dict[int, np.ndarray] = {}
    axis_world: dict[int, np.ndarray] = {}
    for n, j in enumerate(arm_joints, start=1):
        parent = j.find("parent").get("link")
        child = j.find("child").get("link")
        origin = j.find("origin")
        xyz = _triplet(origin.get("xyz", "0 0 0")) if origin is not None else np.zeros(3)
        rpy = _triplet(origin.get("rpy", "0 0 0")) if origin is not None else np.zeros(3)
        axis = _triplet(j.find("axis").get("xyz"))
        origin_rot = _rpy_to_matrix(rpy)
        world[child] = (
            world[parent] @ _transform(xyz, origin_rot) @ _transform(np.zeros(3), _axis_rotation(axis, q_arm[n - 1]))
        )
        link_world[n] = world[child]
        # The axis is invariant to rotation about itself, so it is fixed once the joint origin
        # frame is placed in the world (independent of q for this joint).
        axis_world[n] = world[parent][:3, :3] @ origin_rot @ axis
    return link_world, axis_world


# ------------------------------------------------------------------- MJCF forward kinematics
@lru_cache(maxsize=None)
def _combined_model(arm: ArmType) -> mujoco.MjModel:
    """The shipped sim model: arm + linear_4310 gripper, exactly as ``SimRobot`` loads it."""
    return mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(arm, SIM_GRIPPER))


def _mjcf_posed_frames(
    arm: ArmType, pose: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """MuJoCo FK at ``pose`` for the combined sim model.

    Mirrors ``SimRobot.command_joint_pos`` (set qpos, then ``mj_forward``) but sets the arm joints
    by name without the joint-limit clip, so the exact commanded configuration is evaluated. The
    gripper DOF is left at rest -- it does not affect any arm link or joint frame. Returns, keyed
    by joint index n=1..6, the world 4x4 of body ``link{n}`` and joint ``joint{n}``'s world axis.
    """
    model = _combined_model(arm)
    data = mujoco.MjData(model)
    for n in range(1, N_ARM_JOINTS + 1):
        data.qpos[model.joint(f"joint{n}").qposadr[0]] = pose[n - 1]
    mujoco.mj_forward(model, data)

    link_world: dict[int, np.ndarray] = {}
    axis_world: dict[int, np.ndarray] = {}
    for n in range(1, N_ARM_JOINTS + 1):
        bid = model.body(f"link{n}").id
        link_world[n] = _transform(data.xpos[bid].copy(), _wxyz_to_matrix(data.xquat[bid]))
        axis_world[n] = data.xaxis[model.joint(f"joint{n}").id].copy()
    return link_world, axis_world


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("arm", ARMS, ids=lambda a: a.value)
@pytest.mark.parametrize("pose", YAM_ARM_MOTIONS, ids=[f"pose{i}" for i in range(len(YAM_ARM_MOTIONS))])
def test_posed_frames_match_urdf(arm: ArmType, pose: np.ndarray) -> None:
    """Sim-commanding an arm to each pose keeps its link/joint frames aligned with the URDF."""
    urdf_path = os.path.splitext(arm.get_xml_path())[0] + ".urdf"
    urdf_frames, urdf_axes = _urdf_posed_frames(urdf_path, pose[:N_ARM_JOINTS])
    mjcf_frames, mjcf_axes = _mjcf_posed_frames(arm, pose)

    # Joints/links carrying the arm's own kinematics (all six) vs only the five physical arm
    # joints when the gripper mount rewrites the wrist frame (big_yam).
    checked = range(1, 7) if arm in MOUNT_ALIGNED_ARMS else range(1, 6)
    bad = []
    for n in checked:
        dp = float(np.max(np.abs(urdf_frames[n][:3, 3] - mjcf_frames[n][:3, 3])))
        dr = float(np.max(np.abs(urdf_frames[n][:3, :3] - mjcf_frames[n][:3, :3])))
        da = float(np.max(np.abs(urdf_axes[n] - mjcf_axes[n])))
        if dp > POS_TOL or dr > ROT_TOL or da > AXIS_TOL:
            bad.append(f"joint{n}/link{n}: dpos={dp:.2e} drot={dr:.2e} daxis={da:.2e}")

    # The mount *origin* is convention-independent, so it is checked even when the wrist
    # orientation is rewritten by the gripper mount (big_yam).
    if arm not in MOUNT_ALIGNED_ARMS:
        dp6 = float(np.max(np.abs(urdf_frames[6][:3, 3] - mjcf_frames[6][:3, 3])))
        if dp6 > POS_TOL:
            bad.append(f"link6 mount origin: dpos={dp6:.2e}")

    assert not bad, f"{arm.value} {pose[:N_ARM_JOINTS]}: URDF/MJCF posed frames diverge -> " + "; ".join(bad)


@pytest.mark.parametrize("pose", YAM_ARM_MOTIONS, ids=[f"pose{i}" for i in range(len(YAM_ARM_MOTIONS))])
def test_posed_joint_axes_match_across_arms(pose: np.ndarray) -> None:
    """At each pose, every joint's world axis is identical across the yam-family arms."""
    ref = CROSS_ARM_FAMILY[0]
    _, ref_axes = _mjcf_posed_frames(ref, pose)
    bad = []
    for arm in CROSS_ARM_FAMILY[1:]:
        _, axes = _mjcf_posed_frames(arm, pose)
        for n in range(1, N_ARM_JOINTS + 1):
            d = float(np.max(np.abs(axes[n] - ref_axes[n])))
            if d > AXIS_TOL:
                bad.append(f"{arm.value} joint{n}: |axis-{ref.value}|={d:.2e}")
    assert not bad, f"{pose[:N_ARM_JOINTS]}: cross-arm joint axes diverge -> " + "; ".join(bad)
