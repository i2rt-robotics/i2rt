"""URDF <-> MJCF alignment for every arm model documented in ``arm/<arm>/README.md``.

Each arm has a URDF (kinematic/dynamic source of truth) and an arm-only MJCF. This test recomputes
masses, COMs, inertia tensors, joint frames/axes/ranges, and home-pose global frames independently
from *both* files and asserts they agree, so the MJCF cannot silently drift from the URDF. The
per-arm README tabulates the same quantities (plus Modified-DH and PoE screw axes), but this test
does **not** parse the README -- those tables are hand-maintained and can drift independently.

Comparison is done in frame-invariant / world terms so the two file formats can be checked directly:
  - home-pose body world transforms: URDF forward kinematics (matrix product) vs MJCF body-tree
    accumulation (quaternion product) -- independent numerical paths
  - joint world axes and joint ranges
  - per-link mass, COM, and inertia (URDF full tensor vs MJCF principal diag+quat, both re-expressed
    in the link frame)

Notes on scope:
  - ``link6`` is the gripper-mount frame, not a physical link; its MJCF inertial is a placeholder
    (``mass=1e-6``), so only its *frame*/axis/range are checked, not mass/COM/inertia (the READMEs
    likewise show ``-`` for the mount).
  - ``big_yam``'s MJCF models the fixed base as a static worldbody geom (no ``base`` body), so its
    base mass/COM/inertia have no MJCF counterpart to compare; the base frame is the world origin.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from i2rt.robot_models import (
    ARM_BIG_YAM_XML_PATH,
    ARM_YAM_PRO_XML_PATH,
    ARM_YAM_ULTRA_XML_PATH,
    ARM_YAM_XML_PATH,
)
from i2rt.robots.utils import ArmType

# Arm-only MJCF path per arm; the URDF sits beside it with the same basename.
ARM_XML = {
    ArmType.YAM: ARM_YAM_XML_PATH,
    ArmType.YAM_PRO: ARM_YAM_PRO_XML_PATH,
    ArmType.YAM_ULTRA: ARM_YAM_ULTRA_XML_PATH,
    ArmType.BIG_YAM: ARM_BIG_YAM_XML_PATH,
}
ARMS = list(ARM_XML)

# Tolerances. The yam family's MJCFs are generated from their URDFs so they agree to ~1e-15;
# big_yam's MJCF is hand-maintained and agrees to ~1e-6 on geometry / exactly on mass+range.
# Real misalignment (wrong link, swapped axis, stale mount) is orders of magnitude larger.
POS_TOL = 1e-5
ROT_TOL = 1e-4
AXIS_TOL = 1e-4
MASS_TOL = 1e-6
COM_TOL = 1e-6
INERTIA_TOL = 1e-5
RANGE_TOL = 1e-6

ARM_JOINT_RE = re.compile(r"^(?:dof_)?joint(\d+)$")
# MJCF body name -> URDF link name resolver key ("mount" == the joint6 child link).
MJCF_BODIES = ["base", "link1", "link2", "link3", "link4", "link5", "link6"]


# --------------------------------------------------------------------------- math
def _triplet(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return R.from_euler("ZYX", [rpy[2], rpy[1], rpy[0]]).as_matrix()


def _wxyz_to_matrix(quat: str) -> np.ndarray:
    w, x, y, z = _triplet(quat)
    return R.from_quat([x, y, z, w]).as_matrix()


def _transform(xyz: np.ndarray, rot: np.ndarray) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


def _inertia_in_link_frame(rot: np.ndarray, principal_or_tensor: np.ndarray) -> np.ndarray:
    """rot @ I @ rot.T -- re-express an inertia given in a rotated frame into the link frame."""
    return rot @ principal_or_tensor @ rot.T


# --------------------------------------------------------------------------- parsed model
@dataclass
class _Frame:
    name: str
    urdf_T: np.ndarray
    mjcf_T: np.ndarray


@dataclass
class _Inertial:
    name: str
    urdf_mass: float
    mjcf_mass: float
    urdf_com: np.ndarray
    mjcf_com: np.ndarray
    urdf_I: np.ndarray
    mjcf_I: np.ndarray


@dataclass
class _Joint:
    n: int
    urdf_axis_world: np.ndarray
    mjcf_axis_world: np.ndarray
    urdf_range: tuple[float, float]
    mjcf_range: tuple[float, float]


@dataclass
class _ArmModel:
    arm: str
    n_arm_joints: int
    mount_child: str
    has_base_body: bool
    frames: list[_Frame]
    inertials: list[_Inertial]
    joints: list[_Joint]


def _urdf_world_transforms(robot: ET.Element) -> dict[str, np.ndarray]:
    """Home-pose world transform per link via URDF joint-origin matrix products."""
    child_joint = {j.find("child").get("link"): j for j in robot.findall("joint")}
    cache: dict[str, np.ndarray] = {}

    def world(link: str) -> np.ndarray:
        if link in cache:
            return cache[link]
        j = child_joint.get(link)
        if j is None:
            cache[link] = np.eye(4)
            return cache[link]
        o = j.find("origin")
        xyz = _triplet(o.get("xyz", "0 0 0")) if o is not None else np.zeros(3)
        rpy = _triplet(o.get("rpy", "0 0 0")) if o is not None else np.zeros(3)
        cache[link] = world(j.find("parent").get("link")) @ _transform(xyz, _rpy_to_matrix(rpy))
        return cache[link]

    for lk in robot.findall("link"):
        world(lk.get("name"))
    return cache


def _mjcf_world_transforms(mjcf: ET.Element) -> tuple[dict[str, np.ndarray], dict[str, ET.Element]]:
    """Home-pose world transform + element per MJCF body via quaternion products."""
    world: dict[str, np.ndarray] = {}
    elem: dict[str, ET.Element] = {}

    def walk(body: ET.Element, parent_T: np.ndarray) -> None:
        xyz = _triplet(body.get("pos", "0 0 0"))
        w = parent_T @ _transform(xyz, _wxyz_to_matrix(body.get("quat", "1 0 0 0")))
        world[body.get("name")] = w
        elem[body.get("name")] = body
        for child in body.findall("body"):
            walk(child, w)

    for b in mjcf.find("worldbody").findall("body"):
        walk(b, np.eye(4))
    return world, elem


def _urdf_link_inertial(link_el: ET.Element) -> tuple[float, np.ndarray, np.ndarray]:
    ine = link_el.find("inertial")
    mass = float(ine.find("mass").get("value"))
    com = _triplet(ine.find("origin").get("xyz", "0 0 0"))
    rot = _rpy_to_matrix(_triplet(ine.find("origin").get("rpy", "0 0 0")))
    it = ine.find("inertia")
    tensor = np.array(
        [
            [float(it.get("ixx")), float(it.get("ixy")), float(it.get("ixz"))],
            [float(it.get("ixy")), float(it.get("iyy")), float(it.get("iyz"))],
            [float(it.get("ixz")), float(it.get("iyz")), float(it.get("izz"))],
        ]
    )
    return mass, com, _inertia_in_link_frame(rot, tensor)


def _mjcf_body_inertial(body_el: ET.Element) -> tuple[float, np.ndarray, np.ndarray]:
    bi = body_el.find("inertial")
    mass = float(bi.get("mass"))
    com = _triplet(bi.get("pos", "0 0 0"))
    rot = _wxyz_to_matrix(bi.get("quat", "1 0 0 0"))
    diag = np.diag(_triplet(bi.get("diaginertia")))
    return mass, com, _inertia_in_link_frame(rot, diag)


def _load_arm_model(arm: ArmType) -> _ArmModel:
    xml_path = ARM_XML[arm]
    urdf_path = os.path.splitext(xml_path)[0] + ".urdf"
    robot = ET.parse(urdf_path).getroot()
    mjcf = ET.parse(xml_path).getroot()

    arm_joints = sorted(
        (int(ARM_JOINT_RE.match(j.get("name")).group(1)), j)
        for j in robot.findall("joint")
        if ARM_JOINT_RE.match(j.get("name")) and j.get("type") == "revolute"
    )
    arm_joints = [j for _, j in arm_joints]
    base = arm_joints[0].find("parent").get("link")
    links = [base] + [arm_joints[i].find("child").get("link") for i in range(5)]
    mount = arm_joints[5].find("child").get("link")

    urdf_w = _urdf_world_transforms(robot)
    mjcf_w, mjcf_el = _mjcf_world_transforms(mjcf)
    # MJCF body name -> corresponding URDF link name.
    urdf_link_of = {"base": base, "link6": mount}
    for i in range(1, 6):
        urdf_link_of[f"link{i}"] = links[i]

    frames: list[_Frame] = []
    inertials: list[_Inertial] = []
    for name in MJCF_BODIES:
        if name not in mjcf_w:  # e.g. big_yam has no `base` body (fixed static geom)
            continue
        urdf_link = urdf_link_of[name]
        frames.append(_Frame(name, urdf_w[urdf_link], mjcf_w[name]))
        if name == "link6":  # gripper mount: placeholder inertial, excluded from README mass/inertia
            continue
        um, uc, ui = _urdf_link_inertial(_link_by_name(robot, urdf_link))
        mm, mc, mi = _mjcf_body_inertial(mjcf_el[name])
        inertials.append(_Inertial(name, um, mm, uc, mc, ui, mi))

    joints: list[_Joint] = []
    for n, j in enumerate(arm_joints, start=1):
        body_name, mj_joint = _mjcf_joint(mjcf, f"joint{n}")
        urdf_axis = _triplet(j.find("axis").get("xyz"))
        urdf_axis_w = urdf_w[j.find("child").get("link")][:3, :3] @ urdf_axis
        mjcf_axis_w = mjcf_w[body_name][:3, :3] @ _triplet(mj_joint.get("axis"))
        lim = j.find("limit")
        rng = _triplet(mj_joint.get("range"))
        joints.append(
            _Joint(
                n,
                urdf_axis_w,
                mjcf_axis_w,
                (float(lim.get("lower")), float(lim.get("upper"))),
                (float(rng[0]), float(rng[1])),
            )
        )

    return _ArmModel(arm.value, len(arm_joints), mount, "base" in mjcf_w, frames, inertials, joints)


def _link_by_name(robot: ET.Element, name: str) -> ET.Element:
    return next(lk for lk in robot.findall("link") if lk.get("name") == name)


def _mjcf_joint(mjcf: ET.Element, joint_name: str) -> tuple[str, ET.Element]:
    for body in mjcf.iter("body"):
        j = body.find("joint")
        if j is not None and j.get("name") == joint_name:
            return body.get("name"), j
    raise AssertionError(f"MJCF has no joint {joint_name!r}")


@pytest.fixture(params=ARMS, ids=lambda a: a.value)
def model(request: pytest.FixtureRequest) -> _ArmModel:
    return _load_arm_model(request.param)


# --------------------------------------------------------------------------- tests
def test_arm_structure(model: _ArmModel) -> None:
    """Six revolute arm joints and a terminal mount, as every README asserts (six-DOF)."""
    assert model.n_arm_joints == 6, f"{model.arm}: expected 6 revolute arm joints"
    assert model.mount_child, f"{model.arm}: no joint6 child (terminal mount) found"


def test_home_pose_frames_aligned(model: _ArmModel) -> None:
    """URDF FK and MJCF body accumulation agree on every body's home-pose world transform."""
    bad = []
    for f in model.frames:
        dp = float(np.max(np.abs(f.urdf_T[:3, 3] - f.mjcf_T[:3, 3])))
        dr = float(np.max(np.abs(f.urdf_T[:3, :3] - f.mjcf_T[:3, :3])))
        if dp > POS_TOL or dr > ROT_TOL:
            bad.append(f"{f.name}: dpos={dp:.2e} drot={dr:.2e}")
    assert not bad, f"{model.arm}: URDF/MJCF home frames diverge -> " + "; ".join(bad)


def test_joint_axes_aligned(model: _ArmModel) -> None:
    """Each joint's world rotation axis matches between URDF and MJCF (same direction, no sign flip)."""
    bad = []
    for j in model.joints:
        d = float(np.max(np.abs(j.urdf_axis_world - j.mjcf_axis_world)))
        if d > AXIS_TOL:
            bad.append(f"joint{j.n}: |urdf-mjcf world axis|={d:.2e}")
    assert not bad, f"{model.arm}: URDF/MJCF joint axes diverge -> " + "; ".join(bad)


def test_joint_ranges_aligned(model: _ArmModel) -> None:
    bad = []
    for j in model.joints:
        dl = abs(j.urdf_range[0] - j.mjcf_range[0])
        du = abs(j.urdf_range[1] - j.mjcf_range[1])
        if dl > RANGE_TOL or du > RANGE_TOL:
            bad.append(f"joint{j.n}: urdf={j.urdf_range} mjcf={j.mjcf_range}")
    assert not bad, f"{model.arm}: URDF/MJCF joint ranges diverge -> " + "; ".join(bad)


def test_link_masses_aligned(model: _ArmModel) -> None:
    bad = [
        f"{i.name}: urdf={i.urdf_mass} mjcf={i.mjcf_mass}"
        for i in model.inertials
        if abs(i.urdf_mass - i.mjcf_mass) > MASS_TOL
    ]
    assert not bad, f"{model.arm}: URDF/MJCF masses diverge -> " + "; ".join(bad)


def test_link_coms_aligned(model: _ArmModel) -> None:
    bad = [
        f"{i.name}: d={float(np.max(np.abs(i.urdf_com - i.mjcf_com))):.2e}"
        for i in model.inertials
        if np.max(np.abs(i.urdf_com - i.mjcf_com)) > COM_TOL
    ]
    assert not bad, f"{model.arm}: URDF/MJCF COMs diverge -> " + "; ".join(bad)


def test_link_inertias_aligned(model: _ArmModel) -> None:
    """URDF inertia tensor and MJCF principal (diag+quat) inertia match, both in the link frame."""
    bad = [
        f"{i.name}: d={float(np.max(np.abs(i.urdf_I - i.mjcf_I))):.2e}"
        for i in model.inertials
        if np.max(np.abs(i.urdf_I - i.mjcf_I)) > INERTIA_TOL
    ]
    assert not bad, f"{model.arm}: URDF/MJCF inertia tensors diverge -> " + "; ".join(bad)
