"""Gripper URDF <-> MJCF alignment.

The gripper MJCFs are the source of truth, and i2rt/robot_models/scripts/mjcf_to_urdf.py generates their URDFs.
These tests check that the URDFs are current, and check each one against MuJoCo's own compile of its MJCF by an
independent path: URDF rpy -> matrix forward kinematics against MuJoCo's kinematics. They compare every body and
site frame at both ends and the middle of the joints' travel, plus joint types, ranges and mimics, every link's
mass, COM and inertia, and every mesh's placement. The URDF's root link `gripper` is the MJCF's world frame.

`flexible_4310`'s URDF is an ONShape export in its own frames, so only its existence is checked.
"""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from i2rt.robot_models.scripts.mjcf_to_urdf import convert
from i2rt.robots.utils import GripperType

GENERATED = [g for g in GripperType if g != GripperType.FLEXIBLE_4310]
TOL = 1e-7  # the URDF carries 9 significant digits


def _paths(gripper: GripperType) -> tuple[str, str]:
    xml_path = gripper.get_xml_path()
    return xml_path, os.path.splitext(xml_path)[0] + ".urdf"


def _vec(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _origin(el: ET.Element) -> np.ndarray:
    origin = el.find("origin")
    pose = np.eye(4)
    pose[:3, :3] = _rpy_to_matrix(*_vec(origin.get("rpy")))
    pose[:3, 3] = _vec(origin.get("xyz"))
    return pose


def _joint_values(robot: ET.Element, fraction: float) -> dict[str, float]:
    """Every moving joint at ``fraction`` of its range, with each mimic joint following its leader."""
    joints = [j for j in robot.iter("joint") if j.get("type") != "fixed"]
    values = {}
    for joint in joints:
        limit = joint.find("limit")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        values[joint.get("name")] = lower + fraction * (upper - lower)
    for joint in joints:
        mimic = joint.find("mimic")
        if mimic is not None:
            leader = values[mimic.get("joint")]
            values[joint.get("name")] = float(mimic.get("multiplier")) * leader + float(mimic.get("offset"))
    return values


def _link_poses(robot: ET.Element, q: dict[str, float]) -> dict[str, np.ndarray]:
    """Every link's pose in the root link's frame at joint values ``q``."""
    by_child = {j.find("child").get("link"): j for j in robot.iter("joint")}

    def pose(link: str) -> np.ndarray:
        if link not in by_child:
            return np.eye(4)
        joint = by_child[link]
        transform = pose(joint.find("parent").get("link")) @ _origin(joint)
        motion = np.eye(4)
        if joint.get("type") == "prismatic":
            motion[:3, 3] = _vec(joint.find("axis").get("xyz")) * q[joint.get("name")]
        elif joint.get("type") == "revolute":
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, _vec(joint.find("axis").get("xyz")), q[joint.get("name")])
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, quat)
            motion[:3, :3] = rot.reshape(3, 3)
        return transform @ motion

    return {link.get("name"): pose(link.get("name")) for link in robot.iter("link")}


def _load(
    gripper: GripperType, fraction: float
) -> tuple[ET.Element, dict[str, np.ndarray], mujoco.MjModel, mujoco.MjData]:
    """The URDF and its link poses, and the compiled MJCF posed the same way, with every joint at ``fraction``."""
    xml_path, urdf_path = _paths(gripper)
    robot = ET.parse(urdf_path).getroot()
    q = _joint_values(robot, fraction)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    for name, value in q.items():
        data.joint(name).qpos[0] = value
    mujoco.mj_kinematics(model, data)
    return robot, _link_poses(robot, q), model, data


@pytest.mark.parametrize("gripper", list(GripperType), ids=lambda g: g.value)
def test_every_gripper_has_a_urdf(gripper: GripperType) -> None:
    assert os.path.isfile(_paths(gripper)[1])


@pytest.mark.parametrize("gripper", GENERATED, ids=lambda g: g.value)
def test_urdf_is_generated_from_the_mjcf(gripper: GripperType) -> None:
    xml_path, urdf_path = _paths(gripper)
    with open(urdf_path) as f:
        assert f.read() == convert(xml_path), f"{urdf_path} is stale: re-run i2rt/robot_models/scripts/mjcf_to_urdf.py"


@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("gripper", GENERATED, ids=lambda g: g.value)
def test_frames_match(gripper: GripperType, fraction: float) -> None:
    _, poses, model, data = _load(gripper, fraction)
    # Body 1 is the MJCF root, whose own pos/quat the URDF folds into its root link (the MJCF world frame).
    frames = [(model.body(i).name, data.xpos[i], data.xmat[i]) for i in range(2, model.nbody)]
    frames += [(model.site(i).name, data.site_xpos[i], data.site_xmat[i]) for i in range(model.nsite)]
    assert {name for name, _, _ in frames} | {model.body(1).name} == set(poses)
    for name, pos, mat in frames:
        np.testing.assert_allclose(poses[name][:3, 3], pos, atol=TOL, err_msg=name)
        np.testing.assert_allclose(poses[name][:3, :3], mat.reshape(3, 3), atol=TOL, err_msg=name)


@pytest.mark.parametrize("gripper", GENERATED, ids=lambda g: g.value)
def test_joints_match(gripper: GripperType) -> None:
    robot, _, model, _ = _load(gripper, 0.0)
    types = {"prismatic": mujoco.mjtJoint.mjJNT_SLIDE, "revolute": mujoco.mjtJoint.mjJNT_HINGE}
    joints = [j for j in robot.iter("joint") if j.get("type") != "fixed"]
    assert sorted(j.get("name") for j in joints) == sorted(model.joint(i).name for i in range(model.njnt))
    for joint in joints:
        mj_joint = model.joint(joint.get("name"))
        assert mj_joint.type[0] == types[joint.get("type")]
        limit = joint.find("limit")
        np.testing.assert_allclose(mj_joint.range, [float(limit.get("lower")), float(limit.get("upper"))], atol=1e-12)

    urdf_mimics = {
        j.get("name"): (
            j.find("mimic").get("joint"),
            float(j.find("mimic").get("multiplier")),
            float(j.find("mimic").get("offset")),
        )
        for j in joints
        if j.find("mimic") is not None
    }
    mjcf_mimics = {}
    for i in range(model.neq):
        assert model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
        c0, c1 = model.eq_data[i][:2]  # joint1 = c0 + c1 * joint2
        mjcf_mimics[model.joint(model.eq_obj2id[i]).name] = (model.joint(model.eq_obj1id[i]).name, 1 / c1, -c0 / c1)
    assert urdf_mimics == mjcf_mimics


@pytest.mark.parametrize("gripper", GENERATED, ids=lambda g: g.value)
def test_inertials_match(gripper: GripperType) -> None:
    robot, poses, model, data = _load(gripper, 0.0)
    links = {link.get("name"): link for link in robot.iter("link")}
    for i in range(1, model.nbody):
        name = model.body(i).name
        inertial = links[name].find("inertial")
        assert float(inertial.find("mass").get("value")) == pytest.approx(model.body_mass[i], rel=1e-9), name
        com = poses[name] @ _origin(inertial)
        np.testing.assert_allclose(com[:3, 3], data.xipos[i], atol=TOL, err_msg=name)
        a = inertial.find("inertia").attrib
        tensor = np.array(
            [
                [a["ixx"], a["ixy"], a["ixz"]],
                [a["ixy"], a["iyy"], a["iyz"]],
                [a["ixz"], a["iyz"], a["izz"]],
            ],
            dtype=float,
        )
        urdf_world = com[:3, :3] @ tensor @ com[:3, :3].T
        ximat = data.ximat[i].reshape(3, 3)
        mjcf_world = ximat @ np.diag(model.body_inertia[i]) @ ximat.T
        np.testing.assert_allclose(urdf_world, mjcf_world, rtol=1e-6, atol=1e-12, err_msg=name)


@pytest.mark.parametrize("gripper", GENERATED, ids=lambda g: g.value)
def test_meshes_match(gripper: GripperType) -> None:
    """Each URDF visual is an MJCF mesh geom with the same file and world pose; collisions are the colliding ones."""
    xml_path, urdf_path = _paths(gripper)
    robot, poses, _, data = _load(gripper, 0.5)

    def placed(link: str, el: ET.Element) -> tuple[str, np.ndarray]:
        filename = os.path.normpath(os.path.join(os.path.dirname(urdf_path), el.find("geometry/mesh").get("filename")))
        return filename, poses[link] @ _origin(el)

    urdf_visuals = [placed(link.get("name"), v) for link in robot.iter("link") for v in link.findall("visual")]
    urdf_collisions = [placed(link.get("name"), c) for link in robot.iter("link") for c in link.findall("collision")]

    mjcf = ET.parse(xml_path).getroot()
    compiler = mjcf.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_files = {
        m.get("name"): os.path.normpath(os.path.join(os.path.dirname(xml_path), meshdir, m.get("file")))
        for m in mjcf.iter("mesh")
    }
    mjcf_visuals, mjcf_collisions = [], []
    for body in mjcf.iter("body"):
        body_pose = np.eye(4)
        body_pose[:3, :3] = data.body(body.get("name")).xmat.reshape(3, 3)
        body_pose[:3, 3] = data.body(body.get("name")).xpos
        for geom in body.findall("geom"):
            local = np.eye(4)
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(
                rot, _vec(geom.get("quat", "1 0 0 0")) / np.linalg.norm(_vec(geom.get("quat", "1 0 0 0")))
            )
            local[:3, :3] = rot.reshape(3, 3)
            local[:3, 3] = _vec(geom.get("pos", "0 0 0"))
            entry = (mesh_files[geom.get("mesh")], body_pose @ local)
            mjcf_visuals.append(entry)
            if geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
                mjcf_collisions.append(entry)

    for urdf_side, mjcf_side in ((urdf_visuals, mjcf_visuals), (urdf_collisions, mjcf_collisions)):
        assert sorted(f for f, _ in urdf_side) == sorted(f for f, _ in mjcf_side)
        for (filename, urdf_pose), (_, mjcf_pose) in zip(
            sorted(urdf_side, key=lambda e: e[0]), sorted(mjcf_side, key=lambda e: e[0]), strict=True
        ):
            assert os.path.isfile(filename)
            np.testing.assert_allclose(urdf_pose, mjcf_pose, atol=TOL, err_msg=filename)
