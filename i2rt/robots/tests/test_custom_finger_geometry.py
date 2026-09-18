"""Offline geometry checks: no motor drivers or CAN connections."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310))


def mesh_points(m: mujoco.MjModel, d: mujoco.MjData, name: str) -> np.ndarray:
    g = m.geom(name).id
    mesh = m.geom_dataid[g]
    start = m.mesh_vertadr[mesh]
    vertices = m.mesh_vert[start : start + m.mesh_vertnum[mesh]]
    return vertices @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def test_custom_finger_extents_and_unchanged_tool_site() -> None:
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mount = m.body("gripper").id
    rotation = d.xmat[mount].reshape(3, 3)
    for side in ("left", "right"):
        points = (mesh_points(m, d, f"umi_{side}_pad_collision") - d.xpos[mount]) @ rotation
        # Mount registration, not stretching to the intentional 220 mm virtual TCP.
        assert -0.210 < points[:, 2].min() < -0.208
        for part in ("adapter", "finger", "pad"):
            visual = m.geom(f"umi_{side}_{part}").id
            collision = m.geom(f"umi_{side}_{part}_collision").id
            assert m.geom_contype[visual] == m.geom_conaffinity[visual] == 0
            assert m.geom_contype[collision] == m.geom_conaffinity[collision] == 1
            assert m.geom_bodyid[visual] == m.geom_bodyid[collision]
    np.testing.assert_array_equal(m.site("grasp_site").pos, [0, 0, -0.220])
    np.testing.assert_array_equal(m.site("grasp_site").quat, [0, 0, 1, 0])


@pytest.mark.parametrize("opening", [0.0, 0.5, 1.0])
def test_independent_slide_travel(opening: float) -> None:
    m = model()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    initial = [mesh_points(m, d, f"umi_{side}_pad_collision") for side in ("left", "right")]
    mount_rotation = d.xmat[m.body("gripper").id].reshape(3, 3).copy()
    d.qpos[6:] = opening * 0.0475
    mujoco.mj_forward(m, d)
    for side, sign, before in zip(("left", "right"), (-1, 1), initial, strict=True):
        displacement = (mesh_points(m, d, f"umi_{side}_pad_collision") - before) @ mount_rotation
        np.testing.assert_allclose(displacement, np.tile([0, sign * opening * 0.0475, 0], (len(before), 1)), atol=3e-7)


def test_distal_geometry_participates_in_collision_detection() -> None:
    root = ET.parse(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)).getroot()
    world = root.find("worldbody")
    assert world is not None
    # A 3 mm probe at flange Z=-200 mm: beyond the old 145 mm fingertips.
    ET.SubElement(world, "geom", name="probe", type="sphere", size=".003", pos=".310597 .003 .173502")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    probe = m.geom("probe").id
    assert any(probe in (c.geom1, c.geom2) and c.dist < -0.001 for c in d.contact)
    # Opening the independent jaws clears the same probe.
    d.qpos[6:] = 0.0475
    mujoco.mj_forward(m, d)
    assert not any(probe in (c.geom1, c.geom2) and c.dist < 0 for c in d.contact)


def test_long_fingers_detect_folded_arm_contact() -> None:
    m = model()
    d = mujoco.MjData(m)
    # Offline regression: shorter legacy fingers missed this base contact.
    d.qpos[:] = [-0.04805339, 3.56922462, 0.90355518, -1.24485844, 0.68465809, -1.96912029, 0.01399191, 0.01399191]
    mujoco.mj_forward(m, d)
    assert any(
        {m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name} == {"base", "tip_left"}
        and c.dist < -0.005
        for c in d.contact
    )
