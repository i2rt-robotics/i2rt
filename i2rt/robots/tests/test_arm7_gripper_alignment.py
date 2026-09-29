"""Check the composed right arm's end effector against the complete source URDF."""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial import cKDTree

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


def _source_model() -> mujoco.MjModel:
    path = Path(ArmType.YAM_7DOF_RIGHT.get_xml_path()).with_suffix(".urdf")
    root = ET.parse(path).getroot()
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", discardvisual="false")
    assets = {mesh.get("filename"): (path.parent / mesh.get("filename")).read_bytes() for mesh in root.iter("mesh")}
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)


def _world_vertices(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    mid = model.mesh(name).id
    gid = next(
        g
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_dataid[g] == mid
    )
    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    return model.mesh_vert[start : start + count] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]


@pytest.mark.parametrize("q", [np.zeros(7), np.array([0.3, 0.9, 0.4, -1.2, 0.3, -0.5, 0.2])])
def test_composed_end_effector_matches_source(q: np.ndarray) -> None:
    source = _source_model()
    composed = mujoco.MjModel.from_xml_path(
        combine_arm_and_gripper_xml(ArmType.YAM_7DOF_RIGHT, GripperType.LINEAR_4310)
    )
    states = [mujoco.MjData(model) for model in (source, composed)]
    for model, data in zip((source, composed), states, strict=True):
        data.qpos[:7] = q
        data.qpos[7:] = 0.02
        mujoco.mj_forward(model, data)

    for name in ("link6", "wrist", "gripper", "tip_left", "tip_right"):
        expected = _world_vertices(source, states[0], name)
        actual = _world_vertices(composed, states[1], name)
        # Compilation may reorder/recenter vertices; compare their world-space point sets.
        assert cKDTree(expected).query(actual)[0].max() < 1e-6, name
        assert cKDTree(actual).query(expected)[0].max() < 1e-6, name

    for name in ("gripper", "tip_left", "tip_right"):
        source_id, composed_id = source.body(name).id, composed.body(name).id
        np.testing.assert_allclose(composed.body_mass[composed_id], source.body_mass[source_id], atol=1e-9)
        np.testing.assert_allclose(states[1].xipos[composed_id], states[0].xipos[source_id], atol=1e-7)
        tensors = []
        for model, data, bid in zip((source, composed), states, (source_id, composed_id), strict=True):
            rotation = data.ximat[bid].reshape(3, 3)
            tensors.append(rotation @ np.diag(model.body_inertia[bid]) @ rotation.T)
        np.testing.assert_allclose(tensors[1], tensors[0], atol=1e-8)

    if not q.any():
        # At Hang the grasp point must extend down from the wrist, as the source fingers do.
        assert states[1].site("grasp_site").xpos[2] - states[1].body("gripper").xpos[2] < -0.20
        # ...and it must sit on the fingers, which is not the same as sitting under the `gripper`
        # body origin. On this arm the mount origin is 25.6 mm off the finger centreline in Y --
        # in the source URDF as well as the composed model, so it is the hardware, not the
        # composition. `grasp_site` is documented as the midpoint of the two tips' distal facets,
        # so that is what it is checked against.
        facets = []
        for name in ("tip_left", "tip_right"):
            vertices = _world_vertices(composed, states[1], name)
            facets.append(vertices[vertices[:, 2] < vertices[:, 2].min() + 1e-3].mean(axis=0))
        assert np.linalg.norm(states[1].site("grasp_site").xpos - np.mean(facets, axis=0)) < 1e-3
