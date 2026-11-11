"""The LinearBot model sits in the flow base's odometry frame, on the ground, with its rail joint in metres."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from i2rt.flow_base.flow_base_controller import h_x, h_y

URDF = Path(__file__).resolve().parents[2] / "robot_models" / "linearbot" / "linearbot.urdf"


def test_linearbot_model_matches_the_base_frame() -> None:
    model = mujoco.MjModel.from_xml_path(str(URDF))
    data = mujoco.MjData(model)
    rail = model.joint("linear_rail")
    np.testing.assert_array_equal(model.jnt_range[rail.id], [0.0, 1.0])
    mujoco.mj_kinematics(model, data)

    # Steering axes on the swerve kinematics' square. The export's motorN names are not the CAN ids, so
    # compare the layout, and pin the heading by the rail holder: forward (+x) is the side it faces.
    steer_xy = sorted((data.body(f"motor{k}").xpos[:2].copy() for k in (1, 3, 5, 7)), key=lambda p: tuple(np.sign(p)))
    expected = sorted(zip(h_x, h_y, strict=True), key=lambda p: tuple(np.sign(p)))
    np.testing.assert_allclose(steer_xy, expected, atol=0.005)
    assert data.body("linear_lift_holder").xipos[0] > 0.1

    # Wheels on the ground: the lowest vertex of the whole robot is at z = 0.
    lowest = np.inf
    for g, mesh in enumerate(model.geom_dataid):
        v0 = model.mesh_vertadr[mesh]
        verts = model.mesh_vert[v0 : v0 + model.mesh_vertnum[mesh]]
        z = verts @ data.geom_xmat[g].reshape(3, 3)[2] + data.geom_xpos[g][2]
        lowest = min(lowest, z.min())
    assert lowest == pytest.approx(0.0, abs=1e-3)

    # Each arm mount is an arm base frame on its own side of the carriage: +x forward, +z up.
    for name, side in (("left_arm_mount", 1), ("right_arm_mount", -1)):
        assert np.sign(data.body(name).xpos[1]) == side
        # 1e-5: the export writes the column's half-turn as 3.14159, 2.7e-6 rad short of pi.
        np.testing.assert_allclose(data.body(name).xmat.reshape(3, 3), np.eye(3), atol=1e-5)

    # The joint value is the rail height in metres, the z a LinearBot's odometry reports.
    home = float(data.body("linear_lift_holder").xpos[2])
    data.qpos[rail.qposadr[0]] = 1.0
    mujoco.mj_kinematics(model, data)
    assert data.body("linear_lift_holder").xpos[2] - home == pytest.approx(1.0)


def test_linearbot_xml_matches_the_urdf() -> None:
    """linearbot.xml is generated from the URDF; this catches one being edited without the other."""
    urdf, xml = (mujoco.MjModel.from_xml_path(str(URDF.with_suffix(ext))) for ext in (".urdf", ".xml"))
    assert [urdf.body(i).name for i in range(urdf.nbody)] == [xml.body(i).name for i in range(xml.nbody)]
    np.testing.assert_allclose(xml.body_mass, urdf.body_mass)
    for q in (np.zeros(urdf.nq), np.r_[np.linspace(-3, 3, urdf.nq - 1), 0.7]):  # every caster turned, rail up
        poses = []
        for model in (urdf, xml):
            data = mujoco.MjData(model)
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            poses.append(np.hstack([data.xpos, data.xmat]))
        np.testing.assert_allclose(poses[1], poses[0], atol=1e-6)
