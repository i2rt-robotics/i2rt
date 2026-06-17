"""EEF kinematics helper: pose<->matrix math + graceful degradation."""

from __future__ import annotations

import numpy as np

from i2rt.serving.eef import ArmKinematics, _mat_to_pose, _pose_to_mat


def test_pose_matrix_roundtrip():
    # build a transform from a known quaternion + translation, round-trip it
    pose = np.array([0.1, -0.2, 0.3, np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])  # 90deg about y
    T = _pose_to_mat(pose)
    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], [0.1, -0.2, 0.3])
    back = _mat_to_pose(T)
    # quaternion sign can flip; compare via the rotation matrix instead
    assert np.allclose(_pose_to_mat(back)[:3, :3], T[:3, :3], atol=1e-6)


class _NoModelRobot:
    def get_joint_pos(self):
        return np.zeros(7)


def test_degrades_without_model():
    kin = ArmKinematics(_NoModelRobot())  # no xml_path -> unavailable
    assert kin.available is False
    assert kin.fk(np.zeros(7)) is None
    assert kin.ik(np.zeros(7), np.zeros(7)) is None
