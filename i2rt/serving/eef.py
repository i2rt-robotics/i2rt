"""End-effector kinematics for the serving layer (reuses the company `Kinematics`).

Wraps :class:`i2rt.robots.kinematics.Kinematics` (mink FK + differential IK) for a
follower arm, exposing:

* :meth:`ArmKinematics.fk`  — current joint config -> EE pose ``[x,y,z, qw,qx,qy,qz]``
* :meth:`ArmKinematics.ik`  — EE pose target (+ current q as seed) -> joint positions

This is the **safe operational-space** building block: EE targets are resolved to
joint targets by mink's QP IK (which respects joint limits + LM damping), then the
caller commands them through the existing joint-impedance path (``command_joint_pos``
+ ``TargetSmoother`` rate limit + joint clamp + e-stop + watchdog). No torque-level
OSC, so there are no singularity torque spikes.

Everything is best-effort: if the robot exposes no model / the site is missing, the
instance reports ``available == False`` and fk/ik return ``None`` (callers degrade
gracefully — EEF obs become zeros, EEF control is a logged no-op).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# EE pose layout per arm: position (3) + quaternion wxyz (4).
EEF_POSE_DIM = 7


def _mat_to_pose(T: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> ``[x, y, z, qw, qx, qy, qz]``."""
    p = T[:3, 3]
    r = T[:3, :3]
    t = np.trace(r)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    return np.array([p[0], p[1], p[2], w, x, y, z], dtype=float)


def _pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` -> 4x4 homogeneous transform."""
    x, y, z, qw, qx, qy, qz = np.asarray(pose, dtype=float).reshape(-1)[:7]
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    T[:3, 3] = [x, y, z]
    return T


class ArmKinematics:
    def __init__(self, robot: Any, site: str = "grasp_site"):
        self.available = False
        self._site = site
        self._kin = None
        self._nq = 0
        xml_path = getattr(robot, "xml_path", None)
        if xml_path is None:
            logger.info("eef: robot exposes no xml_path; EEF kinematics disabled")
            return
        try:
            import mujoco

            from i2rt.robots.kinematics import Kinematics

            # the MuJoCo model's nq can exceed the robot's reported dofs (e.g. a 2-joint
            # gripper vs one normalized gripper value), so we pad to nq; arm joints lead,
            # so the EE-site pose is unaffected by the trailing (gripper) entries.
            self._nq = int(mujoco.MjModel.from_xml_path(xml_path).nq)
            self._kin = Kinematics(xml_path, site)
            self._kin.fk(self._pad(np.asarray(robot.get_joint_pos(), dtype=float)))  # validate
            self.available = True
        except Exception as e:
            logger.warning("eef: could not build Kinematics(%s, %s): %s", xml_path, site, e)

    def _pad(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros(self._nq, dtype=float)
        n = min(np.asarray(q).reshape(-1).size, self._nq)
        out[:n] = np.asarray(q, dtype=float).reshape(-1)[:n]
        return out

    def fk(self, q: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        try:
            return _mat_to_pose(self._kin.fk(self._pad(q)))
        except Exception:
            return None

    def ik(self, pose: np.ndarray, init_q: np.ndarray) -> Optional[np.ndarray]:
        """Solve IK for an EE ``pose`` seeded at ``init_q``; returns the full model joint
        config (length nq) or None. Arm joints lead, so the caller takes ``q[:n_arm]``."""
        if not self.available:
            return None
        try:
            ok, q = self._kin.ik(_pose_to_mat(pose), self._site, init_q=self._pad(init_q))
            return np.asarray(q, dtype=float) if ok else None
        except Exception as e:
            logger.warning("eef: ik failed: %s", e)
            return None
