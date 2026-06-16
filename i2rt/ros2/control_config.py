"""Global control config shared by the ROS 2 teleop / DAgger / replay pipeline.

The **follower's control gains** must be identical wherever the follower is driven
— teleop, DAgger, and replay (via the wrapper) — so a replayed episode behaves
exactly like the collected one. By default they are the arm's ``yam.yml`` gains
(already global); set the overrides here to change them in **one place**.

Steady-state following is intentionally **not** rate-limited: the follower tracks
the leader directly via these gains, matching the original ``minimum_gello``. The
ramp speed below only shapes the one-time engage approach and the homing return.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

# --- Follower arm control gains (the "follow gain") --------------------------
# None  -> use the per-arm yam.yml defaults (already identical across modes).
# float -> broadcast that value to every joint.
# list  -> per-joint vector of length num_dofs.
FOLLOWER_KP: Optional[Union[float, List[float]]] = None
FOLLOWER_KD: Optional[Union[float, List[float]]] = None

# --- Teleop transition smoothing (rad/s) -------------------------------------
# Only the one-time engage approach and the homing return are rate-limited;
# steady tracking is direct (uses FOLLOWER_KP/KD above).
RAMP_SPEED: float = 0.8

# --- Engage / release gate ---------------------------------------------------
ENGAGE_THR: float = 0.6
RELEASE_THR: float = 0.3
DWELL_S: float = 0.5
GATE_JOINTS: List[int] = []  # [] = L2 over all arm joints; [1] = 2nd joint only

# --- Leader stiffness (gains on the human-held gello, NOT speeds) ------------
HOME_KP: float = 0.3       # pulls the leader back to home while homing
BILATERAL_KP: float = 0.0  # back-drives the leader while engaged (force feel)


def _broadcast(val: Optional[Union[float, List[float]]], default: np.ndarray) -> np.ndarray:
    if val is None:
        return np.asarray(default, dtype=float)
    if np.isscalar(val):
        return np.full(len(default), float(val))
    return np.asarray(val, dtype=float)


def apply_follower_gains(robot: object) -> None:
    """Enforce the global follower gains (if overridden) so every mode matches.

    No-op when both overrides are ``None`` (the robot keeps its ``yam.yml`` gains,
    which are already shared by teleop/DAgger/replay).
    """
    if FOLLOWER_KP is None and FOLLOWER_KD is None:
        return
    if not (hasattr(robot, "update_kp_kd") and hasattr(robot, "get_robot_info")):
        return
    try:
        info = robot.get_robot_info()
        kp = _broadcast(FOLLOWER_KP, info["kp"])
        kd = _broadcast(FOLLOWER_KD, info["kd"])
        robot.update_kp_kd(kp, kd)
    except Exception:
        pass
