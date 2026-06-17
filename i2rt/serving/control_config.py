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

from typing import Any, List, Optional, Union

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
# Homing return speed (rad/s) — kept slower than the engage approach so the robot
# and leaders ease back to home gently (e.g. after a leader "end episode" button).
HOME_SPEED: float = 0.4

# --- Engage / release gate ---------------------------------------------------
ENGAGE_THR: float = 0.6
RELEASE_THR: float = 0.3
DWELL_S: float = 0.5
# Which leader joint(s) drive the engage/release gate (max abs displacement from
# home over these joints). [1] = the 2nd joint only — intuitive: lift the 2nd
# joint to start, lower it to return, regardless of the other joints. Set to []
# to use the L2 distance over ALL arm joints instead.
GATE_JOINTS: List[int] = [1]

# --- Leader stiffness (gains on the human-held gello, NOT speeds) ------------
HOME_KP: float = 0.3  # pulls the leader back to home while homing
BILATERAL_KP: float = 0.0  # teleop: back-drives the leader while engaged (force feel) # This should be 0.0

# --- DAgger leader gains (separate for the two phases) -----------------------
# Intervention is TOGGLED by a handle button (press once to take over, press again
# to hand back). While intervening the human drives the follower (FEEDBACK gain on
# the leader for force feel); otherwise the policy drives and the leader mirrors
# the policy action (MIRROR gain). Mirror is usually a touch higher so the human
# feels/anticipates the policy; feedback is low so it doesn't fight the human.
DAGGER_MIRROR_KP: float = 0.2  # leader stiffness while the POLICY drives (leader mirrors policy)
DAGGER_FEEDBACK_KP: float = 0.1  # leader stiffness while the HUMAN intervenes (force feel)


# --- Wrist payload for gravity compensation (e.g. a D405 camera) -------------
# A wrist-mounted camera adds mass the arm model doesn't know about, so the
# wrist/distal joints sag. Gravity compensation is quasi-static, so only the
# added MASS (and roughly its COM) matter — not the inertia tensor. Setting this
# adds the payload to the FOLLOWER's end-effector inertial in the gravity-comp
# model (applied identically in teleop / DAgger / replay-wrapper).
FOLLOWER_PAYLOAD_KG: Optional[float] = None  # extra wrist mass in kg (D405 ≈ 0.05); None = none
FOLLOWER_EE_INERTIA: Optional[List[float]] = None  # optional [ipos(3), quat(4), diaginertia(3)] to place the COM


# --- Leader handle "end episode" buttons -------------------------------------
# Pressing any of these leader-handle buttons during teleop forces the rig to
# start HOMING (ending the episode). The recorder maps the same buttons to an
# outcome — success / fail / discard — so one press both ends and labels the
# trajectory (button 0 = discard, 1 = success, 2 = fail; see recorder.py).
HOME_BUTTONS: List[int] = [0, 1, 2]


# --- Follower workspace (joint) limits ---------------------------------------
# Per-joint [lo, hi] clamp (rad; trailing entry is the normalized 0-1 gripper)
# applied to every commanded follower target as a safety net (teleop / DAgger /
# replay / policy). None = no clamping. Provide a list of (lo, hi) up to num_dofs;
# joints past the list are left unclamped.
FOLLOWER_JOINT_LIMITS: Optional[List[tuple]] = None

# --- Collision / overload soft-stop ------------------------------------------
# If set, any follower arm joint whose |effort| (Nm) exceeds this triggers an
# automatic e-stop (the rig holds until you release it) — a simple collision /
# overload guard. None = disabled (default; tune to your arm before enabling so it
# doesn't false-trip). The gripper joint is excluded (grasping is legitimately high).
FOLLOWER_EFFORT_LIMIT: Optional[float] = None


def _gripper_base_mass(gripper_type: Any) -> Optional[float]:
    """Read the gripper body's own inertial mass (kg) from its MJCF, or None."""
    import xml.etree.ElementTree as ET

    try:
        root = ET.parse(gripper_type.get_xml_path()).getroot()
        body = root.find(".//body[@name='gripper']")
        inertial = body.find("inertial") if body is not None else None
        m = inertial.get("mass") if inertial is not None else None
        return float(m) if m is not None else None
    except Exception:
        return None


def resolve_follower_ee(arm_type: Any, gripper_type: Any) -> tuple:
    """Return ``(ee_mass, ee_inertia)`` for ``get_yam_robot`` given the wrist payload.

    ``ee_mass = gripper_base_mass + FOLLOWER_PAYLOAD_KG`` so the payload is *added*
    at the gripper COM (ee_mass overrides the gripper inertial's mass). Returns
    ``(None, FOLLOWER_EE_INERTIA)`` when no payload is configured or the base mass
    can't be read (so we never accidentally replace the gripper mass with just the
    payload).
    """
    if FOLLOWER_PAYLOAD_KG is None:
        return None, FOLLOWER_EE_INERTIA
    base = _gripper_base_mass(gripper_type)
    if base is None:
        return None, FOLLOWER_EE_INERTIA
    return base + float(FOLLOWER_PAYLOAD_KG), FOLLOWER_EE_INERTIA


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
