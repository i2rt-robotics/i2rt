"""Shared helpers for the bimanual teleop (②) and DAgger (③) nodes.

* ``build_pair`` / ``build_bimanual`` — construct leader+follower robot pairs
* ``LatchingToggle`` — rising-edge latch for a button (press toggles a boolean)
* ``read_handle`` — read a leader's arm joints, trigger, and buttons
* ``build_follower_target`` — map a leader's arm + trigger to a follower joint target
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


@dataclass
class PairSpec:
    """CAN channels + gripper choices for one leader/follower side."""

    side: str  # "left" / "right"
    leader_channel: str
    follower_channel: str
    arm_type: str = "yam"
    leader_gripper: str = "yam_teaching_handle"
    follower_gripper: str = "linear_4310"


@dataclass
class ArmPair:
    side: str
    leader: object
    follower: object
    base_kp: Optional[np.ndarray] = None  # leader's nominal kp (for bilateral scaling)
    base_kd: Optional[np.ndarray] = None  # leader's nominal kd (for homing damping)


def _robot(channel: str, arm_type: str, gripper: str, sim: bool, zero_gravity: bool) -> Any:
    return get_yam_robot(
        channel=channel,
        arm_type=ArmType(arm_type),
        gripper_type=GripperType(gripper),
        zero_gravity_mode=zero_gravity,
        sim=sim,
    )


def build_pair(spec: PairSpec, sim: bool) -> ArmPair:
    """Build one leader (zero-gravity, human-held) + follower (PD) pair."""
    leader = _robot(spec.leader_channel, spec.arm_type, spec.leader_gripper, sim, zero_gravity=True)
    follower = _robot(spec.follower_channel, spec.arm_type, spec.follower_gripper, sim, zero_gravity=False)
    base_kp = base_kd = None
    info = leader.get_robot_info() if hasattr(leader, "get_robot_info") else {}
    if isinstance(info, dict) and "kp" in info:
        base_kp = np.asarray(info["kp"], dtype=float)
    if isinstance(info, dict) and info.get("kd") is not None:
        base_kd = np.asarray(info["kd"], dtype=float)
    return ArmPair(side=spec.side, leader=leader, follower=follower, base_kp=base_kp, base_kd=base_kd)


def build_bimanual(specs: List[PairSpec], sim: bool) -> Dict[str, ArmPair]:
    return {s.side: build_pair(s, sim) for s in specs}


class LatchingToggle:
    """Toggle a boolean on each rising edge (button press) of an input signal."""

    def __init__(self, initial: bool = False):
        self.state = initial
        self._prev = False

    def update(self, pressed: bool) -> bool:
        if pressed and not self._prev:
            self.state = not self.state
        self._prev = bool(pressed)
        return self.state


def read_handle(leader: Any) -> Tuple[np.ndarray, Optional[float], List[int]]:
    """Return ``(arm_joints, gripper_cmd, buttons)`` for a leader arm.

    A teaching-handle leader has no gripper DOF (e.g. 6 arm joints); the gripper
    *command* and the buttons come from its passive encoder (the trigger maps to
    ``1 - encoder_position``, matching ``minimum_gello``). ``gripper_cmd`` is
    ``None`` when there is no trigger source (e.g. sim). Use
    :func:`build_follower_target` to turn this into a follower-sized command.
    """
    obs = leader.get_observations()
    arm = np.asarray(obs.get("joint_pos", leader.get_joint_pos()), dtype=float).reshape(-1)
    gripper_cmd: Optional[float] = None
    buttons: List[int] = []
    mc = getattr(leader, "motor_chain", None)
    if mc is not None and getattr(mc, "same_bus_device_driver", None) is not None:
        try:
            states = mc.get_same_bus_device_states()
            if states:
                enc = states[0]
                buttons = [int(bool(b)) for b in enc.io_inputs]
                gripper_cmd = float(1.0 - enc.position)
        except Exception:
            pass
    if gripper_cmd is None and "gripper_pos" in obs:
        gripper_cmd = float(np.asarray(obs["gripper_pos"], dtype=float).reshape(-1)[0])
    return arm, gripper_cmd, buttons


def build_follower_target(follower: Any, arm: np.ndarray, gripper_cmd: Optional[float]) -> np.ndarray:
    """Map a leader's arm joints + trigger into a full follower joint target.

    Result length is ``follower.num_dofs()``: the follower's arm joints come from
    ``arm`` and (if the follower has a gripper) the trailing element is
    ``gripper_cmd`` — or the follower's current gripper position when no trigger is
    available, so the gripper simply holds.
    """
    arm = np.asarray(arm, dtype=float).reshape(-1)
    n = int(follower.num_dofs())
    has_grip = "gripper_pos" in follower.get_observations()
    if not has_grip:
        return arm[:n]
    if gripper_cmd is None:
        gripper_cmd = float(np.asarray(follower.get_observations()["gripper_pos"], dtype=float).reshape(-1)[0])
    return np.concatenate([arm[: n - 1], [float(gripper_cmd)]])


class TeleopStateMachine:
    """Global (both-arms) auto-gate for teleop, driven by leader distance from home.

    States:

    * ``HOMING``  — robot + leaders ramp to the home pose; leaves once homing is done.
    * ``IDLE``    — sitting at home, leaders free; waiting for the human to lift them.
    * ``ENGAGED`` — teleop active; followers track the leaders.

    Transitions (bimanual, decided on *both* arms together, with hysteresis):

    * ``IDLE → ENGAGED``   when **both** leaders are farther than ``engage_thr`` from home.
    * ``ENGAGED → HOMING`` when **both** leaders are within ``release_thr`` of home,
      sustained for ``dwell_s`` seconds.
    * ``HOMING → IDLE``    when the homing ramp has converged (``homing_done``).

    The thresholds use the L2 distance of the leader *arm* joints from the home
    arm joints (the gripper is ignored for gating).
    """

    HOMING = "HOMING"
    IDLE = "IDLE"
    ENGAGED = "ENGAGED"

    def __init__(self, engage_thr: float, release_thr: float, dwell_s: float):
        assert engage_thr > release_thr, "engage_thr must exceed release_thr (hysteresis)"
        self.state = self.HOMING
        self.engage_thr = float(engage_thr)
        self.release_thr = float(release_thr)
        self.dwell_s = float(dwell_s)
        self._release_since: Optional[float] = None

    def update(self, dists: List[float], homing_done: bool, now: float) -> str:
        """Advance the state machine one tick and return the new state.

        ``dists`` is the per-arm leader-to-home distance; ``homing_done`` is True
        once the homing ramp has reached the home pose; ``now`` is a monotonic time
        in seconds (for the release dwell).
        """
        both_away = bool(dists) and all(d > self.engage_thr for d in dists)
        both_home = bool(dists) and all(d < self.release_thr for d in dists)

        if self.state == self.HOMING:
            if homing_done:
                self.state = self.IDLE
        elif self.state == self.IDLE:
            if both_away:
                self.state = self.ENGAGED
        elif self.state == self.ENGAGED:
            if both_home:
                if self._release_since is None:
                    self._release_since = now
                elif now - self._release_since >= self.dwell_s:
                    self.state = self.HOMING
                    self._release_since = None
            else:
                self._release_since = None
        return self.state


def gate_distance(arm_q: np.ndarray, home_arm: np.ndarray, joints: Optional[List[int]] = None) -> float:
    """Distance of a leader's arm from home used for the engage/release gate.

    * ``joints`` empty/None → L2 norm over all arm joints (default).
    * ``joints`` given      → max absolute displacement over just those joints
      (e.g. ``[1]`` gates on the 2nd joint only — more intuitive than summing all).
    """
    a = np.asarray(arm_q, dtype=float).reshape(-1)
    h = np.asarray(home_arm, dtype=float).reshape(-1)
    n = min(a.size, h.size)
    if n == 0:
        return 0.0
    if joints:
        diffs = [abs(float(a[j] - h[j])) for j in joints if 0 <= j < n]
        return max(diffs) if diffs else 0.0
    return float(np.linalg.norm(a[:n] - h[:n]))


def default_bimanual_specs(sim: bool) -> List[PairSpec]:
    """Standard left/right channel naming used across i2rt bimanual examples."""
    if sim:
        return [
            PairSpec("left", "sim_leader_l", "sim_follower_l"),
            PairSpec("right", "sim_leader_r", "sim_follower_r"),
        ]
    return [
        PairSpec("left", "can_leader_l", "can_follower_l"),
        PairSpec("right", "can_leader_r", "can_follower_r"),
    ]
