"""Bimanual controllers.

Each controller owns the robot pairs and runs the real-time control law in
``step()`` (call it from a fixed-rate loop on the robot). It keeps a thread-safe
``snapshot()`` dict and reads external inputs (policy action, gate override, replay
command) through setters. A :class:`~i2rt.serving.robot_server.RobotServer` wraps any
of these and exposes the snapshot + setters over the network.

Three modes:

* :class:`TeleopController`  — auto home/engage gate, bilateral leader→follower teleop
* :class:`DaggerController`  — HG-DAgger: policy drives, a button hands control to the human
* :class:`WrapperController` — followers track an external command (replay / direct control)
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from i2rt.serving import control_config as cc
from i2rt.serving.eef import ArmKinematics
from i2rt.serving.safety import TargetSmoother, clamp_limits, is_finite_vector, max_step_from_speed
from i2rt.serving.state_utils import full_state, to_full_target
from i2rt.serving.teleop_common import (
    ArmPair,
    LatchingToggle,
    TeleopStateMachine,
    build_bimanual,
    build_follower_target,
    default_bimanual_specs,
    gate_distance,
    read_handle,
)

logger = logging.getLogger(__name__)

_HOME_TOL = 0.05  # rad; homing considered done when the ramp is within this of home


def _side_state(robot: Any, kin: Optional[ArmKinematics] = None) -> Dict[str, list]:
    pos, vel, eff = full_state(robot)
    out = {"pos": pos.tolist(), "vel": vel.tolist(), "eff": eff.tolist()}
    pose = kin.fk(robot.get_joint_pos()) if kin is not None else None
    if pose is not None:
        out["eef"] = pose.tolist()
    return out


def _build_kin(robots: Dict[str, Any]) -> Dict[str, ArmKinematics]:
    """One ArmKinematics per follower (for EEF FK in the snapshot / IK control)."""
    return {side: ArmKinematics(robot) for side, robot in robots.items()}


class BaseController:
    """Shared transport surface for the controllers.

    Provides the thread-safe ``snapshot()``/``metadata()`` readers and **no-op
    defaults** for every input hook the :class:`~i2rt.serving.robot_server.RobotServer`
    binds; each controller overrides only the hooks it actually supports. Subclasses
    set ``self._lock``, ``self._snap`` and ``self._metadata`` in ``__init__``.
    """

    mode = "base"
    _estop = False
    command_timeout = 0.5  # s; external commands older than this are considered stale (link loss)
    _last_cmd_t = -1e9

    def snapshot(self) -> Dict:
        with self._lock:
            return dict(self._snap)

    def metadata(self) -> Dict:
        return dict(self._metadata)

    def set_estop(self, flag: bool) -> None:
        """Engage/release the e-stop. While engaged, no follower commands are sent."""
        self._estop = bool(flag)

    def _effort_guard(self, robot: Any) -> None:
        """Collision/overload guard: trip the e-stop if a follower arm effort exceeds
        ``FOLLOWER_EFFORT_LIMIT`` (no-op when unset or already e-stopped)."""
        lim = cc.FOLLOWER_EFFORT_LIMIT
        if lim is None or self._estop:
            return
        try:
            _, _, eff = full_state(robot)
            arm = np.abs(np.asarray(eff, dtype=float).reshape(-1)[:-1])  # exclude gripper
            if arm.size and float(arm.max()) > float(lim):
                logger.warning("effort guard tripped (|eff|=%.1f > %.1f Nm) -> e-stop", float(arm.max()), lim)
                self.set_estop(True)
        except Exception:
            pass

    def _touch_cmd(self) -> None:
        """Mark that a fresh external command just arrived (for the staleness watchdog)."""
        self._last_cmd_t = time.monotonic()

    def _cmd_fresh(self) -> bool:
        """True if an external command arrived within ``command_timeout`` — else the link
        is presumed lost and the follower should hold instead of replaying a stale target."""
        return (time.monotonic() - self._last_cmd_t) < self.command_timeout

    def _apply(self, robot: Any, target: np.ndarray) -> Optional[list]:
        """Clamp ``target`` to the workspace limits and command it — unless e-stopped.

        Returns the commanded target as a list (for the snapshot), or None when
        e-stopped (the follower simply holds its last command).
        """
        if self._estop:
            return None
        target = clamp_limits(target, cc.FOLLOWER_JOINT_LIMITS)
        robot.command_joint_pos(target)
        return np.asarray(target, dtype=float).tolist()

    def set_policy_action(self, data: Dict) -> None: ...
    def set_intervention(self, flag: bool) -> None: ...
    def command(self, data: Dict) -> None: ...
    def set_sim_engage(self, flag: bool) -> None: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Teleop
# ---------------------------------------------------------------------------
@dataclass
class TeleopConfig:
    sim: bool = False
    home: str = ""
    engage_thr: float = cc.ENGAGE_THR
    release_thr: float = cc.RELEASE_THR
    dwell: float = cc.DWELL_S
    home_kp: float = cc.HOME_KP
    bilateral_kp: float = cc.BILATERAL_KP
    rate: float = 120.0
    ramp_speed: float = cc.RAMP_SPEED
    home_speed: float = cc.HOME_SPEED  # slower ramp for the homing return
    gate_joints: str = ",".join(str(j) for j in cc.GATE_JOINTS)


class TeleopController(BaseController):
    mode = "teleop"

    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.bilateral_kp = cfg.bilateral_kp
        self.home_kp = cfg.home_kp
        self.pairs = build_bimanual(default_bimanual_specs(cfg.sim), sim=cfg.sim)
        self._ramp_step = max_step_from_speed(cfg.ramp_speed, cfg.rate)
        self._home_step = max_step_from_speed(cfg.home_speed, cfg.rate)
        self._gate_joints = [int(x) for x in cfg.gate_joints.split(",") if x.strip() != ""] if cfg.gate_joints else []
        self._caught_up = {s: False for s in self.pairs}
        self._home_d0 = {s: 0.0 for s in self.pairs}  # start distance for the homing cosine profile
        self._engage_d0 = {s: 0.0 for s in self.pairs}  # start distance for the engage cosine approach
        self._prev_state = TeleopStateMachine.HOMING

        first = next(iter(self.pairs.values())).follower
        n = int(first.num_dofs())
        self._has_grip = "gripper_pos" in first.get_observations()
        n_arm = n - 1 if self._has_grip else n
        self.home_arm, self.home_grip = self._parse_home(cfg.home, n_arm)
        self.home_full = np.concatenate([self.home_arm, [self.home_grip]]) if self._has_grip else self.home_arm.copy()

        self.sm = TeleopStateMachine(cfg.engage_thr, cfg.release_thr, cfg.dwell)
        self._sim_engage = False

        self._kin = _build_kin({s: p.follower for s, p in self.pairs.items()})
        self._fsmooth, self._lsmooth = {}, {}
        for side, pair in self.pairs.items():
            self._fsmooth[side] = TargetSmoother(pair.follower.get_joint_pos(), self._ramp_step)
            self._lsmooth[side] = TargetSmoother(np.asarray(pair.leader.get_joint_pos())[:n_arm], self._ramp_step)

        self._lock = threading.Lock()
        self._snap: Dict = {"mode": self.mode, "t": 0.0, "teleop_state": "HOMING", "active": False}
        self._metadata = {"mode": self.mode, "sides": list(self.pairs), "has_gripper": self._has_grip}
        gate_desc = f"joints{self._gate_joints}" if self._gate_joints else "L2(all)"
        logger.info(
            "TeleopController up: sides=%s home_arm=%s gate=%s engage>%s release<%s ramp_speed=%s bilateral_kp=%s sim=%s",
            list(self.pairs),
            np.round(self.home_arm, 2).tolist(),
            gate_desc,
            cfg.engage_thr,
            cfg.release_thr,
            cfg.ramp_speed,
            cfg.bilateral_kp,
            cfg.sim,
        )

    @staticmethod
    def _parse_home(home_str: str, n_arm: int) -> "tuple[np.ndarray, float]":
        if not home_str:
            return np.zeros(n_arm), 0.0
        vals = [float(x) for x in home_str.split(",") if x.strip() != ""]
        if len(vals) == n_arm:
            return np.asarray(vals), 0.0
        if len(vals) == n_arm + 1:
            return np.asarray(vals[:n_arm]), float(vals[n_arm])
        raise ValueError(f"home expects {n_arm} or {n_arm + 1} values, got {len(vals)}")

    # ---- external inputs (called from portal handlers) ----------------------
    def set_sim_engage(self, flag: bool) -> None:
        self._sim_engage = bool(flag)

    @staticmethod
    def _home_button_pressed(buttons: Dict[str, list]) -> bool:
        return any(idx < len(b) and bool(b[idx]) for b in buttons.values() for idx in cc.HOME_BUTTONS)

    def _homing_done(self) -> bool:
        for side in self.pairs:
            if np.linalg.norm(self._fsmooth[side].cur - self.home_full) > _HOME_TOL:
                return False
            if np.linalg.norm(self._lsmooth[side].cur - self.home_arm) > _HOME_TOL:
                return False
        return True

    @staticmethod
    def _ease_vel_scale(p: float) -> float:
        """Raised-cosine speed multiplier as a function of progress ``p`` (0 at the
        start, 1 at the target): ~0.5x at the ends, ~1.28x through the middle,
        averaging ~1x -- a smooth ease-in/out that's quicker in the middle. Used for
        both the homing return and the engage approach."""
        return float(0.5 + 0.785 * np.sin(np.pi * min(max(p, 0.0), 1.0)))

    # ---- one control tick (port of TeleopNode._loop) ------------------------
    def step(self) -> None:
        now = time.monotonic()
        arm_q, grip, buttons, dists = {}, {}, {}, []
        for side, pair in self.pairs.items():
            try:
                a, g, b = read_handle(pair.leader)
            except Exception as e:
                logger.warning("[%s] handle read failed: %s", side, e)
                a, g, b = np.asarray(pair.leader.get_joint_pos(), dtype=float), None, []
            arm_q[side], grip[side], buttons[side] = a, g, b
            dists.append(gate_distance(a, self.home_arm, self._gate_joints))

        state = self.sm.update(dists, self._homing_done(), now)
        if self._sim_engage:
            state = TeleopStateMachine.ENGAGED
        # a leader "end episode" button (success/fail) forces homing while engaged
        if state == TeleopStateMachine.ENGAGED and self._home_button_pressed(buttons):
            self.sm.state = state = TeleopStateMachine.HOMING
        if state == TeleopStateMachine.ENGAGED and self._prev_state != TeleopStateMachine.ENGAGED:
            for s in self.pairs:
                self._caught_up[s] = False
                self._fsmooth[s].reset(self.pairs[s].follower.get_joint_pos())

        sides_snap: Dict[str, Dict] = {}
        for side, pair in self.pairs.items():
            n = pair.follower.num_dofs()
            fsm, lsm = self._fsmooth[side], self._lsmooth[side]
            applied = None
            try:
                self._effort_guard(pair.follower)
                if state == TeleopStateMachine.ENGAGED:
                    desired = build_follower_target(pair.follower, arm_q[side], grip[side])
                    if not is_finite_vector(desired, n):
                        desired = fsm.cur
                    if self._caught_up[side]:
                        applied = desired
                        fsm.reset(applied)
                    else:
                        # Cosine ease for the catch-up from home to the (live) leader:
                        # gentle off home, quicker through the middle, gentle on arrival.
                        d = float(np.linalg.norm(fsm.cur - desired))
                        if self._prev_state != TeleopStateMachine.ENGAGED:
                            self._engage_d0[side] = max(d, 1e-6)  # capture the initial gap once
                        p = 1.0 - d / max(self._engage_d0[side], 1e-6)
                        fsm.max_step = self._ramp_step * self._ease_vel_scale(p)
                        applied = fsm.step(desired)
                        if float(np.max(np.abs(fsm.cur - desired))) < _HOME_TOL:
                            self._caught_up[side] = True
                    # Only back-drive the leader once the follower has CAUGHT UP. Before
                    # that the follower is still near home while the leader is lifted, so
                    # back-driving would yank the leader toward home — keep it free instead.
                    if self.bilateral_kp > 0.0 and self._caught_up[side]:
                        self._drive_leader(pair, np.asarray(pair.follower.get_joint_pos())[: pair.leader.num_dofs()])
                    else:
                        self._free_leader(pair)
                    lsm.reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])
                elif state == TeleopStateMachine.HOMING:
                    # Cosine velocity profile: ease in/out at the ends, faster through
                    # the middle (avg ≈ home_speed, so total time stays similar but the
                    # return is smooth rather than a constant crawl).
                    d = float(np.linalg.norm(fsm.cur - self.home_full))
                    if self._prev_state != TeleopStateMachine.HOMING:
                        self._home_d0[side] = max(d, 1e-6)  # capture the start distance once
                    p = min(max(1.0 - d / max(self._home_d0[side], 1e-6), 0.0), 1.0)
                    fsm.max_step = lsm.max_step = self._home_step * self._ease_vel_scale(p)
                    applied = fsm.step(self.home_full)
                    self._home_leader(pair, lsm.step(self.home_arm))
                else:  # IDLE
                    fsm.max_step = self._ramp_step
                    applied = fsm.step(self.home_full)
                    self._free_leader(pair)
                    lsm.reset(np.asarray(pair.leader.get_joint_pos())[: self.home_arm.size])

                applied_list = self._apply(pair.follower, applied)
                snap = _side_state(pair.follower, self._kin.get(side))
                snap["leader_pos"] = np.asarray(pair.leader.get_joint_pos(), dtype=float).tolist()
                snap["buttons"] = list(buttons[side])
                snap["gripper_cmd"] = float(grip[side]) if grip[side] is not None else 0.0
                snap["applied"] = applied_list
                sides_snap[side] = snap
            except Exception as e:
                logger.warning("[%s] teleop step failed: %s", side, e)

        with self._lock:
            self._snap = {
                "mode": self.mode,
                "t": now,
                "teleop_state": state,
                "active": state == TeleopStateMachine.ENGAGED,
                "estop": self._estop,
                **sides_snap,
            }
        self._prev_state = state

    # ---- leader modes ------------------------------------------------------
    def _home_leader(self, pair: ArmPair, target_arm: np.ndarray) -> None:
        leader = pair.leader
        if not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            kd = pair.base_kd[:m] if pair.base_kd is not None else np.full(m, 0.5)
            leader.update_kp_kd(pair.base_kp[:m] * self.home_kp, kd)
            leader.command_joint_pos(np.asarray(target_arm, dtype=float)[:m])
        except Exception:
            pass

    def _free_leader(self, pair: ArmPair) -> None:
        leader = pair.leader
        if hasattr(leader, "enter_gravity_comp_idle"):
            try:
                leader.enter_gravity_comp_idle()
            except Exception:
                pass

    def _drive_leader(self, pair: ArmPair, target_q: np.ndarray) -> None:
        leader = pair.leader
        if self.bilateral_kp <= 0.0 or not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            leader.update_kp_kd(pair.base_kp[:m] * self.bilateral_kp, np.zeros(m))
            leader.command_joint_pos(np.asarray(target_q, dtype=float)[:m])
        except Exception:
            pass

    def close(self) -> None:
        for pair in self.pairs.values():
            for r in (pair.leader, pair.follower):
                try:
                    r.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# DAgger
# ---------------------------------------------------------------------------
@dataclass
class DaggerConfig:
    sim: bool = False
    mirror_kp: float = cc.DAGGER_MIRROR_KP
    feedback_kp: float = cc.DAGGER_FEEDBACK_KP
    rate: float = 120.0
    max_joint_speed: float = 1.5
    command_timeout: float = 0.5  # s; stale policy actions (link loss) are ignored -> hold


class DaggerController(BaseController):
    mode = "dagger"

    def __init__(self, cfg: DaggerConfig):
        self.cfg = cfg
        self.command_timeout = cfg.command_timeout
        self.mirror_kp = cfg.mirror_kp
        self.feedback_kp = cfg.feedback_kp
        self.pairs = build_bimanual(default_bimanual_specs(cfg.sim), sim=cfg.sim)
        max_step = max_step_from_speed(cfg.max_joint_speed, cfg.rate)

        self._intervening = False
        self._toggle = LatchingToggle(initial=False)
        self._policy_action: Dict[str, Optional[np.ndarray]] = {s: None for s in self.pairs}
        self._kin = _build_kin({s: p.follower for s, p in self.pairs.items()})
        self._smooth = {s: TargetSmoother(p.follower.get_joint_pos(), max_step) for s, p in self.pairs.items()}
        self._has_grip = "gripper_pos" in next(iter(self.pairs.values())).follower.get_observations()

        self._lock = threading.Lock()
        self._snap: Dict = {"mode": self.mode, "t": 0.0, "intervention": False}
        self._metadata = {"mode": self.mode, "sides": list(self.pairs), "has_gripper": self._has_grip}
        logger.info(
            "DaggerController up: sides=%s mirror_kp=%s feedback_kp=%s max_joint_speed=%s sim=%s",
            list(self.pairs),
            cfg.mirror_kp,
            cfg.feedback_kp,
            cfg.max_joint_speed,
            cfg.sim,
        )

    # ---- external inputs ----------------------------------------------------
    def set_policy_action(self, data: Dict) -> None:
        """data = {side: position_array}. Stored as full-length follower targets."""
        for side, pos in (data or {}).items():
            if side not in self.pairs:
                continue
            try:
                self._policy_action[side] = to_full_target(np.asarray(pos, dtype=float), self.pairs[side].follower)
            except ValueError as e:
                logger.warning("[%s] bad policy_action: %s", side, e)
        self._touch_cmd()

    def set_intervention(self, flag: bool) -> None:
        self._toggle.state = bool(flag)

    # ---- one control tick (port of DaggerNode._loop) ------------------------
    def step(self) -> None:
        now = time.monotonic()
        arm_q, grip_cmd, buttons, valid = {}, {}, {}, {}
        for side, pair in self.pairs.items():
            try:
                a, g, b = read_handle(pair.leader)
            except Exception as e:
                logger.warning("[%s] handle read failed: %s", side, e)
                a, g, b = np.zeros(pair.leader.num_dofs()), None, []
            arm_q[side], grip_cmd[side], buttons[side] = a, g, b
            valid[side] = is_finite_vector(a, pair.leader.num_dofs())

        pressed = any(bool(b[0]) for b in buttons.values() if b)
        self._intervening = self._toggle.update(pressed)

        sides_snap: Dict[str, Dict] = {}
        for side, pair in self.pairs.items():
            n = pair.follower.num_dofs()
            smoother = self._smooth[side]
            applied = None
            human = None
            try:
                self._effort_guard(pair.follower)
                desired = None
                if self._intervening:
                    if valid[side]:
                        human = build_follower_target(pair.follower, arm_q[side], grip_cmd[side])
                        desired = human
                        self._drive_leader(
                            pair, np.asarray(pair.follower.get_joint_pos())[: pair.leader.num_dofs()], self.feedback_kp
                        )
                else:
                    act = self._policy_action[side]
                    # ignore a stale policy action (workstation/link down) -> follower holds
                    if is_finite_vector(act, n) and self._cmd_fresh():
                        desired = act[:n]
                        self._drive_leader(pair, act[: pair.leader.num_dofs()], self.mirror_kp)

                if desired is not None:
                    target = smoother.step(desired)
                    applied = self._apply(pair.follower, target)
                else:
                    smoother.reset(pair.follower.get_joint_pos())

                snap = _side_state(pair.follower, self._kin.get(side))
                snap["leader_pos"] = np.asarray(pair.leader.get_joint_pos(), dtype=float).tolist()
                snap["buttons"] = list(buttons[side])
                snap["gripper_cmd"] = float(grip_cmd[side]) if grip_cmd[side] is not None else 0.0
                snap["applied"] = applied
                snap["human"] = np.asarray(human, dtype=float).tolist() if human is not None else None
                sides_snap[side] = snap
            except Exception as e:
                logger.warning("[%s] dagger step failed: %s", side, e)

        with self._lock:
            self._snap = {
                "mode": self.mode,
                "t": now,
                "intervention": bool(self._intervening),
                "estop": self._estop,
                **sides_snap,
            }

    def _drive_leader(self, pair: ArmPair, target_q: np.ndarray, kp_scale: float) -> None:
        leader = pair.leader
        if kp_scale <= 0.0 or not hasattr(leader, "update_kp_kd") or pair.base_kp is None:
            return
        try:
            m = leader.num_dofs()
            leader.update_kp_kd(pair.base_kp[:m] * kp_scale, np.zeros(m))
            leader.command_joint_pos(np.asarray(target_q, dtype=float)[:m])
        except Exception:
            pass

    def close(self) -> None:
        for pair in self.pairs.values():
            for r in (pair.leader, pair.follower):
                try:
                    r.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Wrapper / replay — followers track an external command
# ---------------------------------------------------------------------------
@dataclass
class WrapperConfig:
    sim: bool = False
    arm_type: str = "yam"
    gripper: str = "linear_4310"
    rate: float = 100.0
    max_joint_speed: float = 1.5
    control: str = "joint"  # "joint" (rate-limited joint targets) or "eef" (end-effector pose; experimental)
    command_timeout: float = 0.5  # s; stale commands (link loss) are ignored -> hold
    channels: Dict[str, str] = field(default_factory=lambda: {"left": "can_follower_l", "right": "can_follower_r"})


class WrapperController(BaseController):
    mode = "wrapper"

    def __init__(self, cfg: WrapperConfig):
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        self.cfg = cfg
        self.command_timeout = cfg.command_timeout
        self.followers = {}
        for side, channel in cfg.channels.items():
            ch = ("sim_" + channel) if cfg.sim else channel
            self.followers[side] = get_yam_robot(
                channel=ch,
                arm_type=ArmType(cfg.arm_type),
                gripper_type=GripperType(cfg.gripper),
                zero_gravity_mode=False,
                sim=cfg.sim,
            )
            cc.apply_follower_gains(self.followers[side])

        self._kin = _build_kin(self.followers)
        max_step = max_step_from_speed(cfg.max_joint_speed, cfg.rate)
        self._smooth = {s: TargetSmoother(f.get_joint_pos(), max_step) for s, f in self.followers.items()}
        self._command: Dict[str, Optional[np.ndarray]] = {s: None for s in self.followers}
        self._has_grip = "gripper_pos" in next(iter(self.followers.values())).get_observations()
        self._lock = threading.Lock()
        self._snap: Dict = {"mode": self.mode, "t": 0.0}
        self._metadata = {"mode": self.mode, "sides": list(self.followers), "has_gripper": self._has_grip}
        logger.info("WrapperController up: sides=%s sim=%s", list(self.followers), cfg.sim)

    def command(self, data: Dict) -> None:
        """data = {side: target} for each follower (joint positions, or eef pose in eef mode)."""
        if self.cfg.control == "eef":
            self._command_eef(data)
            return
        for side, pos in (data or {}).items():
            if side not in self.followers:
                continue
            try:
                self._command[side] = to_full_target(np.asarray(pos, dtype=float), self.followers[side])
            except ValueError as e:
                logger.warning("[%s] bad command: %s", side, e)
        self._touch_cmd()

    def _command_eef(self, data: Dict) -> None:
        """Safe operational-space (resolved-rate) control: resolve each EE pose target to
        joint positions with the company IK (mink, limits + damping), seeded at the
        current pose, then drive them through the SAME joint path as joint mode — the
        TargetSmoother rate limit, joint clamp, e-stop and watchdog all still apply.
        """
        for side, pose in (data or {}).items():
            follower = self.followers.get(side)
            kin = self._kin.get(side)
            if follower is None or kin is None or not kin.available:
                if not getattr(self, "_eef_warned", False):
                    logger.warning("eef control requested but no IK model is available; ignoring")
                    self._eef_warned = True
                continue
            cur = np.asarray(follower.get_joint_pos(), dtype=float)
            q = kin.ik(np.asarray(pose, dtype=float), init_q=cur)  # full model config (nq) or None
            if q is None:
                continue
            n_arm = cur.size - 1  # gripper is the trailing dof; arm joints lead
            target = cur.copy()
            target[:n_arm] = q[:n_arm]  # IK arm solution; keep the current gripper opening
            self._command[side] = target
        self._touch_cmd()

    # a policy can drive the wrapper too (treated as a direct command)
    set_policy_action = command

    def step(self) -> None:
        now = time.monotonic()
        sides_snap: Dict[str, Dict] = {}
        for side, follower in self.followers.items():
            smoother = self._smooth[side]
            applied = None
            try:
                self._effort_guard(follower)
                cmd = self._command[side]
                # ignore stale commands (link loss) -> hold instead of replaying an old target
                if is_finite_vector(cmd, follower.num_dofs()) and self._cmd_fresh():
                    applied = self._apply(follower, smoother.step(cmd))
                else:
                    smoother.reset(follower.get_joint_pos())
                snap = _side_state(follower, self._kin.get(side))
                snap["applied"] = applied
                sides_snap[side] = snap
            except Exception as e:
                logger.warning("[%s] wrapper step failed: %s", side, e)
        with self._lock:
            self._snap = {"mode": self.mode, "t": now, "estop": self._estop, **sides_snap}

    def close(self) -> None:
        for f in self.followers.values():
            try:
                f.close()
            except Exception:
                pass
