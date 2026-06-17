"""Safety helpers for the ROS 2 control nodes.

The single most important safety net is :class:`TargetSmoother`, a per-joint
rate limiter. Whenever the *desired* follower target changes discontinuously —
switching between policy and human control, a stale→fresh policy action, or a
dropped reading — it bounds how far the commanded target may move per control
tick, so the follower ramps smoothly instead of snapping at unsafe speed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TargetSmoother:
    """Rate-limit a commanded joint target so it never jumps per tick.

    ``max_step`` is the maximum absolute change per joint per ``step()`` call,
    typically ``max_joint_speed / control_rate``. The same bound is applied to
    the (normalized 0-1) gripper element, which is a conservative, safe choice.
    """

    def __init__(self, current: np.ndarray, max_step: float):
        self.cur = np.asarray(current, dtype=float).reshape(-1).copy()
        self.max_step = float(max_step)

    def reset(self, current: np.ndarray) -> None:
        """Snap the internal state to ``current`` (use when control is handed off)."""
        self.cur = np.asarray(current, dtype=float).reshape(-1).copy()

    def step(self, desired: np.ndarray) -> np.ndarray:
        """Advance toward ``desired`` by at most ``max_step`` per joint; return the new target."""
        desired = np.asarray(desired, dtype=float).reshape(-1)
        if desired.shape != self.cur.shape:
            return self.cur  # ignore shape mismatch; hold last safe target
        delta = np.clip(desired - self.cur, -self.max_step, self.max_step)
        self.cur = self.cur + delta
        return self.cur


def max_step_from_speed(max_joint_speed: float, rate_hz: float) -> float:
    """Convert a max joint speed (rad/s) and a control rate (Hz) to a per-tick step."""
    return float(max_joint_speed) / max(float(rate_hz), 1.0)


def is_finite_vector(x: Optional[np.ndarray], n: Optional[int] = None) -> bool:
    """True iff ``x`` is a finite-valued vector (optionally of length ``n``)."""
    if x is None:
        return False
    x = np.asarray(x, dtype=float).reshape(-1)
    if n is not None and x.size != n:
        return False
    return bool(x.size) and bool(np.all(np.isfinite(x)))


def clamp_limits(target: np.ndarray, limits: Optional[list]) -> np.ndarray:
    """Clamp each joint of ``target`` into its ``[lo, hi]`` workspace limit.

    ``limits`` is a list of ``(lo, hi)`` per joint (up to ``len(target)`` entries);
    ``None``/empty means no clamping. Joints beyond ``limits`` are left untouched.
    """
    if not limits:
        return target
    t = np.asarray(target, dtype=float).reshape(-1).copy()
    for i, lohi in enumerate(limits):
        if i >= t.size or lohi is None:
            continue
        lo, hi = lohi
        t[i] = min(max(t[i], float(lo)), float(hi))
    return t
