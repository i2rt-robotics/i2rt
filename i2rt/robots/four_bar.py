"""Four-bar transmission between a motor and the joint it drives.

Most joints on these arms are direct-drive, so the motor's encoder reading *is* the joint angle.
``yam_7dof``'s joint6 is not: its motor turns an input crank, and the joint follows through a
four-bar, declared under ``transmissions:`` in the arm's config. The two angles are related by the linkage's geometry, which is nonlinear -- on that joint
the ratio runs from about 0.81 down to 0.14 across the travel -- so no single gear ratio or lookup
stands in for it. There is no encoder on the joint itself, so the joint angle is *computed* from the
motor's.

The convention: ``phase`` is the crank's angle from the ground line (the line joining the two fixed
pivots) at the instant the motor reads zero, which is also where the joint reads zero. Everything is
in radians.

``branch`` picks between the two ways the linkage can close, and ``sign`` orients the joint's
positive sense to match the URDF's axis. Both are geometry, not tuning: get either wrong and the map
is a mirror of the truth, which is invisible at zero and grows from there.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# The inverse is bracketed inside the usable travel rather than solved freely, so it can never
# return a solution on the far side of a dead point -- see motor_from_joint.
_INVERSE_TOL = 1e-12
_INVERSE_ITERS = 100


@dataclass(frozen=True)
class FourBarLinkage:
    """Maps a motor angle to the joint angle it produces, and back.

    Args:
        ground: distance between the two fixed pivot centres.
        crank: the link the motor turns.
        coupler: the floating link.
        rocker: the link carrying the joint.
        phase: crank angle from the ground line when the motor reads zero.
        motor_range: usable motor travel ``(lo, hi)``; brackets the inverse.
        branch: which assembly solution the mechanism is built in (+1 or -1).
        sign: orients the joint's positive direction (+1 or -1).

    Lengths share a unit; only their ratios matter.
    """

    ground: float
    crank: float
    coupler: float
    rocker: float
    phase: float
    motor_range: Tuple[float, float]
    branch: int = 1
    sign: int = -1

    def _rocker_angle(self, crank_angle: np.ndarray) -> np.ndarray:
        """Angle of the rocker about its pivot, by intersecting the coupler and rocker circles."""
        theta = np.asarray(crank_angle, dtype=float)
        ax, ay = self.crank * np.cos(theta), self.crank * np.sin(theta)
        dx, dy = self.ground - ax, -ay
        d = np.hypot(dx, dy)
        with np.errstate(invalid="ignore"):
            a = (self.coupler**2 - self.rocker**2 + d**2) / (2 * d)
            h = np.sqrt(np.where(self.coupler**2 - a**2 > 0, self.coupler**2 - a**2, np.nan))
        mx, my = ax + a * dx / d, ay + a * dy / d
        bx, by = mx - self.branch * h * dy / d, my + self.branch * h * dx / d
        return np.arctan2(by, bx - self.ground)

    def joint_from_motor(self, motor_angle: float) -> float:
        """Joint angle produced by a motor angle. Zero maps to zero."""
        moved = self._rocker_angle(self.phase + np.asarray(motor_angle, dtype=float))
        return float(self.sign * (moved - self._rocker_angle(np.asarray(self.phase))))

    def motor_from_joint(self, joint_angle: float) -> float:
        """Motor angle that produces a joint angle -- the inverse of joint_from_motor.

        Bisected inside ``motor_range``. The map is monotonic there (the tests pin that for the
        shipped linkage), so the bracket both guarantees a unique answer and keeps the result on
        the arm's own side of the linkage's dead points. A joint angle outside the
        reachable range clamps to the nearer end instead of raising: callers clip against the
        model's joint limits, which are the tighter bound anyway.
        """
        lo, hi = self.motor_range
        f_lo, f_hi = self.joint_from_motor(lo), self.joint_from_motor(hi)
        rising = f_hi > f_lo
        if (joint_angle - f_lo) * (1 if rising else -1) <= 0:
            return lo
        if (joint_angle - f_hi) * (1 if rising else -1) >= 0:
            return hi
        for _ in range(_INVERSE_ITERS):
            mid = 0.5 * (lo + hi)
            if hi - lo < _INVERSE_TOL:
                break
            if (self.joint_from_motor(mid) < joint_angle) == rising:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def ratio(self, motor_angle: float, step: float = 1e-6) -> float:
        """d(joint)/d(motor) at a motor angle, for scaling torques and gains across the linkage."""
        a = self.joint_from_motor(motor_angle - step)
        b = self.joint_from_motor(motor_angle + step)
        return (b - a) / (2 * step)
