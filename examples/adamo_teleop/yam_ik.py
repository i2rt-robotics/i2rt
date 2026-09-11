"""PyRoki inverse kinematics for the YAM arm, tuned for interactive teleop.

A damped least-squares pose solve (Levenberg-Marquardt, via jaxls) warm-started from the
previous solution every tick. That is what makes it behave like differential IK: each call
takes a small step from where the arm already is, so the configuration stays continuous and
the arm never flips elbow between frames, but it is still a full nonlinear solve, so it
recovers cleanly after a clutch jump or a target that leaves the workspace.

Adapted from adamo-network's ``sim/yam/yam_ik.py`` to i2rt's own shipped URDF, which is the
kinematic source of truth here. Two things follow from that swap:

* i2rt's ``yam.urdf`` carries the two prismatic gripper fingers (``joint7``/``joint8``)
  alongside ``joint1..joint6``, so pyroki's configuration vector is 8 long. Everything
  public here is the 6 arm joints in ``joint1..joint6`` order and maps in and out by name --
  pyroki topologically sorts the URDF's joints, which for this URDF reverses them. The
  fingers are frozen, which costs nothing: they hang below the link the pose cost drives.
* The tool frame is the combined arm+gripper model's ``grasp_site``, so the pose being
  solved for is the same point the mink solver was driving.

The solver comes from the ``teleop`` dependency group, which ``uv sync`` / ``uv run`` install
by default; it is not in ``pip install i2rt``.
"""

import os
from functools import cached_property

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy

from i2rt.robots.utils import ArmType

# The link the pose cost drives: joint6's child, and the frame the gripper model mounts on.
TIP_LINK = "gripper"

ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

# Tool center point: ``grasp_site``'s pose in TIP_LINK's frame, read off the combined
# yam + linear_4310 MJCF. The quaternion is not identity -- the site is flipped 180 degrees
# about x relative to the mount -- so the tool offset is a full transform, not a translation.
TCP_OFFSET = np.array([9.68103407e-05, 3.88982673e-05, -1.44650259e-01])
TCP_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])

# Comfortable teleop posture: elbow up, wrist neutral. Also the redundancy / posture bias.
REST_POSE = np.array([0.0, 1.047, 1.047, 0.0, 0.0, 0.0])

# Per-joint speed caps in rad/s. The URDF declares a flat placeholder for every joint, which
# is far too permissive to rate-limit teleop against, so we set our own: the big proximal
# DM4340s move slower than the distal DM4310 wrist.
VELOCITY_LIMITS = np.array([2.5, 2.5, 2.5, 4.0, 4.0, 4.0])


@jdc.pytree_dataclass
class IKWeights:
    """Relative weights of the least-squares terms.

    Tracking terms fight the regularizers, so these are only meaningful relative to each
    other. The shipped values are tuned for a human hand at ~60-90 Hz: tight enough that the
    gripper feels rigidly attached to the controller, loose enough that the arm eases through
    singularities instead of snapping.
    """

    position: float = 100.0
    """Weight on TCP translation error. Dominant term: position is what a human notices first."""

    orientation: float = 20.0
    """Weight on TCP rotation error. Deliberately well below `position` -- when a target is only
    reachable by trading the two off, giving up wrist angle feels far better than giving up
    where the gripper is."""

    joint_limit: float = 200.0
    """Weight on joint limit violation. Soft rather than a hard constraint: a hard constraint
    makes jaxls run augmented Lagrangian, which roughly triples solve time. The solution is
    clipped to the limits afterwards regardless."""

    damping: float = 10.0
    """Weight pulling toward the *previous* configuration. This is the differential IK damping
    term, and the main jitter/feel knob. Raise it for a calmer arm that trails the hand; lower
    it for a twitchier arm that chases jitter. Past ~25 it starts costing visible lag."""

    posture: float = 0.4
    """Weight pulling toward `REST_POSE`. Weak; it only breaks ties, keeping the elbow in a sane
    place without visibly dragging the arm."""

    manipulability: float = 0.0
    """Weight penalizing 1/manipulability, i.e. pushing away from singular configurations. Off:
    YAM has exactly 6 joints, so there is no null space to retreat through and the term can only
    buy conditioning by giving up pose accuracy. The `damping` term already provides the
    singularity robustness -- it is what makes this a *damped* least-squares solve."""


@jdc.pytree_dataclass
class IKSolution:
    cfg: jax.Array
    """Joint configuration, in pyroki's own joint order."""

    pos_error: jax.Array
    """TCP position error at the solution, in meters."""

    ori_error: jax.Array
    """TCP orientation error at the solution, in radians."""


class YamIK:
    """Warm-started TCP-pose IK for the YAM arm, in ``joint1..joint6`` order."""

    def __init__(self, weights: IKWeights | None = None, max_iterations: int = 12) -> None:
        # The URDF sits beside the MJCF in the same version dir. Meshes are only needed for
        # collision geometry, which we don't use.
        urdf_path = os.path.splitext(ArmType.YAM.get_xml_path())[0] + ".urdf"
        self._urdf = yourdfpy.URDF.load(urdf_path, load_meshes=False)
        self.robot = pk.Robot.from_urdf(self._urdf)
        self.weights = weights if weights is not None else IKWeights()

        # jaxls needs a static iteration cap to unroll against. 12 is generous for a
        # warm-started solve (it typically converges in 2-4) while bounding worst-case latency
        # after a clutch re-anchor, which starts far from the solution.
        self._max_iterations = max_iterations

        self._tip_index = self.robot.links.names.index(TIP_LINK)
        names = tuple(self.robot.joints.actuated_names)
        self._arm = np.array([names.index(joint) for joint in ARM_JOINTS])
        self._rest = np.zeros(len(names))
        self._rest[self._arm] = REST_POSE
        # Zero for the fingers, which is what freezes them: the clamp below is what they move by.
        self._speed = np.zeros(len(names))
        self._speed[self._arm] = VELOCITY_LIMITS
        self._tip_from_tcp = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(jnp.asarray(TCP_WXYZ)), jnp.asarray(TCP_OFFSET)
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Actuated joint names in pyroki's order -- reversed, and with the fingers. Everything
        this class takes and returns is in ``ARM_JOINTS`` order instead."""
        return tuple(self.robot.joints.actuated_names)

    @cached_property
    def _limits(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(self.robot.joints.lower_limits),
            np.asarray(self.robot.joints.upper_limits),
        )

    def _full(self, q: np.ndarray) -> np.ndarray:
        """Scatter 6 arm joints into a full pyroki configuration, fingers at rest."""
        cfg = self._rest.copy()
        cfg[self._arm] = q
        return cfg

    def tcp_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics to the TCP. Returns (position, wxyz quaternion)."""
        tip = jaxlie.SE3(self.robot.forward_kinematics(jnp.asarray(self._full(q)))[self._tip_index])
        tcp = tip @ self._tip_from_tcp
        return np.asarray(tcp.translation()), np.asarray(tcp.rotation().wxyz)

    def solve(
        self,
        target_position: np.ndarray,
        target_wxyz: np.ndarray,
        q_prev: np.ndarray,
        dt: float = 1.0 / 60.0,
    ) -> tuple[np.ndarray, float, float]:
        """Solve for the arm configuration putting the TCP at the target pose.

        Args:
            target_position: Desired TCP position in the arm's base frame, (3,).
            target_wxyz: Desired TCP orientation as a wxyz quaternion, (4,).
            q_prev: Previous arm configuration, used as both the warm start and the damping
                anchor. Pass ``REST_POSE`` on the very first call.
            dt: Timestep used for the joint speed clamp.

        Returns:
            (q, position_error_m, orientation_error_rad), q in ``ARM_JOINTS`` order.
        """
        cfg_prev = self._full(q_prev)
        sol = self._solve_jit(
            self.robot,
            self.weights,
            jnp.asarray(target_position, dtype=jnp.float32),
            jnp.asarray(target_wxyz, dtype=jnp.float32),
            jnp.asarray(cfg_prev, dtype=jnp.float32),
            jnp.asarray(self._rest, dtype=jnp.float32),
            jnp.asarray(self._tip_index),
            self._tip_from_tcp,
            self._max_iterations,
        )
        cfg = np.asarray(sol.cfg, dtype=np.float64)

        # Two safety nets the solver treats as soft costs. Clamping here, outside the
        # optimization, means a violated limit degrades into a slower or slightly off-target
        # motion rather than into a jump.
        lower, upper = self._limits
        cfg = np.clip(cfg, lower, upper)
        max_step = self._speed * dt
        cfg = cfg_prev + np.clip(cfg - cfg_prev, -max_step, max_step)

        return cfg[self._arm], float(sol.pos_error), float(sol.ori_error)

    @staticmethod
    @jdc.jit
    def _solve_jit(
        robot: pk.Robot,
        weights: IKWeights,
        target_position: jax.Array,
        target_wxyz: jax.Array,
        cfg_prev: jax.Array,
        rest: jax.Array,
        tip_index: jax.Array,
        tip_from_tcp: jaxlie.SE3,
        max_iterations: jdc.Static[int],
    ) -> IKSolution:
        joint_var = robot.joint_var_cls(0)

        target_tcp = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3(target_wxyz), target_position)
        target_tip = target_tcp @ jaxlie.SE3.inverse(tip_from_tcp)

        costs = [
            # Analytic Jacobian: ~2x faster than the numerical one, which matters when this
            # runs every frame.
            pk.costs.pose_cost_analytic_jac(
                robot,
                joint_var,
                target_tip,
                tip_index,
                pos_weight=weights.position,
                ori_weight=weights.orientation,
            ),
            pk.costs.limit_cost(robot, joint_var, weights.joint_limit),
            pk.costs.rest_cost(joint_var, cfg_prev, weights.damping),
            pk.costs.rest_cost(joint_var, rest, weights.posture),
            pk.costs.manipulability_cost(robot, joint_var, tip_index, weights.manipulability),
        ]

        solution = (
            jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
            .analyze()
            .solve(
                initial_vals=jaxls.VarValues.make([joint_var.with_value(cfg_prev)]),
                verbose=False,
                # A handful of variables: a dense factorization beats anything iterative.
                linear_solver="dense_cholesky",
                trust_region=jaxls.TrustRegionConfig(lambda_initial=0.01),
                termination=jaxls.TerminationConfig(max_iterations=max_iterations),
            )
        )
        cfg = solution[joint_var]

        achieved_tip = jaxlie.SE3(robot.forward_kinematics(cfg)[tip_index])
        achieved_tcp = achieved_tip @ tip_from_tcp
        error = jaxlie.SE3.log(jaxlie.SE3.inverse(achieved_tcp) @ target_tcp)

        return IKSolution(cfg=cfg, pos_error=jnp.linalg.norm(error[:3]), ori_error=jnp.linalg.norm(error[3:]))
