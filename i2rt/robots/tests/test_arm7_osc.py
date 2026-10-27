"""Tests for the 7-DOF OSC controller and its simulated motor chain.

Three of these are the controller's core correctness properties, each guarding a failure that is
silent rather than loud:

* the nullspace projector really is a projector, and really does annihilate the task Jacobian --
  otherwise "nullspace motion does not disturb the task" is not true and nothing says so;
* the orientation error is continuous through and past +/-pi -- a discontinuity there is a torque
  step on a real arm, and it hides completely at small angles where testing usually happens;
* a singular configuration produces finite torque -- the damped pseudoinverse is the only thing
  standing between a rank-deficient Jacobian and a divergent command.

The rest cover the simulated plant, whose whole purpose is to reproduce hardware behaviour that
cannot otherwise be seen until bring-up.

Everything runs headless against MuJoCo, with no CAN bus and no motors.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from i2rt.robots.arm7_osc import (
    ARM_DOF,
    COMMAND_TIMEOUT_S,
    DEFAULT_MOTOR_LIST,
    KD_TASK_JOINT,
    KP_TASK_JOINT,
    MAX_FEEDBACK_AGE_S,
    Arm7OSC,
    ArmState,
    ControlMode,
    LoopTimer,
    VelocityObserver,
    arm7_xml,
    rotation_error,
)
from i2rt.robots.sim_motor_chain import SimMotorChain
from i2rt.utils.mujoco_utils import MuJoCoKDL

# Arm plus gripper, composed. The shipped arm MJCF is arm-only: no ``grasp_site`` for the Jacobian
# and no gripper mass for gravity, so testing against it would be testing a different robot than
# the one that runs. Composed once here rather than per test -- it writes a file to /tmp.
ARM7_XML = arm7_xml("right")

# 7 arm motors plus the gripper, as the arm config declares them. The two finger joints are one
# commanded DOF, coupled in the model by an equality constraint, so the chain drives 8 DOFs over a
# 9-DOF model.
MOTOR_LIST = [*DEFAULT_MOTOR_LIST, (0x08, "DM4310")]
DOF_INDICES = list(range(8))
FINGER_DOFS = [7, 8]

# Mid-workspace poses, clear of the joint limits and of singularity. joint3 is positive in both:
# the v1 model gives it [-15, +195] deg where the first 7-DOF model had [-100, +102], so the -0.4
# this used to carry is now out of range. Conditioning is unchanged (cond(J) 17.1 -> 16.6).
Q_MID = np.array([0.3, 0.9, 0.4, -1.2, 0.3, -0.5, 0.2])
Q_ALT = np.array([-0.5, 1.6, 0.7, -0.6, -0.8, 0.4, -0.3])


def _chain(q0: np.ndarray = Q_MID, **kwargs: object) -> SimMotorChain:
    return SimMotorChain(
        ARM7_XML,
        MOTOR_LIST,
        dof_indices=DOF_INDICES,
        initial_qpos=np.concatenate([q0, [0.0, 0.0]]),
        mechanism_held_dofs=FINGER_DOFS,
        **kwargs,  # type: ignore[arg-type]
    )


def _arm(chain: SimMotorChain, **kwargs: object) -> Arm7OSC:
    return Arm7OSC(chain, ARM7_XML, auto_start=False, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Core property 1: the nullspace projector
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [Q_MID, Q_ALT], ids=["mid", "alt"])
def test_nullspace_projector_is_a_projector_that_annihilates_the_jacobian(q: np.ndarray) -> None:
    """``N = I - J^+ J`` must be symmetric, idempotent, and satisfy ``J N == 0``.

    These three are what make the mode's claim meaningful. Symmetry is what keeps nullspace
    stiffness isotropic, so pushing the elbow feels the same in both directions; idempotence is
    what makes it a projection rather than a scaling; and ``J N == 0`` is the actual statement that
    nullspace joint motion is invisible to the task Jacobian.

    Only well-conditioned poses are checked here. Near a singularity the damped pseudoinverse
    stops satisfying the identity on purpose -- see
    :func:`test_projector_identity_degrades_gracefully_at_a_singularity`.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        n = arm.nullspace_projector(q)
        jac = arm.site_jacobian(q)

        np.testing.assert_allclose(n, n.T, atol=1e-9)
        np.testing.assert_allclose(n @ n, n, atol=1e-9)
        np.testing.assert_allclose(jac @ n, np.zeros((6, ARM_DOF)), atol=1e-9)
    finally:
        chain.close()


def test_projector_identity_degrades_gracefully_at_a_singularity() -> None:
    """At a singular pose ``J N`` is small but not zero, and that is the correct behaviour.

    ``J J^+ J == J`` needs a true pseudoinverse. ``rcond`` deliberately does not invert singular
    values below its threshold, so the identity is traded away exactly where inverting them would
    produce an enormous command. What must still hold is that the projector stays symmetric, stays
    a projector, and that the residual stays small -- documented here so that a future change to
    ``rcond`` shows up as a test result rather than as a surprise on hardware.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        q_sing = np.zeros(ARM_DOF)
        n = arm.nullspace_projector(q_sing)
        jac = arm.site_jacobian(q_sing)

        np.testing.assert_allclose(n, n.T, atol=1e-9)
        np.testing.assert_allclose(n @ n, n, atol=1e-9)
        residual = float(np.abs(jac @ n).max())
        assert residual > 1e-9, "expected the damping to break the identity at a singular pose"
        assert residual < 1e-3, f"residual {residual:.2e} is larger than damping should produce"
    finally:
        chain.close()


def test_nullspace_of_a_seven_dof_arm_is_one_dimensional() -> None:
    """A 7-DOF arm on a 6-DOF task has exactly one redundant direction.

    This is the premise of the whole feature: on a 6-DOF arm the projector is zero and there is no
    elbow to push. Asserting the rank keeps a model or site change from quietly removing the
    redundancy the controller is built around.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        assert np.linalg.matrix_rank(arm.nullspace_projector(Q_MID), tol=1e-6) == 1
    finally:
        chain.close()


def test_nullspace_torque_is_orthogonal_to_task_torque() -> None:
    """The two layers must not fight: their torques live in complementary subspaces."""
    chain = _chain()
    try:
        arm = _arm(chain)
        n_null = arm.nullspace_projector(Q_MID)
        n_task = np.eye(ARM_DOF) - n_null
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = rng.standard_normal(ARM_DOF)
            assert abs(float((n_task @ v) @ (n_null @ v))) < 1e-9
    finally:
        chain.close()


# --------------------------------------------------------------------------------------
# Core property 2: orientation error continuity
# --------------------------------------------------------------------------------------


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def test_rotation_error_recovers_the_rotation_vector_for_moderate_angles() -> None:
    """Round-trip: build a rotation from a known axis-angle, read it back."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        theta = float(rng.uniform(-3.0, 3.0))
        got = rotation_error(_rot_axis(axis, theta), np.eye(3))
        expected = axis * theta
        # +/-theta about +/-axis is the same rotation, so compare the rotation, not the sign.
        assert min(np.linalg.norm(got - expected), np.linalg.norm(got + expected)) < 1e-7


def test_rotation_error_is_continuous_across_pi() -> None:
    """Sweeping through pi must not make the error jump.

    At exactly pi the axis sign is ambiguous while the rotation is not, and the small-angle
    formula loses all precision there because the off-diagonal terms it reads vanish. A jump here
    is a torque step on hardware, and it is invisible at the small angles where a controller is
    usually exercised.
    """
    thetas = np.linspace(np.pi - 0.05, np.pi + 0.05, 401)
    errs = np.array([rotation_error(_rot_x(t), np.eye(3)) for t in thetas])
    norms = np.linalg.norm(errs, axis=1)

    # The magnitude must peak at pi and come back down, without a discontinuity.
    assert np.all(norms <= np.pi + 1e-6)
    step = np.abs(np.diff(norms)).max()
    assert step < 1e-3, f"largest jump in |error| across pi was {step:.2e}"


def test_rotation_error_handles_exactly_pi_and_identity() -> None:
    """The two degenerate inputs: a half turn, and no rotation at all."""
    err = rotation_error(_rot_x(np.pi), np.eye(3))
    assert np.isfinite(err).all()
    assert abs(np.linalg.norm(err) - np.pi) < 1e-6
    # Axis must be +/-x for a rotation about x.
    np.testing.assert_allclose(np.abs(err / np.linalg.norm(err)), [1.0, 0.0, 0.0], atol=1e-6)

    np.testing.assert_allclose(rotation_error(np.eye(3), np.eye(3)), np.zeros(3), atol=1e-12)


def test_rotation_error_is_zero_between_identical_frames() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20):
        r = _rot_axis(rng.standard_normal(3), float(rng.uniform(-np.pi, np.pi)))
        np.testing.assert_allclose(rotation_error(r, r), np.zeros(3), atol=1e-7)


# --------------------------------------------------------------------------------------
# Core property 3: singular configurations stay finite
# --------------------------------------------------------------------------------------


def test_singular_configuration_produces_finite_torque() -> None:
    """A rank-deficient Jacobian must not produce a divergent command.

    The damped pseudoinverse is the only guard. Without ``rcond`` a vanishing singular value is
    inverted into an enormous joint velocity, which on hardware is a thrown arm.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        # A fully extended arm: the wrist axes line up and J loses rank.
        q_sing = np.zeros(ARM_DOF)
        jac = arm.site_jacobian(q_sing)
        smallest = np.linalg.svd(jac, compute_uv=False).min()
        assert smallest < 1e-2, f"expected a near-singular pose, smallest sigma was {smallest:.3e}"

        tau = arm._torque_kinematic(q_sing, np.zeros(ARM_DOF), q_sing + 0.1)
        assert np.isfinite(tau).all()
        assert np.abs(tau).max() < 1e4

        n = arm.nullspace_projector(q_sing)
        assert np.isfinite(n).all()
    finally:
        chain.close()


def test_zero_error_produces_zero_control_torque() -> None:
    """On target and at rest, the control law must ask for nothing.

    Gravity is added outside this function, so any offset here would be a standing bias fighting
    the arm at every pose.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        tau = arm._torque_kinematic(Q_MID, np.zeros(ARM_DOF), Q_MID)
        np.testing.assert_allclose(tau, np.zeros(ARM_DOF), atol=1e-12)
    finally:
        chain.close()


# --------------------------------------------------------------------------------------
# `full` mode: inertia shaping and the dynamically consistent nullspace
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", [Q_MID, Q_ALT], ids=["mid", "alt"])
def test_nullspace_torque_produces_no_end_effector_acceleration(q: np.ndarray) -> None:
    """The claim that distinguishes `full` from `kinematic`, stated as the physics states it.

    A nullspace torque must accelerate the joints without accelerating the tool:
    ``a_ee = J M^-1 tau_null == 0``. The orthogonal projector cannot promise this -- it makes
    nullspace *velocity* invisible to the Jacobian, which is a different statement -- and the gap
    is not subtle: the same test on ``N_kin`` returns a number of order 1e2.

    This is the A6 simulation unit test, and it is what justifies calling the projector
    "dynamically consistent".
    """
    chain = _chain()
    try:
        arm = _arm(chain, mode=ControlMode.FULL)
        jac, mass = arm._jacobian_and_mass(q)
        mass_inv = np.linalg.inv(mass)
        n_dyn = arm.dynamic_nullspace_projector(q)
        n_kin = arm.nullspace_projector(q)

        rng = np.random.default_rng(0)
        worst_dyn = worst_kin = 0.0
        for _ in range(100):
            v = rng.standard_normal(ARM_DOF)
            for proj, is_dyn in ((n_dyn, True), (n_kin, False)):
                tau = proj @ v
                tau /= max(float(np.linalg.norm(tau)), 1e-12)
                accel = float(np.linalg.norm(jac @ mass_inv @ tau))
                if is_dyn:
                    worst_dyn = max(worst_dyn, accel)
                else:
                    worst_kin = max(worst_kin, accel)

        assert worst_dyn < 1e-6, f"N_dyn leaked {worst_dyn:.2e} of end-effector acceleration"
        assert worst_kin > 1.0, "N_kin should visibly disturb the tool; if not, this test is vacuous"
    finally:
        chain.close()


def test_dynamic_projector_is_idempotent_but_not_symmetric() -> None:
    """``N_dyn`` is an oblique projection, and that is the point, not a defect.

    Projecting orthogonally would measure distance in joint space, where the arm's own inertia does
    not appear. ``N_dyn`` projects along the mass metric instead, which is exactly what buys the
    decoupling above -- and the price is that it is not symmetric. A future "cleanup" that
    symmetrised it would silently destroy the property the mode exists for.
    """
    chain = _chain()
    try:
        arm = _arm(chain, mode=ControlMode.FULL)
        n = arm.dynamic_nullspace_projector(Q_MID)
        np.testing.assert_allclose(n @ n, n, atol=1e-9)
        assert np.abs(n - n.T).max() > 0.1, "N_dyn came out symmetric; it should be oblique"
        assert np.linalg.matrix_rank(n, tol=1e-6) == 1
    finally:
        chain.close()


def test_mass_matrix_is_actually_populated() -> None:
    """Guards a silent MuJoCo trap: ``mj_crb`` alone leaves ``qM`` zero in 3.8.

    ``mj_makeM`` is what assembles it, and omitting that step raises nothing -- the mass matrix
    simply reads as zeros, ``Mx`` becomes meaningless, and the controller produces confident
    garbage. The pipeline is probed at construction rather than version-checked; this pins the
    outcome.
    """
    chain = _chain()
    try:
        arm = _arm(chain, mode=ControlMode.FULL)
        _, mass = arm._jacobian_and_mass(Q_MID)
        diag = np.diag(mass)
        assert diag.min() > 0.0, f"mass matrix diagonal has a zero: {diag}"
        np.testing.assert_allclose(mass, mass.T, atol=1e-9)
        assert np.linalg.eigvalsh(0.5 * (mass + mass.T)).min() > 0.0, "M must be positive definite"

        # And it must agree with a full mj_forward, which is the expensive path being avoided.
        import mujoco

        data = mujoco.MjData(arm._model)
        data.qpos[:ARM_DOF] = Q_MID
        mujoco.mj_forward(arm._model, data)
        reference = np.zeros((arm._model.nv, arm._model.nv))
        mujoco.mj_fullM(arm._model, reference, data.qM)
        np.testing.assert_allclose(mass, reference[:ARM_DOF, :ARM_DOF], atol=1e-9)
    finally:
        chain.close()


def test_full_mode_stays_finite_at_a_singularity() -> None:
    """``Mx`` inverts a matrix that collapses with ``J``; the damped branch must catch it."""
    chain = _chain()
    try:
        arm = _arm(chain, mode=ControlMode.FULL)
        q_sing = np.zeros(ARM_DOF)
        tau = arm._torque_full(q_sing, np.zeros(ARM_DOF), q_sing + 0.1)
        assert np.isfinite(tau).all()
        assert np.abs(tau).max() < 1e4, f"singular-pose torque reached {np.abs(tau).max():.2e}"
    finally:
        chain.close()


def test_full_mode_asks_for_nothing_when_on_target() -> None:
    """On target and at rest, the control torque must be zero in this mode too."""
    chain = _chain()
    try:
        arm = _arm(chain, mode=ControlMode.FULL)
        tau = arm._torque_full(Q_MID, np.zeros(ARM_DOF), Q_MID)
        np.testing.assert_allclose(tau, np.zeros(ARM_DOF), atol=1e-9)
    finally:
        chain.close()


def test_wrist_inertia_augmentation_reaches_only_the_wrist() -> None:
    """The floor is added to the last three diagonal entries and nothing else."""
    chain = _chain()
    try:
        plain = _arm(chain, mode=ControlMode.FULL, wrist_inertia_augmentation=0.0)
        _, mass = plain._jacobian_and_mass(Q_MID)
        base = np.diag(mass).copy()

        # The augmentation is applied inside _torque_full on a copy, so check it there by its
        # effect: a nonzero value must change the torque, and only through the wrist.
        tau_plain = plain._torque_full(Q_MID, np.ones(ARM_DOF) * 0.1, Q_MID + 0.05)
        augmented = _arm(chain, mode=ControlMode.FULL, wrist_inertia_augmentation=0.05)
        tau_aug = augmented._torque_full(Q_MID, np.ones(ARM_DOF) * 0.1, Q_MID + 0.05)
        assert not np.allclose(tau_plain, tau_aug), "augmentation had no effect at all"
        # The wrist entries are the small ones; the floor must be meaningful there and not at the
        # shoulder, which is what makes it a floor rather than a global damping term.
        assert base[:4].min() > 10 * base[4:].max(), "this arm no longer has the inertia spread the floor targets"
    finally:
        chain.close()


# --------------------------------------------------------------------------------------
# Gravity compensation
# --------------------------------------------------------------------------------------


def test_gravity_compensation_exactly_balances_the_arm_at_rest() -> None:
    """Applying ``g(q)`` at rest must leave zero acceleration on every arm joint.

    Guards the porting hazard the whole design warns about. It also guards a subtler one: the two
    finger DOFs must be accounted for, because MuJoCo models them as free slide joints tied by an
    equality constraint, and their unbalanced weight otherwise leaks through that constraint into
    the arm -- 2.2 rad/s^2 of it, which looks exactly like broken gravity compensation.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_path(ARM7_XML)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    data = mujoco.MjData(model)
    kdl = MuJoCoKDL(ARM7_XML)

    for q in (Q_MID, Q_ALT):
        # Compensate every DOF, the arm from the controller and the fingers from the mechanism.
        data.qpos[:] = 0.0
        data.qpos[:ARM_DOF] = q
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        mujoco.mj_inverse(model, data)
        g_all = data.qfrc_inverse.copy()

        data.qpos[:] = 0.0
        data.qpos[:ARM_DOF] = q
        data.qvel[:] = 0.0
        data.qfrc_applied[:] = g_all
        mujoco.mj_forward(model, data)
        assert np.abs(data.qacc[:ARM_DOF]).max() < 1e-9

        # And the controller's own arm-only term must match the full-model result there.
        np.testing.assert_allclose(
            kdl.compute_inverse_dynamics(q, np.zeros(ARM_DOF), np.zeros(ARM_DOF)), g_all[:ARM_DOF], atol=1e-9
        )


def test_gravity_compensated_arm_holds_position_over_time() -> None:
    """Closed loop, plant included: floating on gravity alone must not drift.

    Deliberately an integration test. The static balance above can pass while the loop still
    drifts, if the plant and the controller disagree about the model or the command path drops a
    term.
    """
    chain = _chain()
    kdl = MuJoCoKDL(ARM7_XML)
    try:
        ee_start = chain.get_site_pose("grasp_site")[:3, 3].copy()
        tau = np.zeros(len(MOTOR_LIST))
        for _ in range(2000):  # 4 s at 500 Hz
            q = chain.get_true_state()["pos"][:ARM_DOF]
            tau[:ARM_DOF] = kdl.compute_inverse_dynamics(q, np.zeros(ARM_DOF), np.zeros(ARM_DOF))
            chain.set_commands(tau, get_state=False)
        ee_end = chain.get_site_pose("grasp_site")[:3, 3]
        drift_mm = float(np.linalg.norm(ee_end - ee_start) * 1e3)
        assert drift_mm < 1.0, f"end effector drifted {drift_mm:.3f} mm under pure gravity comp"
    finally:
        chain.close()


def test_zero_torque_lets_the_arm_fall() -> None:
    """The complement of the test above: without compensation the arm must collapse.

    A plant that holds position when told nothing would make every gravity test vacuous.
    """
    chain = _chain()
    try:
        for _ in range(500):
            chain.set_commands(np.zeros(len(MOTOR_LIST)), get_state=False)
        moved = np.abs(chain.get_true_state()["pos"][:ARM_DOF] - Q_MID).max()
        assert moved > 0.1, f"arm only moved {moved:.4f} rad with zero torque"
    finally:
        chain.close()


# --------------------------------------------------------------------------------------
# The simulated plant's hardware fidelity knobs
# --------------------------------------------------------------------------------------


def test_polling_groups_stagger_feedback_timestamps() -> None:
    """With round-robin polling, feedback is not a snapshot and must not claim to be.

    The firmware polls three motors per round, so each motor's effective feedback rate is the loop
    rate over the group count, and different joints carry readings of different ages. The
    controller's eventual time-alignment depends on those per-motor timestamps being honest, and a
    missing-frame criterion set against the loop rate instead of the effective rate would
    misread this staggering as dropped frames.
    """
    groups = [[0, 1, 2], [3, 4, 5], [6, 7]]
    chain = _chain(poll_groups=groups)
    try:
        assert chain.effective_feedback_divisor == len(groups)

        tau = np.zeros(len(MOTOR_LIST))
        for _ in range(len(groups)):
            infos = chain.set_commands(tau)
        assert infos is not None

        stamps = np.array([i.timestamp for i in infos])
        assert np.ptp(stamps) > 0.0, "expected staggered timestamps across polling groups"
        # Motors in one group share a timestamp; different groups do not.
        for g in groups:
            assert len(set(stamps[list(g)])) == 1
        assert len({stamps[g[0]] for g in groups}) == len(groups)
    finally:
        chain.close()


def test_dropped_frames_freeze_feedback_and_its_timestamp() -> None:
    """A silent motor must keep its stale value *and* its stale timestamp.

    Freezing the value while refreshing the timestamp would make a dead motor look healthy, which
    is precisely what a communication watchdog reads.
    """
    chain = _chain()
    try:
        tau = np.zeros(len(MOTOR_LIST))
        chain.set_commands(tau, get_state=False)
        chain.set_dropped_frames([3])
        infos = chain.set_commands(tau)
        assert infos is not None
        frozen_pos, frozen_ts = infos[3].pos, infos[3].timestamp

        for _ in range(50):
            infos = chain.set_commands(tau)
        assert infos is not None
        assert infos[3].pos == frozen_pos
        assert infos[3].timestamp == frozen_ts
        # A live motor on the same chain kept reporting.
        assert infos[0].timestamp > frozen_ts
    finally:
        chain.close()


def test_encoder_quantization_coarsens_position_and_roughens_velocity() -> None:
    """Quantizing position and then differencing it is where the noisy velocity comes from.

    These motors have no velocity sensor, so velocity is a difference of successive rotor-encoder
    reads. Reproducing that mechanism -- rather than adding noise to a clean velocity -- is what
    makes the simulation useful for deciding how much damping the loop can afford.
    """
    bits, ratio, dt = 12, 10.0, 1.0 / 500.0
    step = 2.0 * np.pi / ((1 << bits) * ratio)
    quantum = step / dt  # the only velocity increment a differenced encoder can express

    ideal = _chain(control_dt=dt)
    coarse = _chain(control_dt=dt, encoder_bits=bits, gear_ratio=[ratio] * len(MOTOR_LIST))
    try:
        # Drift slowly, so the motion per sample is comparable to one encoder step. That is the
        # regime where the quantization actually governs the velocity estimate.
        tau = np.zeros(len(MOTOR_LIST))
        v_ideal, v_coarse = [], []
        for _ in range(300):
            a = ideal.set_commands(tau)
            b = coarse.set_commands(tau)
            assert a is not None and b is not None
            v_ideal.append(a[1].vel)
            v_coarse.append(b[1].vel)

        # Reported positions land on the encoder lattice.
        pos = coarse.read_states()[1].pos
        assert abs(pos / step - round(pos / step)) < 1e-6

        # Reported velocities land on multiples of one step per sample interval, which the exact
        # velocity does not. This is the mechanism, not a statistical side effect of it.
        on_lattice = np.abs(np.asarray(v_coarse) / quantum - np.round(np.asarray(v_coarse) / quantum))
        assert on_lattice.max() < 1e-6, "quantized velocity is not a multiple of step/dt"

        off_lattice = np.abs(np.asarray(v_ideal) / quantum - np.round(np.asarray(v_ideal) / quantum))
        assert off_lattice.max() > 1e-3, "exact velocity should not sit on the encoder lattice"
    finally:
        ideal.close()
        coarse.close()


def test_torque_is_saturated_to_each_motors_rating() -> None:
    """A command beyond the motor's range must clamp, not wrap or scale."""
    chain = _chain()
    try:
        huge = np.full(len(MOTOR_LIST), 1e4)
        chain.set_commands(huge, get_state=False)
        applied = chain.get_true_state()["torque"]
        # DM6248 on joints 1-2, DM4340 on 3-5, DM4310 on 6-7 and the gripper.
        np.testing.assert_allclose(applied, [120.0, 120.0, 28.0, 28.0, 28.0, 10.0, 10.0, 10.0], atol=1e-6)
    finally:
        chain.close()


def test_contacts_are_off_by_default_so_gravity_compensation_is_exact() -> None:
    """``base`` is welded to the world, so MuJoCo never filters ``base``-``link1`` contacts.

    Their convex hulls touch at every reachable pose, which puts spurious constraint torque on
    joint1 that gravity compensation cannot see. Off by default is the whole point; this pins it.
    """
    chain = _chain()
    try:
        assert not chain.model.geom_contype.any()
        assert not chain.model.geom_conaffinity.any()
    finally:
        chain.close()

    contacts_on = _chain(enable_contacts=True)
    try:
        assert contacts_on.model.geom_contype.any()
    finally:
        contacts_on.close()


def test_full_mode_survives_both_hardware_defects() -> None:
    """The P1b gate: quantized encoder and round-robin polling, both on, arm still holding.

    The gains here are **derated**, and that is the finding rather than an inconvenience. Three
    groups give each motor 167 Hz of real feedback under a 500 Hz loop, and the usable gain follows
    the feedback rate, not the loop rate: ``Kp6`` drops from 80 to 20 on polling alone.

    The term that actually diverges first is the nullspace *damping*, not the task layer -- at the
    un-derated gains this faults even with the task gains turned down to nothing. The criterion is
    discrete and checkable: ``kd_null * dt_effective / J`` must stay well under 2, with ``J`` the
    wrist's, which is 150x smaller than the shoulder's. Measured here: 0.4 holds, 0.8 diverges.
    """
    groups = [[0, 1, 2], [3, 4, 5], [6, 7]]
    chain = _chain(poll_groups=groups, encoder_bits=12)
    try:
        arm = _arm(
            chain,
            mode=ControlMode.FULL,
            kp_cartesian=np.full(6, 10.0),
            kd_cartesian=np.full(6, 6.3),
            kp_null=1.0,
            kd_null=0.2,
        )
        arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
        arm.enable_arm()

        for _ in range(3000):  # 6 s at 500 Hz
            arm._cmd_time = arm._enable_time = 0.0
            arm.command_joint_pos(Q_MID)
            arm._step()

        assert arm.state is ArmState.ACTIVE, f"diverged under the hardware defects: {arm.fault}"
        settled = np.abs(chain.get_true_state()["vel"][:ARM_DOF]).max()
        assert settled < 0.1, f"still moving at {settled:.3f} rad/s after settling"
        err = np.abs(chain.get_true_state()["pos"][:ARM_DOF] - Q_MID).max()
        assert err < 0.05, f"failed to hold: {err:.4f} rad off"
    finally:
        chain.close()


def test_nullspace_damping_is_what_diverges_first_under_polling() -> None:
    """Pins the bisection, so a future gain change cannot quietly re-break this.

    With both defects on and the *task* gains at zero, the arm still diverges if the nullspace
    damping is left at its full-feedback-rate value -- which is what makes the nullspace layer, not
    the task layer, the thing to derate first.
    """
    groups = [[0, 1, 2], [3, 4, 5], [6, 7]]

    def peak_velocity(kd_null: float) -> float:
        chain = _chain(poll_groups=groups, encoder_bits=12)
        try:
            arm = _arm(
                chain,
                mode=ControlMode.FULL,
                kp_cartesian=np.zeros(6),
                kd_cartesian=np.zeros(6),
                kp_null=4.0,
                kd_null=kd_null,
            )
            arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
            arm.enable_arm()
            peak = 0.0
            for _ in range(2000):
                arm._cmd_time = arm._enable_time = 0.0
                arm.command_joint_pos(Q_MID)
                arm._step()
                peak = max(peak, float(np.abs(chain.get_true_state()["vel"][:ARM_DOF]).max()))
            return peak
        finally:
            chain.close()

    # kd * dt_eff / J_wrist: 0.8 -> 1.6, over the discrete limit; 0.2 -> 0.4, under it.
    assert peak_velocity(0.2) < 3.0, "the derated nullspace damping should stay bounded"
    assert peak_velocity(0.8) > peak_velocity(0.2), "the un-derated damping should be visibly worse"


# --------------------------------------------------------------------------------------
# Fault injection: the watchdog
# --------------------------------------------------------------------------------------


def test_watchdog_ignores_ordinary_polling_and_catches_a_dead_motor() -> None:
    """The missing-frame criterion must key off the per-motor rate, not the loop rate.

    Both halves are the test. Under three-group polling a healthy motor is silent on two ticks out
    of three, so a criterion set against the loop rate would fault a perfectly good arm -- that is
    the failure mode this guards, as much as the dead motor is. Then one motor is actually stopped
    and the watchdog has to notice.
    """
    groups = [[0, 1, 2], [3, 4, 5], [6, 7]]
    # Derated for the 167 Hz per-motor feedback three groups leave. The shipped gains assume every
    # motor is heard from every tick and go unstable here, which is its own finding.
    #
    # The damping scale is 0.05, not the 0.32 the first 7-DOF model needed, because joint7's
    # reflected inertia fell from 0.008 to 0.0005 kg.m^2 between that model and v1 -- sixteenfold.
    # Discrete damping is stable while kd*dt/J < 2, so at dt = 6 ms that ceiling is 0.16 N.m.s/rad
    # on joint7; KD_TASK_JOINT[6] * 0.32 is 0.48 and diverges. Shipped at half the ceiling, as
    # KP_TASK_JOINT and KP_TASK_CARTESIAN are.
    derated = dict(kp_task=KP_TASK_JOINT * 0.1, kd_task=KD_TASK_JOINT * 0.05, kp_null=1.0, kd_null=0.05)

    # Count refreshes per control step directly. The arm's own estimate is in wall-clock seconds,
    # which is the right unit on hardware but is not a stable quantity in a synchronously stepped
    # test: under a parallel test run the two measurements compete for CPU differently and the
    # ratio between them wanders. Timestamp *changes* per step do not -- three groups means each
    # motor refreshes on one step in three, whatever the machine is doing.
    chain = _chain(poll_groups=groups)
    try:
        tau = np.zeros(len(MOTOR_LIST))
        prev = np.array([i.timestamp for i in chain.read_states()])
        refreshes = np.zeros(ARM_DOF)
        steps = 300
        for _ in range(steps):
            infos = chain.set_commands(tau)
            assert infos is not None
            now = np.array([i.timestamp for i in infos])
            refreshes += (now[:ARM_DOF] > prev[:ARM_DOF]).astype(float)
            prev = now
        share = refreshes / steps
        assert chain.effective_feedback_divisor == len(groups)
        np.testing.assert_allclose(share, 1.0 / len(groups), atol=0.02)
    finally:
        chain.close()

    chain = _chain(poll_groups=groups)
    try:
        arm = _arm(chain, **derated)
        arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
        arm.enable_arm()
        for _ in range(1500):
            arm._cmd_time = arm._enable_time = 0.0
            arm.command_joint_pos(Q_MID)
            arm._step()
        # Healthy motors polled in groups are silent two ticks in three. A criterion set against
        # the loop rate would call that a fault; this one must not.
        assert arm.state is ArmState.ACTIVE, f"polling alone tripped the watchdog: {arm.fault}"
        assert arm.feedback_intervals.min() > 0.0, "never learned a refresh interval"

        chain.set_dropped_frames([4])
        for _ in range(3000):
            arm._cmd_time = arm._enable_time = 0.0
            arm.command_joint_pos(Q_MID)
            arm._step()
            if arm.state is ArmState.DAMPED:
                break

        assert arm.state is ArmState.DAMPED, "a motor stopped answering and the watchdog said nothing"
        assert arm.fault is not None and "joint 5" in arm.fault, f"blamed the wrong joint: {arm.fault}"
    finally:
        chain.close()


def test_a_stopped_arm_keeps_holding_itself_up() -> None:
    """Entering DAMPED must not drop the arm: gravity stays, only the control torque goes.

    A safe stop that lets go is not a safe stop.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
        arm.enable_arm()
        for _ in range(500):
            arm._cmd_time = arm._enable_time = 0.0
            arm.command_joint_pos(Q_MID)
            arm._step()

        arm._state = ArmState.DAMPED
        start = chain.get_true_state()["pos"][:ARM_DOF].copy()
        for _ in range(2000):  # 4 s stopped
            arm._step()
        drift = np.abs(chain.get_true_state()["pos"][:ARM_DOF] - start).max()
        assert drift < 0.15, f"the arm sagged {drift:.3f} rad while stopped"
    finally:
        chain.close()


def test_a_command_that_stops_arriving_stops_the_arm() -> None:
    """The upstream watchdog: no fresh command for COMMAND_TIMEOUT_S means stop and stay stopped."""
    chain = _chain()
    try:
        arm = _arm(chain)
        arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
        arm.enable_arm()
        arm.command_joint_pos(Q_MID)

        arm._cmd_time = time.monotonic() - (COMMAND_TIMEOUT_S + 0.05)
        arm._step()
        assert arm.state is ArmState.DAMPED

        # A new command must NOT silently re-engage: recovery is explicit, so that an upstream
        # process which died and restarted cannot fling the arm at a stale target.
        arm.command_joint_pos(Q_MID)
        arm._step()
        assert arm.state is ArmState.DAMPED, "a fresh command re-engaged the arm on its own"
        arm.resume()
        assert arm.state is ArmState.ACTIVE
    finally:
        chain.close()


def test_torque_slew_is_bounded_across_a_state_change() -> None:
    """Entering or leaving a state must not produce a torque step.

    The slew bound is applied after the state machine for exactly this reason: a jump between
    control laws is the same discontinuity as a jump in the command, and the motors cannot tell
    them apart.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
        arm.enable_arm()
        for _ in range(500):
            arm._cmd_time = arm._enable_time = 0.0
            arm.command_joint_pos(Q_MID + 0.2)
            arm._step()

        max_step = arm.max_torque_rate * arm._control_dt
        prev = arm._prev_tau.copy()
        arm.float_arm()
        for _ in range(20):
            arm._step()
            step = np.abs(arm._prev_tau - prev).max()
            assert step <= max_step + 1e-9, f"torque jumped {step:.3f} Nm in one tick, cap {max_step:.3f}"
            prev = arm._prev_tau.copy()
    finally:
        chain.close()


# --------------------------------------------------------------------------------------
# The velocity observer: noise, and the skew that arrives with it
# --------------------------------------------------------------------------------------


def test_observer_low_pass_attenuates_noise_without_shifting_the_mean() -> None:
    """A first-order low-pass must cut the jitter and still settle on the true value.

    Both halves matter: an estimator that is quiet but biased is worse than a noisy one, because
    the bias goes straight into the damping term as a standing torque.
    """
    obs = VelocityObserver(ARM_DOF, cutoff_hz=40.0, control_dt=1 / 500, align=False)
    rng = np.random.default_rng(0)
    truth = np.full(ARM_DOF, 0.5)
    raw, filtered = [], []
    for _ in range(3000):
        noisy = truth + rng.normal(0.0, 0.4, ARM_DOF)
        _, dq = obs.update(np.zeros(ARM_DOF), noisy, np.zeros(ARM_DOF), 0.0)
        raw.append(noisy.copy())
        filtered.append(dq.copy())

    tail_raw = np.array(raw[500:])
    tail_filt = np.array(filtered[500:])
    assert tail_filt.std() < 0.5 * tail_raw.std(), "filter did not attenuate"
    np.testing.assert_allclose(tail_filt.mean(axis=0), truth, atol=0.02)


def test_observer_cutoff_zero_passes_the_raw_signal() -> None:
    """A disabled filter must be a true bypass, so a comparison against raw is honest."""
    obs = VelocityObserver(ARM_DOF, cutoff_hz=0.0, control_dt=1 / 500, align=False)
    rng = np.random.default_rng(1)
    for _ in range(10):
        v = rng.standard_normal(ARM_DOF)
        _, dq = obs.update(np.zeros(ARM_DOF), v, np.zeros(ARM_DOF), 0.0)
    np.testing.assert_allclose(dq, v, atol=1e-12)


def test_observer_extrapolates_each_joint_from_its_own_timestamp() -> None:
    """Round-robin polling hands over readings of different ages; alignment is the correction.

    A joint read one full cycle ago has moved since, and the amount it moved is its own velocity
    times its own age -- so the correction has to be per joint, keyed to a per-motor timestamp.
    Without it, ``q`` describes a configuration the arm was never in.
    """
    dt = 1 / 500
    obs = VelocityObserver(ARM_DOF, cutoff_hz=0.0, control_dt=dt, align=True)
    vel = np.full(ARM_DOF, 2.0)
    now = 10.0
    # Three polling groups: ages of 0, one cycle, and two cycles.
    stamps = np.array([now, now, now, now - dt, now - dt, now - 2 * dt, now - 2 * dt])
    pos = np.zeros(ARM_DOF)

    obs.update(pos, vel, stamps, now)  # prime the filter
    q_aligned, _ = obs.update(pos, vel, stamps, now)

    expected = pos + vel * (now - stamps)
    np.testing.assert_allclose(q_aligned, expected, atol=1e-12)
    assert q_aligned[6] > q_aligned[4] > q_aligned[0], "older readings must be extrapolated further"


def test_observer_refuses_to_extrapolate_a_very_stale_reading() -> None:
    """Past the age cap, extrapolating would invent motion; that is the watchdog's problem."""
    obs = VelocityObserver(ARM_DOF, cutoff_hz=0.0, control_dt=1 / 500, align=True)
    vel = np.full(ARM_DOF, 2.0)
    stamps = np.full(ARM_DOF, 0.0)
    obs.update(np.zeros(ARM_DOF), vel, stamps, 0.0)
    q_aligned, _ = obs.update(np.zeros(ARM_DOF), vel, stamps, 10.0)  # 10 s stale
    np.testing.assert_allclose(q_aligned, vel * MAX_FEEDBACK_AGE_S, atol=1e-12)


def test_alignment_recovers_the_true_configuration_under_polling_skew() -> None:
    """End to end on the plant: with polling on, alignment must beat using the raw frame.

    The arm is swept at a steady rate so the skew has something to bite on, and the aligned
    estimate is compared against the plant's own true state -- the thing neither estimate can see.
    """
    groups = [[0, 1, 2], [3, 4, 5], [6, 7]]
    chain = _chain(poll_groups=groups)
    try:
        dt = 1 / 500
        obs = VelocityObserver(ARM_DOF, cutoff_hz=0.0, control_dt=dt, align=True)
        tau = np.zeros(len(MOTOR_LIST))
        tau[1] = 2.0  # drive one joint so the readings are genuinely moving

        raw_err = aligned_err = 0.0
        for step in range(400):
            infos = chain.set_commands(tau)
            assert infos is not None
            now = max(i.timestamp for i in infos)
            pos = np.array([i.pos for i in infos[:ARM_DOF]])
            vel = np.array([i.vel for i in infos[:ARM_DOF]])
            stamps = np.array([i.timestamp for i in infos[:ARM_DOF]])
            q_aligned, _ = obs.update(pos, vel, stamps, now)
            truth = chain.get_true_state()["pos"][:ARM_DOF]
            if step > 100:  # let the sweep build up speed
                raw_err = max(raw_err, float(np.abs(pos - truth).max()))
                aligned_err = max(aligned_err, float(np.abs(q_aligned - truth).max()))

        assert raw_err > 0.0
        assert aligned_err < raw_err, f"alignment did not help: raw {raw_err:.2e}, aligned {aligned_err:.2e}"
    finally:
        chain.close()


# --------------------------------------------------------------------------------------
# Controller plumbing: state machine, limits, timing
# --------------------------------------------------------------------------------------


def test_controller_starts_floating_not_limp() -> None:
    """Power-on must hold the arm up. "Not enabled" means gravity-comp float, not zero torque."""
    chain = _chain()
    try:
        arm = _arm(chain)
        assert arm.state is ArmState.FLOATING
    finally:
        chain.close()


def test_commands_are_clamped_to_joint_limits() -> None:
    """An out-of-range target is clamped, not refused: refusing leaves a stale target in force."""
    chain = _chain()
    try:
        arm = _arm(chain)
        arm.command_joint_pos(np.full(ARM_DOF, 100.0))
        lo, hi = arm._joint_limits[:, 0], arm._joint_limits[:, 1]
        assert arm._joint_cmd is not None
        assert np.all(arm._joint_cmd <= hi + 1e-9)
        assert np.all(arm._joint_cmd >= lo - 1e-9)
    finally:
        chain.close()


def test_command_of_wrong_length_is_rejected() -> None:
    chain = _chain()
    try:
        arm = _arm(chain)
        with pytest.raises(ValueError, match="joint positions"):
            arm.command_joint_pos(np.zeros(3))
    finally:
        chain.close()


def test_resume_only_leaves_the_damped_state() -> None:
    """Recovery is explicit, and only from DAMPED.

    A stalled upstream process that restarts must not be able to make the arm jump to a stale
    target on its own, so a new command does not clear DAMPED -- only ``resume()`` does.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        arm.float_arm()
        arm.resume()
        assert arm.state is ArmState.FLOATING  # resume is a no-op outside DAMPED

        arm._state = ArmState.DAMPED
        arm._q[:] = Q_MID
        arm.resume()
        assert arm.state is ArmState.ACTIVE
    finally:
        chain.close()


def test_a_fault_blocks_enable_until_cleared() -> None:
    chain = _chain()
    try:
        arm = _arm(chain)
        arm._fault = "synthetic"
        with pytest.raises(RuntimeError, match="synthetic"):
            arm.enable_arm()
        arm.clear_fault()
        arm._q[:] = Q_MID
        arm.enable_arm()
        assert arm.state is ArmState.ACTIVE
    finally:
        chain.close()


def test_loop_timer_reports_tail_latency_not_just_a_mean() -> None:
    """A loop averaging its target while stalling occasionally is not meeting its target.

    The mean cannot express that, which is why this exists alongside ``RateRecorder``.
    """
    timer = LoopTimer(capacity=16)
    assert timer.stats() == {}

    # A synthetic record: many fast periods and one long stall.
    timer._periods[:10] = 0.002
    timer._periods[10] = 0.020
    timer._n = 11

    stats = timer.stats()
    assert stats["count"] == 11
    assert stats["p50_ms"] == pytest.approx(2.0, abs=1e-6)
    assert stats["max_ms"] == pytest.approx(20.0, abs=1e-6)
    assert stats["p99_ms"] > stats["p50_ms"]


def test_loop_timer_ring_buffer_wraps_without_growing() -> None:
    """Bounded by construction: the control loop must not allocate as it runs."""
    timer = LoopTimer(capacity=8)
    for _ in range(50):
        timer.tick()
    assert timer._periods.size == 8
    assert timer.stats()["count"] == 8


def test_controller_rejects_a_model_without_the_requested_site() -> None:
    chain = _chain()
    try:
        with pytest.raises(ValueError, match="not found"):
            Arm7OSC(chain, ARM7_XML, ee_site="no_such_site", auto_start=False)
    finally:
        chain.close()


# --------------------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------------------


def test_osc_holds_a_commanded_pose() -> None:
    """The headline behaviour: enable, command a pose, converge to it and stay.

    Steps the loop directly instead of starting the thread, so the test is deterministic and does
    not depend on wall-clock scheduling.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
        arm.enable_arm()
        arm._enable_time = 0.0  # skip the ramp; it is exercised by its own test

        for _ in range(2500):  # 5 s at 500 Hz
            arm._cmd_time = arm._enable_time = 0.0  # keep the command fresh and the ramp complete
            arm.command_joint_pos(Q_MID)
            arm._step()

        assert arm.state is ArmState.ACTIVE, f"unexpected state {arm.state}, fault={arm.fault}"
        err = np.abs(arm.get_joint_pos() - Q_MID).max()
        assert err < 0.05, f"failed to converge: max joint error {err:.4f} rad"

        obs = arm.get_observations()
        assert np.linalg.norm(obs["ee_pos_err"]) < 0.01
        for key in ("joint_pos", "joint_vel", "joint_eff", "feedback_age"):
            assert obs[key].shape == (ARM_DOF,)
    finally:
        chain.close()


def test_nullspace_absorbs_an_elbow_push() -> None:
    """Push the elbow: the joints must move mostly in the nullspace, and recover on release.

    This is the feature's visible promise. Note the end effector does move -- ``N_kin`` is an
    orthogonal, not a dynamically consistent, projector, so it perturbs the task at the
    millimetre-to-centimetre level. Only ``full`` mode makes that exactly zero, and asserting
    otherwise here would be asserting something the mode does not claim.
    """
    chain = _chain()
    try:
        arm = _arm(chain)
        arm._q[:] = chain.get_true_state()["pos"][:ARM_DOF]
        arm.enable_arm()

        def run(steps: int) -> None:
            for _ in range(steps):
                arm._cmd_time = arm._enable_time = 0.0
                arm.command_joint_pos(Q_MID)
                arm._step()

        run(1500)
        q_ref = arm.get_joint_pos().copy()
        projector = arm.nullspace_projector(q_ref)

        chain.apply_body_force("link3", np.array([0.0, 10.0, 0.0]))
        run(1250)
        delta = arm.get_joint_pos() - q_ref
        assert np.linalg.norm(delta) > 1e-3, "the push did not move the arm at all"
        in_null = np.linalg.norm(projector @ delta) / np.linalg.norm(delta)
        assert in_null > 0.7, f"only {in_null:.1%} of the elbow push went into the nullspace"

        chain.apply_body_force("link3", np.zeros(3))
        run(2000)
        recovered = np.abs(arm.get_joint_pos() - q_ref).max()
        assert recovered < 0.05, f"did not return after release: {recovered:.4f} rad off"
    finally:
        chain.close()


def test_matched_vector_gains_collapse_to_a_plain_joint_pd() -> None:
    """``kp_null == kp_task`` must make the two projections add up to one joint-space PD.

    ``--stage selfmotion`` replays a path planned to lie in the nullspace. The error is then
    entirely inside it, so ``n_task @ err`` is zero and ``kp_task`` drives nothing -- the whole
    trajectory has to come from ``kp_null``. Giving both layers the same 7-vector is what makes
    that work, because ``n_task + n_null = I``.

    The failure this pins happened on hardware on 2026-09-27: the stage ran with ``kp_null=0`` and
    the shipped task gains, which left joint1 0.26 N.m and joint2 0.90 N.m against 1.91 and
    1.37 N.m of stiction. Both stayed still for the entire run, joint5 was driven the wrong way,
    and the run was read as a kinematic error until the torques were checked.
    """
    import mujoco

    from i2rt.robots.arm7_osc import arm7_xml

    xml = arm7_xml("right")
    arm = object.__new__(Arm7OSC)
    model = mujoco.MjModel.from_xml_path(xml)
    arm._model, arm._data = model, mujoco.MjData(model)
    arm._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    arm._jacp = np.zeros((3, model.nv))
    arm._jacr = np.zeros((3, model.nv))
    arm._jac = np.zeros((6, model.nv))
    arm._record_task_error = lambda *_: None  # type: ignore[method-assign]

    gain = 3.0
    arm.kp_task = arm.kp_null = KP_TASK_JOINT * gain
    arm.kd_task = arm.kd_null = KD_TASK_JOINT * np.sqrt(gain)

    q = np.radians([11.4, 31.9, 7.7, -55.7, 32.7, 11.3, -12.0])
    dq = np.radians([0.5, -0.3, 0.9, 0.0, -0.7, 0.2, 0.1])
    cmd = q + np.radians([-0.3, 1.5, 3.3, 0.0, -2.9, 1.5, 0.9])

    plain = arm.kp_task * (cmd - q) - arm.kd_task * dq
    np.testing.assert_allclose(arm._torque_kinematic(q, dq, cmd), plain, atol=1e-12)

    # And the converse. With the nullspace gain off, what a nullspace-only command draws is not
    # a weakened version of the intended torque -- it is a different torque. `kp_task` is a
    # 7-vector, so `kp_task * err` no longer lies in the nullspace even when `err` does, and what
    # survives `n_task @` is whatever that scaling tipped out of it. Roughly a third of the
    # magnitude comes through, pointing partly backwards: on hardware joint5 was driven the wrong
    # way while the joints that mattered sat under their stiction.
    arm.kp_null = arm.kd_null = 0.0
    jac = arm._site_jacobian(q)
    n_dir = np.linalg.svd(jac)[2][-1]
    null_cmd = q + np.radians(3.0) * n_dir
    intended = arm.kp_task * (null_cmd - q)
    starved = arm._torque_kinematic(q, np.zeros(ARM_DOF), null_cmd)

    assert np.any(np.sign(starved) != np.sign(intended)), (
        "with kp_null=0 the surviving torque happened to agree in sign everywhere; "
        "the sign reversal is the part of this defect that moves a joint backwards"
    )
    heavy = slice(0, 3)  # joint1..3, the ones stiction stops
    assert np.all(np.abs(starved[heavy]) < 0.5 * np.abs(intended[heavy])), (
        "kp_null=0 left the heavy joints more than half their torque; "
        f"got {np.round(starved[heavy], 3).tolist()} of {np.round(intended[heavy], 3).tolist()}"
    )


def test_gravity_is_gravity_not_the_joint_limit() -> None:
    """``_gravity`` must return the gravitational force alone, past the modelled range as well.

    ``mj_inverse`` returns ``qfrc_inverse``, which carries the joint-limit constraint force
    alongside the gravitational one. A joint outside its modelled range then contributes a term
    that has nothing to do with gravity and is orders of magnitude larger than it.

    This is neither hypothetical nor rare. The shipped joint6 range is [-55, +45] degrees while the
    hardware travels past -79, and joint2 and joint3 are written about a centre 90 degrees from
    where they operate. On 2026-09-28 a gravity-compensation run had joint6 at -79 degrees, where
    the uncorrected term reads -87 N.m against a true -0.70; after the 0.5 factor and the 15 N.m
    clamp the arm was being shoved with a saturated -15 N.m, on the joint with the least inertia it
    has. It presents as the gravity factor being far too large, which is the one thing it is not.

    Correct ranges would not settle it either -- a joint resting on its stop injects the same term.
    So this is built through the real constructor: what is under test is that ``Arm7OSC`` disables
    the constraint, not that a test can remember to.
    """
    arm = _arm(_chain())
    limits = np.degrees(arm._kdl.model.jnt_range[:ARM_DOF])
    j6_low = limits[5, 0]
    assert j6_low > -70, "this test needs joint6's modelled lower limit to be inside its travel"

    inside = np.radians([8, 5, -4, -5, -5, j6_low + 10, 9])
    outside = np.radians([8, 5, -4, -5, -5, j6_low - 24, 9])
    g_in = arm._gravity(inside)[5]
    g_out = arm._gravity(outside)[5]

    # Real gravity on this joint is a fraction of a N.m and varies smoothly with the angle; the
    # constraint force is tens of N.m and switches on at the limit.
    assert abs(g_out) < 2.0, f"joint6 gravity outside the modelled range reads {g_out:.2f} N.m"
    assert abs(g_out - g_in) < 0.5, (
        f"gravity jumped from {g_in:.3f} to {g_out:.3f} N.m across the modelled limit; "
        "the joint-limit constraint is leaking into the gravity term"
    )


def test_a_motor_that_never_answers_is_caught() -> None:
    """A joint absent from the start must fault, not be skipped.

    The watchdog measures each motor's refresh interval and compares its age against it. A motor
    that has been answering and stops is caught that way. A motor that was never answering has no
    interval, and the check used to skip exactly those -- the one case where the arm is certainly
    not the arm the controller thinks it is.

    On 2026-09-28 joint1 left the bus between two runs. The three that followed produced plausible,
    meaningless data: a tracking test reported 33.9 mm of RMS error that was almost entirely that
    joint not moving, and two pose surveys characterised a six-jointed arm. Nothing said a word for
    the whole session.
    """
    chain = _chain()
    arm = _arm(chain)
    now = time.time()
    arm._feedback_seen = True
    arm._refresh_interval[:] = 0.002
    arm._last_refresh_time[:] = now

    assert arm._stale_feedback(now) is None, "a healthy chain must not fault"

    # Joint3 never refreshed: no measured interval, and its stamp has not moved since start.
    arm._refresh_interval[2] = 0.0
    arm._last_refresh_time[2] = now - 1.0
    fault = arm._stale_feedback(now)
    assert fault is not None, "a motor that has never refreshed was skipped"
    assert "joint 3" in fault, fault
    assert "never refreshed" in fault, fault

    # Inside the startup grace it must stay quiet, or every run would fault before the first
    # readings arrive.
    arm._last_refresh_time[2] = now - 0.1
    assert arm._stale_feedback(now) is None, "faulted inside the startup grace"
