"""Unit tests for the base odometry pose integrator (Vehicle._integrate_pose).

These exercise the pure integration math with no hardware: the regression guard is that
displacement scales with the *actual* elapsed dt, which the previous fixed-nominal-period
integration could not express (it always advanced by control_period regardless of dt).
"""

import math

import numpy as np

from i2rt.flow_base.flow_base_controller import Vehicle


def test_pose_integration_scales_with_dt() -> None:
    # Regression guard: same twist, twice the dt -> twice the displacement.
    # The old fixed-dt integrator would have produced identical displacement for both.
    x0 = np.zeros(3)
    twist = np.array([1.0, 0.0, 0.0])  # 1 m/s forward, no rotation

    x_t, _ = Vehicle._integrate_pose(x0, twist, 0.005)
    x_2t, _ = Vehicle._integrate_pose(x0, twist, 0.010)

    np.testing.assert_allclose(x_t, [0.005, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(x_2t, [0.010, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(x_2t[0], 2.0 * x_t[0], atol=1e-12)


def test_straight_line_uses_measured_dt() -> None:
    # A slow loop (8 ms) must integrate the full 8 ms of motion, not the nominal 5 ms.
    x0 = np.zeros(3)
    twist = np.array([1.0, 0.0, 0.0])

    x_new, dx_world = Vehicle._integrate_pose(x0, twist, 0.008)

    np.testing.assert_allclose(x_new, [0.008, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(dx_world, twist, atol=1e-12)  # heading 0 -> world == body


def test_forward_motion_respects_starting_heading() -> None:
    # Forward body twist from a non-zero heading translates along that heading.
    theta0 = math.pi / 3
    dt = 0.008
    x0 = np.array([0.0, 0.0, theta0])
    twist = np.array([1.0, 0.0, 0.0])

    x_new, _ = Vehicle._integrate_pose(x0, twist, dt)

    expected = np.array([math.cos(theta0) * dt, math.sin(theta0) * dt, theta0])
    np.testing.assert_allclose(x_new, expected, atol=1e-12)


def test_pure_rotation_advances_heading() -> None:
    omega = 2.0  # rad/s
    dt = 0.008
    x0 = np.zeros(3)
    twist = np.array([0.0, 0.0, omega])

    x_new, _ = Vehicle._integrate_pose(x0, twist, dt)

    # Pure spin: heading advances by omega*dt, no translation.
    np.testing.assert_allclose(x_new, [0.0, 0.0, omega * dt], atol=1e-12)


def test_midpoint_heading_is_scaled_by_dt() -> None:
    # Combined forward + angular twist: translation is rotated by the *midpoint* heading
    # (0.5 * omega * dt), so its direction must depend on the measured dt.
    omega = 2.0
    dt = 0.008
    x0 = np.zeros(3)
    twist = np.array([1.0, 0.0, omega])

    x_new, _ = Vehicle._integrate_pose(x0, twist, dt)

    heading_of_translation = math.atan2(x_new[1], x_new[0])
    np.testing.assert_allclose(heading_of_translation, 0.5 * omega * dt, atol=1e-12)
    np.testing.assert_allclose(x_new[2], omega * dt, atol=1e-12)
