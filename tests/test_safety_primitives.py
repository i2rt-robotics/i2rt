"""Pure safety primitives (no robot/mujoco): clamp, rate limit, finiteness."""

from __future__ import annotations

import numpy as np

from i2rt.serving.safety import TargetSmoother, clamp_limits, is_finite_vector, max_step_from_speed


def test_clamp_limits():
    out = clamp_limits(np.array([2.0, -2.0, 0.5]), [(-1.0, 1.0), (-1.0, 1.0), None])
    assert out.tolist() == [1.0, -1.0, 0.5]  # clamped, None left as-is
    assert clamp_limits(np.array([5.0]), None).tolist() == [5.0]  # disabled -> unchanged


def test_target_smoother_caps_step():
    s = TargetSmoother(np.zeros(2), max_step=0.1)
    assert np.allclose(s.step(np.array([1.0, 1.0])), [0.1, 0.1])  # capped per tick
    assert np.allclose(s.step(np.array([1.0, 1.0])), [0.2, 0.2])  # advances again
    s.reset(np.array([0.5, 0.5]))
    assert np.allclose(s.cur, [0.5, 0.5])


def test_max_step_from_speed():
    assert abs(max_step_from_speed(1.0, 100.0) - 0.01) < 1e-9


def test_is_finite_vector():
    assert is_finite_vector(np.zeros(7), 7)
    assert not is_finite_vector(np.array([np.nan, 0.0]))
    assert not is_finite_vector(np.zeros(6), 7)  # wrong length
    assert not is_finite_vector(None)
