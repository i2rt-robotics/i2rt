"""Collision/overload effort-guard tests (no hardware)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")  # controllers import chain pulls mujoco

from i2rt.serving import control_config as cc
from i2rt.serving.controllers import BaseController


class _FakeRobot:
    def __init__(self, arm_eff):
        self._eff = np.asarray(arm_eff, dtype=float)

    def num_dofs(self):
        return self._eff.size + 1  # + gripper

    def get_joint_pos(self):
        return np.zeros(self._eff.size + 1)

    def get_observations(self):
        n = self._eff.size
        return {
            "joint_pos": np.zeros(n),
            "joint_vel": np.zeros(n),
            "joint_eff": self._eff,
            "gripper_pos": np.zeros(1),
            "gripper_vel": np.zeros(1),
            "gripper_eff": np.array([99.0]),  # gripper is excluded -> must NOT trip
        }


def test_effort_guard_trips(monkeypatch):
    monkeypatch.setattr(cc, "FOLLOWER_EFFORT_LIMIT", 5.0)
    bc = BaseController()
    bc._effort_guard(_FakeRobot([0, 0, 9.0, 0, 0, 0]))  # one arm joint over the limit
    assert bc._estop is True


def test_effort_guard_ok(monkeypatch):
    monkeypatch.setattr(cc, "FOLLOWER_EFFORT_LIMIT", 5.0)
    bc = BaseController()
    bc._effort_guard(_FakeRobot([1.0, 2.0, 3.0, 0, 0, 0]))  # arm under limit (gripper high, ignored)
    assert bc._estop is False


def test_effort_guard_disabled(monkeypatch):
    monkeypatch.setattr(cc, "FOLLOWER_EFFORT_LIMIT", None)
    bc = BaseController()
    bc._effort_guard(_FakeRobot([100.0, 0, 0, 0, 0, 0]))
    assert bc._estop is False  # disabled -> never trips
