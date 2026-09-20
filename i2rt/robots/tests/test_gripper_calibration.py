from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from i2rt.robots import utils


def test_calibration_holds_arm_and_stops_gripper_without_replaying_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    commands = []

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    def set_commands(**kwargs: Any) -> None:
        commands.append({key: value.copy() for key, value in kwargs.items()})

    chain = SimpleNamespace(
        motor_list=[1, 2],
        motor_direction=np.ones(2),
        read_states=lambda: [SimpleNamespace(pos=0.3, eff=5.0), SimpleNamespace(pos=0.4, eff=6.0)],
        set_commands=set_commands,
    )
    monkeypatch.setattr(utils, "time", SimpleNamespace(time=lambda: now, sleep=sleep))
    utils.detect_gripper_limits(chain, gripper_index=1)

    assert {float(command["torques"][1]) for command in commands} == {-0.2, 0.0, 0.2}
    for command in commands:
        np.testing.assert_array_equal(command["pos"], [0.3, 0.4])
        np.testing.assert_array_equal(command["vel"], [0.0, 0.0])
        np.testing.assert_array_equal(command["kp"], [5.0, 0.0])
        np.testing.assert_array_equal(command["kd"], [0.5, 0.0])
        assert command["torques"][0] == 0.0
    np.testing.assert_array_equal(commands[-1]["torques"], [0.0, 0.0])
