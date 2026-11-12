"""scripts/zero_7dof_arm.py's hard-stop zeroing against a fake chain whose joints stop dead at their stops."""

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from i2rt.robots.utils import ArmType, _load_arm_config

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "zero_7dof_arm.py"
BACKOFF = np.radians([160, 205, 105, 120, 105, 66.22, 90])
HW = _load_arm_config(ArmType.YAM_7DOF_RIGHT)  # the real gains: a retune that breaks stall detection fails here


class FakeChain:
    """Each joint follows its position command exactly, up to its hard stop."""

    running = True

    def __init__(self, stops: np.ndarray) -> None:
        self.stops = stops
        self.pos = np.zeros(len(stops))

    def set_commands(self, torques: Any, pos: np.ndarray, *args: Any, **kwargs: Any) -> None:
        self.pos = np.minimum(pos, self.stops)

    def read_states(self) -> list:
        return [SimpleNamespace(pos=p) for p in self.pos]


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location("zero_7dof_arm", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def test_each_joint_ends_its_back_off_below_its_stop(script: ModuleType) -> None:
    true_zero = np.radians([10, -20, 5, 0, -30, 15, 40])  # where zero is, seen from the arm's starting pose
    chain = FakeChain(true_zero + BACKOFF)
    script.zero_arm(script.Hold(chain, HW.kp, HW.kd), BACKOFF)
    np.testing.assert_allclose(chain.pos, true_zero, atol=1e-9)


def test_a_joint_stalling_short_backs_out(script: ModuleType) -> None:
    chain = FakeChain(BACKOFF.copy())
    chain.stops[1] = np.radians(30)  # joint2 pushing the wrong way meets the stop 25 deg below zero
    with pytest.raises(RuntimeError, match="joint2 stalled far short"):
        script.zero_arm(script.Hold(chain, HW.kp, HW.kd), BACKOFF)
    assert abs(chain.pos[1]) < 1e-9
