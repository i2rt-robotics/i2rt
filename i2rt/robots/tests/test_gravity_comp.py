"""Test gravity compensation torques for all arm/gripper combinations.

Loads each combined model into MuJoCoKDL, samples joint configurations,
and verifies gravity compensation torques stay within a safe range.
Only arm revolute joints are checked — slide joints (gripper fingers)
are excluded, matching real robot behavior.
"""

import mujoco
import numpy as np
import pytest

from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.mujoco_utils import MuJoCoKDL

_MAX_TORQUE = {
    ArmType.BIG_YAM: 25.0,
}
_DEFAULT_MAX_TORQUE = 20.0  # Nm — same threshold as MotorChainRobot
NUM_SAMPLES = 20

ALL_ARM_GRIPPER_COMBOS = [(arm, gripper) for arm in ArmType for gripper in GripperType if arm != ArmType.NO_ARM]


def _combo_id(val: object) -> str:
    if isinstance(val, (ArmType, GripperType)):
        return val.value
    return str(val)


def _count_arm_joints(model: mujoco.MjModel) -> int:
    """Count revolute (hinge) joints, excluding slide joints used by gripper fingers."""
    n = 0
    for i in range(model.njnt):
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
            n += 1
    return n


@pytest.mark.parametrize("arm,gripper", ALL_ARM_GRIPPER_COMBOS, ids=_combo_id)
def test_gravity_comp_torques_in_range(arm: ArmType, gripper: GripperType) -> None:
    """Gravity compensation torques on arm joints should stay below MAX_TORQUE."""
    model_path = combine_arm_and_gripper_xml(arm, gripper)
    kdl = MuJoCoKDL(model_path)
    n_arm = _count_arm_joints(kdl.model)
    joint_ranges = kdl.model.jnt_range[:n_arm]

    rng = np.random.default_rng(42)

    for i in range(NUM_SAMPLES):
        q = rng.uniform(joint_ranges[:, 0], joint_ranges[:, 1])
        zeros = np.zeros(n_arm)
        torques = kdl.compute_inverse_dynamics(q, zeros, zeros)

        max_torque = _MAX_TORQUE.get(arm, _DEFAULT_MAX_TORQUE)
        max_abs = np.max(np.abs(torques))
        assert max_abs < max_torque, (
            f"Sample {i}: max gravity torque {max_abs:.2f} Nm exceeds {max_torque} Nm "
            f"(arm={arm.value}, gripper={gripper.value}, q={q}, torques={torques})"
        )


@pytest.mark.parametrize("arm,gripper", ALL_ARM_GRIPPER_COMBOS, ids=_combo_id)
def test_gravity_comp_at_zero_config(arm: ArmType, gripper: GripperType) -> None:
    """Gravity torques at zero configuration should be finite and within range."""
    model_path = combine_arm_and_gripper_xml(arm, gripper)
    kdl = MuJoCoKDL(model_path)
    n_arm = _count_arm_joints(kdl.model)

    q = np.zeros(n_arm)
    zeros = np.zeros(n_arm)
    torques = kdl.compute_inverse_dynamics(q, zeros, zeros)

    assert np.all(np.isfinite(torques)), (
        f"Non-finite torques at zero config (arm={arm.value}, gripper={gripper.value}): {torques}"
    )
    max_torque = _MAX_TORQUE.get(arm, _DEFAULT_MAX_TORQUE)
    max_abs = np.max(np.abs(torques))
    assert max_abs < max_torque, (
        f"Zero-config gravity torque {max_abs:.2f} Nm exceeds {max_torque} Nm "
        f"(arm={arm.value}, gripper={gripper.value}, torques={torques})"
    )
