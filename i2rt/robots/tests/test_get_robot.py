import numpy as np

from i2rt.robots.get_robot import _apply_arm_motor_wrap_offsets


def test_wrap_offsets_apply_only_to_arm_motors() -> None:
    motor_offsets = np.zeros(7)
    motor_positions = np.array(
        [
            np.pi + 0.2,
            -np.pi - 0.3,
            0.1,
            -0.2,
            0.3,
            -0.4,
            6.326962691691463,
        ]
    )

    _apply_arm_motor_wrap_offsets(motor_offsets, motor_positions, n_arm_joints=6)

    np.testing.assert_allclose(
        motor_offsets,
        np.array([2 * np.pi, -2 * np.pi, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
