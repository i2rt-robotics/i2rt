import functools
from dataclasses import dataclass


@dataclass
class MotorSpec:
    name: str
    # Base torque in N/M
    base_torque: float
    # Base current in A
    base_current: float
    # Max torque in N/M
    max_torque: float
    # Max current in A
    max_current: float
    # Gear ratio

    @classmethod
    @functools.cache
    def _get_alpha(
        cls,
        base_torque: float,
        max_torque: float,
        base_current: float,
        max_current: float,
    ) -> float:
        return (max_current - base_current) / (max_torque - base_torque)

    @classmethod
    @functools.cache
    def _get_beta(cls, base_current: float, base_torque: float, alpha: float) -> float:
        return base_current - base_torque * alpha

    def torque_to_current(self, torque: float) -> float:
        alpha = self._get_alpha(
            self.base_torque, self.max_torque, self.base_current, self.max_current
        )
        beta = self._get_beta(self.base_current, self.base_torque, alpha)
        return alpha * torque + beta


# Specs from DM
MOTOR_SPECS = {
    "DM4340": MotorSpec(
        name="DM4340",
        base_torque=9,
        base_current=2.5,
        max_torque=27,
        max_current=8,
    ),
    "DM4310": MotorSpec(
        name="DM4310",
        base_torque=3,
        base_current=2.5,
        max_torque=7,
        max_current=7.5,
    ),
}
