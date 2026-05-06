# SDK Reference

The `i2rt` package provides Python interfaces for all I2RT hardware. Everything is importable from the top-level package after installing with `uv pip install -e .`.

## Modules

| Module | Description |
|--------|-------------|
| `i2rt.robots.get_robot` | Factory function `get_yam_robot()` for creating arm instances |
| `i2rt.robots.motor_chain_robot` | `MotorChainRobot` class — joint control, gravity compensation |
| `i2rt.robots.robot` | `Robot` protocol — abstract interface for all robots |
| `i2rt.robots.sim_robot` | `SimRobot` — MuJoCo-based simulation robot |
| `i2rt.robots.utils` | `ArmType`, `GripperType` enums, XML combiner, gripper utilities |
| `i2rt.robots.config` | YAML hardware configs for arms and grippers (motor lists, directions, PD gains, gravity-comp factors) |
| `i2rt.flow_base.flow_base_controller` | Flow Base on-board control (runs on Pi) |
| `i2rt.flow_base.flow_base_client` | Flow Base remote network client |
| `i2rt.motor_drivers` | Low-level DM series motor communication (CAN bus) |
| `i2rt.motor_config_tool` | One-time motor configuration utilities (zero offset, timeout) |
| `i2rt.utils` | MuJoCo helpers, Viser visualization, gamepad, encoder utilities |

## Pages

- [YAM Arm API](/sdk/yam-arm)
- [Flow Base API](/sdk/flow-base)
- [Grippers](/sdk/grippers)
