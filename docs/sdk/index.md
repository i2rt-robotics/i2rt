# SDK Reference

The `i2rt` package provides Python interfaces for all I2RT hardware. Everything is importable from the top-level package after installing with `uv pip install -e .`.

## Modules

| Module | Description |
|--------|-------------|
| `i2rt.robots.motor_chain_robot` | YAM arm control — joint position, velocity, gravity compensation |
| `i2rt.flow_base.flow_base_controller` | Flow Base on-board control (runs on Pi) |
| `i2rt.flow_base.flow_base_client` | Flow Base remote network client |
| `i2rt.motor_drivers` | Low-level DM series motor communication |
| `i2rt.motor_config_tool` | One-time motor configuration utilities |

## Pages

- [YAM Arm API](/sdk/yam-arm)
- [Flow Base API](/sdk/flow-base)
- [Grippers](/sdk/grippers)
