# Examples

All examples live in the [`examples/`](https://github.com/i2rt-robotics/i2rt/tree/main/examples) directory of the repository. Each example has its own README with step-by-step instructions.

## Available Examples

| Example | Hardware Needed | Complexity |
|---------|----------------|------------|
| [Bimanual Teleoperation](/examples/bimanual-teleoperation) | 4 YAM arms + 4 CANable | Intermediate |
| [Record & Replay Trajectory](/examples/record-replay) | 1+ YAM arm | Beginner |
| [Single Motor PD Control](/examples/motor-control) | 1 DM motor | Beginner |
| [MuJoCo Control Interface](/examples/control-with-mujoco) | None (sim) / 1 YAM arm | Beginner |

## Running an Example

```bash
cd /path/to/i2rt
source .venv/bin/activate
python examples/<example_name>/<script>.py
```
