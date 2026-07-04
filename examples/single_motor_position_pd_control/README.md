# Single Motor Position PD Control Example

This example demonstrates a simple interface to control a single motor using your keyboard. It is intended for quick testing and manual control of a motor via CAN bus.

## What does this example do?

- Lets you increment or decrement the motor's position target using the keyboard (arrow keys or `h`/`l`).
- You can adjust the step size for each key press (with up/down arrows or `k`/`j`).
- Resets the target to the current position (`r` key).
- Exits cleanly with `q` or `ESC`.

## Usage

Run the script from the command line:

```bash
python examples/single_motor_position_pd_control/single_motor_position_pd_control.py --channel can1 --motor_id 1 --kd 5
```

You will see example panel like this:
```bash
Arrow-key PD teleop (q to quit)
Current pos : -1.64359 rad
Target  pos : -1.49720 rad
Velocity    : -0.00244 rad/s
Torque      : +11.65812 Nm (or driver units)
Temp rotor  : 45.0 °C   Temp MOS: 35.0 °C
Error code  : 0x1

Step size   : 0.01000 rad   (↑ bigger / ↓ smaller)
KP=80.00  KD=3.00

Controls: ←/→ move • r reset-to-current • SPACE hold • q quit
```

You can interactively control the model and read the motor data in an interactive manner.
