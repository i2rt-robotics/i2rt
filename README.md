# I2RT Python API

A Python client library for interacting with [I2RT](https://i2rt.com/) products, designed with simplicity and extensibility in mind.

[![I2RT](https://github.com/user-attachments/assets/025ac3f0-7af1-4e6f-ab9f-7658c5978f92)](https://i2rt.com/)
## Features

- Plug and play python interface for I2RT robots
- Real-time robot control via CAN bus communication
- Support for directly communicating with motor (DM series motors)
- Visualization and gravity compensation using MuJoCo physics engine

## Installation

```bash
pip install -e .
```

## CAN Usage
Plug in the CAN device and run the following command to check the available CAN devices.
```bash
ls -l /sys/class/net/can*
```

This should give you something like this
```bash
lrwxrwxrwx 1 root root 0 Jul 15 14:35 /sys/class/net/can0 -> ../../devices/platform/soc/your_can_device/can0
```

Where can0 is the CAN device name.

You need to bring up the CAN interface with
```bash
sudo ip link set can0 up type can bitrate 1000000
```

We have provided a convenience script to reset all CAN devices. Simply run
```bash
sh scripts/reset_all_can.sh
```

## YAM Robot Arm Usage

### Getting started
```python
from i2rt.robots.motor_chain_robot import get_yam_robot

# Get a robot instance
robot = get_yam_robot(channel="can0")

# Get the current joint positions
joint_pos = robot.get_joint_pos()

# Command the robot to move to a new joint position
target_pos = np.array([0, 0, 0, 0, 0, 0, 0])

# Command the robot to move to the target position
robot.command_joint_pos(target_pos)
```

### Running the arm and visualizing it
To launch the follower robot run.
```bash
python scripts/minimum_gello.py --mode follower
```

To launch the robot mujoco visualizer run
```bash
python scripts/minimum_gello.py --mode visualizer
```

### Running the arm on controlling it leader follower style
This requires 2 robot arms.

To launch the follower robot run
```bash
python scripts/minimum_gello.py --mode follower --can_channel can0
```

To launch the leader robot run
```bash
python scripts/minimum_gello.py --mode leader  --can_channel can0
```

## Flow Base Usage

### Running the demo
You can control your flow base using a game controller.
To run the joystick demo, run the following command.
```bash
python i2rt/flow_base/flow_base_controller.py
```

### Getting started
```python
from i2rt.flow_base.flow_base_controller import Vehicle

# Get a robot instance
vehicle = Vehicle()
vehicle.start_control()

# move forward slowly for 1 second
start_time = time.time()
while time.time() - start_time < 1:
    user_cmd = (0.1, 0, 0)
    vehicle.set_target_velocity(user_cmd, frame="local")
```

## Contributing
We welcome contributions! Please make a PR.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support
- Contact: support@i2rt.com

## Acknowledgments
- [TidyBot++](https://github.com/jimmyyhwu/tidybot2) - Flow base hardware and code is inspired by TidyBot++
- [GELLO](https://github.com/wuphilipp/gello_software) - Robot arm teleop is inspired by GELLO
