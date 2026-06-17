"""LeRobot dataset recorder for the bimanual YAM teleop rig.

Runs on the **workstation** (a different machine / Python env than the YAM
robot): it connects to the YAM robot server **remotely over portal** (plain TCP,
no ROS), reads the three RealSense cameras locally, and records LeRobot episodes
that auto-start/stop from the teleop gate signal (the robot snapshot's
``teleop_state``).

It needs ``i2rt`` (for the portal ``RobotClient``), ``pyrealsense2`` (cameras),
``lerobot`` (dataset), and ``PyQt`` (GUI) — but **no ROS**. See ``requirements.txt``
and ``README.md``.
"""
