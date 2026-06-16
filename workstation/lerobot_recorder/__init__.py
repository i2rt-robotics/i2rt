"""LeRobot dataset recorder for the bimanual YAM teleop rig.

Runs on the **workstation** (a different machine / Python env than the YAM
robot): it connects to the YAM ROS 2 graph remotely, reads the three RealSense
cameras locally, and records LeRobot episodes that auto-start/stop from the
teleop gate signal (``/teleop/state`` / ``/teleop/active``).

This package is intentionally decoupled from the ``i2rt`` package — it only
needs ``rclpy`` (to subscribe), ``pyrealsense2`` (cameras), ``lerobot`` (dataset),
and ``PyQt`` (GUI). See ``requirements.txt`` and ``README.md``.
"""
