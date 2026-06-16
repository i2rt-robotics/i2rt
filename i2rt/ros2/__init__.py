"""ROS 2 integration for I2RT robots (build-less, pure rclpy).

This subpackage exposes I2RT YAM arms over ROS 2 using only standard message
types (``sensor_msgs``/``std_msgs``), so it needs **no colcon/ament build** —
just ``conda activate i2rt_ros`` (which sources ROS 2 Humble) and run the
modules directly, e.g.::

    python -m i2rt.ros2.run_wrapper --sim
    python -m i2rt.ros2.run_teleop --sim
    python -m i2rt.ros2.run_dagger --sim

See ``i2rt/ros2/README.md`` for the full topic contract and usage.
"""
