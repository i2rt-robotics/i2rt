"""i2rt.serving — robot networking for YAM (portal + websocket).

The robot machine runs a :class:`~i2rt.serving.robot_server.RobotServer` wrapping
one of the bimanual controllers (teleop / dagger / wrapper) and serves its state +
inputs over ``portal`` (plain TCP). The workstation connects with
:class:`~i2rt.serving.robot_client.RobotClient`.

The real-time control core (gating, bilateral teleop, takeover, smoothing) lives in
:mod:`~i2rt.serving.teleop_common`, :mod:`~i2rt.serving.safety`,
:mod:`~i2rt.serving.control_config`.
"""
