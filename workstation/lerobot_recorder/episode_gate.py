"""Episode start/stop logic driven by the teleop gate signal.

One episode spans **ENGAGED → HOMING → IDLE** (inclusive of the homing return):

* it **starts** the moment teleop becomes ``ENGAGED`` (human lifted both gellos),
* it keeps **recording** through ``ENGAGED`` and the automatic ``HOMING`` return,
* it **stops** when the state first returns to ``IDLE`` — i.e. homing finished.

The gate is only live while *armed* (the GUI "Start collection" button). Pure
logic, no ROS/cameras — unit-tested directly.
"""

from __future__ import annotations

# Teleop states reported by i2rt.serving (TeleopController) in the robot snapshot.
HOMING = "HOMING"
IDLE = "IDLE"
ENGAGED = "ENGAGED"

# Events returned by EpisodeGate.update().
EV_IDLE = "idle"  # nothing to do
EV_START = "start"  # begin a new episode; also record this frame
EV_RECORD = "record"  # record this frame
EV_STOP = "stop"  # record this final frame, then save the episode


class EpisodeGate:
    """Turn a stream of teleop states into start / record / stop events."""

    def __init__(self) -> None:
        self._armed = False
        self._recording = False

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def recording(self) -> bool:
        return self._recording

    def arm(self) -> None:
        """Enable auto start/stop (GUI 'Start collection')."""
        self._armed = True

    def disarm(self) -> str:
        """Disable the gate (GUI 'Stop collection').

        Returns ``"abort"`` if an episode was in progress (the caller should
        discard the partial episode), else ``EV_IDLE``.
        """
        self._armed = False
        if self._recording:
            self._recording = False
            return "abort"
        return EV_IDLE

    def update(self, teleop_state: str) -> str:
        """Advance with the latest teleop state; return an event."""
        if not self._armed:
            return EV_IDLE

        if not self._recording:
            if teleop_state == ENGAGED:
                self._recording = True
                return EV_START
            return EV_IDLE

        # recording in progress: keep going through ENGAGED and HOMING
        if teleop_state == IDLE:
            self._recording = False
            return EV_STOP
        return EV_RECORD
