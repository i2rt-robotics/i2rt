"""Record motor feedback as ROS 2 CDR messages in an MCAP file."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional, Sequence

from mcap.records import Schema
from mcap_ros2.writer import Writer as McapWriter

_JOINT_STATE_SCHEMA_NAME = "sensor_msgs/msg/JointState"
_JOINT_STATE_MSGDEF = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

_TEMPERATURE_SCHEMA_NAME = "sensor_msgs/msg/Temperature"
_TEMPERATURE_MSGDEF = """std_msgs/Header header
float64 temperature
float64 variance
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""

_STOP = object()


@dataclass(frozen=True)
class _FeedbackSample:
    timestamp: float
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    effort: tuple[float, ...]
    required_torque: tuple[float, ...]
    temp_mos: tuple[float, ...]
    temp_rotor: tuple[float, ...]


def create_robot_mcap_path(root: Optional[Path] = None, now: Optional[datetime] = None) -> Path:
    """Create <root>/YYYYMMDD/YYYYMMDD-HHmmss/robot.mcap without overwriting an existing recording."""
    recording_root = Path.home() / ".i2rt" if root is None else root.expanduser()
    started_at = datetime.now().astimezone() if now is None else now
    recording_dir = recording_root / started_at.strftime("%Y%m%d") / started_at.strftime("%Y%m%d-%H%M%S")
    recording_dir.mkdir(parents=True, exist_ok=False)
    return recording_dir / "robot.mcap"


class RobotMcapRecorder:
    """Asynchronously write every motor feedback sample to one ROS 2 MCAP file."""

    def __init__(self, path: Path, joint_names: Sequence[str]) -> None:
        if not joint_names:
            raise ValueError("At least one joint name is required")

        self.path = path
        self._joint_names = tuple(joint_names)
        self._stream: BinaryIO = path.open("xb")
        self._writer = McapWriter(self._stream)
        self._joint_state_schema: Schema = self._writer.register_msgdef(_JOINT_STATE_SCHEMA_NAME, _JOINT_STATE_MSGDEF)
        self._temperature_schema: Schema = self._writer.register_msgdef(_TEMPERATURE_SCHEMA_NAME, _TEMPERATURE_MSGDEF)
        self._queue: queue.SimpleQueue[_FeedbackSample | object] = queue.SimpleQueue()
        self._writer_error: Optional[BaseException] = None
        self._closed = False
        self._sequence = 0
        self._thread = threading.Thread(target=self._write_loop, name="robot_mcap_writer")
        self._thread.start()

    @classmethod
    def create(
        cls,
        joint_names: Sequence[str],
        root: Optional[Path] = None,
        now: Optional[datetime] = None,
    ) -> "RobotMcapRecorder":
        return cls(create_robot_mcap_path(root=root, now=now), joint_names)

    def add(
        self,
        *,
        timestamp: float,
        position: Sequence[float],
        velocity: Sequence[float],
        effort: Sequence[float],
        required_torque: Sequence[float],
        temp_mos: Sequence[float],
        temp_rotor: Sequence[float],
    ) -> None:
        """Queue one complete motor feedback frame for recording."""
        if self._closed:
            raise RuntimeError("Cannot add feedback to a closed MCAP recorder")
        if self._writer_error is not None:
            raise RuntimeError("MCAP writer failed") from self._writer_error

        sample = _FeedbackSample(
            timestamp=timestamp,
            position=tuple(float(value) for value in position),
            velocity=tuple(float(value) for value in velocity),
            effort=tuple(float(value) for value in effort),
            required_torque=tuple(float(value) for value in required_torque),
            temp_mos=tuple(float(value) for value in temp_mos),
            temp_rotor=tuple(float(value) for value in temp_rotor),
        )
        lengths = {
            len(sample.position),
            len(sample.velocity),
            len(sample.effort),
            len(sample.required_torque),
            len(sample.temp_mos),
            len(sample.temp_rotor),
            len(self._joint_names),
        }
        if len(lengths) != 1:
            raise ValueError("Recorded arrays must match the number of joint names")
        self._queue.put(sample)

    def _write_loop(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                assert isinstance(item, _FeedbackSample)
                self._write_sample(item)
        except BaseException as exc:
            self._writer_error = exc
        finally:
            try:
                self._writer.finish()
            except BaseException as exc:
                if self._writer_error is None:
                    self._writer_error = exc
            finally:
                self._stream.close()

    def _write_sample(self, sample: _FeedbackSample) -> None:
        timestamp_ns = int(sample.timestamp * 1_000_000_000)
        stamp_sec, stamp_nanosec = divmod(timestamp_ns, 1_000_000_000)
        stamp = {"sec": stamp_sec, "nanosec": stamp_nanosec}

        self._writer.write_message(
            topic="/joint_states",
            schema=self._joint_state_schema,
            message={
                "header": {"stamp": stamp, "frame_id": ""},
                "name": list(self._joint_names),
                "position": list(sample.position),
                "velocity": list(sample.velocity),
                "effort": list(sample.effort),
            },
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            sequence=self._sequence,
        )
        self._writer.write_message(
            topic="/required_torques",
            schema=self._joint_state_schema,
            message={
                "header": {"stamp": stamp, "frame_id": ""},
                "name": list(self._joint_names),
                "position": [],
                "velocity": [],
                "effort": list(sample.required_torque),
            },
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            sequence=self._sequence,
        )

        for index, joint_name in enumerate(self._joint_names):
            header = {"stamp": stamp, "frame_id": joint_name}
            for sensor_name, temperature in (
                ("mos", sample.temp_mos[index]),
                ("rotor", sample.temp_rotor[index]),
            ):
                self._writer.write_message(
                    topic=f"/{joint_name}/temperature/{sensor_name}",
                    schema=self._temperature_schema,
                    message={"header": header, "temperature": temperature, "variance": 0.0},
                    log_time=timestamp_ns,
                    publish_time=timestamp_ns,
                    sequence=self._sequence,
                )
        self._sequence += 1

    def close(self) -> None:
        """Drain queued samples and finalize the MCAP indexes and footer."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(_STOP)
        self._thread.join()
        if self._writer_error is not None:
            raise RuntimeError(f"Failed to write MCAP recording: {self.path}") from self._writer_error
