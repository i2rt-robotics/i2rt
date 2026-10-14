#!/usr/bin/env python3
"""Serve YAM camera and passive motor telemetry to a remote Rerun viewer.

The CAN readers are deliberately passive: they open SocketCAN receive sockets and never
transmit. Joint and gripper plots populate while the real robot controller is running and
producing DM feedback frames. The three D405 image feeds are captured directly on this host.

Run on yam-01:
    uv run examples/yam_observation_rerun/yam_observation_rerun.py

Connect a matching Rerun viewer from a laptop:
    rerun rerun+http://yam-01:9876/proxy
"""

from __future__ import annotations

import queue
import signal
import subprocess
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import can
import numpy as np
import pyrealsense2 as rs
import rerun as rr
import rerun.blueprint as rrb
import tyro

from i2rt.motor_drivers.utils import MotorType, uint_to_float
from i2rt.robots.utils import ArmType, GripperType, _load_arm_config

CAMERA_NAMES = {
    "255323071054": "wrist_255323071054",
    "255323074657": "wrist_255323074657",
    "255323073978": "head",
}
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 7))


@dataclass
class Args:
    bind_ip: str = "0.0.0.0"
    """Address on which the Rerun gRPC server listens."""

    port: int = 9876
    """Rerun gRPC server port."""

    server_memory_limit: str = "2GiB"
    """Maximum buffered Rerun data retained for late-connecting viewers."""

    save: Path | None = None
    """Optional .rrd output path. Existing files are never overwritten."""

    camera_fps: int = 15
    camera_width: int = 1280
    camera_height: int = 720
    jpeg_quality: int = 80
    exposure_us: float = 8000.0
    gain: float = 16.0

    telemetry_rate_hz: float = 50.0
    """Rate at which the latest passive CAN state is logged."""

    can_channels: tuple[str, ...] = ("can0", "can1")
    """SocketCAN channels to observe without transmitting."""

    arm_type: ArmType = ArmType.YAM
    """Arm model used to decode velocity/effort scales. Position scale is common to YAM variants."""

    gripper_type: GripperType = GripperType.LINEAR_4310
    """Gripper model on motor ID 7; its raw motor state is logged without calibration."""


@dataclass(frozen=True)
class Camera:
    serial: str
    name: str
    device: Path


@dataclass(frozen=True)
class MotorFeedback:
    timestamp: float
    position: float
    velocity: float
    effort: float
    error_code: int


def _blueprint(cameras: list[Camera], channels: tuple[str, ...]) -> rrb.Blueprint:
    camera_views = [rrb.Spatial2DView(origin=f"/cameras/{camera.name}", name=camera.name) for camera in cameras]
    telemetry_views = []
    for channel in channels:
        telemetry_views.extend(
            [
                rrb.TimeSeriesView(origin=f"/robots/{channel}/joints", name=f"{channel} joints"),
                rrb.TimeSeriesView(origin=f"/robots/{channel}/gripper", name=f"{channel} gripper"),
            ]
        )
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(*camera_views),
            rrb.Horizontal(*telemetry_views),
            row_shares=[2, 1],
        ),
        collapse_panels=True,
    )


def _device_properties(device: Path) -> dict[str, str]:
    result = subprocess.run(
        ["udevadm", "info", "--query=property", "--name", str(device)],
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def discover_cameras() -> list[Camera]:
    found: dict[str, Path] = {}
    for device in sorted(Path("/dev").glob("video*")):
        index_path = Path("/sys/class/video4linux") / device.name / "index"
        if not index_path.exists() or index_path.read_text().strip() != "4":
            continue
        properties = _device_properties(device)
        serial = properties.get("ID_SERIAL_SHORT", "")
        if not serial:
            serial = properties.get("ID_SERIAL", "").rsplit("_", 1)[-1]
        if serial in CAMERA_NAMES:
            found[serial] = device

    missing = sorted(set(CAMERA_NAMES) - set(found))
    if missing:
        raise RuntimeError(f"Missing expected RealSense camera(s): {', '.join(missing)}")
    return [Camera(serial, name, found[serial]) for serial, name in CAMERA_NAMES.items()]


def configure_exposure(exposure_us: float, gain: float) -> None:
    devices = rs.context().query_devices()
    if len(devices) != len(CAMERA_NAMES):
        raise RuntimeError(f"Expected {len(CAMERA_NAMES)} RealSense devices, found {len(devices)}")
    for device in devices:
        sensor = device.first_depth_sensor()
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, exposure_us)
        sensor.set_option(rs.option.gain, gain)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"[camera] {serial}: exposure={exposure_us:.0f} us gain={gain:.0f}", flush=True)


def _gst_command(camera: Camera, args: Args) -> list[str]:
    return [
        "gst-launch-1.0",
        "-q",
        "v4l2src",
        f"device={camera.device}",
        "!",
        (
            f"video/x-raw,format=YUY2,width={args.camera_width},height={args.camera_height},"
            f"framerate={args.camera_fps}/1"
        ),
        "!",
        "videoconvert",
        "!",
        "jpegenc",
        f"quality={args.jpeg_quality}",
        "!",
        "fdsink",
        "fd=1",
        "sync=false",
    ]


def _read_jpegs(stream: BinaryIO, stop: threading.Event) -> Generator[bytes, None, None]:
    buffer = bytearray()
    while not stop.is_set():
        chunk = stream.read(64 * 1024)
        if not chunk:
            return
        buffer.extend(chunk)
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            end = buffer.find(b"\xff\xd9", start + 2)
            if end < 0:
                if start:
                    del buffer[:start]
                break
            end += 2
            yield bytes(buffer[start:end])
            del buffer[:end]


def _camera_worker(
    camera: Camera,
    args: Args,
    frames: queue.Queue[tuple[Camera, float, bytes]],
    stop: threading.Event,
    processes: list[subprocess.Popen[bytes]],
) -> None:
    process = subprocess.Popen(_gst_command(camera, args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes.append(process)
    assert process.stdout is not None
    try:
        for jpeg in _read_jpegs(process.stdout, stop):
            item = (camera, time.time(), jpeg)
            try:
                frames.put_nowait(item)
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                frames.put_nowait(item)
    finally:
        if process.poll() is None:
            process.terminate()


class PassiveCanMonitor:
    """Receive DM feedback frames from an existing controller without sending CAN traffic."""

    def __init__(self, channel: str, arm_type: ArmType, gripper_type: GripperType) -> None:
        self.channel = channel
        hardware = _load_arm_config(arm_type)
        self._motor_types = {motor_id: motor_type for motor_id, motor_type in hardware.motor_list}
        self._directions = {index + 1: direction for index, direction in enumerate(hardware.directions)}
        self._motor_types[7] = gripper_type.get_motor_type(arm_type)
        self._directions[7] = gripper_type.get_motor_direction(arm_type)
        self._bus = can.Bus(
            interface="socketcan",
            channel=channel,
            can_filters=[{"can_id": 0x10, "can_mask": 0x7F0, "extended": False}],
        )
        self._lock = threading.Lock()
        self._states: dict[int, MotorFeedback] = {}
        self._frame_count = 0
        self._thread: threading.Thread | None = None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def start(self, stop: threading.Event) -> None:
        self._thread = threading.Thread(target=self._run, args=(stop,), daemon=True, name=f"can-{self.channel}")
        self._thread.start()

    def _run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            message = self._bus.recv(timeout=0.1)
            if message is None or message.is_extended_id or len(message.data) != 8:
                continue
            motor_id = message.arbitration_id - 16
            motor_type = self._motor_types.get(motor_id)
            if motor_type is None:
                continue
            feedback = self._decode(message.data, motor_id, motor_type, message.timestamp)
            with self._lock:
                self._states[motor_id] = feedback
                self._frame_count += 1

    def _decode(self, data: bytearray, motor_id: int, motor_type: str, timestamp: float) -> MotorFeedback:
        constants = MotorType.get_motor_constants(motor_type)
        position_raw = (data[1] << 8) | data[2]
        velocity_raw = (data[3] << 4) | (data[4] >> 4)
        effort_raw = ((data[4] & 0x0F) << 8) | data[5]
        direction = self._directions[motor_id]
        return MotorFeedback(
            timestamp=timestamp,
            position=direction * uint_to_float(position_raw, constants.POSITION_MIN, constants.POSITION_MAX, 16),
            velocity=direction * uint_to_float(velocity_raw, constants.VELOCITY_MIN, constants.VELOCITY_MAX, 12),
            effort=direction * uint_to_float(effort_raw, constants.TORQUE_MIN, constants.TORQUE_MAX, 12),
            error_code=(data[0] & 0xF0) >> 4,
        )

    def snapshot(self) -> dict[int, MotorFeedback]:
        with self._lock:
            return dict(self._states)

    def close(self) -> None:
        self._bus.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=1)


def _log_static_series(channels: tuple[str, ...]) -> None:
    colors = [
        [230, 80, 80],
        [80, 200, 120],
        [90, 130, 235],
        [220, 180, 70],
        [190, 100, 220],
        [80, 200, 210],
    ]
    for channel in channels:
        for metric in ("position_rad", "velocity_rad_s", "effort_nm"):
            rr.log(
                f"robots/{channel}/joints/{metric}",
                rr.SeriesLines(names=list(JOINT_NAMES), colors=colors),
                static=True,
            )


def _log_motor_state(monitor: PassiveCanMonitor, now: float) -> None:
    states = monitor.snapshot()
    joints = [states.get(motor_id) for motor_id in range(1, 7)]
    if any(state is not None for state in joints):
        rr.log(
            f"robots/{monitor.channel}/joints/position_rad",
            rr.Scalars([state.position if state is not None else np.nan for state in joints]),
        )
        rr.log(
            f"robots/{monitor.channel}/joints/velocity_rad_s",
            rr.Scalars([state.velocity if state is not None else np.nan for state in joints]),
        )
        rr.log(
            f"robots/{monitor.channel}/joints/effort_nm",
            rr.Scalars([state.effort if state is not None else np.nan for state in joints]),
        )

    gripper = states.get(7)
    if gripper is not None:
        rr.log(f"robots/{monitor.channel}/gripper/position_rad", rr.Scalars(gripper.position))
        rr.log(f"robots/{monitor.channel}/gripper/velocity_rad_s", rr.Scalars(gripper.velocity))
        rr.log(f"robots/{monitor.channel}/gripper/effort_nm", rr.Scalars(gripper.effort))
        rr.log(f"robots/{monitor.channel}/gripper/error_code", rr.Scalars(gripper.error_code))

    newest = max((state.timestamp for state in states.values()), default=0.0)
    age = now - newest if newest else np.nan
    rr.log(f"status/{monitor.channel}/feedback_age_s", rr.Scalars(age))
    rr.log(f"status/{monitor.channel}/frame_count", rr.Scalars(monitor.frame_count))


def _configure_rerun(args: Args, cameras: list[Camera]) -> rr.GrpcServerSink:
    rr.init("yam01_observation")
    server = rr.GrpcServerSink(
        bind_ip=args.bind_ip,
        port=args.port,
        server_memory_limit=args.server_memory_limit,
        newest_first=True,
    )
    sinks: list[rr.LogSink] = [server]
    if args.save is not None:
        output = args.save.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing recording: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        sinks.append(rr.FileSink(output))
        print(f"[rerun] recording to {output}", flush=True)
    rr.set_sinks(*sinks, default_blueprint=_blueprint(cameras, args.can_channels))
    return server


def main(args: Args) -> None:
    cameras = discover_cameras()
    configure_exposure(args.exposure_us, args.gain)
    server = _configure_rerun(args, cameras)
    print(f"[rerun] connect viewer to {server.uri}", flush=True)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    frames: queue.Queue[tuple[Camera, float, bytes]] = queue.Queue(maxsize=len(cameras) * 2)
    processes: list[subprocess.Popen[bytes]] = []
    camera_threads = [
        threading.Thread(
            target=_camera_worker,
            args=(camera, args, frames, stop, processes),
            daemon=True,
            name=f"camera-{camera.name}",
        )
        for camera in cameras
    ]
    monitors = [PassiveCanMonitor(channel, args.arm_type, args.gripper_type) for channel in args.can_channels]

    _log_static_series(args.can_channels)
    rr.log(
        "status",
        rr.TextLog(
            "Camera logging active. CAN logging is passive and will populate when the robot controller emits feedback."
        ),
    )
    for thread in camera_threads:
        thread.start()
    for monitor in monitors:
        monitor.start(stop)

    telemetry_period = 1.0 / args.telemetry_rate_hz
    next_telemetry = time.monotonic()
    try:
        while not stop.is_set():
            timeout = max(0.0, min(0.1, next_telemetry - time.monotonic()))
            try:
                camera, timestamp, jpeg = frames.get(timeout=timeout)
                rr.set_time("wall_clock", timestamp=timestamp)
                rr.log(f"cameras/{camera.name}", rr.EncodedImage(contents=jpeg, media_type="image/jpeg"))
            except queue.Empty:
                pass

            if time.monotonic() >= next_telemetry:
                now = time.time()
                rr.set_time("wall_clock", timestamp=now)
                for monitor in monitors:
                    _log_motor_state(monitor, now)
                next_telemetry += telemetry_period
    finally:
        stop.set()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for thread in camera_threads:
            thread.join(timeout=2)
        for monitor in monitors:
            monitor.close()
        rr.disconnect()


if __name__ == "__main__":
    main(tyro.cli(Args))
