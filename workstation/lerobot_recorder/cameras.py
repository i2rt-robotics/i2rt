"""RealSense camera manager — opens the configured cameras, grabs latest frames.

Each :class:`~workstation.lerobot_recorder.config.CameraSpec` maps a serial to a
dataset image key (``wrist_left`` / ``wrist_right`` / ``agentview``). ``read()``
returns the most recent color frame per key as an ``HxWx3 uint8`` array.

``mock=True`` returns synthetic frames so the pipeline runs without hardware.

CLI: ``python -m workstation.lerobot_recorder.cameras --list`` prints connected
RealSense serials so you can fill them into the config.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from workstation.lerobot_recorder.config import CameraSpec, RecorderConfig

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self.specs: List[CameraSpec] = cfg.cameras
        self._pipelines: Dict[str, object] = {}
        self._serials: Dict[str, str] = {}  # resolved serial per key (for reconnect)
        self._last: Dict[str, np.ndarray] = {}  # last good frame per key
        self._healthy: Dict[str, bool] = {}
        self._next_retry: Dict[str, float] = {}
        self._frame_t = 0

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        if self.cfg.mock:
            return
        import pyrealsense2 as rs

        available = {d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()}
        for spec in self.specs:
            serial = spec.serial or self._pick_unused(available)
            if not serial:
                # Don't abort the whole recorder for one missing camera: leave it
                # unhealthy (read() yields black) so the others still display/record.
                logger.warning("camera '%s': no RealSense serial (available: %s)", spec.key, sorted(available))
                self._healthy[spec.key] = False
                continue
            available.discard(serial)
            self._serials[spec.key] = serial
            try:
                self._open_pipe(spec, serial)
            except Exception as e:  # a bad profile / busy device shouldn't blank every camera
                self._healthy[spec.key] = False
                logger.warning("camera '%s' (%s) could not open: %s", spec.key, serial, e)

    def _supported_color_fps(self, serial: str, spec: CameraSpec) -> list:
        """Color fps values the device actually offers at (width, height) in rgb8."""
        import pyrealsense2 as rs

        fps = set()
        for dev in rs.context().query_devices():
            if dev.get_info(rs.camera_info.serial_number) != serial:
                continue
            for sensor in dev.query_sensors():
                for p in sensor.get_stream_profiles():
                    try:
                        vp = p.as_video_stream_profile()
                        if (
                            p.stream_type() == rs.stream.color
                            and p.format() == rs.format.rgb8
                            and vp.width() == spec.width
                            and vp.height() == spec.height
                        ):
                            fps.add(p.fps())
                    except Exception:
                        pass
        return sorted(fps)

    def _open_pipe(self, spec: CameraSpec, serial: str) -> None:
        import pyrealsense2 as rs

        # Many RealSense models (D405/D455) cap 640x480 color at 30 fps, so a 60 fps
        # request fails with "Couldn't resolve requests". Fall back to the highest
        # supported fps <= the requested one instead of blanking the camera.
        fps = spec.fps
        supported = self._supported_color_fps(serial, spec)
        if supported and spec.fps not in supported:
            usable = [f for f in supported if f <= spec.fps] or supported
            fps = max(usable)
            logger.info("camera '%s': %d fps unsupported; using %d fps (available: %s)", spec.key, spec.fps, fps, supported)

        pipe = rs.pipeline()
        rs_cfg = rs.config()
        rs_cfg.enable_device(serial)
        rs_cfg.enable_stream(rs.stream.color, spec.width, spec.height, rs.format.rgb8, fps)
        pipe.start(rs_cfg)
        self._pipelines[spec.key] = pipe
        self._healthy[spec.key] = True

    def read(self) -> Dict[str, np.ndarray]:
        """Return {key: HxWx3 uint8 RGB}. On a camera fault, returns the last good
        frame (or black), marks the camera unhealthy, and retries reconnection."""
        if self.cfg.mock:
            return self._mock_frames()

        out: Dict[str, np.ndarray] = {}
        for spec in self.specs:
            try:
                pipe = self._pipelines.get(spec.key)
                if pipe is None:
                    raise RuntimeError("pipeline not open")
                frames = pipe.wait_for_frames(timeout_ms=1000)
                img = np.asanyarray(frames.get_color_frame().get_data())  # HxWx3 uint8 (rgb8)
                self._last[spec.key] = img
                self._healthy[spec.key] = True
                out[spec.key] = img
            except Exception:
                self._healthy[spec.key] = False
                self._try_reconnect(spec)
                out[spec.key] = self._last.get(spec.key, np.zeros((spec.height, spec.width, 3), np.uint8))
        return out

    def _try_reconnect(self, spec: CameraSpec) -> None:
        """Throttled best-effort reconnection for a faulted camera."""
        import time

        now = time.monotonic()
        if now < self._next_retry.get(spec.key, 0.0):
            return
        self._next_retry[spec.key] = now + 2.0
        try:
            old = self._pipelines.pop(spec.key, None)
            if old is not None:
                try:
                    old.stop()
                except Exception:
                    pass
            self._open_pipe(spec, self._serials.get(spec.key, spec.serial))
            logger.info("camera '%s' reconnected", spec.key)
        except Exception:
            pass  # stay unhealthy; will retry on the next interval

    @property
    def healthy(self) -> bool:
        """True iff every camera delivered a frame on the latest read (always True in mock)."""
        return all(self._healthy.values()) if self._healthy else True

    def stop(self) -> None:
        for pipe in self._pipelines.values():
            try:
                pipe.stop()
            except Exception:
                pass
        self._pipelines.clear()

    @property
    def image_keys(self) -> List[str]:
        return [s.key for s in self.specs]

    def shape_of(self, key: str) -> tuple:
        spec = next(s for s in self.specs if s.key == key)
        return (spec.height, spec.width, 3)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _pick_unused(available: set) -> str:
        return next(iter(sorted(available)), "")

    def _mock_frames(self) -> Dict[str, np.ndarray]:
        self._frame_t += 1
        out = {}
        for spec in self.specs:
            img = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
            x = (self._frame_t * 4) % spec.width
            img[:, max(0, x - 8) : x + 8, :] = 200  # a moving bar so frames differ
            out[spec.key] = img
        return out


def _list_devices() -> None:
    import pyrealsense2 as rs

    devs = rs.context().query_devices()
    if len(devs) == 0:
        print("No RealSense devices found.")
        return
    for d in devs:
        print(f"{d.get_info(rs.camera_info.name):24s}  serial={d.get_info(rs.camera_info.serial_number)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RealSense helper")
    p.add_argument("--list", action="store_true", help="list connected RealSense serials")
    args = p.parse_args()
    if args.list:
        _list_devices()
