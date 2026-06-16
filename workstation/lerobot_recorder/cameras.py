"""RealSense camera manager — opens the configured cameras, grabs latest frames.

Each :class:`~workstation.lerobot_recorder.config.CameraSpec` maps a serial to a
dataset image key (``wrist_left`` / ``wrist_right`` / ``agentview``). ``read()``
returns the most recent color frame per key as an ``HxWx3 uint8`` array.

``mock=True`` returns synthetic frames so the pipeline runs without hardware.

CLI: ``python -m workstation.lerobot_recorder.cameras --list`` prints connected
RealSense serials so you can fill them into the config.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from workstation.lerobot_recorder.config import CameraSpec, RecorderConfig


class CameraManager:
    def __init__(self, cfg: RecorderConfig) -> None:
        self.cfg = cfg
        self.specs: List[CameraSpec] = cfg.cameras
        self._pipelines: Dict[str, object] = {}
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
                raise RuntimeError(f"no RealSense serial for camera '{spec.key}' (available: {sorted(available)})")
            available.discard(serial)
            pipe = rs.pipeline()
            rs_cfg = rs.config()
            rs_cfg.enable_device(serial)
            rs_cfg.enable_stream(rs.stream.color, spec.width, spec.height, rs.format.rgb8, spec.fps)
            pipe.start(rs_cfg)
            self._pipelines[spec.key] = pipe

    def read(self) -> Dict[str, np.ndarray]:
        """Return {key: HxWx3 uint8 RGB} for all cameras (latest frame)."""
        if self.cfg.mock:
            return self._mock_frames()

        out: Dict[str, np.ndarray] = {}
        for spec in self.specs:
            frames = self._pipelines[spec.key].wait_for_frames()
            color = frames.get_color_frame()
            out[spec.key] = np.asanyarray(color.get_data())  # HxWx3 uint8 (rgb8)
        return out

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
        print(
            f"{d.get_info(rs.camera_info.name):24s}  "
            f"serial={d.get_info(rs.camera_info.serial_number)}"
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RealSense helper")
    p.add_argument("--list", action="store_true", help="list connected RealSense serials")
    args = p.parse_args()
    if args.list:
        _list_devices()
