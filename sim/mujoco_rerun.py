#!/usr/bin/env python3
# Minimal LeRobot → MuJoCo → Rerun playback

import argparse, time
from pathlib import Path
import numpy as np
import torch
import rerun as rr
import mujoco

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

PATH_MJCF = 'assets/station_mjcf/station.xml'

def to_hwc_uint8_numpy(x: torch.Tensor):
    # CHW float32 [0,1] → HWC uint8
    return (x.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

def load_mj(mjcf: Path):
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    # model.cam_fovy[0] = 1.0
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    return model, data, renderer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episode-index", type=int, required=True)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--realtime", type=int, default=0)
    ap.add_argument("--scale-actions", type=int, default=1, help="1: [-1,1] → ctrlrange")
    ap.add_argument("--show-dataset", type=int, default=1, help="1: also show dataset camera frames")
    ap.add_argument("--camera", type=str, default=None, help="MuJoCo camera name (defaults to free)")
    args = ap.parse_args()

    # Data
    ds = LeRobotDataset(args.repo_id, episodes=[args.episode_index], root=args.root)
    from_idx = ds.meta.episodes["dataset_from_index"][args.episode_index]
    to_idx   = ds.meta.episodes["dataset_to_index"][args.episode_index]
    fps = float(ds.meta.fps)

    # Viz
    rr.init(f"{ds.repo_id}/episode_{args.episode_index}", spawn=True)

    # Sim
    model, data, renderer = load_mj(PATH_MJCF)
    sim_dt = model.opt.timestep

    # Playback
    first_ts = None
    t0_wall = time.time()
    for k in range(from_idx, to_idx):
        sample = ds[k]
        ts = float(sample["timestamp"].item())
        if first_ts is None:
            first_ts = ts
            t0_wall = time.time()

        # Rerun time
        rr.set_time_seconds("timestamp", ts)

        # Dataset frame(s) (optional)
        if args.show_dataset:
            for cam_key in ds.meta.camera_keys:
                rr.log(f"dataset/{cam_key}", rr.Image(to_hwc_uint8_numpy(sample[cam_key])))

        # Duration until next frame
        if k + 1 < to_idx:
            ts_next = float(ds[k + 1]["timestamp"].item())
        else:
            ts_next = ts + 1.0 / max(1.0, fps)
        n_steps = max(1, int(round((ts_next - ts) / sim_dt)))

        # Step sim
        data.ctrl[:] = np.concatenate((sample[ACTION].numpy()[7:], sample[ACTION].numpy()[:7]))
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

        # Render & log
        renderer.update_scene(data, camera=0)
        rr.log("sim", rr.Image(renderer.render()))


if __name__ == "__main__":
    main()
