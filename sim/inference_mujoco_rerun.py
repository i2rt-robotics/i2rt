#!/usr/bin/env python3
# Minimal LeRobot → MuJoCo → Rerun playback (with action-vs-inference logging)

import argparse, time
from pathlib import Path
import numpy as np
import torch
import rerun as rr
import mujoco
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import *
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

PATH_MJCF = 'assets/station_mjcf/station.xml'

def to_hwc_uint8_numpy(x: torch.Tensor):
    # CHW float32 [0,1] → HWC uint8
    return (x.clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

def load_mj(mjcf: Path):
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    return model, data, renderer

def log_actions_two_plots(u_pred: np.ndarray, u_gt: np.ndarray):
    u_pred = np.asarray(u_pred, dtype=np.float32).reshape(-1)
    u_gt   = np.asarray(u_gt,   dtype=np.float32).reshape(-1)
    n = min(len(u_pred), len(u_gt))
    for i in range(n):
        rr.log(f"gt/j{i}",   rr.Scalar(float(u_gt[i])))
        rr.log(f"pred/j{i}", rr.Scalar(float(u_pred[i])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--model-path", required=True)
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

    # Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inference = ACTPolicy.from_pretrained(args.model_path).to(device).eval()

    dataset_id = "zetanschy/train_set1"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(
        model_inference.config, dataset_stats=dataset_metadata.stats
    )

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

        # Inference
        a = time.time()
        with torch.no_grad():
            # print("sample:", sample)
            proc   = preprocess(sample)                           # dict with normalized tensors
            # print("preprocess:", proc)
            proc   = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in proc.items()}
            action = model_inference.select_action(proc)          # model output (normalized space)
            action = postprocess(action)                          # back to dataset space
            action = make_robot_action(action, dataset_metadata.features)
        print(time.time()-a)
        
        # Build arrays
        u_pred = np.array(list(action.values()), dtype=np.float32)
        u_gt   = sample[ACTION].detach().cpu().numpy().astype(np.float32).reshape(-1)

        # Log comparison to Rerun
        log_actions_two_plots(u_pred, u_gt)

        # Apply to sim (note: control order matches your original concatenation)
        data.ctrl[:] = np.concatenate((u_pred[7:], u_pred[:7]))
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

        # Render & log sim frame
        renderer.update_scene(data, camera=0)
        rr.log("sim", rr.Image(renderer.render()))

if __name__ == "__main__":
    main()
