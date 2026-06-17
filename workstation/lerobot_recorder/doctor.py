"""Dataset doctor — quick health/stats for a collected dataset.

Summarizes the ``outcomes.jsonl`` sidecar (episode counts, success rate, per-task
breakdown) and — if ``lerobot`` is installed and a ``--repo-id`` is given —
validates the LeRobot dataset (features, episode/frame counts).

    python -m workstation.lerobot_recorder.doctor --root ~/lerobot_data
    python -m workstation.lerobot_recorder.doctor --root ~/lerobot_data --repo-id user/yam_pick
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict


def summarize_outcomes(root: str) -> Dict:
    """Aggregate the outcomes.jsonl sidecar into counts + success rate + per-task stats."""
    path = os.path.join(os.path.expanduser(root), "outcomes.jsonl")
    if not os.path.exists(path):
        return {"exists": False, "path": path, "episodes": 0, "outcomes": {}, "by_task": {}, "success_rate": None}

    entries = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    outcomes = Counter(e.get("outcome") for e in entries)
    succ, fail = outcomes.get("success", 0), outcomes.get("fail", 0)
    by_task: Dict[str, Counter] = defaultdict(Counter)
    for e in entries:
        by_task[e.get("task", "?")][e.get("outcome")] += 1
    return {
        "exists": True,
        "path": path,
        "episodes": len(entries),
        "frames_total": sum(int(e.get("frames", 0)) for e in entries),
        "outcomes": dict(outcomes),
        "success_rate": (succ / (succ + fail)) if (succ + fail) else None,
        "by_task": {k: dict(v) for k, v in by_task.items()},
    }


def outcomes_by_episode(root: str) -> Dict[int, str]:
    """Map episode_index -> outcome from the sidecar (for annotating episode lists)."""
    path = os.path.join(os.path.expanduser(root), "outcomes.jsonl")
    out: Dict[int, str] = {}
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("episode") is not None and e.get("outcome") is not None:
                out[int(e["episode"])] = str(e["outcome"])
    return out


def _print_summary(s: Dict) -> None:
    if not s["exists"]:
        print(f"[doctor] no outcomes.jsonl at {s['path']} (nothing recorded yet?)")
        return
    rate = "n/a" if s["success_rate"] is None else f"{100 * s['success_rate']:.0f}%"
    print(f"[doctor] {s['path']}")
    print(f"  episodes: {s['episodes']}   frames: {s['frames_total']}   success rate: {rate}")
    print(f"  outcomes: {s['outcomes']}")
    for task, counts in s["by_task"].items():
        print(f"    task {task!r}: {counts}")


def _validate_lerobot(repo_id: str, root: str) -> None:
    try:
        from lerobot.datasets import LeRobotDataset
    except Exception as e:
        print(f"[doctor] lerobot not available, skipping dataset validation ({e})")
        return
    try:
        ds = LeRobotDataset(repo_id, root=os.path.expanduser(root))
        feats = sorted(getattr(ds, "features", {}))
        n_ep = getattr(ds, "num_episodes", getattr(getattr(ds, "meta", None), "total_episodes", "?"))
        n_fr = getattr(ds, "num_frames", getattr(getattr(ds, "meta", None), "total_frames", "?"))
        print(f"[doctor] LeRobot dataset OK: {n_ep} episodes, {n_fr} frames")
        print(f"  features: {feats}")
    except Exception as e:
        print(f"[doctor] FAILED to open LeRobot dataset: {e}")


def main() -> None:
    p = argparse.ArgumentParser(description="Dataset health / stats")
    p.add_argument("--root", default="~/lerobot_data")
    p.add_argument("--repo-id", default=None, help="also validate the LeRobot dataset (needs lerobot)")
    args = p.parse_args()
    _print_summary(summarize_outcomes(args.root))
    if args.repo_id:
        _validate_lerobot(args.repo_id, args.root)


if __name__ == "__main__":
    main()
