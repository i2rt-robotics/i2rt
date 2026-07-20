#!/usr/bin/env python3
"""Sync per-arm gripper mount data to a regenerated arm MJCF -- **stage 5**.

When the arm MJCF is regenerated from a re-aligned URDF (``urdf_to_arm_mjcf.py``), the arm's
terminal mount frame -- ``link6``'s ``pos``/``quat`` and ``joint6``'s ``axis`` -- can change. At
runtime, ``combine_arm_and_gripper_xml`` (``i2rt/robots/utils.py``) locates the arm's deepest body
and **overwrites** exactly those three fields from ``last_joint_mount.<arm>`` in the *gripper*
config before attaching the gripper. That mount block is arm data, duplicated across every gripper
config, so a stale copy silently misplaces the gripper for that arm.

This script re-reads the mount straight from the regenerated ``<arm>.xml`` -- using the very same
``_find_deepest_body`` that composition uses, so the body selected here is guaranteed to be the one
overwritten at runtime -- and writes ``pos``/``quat``/``axis`` into ``last_joint_mount.<arm>`` for
every gripper config that has such a block. It touches only the target arm's sub-block, preserving
comments, key order, and the other arms' data (``ruamel.yaml`` is not available, so it edits the
three scalar lines textually rather than round-tripping the YAML).

A block already matching the MJCF within 1e-9 is left untouched, so the tool is idempotent and
produces a clean diff (only genuinely-changed arms are rewritten).

Usage:
    python .claude/skills/align-urdf-mjcf/scripts/sync_gripper_mounts.py <arm>.xml [--arm NAME] [--config-dir DIR] [--dry-run]
"""
from __future__ import annotations

import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Annotated

import numpy as np
import tyro

from i2rt.robots.utils import _find_deepest_body  # the exact body composition overwrites

DEFAULT_CONFIG_DIR = os.path.join("i2rt", "robots", "config")


def read_arm_mount(xml_path: str) -> tuple[str, str, str, str]:
    """Return (arm_model_name, pos, quat, axis) from the arm MJCF's terminal mount body."""
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{xml_path}: no <worldbody>")
    tip = _find_deepest_body(worldbody)
    pos = tip.get("pos")
    quat = tip.get("quat")
    joint = tip.find("joint")
    if pos is None or quat is None or joint is None or joint.get("axis") is None:
        raise ValueError(f"{xml_path}: terminal body {tip.get('name')!r} lacks pos/quat/joint-axis")
    return root.get("model"), pos, quat, joint.get("axis")


def _triplet(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.split()])


def _values_match(old: str, new: str, kind: str) -> bool:
    """True if the config value already equals the MJCF value within 1e-9. ``quat`` matches up to
    sign (q and -q are the same rotation); ``pos``/``axis`` must match directly (axis sign encodes
    motor direction, so a flipped axis is a real change)."""
    a, b = _triplet(old), _triplet(new)
    if a.shape != b.shape:
        return False
    if kind == "quat":
        na, nb = a / (np.linalg.norm(a) or 1.0), b / (np.linalg.norm(b) or 1.0)
        return abs(abs(float(np.dot(na, nb))) - 1.0) < 1e-9
    return bool(np.allclose(a, b, atol=1e-9))


def sync_config(path: str, arm: str, pos: str, quat: str, axis: str, dry_run: bool) -> tuple[str, bool]:
    """Update ``last_joint_mount.<arm>`` in one config. Returns (status, changed)."""
    new_vals = {"pos": pos, "quat": quat, "axis": axis}
    lines = open(path, encoding="utf-8").read().splitlines()
    out, olds = [], {}
    in_ljm = in_arm = found_arm = False
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if in_ljm and stripped and indent == 0:  # a top-level key/comment ends the mount block
            in_ljm = in_arm = False
        if not in_ljm and re.match(r"^last_joint_mount:\s*$", line):
            in_ljm = True
            out.append(line)
            continue
        if in_ljm:
            m = re.match(r"^(\s+)(\S+):\s*$", line)  # an arm sub-key (2-space indent, no value)
            if m and len(m.group(1)) == 2:
                in_arm = m.group(2) == arm
                found_arm = found_arm or in_arm
                out.append(line)
                continue
            if in_arm:
                m2 = re.match(r"^(\s{4})(pos|quat|axis):\s*(.*)$", line)
                if m2:
                    key = m2.group(2)
                    olds[key] = m2.group(3).strip().strip('"')
                    out.append(f'{m2.group(1)}{key}: "{new_vals[key]}"')
                    continue
        out.append(line)

    if not found_arm:
        return f"MISSING arm sub-key '{arm}'", False
    absent = [k for k in ("pos", "quat", "axis") if k not in olds]
    if absent:
        # A key never matched the 4-space ``pos|quat|axis`` pattern -- absent from the block or
        # mis-indented. Rewriting now would leave that field stale/missing while reporting
        # success, so refuse loudly instead (composition would silently misplace the gripper).
        return f"ERROR malformed '{arm}' block: missing/mis-indented {', '.join(absent)}", False
    changed = not all(_values_match(olds[k], new_vals[k], k) for k in ("pos", "quat", "axis"))
    if changed and not dry_run:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
    return ("would change" if dry_run else "changed") if changed else "unchanged", changed


def main(
    arm_xml: Annotated[str, tyro.conf.Positional],
    arm: str | None = None,
    config_dir: str = DEFAULT_CONFIG_DIR,
    dry_run: bool = False,
) -> int:
    """Sync last_joint_mount.<arm> in every gripper config from a regenerated arm MJCF.

    Args:
        arm_xml: regenerated arm MJCF (e.g. i2rt/robot_models/arm/yam_ultra/yam_ultra.xml).
        arm: arm sub-key to update (default: MJCF model name / parent dir).
        config_dir: gripper config dir.
        dry_run: report old->new but do not write.
    """
    model_name, pos, quat, axis = read_arm_mount(arm_xml)
    arm = arm or model_name or os.path.basename(os.path.dirname(os.path.abspath(arm_xml)))
    print(f"[sync-mounts] arm={arm!r} from {arm_xml}")
    print(f"  pos  = {pos}")
    print(f"  quat = {quat}")
    print(f"  axis = {axis}")

    configs = sorted(
        os.path.join(config_dir, f)
        for f in os.listdir(config_dir)
        if f.endswith(".yml") and "last_joint_mount:" in open(os.path.join(config_dir, f), encoding="utf-8").read()
    )
    if not configs:
        print(f"  [error] no gripper configs with a last_joint_mount block in {config_dir}")
        return 1

    errors = 0
    for path in configs:
        status, _ = sync_config(path, arm, pos, quat, axis, dry_run)
        print(f"  {os.path.basename(path):24s} {status}")
        if status.startswith(("MISSING", "ERROR")):
            errors += 1
    if errors:
        print(f"  [error] {errors} config(s) missing/malformed for '{arm}' -- composition would use a stale/missing mount at runtime.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main, description=__doc__))
