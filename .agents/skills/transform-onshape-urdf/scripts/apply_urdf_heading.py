#!/usr/bin/env python3
"""Apply a user-chosen heading rotation to a normalized YAM-family URDF -- **stage 3**.

Stage 1 (``normalize_onshape_urdf.py``) leaves the arm in its natural ONShape assembled pose. The
question "which way should the base face in the world?" is a human decision, so after inspecting
the normalized model (``view_urdf.py``) the user states an axis + angle (e.g. "90 deg about world
Z"), and this stage bakes it in:

    * Left-multiply the requested rotation ``R`` into the base **visual** origin, the base
      **inertial** origin (the *same* ``R`` -- the mesh and inertia never diverge), and the
      ``base -> link1`` joint origin (which rigidly rotates the whole arm subtree). Joints 2..N and
      the inertia tensor components are untouched.

Because ``transform_origin`` left-multiplies, this composes with stage 1's baked ``R0`` exactly as
``R @ R0`` -- identical to what the old one-shot ``transform_onshape_urdf.py`` produced with its
``--heading-deg``. The split just moves ``R`` behind a visual-inspection checkpoint.

This operation is **not idempotent** -- running it twice rotates twice -- so it previews by default
and only writes with ``--apply``. Each application appends an ``onshape-heading:`` marker, so a
double application is visible in the file header.

RPY convention (URDF): ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

Usage:
    # preview a 90 deg yaw about world Z (writes nothing):
    python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py <model.urdf> --deg 90
    # commit it:
    python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py <model.urdf> --deg 90 --apply
    # arbitrary rotation:
    python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py <model.urdf> --rpy "0 0 1.5708" --apply
"""
from __future__ import annotations

import copy
import datetime
import os
import sys
import xml.etree.ElementTree as ET
from typing import Annotated, Literal

import tyro
from urdf_align_lib import (
    apply_rotation_to_base,
    heading_matrix,
    joint_by_child,
    joint_by_parent,
    link_by_name,
    parse_triplet,
    read_preamble_comments,
    rpy_to_matrix,
    serialize,
    verify_base_mesh_inertia_aligned,
    verify_meshes,
    verify_rigidity,
)

MARKER_KEY = "onshape-heading:"
SCRIPT_VERSION = "v1"


def build_marker(desc: str) -> str:
    today = datetime.date.today().isoformat()
    return (
        f"<!-- {MARKER_KEY} {SCRIPT_VERSION} applied {today}; {desc}; "
        f"tool=.claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py -->"
    )


def main(
    urdf: Annotated[str, tyro.conf.Positional],
    axis: Literal["x", "y", "z"] = "z",
    deg: float | None = None,
    rpy: str | None = None,
    apply: bool = False,
) -> int:
    """Apply a heading rotation to a normalized URDF (preview unless --apply).

    Args:
        urdf: path to a normalized URDF (edited in place only with --apply).
        axis: heading axis.
        deg: heading rotation in degrees about --axis.
        rpy: arbitrary rotation "roll pitch yaw" (rad); overrides --axis/--deg.
        apply: write the change (default: preview only).
    """
    if rpy is not None:
        if len(rpy.split()) != 3:
            raise SystemExit(f'--rpy expects 3 values "roll pitch yaw" (rad), got {rpy!r}')
        rot = rpy_to_matrix(parse_triplet(rpy))
        desc = f'rpy="{rpy}"'
    elif deg is not None:
        rot = heading_matrix(axis, deg)
        desc = f"heading={deg:g}deg@{axis}"
    else:
        raise SystemExit('specify a rotation: --deg N [--axis x|y|z], or --rpy "r p y"')

    urdf_path = os.path.abspath(urdf)
    urdf_dir = os.path.dirname(urdf_path)
    with open(urdf_path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    robot = ET.fromstring(raw)

    # Precondition: base must be the tree root (normalize already removed the synthetic root).
    if joint_by_child(robot, "base") is not None:
        print(f"[error] {urdf}: 'base' is still a joint child -- run normalize_onshape_urdf.py first.")
        return 1
    if link_by_name(robot, "base") is None:
        print(f"[error] {urdf}: no 'base' link found.")
        return 1
    if joint_by_parent(robot, "base") is None:
        print(f"[error] {urdf}: 'base' has no child joint (base->link1) to carry the rotation "
              "-- not a normalized arm URDF.")
        return 1

    baseline = copy.deepcopy(robot)  # rigidity baseline: pre-heading
    apply_rotation_to_base(robot, "base", rot)

    # ---- verify ----
    problems, world_axes = verify_rigidity(baseline, robot, "base")
    problems += verify_meshes(robot, urdf_dir)
    problems += verify_base_mesh_inertia_aligned(robot, "base")
    print(f"[apply-heading] {urdf}: {desc}")
    print("  joint world axes:", {k: [round(v, 4) for v in vec] for k, vec in sorted(world_axes.items())})
    if problems:
        print("  VERIFICATION FAILED:")
        for p in problems:
            print("   -", p)
        print("  refusing to write.")
        return 1
    print("  verification OK (arm rigid; meshes resolve; base mesh & inertia aligned).")

    marker = build_marker(desc)
    preamble = read_preamble_comments(raw)  # keeps the ONShape + onshape-normalize markers
    out = "\n".join(preamble + [marker, serialize(robot)]) + "\n"

    if not apply:
        print("  [preview] not writing (pass --apply to commit). Resulting base link + joint1:")
        print(serialize(link_by_name(robot, "base"), indent=1))
        j1 = joint_by_parent(robot, "base")
        if j1 is not None:
            print(serialize(j1, indent=1))
        return 0
    if MARKER_KEY in raw:
        print("  NOTE: an onshape-heading marker is already present -- this rotation STACKS on the previous one.")
    with open(urdf_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"  wrote {urdf}")
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main, description=__doc__))
