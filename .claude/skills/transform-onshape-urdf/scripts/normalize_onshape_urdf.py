#!/usr/bin/env python3
"""Normalize a raw ONShape-exported YAM-family URDF -- **stage 1** of the alignment pipeline.

A fresh ONShape export (``<robot name="urdf_top_assembly">``) is messy: every physical link is
emitted two or three times (``base`` / ``base_1``, ``link2`` / ``link2_1`` ...), a synthetic
``root`` link sits above ``base`` via a ``root -> base`` joint (``dof_joint0``) that re-orients the
whole arm, mesh paths use ``package://.../meshes/Name.stl`` (mixed case), the joints are named
``dof_joint0 .. dof_jointN``, and a web of fixed ``分组`` / ``紧固`` joints wires the redundant
copies together.

This stage does the **deterministic** cleanup only -- everything that does not require human
judgment -- and then stops so the result can be visually inspected before any heading correction:

    1. Keep only the real actuated chain (the ``dof_joint*`` joints) and the links it references;
       drop the ``root`` duplicates and every ``分组`` / ``紧固`` joint. Rename the survivors to
       canonical names by stripping a trailing ``_<n>`` (``base_1`` -> ``base`` ...).
    2. Rewrite every mesh reference to ``assets/<lowercase-basename>.stl`` matching the real file.
    3. Remove the synthetic root, baking its rotation ``R0`` into the ``base`` visual, collision, AND
       inertial origins (the same ``R0`` -> mesh, collision & inertia stay aligned) and the ``base -> link1`` joint
       origin (which rigidly rotates the whole arm). This preserves the ONShape assembled pose, so
       the model can be inspected in its natural orientation.
    4. Rename ``dof_jointN`` -> ``jointN`` (``dof_joint0`` is the root joint and is already gone).
    5. Set the robot name and insert an ``onshape-normalize:`` marker comment.

**No heading guess.** The old monolith baked a default 90 deg yaw here; that is a human decision
and now lives in stage 3 (``apply_urdf_heading.py``), gated by a visual-inspection checkpoint
(``view_urdf.py``).

Root-removal guard: the synthetic root is the *parent of the joint whose child is ``base``*. If no
joint has child ``base``, the root is already gone and this step is skipped -- it never re-bakes,
and it never mistakes a rootless ``base`` for a synthetic root (which would delete ``base`` +
``joint1``).

The transform is idempotent: each step is a no-op on an already-normalized tree, and a structural
classification (``root`` link, ``dof_joint*`` names, ``package://`` paths, ``分组``/``紧固`` joints,
or ``*_<n>`` duplicate links) decides whether work is needed.

RPY convention (URDF): ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

Usage:
    python .claude/skills/transform-onshape-urdf/scripts/normalize_onshape_urdf.py <model.urdf> [--assets-dir assets] [--name NAME]
        [--dry-run] [--force]
"""
from __future__ import annotations

import copy
import datetime
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Annotated

import tyro
from urdf_align_lib import (
    ARM_JOINT_RE,
    apply_rotation_to_base,
    joint_by_child,
    joint_by_parent,
    joints,
    link_by_name,
    links,
    parse_triplet,
    read_preamble_comments,
    rpy_to_matrix,
    serialize,
    verify_base_mesh_inertia_aligned,
    verify_meshes,
    verify_rigidity,
)

MARKER_KEY = "onshape-normalize:"
SCRIPT_VERSION = "v1"
DOF_JOINT_RE = re.compile(r"^dof_joint(\d+)$")
SUFFIX_RE = re.compile(r"_\d+$")


# --------------------------------------------------------------------------- steps
def step_clean_dof_tree(robot: ET.Element) -> None:
    """Keep only the ``dof_joint*`` chain + its links; drop everything else; strip ``_<n>``."""
    dof_joints = [j for j in joints(robot) if DOF_JOINT_RE.match(j.get("name", ""))]
    if not dof_joints:
        return  # already clean

    keep_links = set()
    for j in dof_joints:
        keep_links.add(j.find("parent").get("link"))
        keep_links.add(j.find("child").get("link"))

    # drop non-dof joints (分组 / 紧固 structural duplicates)
    for j in joints(robot):
        if not DOF_JOINT_RE.match(j.get("name", "")):
            robot.remove(j)
    # drop links not on the actuated chain
    for lk in links(robot):
        if lk.get("name") not in keep_links:
            robot.remove(lk)

    # canonicalize names (base_1 -> base, link2_1 -> link2, ...); update joint refs
    rename = {}
    for name in keep_links:
        canonical = SUFFIX_RE.sub("", name)
        if canonical != name:
            rename[name] = canonical
    canonical_set = {SUFFIX_RE.sub("", n) for n in keep_links}
    if len(canonical_set) != len(keep_links):
        raise ValueError(f"canonical-name collision among chain links: {sorted(keep_links)}")
    for lk in links(robot):
        if lk.get("name") in rename:
            lk.set("name", rename[lk.get("name")])
    for j in joints(robot):
        for tag in ("parent", "child"):
            el = j.find(tag)
            if el is not None and el.get("link") in rename:
                el.set("link", rename[el.get("link")])


def step_fix_meshes(robot: ET.Element, assets_dir_abs: str, assets_prefix: str) -> None:
    on_disk = {}
    if os.path.isdir(assets_dir_abs):
        on_disk = {f.lower(): f for f in os.listdir(assets_dir_abs)}
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename")
        if fn is None:
            continue
        base = os.path.basename(fn)          # e.g. "Gripper.stl" or "link5_1.stl"
        stem, ext = os.path.splitext(base)
        # prefer an exact (case-insensitive) file on disk; else drop the ONShape instance
        # suffix ("link5_1" -> "link5") and match that; else fall back to lowercase.
        low = base.lower()
        if low in on_disk:
            actual = on_disk[low]
        else:
            stripped = (SUFFIX_RE.sub("", stem) + ext).lower()
            actual = on_disk.get(stripped, stripped)
        mesh.set("filename", f"{assets_prefix}/{actual}")


def step_remove_root(robot: ET.Element, base_name: str = "base") -> tuple | None:
    """Remove the synthetic root and bake its ``R0`` into the base. Returns (R0, root_rpy_str) or
    ``None`` if there is no synthetic root (no joint has child ``base`` -> already rootless).

    The synthetic root is the *parent* of the joint whose child is ``base``. Keying off the child
    (never off "which link is never a child") is what makes re-running on an already-rootless tree
    safe -- otherwise ``base`` itself looks like a root and gets deleted along with ``joint1``.
    """
    root_joint = joint_by_child(robot, base_name)
    if root_joint is None:
        return None  # already rootless
    root_name = root_joint.find("parent").get("link")
    origin = root_joint.find("origin")
    rpy_str = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    r0 = rpy_to_matrix(parse_triplet(rpy_str))
    robot.remove(root_joint)
    root_link = link_by_name(robot, root_name)
    if root_link is not None:
        robot.remove(root_link)
    apply_rotation_to_base(robot, base_name, r0)  # bake R0: base visual+collision+inertial + base->link1
    return r0, rpy_str


def step_rename_joints(robot: ET.Element) -> None:
    for j in joints(robot):
        m = DOF_JOINT_RE.match(j.get("name", ""))
        if m:
            j.set("name", f"joint{int(m.group(1))}")


# --------------------------------------------------------------------------- driver
def classify(raw: str, robot: ET.Element) -> tuple[bool, bool, str]:
    """Return (has_marker, is_raw, raw_reason).

    ``is_raw`` is a purely structural check for a still-untransformed export. Signals: a synthetic
    ``root`` link, ``dof_joint*`` names, ``package://`` mesh paths, non-arm (``分组``/``紧固``)
    joints, or ``*_<n>`` duplicate links. A non-raw file has nothing to normalize (each step would
    be a no-op), so it is skipped."""
    has_marker = MARKER_KEY in raw
    reasons = []
    if any(lk.get("name") == "root" for lk in links(robot)):
        reasons.append("root link")
    if any(j.get("name", "").startswith("dof_joint") for j in joints(robot)):
        reasons.append("dof_joint* names")
    if "package://" in raw:
        reasons.append("package:// paths")
    if any(not ARM_JOINT_RE.match(j.get("name", "")) for j in joints(robot)):
        reasons.append("structural (non-arm) joints")  # 分组 / 紧固 / fastened duplicates
    if any(SUFFIX_RE.search(lk.get("name", "")) for lk in links(robot)):
        reasons.append("duplicate *_<n> links")
    return has_marker, bool(reasons), ", ".join(reasons)


def build_marker(root_joint_rpy: str | None) -> str:
    today = datetime.date.today().isoformat()
    root_note = f'root_joint_rpy="{root_joint_rpy}"' if root_joint_rpy is not None else "root=already-removed"
    return (
        f"<!-- {MARKER_KEY} {SCRIPT_VERSION} applied {today}; "
        f"{root_note}; heading=pending (run apply_urdf_heading.py); "
        f"tool=.claude/skills/transform-onshape-urdf/scripts/normalize_onshape_urdf.py -->"
    )


def main(
    urdf: Annotated[str, tyro.conf.Positional],
    assets_dir: str = "assets",
    name: str | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """Normalize a raw ONShape URDF in place (verify unless --dry-run).

    Args:
        urdf: path to the URDF to normalize (edited in place).
        assets_dir: mesh dir relative to the URDF.
        name: robot name (default: URDF's parent dir name).
        dry_run: report and verify but do not write.
        force: normalize even if a marker is present.
    """
    urdf_path = os.path.abspath(urdf)
    urdf_dir = os.path.dirname(urdf_path)
    model_name = name or os.path.basename(urdf_dir)
    with open(urdf_path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    robot = ET.fromstring(raw)

    has_marker, is_raw, raw_reason = classify(raw, robot)
    if not is_raw:
        note = "marker present" if has_marker else "no raw signatures"
        print(f"[skip] {urdf}: already normalized ({note}).")
        return 0
    if has_marker and not force:
        print(f"[skip] {urdf}: marker present (use --force to re-normalize this raw file).")
        return 0
    print(f"[normalize] {urdf}: raw signatures: {raw_reason}")

    step_clean_dof_tree(robot)
    step_fix_meshes(robot, os.path.join(urdf_dir, assets_dir), assets_dir)
    if robot.get("name") in (None, "urdf_top_assembly"):
        robot.set("name", model_name)
    baseline = copy.deepcopy(robot)  # rigidity baseline: deduped, pre-root-removal
    removed = step_remove_root(robot)
    step_rename_joints(robot)
    root_joint_rpy = removed[1] if removed is not None else None

    # ---- verify ----
    problems, world_axes = verify_rigidity(baseline, robot, "base")
    problems += verify_meshes(robot, urdf_dir)
    problems += verify_base_mesh_inertia_aligned(robot, "base")
    if removed is not None:
        print("  removed synthetic root | R0 rpy:", root_joint_rpy)
    else:
        print("  synthetic root already removed (no joint has child 'base') -- base pose left as-is")
    print("  joint world axes:", {k: [round(v, 4) for v in vec] for k, vec in sorted(world_axes.items())})
    if problems:
        print("  VERIFICATION FAILED:")
        for p in problems:
            print("   -", p)
        print("  refusing to write.")
        return 1
    print("  verification OK (arm rigid; meshes resolve; base mesh & inertia aligned).")
    print("  next: inspect with `python .claude/skills/transform-onshape-urdf/scripts/view_urdf.py", urdf,
          "`, then `python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py` if a heading correction is needed.")

    marker = build_marker(root_joint_rpy)
    preamble = read_preamble_comments(raw)
    out = "\n".join(preamble + [marker, serialize(robot)]) + "\n"

    if dry_run:
        print("  [dry-run] not writing. Resulting base link + joint1:")
        print(serialize(link_by_name(robot, "base"), indent=1))
        j1 = joint_by_parent(robot, "base")
        if j1 is not None:
            print(serialize(j1, indent=1))
        return 0
    with open(urdf_path, "w", encoding="utf-8") as fh:
        fh.write(out)
    print(f"  wrote {urdf}")
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main, description=__doc__))
