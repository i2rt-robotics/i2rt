#!/usr/bin/env python3
"""Shared helpers for the ONShape→URDF alignment scripts.

Split out of the former monolithic ``transform_onshape_urdf.py`` so that the two stage scripts --
``normalize_onshape_urdf.py`` (mesh names, joint names, remove root, bake ``R0``) and
``apply_urdf_heading.py`` (apply a user-chosen heading rotation) -- share one definition of the
RPY convention, the ``<origin>`` left-multiply, and the FK/verification routines. Two scripts that
disagreed on any of these would silently produce inconsistent geometry, so they live here once.

URDF RPY convention: ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

# A real actuated joint, either raw (``dof_joint3``) or renamed (``joint3``).
ARM_JOINT_RE = re.compile(r"^(?:dof_)?joint\d+$")


# --------------------------------------------------------------------------- math
def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    return R.from_euler("ZYX", [y, p, r]).as_matrix()


def matrix_to_rpy(m: np.ndarray) -> np.ndarray:
    yaw, pitch, roll = R.from_matrix(m).as_euler("ZYX")
    return np.array([roll, pitch, yaw])


def heading_matrix(axis: str, deg: float) -> np.ndarray:
    return R.from_euler(axis.upper(), deg, degrees=True).as_matrix()


def parse_triplet(text: str) -> np.ndarray:
    return np.array([float(v) for v in text.split()])


def fmt_num(x: float) -> str:
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:.12g}"


def fmt_triplet(vec: np.ndarray) -> str:
    return " ".join(fmt_num(v) for v in vec)


def transform_origin(origin_el: ET.Element, rot: np.ndarray) -> None:
    """Left-multiply an ``<origin>``'s pose by ``rot`` (rotation about the frame origin).

    Because this left-multiplies, applying ``R0`` here in one script and ``R_head`` in another
    composes to ``R_head @ R0`` -- which is exactly what the old one-shot transform computed.
    """
    xyz = parse_triplet(origin_el.get("xyz", "0 0 0"))
    rpy = parse_triplet(origin_el.get("rpy", "0 0 0"))
    origin_el.set("xyz", fmt_triplet(rot @ xyz))
    origin_el.set("rpy", fmt_triplet(matrix_to_rpy(rot @ rpy_to_matrix(rpy))))


# --------------------------------------------------------------------------- tree access
def links(robot: ET.Element) -> list[ET.Element]:
    return robot.findall("link")


def joints(robot: ET.Element) -> list[ET.Element]:
    return robot.findall("joint")


def link_by_name(robot: ET.Element, name: str) -> ET.Element | None:
    for lk in links(robot):
        if lk.get("name") == name:
            return lk
    return None


def joint_by_parent(robot: ET.Element, parent: str) -> ET.Element | None:
    for j in joints(robot):
        p = j.find("parent")
        if p is not None and p.get("link") == parent:
            return j
    return None


def joint_by_child(robot: ET.Element, child: str) -> ET.Element | None:
    """The joint whose ``<child>`` is ``child`` (each link is the child of at most one joint)."""
    for j in joints(robot):
        c = j.find("child")
        if c is not None and c.get("link") == child:
            return j
    return None


def root_links(robot: ET.Element) -> list[str]:
    """Links that are never a child of any joint (the tree root(s))."""
    children = {j.find("child").get("link") for j in joints(robot) if j.find("child") is not None}
    return [lk.get("name") for lk in links(robot) if lk.get("name") not in children]


def apply_rotation_to_base(robot: ET.Element, base_name: str, rot: np.ndarray) -> None:
    """Left-multiply ``rot`` into every base geometry origin -- **visual**, **collision**, and
    **inertial** -- and the ``base -> link1`` joint origin.

    The base visual (mesh), collision, and inertial origins all receive the *identical* ``rot`` so
    they never diverge; the ``base -> link1`` joint origin carries the same ``rot``, which rigidly
    rotates the whole arm subtree. A link may declare multiple ``<visual>``/``<collision>`` blocks
    (raw ONShape exports often do -- ``big_yam.urdf`` has 9 collision blocks), so every one is
    rotated, not just the first. Downstream joints are untouched. Inertia tensor components stay in
    the (rotated) inertial frame, so they are not modified. Shared by both the ``R0`` bake
    (normalize) and the heading application (apply_urdf_heading)."""
    base = link_by_name(robot, base_name)
    if base is None:
        raise ValueError(f"no base link named {base_name!r}")
    for tag in ("visual", "collision", "inertial"):
        for el in base.findall(tag):
            origin = el.find("origin")
            if origin is not None:
                transform_origin(origin, rot)
    joint1 = joint_by_parent(robot, base_name)
    if joint1 is None:
        raise ValueError(f"no joint found with parent {base_name!r} to carry the rotation")
    if joint1.find("origin") is not None:
        transform_origin(joint1.find("origin"), rot)


# --------------------------------------------------------------------------- FK / verify
def fk(robot: ET.Element) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Global home-pose transforms per link, plus each joint's world axis.

    Also proves the tree is acyclic (raises on a cycle) with a resolvable single root.
    """
    c2j = {}
    for j in joints(robot):
        child = j.find("child").get("link")
        c2j[child] = j
    poses, axes = {}, {}

    def world(link: str, seen: set[str]) -> np.ndarray:
        if link in seen:
            raise ValueError(f"cycle at {link}")
        if link not in c2j:
            return np.eye(4)
        j = c2j[link]
        parent = j.find("parent").get("link")
        tp = world(parent, seen | {link})
        o = j.find("origin")
        xyz = parse_triplet(o.get("xyz", "0 0 0")) if o is not None else np.zeros(3)
        rpy = parse_triplet(o.get("rpy", "0 0 0")) if o is not None else np.zeros(3)
        t = np.eye(4)
        t[:3, :3] = rpy_to_matrix(rpy)
        t[:3, 3] = xyz
        return tp @ t

    for lk in links(robot):
        poses[lk.get("name")] = world(lk.get("name"), set())
    for j in joints(robot):
        ax = j.find("axis")
        if ax is None:
            continue
        child = j.find("child").get("link")
        wa = poses[child][:3, :3] @ parse_triplet(ax.get("xyz"))
        axes[j.get("name")] = wa / (np.linalg.norm(wa) or 1.0)
    return poses, axes


def verify_rigidity(baseline_robot: ET.Element, after_robot: ET.Element, base_name: str) -> tuple[list[str], dict]:
    """The arm must stay rigid: relative parent->child transforms for every joint *below* the
    intentionally-rotated ``base -> link1`` joint must be unchanged between ``baseline_robot`` and
    ``after_robot``. Compared by child-link name so a ``dof_jointN`` -> ``jointN`` rename does not
    matter. Returns (problem messages, world axes of the final tree)."""
    msgs = []
    _, aa = fk(after_robot)  # world axes for the report (also proves the tree is acyclic)
    max_err = 0.0
    b_c2j = {j.find("child").get("link"): j for j in joints(baseline_robot)}
    a_c2j = {j.find("child").get("link"): j for j in joints(after_robot)}
    for child, aj in a_c2j.items():
        parent = aj.find("parent").get("link")
        if parent == base_name:  # base->link1 joint is intentionally rotated
            continue
        bj = b_c2j.get(child)
        if bj is None:
            continue
        for attr in ("xyz", "rpy"):
            ao = aj.find("origin")
            bo = bj.find("origin")
            av = parse_triplet(ao.get(attr, "0 0 0")) if ao is not None else np.zeros(3)
            bv = parse_triplet(bo.get(attr, "0 0 0")) if bo is not None else np.zeros(3)
            max_err = max(max_err, float(np.max(np.abs(av - bv))))
    if max_err > 1e-9:
        msgs.append(f"RIGIDITY FAIL: downstream joint origins changed (max {max_err:.2e})")
    return msgs, aa


def verify_meshes(robot: ET.Element, urdf_dir: str) -> list[str]:
    missing = []
    for mesh in robot.iter("mesh"):
        fn = mesh.get("filename")
        path = os.path.join(urdf_dir, fn)
        if not os.path.isfile(path):
            missing.append(fn)
    return [f"MESH MISSING: {m}" for m in missing]


def verify_base_mesh_inertia_aligned(robot: ET.Element, base_name: str, tol: float = 1e-9) -> list[str]:
    """The base visual (mesh), collision, and inertial origins must share one orientation, so every
    heading rotation applied to one is applied identically to the others. Compare as rotation
    matrices (``rpy`` strings can differ, e.g. ``pi`` vs ``-pi``, for the same rotation). A base
    ``<collision>`` left un-rotated by ``apply_rotation_to_base`` would surface here as a
    collision-vs-visual mismatch. (ONShape collision geometry mirrors the visual mesh, so they are
    co-oriented in these exports; a legitimately differently-oriented collider is not expected.)"""
    base = link_by_name(robot, base_name)
    if base is None:
        return [f"BASE MISSING: no link named {base_name!r}"]
    vis, ine = base.find("visual"), base.find("inertial")
    if vis is None or ine is None or vis.find("origin") is None or ine.find("origin") is None:
        return []  # nothing to compare (no visual or no inertial origin)
    rv = rpy_to_matrix(parse_triplet(vis.find("origin").get("rpy", "0 0 0")))
    ri = rpy_to_matrix(parse_triplet(ine.find("origin").get("rpy", "0 0 0")))
    msgs = []
    err = float(np.max(np.abs(rv - ri)))
    if err > tol:
        msgs.append(f"BASE MESH/INERTIA MISALIGNED: visual vs inertial rpy differ (max {err:.2e})")
    for col in base.findall("collision"):
        origin = col.find("origin")
        if origin is None:
            continue
        rc = rpy_to_matrix(parse_triplet(origin.get("rpy", "0 0 0")))
        cerr = float(np.max(np.abs(rc - rv)))
        if cerr > tol:
            msgs.append(f"BASE COLLISION MISALIGNED: collision vs visual rpy differ (max {cerr:.2e})")
    return msgs


# --------------------------------------------------------------------------- serialize
def serialize(elem: ET.Element, indent: int = 0) -> str:
    pad = "    " * indent
    if not isinstance(elem.tag, str):  # comment / PI node (stdlib ET tags these with a callable)
        return f"{pad}<!--{elem.text}-->"
    attrs = "".join(f' {k}="{v}"' for k, v in elem.attrib.items())
    children = list(elem)
    has_text = elem.text is not None and elem.text.strip()
    if not children and not has_text:
        return f"{pad}<{elem.tag}{attrs} />"
    lines = [f"{pad}<{elem.tag}{attrs}>"]
    lines.extend(serialize(c, indent + 1) for c in children)
    lines.append(f"{pad}</{elem.tag}>")
    return "\n".join(lines)


def read_preamble_comments(raw: str) -> list[str]:
    head = raw.split("<robot", 1)[0]
    return re.findall(r"<!--.*?-->", head, flags=re.DOTALL)
