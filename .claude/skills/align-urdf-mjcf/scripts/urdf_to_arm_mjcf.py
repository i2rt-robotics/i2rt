#!/usr/bin/env python3
"""Generate an arm-only MuJoCo MJCF from an aligned YAM-family URDF (align-urdf-mjcf skill).

This is the URDF->MJCF half of the ONShape alignment pipeline (the URDF half lives in
``.claude/skills/transform-onshape-urdf/scripts/normalize_onshape_urdf.py`` + ``.claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py``). It maps an aligned arm
URDF to the arm-only MJCF convention used by ``yam.xml``:

    world
    └── base                       (body: inertial + base geom)
        └── link1 / joint1          ...
            └── ... link5 / joint5
                └── link6 / joint6  (empty end-effector mount; placeholder inertia, no geom)

Arm-only contract: exactly six joints ``joint1..joint6``; ``link6`` is the gripper mount frame
(not a physical link); the URDF gripper/tip bodies and their meshes are excluded. Only
``base.stl`` and ``link1.stl``..``link5.stl`` are declared.

After regenerating an arm MJCF, re-sync the per-arm gripper mount with
``.claude/skills/align-urdf-mjcf/scripts/sync_gripper_mounts.py <arm>.xml``: composition overwrites ``link6``'s pose/axis from
each gripper config's ``last_joint_mount.<arm>`` block, so a stale copy misplaces the gripper.

Mapping (align-urdf-mjcf "Map URDF Semantics to MJCF"):
  - joint origin  -> child body ``pos`` + ``quat``
  - joint axis    -> joint ``axis`` (kept in the joint/body frame)
  - visual origin -> geom ``pos`` + ``quat``
  - inertial xyz  -> inertial ``pos``; mass -> ``mass``
  - inertia + inertial rpy -> principal-axis ``quat`` + ``diaginertia`` via
        I_body = R_rpy @ I_urdf @ R_rpy.T ; eig-decompose; force det(+1); to wxyz quat

Validated: regenerating ``yam.xml`` from ``yam.urdf`` reproduces the committed file to
machine precision (max pose err ~1e-16, diaginertia ~1e-18). URDF RPY convention:
``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

Usage:
    python .claude/skills/align-urdf-mjcf/scripts/urdf_to_arm_mjcf.py <arm.urdf> <out.xml> [--model NAME] [--dof N]
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from typing import Annotated

import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

# Distinct render palette (matches yam.xml), keyed by mesh/link name. URDF material colours
# are unreliable across variants (some export flat grey), so we fix the palette here.
PALETTE = {
    "base": "0.317647 0.121569 0.498039 1",
    "link1": "0.72549 0.45098 0.317647 1",
    "link2": "0 0.4 0.6 1",
    "link3": "0.552941 0.580392 0.203922 1",
    "link4": "0.545098 0.52549 0.419608 1",
    "link5": "0.827451 0.623529 0.784314 1",
}
# Fallback for links beyond the palette (e.g. link6+ when --dof > 6), so generation never crashes.
DEFAULT_RGBA = "0.7 0.7 0.7 1"


def _rpy_to_matrix(s: str) -> np.ndarray:
    r, p, y = (float(v) for v in s.split())
    return R.from_euler("ZYX", [y, p, r]).as_matrix()


def _matrix_to_wxyz(m: np.ndarray) -> np.ndarray:
    x, y, z, w = R.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def _g(x: float) -> str:
    return f"{x:.17g}"


def _gq(q: np.ndarray) -> str:
    return " ".join(_g(v) for v in q)


def _rpy_str_to_quat_str(rpy_s: str) -> str:
    return _gq(_matrix_to_wxyz(_rpy_to_matrix(rpy_s)))


def _principal(inertia_attrib: dict[str, str], rpy_s: str) -> tuple[np.ndarray, np.ndarray, float]:
    """URDF inertia tensor + inertial rpy -> (wxyz quat, diaginertia, reconstruction residual)."""
    a = inertia_attrib
    I_urdf = np.array([
        [float(a["ixx"]), float(a["ixy"]), float(a["ixz"])],
        [float(a["ixy"]), float(a["iyy"]), float(a["iyz"])],
        [float(a["ixz"]), float(a["iyz"]), float(a["izz"])],
    ])
    Rin = _rpy_to_matrix(rpy_s)
    I_body = Rin @ I_urdf @ Rin.T
    evals, evecs = np.linalg.eigh(I_body)
    if np.linalg.det(evecs) < 0:  # ensure a proper rotation
        evecs[:, 0] = -evecs[:, 0]
    resid = float(np.max(np.abs(evecs @ np.diag(evals) @ evecs.T - I_body)))
    return _matrix_to_wxyz(evecs), evals, resid


def _parse_urdf(path: str) -> tuple[dict, dict]:
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for lk in root.findall("link"):
        d = {}
        for tag in ("visual", "inertial"):
            e = lk.find(tag)
            if e is None:
                continue
            o = e.find("origin")
            d[tag] = {"xyz": o.get("xyz"), "rpy": o.get("rpy", "0 0 0")}
            if tag == "inertial":
                d[tag]["mass"] = e.find("mass").get("value")
                d[tag]["I"] = e.find("inertia").attrib
        links[lk.get("name")] = d
    for j in root.findall("joint"):
        o = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        joints[j.get("name")] = {
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "xyz": o.get("xyz"),
            "rpy": o.get("rpy", "0 0 0"),
            "axis": ax.get("xyz") if ax is not None else "0 0 1",
            "lower": lim.get("lower") if lim is not None else "0",
            "upper": lim.get("upper") if lim is not None else "0",
        }
    return links, joints


def _joint_index(name: str) -> int:
    """First run of digits in a joint name (``joint10`` -> 10, ``dof_joint3`` -> 3), for numeric
    ordering. Sorting the raw string would place ``joint10`` before ``joint2``; falls back to 0 when
    there is no number."""
    m = re.search(r"\d+", name)
    return int(m.group()) if m else 0


def _arm_chain(joints: dict, dof: int) -> list[tuple[str, dict]]:
    """Follow the actuated chain from ``base`` for ``dof`` joints (stops before gripper/tips)."""
    by_parent = {}
    for name, jd in joints.items():
        by_parent.setdefault(jd["parent"], []).append((name, jd))
    chain, cur = [], "base"
    while len(chain) < dof:
        cands = sorted(by_parent.get(cur, []), key=lambda kv: _joint_index(kv[0]))
        if not cands:
            raise ValueError(f"arm chain breaks at '{cur}' after {len(chain)} joints")
        nxt = cands[0]  # unique actuated continuation of the chain
        chain.append(nxt)
        cur = nxt[1]["child"]
    return chain


def generate(urdf_path: str, model_name: str, dof: int = 6) -> tuple[str, list[tuple[str, float]]]:
    links, joints = _parse_urdf(urdf_path)
    chain = _arm_chain(joints, dof)

    def ind(n: int) -> str:
        return "  " * n

    out = [f'<mujoco model="{model_name}">',
           f'{ind(1)}<compiler angle="radian" meshdir="assets"/>', "",
           f"{ind(1)}<asset>"]
    for m in ["base"] + [f"link{i}" for i in range(1, dof)]:
        pad = " " if m == "base" else ""
        out.append(f'{ind(2)}<mesh name="{m}" {pad}file="{m}.stl"/>')
    out += [f"{ind(1)}</asset>", "", f"{ind(1)}<worldbody>"]

    residuals = []
    bi, bv = links["base"]["inertial"], links["base"]["visual"]
    q, diag, resid = _principal(bi["I"], bi["rpy"])
    residuals.append(("base", resid))
    depth = 2
    out.append(f'{ind(depth)}<body name="base" pos="0 0 0" quat="1 0 0 0">')
    out.append(f'{ind(depth + 1)}<inertial pos="{bi["xyz"]}" quat="{_gq(q)}" mass="{bi["mass"]}" diaginertia="{_gq(diag)}"/>')
    out.append(f'{ind(depth + 1)}<geom pos="{bv["xyz"]}" quat="{_rpy_str_to_quat_str(bv["rpy"])}" type="mesh" rgba="{PALETTE["base"]}" mesh="base"/>')

    for i, (_, jd) in enumerate(chain, start=1):
        depth += 1
        name = f"link{i}"
        out.append(f'{ind(depth)}<body name="{name}" pos="{jd["xyz"]}" quat="{_rpy_str_to_quat_str(jd["rpy"])}">')
        if i < dof:
            li, lv = links[jd["child"]]["inertial"], links[jd["child"]]["visual"]
            q, diag, resid = _principal(li["I"], li["rpy"])
            residuals.append((name, resid))
            out.append(f'{ind(depth + 1)}<inertial pos="{li["xyz"]}" quat="{_gq(q)}" mass="{li["mass"]}" diaginertia="{_gq(diag)}"/>')
            out.append(f'{ind(depth + 1)}<joint name="joint{i}" pos="0 0 0" axis="{jd["axis"]}" type="hinge" range="{jd["lower"]} {jd["upper"]}" actuatorfrcrange="-10 10"/>')
            out.append(f'{ind(depth + 1)}<geom pos="{lv["xyz"]}" quat="{_rpy_str_to_quat_str(lv["rpy"])}" type="mesh" rgba="{PALETTE.get(name, DEFAULT_RGBA)}" mesh="{name}"/>')
        else:  # terminal mount: placeholder inertia, no geom
            out.append(f'{ind(depth + 1)}<inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>')
            out.append(f'{ind(depth + 1)}<joint name="joint{i}" pos="0 0 0" axis="{jd["axis"]}" type="hinge" range="{jd["lower"]} {jd["upper"]}" actuatorfrcrange="-10 10"/>')

    for d in range(depth, 1, -1):
        out.append(f"{ind(d)}</body>")
    out += [f"{ind(1)}</worldbody>", "</mujoco>"]
    return "\n".join(out) + "\n", residuals


def main(
    urdf: Annotated[str, tyro.conf.Positional],
    out: Annotated[str, tyro.conf.Positional],
    model: str | None = None,
    dof: int = 6,
) -> int:
    """Generate an arm-only MJCF from an aligned arm URDF.

    Args:
        urdf: path to the aligned arm URDF.
        out: path to write the generated MJCF.
        model: mujoco model name (default: URDF's parent dir name).
        dof: number of actuated arm joints.
    """
    import os
    model_name = model or os.path.basename(os.path.dirname(os.path.abspath(urdf)))
    xml, residuals = generate(urdf, model_name, dof)
    print("inertia reconstruction residuals (kg·m², expect ~0):")
    for n, r in residuals:
        print(f"   {n}: {r:.2e}")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(xml)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main, description=__doc__))
