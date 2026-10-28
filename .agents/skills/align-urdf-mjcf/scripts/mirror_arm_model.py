#!/usr/bin/env python3
"""Mirror a whole arm model about the world x-z plane (``y -> -y``).

Produces the left-hand twin of a right-hand model: URDF, MJCF and every binary STL. This is not a
stage of the ONShape alignment pipeline -- it runs *after* a model is finished, and re-runs whenever
the source model changes so the two sides never drift apart.

Why it is a separate script rather than a mode of the pipeline: a reflection has determinant -1, and
nothing in ``urdf_align_lib`` accepts one. ``transform_origin`` round-trips through
``scipy``'s ``Rotation.from_matrix``, which raises ``Non-positive determinant`` on ``diag(1,-1,1)``;
``apply_rotation_to_base`` only touches the base link and the ``base -> link1`` joint, and its
companion ``verify_rigidity`` exists to *reject* any change below that joint -- which is exactly what
a mirror is.

**Every rule below is a sign flip or a component swap**, so the mirror is bit-exact: the numbers are
rewritten as text and never re-formatted, and float32 mesh coordinates only lose their sign bit. That
is a property of reflecting about a coordinate plane specifically; baking a general rotation would
re-round every value.

    quantity                                     rule
    ------------------------------------------   ------------------------------------------------
    any position (body/geom/site/inertial pos,    negate y
    URDF origin xyz)
    URDF origin rpy                               negate roll and yaw, keep pitch
    MJCF frame quat (body/geom/site)              (w,x,y,z) -> (w,-x,y,-z)          [R -> M R M]
    MJCF inertial quat (principal axes)           (w,x,y,z) -> (-y,z,w,-x)          [Ry(pi) (x) q]
    URDF inertia tensor                           negate ixy and iyz
    hinge axis (a pseudovector)                   negate x and z                    [a -> -M a]
    slide axis (a true vector)                    negate y                          [a -> M a]
    joint range, mass, diaginertia, rgba, meshes  unchanged
    STL triangle                                  negate y of the normal and the three vertices,
                                                  then swap v2/v3 to restore CCW winding

``M R M`` and ``Ry(pi) R_p`` are both proper rotations, which is why the mirrored files still parse:
the reflection never appears on its own. The two quaternion rules differ because a *frame* is
conjugated by ``M`` while a *principal-axis* triad is only left-multiplied by it -- ``M R_p`` has
determinant -1, and ``-M = Ry(pi)`` is the left-multiplication that re-rights it while preserving
``R' diag(d) R'^T = M I M``.

The result is a **true mirror image**: identical joint vectors put the two models in mirror-symmetric
poses, so ``q_mirrored = q_source`` for every joint and no range needs re-signing.

That falls out of the axis rules above rather than being imposed. A rotation axis is a pseudovector,
transforming as ``a -> det(M) M a = -M a``; a translation axis is an ordinary vector, ``a -> M a``.
Substituting either into the mirrored frame ``R' = M R M`` reproduces the reflected motion at the
*same* joint value, which is why the joint coordinate and its limits come through untouched. In
world terms an axis along the mirror normal keeps its direction and one lying in the mirror plane
reverses -- the two sides agree on the first kind of joint and oppose on the second.

Usage:
    python .claude/skills/align-urdf-mjcf/scripts/mirror_arm_model.py <src_dir> <dst_dir> [--dry-run]

``<src_dir>`` and ``<dst_dir>`` are model version dirs holding ``<name>.urdf``, ``<name>.xml`` and
``assets/``. The mirrored arm is its own family, so its files carry its own name: pass ``--name``
when the destination's basename differs from the source's, e.g.

    ... arm/yam_7dof_right/v1 arm/yam_7dof_left/v1 --name yam_7dof_left

which reads ``yam_7dof_right.{urdf,xml}`` and writes ``yam_7dof_left.{urdf,xml}``, rewriting the
URDF ``<robot name>`` and MJCF ``<mujoco model>`` to match. Those names are load-bearing:
``get_arm_xml_path`` resolves ``arm/<family>/v<N>/<family>.xml``, and ``sync_gripper_mounts.py``
keys the mount block off the MJCF's model name.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET
from typing import Annotated, Dict, List

import numpy as np
import tyro

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "transform-onshape-urdf", "scripts")
)
from urdf_align_lib import joints, verify_meshes

# Reflection about the world x-z plane. Documentation only -- the transformations below are written
# out as sign rules, and M is never handed to scipy, which rejects a determinant of -1.
M = np.diag([1.0, -1.0, 1.0])


def neg(token: str) -> str:
    """Negate a numeric token by editing its sign character, so no digit is re-formatted.

    Zero is returned untouched: flipping it would churn ``pos="0 0 0"`` into ``pos="0 -0 0"`` for
    no gain.
    """
    if float(token) == 0.0:
        return token
    return token[1:] if token.startswith("-") else "-" + token


def neg_component(text: str, index: int) -> str:
    """Negate one whitespace-separated component of a vector attribute."""
    parts = text.split()
    parts[index] = neg(parts[index])
    return " ".join(parts)


def mirror_frame_quat(text: str) -> str:
    """``R -> M R M`` on a wxyz quaternion: negate x and z."""
    return neg_component(neg_component(text, 1), 3)


def mirror_inertial_quat(text: str) -> str:
    """``R_p -> Ry(pi) R_p`` on a wxyz quaternion: ``(w,x,y,z) -> (-y,z,w,-x)``.

    ``M R_p`` would decompose the mirrored inertia correctly but is improper; left-multiplying by
    ``Ry(pi) = -M`` re-rights it without disturbing ``diag(d)``, which is why ``diaginertia`` stays
    byte-identical.
    """
    w, x, y, z = text.split()
    return " ".join([neg(y), z, w, neg(x)])


def mirror_rpy(text: str) -> str:
    """``R -> M R M`` on a URDF ``rpy``: negate roll and yaw, keep pitch.

    With ``R = Rz(yaw) Ry(pitch) Rx(roll)``, the ``M...M`` conjugation telescopes term by term --
    ``M Rz(a) M = Rz(-a)``, ``M Ry(a) M = Ry(a)``, ``M Rx(a) M = Rx(-a)`` -- so the result is still
    an rpy in the same convention and no trigonometry is needed.
    """
    return neg_component(neg_component(text, 0), 2)


def mirror_joint_axis(text: str, is_hinge: bool) -> str:
    """Mirror a joint axis. A hinge axis is a pseudovector, a slide axis an ordinary vector.

    ``a -> -M a`` (negate x and z) for a hinge, ``a -> M a`` (negate y) for a slide. Both leave the
    joint *coordinate* alone, which is what makes the mirrored model a true mirror image at the same
    joint values -- and why no ``range`` is touched anywhere in this script.

    Signed zeros are normalized away: a human reads these to check a joint's sense, and a stray
    ``-0`` in an axis is noise.
    """
    x, y, z = text.split()
    out = (neg(x), y, neg(z)) if is_hinge else (x, neg(y), z)
    return " ".join("0" if float(t) == 0.0 else t for t in out)


def joint_kinds(urdf_path: str) -> Dict[str, bool]:
    """Per joint name: True for a hinge (revolute/continuous), False for a slide (prismatic)."""
    kinds = {}
    for j in joints(ET.parse(urdf_path).getroot()):
        kind = j.get("type")
        if kind not in ("revolute", "continuous", "prismatic"):
            raise ValueError(f"{j.get('name')}: cannot mirror a joint of type {kind!r}")
        kinds[j.get("name")] = kind != "prismatic"
    return kinds


def mirror_urdf(raw: str, hinges: Dict[str, bool]) -> str:
    """Rewrite a pretty-printed URDF line by line, leaving comments and layout untouched."""
    out: List[str] = []
    joint_name = None
    for line in raw.splitlines():
        if (m := re.search(r'<joint\s+name="([^"]+)"', line)) is not None:
            joint_name = m.group(1)
        elif "</joint>" in line:
            joint_name = None

        if "<origin" in line:
            line = re.sub(r'\bxyz="([^"]*)"', lambda m: f'xyz="{neg_component(m.group(1), 1)}"', line)
            line = re.sub(r'\brpy="([^"]*)"', lambda m: f'rpy="{mirror_rpy(m.group(1))}"', line)
        elif "<inertia " in line:
            for key in ("ixy", "iyz"):
                line = re.sub(rf'{key}="([^"]*)"', lambda m, k=key: f'{k}="{neg(m.group(1))}"', line)
        elif "<axis" in line and joint_name is not None:
            line = re.sub(
                r'\bxyz="([^"]*)"',
                lambda m, jn=joint_name: f'xyz="{mirror_joint_axis(m.group(1), hinges[jn])}"',
                line,
            )
        out.append(line)
    return "\n".join(out) + "\n"


def mirror_mjcf(raw: str, hinges: Dict[str, bool]) -> str:
    """Rewrite a hand-formatted MJCF line by line, preserving its layout and comments.

    Element-per-line is the file's own convention (only the ``<contact>`` rationale spans lines, and
    it carries no attributes), so a textual pass keeps the diff to the numbers that actually move --
    which is the whole point of a hand-maintained MJCF.
    """
    out: List[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("<inertial"):
            line = re.sub(r'\bpos="([^"]*)"', lambda m: f'pos="{neg_component(m.group(1), 1)}"', line)
            line = re.sub(r'\bquat="([^"]*)"', lambda m: f'quat="{mirror_inertial_quat(m.group(1))}"', line)
        elif stripped.startswith(("<body", "<geom", "<site")):
            line = re.sub(r'\bpos="([^"]*)"', lambda m: f'pos="{neg_component(m.group(1), 1)}"', line)
            line = re.sub(r'\bquat="([^"]*)"', lambda m: f'quat="{mirror_frame_quat(m.group(1))}"', line)
        elif (m := re.match(r'<joint\s+name="([^"]+)"', stripped)) is not None:
            # Matched on `<joint name=` specifically: <equality>'s constraint element is also a
            # <joint>, but names its two operands `joint1`/`joint2` and carries a `polycoef` that
            # must not be touched.
            name = m.group(1)
            line = re.sub(r'\bpos="([^"]*)"', lambda mm: f'pos="{neg_component(mm.group(1), 1)}"', line)
            line = re.sub(
                r'\baxis="([^"]*)"',
                lambda mm, jn=name: f'axis="{mirror_joint_axis(mm.group(1), hinges[jn])}"',
                line,
            )
        out.append(line)
    return "\n".join(out) + "\n"


def mirror_stl(src: str, dst: str) -> int:
    """Mirror a binary STL in place along y. Returns the triangle count.

    Reflecting the three vertices reverses each triangle's winding -- ``(M.a) x (M.b) = -M(a x b)``
    -- so v2 and v3 are swapped to put the face normal back on the outside. The stored normal is
    mirrored directly. Every write is a float32 sign-bit flip, so nothing is re-rounded.
    """
    with open(src, "rb") as f:
        blob = f.read()
    header, count = blob[:80], struct.unpack("<I", blob[80:84])[0]
    expected = 84 + count * 50
    if len(blob) != expected:
        raise ValueError(f"{src}: not a binary STL ({len(blob)} bytes, expected {expected} for {count} tris)")

    body = bytearray(blob[84:])
    for i in range(count):
        off = i * 50
        n, v1, v2, v3 = (np.frombuffer(body[off + k * 12 : off + k * 12 + 12], dtype="<f4") for k in range(4))
        mirrored = np.concatenate([n * [1, -1, 1], v1 * [1, -1, 1], v3 * [1, -1, 1], v2 * [1, -1, 1]])
        body[off : off + 48] = mirrored.astype("<f4").tobytes()

    with open(dst, "wb") as f:
        f.write(header + struct.pack("<I", count) + bytes(body))
    return count


def main(
    src: Annotated[str, tyro.conf.Positional],
    dst: Annotated[str, tyro.conf.Positional],
    name: str | None = None,
    dry_run: bool = False,
) -> int:
    """Mirror the arm model in ``src`` about the world x-z plane into ``dst``.

    Args:
        src: Source model dir, holding ``<name>.urdf``, ``<name>.xml`` and ``assets/``.
        dst: Destination model dir. Created if absent; existing files are overwritten.
        name: Basename and model name for the mirrored files. Defaults to the source's, which is
            right only when the two sides share a family; a mirrored arm that is its own ArmType
            needs its own name, since the resolver derives the path from it.
        dry_run: Report the joint sign decisions and the files that would be written, write nothing.
    """
    urdfs = [f for f in sorted(os.listdir(src)) if f.endswith(".urdf")]
    if len(urdfs) != 1:
        print(f"  [error] expected exactly one .urdf in {src}, found {urdfs}")
        return 1
    name_src = os.path.splitext(urdfs[0])[0]
    src_urdf, src_xml = os.path.join(src, f"{name_src}.urdf"), os.path.join(src, f"{name_src}.xml")
    if not os.path.isfile(src_xml):
        print(f"  [error] no MJCF beside the URDF: {src_xml}")
        return 1

    problems = verify_meshes(ET.parse(src_urdf).getroot(), src)
    if problems:
        for p in problems:
            print(f"  [error] {p}")
        return 1

    hinges = joint_kinds(src_urdf)
    out_name = name or name_src
    print(
        f"mirroring {name_src} about the world x-z plane: {src} -> {dst}"
        + (f" (as {out_name})" if out_name != name_src else "")
    )
    for jname in sorted(hinges, key=lambda s: (len(s), s)):
        kind = "hinge, axis x and z negated" if hinges[jname] else "slide, axis y negated"
        print(f"  {jname:8s} {kind:30s} range kept")

    dst_assets = os.path.join(dst, "assets")
    if dry_run:
        print(
            f"  would write {out_name}.urdf, {out_name}.xml and {len(os.listdir(os.path.join(src, 'assets')))} meshes"
        )
        return 0

    os.makedirs(dst_assets, exist_ok=True)
    with open(src_urdf, encoding="utf-8") as f:
        urdf_raw = f.read()
    with open(src_xml, encoding="utf-8") as f:
        xml_raw = f.read()
    with open(os.path.join(dst, f"{out_name}.urdf"), "w", encoding="utf-8") as f:
        f.write(mirror_urdf(urdf_raw, hinges).replace(f'<robot name="{name_src}">', f'<robot name="{out_name}">'))
    with open(os.path.join(dst, f"{out_name}.xml"), "w", encoding="utf-8") as f:
        f.write(mirror_mjcf(xml_raw, hinges).replace(f'<mujoco model="{name_src}">', f'<mujoco model="{out_name}">'))
    print(f"  wrote {out_name}.urdf, {out_name}.xml")

    for mesh in sorted(os.listdir(os.path.join(src, "assets"))):
        src_mesh = os.path.join(src, "assets", mesh)
        if not mesh.endswith(".stl"):
            shutil.copy2(src_mesh, os.path.join(dst_assets, mesh))
            continue
        tris = mirror_stl(src_mesh, os.path.join(dst_assets, mesh))
        print(f"  wrote assets/{mesh:16s} {tris} triangles")
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main, description=__doc__))
