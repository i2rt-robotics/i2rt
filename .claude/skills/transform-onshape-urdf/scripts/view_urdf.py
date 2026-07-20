#!/usr/bin/env python3
"""Visually inspect a YAM-family URDF in MuJoCo with world axes -- **stage 2 checkpoint**.

This is the required human checkpoint between ``normalize_onshape_urdf.py`` (stage 1) and
``apply_urdf_heading.py`` (stage 3). "Which way should the base face in the world?" cannot be
inferred from CAD coordinates or link names, so the operator looks at the assembled model against
the world XYZ axes and decides whether -- and by how much -- to rotate the heading.

MuJoCo can load a URDF directly, but by default it discards visual geoms and does not resolve the
``assets/*.stl`` paths, so nothing renders. We inject ``<mujoco><compiler discardvisual="false"/>
</mujoco>`` and hand the mesh bytes in via an ``assets`` dict keyed by the exact filename strings.

Usage:
    python .claude/skills/transform-onshape-urdf/scripts/view_urdf.py <model.urdf>              # interactive window (world frame on)
    python .claude/skills/transform-onshape-urdf/scripts/view_urdf.py <model.urdf> --screenshot out.png   # offscreen render (headless)
"""
from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Annotated

import mujoco
import numpy as np
import tyro


def load_model(urdf_path: str) -> mujoco.MjModel:
    """Load the URDF with visual geoms kept and mesh bytes supplied out-of-band."""
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    root = ET.parse(urdf_path).getroot()
    # MuJoCo's URDF import discards <visual> geoms by default; a <mujoco><compiler discardvisual>
    # extension block keeps them so the meshes render.
    mj = ET.Element("mujoco")
    ET.SubElement(mj, "compiler").set("discardvisual", "false")
    root.insert(0, mj)
    assets = {}
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn:
            assets[fn] = open(os.path.join(urdf_dir, fn), "rb").read()
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)


def render_offscreen(model: mujoco.MjModel) -> np.ndarray:
    """Render one offscreen frame with the world triad. May raise if no GL context is available."""
    # URDF import ignores <visual><global>, so the offscreen buffer stays at MuJoCo's compiled
    # default; render at exactly that size so the frame always fits.
    width, height = model.vis.global_.offwidth, model.vis.global_.offheight
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    opt = mujoco.MjvOption()
    opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # draw the world XYZ triad
    with mujoco.Renderer(model, height, width) as renderer:
        renderer.update_scene(data, scene_option=opt)
        return renderer.render()


def interactive(model: mujoco.MjModel) -> None:
    import mujoco.viewer

    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # world XYZ triad: X red, Y green, Z blue
        mujoco.mj_forward(model, data)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


def main(
    urdf: Annotated[str, tyro.conf.Positional],
    screenshot: str | None = None,
) -> int:
    """Inspect the URDF, or render one offscreen frame to a PNG.

    Args:
        urdf: path to the URDF to inspect.
        screenshot: render one offscreen frame to this PNG instead of opening a window (headless).
    """
    model = load_model(urdf)
    print(f"loaded {urdf}: nbody={model.nbody} ngeom={model.ngeom} nmesh={model.nmesh}")
    print("World axes: X=red, Y=green, Z=blue. "
          "Check: does the base face the intended robot-forward direction?")
    print("If not, note the axis + degrees and run:")
    print(f"  python .claude/skills/transform-onshape-urdf/scripts/apply_urdf_heading.py {urdf} --axis z --deg <N> --apply")

    if screenshot:
        try:
            pixels = render_offscreen(model)
        except Exception as e:  # offscreen GL context may be unavailable on a headless host
            print(f"[error] offscreen render failed ({type(e).__name__}: {e}).")
            print("        On a headless host set MUJOCO_GL=egl (or osmesa) and retry, "
                  "or run without --screenshot on a machine with a display.")
            return 1
        try:
            from PIL import Image  # not a hard dep; only needed for --screenshot
        except ImportError:
            print("[error] --screenshot needs Pillow: pip install pillow")
            return 1
        try:
            Image.fromarray(pixels).save(screenshot)
        except OSError as e:
            print(f"[error] could not write {screenshot!r}: {e}")
            return 1
        print(f"wrote {screenshot}")
    else:
        interactive(model)
    return 0


if __name__ == "__main__":
    sys.exit(tyro.cli(main, description=__doc__))
