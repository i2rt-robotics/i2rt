"""Offline CAD conversion. Run with uv run --with cadquery --with trimesh --with fast-simplification python PATH.

No hardware imports. Output meshes are metres in the unchanged finger body frames.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def export() -> None:
    """Place the supplied assembly using its mount face and two 21 mm-spaced holes."""
    solids = cq.importers.importStep(str(ROOT / "cad/umi_flow_fingers.step")).solids().vals()
    if len(solids) != 6:
        raise ValueError("Expected adapter, finger and pad for each of two fingers")
    xml = ET.parse(ROOT / "linear_4310.xml")
    # CAD millimetres -> flange millimetres: (X,Y,Z) -> (Z,X,Y)+offset.
    # Datum: old mesh's mounting face Z=-70.377 mm and hole-axis centers.
    # No fingertip/grasp_site constraint is used to place the geometry.
    rotation = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    offsets = {"left": [-1.107, -10.095, -70.377], "right": [-1.297, -87.702, -70.377]}
    for side, first in [("left", 0), ("right", 3)]:
        body = xml.find(f".//body[@name='tip_{side}']")
        assert body is not None
        position = np.fromstring(body.attrib["pos"], sep=" ")
        wxyz = np.fromstring(body.attrib["quat"], sep=" ")
        body_rotation = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
        for index, component in enumerate(["adapter", "finger", "pad"]):
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "part.stl"
                cq.exporters.export(solids[first + index], str(source), tolerance=0.05, angularTolerance=0.1)
                mesh = trimesh.load(source, force="mesh")
            flange = (mesh.vertices @ rotation.T + offsets[side]) * 0.001
            mesh.vertices = (flange - position) @ body_rotation
            # Explicit convex collision envelope of each original CAD solid.
            mesh.convex_hull.export(ROOT / "assets" / f"umi_{side}_{component}_collision.stl")
            if len(mesh.faces) > 15000:
                mesh = mesh.simplify_quadric_decimation(face_count=15000)
            mesh.export(ROOT / "assets" / f"umi_{side}_{component}.stl")


if __name__ == "__main__":
    export()
