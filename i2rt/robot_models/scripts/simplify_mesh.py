"""Simplify all STL meshes under a given directory in-place.

Uses quadric edge-collapse decimation (pymeshlab) to reduce face count while
preserving visual quality.  Processed files overwrite the originals so the
robot model XML references remain valid.

Usage:
    # Simplify every .stl / .STL under robot_models/
    python i2rt/robot_models/scripts/simplify_mesh.py i2rt/robot_models

    # Override target face count
    python i2rt/robot_models/scripts/simplify_mesh.py i2rt/robot_models --faces 5000
"""

import argparse
import glob
import os

import pymeshlab

TARGET_FACES = 8000


def simplify_file(path: str, target_faces: int) -> None:
    before = os.path.getsize(path)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    current_faces = ms.current_mesh().face_number()
    if current_faces <= target_faces:
        print(f"  skip  {os.path.basename(path)}  ({current_faces} faces <= {target_faces})")
        return
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    ms.save_current_mesh(path)
    after = os.path.getsize(path)
    print(
        f"  {os.path.basename(path):40s}  {current_faces:>7} -> {target_faces} faces  "
        f"{before / 1024:.0f}K -> {after / 1024:.0f}K"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplify all STL meshes under PATH in-place")
    parser.add_argument("path", type=str, help="Root directory to search for STL files")
    parser.add_argument("--faces", type=int, default=TARGET_FACES, help="Target face count (default: %(default)s)")
    args = parser.parse_args()

    stl_files = sorted(
        glob.glob(os.path.join(args.path, "**", "*.STL"), recursive=True)
        + glob.glob(os.path.join(args.path, "**", "*.stl"), recursive=True)
    )

    if not stl_files:
        print(f"No STL files found under {args.path}")
        return

    print(f"Found {len(stl_files)} STL files, target faces = {args.faces}\n")
    for f in stl_files:
        simplify_file(f, args.faces)

    print("\nDone.")


if __name__ == "__main__":
    main()
