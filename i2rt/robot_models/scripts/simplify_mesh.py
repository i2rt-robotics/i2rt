import glob
import os

import pymeshlab

# target number of faces (adjust as needed)
TARGET_FACES = 5000
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="The path to the mesh")
    args = parser.parse_args()

    stl_files = glob.glob(f"{args.path}/*.STL")
    output_dir = f"{args.path}/simplified"
    os.makedirs(output_dir, exist_ok=True)

    for f in stl_files:
        print(f"Processing {f}...")
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(f)

        # simplify mesh
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=TARGET_FACES)

        # save simplified version
        base_name = os.path.basename(f)
        out_path = os.path.join(output_dir, base_name)
        ms.save_current_mesh(out_path)
        print(f"Saved simplified mesh to {out_path}")
