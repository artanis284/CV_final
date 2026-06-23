import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a textured mesh into a colored point cloud for Gaussian initialization.")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--points", default=200000, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if mesh.is_empty():
        raise ValueError(f"Cannot load mesh: {args.mesh}")
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=args.points, init_factor=5)

    if not pcd.has_colors():
        colors = np.full((len(pcd.points), 3), 0.75, dtype=np.float64)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(args.out), pcd)
    print(f"Wrote {len(pcd.points)} sampled points to {args.out}")


if __name__ == "__main__":
    main()
