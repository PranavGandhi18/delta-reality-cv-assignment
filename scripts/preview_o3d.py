"""
Interactive Open3D preview of the transformed scene — designed to run
locally on macOS (or anywhere Open3D's GUI works) as a sanity check that
the coordinate-system fix in `scripts/transform.py` produces something
that *looks* like the reference screenshot in the assignment PDF.

What this script renders:
  * Each transformed point cloud (`data/output/image{1,2,3}.ply`) with its
    per-point RGB color.
  * A small frustum for each camera in `data/output/traj.txt`, drawn as
    a wireframe so you can see where the photos were taken from.
  * A world axis triad at the origin (X=red, Y=green, Z=blue), so you can
    orient yourself in the viewer.

Controls (Open3D defaults):
  * Left-drag      → rotate
  * Right-drag     → pan
  * Scroll wheel   → zoom
  * Shift + drag   → roll
  * R              → reset view
  * H              → print full help in the terminal

Usage:
    python scripts/preview_o3d.py                 # the transformed output
    python scripts/preview_o3d.py --src           # the originals
    python scripts/preview_o3d.py --downsample 0.02
                                                   # voxel-grid downsample
                                                   # for snappier interaction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "output"
SRC_DIR = REPO / "ComputerVisionAssignment_Data" / "StreamingAssets" / "Points"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        action="store_true",
        help="Load the original (un-transformed) point clouds from the "
        "upstream StreamingAssets/Points instead of data/output/. "
        "Mainly for comparison.",
    )
    p.add_argument(
        "--downsample",
        type=float,
        default=0.02,
        help="Voxel size (in meters) for downsampling each cloud before "
        "rendering. 0 disables. Default 0.02 (~2cm voxels) keeps the "
        "viewer responsive without losing visible detail.",
    )
    p.add_argument(
        "--frustum-size",
        type=float,
        default=0.4,
        help="Length (in meters) of each camera frustum drawn for traj.txt. "
        "Default 0.4 — small enough not to obscure the scene.",
    )
    return p.parse_args()


def load_traj(path: Path) -> list[np.ndarray]:
    """Load traj.txt → list of 4x4 matrices in row order."""
    poses: list[np.ndarray] = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vals = np.array([float(x) for x in parts], dtype=np.float64)
            if vals.size != 16:
                raise ValueError(f"row has {vals.size} values, expected 16")
            poses.append(vals.reshape(4, 4))
    return poses


def camera_frustum_lineset(T_cw: np.ndarray, size: float, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    """Build a 5-vertex pyramid frustum representing a camera at pose T_cw.

    Apex is at the camera position; the base is a small square 1 unit of
    `size` in front of the camera (along the camera's +Z axis, which is
    Unity-forward after our transformation).

    The frustum is drawn purely as a visual gizmo — its base is a square
    of half-extent `size` at depth `size`. Not pixel-accurate, just a
    sense of position and orientation.
    """
    half = size * 0.5
    depth = size
    # vertices in CAMERA coords: apex + 4 base corners.
    local = np.array(
        [
            [0.0, 0.0, 0.0],        # apex (camera center)
            [-half, -half, depth],  # base TL
            [+half, -half, depth],  # base TR
            [+half, +half, depth],  # base BR
            [-half, +half, depth],  # base BL
        ],
        dtype=np.float64,
    )
    # apply T_cw to bring into world (viewer) coords.
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    world = local @ R.T + t

    lines = np.array(
        [
            [0, 1], [0, 2], [0, 3], [0, 4],   # apex to base corners
            [1, 2], [2, 3], [3, 4], [4, 1],   # base edges
        ],
        dtype=np.int32,
    )
    colors = np.tile(np.array(color, dtype=np.float64), (lines.shape[0], 1))

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def main() -> int:
    args = parse_args()
    src_dir = SRC_DIR if args.src else OUT_DIR
    traj_path = (SRC_DIR.parent / "traj.txt") if args.src else (OUT_DIR / "traj.txt")

    if not traj_path.exists():
        print(f"missing traj.txt at {traj_path}", file=sys.stderr)
        return 2

    geoms: list[o3d.geometry.Geometry] = []

    # 1) Point clouds.
    for i in (1, 2, 3):
        ply = src_dir / f"image{i}.ply"
        if not ply.exists():
            print(f"  missing {ply}, skipping", file=sys.stderr)
            continue
        print(f"loading {ply}")
        pcd = o3d.io.read_point_cloud(str(ply))
        if args.downsample > 0:
            before = len(pcd.points)
            pcd = pcd.voxel_down_sample(voxel_size=args.downsample)
            after = len(pcd.points)
            print(f"  downsampled {before} → {after} points "
                  f"(voxel={args.downsample} m)")
        geoms.append(pcd)

    # 2) Camera frusta.
    poses = load_traj(traj_path)
    cam_colors = [
        (1.0, 0.2, 0.2),  # red
        (0.2, 1.0, 0.2),  # green
        (0.2, 0.4, 1.0),  # blue
    ]
    for i, T in enumerate(poses):
        col = cam_colors[i % len(cam_colors)]
        ls = camera_frustum_lineset(T, size=args.frustum_size, color=col)
        geoms.append(ls)
        cam_pos = T[:3, 3]
        print(f"camera {i+1}: pos={cam_pos.round(3).tolist()} (color={col})")

    # 3) Origin axes.
    triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    geoms.append(triad)

    # 4) Launch interactive window.
    print("\nopening Open3D window — close it to exit. "
          "Press H in the window for the full controls list.")
    o3d.visualization.draw_geometries(
        geoms,
        window_name=("delta-reality preview — "
                     f"{'source' if args.src else 'transformed'}"),
        width=1280,
        height=800,
        point_show_normal=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
