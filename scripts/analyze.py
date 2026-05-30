"""
Analyze the input data without writing any output files.

Goals:
  * Parse `traj.txt` and verify each row is a proper rigid transform
    (rotation 3x3 is orthogonal, det == +1).
  * Stream each PLY (the originals are ~125 MB ASCII so a full load is OK
    on a modern machine but we still keep it tidy). Report:
      - vertex count
      - axis-aligned bounding box (min/max per axis)
      - centroid
      - mean RGB color
  * Project each PLY's bbox/centroid into the world under the candidate
    cam->world matrix and compare against the camera's translation. If
    H1 ("PLY is camera-local, apply T_cw") is correct, the post-transform
    bboxes from all three cameras should overlap in roughly the same
    world region (one room).
  * Print everything to stdout. No file output.

Run:
    python scripts/analyze.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
STREAM = REPO / "ComputerVisionAssignment_Data" / "StreamingAssets"
POINTS = STREAM / "Points"
TRAJ = STREAM / "traj.txt"


def load_traj(path: Path) -> dict[int, np.ndarray]:
    """
    Load traj.txt -> {image_id: 4x4 matrix}.

    Format: each non-empty line is 16 whitespace-separated floats encoding a
    row-major 4x4 matrix. The image index is implicit: row N (1-based)
    corresponds to imageN.ply.
    """
    poses: dict[int, np.ndarray] = {}
    img_id = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_id += 1
            vals = np.array([float(x) for x in parts], dtype=np.float64)
            if vals.size != 16:
                raise ValueError(f"row {img_id} has {vals.size} values, expected 16")
            poses[img_id] = vals.reshape(4, 4)
    return poses


def check_rotation(R: np.ndarray, tag: str) -> None:
    """Sanity-check a 3x3 rotation. Print orthonormality + determinant."""
    should_be_I = R @ R.T
    err = np.max(np.abs(should_be_I - np.eye(3)))
    det = np.linalg.det(R)
    print(f"  {tag}: max |R R^T - I| = {err:.2e}, det(R) = {det:+.6f}")


def stream_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Read an ASCII PLY and return (xyz [N,3], rgb [N,3] uint8, n).
    The header is small; the body is one line per vertex with
    'x y z r g b'. We use numpy's loadtxt-style streaming via fromstring per
    chunk to keep memory reasonable but here a single np.loadtxt is fine.
    """
    with open(path) as f:
        # parse header
        n_vertex = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in PLY header")
            line = line.strip()
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            if line == "end_header":
                break
        if n_vertex is None:
            raise ValueError("No 'element vertex' in PLY header")

        # body: read all remaining lines via numpy
        # this is ~125 MB ascii; about ~15-20s and ~700 MB of RAM peak.
        data = np.loadtxt(f, dtype=np.float64)

    if data.shape[0] != n_vertex:
        print(
            f"  WARN: header said {n_vertex} vertices, parsed {data.shape[0]}",
            file=sys.stderr,
        )
    xyz = data[:, 0:3].astype(np.float32)
    rgb = data[:, 3:6].astype(np.uint8)
    return xyz, rgb, n_vertex


def bbox_stats(xyz: np.ndarray) -> dict:
    mn = xyz.min(axis=0)
    mx = xyz.max(axis=0)
    return {
        "min": mn,
        "max": mx,
        "size": mx - mn,
        "centroid": xyz.mean(axis=0),
        "median": np.median(xyz, axis=0),
    }


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+8.3f}" for x in v) + "]"


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply 4x4 T to Nx3 points -> Nx3."""
    R = T[:3, :3]
    t = T[:3, 3]
    return xyz @ R.T + t


def main() -> int:
    print(f"Repo: {REPO}")
    print(f"Streaming assets dir: {STREAM}")
    print()

    # 1) trajectories
    print("=== traj.txt ===")
    poses = load_traj(TRAJ)
    for img_id, T in sorted(poses.items()):
        print(f"\nPose for image {img_id}:")
        print(T)
        check_rotation(T[:3, :3], f"R{img_id}")
        cam_pos = T[:3, 3]
        # camera forward axis in OpenCV convention is +Z in camera frame.
        # In a cam->world matrix, the camera's +Z axis in world coords is
        # the 3rd column of the 3x3 rotation.
        cam_x = T[:3, 0]
        cam_y = T[:3, 1]
        cam_z = T[:3, 2]
        print(f"  cam pos (world):  {fmt_vec(cam_pos)}")
        print(f"  cam +X (world):   {fmt_vec(cam_x)}")
        print(f"  cam +Y (world):   {fmt_vec(cam_y)}")
        print(f"  cam +Z (world):   {fmt_vec(cam_z)}")

    # 2) point clouds
    print("\n=== point clouds ===")
    summaries: dict[int, dict] = {}
    for img_id in sorted(poses.keys()):
        ply = POINTS / f"image{img_id}.ply"
        if not ply.exists():
            print(f"  MISSING: {ply}")
            continue
        print(f"\n-- {ply.name} ({os.path.getsize(ply)/1e6:.1f} MB)")
        xyz, rgb, n = stream_ply_xyz_rgb(ply)
        print(f"  n_vertex (header): {n}")
        print(f"  parsed shape: {xyz.shape}")
        bb = bbox_stats(xyz)
        print(f"  bbox min:  {fmt_vec(bb['min'])}")
        print(f"  bbox max:  {fmt_vec(bb['max'])}")
        print(f"  bbox size: {fmt_vec(bb['size'])}")
        print(f"  centroid:  {fmt_vec(bb['centroid'])}")
        print(f"  median:    {fmt_vec(bb['median'])}")
        print(f"  mean rgb:  {rgb.mean(axis=0)}")
        summaries[img_id] = {
            "xyz": xyz,
            "bb": bb,
            "pose": poses[img_id],
        }

    # 3) Hypothesis H1: bake cam->world transform; check world overlap.
    print("\n=== H1: assume PLY is camera-local; apply T_cw ===")
    world_bboxes = {}
    for img_id, s in summaries.items():
        # Don't transform the full N=3M cloud — use the 8 bbox corners.
        bb = s["bb"]
        corners = np.array(
            [
                [bb["min"][0], bb["min"][1], bb["min"][2]],
                [bb["max"][0], bb["min"][1], bb["min"][2]],
                [bb["min"][0], bb["max"][1], bb["min"][2]],
                [bb["max"][0], bb["max"][1], bb["min"][2]],
                [bb["min"][0], bb["min"][1], bb["max"][2]],
                [bb["max"][0], bb["min"][1], bb["max"][2]],
                [bb["min"][0], bb["max"][1], bb["max"][2]],
                [bb["max"][0], bb["max"][1], bb["max"][2]],
            ],
            dtype=np.float64,
        )
        T = s["pose"]
        wcorners = transform_points(T, corners)
        wmn = wcorners.min(axis=0)
        wmx = wcorners.max(axis=0)
        world_bboxes[img_id] = (wmn, wmx)
        print(
            f"  image{img_id}: world bbox min={fmt_vec(wmn)} max={fmt_vec(wmx)} "
            f"cam_pos={fmt_vec(T[:3,3])}"
        )

    # Compare overlap of world bboxes pairwise.
    print("\n  pairwise world-bbox overlap volumes (post-H1):")
    ids = sorted(world_bboxes.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_mn, a_mx = world_bboxes[ids[i]]
            b_mn, b_mx = world_bboxes[ids[j]]
            ov_mn = np.maximum(a_mn, b_mn)
            ov_mx = np.minimum(a_mx, b_mx)
            ov_size = np.maximum(0.0, ov_mx - ov_mn)
            ov_vol = float(np.prod(ov_size))
            print(
                f"    image{ids[i]} ∩ image{ids[j]}: overlap size={fmt_vec(ov_size)} "
                f"volume={ov_vol:.3f}"
            )

    # 4) Also evaluate the camera centroid spread vs. world cloud bbox.
    print("\n  union world bbox (post-H1):")
    all_mn = np.min(np.stack([b[0] for b in world_bboxes.values()]), axis=0)
    all_mx = np.max(np.stack([b[1] for b in world_bboxes.values()]), axis=0)
    print(f"    min={fmt_vec(all_mn)}")
    print(f"    max={fmt_vec(all_mx)}")
    print(f"    size={fmt_vec(all_mx - all_mn)}")
    cam_positions = np.stack([poses[i][:3, 3] for i in ids])
    print(f"  camera positions (cam->world translation column):")
    for i, p in zip(ids, cam_positions):
        print(f"    image{i}: {fmt_vec(p)}")
    print(f"  cam centroid: {fmt_vec(cam_positions.mean(axis=0))}")
    print(f"  cam spread:   {fmt_vec(cam_positions.max(axis=0) - cam_positions.min(axis=0))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
