"""
Convert the supplied PLY point clouds and `traj.txt` from the source
coordinate system into the viewer's (Unity, left-handed Y-up) coordinate
system.

Pipeline per point cloud (see execute_plan.md sec 7 for derivation):

    p_world_source = T_cw · p_local                 # bake cam->world
    p_world_viewer = S · p_world_source             # handedness flip
                                                    # default S = diag(1, 1, -1, 1)

And per trajectory pose:

    T_viewer = S · T_cw · S                         # conjugate by S

`--flip x|y|z|none` lets the user try alternates. `--ascii / --binary`
toggles the output PLY format (default ascii, matching the source).

Run (from repo root):
    python scripts/transform.py
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_STREAM = REPO / "ComputerVisionAssignment_Data" / "StreamingAssets"
DEFAULT_OUT = REPO / "data" / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in-stream",
        type=Path,
        default=DEFAULT_STREAM,
        help="Path to StreamingAssets/ (contains Points/ + traj.txt). "
        f"Default: {DEFAULT_STREAM}",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory. Default: {DEFAULT_OUT}",
    )
    p.add_argument(
        "--flip",
        choices=("x", "y", "z", "none"),
        default="z",
        help="Which world axis to negate when converting RH source -> LH "
        "Unity viewer. Default: z.",
    )
    p.add_argument(
        "--cam-flip",
        choices=("x", "y", "z", "none"),
        default="y",
        help="Which camera-local axis to negate when converting the "
        "trajectory matrix from source camera convention (OpenCV: Y-down) "
        "to viewer camera convention (Unity: Y-up). Only affects "
        "traj.txt, not the point cloud transform. Default: y.",
    )
    p.add_argument(
        "--binary",
        action="store_true",
        help="Emit binary little-endian PLY (faster, smaller). "
        "Default is ASCII to match source format exactly.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N vertices per PLY. Useful for quick "
        "smoke tests; omit for full output.",
    )
    p.add_argument(
        "--upright",
        action="store_true",
        help="After the world flip, compute the average camera 'up' axis "
        "across the three poses and rotate the world so that average lands "
        "on (0, 1, 0). Fixes the case where the source data's true vertical "
        "axis is not exactly aligned with viewer Y, which manifests in the "
        "Unity viewer as a scene you can't level-orbit to upright.",
    )
    return p.parse_args()


def flip_matrix(flip: str) -> np.ndarray:
    """Return the 4x4 reflection matrix S for the chosen axis flip."""
    if flip == "none":
        return np.eye(4)
    S = np.eye(4)
    idx = {"x": 0, "y": 1, "z": 2}[flip]
    S[idx, idx] = -1.0
    return S


def axis_align_rotation(from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
    """4x4 rotation matrix that maps `from_vec` direction to `to_vec`.

    Rodrigues formula. Both vectors are auto-normalised. Falls back to
    identity if they're already aligned and to a 180° flip about an
    arbitrary perpendicular axis if they're antiparallel.
    """
    a = from_vec / np.linalg.norm(from_vec)
    b = to_vec / np.linalg.norm(to_vec)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-10:
        if c > 0:
            return np.eye(4)
        # antiparallel: 180° about any axis perpendicular to a
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        R3 = np.eye(3) + 2 * (K @ K)
    else:
        K = np.array(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ]
        )
        R3 = np.eye(3) + K + (K @ K) * ((1 - c) / (s * s))
    R = np.eye(4)
    R[:3, :3] = R3
    return R


def compute_upright_rotation(
    poses_after_world_flip: list[np.ndarray],
) -> np.ndarray:
    """Given the per-camera S_world · T_cw matrices (so the rotation 3x3 is in
    viewer-world coords already), compute the leveling rotation that maps
    the average camera 'up' axis (the camera frame's +Y axis after S_cam,
    but since S_cam is applied later in transform_pose, we use the
    OpenCV-style 'up' = -col_1 of the rotation here) to viewer (0, 1, 0).

    Concretely: each pose's rotation 3x3 has column 1 = camera +Y axis in
    viewer-world. In OpenCV camera convention that axis points DOWN in
    image space, so 'up' = -column_1. We average across the cameras and
    Rodrigues-rotate that average to (0, 1, 0).
    """
    ups = []
    for T in poses_after_world_flip:
        up = -T[:3, 1]   # OpenCV cam +Y is image-down; up is its negation
        ups.append(up)
    avg_up = np.mean(np.stack(ups, axis=0), axis=0)
    avg_up /= np.linalg.norm(avg_up)
    print(f"  avg camera-up axis in viewer-world: {avg_up.round(4).tolist()}")
    R = axis_align_rotation(avg_up, np.array([0.0, 1.0, 0.0]))
    return R


def load_traj(path: Path) -> dict[int, np.ndarray]:
    """Load traj.txt -> {image_id (1-based): 4x4 matrix}.

    Each row of the file is 16 row-major floats. The image index is
    implicit in the row number.
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


def write_traj(path: Path, poses: dict[int, np.ndarray]) -> None:
    """Write traj.txt using the same row-major 16-float layout the loader
    consumes. We match the source's scientific-notation float format
    (18 digits, e-notation) so any byte-level diff stays small."""
    with open(path, "w") as f:
        for _, T in sorted(poses.items()):
            row = " ".join(f"{v:.18e}" for v in T.reshape(-1))
            f.write(row + "\n")


def read_ply_ascii(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read an ASCII PLY with `float x, float y, float z, uchar r, g, b`.

    Returns (xyz [N,3] float32, rgb [N,3] uint8, n_vertex).
    """
    with open(path) as f:
        n_vertex = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF in PLY header")
            line = line.strip()
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            elif line == "end_header":
                break
        if n_vertex is None:
            raise ValueError("Missing 'element vertex' in header")
        data = np.loadtxt(f, dtype=np.float64)

    xyz = data[:, 0:3].astype(np.float32)
    rgb = np.clip(np.round(data[:, 3:6]), 0, 255).astype(np.uint8)
    return xyz, rgb, n_vertex


def write_ply_ascii(
    path: Path, xyz: np.ndarray, rgb: np.ndarray, *, comments: list[str] | None = None
) -> None:
    """Write an ASCII PLY with the same property layout as the source."""
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        if comments:
            for c in comments:
                # PLY comments must be single-line
                for line in c.splitlines():
                    f.write(f"comment {line}\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Stream-write to avoid building one big string in memory.
        # Match the source's ~9-digit float precision.
        chunk = 200_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xs = xyz[start:end]
            cs = rgb[start:end]
            buf = []
            for (x, y, z), (r, g, b) in zip(xs, cs):
                buf.append(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            f.write("".join(buf))


def write_ply_binary(
    path: Path, xyz: np.ndarray, rgb: np.ndarray, *, comments: list[str] | None = None
) -> None:
    """Write a binary_little_endian PLY with the same property layout."""
    n = xyz.shape[0]
    with open(path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        if comments:
            for c in comments:
                for line in c.splitlines():
                    f.write(f"comment {line}\n".encode("utf-8"))
        f.write(f"element vertex {n}\n".encode("utf-8"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")

        # Pack each vertex as 3f + 3B = 12 + 3 = 15 bytes.
        dt = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ]
        )
        arr = np.empty(n, dtype=dt)
        arr["x"] = xyz[:, 0]
        arr["y"] = xyz[:, 1]
        arr["z"] = xyz[:, 2]
        arr["r"] = rgb[:, 0]
        arr["g"] = rgb[:, 1]
        arr["b"] = rgb[:, 2]
        f.write(arr.tobytes(order="C"))


def transform_cloud(xyz: np.ndarray, T_cw: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Apply M = S · T_cw to Nx3 local-frame points -> Nx3 viewer-frame
    points."""
    M = (S @ T_cw).astype(np.float64)
    R = M[:3, :3]
    t = M[:3, 3]
    return (xyz.astype(np.float64) @ R.T + t).astype(np.float32)


def transform_pose(T_cw: np.ndarray, S_world: np.ndarray, S_cam: np.ndarray) -> np.ndarray:
    """Convert a source-camera -> source-world matrix into a
    viewer-camera -> viewer-world matrix.

    T_viewer = S_world · T_cw · S_cam.

    Derivation: source-world points relate to viewer-world points by
    `p_view_w = S_world · p_src_w`, and source-camera points relate to
    viewer-camera points by `p_src_c = S_cam · p_view_c` (S_cam is its
    own inverse). Substituting `p_src_w = T_cw · p_src_c` and chaining
    gives T_viewer.

    Both S_world and S_cam are reflections (det -1); their product has
    det +1, so T_viewer remains a proper rigid transform (rotation +
    translation)."""
    return S_world @ T_cw @ S_cam


def main() -> int:
    args = parse_args()
    stream = args.in_stream
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    traj_in = stream / "traj.txt"
    points_in = stream / "Points"
    if not traj_in.exists():
        print(f"ERROR: missing {traj_in}", file=sys.stderr)
        return 2
    if not points_in.exists():
        print(f"ERROR: missing {points_in}", file=sys.stderr)
        return 2

    S_world = flip_matrix(args.flip)
    S_cam = flip_matrix(args.cam_flip)
    print(f"World flip: {args.flip!r}\n{S_world}")
    print(f"Camera-frame flip (for traj.txt only): {args.cam_flip!r}\n{S_cam}\n")

    # 1.5) Optional upright leveling. Compute it BEFORE loading any PLY so we
    # have a single composed world transform `M_world = R_upright · S_world`.
    R_upright = np.eye(4)

    # 1) Load source trajectories.
    poses = load_traj(traj_in)
    print(f"Loaded {len(poses)} poses from {traj_in}")

    if args.upright:
        print("\nComputing upright leveling rotation:")
        # poses_after_world_flip[i] = S_world @ poses[i]   (3x3 part is what
        # we use to extract the cam-up axis in viewer-world coords)
        poses_after_world_flip = [S_world @ T for T in poses.values()]
        R_upright = compute_upright_rotation(poses_after_world_flip)
        print(f"  upright rotation (3x3):\n{R_upright[:3, :3].round(4)}")

    # Composed world transform used for points: M_world = R_upright @ S_world.
    M_world = R_upright @ S_world

    # 2) Per-image: load PLY, transform, write.
    new_poses: dict[int, np.ndarray] = {}
    for img_id, T_cw in sorted(poses.items()):
        ply_in = points_in / f"image{img_id}.ply"
        if not ply_in.exists():
            print(f"  WARN: skipping {img_id} (no {ply_in})")
            continue
        ply_out = out / f"image{img_id}.ply"

        print(
            f"\n[image{img_id}] reading {ply_in.name} "
            f"({os.path.getsize(ply_in)/1e6:.1f} MB)..."
        )
        t0 = time.time()
        xyz, rgb, n = read_ply_ascii(ply_in)
        if args.limit is not None:
            xyz = xyz[: args.limit]
            rgb = rgb[: args.limit]
            n = xyz.shape[0]
        t_read = time.time() - t0

        print(
            f"  parsed {n} verts in {t_read:.1f}s; "
            f"local bbox = [{xyz.min(axis=0)} .. {xyz.max(axis=0)}]"
        )

        t1 = time.time()
        # Use the composed M_world = R_upright @ S_world so points get
        # leveled too.
        xyz_world = transform_cloud(xyz, T_cw, M_world)
        t_xform = time.time() - t1
        print(
            f"  transformed in {t_xform:.1f}s; "
            f"viewer bbox = [{xyz_world.min(axis=0)} .. {xyz_world.max(axis=0)}]"
        )

        comments = [
            f"transformed by transform.py (flip={args.flip})",
            f"source = {ply_in.name}",
            "p_viewer = S * T_cw * p_local; cam->world baked into points",
        ]
        t2 = time.time()
        if args.binary:
            write_ply_binary(ply_out, xyz_world, rgb, comments=comments)
        else:
            write_ply_ascii(ply_out, xyz_world, rgb, comments=comments)
        t_write = time.time() - t2
        print(
            f"  wrote {ply_out} ({os.path.getsize(ply_out)/1e6:.1f} MB) "
            f"in {t_write:.1f}s"
        )

        # For traj.txt we use the SAME composed world transform on the left
        # so cameras get leveled identically to the points.
        new_poses[img_id] = transform_pose(T_cw, M_world, S_cam)

    # 3) Write transformed traj.txt.
    traj_out = out / "traj.txt"
    write_traj(traj_out, new_poses)
    print(f"\nWrote {traj_out}")
    for img_id, T in sorted(new_poses.items()):
        print(
            f"  image{img_id} cam_pos (viewer space): "
            f"{T[:3, 3].round(3).tolist()}"
        )

    print("\nDone. Copy outputs into the viewer's StreamingAssets to test:")
    print(
        f"  cp {out}/image*.ply "
        f"{stream}/Points/"
    )
    print(f"  cp {out}/traj.txt {stream}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
