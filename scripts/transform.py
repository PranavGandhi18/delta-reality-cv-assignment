"""
Convert the supplied PLY point clouds and `traj.txt` into the Unity viewer's
coordinate system.

Pipeline (see execute_plan.md sec 13 for the full derivation and history of
what we tried before):

    S       = diag(1, 1, -1)                # flip Z so cam-forward becomes
                                            # Unity-forward (+Z)
    delta   = -(S @ cam_anchor_world_pos)   # translate so the anchor
                                            # camera lands at world origin

    For each point in image_i.ply:
        p_view = S @ T_cw_i @ p_local + delta

    For each trajectory pose T_i (4x4):
        T_i_view = T_translate(delta) @ S_world @ T_i
            where S_world = diag(1, 1, -1, 1)
                  T_translate is a 4x4 translation by delta

The anchor camera defaults to cam2 (middle photo, smallest roll); pass
`--center-on cam1` or `cam3` if cam2's natural roll looks wrong in the
viewer.

Run (from repo root):
    python scripts/transform.py
    python scripts/transform.py --center-on cam1   # try a different anchor
    python scripts/transform.py --limit 100000     # smoke test on a subset
"""

from __future__ import annotations

import argparse
import os
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
        "--center-on",
        choices=("cam1", "cam2", "cam3"),
        default="cam2",
        help="Which camera's transformed pose to drop at the world origin. "
        "Unity's default user camera spawns near origin looking +Z, so the "
        "anchor camera defines what the user will see when the viewer "
        "opens. cam2 has the smallest natural roll, so it's the default.",
    )
    p.add_argument(
        "--no-level",
        action="store_true",
        help="Skip the leveling rotation that aligns the anchor camera's "
        "image-up axis with viewer +Y. By default we level (rotate about "
        "the anchor's forward axis) so the scene appears upright. Disable "
        "if you specifically want the anchor's natural photographer roll "
        "preserved.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N vertices per PLY. Useful for quick "
        "smoke tests; omit for full output.",
    )
    return p.parse_args()


def load_traj(path: Path) -> dict[int, np.ndarray]:
    """Load traj.txt -> {image_id (1-based): 4x4 matrix}.

    Each non-empty line is 16 row-major floats. The image index is implicit
    (row N corresponds to imageN.ply).
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
    """Write traj.txt in the same row-major 16-floats-per-line format as
    the source."""
    with open(path, "w") as f:
        for _, T in sorted(poses.items()):
            row = " ".join(f"{v:.18e}" for v in T.reshape(-1))
            f.write(row + "\n")


def read_ply_ascii(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Read an ASCII PLY with `float x, y, z, uchar r, g, b`."""
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
    n = xyz.shape[0]
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        if comments:
            for c in comments:
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

        chunk = 200_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            xs = xyz[start:end]
            cs = rgb[start:end]
            buf = []
            for (x, y, z), (r, g, b) in zip(xs, cs):
                buf.append(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            f.write("".join(buf))


def build_recentering_transform(
    poses: dict[int, np.ndarray], anchor_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (S, T_delta) such that S @ T_anchor @ p_local + delta lands
    the anchor camera's position at the world origin and its forward axis
    along world +Z.

    S = diag(1, 1, -1, 1) — flip Z so cam-forward becomes Unity-forward.
    T_delta is a pure translation 4x4 so that the anchor's transformed
    position is the origin.
    """
    S = np.diag([1.0, 1.0, -1.0, 1.0])
    anchor_T = poses[anchor_id]
    anchor_pos_world = anchor_T[:3, 3]
    anchor_pos_after_flip = S[:3, :3] @ anchor_pos_world
    delta = -anchor_pos_after_flip
    T_delta = np.eye(4)
    T_delta[:3, 3] = delta
    return S, T_delta


def compute_leveling_rotation(T_anchor_view: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the 4x4 rotation about the anchor camera's forward axis
    that maps the anchor's image-up axis exactly to viewer +Y.

    The anchor's natural photographer roll (≈16° for cam2 in this
    dataset) leaks into the scene as a world-level tilt — every wall
    appears rolled by that amount because Unity's player camera is
    locked to Y-up. This rotation cancels it *without* changing where
    any camera looks: the rotation axis is the anchor's forward axis,
    so cam2's view direction (and hence what appears in front of the
    user at spawn) is identical before and after.

    Implementation note: we use Rodrigues directly with the anchor's
    forward axis as the rotation axis. We can't construct a "target
    leveled frame" and take R_target @ R_current.T because after the
    `S = diag(1,1,-1)` flip the camera triad becomes effectively
    left-handed (right × down = -forward), and an RH cross product of
    `down × forward` returns the wrong sign of `right`. Rodrigues
    sidesteps that — it just rotates everything in the plane
    perpendicular to forward by the signed roll angle.

    Returns (R_level_4x4, roll_angle_degrees) for logging.
    """
    R_current = T_anchor_view[:3, :3]
    forward = R_current[:, 2]
    forward = forward / np.linalg.norm(forward)

    # OpenCV col1 = image-down; image-up is its negation.
    up_current = -R_current[:, 1]

    world_up = np.array([0.0, 1.0, 0.0])
    up_target_perp = world_up - np.dot(world_up, forward) * forward
    norm = np.linalg.norm(up_target_perp)
    if norm < 1e-6:
        # Anchor is looking straight up or down; can't level about its
        # forward axis. Skip.
        return np.eye(4), 0.0
    up_target = up_target_perp / norm

    # Signed angle from up_current to up_target in the plane
    # perpendicular to forward.
    cos_a = float(np.clip(np.dot(up_current, up_target), -1.0, 1.0))
    sin_a = float(np.dot(np.cross(up_current, up_target), forward))
    angle = float(np.arctan2(sin_a, cos_a))

    # Rodrigues rotation about `forward` by `angle`.
    K = np.array(
        [
            [0.0, -forward[2], forward[1]],
            [forward[2], 0.0, -forward[0]],
            [-forward[1], forward[0], 0.0],
        ]
    )
    R_level_3 = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    R_level_4 = np.eye(4)
    R_level_4[:3, :3] = R_level_3
    return R_level_4, float(np.degrees(abs(angle)))


def transform_cloud_points(
    xyz_local: np.ndarray, T_cw: np.ndarray, M_world: np.ndarray
) -> np.ndarray:
    """Apply the full pipeline: cam-local -> source-world -> M_world.
    M_world bundles S_z, T_delta, and (optionally) R_level. Returns Nx3
    float32 in viewer-world space."""
    M = (M_world @ T_cw).astype(np.float64)
    R = M[:3, :3]
    t = M[:3, 3]
    return (xyz_local.astype(np.float64) @ R.T + t).astype(np.float32)


def transform_pose(T_cw: np.ndarray, M_world: np.ndarray) -> np.ndarray:
    """A camera pose stored in `traj.txt` lives in source-world space. We
    apply the SAME world transform on the left so the pose ends up in
    viewer-world. We deliberately do NOT touch the camera-local frame
    (right side) — see execute_plan.md sec 13 for why."""
    return M_world @ T_cw


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

    poses = load_traj(traj_in)
    print(f"Loaded {len(poses)} poses from {traj_in}")

    anchor_id = int(args.center_on[-1])
    if anchor_id not in poses:
        print(f"ERROR: anchor {args.center_on!r} not in traj.txt", file=sys.stderr)
        return 2

    S, T_delta = build_recentering_transform(poses, anchor_id)
    print(f"Anchor: cam{anchor_id} (source pos = {poses[anchor_id][:3,3].round(3).tolist()})")
    print(f"Step 1 — Z flip + translate so cam{anchor_id} lands at world origin:")
    print(f"  S      = diag(1, 1, -1)")
    print(f"  delta  = {T_delta[:3, 3].round(3).tolist()}")

    # Step 2 — level the anchor: rotate about the anchor's forward axis so
    # its image-up axis maps exactly to viewer +Y. Without this, the
    # anchor's natural photographer roll (e.g. 16° for cam2) shows up in
    # Unity as a tilted scene the user can't level-orbit.
    T_anchor_view_unleveled = T_delta @ S @ poses[anchor_id]
    if args.no_level:
        R_level = np.eye(4)
        print(f"Step 2 — leveling DISABLED (--no-level)")
    else:
        R_level, roll = compute_leveling_rotation(T_anchor_view_unleveled)
        print(f"Step 2 — level world about cam{anchor_id}'s forward axis "
              f"(cancels {roll:.1f}° photographer roll):")
        print(f"  R_level = \n{R_level[:3,:3].round(4)}")

    M_world = R_level @ T_delta @ S
    print()

    new_poses: dict[int, np.ndarray] = {}
    for img_id, T_cw in sorted(poses.items()):
        ply_in = points_in / f"image{img_id}.ply"
        if not ply_in.exists():
            print(f"  WARN: skipping {img_id} (no {ply_in})")
            continue
        ply_out = out / f"image{img_id}.ply"

        print(
            f"[image{img_id}] reading {ply_in.name} "
            f"({os.path.getsize(ply_in)/1e6:.1f} MB)..."
        )
        t0 = time.time()
        xyz, rgb, n = read_ply_ascii(ply_in)
        if args.limit is not None:
            xyz = xyz[: args.limit]
            rgb = rgb[: args.limit]
            n = xyz.shape[0]
        print(f"  parsed {n} verts in {time.time()-t0:.1f}s")

        t1 = time.time()
        xyz_view = transform_cloud_points(xyz, T_cw, M_world)
        print(
            f"  transformed in {time.time()-t1:.1f}s; "
            f"viewer bbox = [{xyz_view.min(axis=0).round(2).tolist()}"
            f" .. {xyz_view.max(axis=0).round(2).tolist()}]"
        )

        level_tag = "" if args.no_level else f", leveled about cam{anchor_id} forward"
        comments = [
            f"transformed by transform.py (--center-on cam{anchor_id}{level_tag})",
            f"source = {ply_in.name}",
            "p_view = R_level @ T_delta @ S_z @ T_cw @ p_local",
        ]
        t2 = time.time()
        write_ply_ascii(ply_out, xyz_view, rgb, comments=comments)
        print(
            f"  wrote {ply_out} ({os.path.getsize(ply_out)/1e6:.1f} MB) "
            f"in {time.time()-t2:.1f}s"
        )

        new_poses[img_id] = transform_pose(T_cw, M_world)

    traj_out = out / "traj.txt"
    write_traj(traj_out, new_poses)
    print(f"\nWrote {traj_out}")
    for img_id, T in sorted(new_poses.items()):
        pos = T[:3, 3].round(3).tolist()
        fwd = T[:3, 2].round(3).tolist()
        up = (-T[:3, 1]).round(3).tolist()  # camera up = -OpenCV+Y
        print(f"  image{img_id}: pos={pos}  fwd={fwd}  up={up}")

    print("\nCopy outputs into the viewer's StreamingAssets to test:")
    print(f"  cp {out}/image*.ply {stream}/Points/")
    print(f"  cp {out}/traj.txt {stream}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
