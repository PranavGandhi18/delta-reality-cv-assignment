"""
Sanity-check the transformed point clouds by rendering a downsampled
preview. We don't try to load all ~9M points into matplotlib — that would
be both slow and pointless. Instead we pull a random subset of each cloud
(default 30k per cloud) and render them with their per-point colors.

Run:
    python scripts/visualize.py            # uses data/output/
    python scripts/visualize.py --src      # renders the originals instead

Writes a PNG to `data/output/preview_xyz.png` (and `_yz`, `_xz`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "output"
SRC_DIR = REPO / "ComputerVisionAssignment_Data" / "StreamingAssets" / "Points"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        action="store_true",
        help="Render the original (un-transformed) PLYs in data/...StreamingAssets/Points.",
    )
    p.add_argument(
        "--n-sample",
        type=int,
        default=30_000,
        help="Number of random vertices per PLY to render (default 30000).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR,
        help=f"Directory to write previews. Default: {OUT_DIR}",
    )
    return p.parse_args()


def load_ply_subset(path: Path, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Read ASCII PLY header to find vertex count, then read the body and
    randomly subsample N rows. We still parse the full body here because
    streaming an ASCII PLY randomly is not worth the complexity."""
    with open(path) as f:
        n_vertex = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError("EOF in header")
            line = line.strip()
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            elif line == "end_header":
                break
        data = np.loadtxt(f, dtype=np.float64)
    xyz = data[:, 0:3]
    rgb = np.clip(data[:, 3:6] / 255.0, 0, 1)

    if data.shape[0] > n:
        idx = rng.choice(data.shape[0], size=n, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
    return xyz, rgb


def main() -> int:
    args = parse_args()
    src = SRC_DIR if args.src else args.out
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Load and subsample each cloud.
    clouds: list[tuple[np.ndarray, np.ndarray, str]] = []
    cam_positions: list[np.ndarray] = []
    for i in (1, 2, 3):
        ply = src / f"image{i}.ply"
        if not ply.exists():
            print(f"missing: {ply}", file=sys.stderr)
            continue
        print(f"loading {ply.name} (subsample={args.n_sample})")
        xyz, rgb = load_ply_subset(ply, args.n_sample, rng)
        clouds.append((xyz, rgb, f"image{i}"))

    # Load camera positions from traj.txt (which lives one level up).
    traj = src.parent / "traj.txt" if args.src else (args.out / "traj.txt")
    if traj.exists():
        with open(traj) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                T = np.array([float(x) for x in parts]).reshape(4, 4)
                cam_positions.append(T[:3, 3])
    cam_pos = np.array(cam_positions) if cam_positions else np.zeros((0, 3))

    # Three orthographic projections: XY, XZ, YZ.
    projections = [
        ("xy", 0, 1, "X", "Y"),
        ("xz", 0, 2, "X", "Z"),
        ("yz", 1, 2, "Y", "Z"),
    ]
    for tag, ax_a, ax_b, label_a, label_b in projections:
        fig, ax = plt.subplots(figsize=(8, 8))
        for xyz, rgb, label in clouds:
            ax.scatter(xyz[:, ax_a], xyz[:, ax_b], c=rgb, s=0.5, label=label)
        if cam_pos.size:
            ax.scatter(
                cam_pos[:, ax_a],
                cam_pos[:, ax_b],
                marker="x",
                s=120,
                c="red",
                label="cameras",
            )
            for k, p in enumerate(cam_pos, start=1):
                ax.annotate(
                    f"cam{k}",
                    xy=(p[ax_a], p[ax_b]),
                    xytext=(6, 6),
                    textcoords="offset points",
                    color="red",
                    fontsize=8,
                )
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        tag_src = "src" if args.src else "out"
        ax.set_title(f"{tag.upper()} projection ({tag_src})")
        out = args.out / f"preview_{tag_src}_{tag}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
