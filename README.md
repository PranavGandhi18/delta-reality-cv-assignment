# Computer Vision Assignment — Coordinate System Fix

Submission for the Delta Reality computer vision assignment.

The task: the supplied Unity viewer (`ComputerVisionAssignment.x86_64`) loads
three point clouds (`image{1,2,3}.ply`) and a camera trajectory (`traj.txt`),
but the data and the viewer disagree on coordinate conventions, so the viewer
shows nothing meaningful. We figure out the transformation, apply it to the
three point clouds and the trajectory, and emit replacement files that the
viewer can render directly.

A detailed development log — including everything tried, every dead end, and
the reasoning behind the final transformation — is in
[`execute_plan.md`](./execute_plan.md). Read it for the *why*; read this
README for the *how to run*.

---

## 1. Prerequisites

- A Linux x86_64 host **to run the supplied Unity viewer** (the viewer
  binary is Linux-only). The transformation script itself runs anywhere
  Python runs — including macOS, which is where this was developed.
- `conda` (Miniconda or Anaconda).
- The original assignment package, unzipped, in this directory. In
  particular this layout must exist:
  ```
  ComputerVisionAssignment.x86_64
  ComputerVisionAssignment_Data/
    StreamingAssets/
      Points/
        image1.ply  image1.png  image1_depth.png  image1_rays.png
        image2.ply  image2.png  image2_depth.png  image2_rays.png
        image3.ply  image3.png  image3_depth.png  image3_rays.png
      traj.txt
  ```
  The large binaries (`UnityPlayer.so`, the original ~125 MB PLYs, etc.) are
  gitignored, so after cloning you need to re-place the upstream package on
  top of this checkout.

## 2. Set up the conda environment

```bash
conda create -n cv_assignment python=3.11 -y
conda activate cv_assignment
pip install numpy plyfile pillow scipy
```

Pinned versions used during development:
- python 3.11
- numpy 2.4.6
- plyfile 1.1.4
- pillow 12.2.0
- scipy 1.17.1
- matplotlib 3.10.9 (for the static XY/XZ/YZ preview PNGs)
- open3d 0.19.0 (for the interactive on-Mac preview window)

## 3. Run the transformation

```bash
conda activate cv_assignment
python scripts/transform.py
```

This reads the original PLYs and `traj.txt` from
`ComputerVisionAssignment_Data/StreamingAssets/`, applies the
coordinate-system fix described in `execute_plan.md`, and writes corrected
copies into `data/output/`.

## 3a. Quick preview on macOS (no Linux machine needed)

The supplied Unity viewer is a Linux x86_64 binary, so you can't run it
natively on macOS. To still get a visual confirmation that the
transformation is correct, the repo ships an interactive Open3D viewer
that loads the same outputs the Unity viewer would, plus camera frusta
for each pose in `traj.txt`:

```bash
conda activate cv_assignment
python scripts/preview_o3d.py
```

This opens a window with:
- The three transformed point clouds (RGB-colored).
- Three camera frusta drawn in red / green / blue at the cam1 / cam2 /
  cam3 positions, oriented along each camera's stored look-direction.
- An RGB axis triad at the world origin.

Controls (Open3D defaults): left-drag to rotate, right-drag to pan,
scroll to zoom, `R` to reset, `H` for the full help printed to the
terminal. The preview defaults to 2 cm voxel downsampling to stay
responsive; pass `--downsample 0` to render every point or
`--downsample 0.05` to render fewer.

To compare against the un-transformed source instead, pass `--src`.

A static (non-interactive) version of the same comparison is also
available as PNGs:

```bash
python scripts/visualize.py            # XY / XZ / YZ projections of the output
python scripts/visualize.py --src      # same projections, source data
```

The PNGs land in `data/output/preview_*.png`.

## 4. View the result in the Delta Reality viewer (Linux only)

Copy the outputs into the locations the viewer reads from:

```bash
cp data/output/image1.ply ComputerVisionAssignment_Data/StreamingAssets/Points/
cp data/output/image2.ply ComputerVisionAssignment_Data/StreamingAssets/Points/
cp data/output/image3.ply ComputerVisionAssignment_Data/StreamingAssets/Points/
cp data/output/traj.txt   ComputerVisionAssignment_Data/StreamingAssets/
```

(Back up the originals first if you want to compare.) Then on a Linux host:

```bash
chmod +x ComputerVisionAssignment.x86_64
./ComputerVisionAssignment.x86_64
```

If you don't have a Linux machine handy, use **§3a** instead — the
Open3D preview shows the same geometry in an interactive window on macOS
(or Windows, or any OS where Open3D runs).

Viewer controls (from the assignment PDF):
- Hold the **right mouse button** to rotate.
- While holding RMB: `W/A/S/D` to move horizontally, `Q/E` to move up/down.

## 5. Approach (one-paragraph version)

The three PLYs are dense, single-image-derived point clouds in their
respective camera-local frames; `traj.txt` gives the camera-to-world
pose for each. The viewer expects every PLY in *world* coordinates.
Three operations on top of the cam→world bake:

```
S           = diag(1, 1, -1)          # flip Z so cam-forward = Unity +Z
delta       = -(S · cam2_world_pos)   # translate cam2 to origin
R_level     = Rodrigues(axis=cam2_forward, angle=cam2_roll)
                                      # rotate about cam2 forward to
                                      # cancel cam2's photographer roll
M_world     = R_level · T_translate(delta) · S

p_view = M_world · T_cw · p_local              # per point cloud
T_view = M_world · T_cw                        # per trajectory pose
```

Cam2 ends up at world origin looking `+Z`, image-up exactly `+Y`, so
Unity's default user camera at `(0, 1, -10)` looking `+Z` spawns
behind cam2 with the panorama upright and filling the view. Cam2 is
the anchor because it has the smallest natural roll (16°) of the
three; the leveling cancels even that. The other two cameras keep
their own photographer rolls (51° for cam1, 23° for cam3) — those
are real properties of the source photos. Full reasoning, including
the five earlier approaches that didn't work and what each taught us,
is in [`execute_plan.md`](./execute_plan.md) §12–§14.

## 5a. CLI knobs

```bash
python scripts/transform.py                  # default: anchor cam2, leveled
python scripts/transform.py --center-on cam1 # anchor on cam1 instead
python scripts/transform.py --center-on cam3 # anchor on cam3 instead
python scripts/transform.py --no-level       # skip the leveling rotation
python scripts/transform.py --limit 100000   # smoke test on a small subset
```

Output is always ASCII PLY (matching the source format exactly, ~114 MB
each). `scripts/visualize.py` renders XY/XZ/YZ orthographic projections
into `data/output/preview_*.png`. `scripts/preview_o3d.py` opens an
interactive Open3D window from any OS.

## 6. References

- The Unity coordinate-system conventions (left-handed, Y-up):
  https://docs.unity3d.com/Manual/class-Transform.html
- OpenCV / COLMAP camera convention (right-handed, X-right Y-down Z-forward):
  https://colmap.github.io/format.html
- PLY format reference: http://paulbourke.net/dataformats/ply/
- `plyfile` library for reading/writing PLY in Python:
  https://github.com/dranjan/python-plyfile
