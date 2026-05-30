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

## 3. Run the transformation

```bash
conda activate cv_assignment
python scripts/transform.py
```

This reads the original PLYs and `traj.txt` from
`ComputerVisionAssignment_Data/StreamingAssets/`, applies the
coordinate-system fix described in `execute_plan.md`, and writes corrected
copies into `data/output/`.

## 4. View the result

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

Viewer controls (from the assignment PDF):
- Hold the **right mouse button** to rotate.
- While holding RMB: `W/A/S/D` to move horizontally, `Q/E` to move up/down.

## 5. Approach (one-paragraph version)

The three PLYs are dense, single-image-derived point clouds in their
respective camera-local frames; `traj.txt` gives the camera-to-world pose for
each. The viewer expects every PLY to already be in *world* coordinates. On
top of that, the source pipeline is right-handed (OpenCV / COLMAP camera
convention) while the Unity viewer is left-handed (Y-up, Z-forward). So the
correction is, with `S_world = diag(1, 1, -1, 1)` and `S_cam = diag(1, -1, 1, 1)`:

```
p_view = S_world · T_cw · p_local            # per point cloud
T_view = S_world · T_cw · S_cam              # per trajectory pose
```

`S_world` flips world handedness (negate Z); `S_cam` converts the camera
frame from OpenCV (Y-down) to Unity (Y-up). Points only need `S_world`
because they're already in source-camera coordinates; the trajectory matrix
needs both because it terminates in a camera frame the viewer will
interpret. Full reasoning, including how we verified handedness
experimentally and the visual sanity checks done on macOS, is in
[`execute_plan.md`](./execute_plan.md).

## 5a. CLI knobs for fallback hypotheses

If the viewer renders the scene wrong, try these without editing code:

```bash
python scripts/transform.py --flip none      # source was already LH
python scripts/transform.py --flip y         # the up-axis is different
python scripts/transform.py --cam-flip none  # source camera was already Y-up
python scripts/transform.py --binary         # smaller / faster PLY output
python scripts/transform.py --limit 100000   # smoke test on a small subset
```

`scripts/visualize.py` renders XY/XZ/YZ orthographic projections of the
output (or `--src` for the originals) into `data/output/preview_*.png`.

## 6. References

- The Unity coordinate-system conventions (left-handed, Y-up):
  https://docs.unity3d.com/Manual/class-Transform.html
- OpenCV / COLMAP camera convention (right-handed, X-right Y-down Z-forward):
  https://colmap.github.io/format.html
- PLY format reference: http://paulbourke.net/dataformats/ply/
- `plyfile` library for reading/writing PLY in Python:
  https://github.com/dranjan/python-plyfile
