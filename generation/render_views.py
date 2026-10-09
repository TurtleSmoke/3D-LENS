"""Step 4. Render geometrically-consistent novel views of each calibrated mesh (Sec. 3.2).

Meshes whose calibrated IoU is below tau_iou = 0.7 are discarded. Each remaining mesh is rendered N = 10 times
from its calibrated pose shifted by
    - an elevation shift linearly spaced in [0, 30] deg toward the other viewpoint-domain
      (ground source -> camera moves up, aerial source -> camera moves down),
      view _00 is the source elevation and view _09 the largest shift (used by the curriculum),
    - an azimuth shift sampled uniformly in [-30, 30] deg.
Renders are foreground-only RGBA PNGs saved as <output>/<relative mesh path>_XX.png

python generation/render_views.py --mesh-root meshes/AG-ReID --output AG-ReID_renders meshes/AG-ReID/**/*.glb
"""

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

from calibrate_pose import get_camera_pose  # noqa: E402

IOU_THRESHOLD = 0.7
NUM_VIEWS = 10
MAX_ELEVATION_SHIFT = 30
MAX_AZIMUTH_SHIFT = 30


def is_aerial(path):
    """Viewpoint-domain of the source image. AG-ReID / AG-ReID.v2 camera C0 is the UAV, MOO top views are frames >= 64."""
    name = Path(path).name
    match = re.search(r"cow\d+_(\d+)", name)
    if match:
        return int(match.group(1)) >= 64
    return "C0" in name


def add_lights(scene, cam_pose):
    key = np.copy(cam_pose)
    key[:3, 3] += [0.5, 0.5, 0.0]
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0), pose=key)
    fill = np.copy(cam_pose)
    fill[:3, 3] += [-0.3, 0.0, -0.3]
    scene.add(pyrender.DirectionalLight(color=[0.8, 0.8, 0.9], intensity=1.5), pose=fill)
    rim = np.copy(cam_pose)
    rim[:3, 3] += [0.0, -0.5, -0.3]
    scene.add(
        pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=1.0, innerConeAngle=np.pi / 4, outerConeAngle=np.pi / 2),
        pose=rim,
    )


def render(mesh, yaw, pitch, roll, radius, fov, ar, renderer):
    cam_pose = get_camera_pose(yaw, pitch, roll, radius)
    scene = pyrender.Scene(bg_color=[0, 0, 255, 0])
    add_lights(scene, cam_pose)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    scene.add_node(
        pyrender.Node(camera=pyrender.PerspectiveCamera(yfov=np.deg2rad(fov), aspectRatio=ar), matrix=cam_pose)
    )
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)

    # Remove the anti-aliased border, then crop to the object
    mask = binary_erosion(depth > 0, structure=np.ones((3, 3)))
    color = np.array(color)
    color[~mask] = 0
    color[:, :, 3] = mask * 255
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return color[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("meshes", nargs="+", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    renderer = pyrender.OffscreenRenderer(viewport_width=1024, viewport_height=1024)

    for mesh_file in args.meshes:
        csv_path = mesh_file.with_suffix(".csv")
        if not csv_path.exists():
            print(f"No calibration for {mesh_file}, skipped")
            continue
        cam = pd.read_csv(csv_path).iloc[0]
        if cam["IOU"] < IOU_THRESHOLD:
            continue

        yaws = cam["Yaw"] + rng.uniform(-MAX_AZIMUTH_SHIFT, MAX_AZIMUTH_SHIFT, NUM_VIEWS)
        sign = -1 if is_aerial(mesh_file) else 1
        pitches = cam["Pitch"] + sign * np.linspace(0, MAX_ELEVATION_SHIFT, NUM_VIEWS)

        out_stem = args.output / mesh_file.relative_to(args.mesh_root).with_suffix("")
        out_stem.parent.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.load(mesh_file, force="mesh")
        for view, (yaw, pitch) in enumerate(zip(yaws, pitches)):
            out = Path(f"{out_stem}_{view:02d}.png")
            if out.exists() and not args.force:
                continue
            image = render(mesh, yaw, pitch, cam["Roll"], cam["Radius"], cam["FOV"], cam["Aspect Ratio"], renderer)
            if image is None:
                print(f"Empty render for {out}, skipped")
                continue
            Image.fromarray(image).save(out)


if __name__ == "__main__":
    main()
