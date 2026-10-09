"""Step 3. Calibrate the virtual camera of each mesh on its source image (Sec. 3.2, Eq. 1).

The camera yaw/pitch maximizing the IoU between the rendered silhouette and the segmentation mask is found by a
grid search (scipy brute, 20 then 40 samples per axis if IoU < 0.7) refined with Nelder-Mead.
The pose and IoU are written in a CSV next to every mesh (`<mesh>.csv`), read by render_views.py.

python generation/calibrate_pose.py --mesh-root meshes/AG-ReID --segmented-root AG-ReID_segmented meshes/AG-ReID/**/*.glb
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import optimize  # noqa: E402

RADIUS, FOV, ASPECT_RATIO = 3.5, 60, 1.0


def get_camera_pose(yaw_deg, pitch_deg, roll_deg, radius):
    """Camera on a sphere of given radius looking at the origin (yaw around Y, pitch = elevation)."""
    theta, phi, psi = map(np.radians, [yaw_deg, pitch_deg, roll_deg])
    eye = radius * np.array([np.cos(phi) * np.sin(theta), np.sin(phi), np.cos(phi) * np.cos(theta)])

    forward = eye / np.linalg.norm(eye)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    if psi != 0:
        right, up = right * np.cos(psi) + up * np.sin(psi), -right * np.sin(psi) + up * np.cos(psi)

    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, up, forward, eye
    return pose


def crop_to_content(background_mask):
    rows = np.where(np.any(~background_mask, axis=1))[0]
    cols = np.where(np.any(~background_mask, axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Empty mask")
    return background_mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def compute_iou(rendered_silhouette, reference_rgba):
    """IoU of the two masks, each cropped to the object bounding box and resized to 512x512."""
    gray = cv2.cvtColor(rendered_silhouette, cv2.COLOR_RGB2GRAY)
    rendered = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1] > 0  # True = background
    reference = (255 - reference_rgba[:, :, 3]) > 0  # True = background

    rendered = cv2.resize(crop_to_content(rendered).astype(np.uint8), (512, 512)) > 0
    reference = cv2.resize(crop_to_content(reference).astype(np.uint8), (512, 512)) > 0

    union = np.logical_or(rendered, reference).sum()
    return np.logical_and(rendered, reference).sum() / union if union > 0 else 0


def render_silhouette(yaw, pitch, mesh, renderer):
    scene = pyrender.Scene(ambient_light=(1.0, 1.0, 1.0))
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(FOV), aspectRatio=ASPECT_RATIO)
    scene.add_node(pyrender.Node(camera=camera, matrix=get_camera_pose(yaw, pitch, 0, RADIUS)))
    _, depth = renderer.render(scene)
    return np.stack([np.where(depth == 0, 255, 0)] * 3, axis=-1).astype(np.uint8)


def calibrate(mesh_file, image_file, renderer, threshold=0.7):
    mesh = trimesh.load(mesh_file, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    mesh = pyrender.Mesh.from_trimesh(mesh)
    reference = np.array(Image.open(image_file).convert("RGBA"))

    best = {"iou": 0.0, "pose": (0.0, 0.0)}

    def objective(params):
        iou = compute_iou(render_silhouette(params[0], params[1], mesh, renderer), reference)
        if iou > best["iou"]:
            best["iou"], best["pose"] = iou, (params[0], params[1])
        return -iou

    ranges = ((-90, 90), (-90, 90))  # yaw, pitch
    optimize.brute(objective, ranges, Ns=20, finish=optimize.fmin, workers=1)
    if best["iou"] < threshold:
        optimize.brute(objective, ranges, Ns=40, finish=optimize.fmin, workers=1)

    yaw, pitch = best["pose"]
    return yaw, pitch, compute_iou(render_silhouette(yaw, pitch, mesh, renderer), reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", type=Path, required=True, help="meshes are matched to masks by relative path")
    parser.add_argument("--segmented-root", type=Path, required=True, help="output of segment.py")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("meshes", nargs="+", type=Path)
    args = parser.parse_args()

    renderer = pyrender.OffscreenRenderer(512, 512)
    for mesh_file in args.meshes:
        csv_path = mesh_file.with_suffix(".csv")
        if csv_path.exists() and not args.force:
            continue
        image_file = args.segmented_root / mesh_file.relative_to(args.mesh_root).with_suffix(".png")
        try:
            yaw, pitch, iou = calibrate(mesh_file, image_file, renderer)
        except Exception as e:
            print(f"Failed on {mesh_file}: {e}")
            continue
        csv_path.write_text(
            "Model,Image,Yaw,Pitch,Roll,Radius,FOV,Aspect Ratio,IOU\n"
            f"{mesh_file},{image_file},{yaw:.2f},{pitch:.2f},0.00,{RADIUS:.2f},{FOV:.2f},{ASPECT_RATIO:.2f},{iou:.4f}\n"
        )
        print(f"{mesh_file}: yaw={yaw:.1f} pitch={pitch:.1f} IoU={iou:.4f}")


if __name__ == "__main__":
    main()
