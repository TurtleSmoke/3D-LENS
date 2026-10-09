"""Step 5. Composite every render onto a real background (Sec. 3.2, synthetic-to-real domain alignment).

For each render, a real image from the same camera type as the source image is drawn at random, its person is
removed with LaMa inpainting (mask = dilated segmentation), and the render is pasted at the location and size of
the source person. The drawn real image is also the style reference of StyleID (step 6), so the script writes the
two StyleID file lists <output>/content_<tag>.txt and <output>/style_<tag>.txt (one pair per line).

python generation/composite_background.py --render-root AG-ReID_renders --segmented-root AG-ReID_segmented \
    --real-root AG-ReID --output AG-ReID_composite AG-ReID_renders/**/*.png
"""

import argparse
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from simple_lama_inpainting import SimpleLama


def camera_type(path):
    """AG-ReID / AG-ReID.v2 camera code, C0 = aerial, C2 = ground (wearable), C3 = CCTV."""
    for code, name in (("C0", "aerial"), ("C2", "ground"), ("C3", "cctv")):
        if code in Path(path).name:
            return name
    raise ValueError(f"Unknown camera type in path: {path}")


def remove_person(lama, real_image, segmentation_rgba):
    """Inpaint the (dilated) foreground of a real image to get an empty background."""
    rgb = np.array(real_image)
    alpha = np.array(segmentation_rgba.getchannel("A")) > 0
    mask = binary_dilation(alpha, structure=np.ones((5, 5), dtype=bool)).astype(np.uint8) * 255
    rgb[alpha] = 0
    return lama(Image.fromarray(rgb), Image.fromarray(mask))


def composite(render_rgba, source_segmentation, background):
    """Paste the render at the bounding box of the source person, on a canvas of the source image size."""
    alpha = np.array(source_segmentation.getchannel("A"))
    ys, xs = np.where(alpha > 0)
    y_min, y_max, x_min, x_max = ys.min(), ys.max(), xs.min(), xs.max()

    foreground = np.array(render_rgba).astype(np.float32) / 255.0
    foreground = cv2.resize(foreground, (x_max - x_min + 1, y_max - y_min + 1))
    canvas = np.zeros((source_segmentation.size[1], source_segmentation.size[0], 4), dtype=np.float32)
    canvas[y_min : y_max + 1, x_min : x_max + 1] = foreground

    background = np.array(background).astype(np.float32) / 255.0
    if canvas.shape[:2] != background.shape[:2]:
        canvas = cv2.resize(canvas, (background.shape[1], background.shape[0]))

    image = canvas[:, :, :3] * canvas[:, :, 3:4] + background * (1 - canvas[:, :, 3:4])
    return Image.fromarray((image * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-root", type=Path, required=True, help="output of render_views.py")
    parser.add_argument("--segmented-root", type=Path, required=True, help="output of segment.py on the real images")
    parser.add_argument("--real-root", type=Path, required=True, help="real dataset (JPEG images)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="0", help="suffix of the StyleID lists, to run several shards in parallel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("renders", nargs="+", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    lama = SimpleLama()

    backgrounds = {}
    for path in sorted(args.segmented_root.glob("**/*.png")):
        backgrounds.setdefault(camera_type(path), []).append(path)

    contents, styles = [], []
    for render_path in args.renders:
        relative = render_path.relative_to(args.render_root)
        source = args.segmented_root / re.sub(r"_\d+\.png$", ".png", str(relative))
        output = (args.output / relative).with_suffix(".jpg")

        background_segmentation = random.choice(backgrounds[camera_type(render_path)])
        real_path = (args.real_root / background_segmentation.relative_to(args.segmented_root)).with_suffix(".jpg")
        contents.append(output.resolve())
        styles.append(real_path.resolve())
        if output.exists() and not args.force:
            continue

        background = remove_person(
            lama, Image.open(real_path).convert("RGB"), Image.open(background_segmentation).convert("RGBA")
        )
        image = composite(Image.open(render_path), Image.open(source).convert("RGBA"), background)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)

    (args.output / f"content_{args.tag}.txt").write_text("".join(f"{p}\n" for p in contents))
    (args.output / f"style_{args.tag}.txt").write_text("".join(f"{p}\n" for p in styles))


if __name__ == "__main__":
    main()
