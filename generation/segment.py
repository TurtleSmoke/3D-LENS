"""Step 1. Foreground segmentation of the source images with YOLO (instance segmentation).

Writes, for every input image, an RGBA PNG at the same relative path (alpha = foreground) under --output.
The foreground is the largest detected person plus every other detected object (bags, bicycles, ...).

python generation/segment.py --root AG-ReID --output AG-ReID_segmented --model yolo11l-seg.pt AG-ReID/bounding_box_train/*.jpg
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

PERSON = 0  # COCO class id


def foreground_mask(result, image_hw):
    classes = result.boxes.cls.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    person = np.zeros_like(masks[0])
    others = np.zeros_like(masks[0])
    for cls, m in zip(classes, masks):
        if cls == PERSON:
            person = person if person.sum() > m.sum() else m
        else:
            others = np.maximum(others, m)
    mask = np.maximum(person, others)

    # Undo the letterbox padding of YOLO, then resize to the image size
    model_h, model_w = mask.shape
    orig_h, orig_w = image_hw
    scale = min(model_h / orig_h, model_w / orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    pad_h, pad_w = (model_h - new_h) // 2, (model_w - new_w) // 2
    mask = mask[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
    return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="images are saved relative to this directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo11l-seg.pt", help="YOLO segmentation weights")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = YOLO(args.model)

    for start in range(0, len(args.images), args.batch):
        paths = args.images[start : start + args.batch]
        images = [np.array(Image.open(p).convert("RGB")) for p in paths]
        for path, img, result in zip(paths, images, model([Image.fromarray(i) for i in images])):
            classes = result.boxes.cls.cpu().numpy()
            if PERSON not in classes:
                print(f"No person detected in {path}, skipped")
                continue

            mask = foreground_mask(result, img.shape[:2])
            rgba = np.concatenate([img * mask[..., None], (mask * 255)[..., None]], axis=2).astype(np.uint8)

            out = args.output / path.relative_to(args.root).with_suffix(".png")
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba).save(out)


if __name__ == "__main__":
    main()
