---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-feature-extraction
tags:
- person-re-identification
- aerial-ground
- vision-transformer
- eccv2026
---

# 3D-LENS

Checkpoints of **3D-LENS: A 3D Lifting-based Elevated Novel-view Synthesis method for Single-View Aerial-Ground
Re-Identification** (ECCV 2026). Code, training and evaluation: <https://github.com/TurtleSmoke/3D-LENS>.

Each checkpoint is a ViT-B/16 + BNNeck trained on one camera type of AG-ReID or AG-ReID.v2, together with its 3D-LENS
synthetic views. Retrieval uses the cosine distance on the BNNeck feature, with a single forward pass.

## Checkpoints

| File | Setting | Protocol 1 mAP / R1 | Protocol 2 mAP / R1 |
|------|---------|---------------------|---------------------|
| `agreid_aerial.pth` | AG-ReID, aerial training | A→G 49.78 / 62.84 | G→A 49.44 / 57.59 |
| `agreid_ground.pth` | AG-ReID, ground training | A→G 53.26 / 65.17 | G→A 55.05 / 67.36 |
| `agreid_v2_aerial_cctv.pth` | AG-ReID.v2, aerial training | C→A 37.84 / 47.10 | A→C 42.77 / 55.69 |
| `agreid_v2_cctv.pth` | AG-ReID.v2, CCTV training | C→A 47.21 / 58.53 | A→C 41.36 / 50.76 |
| `agreid_v2_aerial_wearable.pth` | AG-ReID.v2, aerial training | W→A 28.61 / 35.77 | A→W 34.26 / 47.13 |
| `agreid_v2_wearable.pth` | AG-ReID.v2, wearable training | W→A 38.27 / 46.24 | A→W 38.93 / 47.89 |

The numbers are those obtained with these files and `scripts/eval.sh`. The checkpoints keep only the inference
parameters (backbone and BNNeck) and are stored in fp16, so they can differ from the paper by up to 0.1 point.

## Usage

```bash
git clone https://github.com/TurtleSmoke/3D-LENS && cd 3D-LENS && uv sync
hf download TurtleSmoke/3D-LENS --local-dir checkpoints
bash scripts/eval.sh agreid_aerial $DATA checkpoints/agreid_aerial.pth
```

`$DATA` holds the real datasets, see the [GitHub README](https://github.com/TurtleSmoke/3D-LENS#data).

## License

The weights are released for non-commercial research under CC BY-NC 4.0. They were trained on
[AG-ReID](https://github.com/huynguyen792/AG-ReID) and [AG-ReID.v2](https://github.com/huynguyen792/AG-ReID.v2)
and on synthetic views generated with [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) and
[StyleID](https://github.com/jiwoogit/StyleID); their respective terms also apply. The code is under the CeCILL-B license.

## Citation

```bibtex
@inproceedings{grolleau2026lens,
  title     = {3D-LENS: A 3D Lifting-based Elevated Novel-view Synthesis method for Single-View Aerial-Ground Re-Identification},
  author    = {Grolleau, William and Sabourin, Astrid and Lapouge, Guillaume and Achard, Catherine},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
