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
Re-Identification** (ECCV 2026): [paper](https://link.springer.com/chapter/10.1007/978-3-032-36846-1_12),
[arXiv:2604.26520](https://arxiv.org/abs/2604.26520), [poster](https://eccv.ecva.net/virtual/2026/poster/5389).
Code, training and evaluation are at <https://github.com/TurtleSmoke/3D-LENS>.

## Checkpoints

Each checkpoint is trained on one camera type of AG-ReID (`agreid_*`) or AG-ReID.v2 (`agreid_v2_*`), given by its
name. The two AG-ReID.v2 aerial checkpoints are evaluated against CCTV and wearable images respectively.

| Checkpoint                      | Query → Gallery | mAP   | Rank-1 |
|---------------------------------|-----------------|-------|--------|
| `agreid_aerial.pth`             | A → G           | 49.78 | 62.84  |
|                                 | G → A           | 49.44 | 57.59  |
| `agreid_ground.pth`             | A → G           | 53.26 | 65.17  |
|                                 | G → A           | 55.05 | 67.36  |
| `agreid_v2_aerial_cctv.pth`     | C → A           | 37.84 | 47.10  |
|                                 | A → C           | 42.77 | 55.69  |
| `agreid_v2_cctv.pth`            | C → A           | 47.21 | 58.53  |
|                                 | A → C           | 41.36 | 50.76  |
| `agreid_v2_aerial_wearable.pth` | W → A           | 28.61 | 35.77  |
|                                 | A → W           | 34.26 | 47.13  |
| `agreid_v2_wearable.pth`        | W → A           | 38.27 | 46.24  |
|                                 | A → W           | 38.93 | 47.89  |

A = aerial, G = ground, C = CCTV, W = wearable. The checkpoints are stored in fp16, so they can differ slightly from
the paper.

## Usage

```bash
git clone https://github.com/TurtleSmoke/3D-LENS && cd 3D-LENS && uv sync && source .venv/bin/activate
hf download TurtleSmoke/3D-LENS --include "*.pth" --local-dir checkpoints
bash scripts/eval.sh agreid_aerial $DATA checkpoints/agreid_aerial.pth
```

## License

The weights are released for non-commercial research under CC BY-NC 4.0. They were trained on
[AG-ReID](https://github.com/huynguyen792/AG-ReID) and [AG-ReID.v2](https://github.com/huynguyen792/AG-ReID.v2)
and on synthetic views generated with [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) and
[StyleID](https://github.com/jiwoogit/StyleID), whose respective terms also apply. The code is under the CeCILL-B license.

## Citation

```bibtex
@inproceedings{grolleau2026lens,
    title = {3D-LENS: A 3D Lifting-based Elevated Novel-view Synthesis method for Single-View Aerial-Ground Re-Identification},
    author = {Grolleau, William and Sabourin, Astrid and Lapouge, Guillaume and Achard, Catherine},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2026}
}
```
