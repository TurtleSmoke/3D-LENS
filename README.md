# 3D-LENS

[![Paper](https://img.shields.io/badge/ECCV%202026-Paper-4b44ce)](https://link.springer.com/chapter/10.1007/978-3-032-36846-1_12)
[![arXiv](https://img.shields.io/badge/arXiv-2604.26520-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.26520)
[![Poster](https://img.shields.io/badge/ECCV%202026-Poster-4b44ce)](https://eccv.ecva.net/virtual/2026/poster/5389)
[![Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-ffd21e?logo=huggingface&logoColor=black)](https://huggingface.co/TurtleSmoke/3D-LENS)
[![Lint](https://github.com/TurtleSmoke/3D-LENS/actions/workflows/lint.yml/badge.svg)](https://github.com/TurtleSmoke/3D-LENS/actions/workflows/lint.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: CeCILL-B](https://img.shields.io/badge/License-CeCILL--B-green)](LICENSE)

Official code of **3D-LENS: A 3D Lifting-based Elevated Novel-view Synthesis method for Single-View Aerial-Ground
Re-Identification** (ECCV 2026).

3D-LENS trains an aerial-ground re-identification model from a single viewpoint (aerial *or* ground images only).
Each training image is lifted to a textured 3D mesh, and the mesh is rendered from shifted elevations and azimuths,
so the network sees the missing viewpoint during training. The synthetic views are composited on real backgrounds
and style-aligned with the real images. They are then mixed with the real images through a balanced sampler, a
curriculum on the elevation shift and a domain token.

[3D-VITAL](https://github.com/TurtleSmoke/3D-VITAL), a more advanced version of 3D-LENS accepted at NeurIPS 2026, is now available.

```
generation/   synthetic view generation (segmentation, 3D lifting, calibration, rendering, background, StyleID)
reid/         training and evaluation
scripts/      utility scripts
```

## Installation

```bash
uv sync                          # training and evaluation (Python >= 3.10, torch 2.4), or: pip install -e .
uv sync --group generation       # + synthetic view generation (YOLO, pyrender, LaMa)
make -C reid                     # optional: Cython ranking, faster evaluation
uv run pre-commit install        # development: ruff check + format on every commit (same as the CI)
```

Commands below assume the environment is active (`source .venv/bin/activate`).

We use the pretrained weights on ImageNet-21K for the ViT backbone
([jx_vit_base_p16_224-80ecf9dd.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)),
which is expected in `checkpoints/` (or set `PRETRAINED=<path>`).

## Data

Each real dataset is paired with its synthetic counterpart, generated with [generation/](generation/README.md):

```
$DATA/
├── AG-ReID/                 # bounding_box_train/, query_all_c{0,3}/, bounding_box_test_all_c{0,3}/, train_all.txt
├── AG-ReID_3D-LENS/         # bounding_box_train/<image>_XX.jpg, train_all.txt
├── AG-ReID.v2/              # AG-ReID.v2/{train_all,query,gallery}/, exp*.txt, train_all.txt
├── AG-ReID.v2_3D-LENS/      # AG-ReID.v2/train_all/<id>/<image>_XX.jpg, train_all.txt
├── MOO/                     # output_screenshots_png/, output_screenshots_split/, train_all.txt
└── MOO_3D-LENS/             # output_screenshots_png/<cow>/<image>_XX.png, train_all.txt
```

`train_all.txt` lists the training images of a directory (`bash generation/build_lists.sh <dataset> <dir>`).
Synthetic views are named after their source image, with the elevation shift index `_00` (no shift) to `_09` (30°).

## Training

One configuration per single-view setting, in `reid/configs/train/`:

| Config                                             | Dataset    | Training images   | Evaluation    |
|----------------------------------------------------|------------|-------------------|---------------|
| `agreid_aerial` / `agreid_ground`                  | AG-ReID    | aerial / ground   | A→G, G→A      |
| `agreid_v2_aerial_cctv` / `agreid_v2_cctv`         | AG-ReID.v2 | aerial / CCTV     | A→C, C→A      |
| `agreid_v2_aerial_wearable` / `agreid_v2_wearable` | AG-ReID.v2 | aerial / wearable | A→W, W→A      |
| `moo_top` / `moo_side`                             | MOO        | top / side        | T→S, S→T, T+S |

```bash
bash scripts/train.sh agreid_aerial $DATA
```

Results are reported at the last epoch (`runs/ckpts/<config>/model_last.pth`).

## Evaluation

Evaluation only needs the real dataset. The checkpoints of the paper are on
[Hugging Face](https://huggingface.co/TurtleSmoke/3D-LENS), with their results in the model card:

```bash
hf download TurtleSmoke/3D-LENS --include "*.pth" --local-dir checkpoints
bash scripts/eval.sh agreid_aerial $DATA checkpoints/agreid_aerial.pth
```

They are stored in fp16 (inference parameters only) to halve their size, so evaluating them can differ slightly from
the paper.

## Citation

```bibtex
@inproceedings{grolleau2026lens,
    title = {3D-LENS: A 3D Lifting-based Elevated Novel-view Synthesis method for Single-View Aerial-Ground Re-Identification},
    author = {Grolleau, William and Sabourin, Astrid and Lapouge, Guillaume and Achard, Catherine},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2026}
}
```

## Acknowledgements

The backbone comes from [TransReID](https://github.com/damo-cv/TransReID) and the losses, optimizer and scheduler
from [fast-reid](https://github.com/JDAI-CV/fast-reid). Synthetic views are generated with
[Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2), [StyleID](https://github.com/jiwoogit/StyleID),
[LaMa](https://github.com/advimman/lama) and [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

Special thanks to the authors of [fast-reid](https://github.com/JDAI-CV/fast-reid), whose open-source code was largely
used and adapted for this project.

## License

Copyright (c) 2026 Commissariat à l'énergie atomique et aux énergies alternatives (CEA).
This code is released under the [CeCILL-B](LICENSE) license. Files adapted from TransReID (MIT), fast-reid
(Apache-2.0) and StyleID (MIT) keep their original license.
