#!/bin/bash
# Train 3D-LENS on one single-view setting.
#   bash scripts/train.sh <setting> <datasets_dir> [extra KEY=VALUE options]
# <setting> is a file of reid/configs/train/, e.g. agreid_aerial, agreid_v2_cctv, moo_top.
# ViT baseline, add TRAIN.SAMPLER=naive TRAIN.DOMAIN_LOSS_WEIGHT=0 "DATA.DATASET_DIR=[<real dataset>]"
set -e
setting=$1
root_dir=$2
shift 2

python -u -m reid.train \
    --name "$setting" \
    --config "reid/configs/train/$setting.yml" \
    --output_dir runs/ \
    --root_dir "$root_dir" \
    --opts MODEL.REID.PRETRAINED_PATH="${PRETRAINED:-checkpoints/jx_vit_base_p16_224-80ecf9dd.pth}" "$@"
