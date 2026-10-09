#!/bin/bash
# Evaluate a 3D-LENS checkpoint (only the real dataset is needed).
#   bash scripts/eval.sh <setting> <datasets_dir> <checkpoint.pth>
set -e
setting=$1
root_dir=$2
weights=$3
real_dir=$(python -c "from omegaconf import OmegaConf; print(OmegaConf.load('reid/configs/train/$setting.yml').DATA.DATASET_DIR[0])")

python -u -m reid.train \
    --name "eval_$setting" \
    --config "reid/configs/train/$setting.yml" \
    --output_dir runs/ \
    --root_dir "$root_dir" \
    --opts GLOBAL.RUN_MODE=eval "DATA.DATASET_DIR=[$real_dir]" TEST.WEIGHTS="$weights"
