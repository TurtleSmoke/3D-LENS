"""Keep only the parameters used at inference (backbone + BNNeck) in a 3D-LENS training checkpoint.

python scripts/strip_checkpoint.py model_last.pth agreid_aerial.pth [--fp32]
"""

import argparse

import torch

INFERENCE_PREFIXES = ("base.", "heads.bottleneck.")

parser = argparse.ArgumentParser()
parser.add_argument("src")
parser.add_argument("dst")
parser.add_argument("--fp32", action="store_true", help="keep fp32 weights (default is fp16, half the size)")
args = parser.parse_args()

state_dict = torch.load(args.src, map_location="cpu", weights_only=True)
kept = {k: v for k, v in state_dict.items() if k.startswith(INFERENCE_PREFIXES)}
if not args.fp32:
    kept = {k: v.half() if v.is_floating_point() else v for k, v in kept.items()}

torch.save(kept, args.dst)
dropped = sorted({k.split(".")[0] for k in state_dict if k not in kept})
print(f"Kept {len(kept)}/{len(state_dict)} tensors, dropped {dropped}")
