#!/usr/bin/env bash
# Custom DGAF-VSR Vid4 inference. Drop this into the DGAF-VSR repo root and run.
#
# Output: <out_path>/<seq.parent.name>/<seq.name>/<frame>.png
#   -> /mnt/HDD_raid1/yjcho/Comparison/DGAF-VSR/Vid4/Bicubic4xLR/calendar/0000.png
set -e

OUT_ROOT="/mnt/HDD_raid1/yjcho/Comparison/DGAF-VSR/Vid4"
mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python examples/dgafvsr/test_dgafvsr.py \
  --in_path  assets/Vid4/Bicubic4xLR/ \
  --model_id ckpts/DGAF_VSR/ \
  --ckpt     ckpts/DGAF_VSR/DGAF_VSR_REDS \
  --out_path "$OUT_ROOT" \
  --num_inference_steps 50
