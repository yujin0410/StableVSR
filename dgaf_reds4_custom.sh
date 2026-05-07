#!/usr/bin/env bash
# Custom DGAF-VSR REDS4 inference. Drop this into the DGAF-VSR repo root and run.
#   cp dgaf_reds4_custom.sh /mnt/HDD_raid1/yjcho/Comparison/DGAF-VSR/test_reds4_custom.sh
#   cd /mnt/HDD_raid1/yjcho/Comparison/DGAF-VSR && bash test_reds4_custom.sh
#
# Output (per test_dgafvsr.py): <out_path>/<seq.parent.name>/<seq.name>/<frame>.png
#   -> /mnt/HDD_raid1/yjcho/Comparison/DGAF-VSR/REDS/x4/000/0000000.png
set -e

OUT_ROOT="/mnt/HDD_raid1/yjcho/Comparison/DGAF-VSR/REDS"
mkdir -p "$OUT_ROOT"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python examples/dgafvsr/test_dgafvsr.py \
  --in_path  assets/REDS4/x4/ \
  --model_id ckpts/DGAF_VSR/ \
  --ckpt     ckpts/DGAF_VSR/DGAF_VSR_REDS \
  --out_path "$OUT_ROOT" \
  --num_inference_steps 50
