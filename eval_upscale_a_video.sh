#!/usr/bin/env bash
# Run our eval.py on Upscale-A-Video outputs.
# Output already in our standard {seq}/{frame}.png layout (consolidated by uav_reds4_custom.sh).
#
# Usage:
#   bash eval_upscale_a_video.sh reds4
#   bash eval_upscale_a_video.sh vid4
set -e

DATASET="${1:?usage: $0 reds4|vid4}"
case "$DATASET" in
  reds4)
    SRC="/mnt/HDD_raid1/yjcho/Comparison/Upscale-A-Video/REDS"
    GT="/mnt/HDD_raid1/yjcho/data/REDS/test/gt"
    ;;
  vid4)
    SRC="/mnt/HDD_raid1/yjcho/Comparison/Upscale-A-Video/Vid4"
    GT="/mnt/HDD_raid1/yjcho/data/Vid4/GT"
    ;;
  *) echo "unknown dataset: $DATASET"; exit 1 ;;
esac

python eval.py --out_path "$SRC/" --gt_path "$GT/"
