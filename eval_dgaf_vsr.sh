#!/usr/bin/env bash
# Normalize DGAF-VSR output to {seq}/{frame}.png and run our eval.py.
#
# Usage:
#   bash eval_dgaf_vsr.sh <DGAF_RESULTS_DIR> <DATASET>
# Example:
#   bash eval_dgaf_vsr.sh /mnt/HDD_raid1/yjcho/DGAF-VSR/results/REDS4 reds4
#   bash eval_dgaf_vsr.sh /mnt/HDD_raid1/yjcho/DGAF-VSR/results/Vid4 vid4
set -e

SRC="${1:?usage: $0 <dgaf_results_dir> <dataset>}"
DATASET="${2:?usage: $0 <dgaf_results_dir> <dataset>}"

case "$DATASET" in
  reds4) GT_PATH="/mnt/HDD_raid1/yjcho/data/REDS/test/gt" ;;
  vid4)  GT_PATH="/mnt/HDD_raid1/yjcho/data/Vid4/GT"      ;;
  *) echo "unknown dataset: $DATASET"; exit 1 ;;
esac

NORM_DIR="./DGAF_VSR_results_${DATASET}"
echo "[*] Normalizing $SRC -> $NORM_DIR"
rm -rf "$NORM_DIR"
mkdir -p "$NORM_DIR"

# DGAF-VSR may write either:
#   results/REDS4/000/0000.png      (per-sequence dir, ours-like)
#   results/REDS4/000_0000.png      (flat with seq prefix)
# Try both.
shopt -s nullglob
for entry in "$SRC"/*; do
  base=$(basename "$entry")
  if [ -d "$entry" ]; then
    ln -sfn "$entry" "$NORM_DIR/$base"
  else
    seq="${base%%_*}"
    rest="${base#*_}"
    mkdir -p "$NORM_DIR/$seq"
    ln -sfn "$entry" "$NORM_DIR/$seq/$rest"
  fi
done

echo "[*] Running eval.py..."
python eval.py \
  --out_path "$NORM_DIR/" \
  --gt_path  "$GT_PATH/"
