#!/usr/bin/env bash
# Normalize DGAF-VSR output to {seq}/{frame}.png and run our eval.py.
#
# DGAF-VSR writes to: <out_path>/<seq.parent.name>/<seq.name>/<frame>.png
# e.g. results/REDS4/dgaf_reds4_s50/x4/000/0000000.png
#
# Usage:
#   bash eval_dgaf_vsr.sh <DGAF_OUT_PATH> <DATASET>
# Example:
#   bash eval_dgaf_vsr.sh /mnt/HDD_raid1/yjcho/DGAF-VSR/results/REDS4/dgaf_reds4_s50 reds4
#   bash eval_dgaf_vsr.sh /mnt/HDD_raid1/yjcho/DGAF-VSR/results/Vid4/dgaf_vid4_s50  vid4
set -e

SRC="${1:?usage: $0 <dgaf_out_path> <dataset>}"
DATASET="${2:?usage: $0 <dgaf_out_path> <dataset>}"

case "$DATASET" in
  reds4) GT_PATH="/mnt/HDD_raid1/yjcho/data/REDS/test/gt" ;;
  vid4)  GT_PATH="/mnt/HDD_raid1/yjcho/data/Vid4/GT"      ;;
  *) echo "unknown dataset: $DATASET"; exit 1 ;;
esac

# Pick the inner level produced by DGAF-VSR (e.g. "x4" / "Bicubic4xLR").
# If SRC contains a single subdir, descend into it; otherwise use SRC as-is.
INNER=$(find "$SRC" -mindepth 1 -maxdepth 1 -type d | head -n1)
N_TOP=$(find "$SRC" -mindepth 1 -maxdepth 1 -type d | wc -l)
if [ "$N_TOP" = "1" ] && [ -d "$INNER" ]; then
  SEQ_ROOT="$INNER"
else
  SEQ_ROOT="$SRC"
fi
echo "[*] Sequence root: $SEQ_ROOT"

NORM_DIR="./DGAF_VSR_results_${DATASET}"
echo "[*] Normalizing -> $NORM_DIR"
rm -rf "$NORM_DIR"
mkdir -p "$NORM_DIR"
shopt -s nullglob
for seq_dir in "$SEQ_ROOT"/*/; do
  seq=$(basename "$seq_dir")
  ln -sfn "$(realpath "$seq_dir")" "$NORM_DIR/$seq"
done

echo "[*] Running eval.py..."
python eval.py \
  --out_path "$NORM_DIR/" \
  --gt_path  "$GT_PATH/"
