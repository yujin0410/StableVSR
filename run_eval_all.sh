#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=2

EVAL=/home/yjcho/StableVSR/eval.py
REC_ROOT=/mnt/HDD_raid1/yjcho/Comparison/Ours_30k

UDM10_GT=/mnt/HDD_raid1/yjcho/data/UDM10/GT
VID4_GT=/mnt/HDD_raid1/yjcho/data/Vid4/GT
SPMCS_ROOT=/mnt/HDD_raid1/yjcho/spmc_test_set

echo "==================== [1/3] UDM10 ===================="
python "$EVAL" \
    --gt_path "$UDM10_GT/" \
    --out_path "$REC_ROOT/udm10/"

echo "==================== [2/3] Vid4 ====================="
python "$EVAL" \
    --gt_path "$VID4_GT/" \
    --out_path "$REC_ROOT/vid4/"

echo "==================== [3/3] SPMCS ===================="
# SPMCS GT layout: spmc_test_set/<seq>/truth/<frames>
# eval.py expects: gt_path/<seq>/<frames>
# -> build a tmp dir of symlinks so <seq> points to <seq>/truth
SPMCS_TMP_GT=$(mktemp -d -t spmcs_gt_XXXX)
echo "Building SPMCS GT symlink tree at: $SPMCS_TMP_GT"

REC_SPMCS="$REC_ROOT/spmcs"
for seq in $(ls "$REC_SPMCS"); do
    if [ -d "$SPMCS_ROOT/$seq/truth" ]; then
        ln -s "$SPMCS_ROOT/$seq/truth" "$SPMCS_TMP_GT/$seq"
    else
        echo "WARN: missing GT for sequence '$seq' (expected $SPMCS_ROOT/$seq/truth)"
    fi
done

python "$EVAL" \
    --gt_path "$SPMCS_TMP_GT/" \
    --out_path "$REC_SPMCS/"

rm -rf "$SPMCS_TMP_GT"

echo "==================== DONE ===================="
