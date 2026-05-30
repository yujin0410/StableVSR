#!/bin/bash
# Middle-frame eval on Vimeo-90K-T for every model we've inferred.
# Each invocation appends one row to the shared Excel file.
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

EVAL=/home/yjcho/StableVSR/middle_frame_eval.py
GT=/mnt/HDD_raid1/yjcho/data/vimeo_sr_test/vimeo_super_resolution_test/target
XLSX=/home/yjcho/StableVSR/vimeo_results.xlsx

run() {
    local name="$1"
    local rec="$2"
    if [ ! -d "$rec" ]; then
        echo "SKIP $name: $rec not found"
        return
    fi
    echo "==================== $name ===================="
    python "$EVAL" \
        --rec_path "$rec" \
        --gt_path "$GT" \
        --model_name "$name" \
        --excel_path "$XLSX"
}

run "Ours"        /mnt/HDD_raid1/yjcho/20260430/vimeo/
run "StableVSR"   /mnt/HDD_raid1/yjcho/stablevsr_vimeo/
run "DGAF-VSR"    /mnt/HDD_raid1/yjcho/Comparision/DGAF-VSR/Vimeo/
run "BasicVSR++"  /mnt/HDD_raid1/yjcho/BasicVSR_PlusPlus/Vimeo/

echo "==================== DONE ===================="
echo "Results appended to: $XLSX"
