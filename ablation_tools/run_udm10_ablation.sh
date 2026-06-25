#!/usr/bin/env bash
# Band-disabling ablation on UDM10 (reuses the REDS-20k dual-SFT checkpoint).
# Reproduces the slide-24 inference-time band toggle on an OOD dataset to test
# whether the HIGH-band anchoring that gives temporal consistency on REDS
# still works out-of-distribution.
#
# Requires the flow-cache patch to avoid recomputing identical flows 4x:
#   add `import ablation_tools.flow_cache` near the top of test.py.
#
# Edit the 4 paths below, then: bash ablation_tools/run_udm10_ablation.sh
set -e

# All generated frames, flow cache, and eval logs go under HDD (home has no space).
BASE=/mnt/HDD_raid1/yjcho/20260430
CKPT=experiments/20260430_dualsft/checkpoint-20000
UDM10_LR=/mnt/HDD_raid1/yjcho/data/UDM10/test/bicubic       # <-- UDM10 LR frames
UDM10_GT=/mnt/HDD_raid1/yjcho/data/UDM10/test/gt            # <-- UDM10 GT frames
OUT=$BASE/ablation_udm10

mkdir -p "$OUT"
export FLOW_CACHE_DIR="$OUT/flow_cache"                     # shared across all 4 variants

run () {  # $1 = variant name, $2 $3 = ablation flags
  echo "=== variant: $1  (flags: $2 $3) ==="
  for SID in 0 1; do
    CUDA_VISIBLE_DEVICES=$SID python test.py \
      --in_path  "$UDM10_LR" \
      --out_path "$OUT/$1/" \
      --controlnet_ckpt "$CKPT" \
      --sft_ckpt "$CKPT/sft_adapter.bin" \
      --dual_sft \
      --num_shards 2 --shard_id $SID \
      --num_inference_steps 50  $2 $3  &
  done
  wait
}

# Run `full` FIRST so it populates the flow cache for the other three.
run full
run no_high  --no_high_sft
run no_low   --no_low_sft
run no_both  --no_high_sft --no_low_sft

echo "=== evaluating (per-sequence breakdown) ==="
for v in full no_high no_low no_both; do
  echo "----- $v -----"
  python ablation_tools/eval_per_seq.py --out_path "$OUT/$v" --gt_path "$UDM10_GT" \
    | tee "$OUT/eval_udm10_${v}.txt"
done

echo "Done. Compare tLPIPS across full vs no_high to judge OOD anchoring."
