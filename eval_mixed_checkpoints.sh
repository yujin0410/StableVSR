#!/bin/sh
# Evaluate dual-SFT fine-tune checkpoints across datasets and tabulate metrics.
#
# For each (checkpoint x dataset): run test.py sharded over GPU 1 & 2 to produce
# SR frames, then eval.py to compute the 9 metrics, and append the summary line
# to a CSV. checkpoint-20000 is the pre-finetune REFERENCE.
#
# Goal: find the checkpoint where the clean out-of-domain sets (UDM10/SPMCS)
# recover (LPIPS/DISTS/tLPIPS/tOF down) WITHOUT REDS4 regressing much. Because
# resume restored the 1e-4 optimizer, the sweet spot is likely early
# (20500-22000) -- watch those first.
#
# EDIT the DATASETS paths below to your actual LR(in_path)/GT(gt_path) folders,
# then:  sh eval_mixed_checkpoints.sh

EXP_DIR="experiments/20260602_mixed_finetune"
COND_MODE="dtcwt"            # must match how the model was trained
STEPS=50
OUT_ROOT="eval_mixed_recon"
RESULTS="eval_mixed_results.csv"
GPU_A=1
GPU_B=2

# checkpoints to evaluate (20000 = pre-finetune reference)
CKPTS="20000 20500 21000 21500 22000 25000"

# one line per dataset:  name|LR_in_path|GT_path     (FILL THESE IN)
DATASETS="
reds4|/mnt/HDD_raid1/yjcho/data/REDS4/test/bicubic|/mnt/HDD_raid1/yjcho/data/REDS4/test/gt
udm10|/mnt/HDD_raid1/yjcho/data/UDM10/bicubic|/mnt/HDD_raid1/yjcho/data/UDM10/gt
spmcs|/mnt/HDD_raid1/yjcho/data/SPMCS/bicubic|/mnt/HDD_raid1/yjcho/data/SPMCS/gt
"

echo "checkpoint,dataset,metrics" > "$RESULTS"

for CKPT in $CKPTS; do
  CN="$EXP_DIR/checkpoint-$CKPT"
  SFT="$CN/sft_adapter.bin"
  if [ ! -d "$CN/controlnet" ] || [ ! -f "$SFT" ]; then
    echo "SKIP checkpoint-$CKPT (missing controlnet/ or sft_adapter.bin)"
    continue
  fi
  printf '%s\n' "$DATASETS" | while IFS='|' read -r NAME LR GT; do
    [ -z "$NAME" ] && continue
    OUT="$OUT_ROOT/ckpt$CKPT/$NAME"
    echo "=== checkpoint-$CKPT / $NAME ==="
    # sharded inference: shard 0 on GPU $GPU_A, shard 1 on GPU $GPU_B (parallel)
    CUDA_VISIBLE_DEVICES=$GPU_A python test.py --in_path "$LR" --out_path "$OUT" \
        --controlnet_ckpt "$CN" --sft_ckpt "$SFT" --dual_sft --cond_mode "$COND_MODE" \
        --num_inference_steps $STEPS --num_shards 2 --shard_id 0 &
    PID_A=$!
    CUDA_VISIBLE_DEVICES=$GPU_B python test.py --in_path "$LR" --out_path "$OUT" \
        --controlnet_ckpt "$CN" --sft_ckpt "$SFT" --dual_sft --cond_mode "$COND_MODE" \
        --num_inference_steps $STEPS --num_shards 2 --shard_id 1 &
    PID_B=$!
    wait $PID_A $PID_B
    # evaluate the reconstructed frames against GT
    LINE=$(CUDA_VISIBLE_DEVICES=$GPU_A python eval.py --out_path "$OUT" --gt_path "$GT" 2>/dev/null | grep -E "PSNR:")
    echo "checkpoint-$CKPT,$NAME,\"$LINE\"" | tee -a "$RESULTS"
  done
done

echo ""
echo "Done. Results -> $RESULTS"
echo "Pick the checkpoint where udm10/spmcs LPIPS,DISTS,tLPIPS,tOF improve vs"
echo "checkpoint-20000 while reds4 stays close (small regression OK)."
