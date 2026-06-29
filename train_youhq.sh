#!/bin/sh
# Dual-SFT fine-tune on REDS + YouHQ, starting from the REDS-trained 20k
# dual-SFT checkpoint. Goal: make the frequency conditioning less REDS-specific
# (mitigate the distribution-dependence found in the band ablations).
#
# Bounded + monitored: low LR, 8k steps, frequent checkpoints, and OOD
# validation on a UDM10 clip every 500 steps so MUSIQ/LPIPS collapse is caught
# early (kill criterion). Loss weights match the 20k run (do NOT crank temporal
# loss -- that killed perceptual in tfl+Vimeo).

MODEL_ID='claudiom4sir/StableVSR'
SRC='experiments/20260430_dualsft/checkpoint-20000'   # fine-tune source
OUTPUT_DIR='experiments/20260430_youhq_finetune'
DATASET_CONFIG='dataset/config_reds_youhq.yaml'
GPUS="1"

GPUS_STR=$(echo $GPUS | tr ' ' ',')
export CUDA_VISIBLE_DEVICES=$GPUS_STR
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
NUM_PROCESSES=$(echo $GPUS | wc -w)

# --- OOD validation set: first 8 frames of UDM10 seq 000 (LR + GT) ---
UDM10='/mnt/HDD_raid1/yjcho/data/UDM10'
VS=$(ls "$UDM10/BIx4/000" | sort | head -8)
VAL_LR=$(echo "$VS" | sed "s#^#$UDM10/BIx4/000/#" | paste -sd ';')
VAL_GT=$(echo "$VS" | sed "s#^#$UDM10/GT/000/#"   | paste -sd ';')

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29502 train.py \
 --mixed_precision=no \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_vae_model_name_or_path=$MODEL_ID \
 --controlnet_model_name_or_path=$SRC \
 --init_sft_ckpt=$SRC/sft_adapter.bin \
 --output_dir=$OUTPUT_DIR \
 --dataset_config_path=$DATASET_CONFIG \
 --learning_rate=5e-5 \
 --validation_steps=500 \
 --checkpointing_steps=1000 \
 --train_batch_size=4 \
 --dataloader_num_workers=8 \
 --max_train_steps=8000 \
 --enable_xformers_memory_efficient_attention \
 --dual_sft \
 --freq_loss_interval=4 \
 --lambda_freq=1.0 \
 --lambda_freq_high=0.1 \
 --lambda_freq_low=1.0 \
 --validation_image "$VAL_LR" \
 --validation_gt_image "$VAL_GT" \
 --validation_prompt ""
