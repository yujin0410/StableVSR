#!/bin/sh

# Q2 ablation -- DWT conditioning variant.
#
# Real-valued DWT (db4) at 4 levels. 3 highpass subbands per level
# (LH, HL, HH) zero-padded to 6 along the subband axis and imag=0,
# so the tensor shape matches DT-CWT's [B,3,6,H_j,W_j,2] exactly.
# Encoder, injection points, optimizer, data are identical -- only
# the conditioning transform changes.
#
# Single GPU (3) at the user's request -- ~1.7 h for 5k iter at batch 8.

MODEL_ID='claudiom4sir/StableVSR'
OUTPUT_DIR='experiments/20260516_dwtcond'
GPUS="1 3"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29504 train.py \
 --mixed_precision=no \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_vae_model_name_or_path=$MODEL_ID \
 --controlnet_model_name_or_path=$MODEL_ID \
 --output_dir=$OUTPUT_DIR \
 --dataset_config_path="/home/yjcho/StableVSR/dataset/config_reds.yaml" \
 --learning_rate=1e-4 \
 --validation_steps=1000 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=20000 \
 --enable_xformers_memory_efficient_attention \
 --dual_sft \
 --cond_mode=dwt \
 --resume_from_checkpoint=latest \
 --debug_dual_sft=100 \
 --freq_loss_interval=4 \
 --lambda_freq=1.0 \
 --lambda_freq_high=0.1 \
 --lambda_freq_low=1.0 \
 --validation_image "/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000000.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000001.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000002.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000003.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000004.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000005.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000006.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000007.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000008.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000009.png" \
 --validation_prompt ""
