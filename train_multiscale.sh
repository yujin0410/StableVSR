#!/bin/sh

# Train from scratch with the multi-scale SFT design.
# This is NOT a fine-tune of checkpoint-20000 because the SFTAdapter now has
# new modules (scale_enhancers, scale_logits) that did not exist there.

MODEL_ID='claudiom4sir/StableVSR'
OUTPUT_DIR='experiments/20260429_multiscale'
GPUS="1 2"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29501 train.py \
 --mixed_precision=no \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_vae_model_name_or_path=$MODEL_ID \
 --controlnet_model_name_or_path=$MODEL_ID \
 --output_dir=$OUTPUT_DIR \
 --dataset_config_path="/home/yjcho/StableVSR/dataset/config_reds.yaml" \
 --learning_rate=5e-5 \
 --validation_steps=1000 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=20000 \
 --enable_xformers_memory_efficient_attention \
 --validation_image "/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000000.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000001.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000002.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000003.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000004.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000005.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000006.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000007.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000008.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000009.png" \
 --validation_prompt ""
