#!/bin/sh

MODEL_ID='claudiom4sir/StableVSR'
RESUME_DIR='experiments/20260424/checkpoint-20000'
OUTPUT_DIR='experiments/20260428_mixed_finetune'
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
 --resume_from_checkpoint=$RESUME_DIR \
 --dataset_config_path="/home/yjcho/StableVSR/dataset/config_mixed.yaml" \
 --learning_rate=1e-5 \
 --validation_steps=1000 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=25000 \
 --enable_xformers_memory_efficient_attention \
 --validation_image "/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000000.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000001.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000002.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000003.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000004.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000005.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000006.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000007.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000008.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000009.png" \
 --validation_prompt ""
