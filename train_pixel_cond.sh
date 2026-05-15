#!/bin/sh

# Pixel-conditioning ablation (Q1: why frequency domain?).
#
# Identical to train_dual.sh except the conditioning input path is replaced:
#  - DT-CWT highpass tensors  ->  4-level avg-pool LR pyramid + learnable
#    6-direction 3x3 conv per level (PixelPyramidConditioner), reshaped to
#    [B,3,6,H_j,W_j,2] with imag=0 to match DT-CWT shape exactly.
#  - The frequency-separated loss path STILL uses DT-CWT, so this isolates
#    the conditioning effect of frequency decomposition.
#
# Same encoder (PerLevelProcessor x4), injection points (down_blocks[1],
# up_blocks[1]), optimizer settings, and 20k iter budget as the DT-CWT run.

MODEL_ID='claudiom4sir/StableVSR'
OUTPUT_DIR='experiments/20260515_pixelcond'
GPUS="1 2"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29502 train.py \
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
 --cond_mode=pixel \
 --debug_dual_sft=100 \
 --freq_loss_interval=4 \
 --lambda_freq=1.0 \
 --lambda_freq_high=0.1 \
 --lambda_freq_low=1.0 \
 --validation_image "/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000000.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000001.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000002.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000003.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000004.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000005.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000006.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000007.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000008.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000009.png" \
 --validation_prompt ""
