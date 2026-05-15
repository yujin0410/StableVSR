#!/bin/sh

# Q2 ablation -- FFT conditioning variant.
#
# Identical to train_pixel_cond.sh except the conditioning input path is
# replaced again, this time with a fixed multi-level directional FFT
# (FFTPyramidConditioner). 6 angular wedges in [0, pi) per level, inverse-
# FFT'd to yield complex subbands matching the DT-CWT shape exactly. The
# frequency-separated loss path still uses DT-CWT, so this isolates the
# transform itself (DT-CWT vs FFT) while holding the encoder, injection
# points, optimizer settings, and training data identical.
#
# Single GPU (3) at the user's request -- ~3.5 h for 5k iter at batch 8.
# Note: 5k iter < the 20k used for DT-CWT/pixel; report this disclaimer in
# the thesis table or re-run at 20k for a strict apples-to-apples result.

MODEL_ID='claudiom4sir/StableVSR'
OUTPUT_DIR='experiments/20260516_fftcond'
GPUS="3"

GPUS_STR=$(echo $GPUS | tr ' ' ',')

export CUDA_VISIBLE_DEVICES=$GPUS_STR
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29503 train.py \
 --mixed_precision=no \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_vae_model_name_or_path=$MODEL_ID \
 --controlnet_model_name_or_path=$MODEL_ID \
 --output_dir=$OUTPUT_DIR \
 --dataset_config_path="/home/yjcho/StableVSR/dataset/config_reds.yaml" \
 --learning_rate=1e-4 \
 --validation_steps=500 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=5000 \
 --enable_xformers_memory_efficient_attention \
 --dual_sft \
 --cond_mode=fft \
 --debug_dual_sft=100 \
 --freq_loss_interval=4 \
 --lambda_freq=1.0 \
 --lambda_freq_high=0.1 \
 --lambda_freq_low=1.0 \
 --validation_image "/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000000.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000001.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000002.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000003.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000004.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000005.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000006.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000007.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000008.png;/mnt/HDD_raid1/yjcho/data/REDS/train/bicubic/X4/020/00000009.png" \
 --validation_prompt ""
