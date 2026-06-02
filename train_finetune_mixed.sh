#!/bin/sh
# Fine-tune StableVSR from REDS-trained weights on a REDS+YouHQ(+Vimeo) mix.
#
# Start from the LESS-drifted 20k checkpoint (not the 30k "+10k" one), point
# REDS_CONTROLNET at the folder holding config.json + safetensors for it
# (an accelerate checkpoint saves it under checkpoint-20000/controlnet/).
#
# Fine-tuning is ADAPTATION, not a fresh run: keep it short + low LR, checkpoint
# often, and pick the stopping point with scripts/select_checkpoint.py (the
# diffusion training loss is NOT a reliable stop signal).

MODEL_ID='claudiom4sir/StableVSR'
REDS_CONTROLNET='experiments/20260430_dualsft/checkpoint-20000/controlnet'
OUTPUT_DIR='experiments/mixed_finetune'
DATASET_CONFIG='dataset/config_mixed.yaml'
GPUS="5 6 7 8"

GPUS_STR=$(echo $GPUS | tr ' ' ',')
export CUDA_VISIBLE_DEVICES=$GPUS_STR
NUM_PROCESSES=$(echo $GPUS | wc -w)

accelerate launch --num_processes $NUM_PROCESSES --main_process_port 29501 train.py \
 --pretrained_model_name_or_path=$MODEL_ID \
 --pretrained_vae_model_name_or_path=$MODEL_ID \
 --controlnet_model_name_or_path=$REDS_CONTROLNET \
 --output_dir=$OUTPUT_DIR \
 --dataset_config_path=$DATASET_CONFIG \
 --learning_rate=1e-5 \
 --validation_steps=1000 \
 --checkpointing_steps=500 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=5000 \
 --enable_xformers_memory_efficient_attention \
 --validation_prompt ""
