#!/bin/sh
# Fine-tune StableVSR on Vimeo-90K, starting from REDS-trained ControlNet weights.
#
# Set REDS_CONTROLNET to the folder that contains your REDS-trained controlnet,
# i.e. a diffusers ControlNetModel directory holding:
#   config.json + diffusion_pytorch_model.safetensors
# This is what `controlnet.save_pretrained(...)` writes (either the final
# output_dir, or the `controlnet` subfolder inside a `checkpoint-XXXX/`).

MODEL_ID='claudiom4sir/StableVSR'
REDS_CONTROLNET='experiments/reds_exp/controlnet'   # <-- your REDS weights
OUTPUT_DIR='experiments/vimeo_finetune'
DATASET_CONFIG='dataset/config_vimeo.yaml'
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
 --learning_rate=5e-5 \
 --validation_steps=1000 \
 --checkpointing_steps=2000 \
 --train_batch_size=8 \
 --dataloader_num_workers=8 \
 --max_train_steps=20000 \
 --enable_xformers_memory_efficient_attention \
 --validation_prompt ""
