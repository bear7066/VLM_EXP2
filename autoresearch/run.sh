#!/bin/bash
# run.sh — autoresearch Gemma 3 4B IT Stage 1 Fine-tune
#
# Usage:
#   DATA_PATH=/path/to/data.json \
#   IMAGE_FOLDER=/path/to/images \
#   OUTPUT_DIR=./output/stage1 \
#   bash run.sh
#
# Or edit the defaults below and run: bash run.sh

set -e

OUTPUT_DIR="${OUTPUT_DIR:-./output/gemma3_stage1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/stage1.json}"

NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29500}"

uv run deepspeed \
    --num_gpus "$NUM_GPUS" \
    --master_port "$MASTER_PORT" \
    train.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    \
    --output_dir "$OUTPUT_DIR" \
    \
    --bf16 True \
    --tf32 False \
    \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    \
    --learning_rate 1e-5 \
    --vision_lr 2e-5 \
    --projector_lr 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --report_to "none"
