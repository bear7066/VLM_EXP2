#!/bin/bash
# test_run.sh — 用 5 個 clip 做 smoke test（只跑 1 step，不存 checkpoint）
#
# 用法：  bash test_run.sh
#
# 目的：確認 dataset / forward / trainer 整條 pipeline 可以跑起來

set -e

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_PATH="$REPO_ROOT/dataset/test_5clips.json"
IMAGE_FOLDER="$REPO_ROOT/dataset"
OUTPUT_DIR="/tmp/gemma_ft_test"
DEEPSPEED_CONFIG="$REPO_ROOT/scripts/stage1.json"

echo "=== gemma_ft smoke test ==="
echo "DATA:   $DATA_PATH"
echo "FRAMES: $IMAGE_FOLDER"
echo ""

cd "$REPO_ROOT"

uv run deepspeed \
    --num_gpus 1 \
    --master_port 29501 \
    stage1/main.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    \
    --model_id "google/gemma-3-4b-it" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --output_dir "$OUTPUT_DIR" \
    \
    --bf16 True \
    --tf32 False \
    \
    --max_seq_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_steps 2 \
    \
    --learning_rate 1e-5 \
    --vision_lr 2e-5 \
    --projector_lr 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.0 \
    --lr_scheduler_type "constant" \
    \
    --eval_strategy "no" \
    --save_strategy "no" \
    \
    --gradient_checkpointing False \
    --logging_steps 1 \
    --dataloader_num_workers 0 \
    --report_to "none"

echo ""
echo "=== smoke test PASSED ==="
