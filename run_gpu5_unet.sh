#!/usr/bin/env bash
set -u

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

ROOT=/root/HW2
LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] START $name on GPU5" | tee -a "$LOG_DIR/train_queue.log"
  echo "Command: $*" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END $name status=$status on GPU5" | tee -a "$LOG_DIR/train_queue.log"
  return "$status"
}

run_step task3_unet_ce \
  python3 "$ROOT/task3_unet_seg/train.py" \
    --data-root "$ROOT/data/stanford_background" \
    --output-dir "$ROOT/outputs/task3" \
    --loss ce \
    --epochs 150 \
    --batch-size 8 \
    --image-size 256 \
    --base-channels 32 \
    --lr 1e-3 \
    --num-workers 4

run_step task3_unet_dice \
  python3 "$ROOT/task3_unet_seg/train.py" \
    --data-root "$ROOT/data/stanford_background" \
    --output-dir "$ROOT/outputs/task3" \
    --loss dice \
    --epochs 150 \
    --batch-size 8 \
    --image-size 256 \
    --base-channels 32 \
    --lr 1e-3 \
    --num-workers 4

run_step task3_unet_ce_dice \
  python3 "$ROOT/task3_unet_seg/train.py" \
    --data-root "$ROOT/data/stanford_background" \
    --output-dir "$ROOT/outputs/task3" \
    --loss ce_dice \
    --epochs 150 \
    --batch-size 8 \
    --image-size 256 \
    --base-channels 32 \
    --lr 1e-3 \
    --num-workers 4

echo "[$(date '+%F %T')] GPU5 QUEUE FINISHED" | tee -a "$LOG_DIR/train_queue.log"
