#!/usr/bin/env bash
set -u

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

ROOT=/root/HW2
LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG_DIR/train_queue.log"
  echo "Command: $*" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END $name status=$status" | tee -a "$LOG_DIR/train_queue.log"
  return "$status"
}

# Conservative settings for a partially occupied RTX 4090.
# Batch sizes are intentionally below the maximum to avoid OOM.

run_step task1_resnet18_pretrained \
  python3 "$ROOT/task1_flower_cls/train.py" \
    --data-root "$ROOT/data/flowers102" \
    --output-dir "$ROOT/outputs/task1" \
    --model resnet18 \
    --pretrained \
    --epochs 40 \
    --batch-size 32 \
    --lr 3e-4 \
    --num-workers 4

run_step task1_resnet18_scratch \
  python3 "$ROOT/task1_flower_cls/train.py" \
    --data-root "$ROOT/data/flowers102" \
    --output-dir "$ROOT/outputs/task1" \
    --model resnet18 \
    --epochs 40 \
    --batch-size 32 \
    --lr 1e-3 \
    --num-workers 4

run_step task1_resnet18_se_pretrained \
  python3 "$ROOT/task1_flower_cls/train.py" \
    --data-root "$ROOT/data/flowers102" \
    --output-dir "$ROOT/outputs/task1" \
    --model resnet18_se \
    --pretrained \
    --epochs 40 \
    --batch-size 32 \
    --lr 3e-4 \
    --num-workers 4

run_step task2_yolov8s \
  yolo detect train \
    model=yolov8s.pt \
    data="$ROOT/task2_yolo_tracking/data.yaml" \
    epochs=80 \
    imgsz=640 \
    batch=16 \
    workers=4 \
    project="$ROOT/outputs/task2" \
    name=yolov8s_road_vehicle

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

echo "[$(date '+%F %T')] TRAIN QUEUE FINISHED" | tee -a "$LOG_DIR/train_queue.log"
