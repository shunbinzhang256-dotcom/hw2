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
  echo "[$(date '+%F %T')] START $name on GPU0" | tee -a "$LOG_DIR/train_queue.log"
  echo "Command: $*" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  local status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END $name status=$status on GPU0" | tee -a "$LOG_DIR/train_queue.log"
  return "$status"
}

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

echo "[$(date '+%F %T')] GPU0 QUEUE FINISHED" | tee -a "$LOG_DIR/train_queue.log"
