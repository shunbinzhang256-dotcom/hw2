#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/HW2
echo "== GPU =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
echo
echo "== Queue =="
tail -n 20 "$ROOT/outputs/logs/train_queue.log" 2>/dev/null || true
echo
echo "== Running =="
ps -eo pid,ppid,stat,cmd | grep -E 'run_formal_train_safe|run_gpu0_cls_yolo|run_gpu5_unet|task1_flower_cls/train.py|task3_unet_seg/train.py|yolo detect train' | grep -v grep || true
echo
echo "== Progress =="
python3 "$ROOT/training_progress.py"
