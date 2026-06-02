#!/usr/bin/env bash
set -euo pipefail

yolo detect train \
  model=yolov8s.pt \
  data=/root/HW2/task2_yolo_tracking/data.yaml \
  epochs=80 \
  imgsz=640 \
  batch=32 \
  project=/root/HW2/outputs/task2 \
  name=yolov8s_road_vehicle
