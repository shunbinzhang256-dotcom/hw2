# HW2 深度学习与空间智能

本仓库包含三个任务：

1. `task1_flower_cls`：102 Flowers 分类，包含 ResNet baseline、预训练消融、SE 注意力和 timm Transformer 对比。
2. `task2_yolo_tracking`：Road Vehicle Images Dataset 上的 YOLOv8 检测训练、视频多目标跟踪和越线计数。
3. `task3_unet_seg`：从零实现 U-Net，在 Stanford Background Dataset 上比较 CE、Dice、CE+Dice 三种损失。

## 环境

```bash
pip install -r requirements.txt
```

推荐显式指定当前空闲 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 数据目录

```text
data/
  flowers102/
  road_vehicle/
  stanford_background/
  videos/
```

当前已验证的数据状态：

- `data/flowers102/jpg`：8189 张图片，官方 `imagelabels.mat` 和 `setid.mat` 可读取。
- `data/road_vehicle/trafic_data`：2704 张训练图，300 张验证图，21 类 YOLO 标签。
- `data/stanford_background/iccv09Data`：715 张图，使用 `labels/*.regions.txt` 作为 8 类语义分割标签。
- `data/videos/traffic_highway.mp4`：768x432，30 秒（377 帧，12.5 FPS），真实道路俯拍交通片段，用于 tracking 和越线计数。来源：Intel IoT Devkit `sample-videos`（无版权限制，仅作测试）。

Kaggle 凭据需要放在：

```text
/root/.kaggle/kaggle.json
```

格式：

```json
{"username": "你的用户名", "key": "你的API key"}
```

## 任务 1：Flowers 分类

冒烟测试：

```bash
python task1_flower_cls/train.py --data-root data/flowers102 --model resnet18 --pretrained --epochs 5
```

正式训练示例：

```bash
python task1_flower_cls/train.py --data-root data/flowers102 --model resnet18 --pretrained --epochs 40
python task1_flower_cls/train.py --data-root data/flowers102 --model resnet18 --epochs 40
python task1_flower_cls/train.py --data-root data/flowers102 --model resnet18_se --pretrained --epochs 40
python task1_flower_cls/train.py --data-root data/flowers102 --model vit_tiny_patch16_224 --pretrained --epochs 40
```

## 任务 2：YOLO 检测和跟踪

生成或修正 YOLO `data.yaml`：

```bash
python task2_yolo_tracking/prepare_data.py --root data/road_vehicle --out task2_yolo_tracking/data.yaml
```

当前生成的 `data.yaml`：

```yaml
path: /root/HW2/data/road_vehicle/trafic_data
train: train/images
val: valid/images
names:
  0: ambulance
  1: army vehicle
  2: auto rickshaw
  3: bicycle
  4: bus
  5: car
  6: garbagevan
  7: human hauler
  8: minibus
  9: minivan
  10: motorbike
  11: pickup
  12: policecar
  13: rickshaw
  14: scooter
  15: suv
  16: taxi
  17: three wheelers -CNG-
  18: truck
  19: van
  20: wheelbarrow
```

训练：

```bash
yolo detect train model=yolov8s.pt data=task2_yolo_tracking/data.yaml epochs=80 imgsz=640 batch=32
```

跟踪与越线计数：

```bash
python task2_yolo_tracking/line_counter.py \
  --model outputs/task2/yolov8s_road_vehicle/weights/best.pt \
  --source data/videos/traffic_highway.mp4 \
  --output outputs/task2/tracked_counted.mp4
```

跟踪 + 越线计数 + ID 跳变分析（输出 track_log.json / id_switch_report.json / 帧）：

```bash
python task2_yolo_tracking/track_analyze.py \
  --model outputs/task2/yolov8s_road_vehicle/weights/best.pt \
  --source data/videos/traffic_highway.mp4 \
  --outdir outputs/task2 --save-frames
```

已通过的 smoke 测试：

```bash
CUDA_VISIBLE_DEVICES=0 yolo detect train \
  model=yolov8n.pt \
  data=/root/HW2/task2_yolo_tracking/data.yaml \
  epochs=1 imgsz=320 batch=8 workers=0 \
  project=/root/HW2/outputs/task2_smoke \
  name=yolov8n_smoke

CUDA_VISIBLE_DEVICES=0 python task2_yolo_tracking/line_counter.py \
  --model /root/HW2/outputs/task2_smoke/yolov8n_smoke/weights/best.pt \
  --source /root/HW2/data/videos/test_traffic.mp4 \
  --output /root/HW2/outputs/task2_smoke/tracked_counted.mp4
```

## 任务 3：U-Net 分割

冒烟测试：

```bash
python task3_unet_seg/train.py --data-root data/stanford_background --loss ce --epochs 5
```

正式训练：

```bash
python task3_unet_seg/train.py --data-root data/stanford_background --loss ce --epochs 150
python task3_unet_seg/train.py --data-root data/stanford_background --loss dice --epochs 150
python task3_unet_seg/train.py --data-root data/stanford_background --loss ce_dice --epochs 150
```

评测与可视化（在 `task3_unet_seg/` 目录下运行）：

```bash
# 数据集级 mIoU（统一混淆矩阵，三组损失可比）
python eval_canonical.py
# image|GT|pred 可视化
python eval.py --checkpoint ../outputs/task3/unet_dice_size256_lr0.001_bs8/best.pt \
  --output-dir ../outputs/task3/visuals_dice
```

## 报告

实验报告（含结果表、训练曲线、tracking/分割可视化）：`reports/HW2_实验报告.pdf`。
重新生成训练曲线与 PDF：

```bash
python make_curves.py                 # 生成 outputs/figures/*.png
python reports/build_pdf.py           # reports/report.md -> reports/HW2_实验报告.pdf
```
