"""Generate training-curve figures for all three tasks from history/results CSVs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/HW2/outputs")
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            try:
                cols[k.strip()].append(float(v))
            except (ValueError, TypeError):
                cols[k.strip()].append(v)
    return cols


# ---------------- Task 1: classification ----------------
T1 = {
    "ResNet18 pretrained": ROOT / "task1/resnet18_pretrained_lr0.0003_bs32/history.csv",
    "ResNet18 scratch": ROOT / "task1/resnet18_scratch_lr0.001_bs32/history.csv",
    "ResNet18+SE pretrained": ROOT / "task1/resnet18_se_pretrained_lr0.0003_bs32/history.csv",
    "ViT-tiny pretrained": ROOT / "task1/vit_tiny_patch16_224_pretrained_lr0.0003_bs32/history.csv",
}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p in T1.items():
    c = read_csv(p)
    ax[0].plot(c["epoch"], c["val_acc"], label=name)
    ax[1].plot(c["epoch"], c["val_loss"], label=name)
ax[0].set(title="Task1: Val Accuracy vs Epoch", xlabel="epoch", ylabel="val acc")
ax[1].set(title="Task1: Val Loss vs Epoch", xlabel="epoch", ylabel="val loss")
for a in ax:
    a.grid(alpha=0.3); a.legend()
fig.tight_layout(); fig.savefig(FIG / "task1_curves.png", dpi=130); plt.close(fig)

# pretrained vs scratch train+val acc (ablation focus)
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, key in [("pretrained", "ResNet18 pretrained"), ("scratch", "ResNet18 scratch")]:
    c = read_csv(T1[key])
    ax.plot(c["epoch"], c["train_acc"], "--", label=f"{name} train")
    ax.plot(c["epoch"], c["val_acc"], "-", label=f"{name} val")
ax.set(title="Task1 ablation: pretrained vs scratch (accuracy)", xlabel="epoch", ylabel="accuracy")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(FIG / "task1_ablation.png", dpi=130); plt.close(fig)

# ---------------- Task 2: YOLOv8s detection ----------------
c = read_csv(ROOT / "task2/yolov8s_road_vehicle/results.csv")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(c["epoch"], c["metrics/mAP50(B)"], label="mAP@0.5")
ax[0].plot(c["epoch"], c["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
ax[0].plot(c["epoch"], c["metrics/precision(B)"], label="precision")
ax[0].plot(c["epoch"], c["metrics/recall(B)"], label="recall")
ax[0].set(title="Task2: YOLOv8s detection metrics", xlabel="epoch", ylabel="score")
ax[1].plot(c["epoch"], c["train/box_loss"], label="train box")
ax[1].plot(c["epoch"], c["train/cls_loss"], label="train cls")
ax[1].plot(c["epoch"], c["val/box_loss"], label="val box")
ax[1].plot(c["epoch"], c["val/cls_loss"], label="val cls")
ax[1].set(title="Task2: YOLOv8s losses", xlabel="epoch", ylabel="loss")
for a in ax:
    a.grid(alpha=0.3); a.legend()
fig.tight_layout(); fig.savefig(FIG / "task2_curves.png", dpi=130); plt.close(fig)

# ---------------- Task 3: U-Net segmentation losses ----------------
T3 = {
    "CE": ROOT / "task3/unet_ce_size256_lr0.001_bs8/history.csv",
    "Dice": ROOT / "task3/unet_dice_size256_lr0.001_bs8/history.csv",
    "CE+Dice": ROOT / "task3/unet_ce_dice_size256_lr0.001_bs8/history.csv",
}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for name, p in T3.items():
    c = read_csv(p)
    ax[0].plot(c["epoch"], c["val_miou"], label=name)
    ax[1].plot(c["epoch"], c["val_loss"], label=name)
ax[0].set(title="Task3: Val mIoU vs Epoch", xlabel="epoch", ylabel="val mIoU")
ax[1].set(title="Task3: Val Loss vs Epoch (loss scales differ)", xlabel="epoch", ylabel="val loss")
for a in ax:
    a.grid(alpha=0.3); a.legend()
fig.tight_layout(); fig.savefig(FIG / "task3_curves.png", dpi=130); plt.close(fig)

print("Saved figures to", FIG)
for f in sorted(FIG.glob("*.png")):
    print(" ", f.name)
