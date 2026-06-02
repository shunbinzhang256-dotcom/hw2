"""Canonical dataset-level evaluation for the U-Net segmentation models.

Unlike batch-averaged mIoU (which depends on batch size), this accumulates a
single global confusion matrix over the entire test set, then computes per-class
IoU once. This is the standard mIoU reported for semantic segmentation and is
directly comparable across the ce / dice / ce_dice runs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import StanfordBackgroundDataset
from unet import UNet

CLASS_NAMES = ["sky", "tree", "road", "grass", "water",
               "building", "mountain", "fg-object"]


@torch.no_grad()
def confusion(model, loader, device, num_classes=8, ignore_index=255):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for image, mask in loader:
        logits = model(image.to(device))
        pred = logits.argmax(dim=1).cpu().numpy().reshape(-1)
        gt = mask.numpy().reshape(-1)
        valid = gt != ignore_index
        pred, gt = pred[valid], gt[valid]
        k = (gt * num_classes + pred)
        cm += np.bincount(k, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def metrics_from_cm(cm):
    inter = np.diag(cm).astype(np.float64)
    gt_sum = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
    union = gt_sum + pred_sum - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    miou = np.nanmean(iou)
    pix_acc = inter.sum() / cm.sum()
    return iou, miou, pix_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/HW2/data/stanford_background")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--output", default="/root/HW2/outputs/task3/canonical_metrics.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = StanfordBackgroundDataset(args.data_root, "test", args.image_size)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    print(f"Test images: {len(ds)}")

    runs = {
        "ce": "/root/HW2/outputs/task3/unet_ce_size256_lr0.001_bs8/best.pt",
        "dice": "/root/HW2/outputs/task3/unet_dice_size256_lr0.001_bs8/best.pt",
        "ce_dice": "/root/HW2/outputs/task3/unet_ce_dice_size256_lr0.001_bs8/best.pt",
    }

    out = {}
    for name, ckpt_path in runs.items():
        model = UNet(num_classes=8, base_channels=args.base_channels).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        cm = confusion(model, loader, device)
        iou, miou, pix_acc = metrics_from_cm(cm)
        out[name] = {
            "mIoU": round(float(miou), 4),
            "pixel_acc": round(float(pix_acc), 4),
            "per_class_iou": {CLASS_NAMES[i]: round(float(iou[i]), 4) for i in range(8)},
        }
        print(f"\n=== {name} ===")
        print(f"mIoU={miou:.4f}  pixel_acc={pix_acc:.4f}")
        for i in range(8):
            print(f"  {CLASS_NAMES[i]:>10}: IoU={iou[i]:.4f}")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
