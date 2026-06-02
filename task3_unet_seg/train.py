import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import StanfordBackgroundDataset
from losses import build_loss
from metrics import mean_iou, pixel_accuracy
from unet import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/HW2/data/stanford_background")
    parser.add_argument("--output-dir", default="/root/HW2/outputs/task3")
    parser.add_argument("--loss", choices=["ce", "dice", "ce_dice"], default="ce")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, num_classes, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_miou = 0.0
    total_acc = 0.0
    total = 0

    for images, masks in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_miou += mean_iou(logits.detach(), masks, num_classes=num_classes) * bs
        total_acc += pixel_accuracy(logits.detach(), masks) * bs
        total += bs

    return total_loss / total, total_miou / total, total_acc / total


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 8

    run_name = f"unet_{args.loss}_size{args.image_size}_lr{args.lr}_bs{args.batch_size}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = StanfordBackgroundDataset(args.data_root, "train", args.image_size, args.seed)
    val_ds = StanfordBackgroundDataset(args.data_root, "val", args.image_size, args.seed)
    test_ds = StanfordBackgroundDataset(args.data_root, "test", args.image_size, args.seed)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    model = UNet(num_classes=num_classes, base_channels=args.base_channels).to(device)
    criterion = build_loss(args.loss, num_classes=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_miou = 0.0
    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_miou", "train_acc", "val_loss", "val_miou", "val_acc", "lr"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_miou, train_acc = run_epoch(model, train_loader, criterion, device, num_classes, optimizer, scaler)
            val_loss, val_miou, val_acc = run_epoch(model, val_loader, criterion, device, num_classes)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_miou": train_miou,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            writer.writerow(row)
            f.flush()
            print(json.dumps(row, ensure_ascii=False))

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save({"model": model.state_dict(), "args": vars(args), "val_miou": best_miou}, out_dir / "best.pt")

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_miou, test_acc = run_epoch(model, test_loader, criterion, device, num_classes)
    metrics = {"best_val_miou": best_miou, "test_loss": test_loss, "test_miou": test_miou, "test_acc": test_acc}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
