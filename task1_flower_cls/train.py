import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import Flowers102Dataset
from models import get_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/HW2/data/flowers102")
    parser.add_argument("--output-dir", default="/root/HW2/outputs/task1")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args):
    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = Flowers102Dataset(args.data_root, "train", train_tf)
    val_ds = Flowers102Dataset(args.data_root, "val", eval_tf)
    test_ds = Flowers102Dataset(args.data_root, "test", eval_tf)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, drop_last=False, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = f"{args.model}_{'pretrained' if args.pretrained else 'scratch'}_lr{args.lr}_bs{args.batch_size}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb:
        import wandb

        wandb.init(project="hw2-flower-cls", name=run_name, config=vars(args))
    else:
        wandb = None

    train_loader, val_loader, test_loader = build_loaders(args)
    model = get_model(args.model, num_classes=102, pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
            writer.writerow(row)
            f.flush()

            if wandb is not None:
                wandb.log(row)

            print(json.dumps(row, ensure_ascii=False))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({"model": model.state_dict(), "args": vars(args), "val_acc": best_acc}, out_dir / "best.pt")

    checkpoint = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    metrics = {"best_val_acc": best_acc, "test_loss": test_loss, "test_acc": test_acc}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
