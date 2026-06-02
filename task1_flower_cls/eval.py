import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import Flowers102Dataset
from models import get_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/HW2/data/flowers102")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = Flowers102Dataset(args.data_root, "test", tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = get_model(args.model, num_classes=102, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    metrics = {"test_loss": total_loss / total, "test_acc": correct / total}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    Path(args.checkpoint).with_suffix(".eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
