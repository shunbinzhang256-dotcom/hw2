import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import StanfordBackgroundDataset
from metrics import mean_iou, pixel_accuracy
from unet import UNet


PALETTE = np.array(
    [
        [128, 128, 128],
        [128, 192, 0],
        [128, 64, 128],
        [0, 128, 0],
        [0, 0, 192],
        [192, 128, 128],
        [128, 128, 0],
        [192, 0, 192],
    ],
    dtype=np.uint8,
)


def denorm(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255).astype(np.uint8)


def colorize(mask):
    mask = mask.copy()
    mask[mask == 255] = 0
    return PALETTE[mask]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/root/HW2/data/stanford_background")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="/root/HW2/outputs/task3/visuals")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = StanfordBackgroundDataset(args.data_root, "test", args.image_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    model = UNet(num_classes=8, base_channels=args.base_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    miou_sum = 0.0
    acc_sum = 0.0
    count = 0
    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            logits = model(image.to(device))
            miou_sum += mean_iou(logits.cpu(), mask, 8)
            acc_sum += pixel_accuracy(logits.cpu(), mask)
            count += 1
            if idx < args.num_samples:
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                gt = mask.squeeze(0).numpy().astype(np.uint8)
                canvas = np.concatenate([denorm(image.squeeze(0)), colorize(gt), colorize(pred)], axis=1)
                cv2.imwrite(str(out_dir / f"sample_{idx:03d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    print({"test_miou": miou_sum / count, "test_acc": acc_sum / count, "visuals": str(out_dir)})


if __name__ == "__main__":
    main()
