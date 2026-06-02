import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class StanfordBackgroundDataset(Dataset):
    """Stanford Background semantic labels from *.regions.txt files."""

    def __init__(self, root, split="train", image_size=256, seed=42):
        self.root = Path(root)
        if (self.root / "iccv09Data").exists():
            self.root = self.root / "iccv09Data"
        self.image_dir = self.root / "images"
        self.label_dir = self.root / "labels"
        self.image_size = image_size

        if not self.image_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(f"Expected images/ and labels/ under {self.root}")

        image_paths = sorted(self.image_dir.glob("*.jpg"))
        pairs = [(p, self.label_dir / f"{p.stem}.regions.txt") for p in image_paths]
        pairs = [(img, lab) for img, lab in pairs if lab.exists()]
        if not pairs:
            raise RuntimeError(f"No image/regions pairs found under {self.root}")

        rng = random.Random(seed)
        rng.shuffle(pairs)
        n = len(pairs)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        if split == "train":
            self.pairs = pairs[:n_train]
        elif split == "val":
            self.pairs = pairs[n_train : n_train + n_val]
        elif split == "test":
            self.pairs = pairs[n_train + n_val :]
        else:
            raise ValueError("split must be train, val, or test")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, label_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        image_np = np.asarray(image).astype("float32") / 255.0
        image_t = torch.from_numpy(image_np).permute(2, 0, 1)
        image_t = F.interpolate(
            image_t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        image_t = (image_t - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

        mask = np.loadtxt(label_path, dtype=np.int64)
        mask = np.where(mask < 0, 255, mask)
        mask_t = torch.from_numpy(mask).unsqueeze(0).float()
        mask_t = F.interpolate(mask_t.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest").squeeze(0).squeeze(0)
        mask_t = mask_t.long()
        return image_t, mask_t
