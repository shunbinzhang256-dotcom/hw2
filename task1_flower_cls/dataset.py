from pathlib import Path

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset


class Flowers102Dataset(Dataset):
    """Oxford 102 Flowers using the official setid.mat split."""

    SPLIT_KEYS = {
        "train": "trnid",
        "val": "valid",
        "test": "tstid",
    }

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        image_dir = self.root / "jpg"
        labels_path = self.root / "imagelabels.mat"
        split_path = self.root / "setid.mat"
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels file: {labels_path}")
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")
        if split not in self.SPLIT_KEYS:
            raise ValueError(f"split must be one of {sorted(self.SPLIT_KEYS)}")

        labels = loadmat(labels_path)["labels"].reshape(-1).astype("int64") - 1
        ids = loadmat(split_path)[self.SPLIT_KEYS[split]].reshape(-1).astype("int64")

        self.samples = []
        for image_id in ids:
            image_path = image_dir / f"image_{image_id:05d}.jpg"
            self.samples.append((image_path, int(labels[image_id - 1])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
