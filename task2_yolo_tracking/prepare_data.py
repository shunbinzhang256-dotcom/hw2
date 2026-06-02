import argparse
from pathlib import Path

import yaml


IMAGE_DIR_NAMES = ("images", "Images", "JPEGImages")
LABEL_DIR_NAMES = ("labels", "Labels")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_NAMES = ("train", "Train", "training", "Training")
VAL_NAMES = ("val", "Val", "valid", "Valid", "validation", "Validation", "test", "Test")


def find_dir(root, names):
    for name in names:
        matches = [p for p in root.rglob(name) if p.is_dir()]
        if matches:
            return matches[0]
    return None


def load_existing_names(root):
    for yaml_path in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        names = data.get("names")
        if isinstance(names, list):
            return {i: name for i, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
    return None


def infer_split_paths(root):
    label_root = find_dir(root, LABEL_DIR_NAMES)
    candidates = []

    # Common YOLO layout: root/images/train and root/images/val.
    for image_root in [p for name in IMAGE_DIR_NAMES for p in root.rglob(name) if p.is_dir()]:
        for train_name in TRAIN_NAMES:
            train = image_root / train_name
            if train.exists():
                for val_name in VAL_NAMES:
                    val = image_root / val_name
                    if val.exists():
                        candidates.append((image_root.parent, train, val))

    # Alternative YOLO layout: */train/images and */valid/images.
    for train_base in [p for name in TRAIN_NAMES for p in root.rglob(name) if p.is_dir()]:
        train_img = find_dir(train_base, IMAGE_DIR_NAMES) or train_base
        for val_name in VAL_NAMES:
            val_base = train_base.parent / val_name
            if val_base.exists() and val_base.is_dir():
                val_img = find_dir(val_base, IMAGE_DIR_NAMES) or val_base
                candidates.append((train_base.parent, train_img, val_img))

    if not candidates:
        raise FileNotFoundError("Could not infer train/val split from common YOLO layouts.")

    dataset_root, train, val = candidates[0]

    class_ids = set()
    if label_root is not None:
        for label_file in label_root.rglob("*.txt"):
            for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.strip().split()
                if parts:
                    try:
                        class_ids.add(int(float(parts[0])))
                    except ValueError:
                        pass

    names = load_existing_names(root)
    if names is None:
        num_classes = max(class_ids) + 1 if class_ids else 1
        names = {i: f"class_{i}" for i in range(num_classes)}
        if num_classes == 1:
            names = {0: "vehicle"}

    return dataset_root, train, val, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/root/HW2/data/road_vehicle")
    parser.add_argument("--out", default="/root/HW2/task2_yolo_tracking/data.yaml")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dataset_root, train, val, names = infer_split_paths(root)
    data = {
        "path": str(dataset_root),
        "train": str(train.relative_to(dataset_root)),
        "val": str(val.relative_to(dataset_root)),
        "names": names,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    train_count = sum(1 for p in train.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    val_count = sum(1 for p in val.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"Wrote {out}")
    print(f"train images: {train_count}, val images: {val_count}, classes: {len(names)}")
    print(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
