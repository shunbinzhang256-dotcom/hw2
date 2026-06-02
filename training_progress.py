import csv
import re
import subprocess
from pathlib import Path


ROOT = Path("/root/HW2")
LOG_DIR = ROOT / "outputs/logs"

TASKS = [
    ("task1_resnet18_pretrained", 40, "cls"),
    ("task1_resnet18_scratch", 40, "cls"),
    ("task1_resnet18_se_pretrained", 40, "cls"),
    ("task2_yolov8s", 80, "yolo"),
    ("task3_unet_ce", 150, "seg"),
    ("task3_unet_dice", 150, "seg"),
    ("task3_unet_ce_dice", 150, "seg"),
]


def count_json_epochs(log_path):
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    epochs = [int(x) for x in re.findall(r'"epoch":\s*(\d+)', text)]
    return max(epochs) if epochs else 0


def count_yolo_epochs():
    candidates = [
        ROOT / "outputs/task2/yolov8s_road_vehicle/results.csv",
        ROOT / "outputs/task2/yolov8s_road_vehicle2/results.csv",
    ]
    for path in candidates:
        if path.exists():
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            return len(rows)
    log_path = LOG_DIR / "task2_yolov8s.log"
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"(\d+)/80", text)
    return max([int(x) for x in matches], default=0)


def task_done(name):
    queue = LOG_DIR / "train_queue.log"
    if not queue.exists():
        return False
    return f"END {name} status=0" in queue.read_text(encoding="utf-8", errors="ignore")


def epoch_count(name, kind):
    if task_done(name):
        return dict((n, e) for n, e, _ in TASKS)[name]
    if kind == "yolo":
        return count_yolo_epochs()
    return count_json_epochs(LOG_DIR / f"{name}.log")


def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main():
    rows = []
    total_done = 0
    total_epochs = sum(e for _, e, _ in TASKS)

    for name, expected, kind in TASKS:
        done = min(epoch_count(name, kind), expected)
        total_done += done
        status = "done" if done >= expected else ("running/pending" if done > 0 else "pending")
        rows.append((name, done, expected, done / expected * 100, status))

    current = next((r for r in rows if r[1] < r[2] and r[1] > 0), None)
    if current is None:
        current = next((r for r in rows if r[1] < r[2]), None)

    print(f"Total epoch progress: {total_done}/{total_epochs} ({total_done / total_epochs * 100:.1f}%)")
    if current:
        print(f"Current task: {current[0]} {current[1]}/{current[2]} ({current[3]:.1f}%)")
    print()
    for name, done, expected, pct, status in rows:
        print(f"{name:30s} {done:3d}/{expected:<3d} {pct:6.1f}% {status}")

    print("\nGPU:")
    gpu = run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"])
    print(gpu or "nvidia-smi unavailable")

    print("\nRunning process:")
    ps = run(
        [
            "bash",
            "-lc",
            "ps -eo pid,etime,cmd | grep -E 'run_formal_train_safe|run_gpu0_cls_yolo|run_gpu5_unet|task1_flower_cls/train.py|task3_unet_seg/train.py|yolo detect train' | grep -v grep",
        ]
    )
    print(ps or "no matching training process")


if __name__ == "__main__":
    main()
