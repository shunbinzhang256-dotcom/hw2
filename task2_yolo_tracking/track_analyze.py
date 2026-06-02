"""Run YOLOv8 + ByteTrack on a video, log per-frame tracks, count line crossings,
and detect ID switches for the report.

Outputs:
  - tracked_counted.mp4 : annotated video (boxes, IDs, class names, counting line, count)
  - track_log.json      : per-frame list of {frame, id, cls, name, conf, cx, cy, x1,y1,x2,y2}
  - id_switch_report.json : detected ID switches + summary stats
  - frames/             : raw frames saved for chosen ID-switch episodes (for the report)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def point_side(point, line_start, line_end):
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)


def parse_line(line_arg, width, height):
    if line_arg:
        values = [float(v) for v in line_arg.split(",")]
        if len(values) != 4:
            raise ValueError("--line must be x1,y1,x2,y2")
        return (int(values[0]), int(values[1])), (int(values[2]), int(values[3]))
    return (int(width * 0.05), int(height * 0.6)), (int(width * 0.95), int(height * 0.6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--outdir", default="/root/HW2/outputs/task2")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--line", default=None, help="Optional line as x1,y1,x2,y2.")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save every raw frame to frames/ for later inspection.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / "frames"
    if args.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    line_start, line_end = parse_line(args.line, width, height)
    writer = cv2.VideoWriter(
        str(outdir / "tracked_counted.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(args.model)
    names = model.names

    prev_centers = {}
    counted_ids = set()
    track_history = defaultdict(list)
    total_count = 0
    log = []

    results = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        verbose=False,
    )

    frame_idx = 0
    for result in results:
        frame = result.orig_img.copy()
        raw = result.orig_img
        if args.save_frames:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:04d}.jpg"), raw)

        cv2.line(frame, line_start, line_end, (0, 255, 255), 3)

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            for box, class_id, track_id, score in zip(xyxy, cls, ids, confs):
                x1, y1, x2, y2 = box.astype(int)
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                track_history[track_id].append((frame_idx, center))
                name = names.get(int(class_id), str(class_id))

                log.append({
                    "frame": frame_idx, "id": int(track_id), "cls": int(class_id),
                    "name": name, "conf": float(score),
                    "cx": center[0], "cy": center[1],
                    "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                })

                if track_id in prev_centers and track_id not in counted_ids:
                    last_side = point_side(prev_centers[track_id], line_start, line_end)
                    curr_side = point_side(center, line_start, line_end)
                    if last_side * curr_side < 0:
                        counted_ids.add(track_id)
                        total_count += 1
                prev_centers[track_id] = center

                label = f"ID{track_id} {name} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 0), 2)

        cv2.rectangle(frame, (10, 10), (320, 58), (0, 0, 0), -1)
        cv2.putText(frame, f"Crossing count: {total_count}", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        writer.write(frame)
        frame_idx += 1

    writer.release()

    # ---- summary stats ----
    n_frames = frame_idx
    track_lengths = {tid: len(h) for tid, h in track_history.items()}
    n_ids = len(track_history)

    # ---- ID-switch heuristic: a track ends, and within `gap` frames a NEW id
    # appears near the last known location -> likely the same physical object. ----
    def last_seen(tid):
        return track_history[tid][-1]

    def first_seen(tid):
        return track_history[tid][0]

    GAP = 15          # frames
    DIST = 80         # pixels
    switches = []
    ends = sorted(track_history.keys(), key=lambda t: last_seen(t)[0])
    starts = sorted(track_history.keys(), key=lambda t: first_seen(t)[0])
    for old in ends:
        of, oc = last_seen(old)
        for new in starts:
            if new == old:
                continue
            nf, nc = first_seen(new)
            if 0 < nf - of <= GAP:
                d = float(np.hypot(nc[0] - oc[0], nc[1] - oc[1]))
                if d <= DIST:
                    switches.append({
                        "old_id": int(old), "new_id": int(new),
                        "old_last_frame": int(of), "new_first_frame": int(nf),
                        "gap_frames": int(nf - of), "dist_px": round(d, 1),
                        "old_last_center": [int(oc[0]), int(oc[1])],
                        "new_first_center": [int(nc[0]), int(nc[1])],
                    })
    switches.sort(key=lambda s: (s["dist_px"], s["gap_frames"]))

    report = {
        "video": str(source),
        "n_frames": n_frames,
        "fps": round(fps, 2),
        "resolution": [width, height],
        "counting_line": [list(line_start), list(line_end)],
        "total_crossing_count": total_count,
        "counted_ids": sorted(int(i) for i in counted_ids),
        "n_unique_track_ids": n_ids,
        "max_track_len": max(track_lengths.values()) if track_lengths else 0,
        "id_switch_candidates": switches[:20],
        "n_id_switch_candidates": len(switches),
    }

    with open(outdir / "track_log.json", "w") as f:
        json.dump(log, f)
    with open(outdir / "id_switch_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items() if k != "id_switch_candidates"}, indent=2))
    print(f"Top ID-switch candidates: {report['n_id_switch_candidates']}")
    for s in switches[:8]:
        print(s)


if __name__ == "__main__":
    main()
