import argparse
from collections import defaultdict
from pathlib import Path

import cv2
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
    return (int(width * 0.1), int(height * 0.55)), (int(width * 0.9), int(height * 0.55))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="/root/HW2/outputs/task2/tracked_counted.mp4")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--line", default=None, help="Optional line as x1,y1,x2,y2.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    line_start, line_end = parse_line(args.line, width, height)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    model = YOLO(args.model)
    prev_centers = {}
    counted_ids = set()
    track_history = defaultdict(list)
    total_count = 0

    results = model.track(
        source=str(source),
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img.copy()
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
                track_history[track_id].append(center)

                if track_id in prev_centers and track_id not in counted_ids:
                    last_side = point_side(prev_centers[track_id], line_start, line_end)
                    curr_side = point_side(center, line_start, line_end)
                    if last_side * curr_side < 0:
                        counted_ids.add(track_id)
                        total_count += 1
                prev_centers[track_id] = center

                label = f"ID {track_id} C{class_id} {score:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

        cv2.rectangle(frame, (10, 10), (280, 58), (0, 0, 0), -1)
        cv2.putText(frame, f"Crossing count: {total_count}", (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"Saved counted tracking video: {out_path}")
    print(f"Total crossing count: {total_count}")


if __name__ == "__main__":
    main()
