"""Evaluate the configured YOLO ``hero`` class on saved game screenshots.

The tool writes an annotated contact sheet plus CSV details so model changes
can be reviewed without starting the bot or sending any input.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=(
            ROOT
            / "models/yolo/yolov8n_1024_rect_hero_mob_level_ge10_all_pets_2860_best.pt"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "screenshot",
    )
    parser.add_argument("--pattern", default="*_img_frame.png")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_device(value: str):
    if value != "auto":
        return int(value) if value.isdigit() else value
    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def fit_panel(image: np.ndarray, size=(640, 380)) -> np.ndarray:
    output_w, output_h = size
    scale = min(output_w / image.shape[1], output_h / image.shape[0])
    resized = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((output_h, output_w, 3), 28, dtype=np.uint8)
    y = (output_h - resized.shape[0]) // 2
    x = (output_w - resized.shape[1]) // 2
    panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return panel


def main() -> int:
    args = parse_args()
    output = args.output or args.input / "hero_yolo_eval"
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} in {args.input}"
        )

    model = YOLO(str(args.model))
    class_id = next(
        (int(key) for key, name in model.names.items() if name == "hero"),
        None,
    )
    if class_id is None:
        raise ValueError(f"Model has no hero class: {model.names}")
    device = resolve_device(args.device)
    results = model.predict(
        source=[str(path) for path in paths],
        imgsz=args.imgsz,
        conf=args.confidence,
        iou=args.iou,
        classes=[class_id],
        device=device,
        half=device != "cpu",
        verbose=False,
    )

    rows = []
    panels = []
    for path, result in zip(paths, results):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        detections = []
        for index, (xyxy, confidence) in enumerate(zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy())):
            x1, y1, x2, y2 = (round(float(value)) for value in xyxy)
            confidence = float(confidence)
            detections.append((x1, y1, x2, y2, confidence))
            rows.append({
                "file": path.name,
                "detection": index,
                "confidence": f"{confidence:.6f}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": round((x1 + x2) / 2),
                "center_y": round((y1 + y2) / 2),
            })
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 80, 220), 8)
            cv2.putText(
                image,
                f"hero {confidence:.3f}",
                (x1, max(40, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 80, 220),
                3,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            f"{path.name} | hero={len(detections)}",
            (24, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.15,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        panels.append(fit_panel(image))

    csv_path = output / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "file", "detection", "confidence", "x1", "y1", "x2", "y2",
            "center_x", "center_y",
        ))
        writer.writeheader()
        writer.writerows(rows)

    columns = 3
    blank = np.full_like(panels[0], 28)
    while len(panels) % columns:
        panels.append(blank.copy())
    sheet = np.vstack([
        np.hstack(panels[index:index + columns])
        for index in range(0, len(panels), columns)
    ])
    sheet_path = output / "contact_sheet.jpg"
    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"Unable to write {sheet_path}")
    print(f"images={len(paths)} detections={len(rows)}")
    print(csv_path)
    print(sheet_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
