#!/usr/bin/env python3
"""Reproducible Komari ONNX vs local Ultralytics YOLO comparison.

The two inference modes intentionally run in separate Python environments:

* ``infer-komari`` needs OpenCV, NumPy, and ONNX Runtime.
* ``infer-local`` needs the dependencies already installed by this repository.

``render`` combines the JSON outputs without rerunning either model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
KOMARI_INPUT_SIZE = 640


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_input_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--input-dir", required=True, type=Path)
        subparser.add_argument(
            "--cutoff",
            default="2026-08-17_01-40-25",
            help="Inclusive timestamp encoded in the screenshot filename.",
        )
        subparser.add_argument("--output-json", required=True, type=Path)
        subparser.add_argument("--runs", type=int, default=5)
        subparser.add_argument("--warmup-runs", type=int, default=3)

    komari = subparsers.add_parser("infer-komari")
    add_input_args(komari)
    komari.add_argument("--model", required=True, type=Path)
    komari.add_argument("--confidence", type=float, default=0.5)
    komari.add_argument(
        "--torch-lib",
        type=Path,
        help="Directory containing PyTorch CUDA/cuDNN DLLs for ORT preload.",
    )
    komari.add_argument(
        "--cpu",
        action="store_true",
        help="Force the CPU execution provider.",
    )

    local = subparsers.add_parser("infer-local")
    add_input_args(local)
    local.add_argument("--model", required=True, type=Path)
    local.add_argument("--imgsz", type=int, default=1024)
    local.add_argument("--confidence", type=float, default=0.4)
    local.add_argument("--iou", type=float, default=0.7)
    local.add_argument("--max-det", type=int, default=100)
    local.add_argument("--class-name", default="mob")
    local.add_argument("--cpu", action="store_true")

    render = subparsers.add_parser("render")
    render.add_argument("--input-dir", required=True, type=Path)
    render.add_argument("--komari-json", required=True, type=Path)
    render.add_argument("--local-json", required=True, type=Path)
    render.add_argument("--output-dir", required=True, type=Path)

    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_images(input_dir: Path, cutoff: str) -> list[Path]:
    cutoff_time = time.strptime(cutoff, TIMESTAMP_FORMAT)
    selected: list[tuple[time.struct_time, str, Path]] = []
    for path in input_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = TIMESTAMP_RE.match(path.name)
        if match is None:
            continue
        image_time = time.strptime(match.group(1), TIMESTAMP_FORMAT)
        if image_time >= cutoff_time:
            selected.append((image_time, path.name, path.resolve()))
    selected.sort(key=lambda item: (item[0], item[1]))
    paths = [item[2] for item in selected]
    if not paths:
        raise FileNotFoundError(
            f"No screenshots in {input_dir} at or after {cutoff!r}"
        )
    return paths


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Unable to read image: {path}")
    return image


def median(values: Iterable[float]) -> float:
    return round(float(statistics.median(values)), 3)


def image_kind(path: Path) -> str:
    return "debug-overlay" if "_debug" in path.stem else "raw"


def base_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def komari_preprocess(
    bgr: np.ndarray,
) -> tuple[np.ndarray, float, int, int]:
    """Match Komari's Rust ``preprocess_for_yolo`` implementation."""
    height, width = bgr.shape[:2]
    ratio = min(KOMARI_INPUT_SIZE / width, KOMARI_INPUT_SIZE / height)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    pad_width = (KOMARI_INPUT_SIZE - resized_width) / 2.0
    pad_height = (KOMARI_INPUT_SIZE - resized_height) / 2.0
    top = int(round(pad_height - 0.1))
    bottom = int(round(pad_height + 0.1))
    left = int(round(pad_width - 0.1))
    right = int(round(pad_width + 0.1))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(
        rgb,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None])
    tensor = tensor.astype(np.float32) / 255.0
    return tensor, ratio, left, top


def komari_postprocess(
    output: np.ndarray,
    width: int,
    height: int,
    ratio: float,
    left: int,
    top: int,
    confidence: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for prediction in output.reshape(-1, output.shape[-1]):
        score = float(prediction[4])
        if score < confidence:
            continue
        x1 = (float(prediction[0]) - left) / ratio
        y1 = (float(prediction[1]) - top) / ratio
        x2 = (float(prediction[2]) - left) / ratio
        y2 = (float(prediction[3]) - top) / ratio
        box = [
            int(round(min(max(x1, 0.0), float(width)))),
            int(round(min(max(y1, 0.0), float(height)))),
            int(round(min(max(x2, 0.0), float(width)))),
            int(round(min(max(y2, 0.0), float(height)))),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(
            {
                "box": box,
                "confidence": round(score, 6),
                "class_id": int(round(float(prediction[5]))),
                "class_name": "mob",
            }
        )
    return detections


def infer_komari(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    model_path = args.model.resolve()
    if args.torch_lib:
        torch_lib = args.torch_lib.resolve()
        if not torch_lib.is_dir():
            raise FileNotFoundError(f"PyTorch DLL directory not found: {torch_lib}")
        ort.preload_dlls(directory=str(torch_lib))

    requested_providers = ["CPUExecutionProvider"]
    if not args.cpu and "CUDAExecutionProvider" in ort.get_available_providers():
        requested_providers.insert(0, "CUDAExecutionProvider")

    load_started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        providers=requested_providers,
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0
    input_name = session.get_inputs()[0].name
    files = selected_images(args.input_dir.resolve(), args.cutoff)

    warmup_tensor, _, _, _ = komari_preprocess(read_image(files[0]))
    for _ in range(max(0, args.warmup_runs)):
        session.run(None, {input_name: warmup_tensor})

    image_results: list[dict[str, Any]] = []
    for path in files:
        bgr = read_image(path)
        height, width = bgr.shape[:2]
        preprocess_times: list[float] = []
        inference_times: list[float] = []
        postprocess_times: list[float] = []
        total_times: list[float] = []
        detections: list[dict[str, Any]] = []

        for _ in range(max(1, args.runs)):
            total_started = time.perf_counter()
            stage_started = total_started
            tensor, ratio, left, top = komari_preprocess(bgr)
            preprocess_times.append((time.perf_counter() - stage_started) * 1000.0)

            stage_started = time.perf_counter()
            output = session.run(None, {input_name: tensor})[0][0]
            inference_times.append((time.perf_counter() - stage_started) * 1000.0)

            stage_started = time.perf_counter()
            detections = komari_postprocess(
                output,
                width,
                height,
                ratio,
                left,
                top,
                args.confidence,
            )
            postprocess_times.append((time.perf_counter() - stage_started) * 1000.0)
            total_times.append((time.perf_counter() - total_started) * 1000.0)

        image_results.append(
            {
                "file": path.name,
                "kind": image_kind(path),
                "width": width,
                "height": height,
                "detections": detections,
                "timing_ms": {
                    "preprocess_median": median(preprocess_times),
                    "inference_median": median(inference_times),
                    "postprocess_median": median(postprocess_times),
                    "total_median": median(total_times),
                },
            }
        )
        print(f"Komari {path.name}: {len(detections)} detection(s)")

    write_json(
        args.output_json.resolve(),
        {
            "schema_version": 1,
            "backend": "komari-onnx",
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "config": {
                "input_size": KOMARI_INPUT_SIZE,
                "confidence": args.confidence,
                "preprocessing": "Komari Rust-compatible letterbox",
                "runs": max(1, args.runs),
                "warmup_runs": max(0, args.warmup_runs),
            },
            "environment": {
                **base_environment(),
                "onnxruntime": ort.__version__,
                "available_providers": ort.get_available_providers(),
                "session_providers": session.get_providers(),
                "model_load_ms": round(load_ms, 3),
            },
            "images": image_results,
        },
    )


def _sync_torch_cuda(torch_module: Any, use_cuda: bool) -> None:
    if use_cuda:
        torch_module.cuda.synchronize()


def infer_local(args: argparse.Namespace) -> None:
    import torch
    import ultralytics
    from ultralytics import YOLO

    model_path = args.model.resolve()
    load_started = time.perf_counter()
    model = YOLO(str(model_path))
    load_ms = (time.perf_counter() - load_started) * 1000.0
    names = dict(model.names)
    class_id = next(
        (
            int(candidate_id)
            for candidate_id, candidate_name in names.items()
            if str(candidate_name) == args.class_name
        ),
        None,
    )
    if class_id is None:
        raise ValueError(
            f"Class {args.class_name!r} is missing from model metadata: {names}"
        )

    use_cuda = bool(torch.cuda.is_available() and not args.cpu)
    device: int | str = 0 if use_cuda else "cpu"
    files = selected_images(args.input_dir.resolve(), args.cutoff)
    first_image = read_image(files[0])
    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.confidence,
        "iou": args.iou,
        "max_det": args.max_det,
        "classes": [class_id],
        "device": device,
        "half": use_cuda,
        "verbose": False,
    }
    for _ in range(max(0, args.warmup_runs)):
        model.predict(source=first_image, **predict_kwargs)
    _sync_torch_cuda(torch, use_cuda)

    image_results: list[dict[str, Any]] = []
    for path in files:
        bgr = read_image(path)
        height, width = bgr.shape[:2]
        wall_times: list[float] = []
        preprocess_times: list[float] = []
        inference_times: list[float] = []
        postprocess_times: list[float] = []
        detections: list[dict[str, Any]] = []

        for _ in range(max(1, args.runs)):
            _sync_torch_cuda(torch, use_cuda)
            started = time.perf_counter()
            results = model.predict(source=bgr, **predict_kwargs)
            _sync_torch_cuda(torch, use_cuda)
            wall_times.append((time.perf_counter() - started) * 1000.0)

            result = results[0]
            speed = result.speed or {}
            preprocess_times.append(float(speed.get("preprocess", 0.0)))
            inference_times.append(float(speed.get("inference", 0.0)))
            postprocess_times.append(float(speed.get("postprocess", 0.0)))
            boxes = result.boxes
            detections = []
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy().reshape(-1, 4)
                confidences = boxes.conf.detach().cpu().numpy().reshape(-1)
                class_ids = boxes.cls.detach().cpu().numpy().reshape(-1)
                for box_values, score, detected_class_id in zip(
                    xyxy, confidences, class_ids
                ):
                    box = [int(round(float(value))) for value in box_values]
                    detections.append(
                        {
                            "box": box,
                            "confidence": round(float(score), 6),
                            "class_id": int(round(float(detected_class_id))),
                            "class_name": args.class_name,
                        }
                    )

        image_results.append(
            {
                "file": path.name,
                "kind": image_kind(path),
                "width": width,
                "height": height,
                "detections": detections,
                "timing_ms": {
                    "preprocess_median": median(preprocess_times),
                    "inference_median": median(inference_times),
                    "postprocess_median": median(postprocess_times),
                    "total_median": median(wall_times),
                },
            }
        )
        print(f"Local {path.name}: {len(detections)} detection(s)")

    write_json(
        args.output_json.resolve(),
        {
            "schema_version": 1,
            "backend": "local-ultralytics",
            "model": str(model_path),
            "model_sha256": sha256(model_path),
            "config": {
                "imgsz": args.imgsz,
                "confidence": args.confidence,
                "iou": args.iou,
                "max_det": args.max_det,
                "class_name": args.class_name,
                "class_id": class_id,
                "half": use_cuda,
                "runs": max(1, args.runs),
                "warmup_runs": max(0, args.warmup_runs),
            },
            "environment": {
                **base_environment(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device": str(device),
                "device_name": (
                    torch.cuda.get_device_name(0) if use_cuda else "CPU"
                ),
                "ultralytics": ultralytics.__version__,
                "model_load_ms": round(load_ms, 3),
            },
            "images": image_results,
        },
    )


def iou(first: list[int], second: list[int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def agreement_count(
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    threshold: float = 0.5,
) -> int:
    candidates: list[tuple[float, int, int]] = []
    for first_index, first_detection in enumerate(first):
        for second_index, second_detection in enumerate(second):
            overlap = iou(first_detection["box"], second_detection["box"])
            if overlap >= threshold:
                candidates.append((overlap, first_index, second_index))
    candidates.sort(reverse=True)
    used_first: set[int] = set()
    used_second: set[int] = set()
    matches = 0
    for _, first_index, second_index in candidates:
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches += 1
    return matches


def draw_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    color: tuple[int, int, int],
    label_prefix: str,
) -> np.ndarray:
    annotated = image.copy()
    line_width = max(2, round(min(image.shape[:2]) / 500))
    font_scale = max(0.7, min(image.shape[:2]) / 1300)
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)
        label = f"{label_prefix} {detection['confidence']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            line_width,
        )
        label_y1 = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (min(image.shape[1], x1 + text_width + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, max(text_height, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_width,
            cv2.LINE_AA,
        )
    return annotated


def panel(image: np.ndarray, title: str, width: int = 960) -> np.ndarray:
    target_height = round(image.shape[0] * width / image.shape[1])
    resized = cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA)
    header_height = 54
    result = cv2.copyMakeBorder(
        resized,
        header_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30),
    )
    cv2.putText(
        result,
        title,
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return result


def render(args: argparse.Namespace) -> None:
    komari = json.loads(args.komari_json.read_text(encoding="utf-8"))
    local = json.loads(args.local_json.read_text(encoding="utf-8"))
    komari_images = {item["file"]: item for item in komari["images"]}
    local_images = {item["file"]: item for item in local["images"]}
    komari_input_size = komari["config"]["input_size"]
    komari_confidence = komari["config"]["confidence"]
    local_input_size = local["config"]["imgsz"]
    local_confidence = local["config"]["confidence"]
    common_files = sorted(set(komari_images) & set(local_images))
    if not common_files:
        raise ValueError("The inference files have no screenshots in common")

    output_dir = args.output_dir.resolve()
    panels_dir = output_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    raw_contact_rows: list[np.ndarray] = []

    for filename in common_files:
        image = read_image((args.input_dir / filename).resolve())
        komari_item = komari_images[filename]
        local_item = local_images[filename]
        komari_detections = komari_item["detections"]
        local_detections = local_item["detections"]
        matches = agreement_count(komari_detections, local_detections)
        kind = komari_item["kind"]

        original_panel = panel(image, f"Original | {kind}")
        komari_panel = panel(
            draw_detections(image, komari_detections, (255, 0, 255), "Komari"),
            f"Komari {komari_input_size} conf={komari_confidence:.2f} | "
            f"boxes={len(komari_detections)}",
        )
        local_panel = panel(
            draw_detections(image, local_detections, (0, 190, 0), "Local"),
            f"Local {local_input_size} conf={local_confidence:.2f} | "
            f"boxes={len(local_detections)}",
        )
        comparison = np.concatenate(
            [original_panel, komari_panel, local_panel],
            axis=1,
        )
        panel_path = panels_dir / f"{Path(filename).stem}_compare.jpg"
        cv2.imwrite(
            str(panel_path),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if kind == "raw":
            contact_width = 1920
            contact_height = round(comparison.shape[0] * contact_width / comparison.shape[1])
            raw_contact_rows.append(
                cv2.resize(
                    comparison,
                    (contact_width, contact_height),
                    interpolation=cv2.INTER_AREA,
                )
            )

        rows.append(
            {
                "file": filename,
                "kind": kind,
                "komari_count": len(komari_detections),
                "local_count": len(local_detections),
                "agreement_iou_0_5": matches,
                "komari_total_median_ms": komari_item["timing_ms"][
                    "total_median"
                ],
                "local_total_median_ms": local_item["timing_ms"][
                    "total_median"
                ],
                "panel": str(panel_path),
            }
        )

    if raw_contact_rows:
        contact_sheet = np.concatenate(raw_contact_rows, axis=0)
        cv2.imwrite(
            str(output_dir / "raw_contact_sheet.jpg"),
            contact_sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )

    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    raw_rows = [row for row in rows if row["kind"] == "raw"]
    summary = {
        "komari": komari,
        "local": local,
        "comparison": {
            "agreement_iou_threshold": 0.5,
            "all_images": rows,
            "raw_images": {
                "count": len(raw_rows),
                "komari_detection_total": sum(
                    row["komari_count"] for row in raw_rows
                ),
                "local_detection_total": sum(
                    row["local_count"] for row in raw_rows
                ),
                "agreement_total": sum(
                    row["agreement_iou_0_5"] for row in raw_rows
                ),
                "komari_median_total_ms": median(
                    row["komari_total_median_ms"] for row in raw_rows
                ),
                "local_median_total_ms": median(
                    row["local_total_median_ms"] for row in raw_rows
                ),
            },
        },
    }
    write_json(output_dir / "comparison.json", summary)

    report_lines = [
        "# Komari vs local YOLO",
        "",
        "Primary results use only raw screenshots; debug-overlay screenshots are listed separately.",
        "",
        "| Screenshot | Type | Komari boxes | Local boxes | Agreement (IoU >= 0.5) | Komari ms | Local ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            "| {file} | {kind} | {komari_count} | {local_count} | "
            "{agreement_iou_0_5} | {komari_total_median_ms:.3f} | "
            "{local_total_median_ms:.3f} |".format(**row)
        )
    report_lines.extend(
        [
            "",
            "## Raw screenshot totals",
            "",
            f"- Screenshots: {len(raw_rows)}",
            f"- Komari detections: {summary['comparison']['raw_images']['komari_detection_total']}",
            f"- Local detections: {summary['comparison']['raw_images']['local_detection_total']}",
            f"- Matched detections at IoU >= 0.5: {summary['comparison']['raw_images']['agreement_total']}",
            f"- Komari median end-to-end time: {summary['comparison']['raw_images']['komari_median_total_ms']:.3f} ms",
            f"- Local median end-to-end time: {summary['comparison']['raw_images']['local_median_total_ms']:.3f} ms",
        ]
    )
    (output_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.command == "infer-komari":
        infer_komari(args)
    elif args.command == "infer-local":
        infer_local(args)
    elif args.command == "render":
        render(args)
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
