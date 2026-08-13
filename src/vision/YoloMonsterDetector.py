"""YOLO-based monster detection with the engine's legacy box interface."""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_model_path(model_path):
    """Resolve a configured model path relative to the project root."""
    path = Path(model_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    candidates = []
    if getattr(sys, "frozen", False):
        # Prefer a model beside the packaged executable so it can be updated
        # independently. Fall back to the PyInstaller one-file extraction
        # directory, where build.bat embeds the known-good checkpoint.
        candidates.append(Path(sys.executable).resolve().parent / path)
    candidates.extend((PROJECT_ROOT / path, Path.cwd() / path))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _to_numpy(value):
    """Convert a Torch tensor or array-like inference result to NumPy."""
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class YoloMonsterDetector:
    """Load one YOLO model and return only its ``mob`` detections.

    Detection dictionaries preserve the existing engine convention:
    ``position`` is ``(x, y)``, ``size`` is ``(height, width)``, and ``score``
    remains lower-is-better for compatibility with older helpers.
    """

    def __init__(
        self,
        model_path,
        *,
        imgsz=1024,
        confidence=0.5,
        iou=0.7,
        max_det=100,
        device="auto",
        half=True,
        class_name="mob",
        model=None,
    ):
        self.model_path = resolve_model_path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO monster model not found: {self.model_path}"
            )

        self.imgsz = int(imgsz)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.class_name = str(class_name)
        self.device = self._resolve_device(device)
        self.half = bool(half) and self.device != "cpu"

        if self.imgsz <= 0:
            raise ValueError("YOLO imgsz must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")
        if not 0.0 <= self.iou <= 1.0:
            raise ValueError("YOLO iou must be between 0 and 1")
        if self.max_det <= 0:
            raise ValueError("YOLO max_det must be positive")

        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "YOLO monster detection requires ultralytics. "
                    "Install requirements.txt in this worktree."
                ) from exc
            model = YOLO(str(self.model_path))
        self.model = model
        self.is_warmed_up = False

        names = getattr(self.model, "names", {})
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        self.class_id = next(
            (
                int(class_id)
                for class_id, name in dict(names).items()
                if str(name) == self.class_name
            ),
            None,
        )
        if self.class_id is None:
            raise ValueError(
                f"YOLO class {self.class_name!r} is missing; classes={names}"
            )

        self.config_signature = (
            str(self.model_path),
            self.imgsz,
            self.confidence,
            self.iou,
            self.max_det,
            str(self.device),
            self.half,
            self.class_name,
        )

    @staticmethod
    def _resolve_device(device):
        if device not in (None, "", "auto"):
            if isinstance(device, str) and device.isdigit():
                return int(device)
            return device

        try:
            import torch
        except ImportError:
            return "cpu"
        return 0 if torch.cuda.is_available() else "cpu"

    @classmethod
    def signature_from_config(cls, config):
        """Return the runtime signature without loading the model."""
        model_path = resolve_model_path(config["model_path"])
        device = cls._resolve_device(config.get("device", "auto"))
        half = bool(config.get("half", True)) and device != "cpu"
        return (
            str(model_path),
            int(config.get("imgsz", 1024)),
            float(config.get("confidence", 0.5)),
            float(config.get("iou", 0.7)),
            int(config.get("max_det", 100)),
            str(device),
            half,
            str(config.get("class_name", "mob")),
        )

    @classmethod
    def from_config(cls, config):
        return cls(
            config["model_path"],
            imgsz=config.get("imgsz", 1024),
            confidence=config.get("confidence", 0.5),
            iou=config.get("iou", 0.7),
            max_det=config.get("max_det", 100),
            device=config.get("device", "auto"),
            half=config.get("half", True),
            class_name=config.get("class_name", "mob"),
        )

    def detect(self, frame, roi=None, confidence=None):
        """Run full-frame inference and optionally retain boxes touching ROI."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("YOLO monster detector expects a BGR image")

        frame_h, frame_w = frame.shape[:2]
        if roi is None:
            roi = (0, 0, frame_w, frame_h)
        roi_x1, roi_y1, roi_x2, roi_y2 = map(int, roi)
        roi_x1 = min(max(0, roi_x1), frame_w)
        roi_x2 = min(max(0, roi_x2), frame_w)
        roi_y1 = min(max(0, roi_y1), frame_h)
        roi_y2 = min(max(0, roi_y2), frame_h)
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return []

        conf = self.confidence if confidence is None else float(confidence)
        if not 0.0 <= conf <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1")

        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=conf,
            iou=self.iou,
            max_det=self.max_det,
            classes=[self.class_id],
            device=self.device,
            half=self.half,
            verbose=False,
        )
        self.is_warmed_up = True
        if not results:
            return []

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return []
        xyxy = _to_numpy(getattr(boxes, "xyxy", None)).reshape(-1, 4)
        confidences = _to_numpy(getattr(boxes, "conf", None)).reshape(-1)
        class_ids = _to_numpy(getattr(boxes, "cls", None)).reshape(-1)

        detections = []
        for box, box_confidence, class_id in zip(
            xyxy, confidences, class_ids
        ):
            if int(round(float(class_id))) != self.class_id:
                continue
            x1, y1, x2, y2 = (int(round(float(value))) for value in box)
            x1 = min(max(0, x1), frame_w)
            x2 = min(max(0, x2), frame_w)
            y1 = min(max(0, y1), frame_h)
            y2 = min(max(0, y2), frame_h)
            if x2 <= x1 or y2 <= y1:
                continue
            if (
                min(x2, roi_x2) <= max(x1, roi_x1)
                or min(y2, roi_y2) <= max(y1, roi_y1)
            ):
                continue

            box_confidence = float(box_confidence)
            detections.append(
                {
                    "name": self.class_name,
                    "class_id": self.class_id,
                    "position": (x1, y1),
                    "size": (y2 - y1, x2 - x1),
                    "confidence": box_confidence,
                    "score": 1.0 - box_confidence,
                }
            )
        return detections

    def warmup(self, frame_size=(1296, 700)):
        """Initialize the inference backend once before game control starts."""
        if self.is_warmed_up:
            return 0.0
        width, height = (int(value) for value in frame_size)
        if width <= 0 or height <= 0:
            raise ValueError("YOLO warmup frame size must be positive")
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        started_at = perf_counter()
        self.detect(frame)
        return (perf_counter() - started_at) * 1000.0
