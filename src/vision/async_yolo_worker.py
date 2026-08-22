"""Latest-frame-only YOLO inference outside the navigation hot loop."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Mapping

import numpy as np

from src.vision.auto_relogin_ocr import (
    normalize_ocr_text,
    RapidOcrError,
    RapidOcrTextLocator,
)


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    """Immutable form of the engine's legacy detection dictionary."""

    name: str
    class_id: int
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @classmethod
    def from_legacy(cls, detection: Mapping) -> "DetectionSnapshot":
        x, y = detection["position"]
        height, width = detection["size"]
        return cls(
            name=str(detection["name"]),
            class_id=int(detection["class_id"]),
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            confidence=float(detection["confidence"]),
        )

    def to_legacy(self) -> dict:
        return {
            "name": self.name,
            "class_id": self.class_id,
            "position": (self.x, self.y),
            "size": (self.height, self.width),
            "confidence": self.confidence,
            "score": 1.0 - self.confidence,
        }


@dataclass(frozen=True, slots=True)
class VisionSnapshot:
    """One completed, timestamped inference result."""

    generation: int
    frame_token: object
    captured_at: float
    started_at: float
    completed_at: float
    frame: np.ndarray
    detections: tuple[tuple[str, tuple[DetectionSnapshot, ...]], ...]
    error: str | None = None

    def detections_for(self, class_name: str) -> list[dict]:
        for name, values in self.detections:
            if name == class_name:
                return [value.to_legacy() for value in values]
        return []


@dataclass(frozen=True, slots=True)
class _VisionTask:
    frame: np.ndarray
    frame_token: object
    captured_at: float


class YoloVisionPostprocessor:
    """Apply mob-size and nearby-pet OCR filters inside the vision worker."""

    def __init__(
        self,
        *,
        monster_name: str,
        hero_name: str | None,
        monster_config: Mapping,
        marker_config: Mapping,
        pet_config: Mapping,
        log: Callable[[str], None] | None = None,
        ocr_locator=None,
    ):
        self.monster_name = str(monster_name)
        self.hero_name = None if hero_name is None else str(hero_name)
        self.min_width = max(
            0.0, float(monster_config.get("min_box_width", 0))
        )
        self.min_height = max(
            0.0, float(monster_config.get("min_box_height", 0))
        )
        self.pet_enabled = bool(
            pet_config.get("enable", False)
            and pet_config.get("filter_yolo_mob", True)
        )
        self.pet_target = normalize_ocr_text(
            pet_config.get("yolo_ocr_text", "")
        )
        self.pet_max_size = tuple(float(value) for value in pet_config.get(
            "yolo_ocr_max_box_size", (0, 0)
        ))
        self.pet_max_distance = tuple(
            float(value) for value in pet_config.get(
                "yolo_ocr_max_hero_distance", (0, 0)
            )
        )
        self.pet_min_score = float(
            pet_config.get("yolo_ocr_min_score", 0.8)
        )
        self.pet_box_threshold = float(
            pet_config.get("yolo_ocr_box_threshold", 0.2)
        )
        yolo_cfg = marker_config.get("yolo", {})
        self.hero_anchor = tuple(float(value) for value in yolo_cfg.get(
            "player_anchor", (0.5, 0.5)
        ))
        self.hero_offset = tuple(float(value) for value in yolo_cfg.get(
            "player_offset", (0, 0)
        ))
        self.log = log
        self._ocr_locator = ocr_locator
        self._last_ocr_error = None

    @staticmethod
    def _box(detection):
        x, y = detection["position"]
        height, width = detection["size"]
        return x, y, x + width, y + height

    def _hero_player(self, detections):
        if self.hero_name is None:
            return None
        heroes = detections.get(self.hero_name, ())
        if len(heroes) != 1:
            return None
        hero = heroes[0]
        x, y = hero["position"]
        height, width = hero["size"]
        return (
            x + width * self.hero_anchor[0] + self.hero_offset[0],
            y + height * self.hero_anchor[1] + self.hero_offset[1],
        )

    def _filter_pet(self, frame, monsters, hero_player):
        if not (
            self.pet_enabled
            and self.pet_target
            and hero_player is not None
            and len(self.pet_max_size) == 2
            and len(self.pet_max_distance) == 2
            and min(*self.pet_max_size, *self.pet_max_distance) > 0
        ):
            return monsters
        if self._ocr_locator is None:
            self._ocr_locator = RapidOcrTextLocator()
        frame_h, frame_w = frame.shape[:2]
        kept = []
        for monster in monsters:
            height, width = monster["size"]
            x1, y1, x2, y2 = self._box(monster)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            nearby_small = (
                0 < width <= self.pet_max_size[0]
                and 0 < height <= self.pet_max_size[1]
                and abs(center_x - hero_player[0]) <= self.pet_max_distance[0]
                and abs(center_y - hero_player[1]) <= self.pet_max_distance[1]
            )
            if not nearby_small:
                kept.append(monster)
                continue
            horizontal_padding = int(round(width * 0.4))
            roi = (
                max(0, x1 - horizontal_padding),
                max(0, y2 - int(round(height * 0.1))),
                min(frame_w, x2 + horizontal_padding),
                min(frame_h, y2 + int(round(height * 0.65))),
            )
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                kept.append(monster)
                continue
            try:
                match = self._ocr_locator.locate(
                    frame,
                    roi,
                    (self.pet_target,),
                    min_score=self.pet_min_score,
                    match_mode="exact",
                    box_threshold=self.pet_box_threshold,
                )
                self._last_ocr_error = None
            except (RapidOcrError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if error != self._last_ocr_error and self.log is not None:
                    self.log(
                        f"[vision-worker] Pet OCR unavailable ({error}); "
                        "keeping detection"
                    )
                self._last_ocr_error = error
                kept.append(monster)
                continue
            if match is None:
                kept.append(monster)
        return kept

    def __call__(self, frame, detections):
        output = {
            name: list(values) for name, values in detections.items()
        }
        monsters = [
            monster for monster in output.get(self.monster_name, ())
            if monster["size"][1] >= self.min_width
            and monster["size"][0] >= self.min_height
        ]
        monsters = self._filter_pet(
            frame,
            monsters,
            self._hero_player(output),
        )
        output[self.monster_name] = monsters
        return output


class AsyncYoloWorker:
    """Run YOLO at a bounded rate while always preferring the newest frame."""

    def __init__(
        self,
        detectors: Mapping[str, object],
        confidences: Mapping[str, float],
        *,
        inference_class_names: tuple[str, ...],
        fps: float = 10.0,
        postprocess: Callable[
            [np.ndarray, dict[str, list[dict]]], dict[str, list[dict]]
        ] | None = None,
        log: Callable[[str], None] | None = None,
    ):
        if not detectors:
            raise ValueError("AsyncYoloWorker requires at least one detector")
        self.detectors = dict(detectors)
        self.confidences = {
            str(name): float(value) for name, value in confidences.items()
        }
        if set(self.detectors) != set(self.confidences):
            raise ValueError("detectors and confidences must name the same classes")
        self.inference_class_names = tuple(dict.fromkeys(
            str(name) for name in inference_class_names
        ))
        if not self.inference_class_names:
            raise ValueError("inference_class_names must not be empty")
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("vision fps must be positive")
        self.interval = 1.0 / self.fps
        self.postprocess = postprocess
        self.log = log
        self._condition = threading.Condition()
        self._pending: _VisionTask | None = None
        self._snapshot: VisionSnapshot | None = None
        self._last_submitted_token = None
        self._generation = 0
        self._stop = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = False
            self._thread = threading.Thread(
                target=self._loop,
                name="mob-vision-10fps",
                daemon=True,
            )
            self._thread.start()

    def submit(
        self,
        frame: np.ndarray,
        frame_token: object,
        captured_at: float,
    ) -> bool:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False
        with self._condition:
            if self._stop or frame_token == self._last_submitted_token:
                return False
            self._last_submitted_token = frame_token
            self._pending = _VisionTask(
                frame=frame,
                frame_token=frame_token,
                captured_at=float(captured_at),
            )
            self._condition.notify_all()
        return True

    def latest(self) -> VisionSnapshot | None:
        with self._condition:
            return self._snapshot

    def stop(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._stop = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
        with self._condition:
            if self._thread is thread and (
                thread is None or not thread.is_alive()
            ):
                self._thread = None

    def _run_detectors(self, task: _VisionTask) -> dict[str, list[dict]]:
        grouped: dict[int, tuple[object, list[str]]] = {}
        for class_name, detector in self.detectors.items():
            key = id(detector)
            if key not in grouped:
                grouped[key] = (detector, [])
            grouped[key][1].append(class_name)

        output: dict[str, list[dict]] = {}
        for detector, class_names in grouped.values():
            inference_names = tuple(
                name for name in self.inference_class_names
                if name in class_names
            )
            if not inference_names:
                inference_names = tuple(class_names)
            inference_confidence = min(
                self.confidences[name] for name in class_names
            )
            for class_name in class_names:
                output[class_name] = detector.detect(
                    task.frame,
                    confidence=self.confidences[class_name],
                    class_name=class_name,
                    inference_class_names=inference_names,
                    inference_confidence=inference_confidence,
                    cache_key=task.frame_token,
                )
        if self.postprocess is not None:
            output = self.postprocess(task.frame, output)
        return output

    def _publish(
        self,
        task: _VisionTask,
        started_at: float,
        completed_at: float,
        detections: Mapping[str, list[dict]],
        error: str | None,
    ) -> None:
        frozen = tuple(
            (
                name,
                tuple(DetectionSnapshot.from_legacy(item) for item in values),
            )
            for name, values in sorted(detections.items())
        )
        with self._condition:
            self._generation += 1
            self._snapshot = VisionSnapshot(
                generation=self._generation,
                frame_token=task.frame_token,
                captured_at=task.captured_at,
                started_at=started_at,
                completed_at=completed_at,
                frame=task.frame,
                detections=frozen,
                error=error,
            )

    def _loop(self) -> None:
        next_start_at = 0.0
        while True:
            with self._condition:
                while not self._stop:
                    now = time.monotonic()
                    if self._pending is not None and now >= next_start_at:
                        task = self._pending
                        self._pending = None
                        break
                    timeout = (
                        None if self._pending is None
                        else max(0.0, next_start_at - now)
                    )
                    self._condition.wait(timeout=timeout)
                else:
                    return

            started_at = time.monotonic()
            error = None
            try:
                detections = self._run_detectors(task)
            except Exception as exc:  # keep navigation alive on vision failure
                detections = {name: [] for name in self.detectors}
                error = f"{type(exc).__name__}: {exc}"
                if self.log is not None:
                    self.log(f"[vision-worker] {error}")
            completed_at = time.monotonic()
            self._publish(
                task,
                started_at,
                completed_at,
                detections,
                error,
            )
            if next_start_at <= 0.0:
                next_start_at = started_at + self.interval
            else:
                next_start_at += self.interval
            if next_start_at < completed_at - self.interval:
                # Abandon only a long inference outage; ordinary scheduler
                # jitter is recovered by taking the pending newest frame now.
                next_start_at = completed_at
