import threading
import time

import numpy as np

from src.navigation.minimap_viewport import DynamicMinimapLocator
from src.vision.async_yolo_worker import (
    AsyncYoloWorker,
    YoloVisionPostprocessor,
)


def _wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


class _FakeDetector:
    class_ids = {"mob": 0, "hero": 1}

    def __init__(self, gate=None):
        self.gate = gate
        self.started = threading.Event()
        self.calls = []
        self._cache_key = None
        self._cache_calls = 0

    def detect(self, frame, *, class_name, cache_key, **kwargs):
        self.started.set()
        if self.gate is not None:
            self.gate.wait(timeout=1.0)
        if cache_key != self._cache_key:
            self._cache_key = cache_key
            self._cache_calls += 1
        self.calls.append((cache_key, class_name))
        return [{
            "name": class_name,
            "class_id": self.class_ids[class_name],
            "position": (int(frame[0, 0, 0]), 2),
            "size": (4, 5),
            "confidence": 0.9,
            "score": 0.1,
        }]


def test_dynamic_minimap_locator_discovers_once_and_preserves_native_size():
    calls = []

    def detect(frame):
        calls.append(frame.shape[:2])
        return 19, 202, 276, 365

    locator = DynamicMinimapLocator(detect)
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    first, first_view = locator.crop(frame)
    second, second_view = locator.crop(frame)

    assert calls == [(2160, 3840)]
    assert first is second
    assert first.rect == (20, 203, 274, 363)
    assert first_view.shape == (363, 274, 3)
    assert second_view.shape == (363, 274, 3)
    assert np.shares_memory(frame, first_view)


def test_dynamic_minimap_locator_reacquires_for_a_different_frame_size():
    sizes = {
        (2160, 3840): (19, 202, 276, 365),
        (2013, 3579): (16, 188, 344, 284),
    }
    locator = DynamicMinimapLocator(lambda frame: sizes[frame.shape[:2]])

    first = locator.acquire(np.zeros((2160, 3840, 3), dtype=np.uint8))
    second = locator.acquire(np.zeros((2013, 3579, 3), dtype=np.uint8))

    assert first.rect == (20, 203, 274, 363)
    assert second.rect == (17, 189, 342, 282)
    assert second.generation == first.generation + 1


def test_dynamic_minimap_locator_forced_check_only_changes_generation_for_new_roi():
    rectangles = iter(((19, 202, 276, 365), (19, 202, 276, 365), (20, 202, 310, 365)))
    locator = DynamicMinimapLocator(lambda _frame: next(rectangles))
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    first = locator.acquire(frame)
    unchanged = locator.acquire(frame, force=True)
    changed = locator.acquire(frame, force=True)

    assert unchanged is first
    assert changed.rect == (21, 203, 308, 363)
    assert changed.generation == first.generation + 1


def test_async_yolo_worker_shares_one_prediction_and_publishes_both_classes():
    detector = _FakeDetector()
    worker = AsyncYoloWorker(
        {"mob": detector, "hero": detector},
        {"mob": 0.7, "hero": 0.85},
        inference_class_names=("mob", "hero"),
        fps=100.0,
    )
    worker.start()
    try:
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[0, 0, 0] = 7
        assert worker.submit(frame, ("capture", 1.0), 1.0)
        snapshot = _wait_for(worker.latest)

        assert snapshot.frame_token == ("capture", 1.0)
        assert snapshot.detections_for("mob")[0]["position"] == (7, 2)
        assert snapshot.detections_for("hero")[0]["position"] == (7, 2)
        assert detector._cache_calls == 1
    finally:
        worker.stop()


def test_async_yolo_worker_drops_queued_old_frames_for_the_latest():
    gate = threading.Event()
    detector = _FakeDetector(gate)
    worker = AsyncYoloWorker(
        {"mob": detector},
        {"mob": 0.7},
        inference_class_names=("mob",),
        fps=100.0,
    )
    worker.start()
    try:
        first = np.zeros((4, 4, 3), dtype=np.uint8)
        second = np.zeros((4, 4, 3), dtype=np.uint8)
        third = np.zeros((4, 4, 3), dtype=np.uint8)
        first[0, 0, 0] = 1
        second[0, 0, 0] = 2
        third[0, 0, 0] = 3
        worker.submit(first, "first", 1.0)
        assert detector.started.wait(timeout=1.0)
        worker.submit(second, "second", 2.0)
        worker.submit(third, "third", 3.0)
        gate.set()

        snapshot = _wait_for(
            lambda: worker.latest()
            if worker.latest() is not None
            and worker.latest().frame_token == "third"
            else None
        )
        assert snapshot.detections_for("mob")[0]["position"] == (3, 2)
        assert "second" not in [token for token, _ in detector.calls]
    finally:
        worker.stop()


def test_vision_postprocessor_runs_nearby_pet_ocr_off_the_navigation_thread():
    class Locator:
        def __init__(self):
            self.rois = []

        def locate(self, frame, roi, targets, **kwargs):
            self.rois.append((roi, targets))
            return object()

    locator = Locator()
    postprocess = YoloVisionPostprocessor(
        monster_name="mob",
        hero_name="hero",
        monster_config={"min_box_width": 20, "min_box_height": 20},
        marker_config={
            "yolo": {"player_anchor": (0.5, 0.5), "player_offset": (0, 0)}
        },
        pet_config={
            "enable": True,
            "filter_yolo_mob": True,
            "yolo_ocr_text": "花蘑菇仔",
            "yolo_ocr_max_box_size": (50, 50),
            "yolo_ocr_max_hero_distance": (80, 80),
            "yolo_ocr_min_score": 0.8,
            "yolo_ocr_box_threshold": 0.2,
        },
        ocr_locator=locator,
    )
    detections = {
        "hero": [{
            "name": "hero", "class_id": 1, "position": (100, 100),
            "size": (40, 40), "confidence": 0.95, "score": 0.05,
        }],
        "mob": [
            {
                "name": "mob", "class_id": 0, "position": (130, 110),
                "size": (30, 30), "confidence": 0.9, "score": 0.1,
            },
            {
                "name": "mob", "class_id": 0, "position": (400, 400),
                "size": (60, 60), "confidence": 0.9, "score": 0.1,
            },
        ],
    }

    result = postprocess(np.zeros((600, 800, 3), dtype=np.uint8), detections)

    assert [item["position"] for item in result["mob"]] == [(400, 400)]
    assert len(locator.rois) == 1
    assert locator.rois[0][1] == ("花蘑菇仔",)
