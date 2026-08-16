import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.input.CaptureCardCapturor import CaptureCardCapturor


class _FakeCapture:
    def __init__(
        self,
        frames=(),
        *,
        opened=True,
        width=3840,
        height=2160,
        fps=60.0,
        subtype=CaptureCardCapturor.DIRECTSHOW_RGB24_SUBTYPE,
    ):
        self.opened = opened
        self.frames = list(frames)
        self.values = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FOURCC: subtype,
        }
        self.set_calls = []
        self.released = False
        self._lock = threading.Lock()
        self._released_event = threading.Event()

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return True

    def get(self, prop):
        return self.values.get(prop, 0)

    def read(self):
        with self._lock:
            if self.frames:
                return True, self.frames.pop(0)
        self._released_event.wait(0.005)
        return False, None

    def release(self):
        self.released = True
        self._released_event.set()


class _BlockingAfterFirstCapture(_FakeCapture):
    """Model a driver whose second read blocks instead of returning False."""

    def read(self):
        with self._lock:
            if self.frames:
                return True, self.frames.pop(0)
        self._released_event.wait(5.0)
        return False, None


def _cfg(**overrides):
    capture_card = {
        "device_index": 0,
        "device_name": "AVerMedia GC573 1 Capture",
        "width": 3840,
        "height": 2160,
        "fps": 60,
        "pixel_format": "RGB24",
        "frame_timeout": 1.0,
        "startup_timeout": 0.5,
        "shutdown_timeout": 0.5,
        "read_retry_interval": 0.001,
    }
    capture_card.update(overrides)
    return {"capture_card": capture_card}


class CaptureCardCapturorTests(unittest.TestCase):
    def setUp(self):
        platform_patch = patch(
            "src.input.CaptureCardCapturor.sys.platform", "win32"
        )
        platform_patch.start()
        self.addCleanup(platform_patch.stop)

    def test_opens_by_index_and_retains_only_latest_4k_frame(self):
        first = np.zeros((2160, 3840, 3), dtype=np.uint8)
        second = np.zeros((2160, 3840, 3), dtype=np.uint8)
        first[0, 0] = (1, 2, 3)
        second[0, 0] = (4, 5, 6)
        fake = _FakeCapture((first, second))

        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ) as video_capture:
            capture = CaptureCardCapturor(
                _cfg(device_index=7, device_name="Friendly diagnostic name")
            )
            try:
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    frame, timestamp = capture.get_frame_snapshot()
                    if frame is not None and tuple(frame[0, 0]) == (4, 5, 6):
                        break
                    time.sleep(0.005)

                self.assertIsNotNone(frame)
                self.assertGreater(timestamp, 0)
                self.assertEqual(frame.shape, (2160, 3840, 3))
                self.assertFalse(frame.flags.writeable)
                self.assertEqual(tuple(frame[0, 0]), (4, 5, 6))
                video_capture.assert_called_once_with(7, cv2.CAP_DSHOW)
                self.assertEqual(capture.capture_profile, "capture_card")
                self.assertIn("Friendly diagnostic name", capture.window_title)

                requested_properties = [prop for prop, _ in fake.set_calls]
                self.assertIn(cv2.CAP_PROP_FRAME_WIDTH, requested_properties)
                self.assertIn(cv2.CAP_PROP_FRAME_HEIGHT, requested_properties)
                self.assertIn(cv2.CAP_PROP_FPS, requested_properties)
                self.assertNotIn(cv2.CAP_PROP_FOURCC, requested_properties)

                # The same immutable-by-contract reference can be handed off
                # without copying roughly 25 MB on every consumer request.
                next_frame = capture.get_frame()
                self.assertIs(next_frame, frame)
                self.assertEqual(tuple(next_frame[0, 0]), (4, 5, 6))
            finally:
                capture.stop()

        self.assertTrue(fake.released)
        self.assertTrue(capture.is_closed)
        self.assertIsNone(capture.get_frame())

    def test_rejects_wrong_negotiated_resolution(self):
        fake = _FakeCapture(width=1920, height=1080)
        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ):
            with self.assertRaisesRegex(RuntimeError, "3840x2160"):
                CaptureCardCapturor(_cfg())
        self.assertTrue(fake.released)

    def test_rejects_wrong_negotiated_fps(self):
        fake = _FakeCapture(fps=30.0)
        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ):
            with self.assertRaisesRegex(RuntimeError, "60 FPS"):
                CaptureCardCapturor(_cfg())
        self.assertTrue(fake.released)

    def test_rejects_non_rgb24_directshow_subtype(self):
        fake = _FakeCapture(subtype=cv2.VideoWriter_fourcc(*"NV12"))
        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ):
            with self.assertRaisesRegex(RuntimeError, "not exposing RGB24"):
                CaptureCardCapturor(_cfg())
        self.assertTrue(fake.released)

    def test_rejects_configuration_that_drifts_from_required_4k60_mode(self):
        with self.assertRaisesRegex(ValueError, "width=3840"):
            CaptureCardCapturor(_cfg(width=1920, height=1080, fps=30))

        with self.assertRaisesRegex(ValueError, "pixel_format=RGB24"):
            CaptureCardCapturor(_cfg(pixel_format="YUY2"))

    def test_stream_timeout_invalidates_stale_frame(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        fake = _FakeCapture((frame,))
        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ):
            capture = CaptureCardCapturor(_cfg(frame_timeout=0.02))
            try:
                time.sleep(0.05)
                self.assertEqual(capture.get_frame_snapshot(), (None, None))
                self.assertTrue(capture.is_closed)
            finally:
                capture.stop()

    def test_invalid_runtime_frame_releases_capture_immediately(self):
        valid = np.zeros((2160, 3840, 3), dtype=np.uint8)
        invalid = np.zeros((1080, 1920, 3), dtype=np.uint8)
        fake = _FakeCapture((valid,))
        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ):
            capture = CaptureCardCapturor(_cfg())
            with fake._lock:
                fake.frames.append(invalid)

            self.assertTrue(fake._released_event.wait(0.5))
            self.assertTrue(fake.released)
            self.assertTrue(capture.is_closed)
            self.assertTrue(capture.is_terminated)
            self.assertIsNone(capture.get_frame())

    def test_snapshot_detects_timeout_even_when_driver_read_is_blocked(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        fake = _BlockingAfterFirstCapture((frame,))
        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ):
            capture = CaptureCardCapturor(_cfg(frame_timeout=0.02))
            try:
                time.sleep(0.04)
                self.assertEqual(capture.get_frame_snapshot(), (None, None))
                self.assertTrue(capture.is_closed)
            finally:
                capture.stop()

    def test_startup_fails_when_stream_produces_no_frame(self):
        fake = _FakeCapture()
        started_at = time.monotonic()
        with patch(
            "src.input.CaptureCardCapturor.cv2.VideoCapture",
            return_value=fake,
        ):
            with self.assertRaisesRegex(RuntimeError, "first frame"):
                CaptureCardCapturor(
                    _cfg(frame_timeout=0.01, startup_timeout=0.08)
                )
        self.assertGreaterEqual(time.monotonic() - started_at, 0.05)
        self.assertTrue(fake.released)

    def test_rejects_non_windows_platform(self):
        with patch("src.input.CaptureCardCapturor.sys.platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "requires Windows"):
                CaptureCardCapturor(_cfg())


if __name__ == "__main__":
    unittest.main()
