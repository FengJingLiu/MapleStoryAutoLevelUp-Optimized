"""Low-latency OpenCV DirectShow capture for the GC573.

The GC573 exposes RGB24 through the DirectShow ``MEDIASUBTYPE_RGB24``
subtype.  OpenCV reports the first DWORD of that subtype GUID through
``CAP_PROP_FOURCC`` (``0xE436EB7D``).  It must not be replaced with the
``RGB3`` FourCC: doing so makes the GC573 driver reject 3840x2160 capture.

Although the DirectShow source subtype is named RGB24, OpenCV exposes the
resulting three-channel ndarray in its conventional BGR channel order.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

from src.utils.logger import logger


class CaptureCardCapturor:
    """Continuously retain only the newest 4K frame from a capture card."""

    FRAME_WIDTH = 3840
    FRAME_HEIGHT = 2160
    FRAME_RATE = 60.0
    FPS_TOLERANCE = 0.5

    # Data1 from DirectShow's MEDIASUBTYPE_RGB24 GUID:
    # {E436EB7D-524F-11CE-9F53-0020AF0BA770}
    DIRECTSHOW_RGB24_SUBTYPE = 0xE436EB7D

    def __init__(self, cfg: dict[str, Any]):
        if sys.platform != "win32":
            raise RuntimeError(
                "CaptureCardCapturor requires Windows and OpenCV DirectShow"
            )

        capture_cfg = cfg.get("capture_card", {})
        if not isinstance(capture_cfg, dict):
            raise TypeError("capture_card configuration must be a mapping")

        self.cfg = cfg
        self.device_index = int(capture_cfg.get("device_index", 0))
        self.device_name = str(
            capture_cfg.get("device_name", "AVerMedia GC573 1 Capture")
        )
        self.requested_width = self._positive_int(
            capture_cfg.get("width", self.FRAME_WIDTH),
            "capture_card.width",
        )
        self.requested_height = self._positive_int(
            capture_cfg.get("height", self.FRAME_HEIGHT),
            "capture_card.height",
        )
        self.requested_fps = self._positive_float(
            capture_cfg.get("fps", self.FRAME_RATE),
            "capture_card.fps",
        )
        self.requested_pixel_format = str(
            capture_cfg.get("pixel_format", "RGB24")
        ).strip().upper()
        if (
            self.requested_width != self.FRAME_WIDTH
            or self.requested_height != self.FRAME_HEIGHT
            or abs(self.requested_fps - self.FRAME_RATE) > 1e-6
            or self.requested_pixel_format != "RGB24"
        ):
            raise ValueError(
                "This capture backend requires capture_card width=3840, "
                "height=2160, fps=60, and pixel_format=RGB24"
            )

        # Compatibility metadata used by capture preprocessing and diagnostics.
        # This is not the title of a clickable desktop game window.
        self.capture_profile = "capture_card"
        self.window_title = (
            f"DirectShow device {self.device_name} "
            f"(index {self.device_index})"
        )
        self.frame_timeout = self._positive_float(
            capture_cfg.get(
                "frame_timeout",
                cfg.get("game_window", {}).get("frame_timeout", 1.0),
            ),
            "capture_card.frame_timeout",
        )
        self.startup_timeout = self._positive_float(
            capture_cfg.get("startup_timeout", 3.0),
            "capture_card.startup_timeout",
        )
        self.shutdown_timeout = self._positive_float(
            capture_cfg.get("shutdown_timeout", 2.0),
            "capture_card.shutdown_timeout",
        )
        self.read_retry_interval = self._positive_float(
            capture_cfg.get("read_retry_interval", 0.01),
            "capture_card.read_retry_interval",
        )

        self.frame: np.ndarray | None = None
        self.last_frame_time = 0.0
        self.is_closed = False
        self.is_terminated = False
        self.is_static_frame = False
        self.lock = threading.Lock()
        self.fps = 0

        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_subtype = 0

        self._stop_event = threading.Event()
        self._startup_event = threading.Event()
        self._capture_started_at = 0.0
        self._startup_error: RuntimeError | None = None
        self._timeout_reported = False
        self._capture_thread: threading.Thread | None = None

        # device_name is intentionally diagnostic only.  OpenCV's DirectShow
        # backend opens capture devices by numeric index, not friendly name.
        logger.info(
            "[CaptureCardCapturor] Opening "
            f"device_index={self.device_index}, expected_name={self.device_name!r}"
        )
        self.capture = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(
                "[CaptureCardCapturor] Unable to open DirectShow capture "
                f"device index {self.device_index} ({self.device_name})"
            )

        try:
            self._configure_and_validate_stream()
            # The first-frame budget starts after device opening and stream
            # negotiation. Driver setup time must not consume startup_timeout.
            self._capture_started_at = time.monotonic()
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="GC573-DirectShow-Capture",
                daemon=True,
            )
            self._capture_thread.start()

            if not self._startup_event.wait(self.startup_timeout):
                raise RuntimeError(
                    "[CaptureCardCapturor] Timed out waiting for the first "
                    f"frame after {self.startup_timeout:.2f}s"
                )
            if self._startup_error is not None:
                raise self._startup_error
        except Exception:
            self.stop()
            raise

        logger.info(
            "[CaptureCardCapturor] Ready: "
            f"{self.actual_width}x{self.actual_height}@{self.actual_fps:g}, "
            f"DirectShow subtype=0x{self.actual_subtype:08X} (RGB24), "
            "OpenCV output=BGR"
        )

    @staticmethod
    def _positive_float(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive number") from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive number")
        return result

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if result <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return result

    def _configure_and_validate_stream(self) -> None:
        """Request 4K60 and fail if DirectShow negotiated another mode."""
        # Do not set CAP_PROP_FOURCC to RGB3.  The GC573 uses the DirectShow
        # RGB24 media subtype GUID and rejects 4K negotiation after RGB3 is set.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        self.capture.set(cv2.CAP_PROP_FPS, self.requested_fps)

        # This property is unsupported by some DirectShow drivers.  It is only
        # a best-effort hint; the reader thread independently overwrites its
        # single application-side frame slot on every successful read.
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        subtype = self.capture.get(cv2.CAP_PROP_FOURCC)

        try:
            self.actual_width = int(round(float(width)))
            self.actual_height = int(round(float(height)))
            self.actual_fps = float(fps)
            self.actual_subtype = int(float(subtype)) & 0xFFFFFFFF
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "[CaptureCardCapturor] DirectShow returned invalid stream metadata"
            ) from exc

        if (
            self.actual_width != self.requested_width
            or self.actual_height != self.requested_height
        ):
            raise RuntimeError(
                "[CaptureCardCapturor] Failed to negotiate 3840x2160: "
                f"driver reported {self.actual_width}x{self.actual_height}"
            )
        if not math.isfinite(self.actual_fps) or abs(
            self.actual_fps - self.requested_fps
        ) > self.FPS_TOLERANCE:
            raise RuntimeError(
                "[CaptureCardCapturor] Failed to negotiate 60 FPS: "
                f"driver reported {self.actual_fps:g} FPS"
            )
        if self.actual_subtype != self.DIRECTSHOW_RGB24_SUBTYPE:
            raise RuntimeError(
                "[CaptureCardCapturor] GC573 is not exposing RGB24: "
                f"DirectShow subtype=0x{self.actual_subtype:08X}, "
                f"expected=0x{self.DIRECTSHOW_RGB24_SUBTYPE:08X}"
            )

        self.fps = int(round(self.actual_fps))

    def _capture_loop(self) -> None:
        """Read continuously so the shared slot always contains the newest frame."""
        while not self._stop_event.is_set():
            failure_reason = "DirectShow returned no frame"
            try:
                success, frame = self.capture.read()
            except Exception as exc:  # OpenCV backends may surface driver errors.
                success, frame = False, None
                failure_reason = f"DirectShow read failed: {exc}"

            if self._stop_event.is_set():
                break

            now = time.monotonic()
            if not success or frame is None:
                self._handle_failed_read(now, failure_reason)
                self._stop_event.wait(self.read_retry_interval)
                continue

            validation_error = self._validate_frame(frame)
            if validation_error is not None:
                with self.lock:
                    self.frame = None
                    self.is_closed = True
                    self._startup_error = RuntimeError(validation_error)
                self._startup_event.set()
                logger.error(validation_error)
                # A terminal stream contract violation must release the card
                # immediately, even if the main loop has not stopped yet.
                self.stop()
                return

            # Published buffers are shared without a 24.9 MB copy.  Make the
            # contract enforceable so a consumer cannot corrupt a snapshot
            # still referenced elsewhere in the processing pipeline.
            frame.setflags(write=False)

            recovered = False
            with self.lock:
                recovered = self._timeout_reported
                # OpenCV allocates a new ndarray for each read.  Replacing this
                # reference is the only buffering performed by this class.
                self.frame = frame
                self.last_frame_time = now
                self.is_closed = False
                self._timeout_reported = False
            self._startup_event.set()
            if recovered:
                logger.info("[CaptureCardCapturor] Capture stream recovered")

    def _validate_frame(self, frame: np.ndarray) -> str | None:
        if not isinstance(frame, np.ndarray):
            return "[CaptureCardCapturor] DirectShow returned a non-array frame"
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            return (
                "[CaptureCardCapturor] Expected an 8-bit three-channel RGB24 "
                f"frame, got shape={frame.shape}, dtype={frame.dtype}"
            )
        if frame.shape[:2] != (self.FRAME_HEIGHT, self.FRAME_WIDTH):
            return (
                "[CaptureCardCapturor] Frame dimensions changed after startup: "
                f"got {frame.shape[1]}x{frame.shape[0]}, expected "
                f"{self.FRAME_WIDTH}x{self.FRAME_HEIGHT}"
            )
        return None

    def _handle_failed_read(self, now: float, reason: str) -> None:
        with self.lock:
            waiting_for_first_frame = self.last_frame_time <= 0
            freshness_origin = (
                self._capture_started_at
                if waiting_for_first_frame
                else self.last_frame_time
            )
            timeout_seconds = (
                self.startup_timeout
                if waiting_for_first_frame
                else self.frame_timeout
            )
            if now - freshness_origin <= timeout_seconds:
                return

            first_report = not self._timeout_reported
            self._timeout_reported = True
            self.frame = None
            self.is_closed = True
            if self.last_frame_time <= 0 and self._startup_error is None:
                self._startup_error = RuntimeError(
                    "[CaptureCardCapturor] Capture stream timed out before "
                    f"the first frame ({reason})"
                )
                self._startup_event.set()

        if first_report:
            logger.warning(
                "[CaptureCardCapturor] Capture stream timed out after "
                f"{timeout_seconds:.2f}s ({reason})"
            )

    def get_frame_snapshot(self) -> tuple[np.ndarray | None, float | None]:
        """Return the latest non-stale frame reference and its timestamp.

        The reference is stable because the capture thread never mutates a
        published ndarray; it atomically replaces the slot with the next array.
        Callers must treat the returned frame as read-only.  Avoiding a 24.9 MB
        copy on every request is important for the 3840x2160 hot path.
        """
        now = time.monotonic()
        timed_out = False
        with self.lock:
            stale = (
                self.last_frame_time > 0
                and now - self.last_frame_time > self.frame_timeout
            )
            if stale:
                timed_out = not self._timeout_reported
                self._timeout_reported = True
                self.frame = None
                self.is_closed = True
            if self.frame is None or self.is_closed or self.last_frame_time <= 0:
                frame = None
                timestamp = None
            else:
                frame = self.frame
                timestamp = self.last_frame_time

        if timed_out:
            logger.warning(
                "[CaptureCardCapturor] Buffered frame exceeded stream timeout "
                f"of {self.frame_timeout:.2f}s"
            )
        if frame is None:
            return None, None
        return frame, timestamp

    def get_frame(self) -> np.ndarray | None:
        """Return the newest non-stale, read-only frame reference."""
        frame, _ = self.get_frame_snapshot()
        return frame

    def stop(self) -> None:
        """Stop capture, release DirectShow, and invalidate the buffered frame."""
        self._stop_event.set()
        self.is_terminated = True

        capture = getattr(self, "capture", None)
        if capture is not None:
            try:
                # Releasing the backend also unblocks most pending read calls.
                capture.release()
            except Exception as exc:
                logger.warning(
                    f"[CaptureCardCapturor] Failed to release capture: {exc}"
                )

        thread = self._capture_thread
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(self.shutdown_timeout)
            if thread.is_alive():
                logger.warning(
                    "[CaptureCardCapturor] Capture thread did not exit within "
                    f"{self.shutdown_timeout:.2f}s"
                )

        with self.lock:
            self.frame = None
            self.is_closed = True
        logger.info("[CaptureCardCapturor] Terminated")
