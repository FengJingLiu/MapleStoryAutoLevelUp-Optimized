"""Asynchronously forward local keyboard state to the ESP32 HID bridge."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Collection

from src.input.Esp32HidClient import Esp32HidClient, usage_from_text
from src.utils.logger import logger


class Esp32KeyForwarder:
    """Mirror a selected set of local keys to the remote BLE HID keyboard.

    Keyboard hooks must return immediately.  Each press/release therefore
    queues a complete keyboard state for a worker thread instead of performing
    serial I/O inside pynput's callback. Keeping every state transition also
    preserves quick taps (DOWN followed immediately by UP).
    """

    def __init__(self, cfg: dict, allowed_keys: Collection[str], client=None):
        normalized_keys = {
            str(key).strip().lower() for key in allowed_keys if str(key).strip()
        }
        for key in normalized_keys:
            usage_from_text(key)

        self.allowed_keys = frozenset(normalized_keys)
        self.client = client or Esp32HidClient.from_config(cfg)
        self._desired_keys: set[str] = set()
        self._state_lock = threading.Lock()
        self._states: queue.Queue[tuple[str, ...] | None] = queue.Queue()
        self._closed = False
        self._last_error_time = 0.0
        self._thread = threading.Thread(
            target=self._run,
            name="route-recorder-esp32-forwarder",
            daemon=True,
        )
        self._thread.start()

        logger.info(
            "[RouteRecorder] ESP32 key forwarding ready over USB serial at "
            f"{self.client.endpoint}; "
            f"keys={sorted(self.allowed_keys)}"
        )

    def handle_key_event(self, key: str, pressed: bool) -> bool:
        """Queue one local key transition without blocking the keyboard hook."""
        key = str(key).strip().lower()
        if self._closed or key not in self.allowed_keys:
            return False

        with self._state_lock:
            changed = False
            if pressed and key not in self._desired_keys:
                self._desired_keys.add(key)
                changed = True
            elif not pressed and key in self._desired_keys:
                self._desired_keys.remove(key)
                changed = True
            if not changed:
                return False
            state = tuple(sorted(self._desired_keys))

        self._states.put(state)
        return True

    def _current_state(self) -> tuple[str, ...]:
        with self._state_lock:
            return tuple(sorted(self._desired_keys))

    def _send_state(self, state: tuple[str, ...]) -> None:
        try:
            self.client.set_state(state)
        except (OSError, RuntimeError, ValueError, ConnectionError) as exc:
            now = time.monotonic()
            if now - self._last_error_time >= 2.0:
                logger.error(f"[RouteRecorder] ESP32 key forwarding failed: {exc}")
                self._last_error_time = now

    def _run(self) -> None:
        while True:
            try:
                state = self._states.get(timeout=0.2)
            except queue.Empty:
                # Reassert a held key after a BLE reconnect. Esp32HidClient
                # deduplicates this call until its configured refresh is due.
                state = self._current_state()

            if state is None:
                return
            self._send_state(state)

    def close(self) -> None:
        """Stop forwarding, release all remote keys, and close the serial client."""
        if self._closed:
            return
        self._closed = True
        with self._state_lock:
            self._desired_keys.clear()

        # Discard stale transitions. The client's close() issues RELEASE_ALL,
        # so no queued DOWN may be sent after shutdown starts.
        while True:
            try:
                self._states.get_nowait()
            except queue.Empty:
                break
        self._states.put(None)
        if self._thread is not threading.current_thread():
            timeout = self.client.connect_timeout + self.client.request_timeout + 1.0
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    "[RouteRecorder] ESP32 forwarding thread did not stop in time"
                )
        self.client.close()

