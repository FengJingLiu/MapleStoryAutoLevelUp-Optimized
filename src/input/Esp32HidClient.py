"""Thread-safe client for the ESP32 Wi-Fi to BLE HID keyboard bridge."""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Callable, Iterable


class Esp32HidTapUncertainError(ConnectionError):
    """A TAP may have reached the HID host, so it must not be replayed."""


KEY_USAGES: dict[str, int] = {}
KEY_USAGES.update({chr(ord("a") + i): 0x04 + i for i in range(26)})
KEY_USAGES.update({str(i): 0x1D + i for i in range(1, 10)})
KEY_USAGES["0"] = 0x27
KEY_USAGES.update(
    {
        "enter": 0x28,
        "return": 0x28,
        "esc": 0x29,
        "escape": 0x29,
        "backspace": 0x2A,
        "tab": 0x2B,
        "space": 0x2C,
        "minus": 0x2D,
        "equal": 0x2E,
        "equals": 0x2E,
        "leftbrace": 0x2F,
        "leftbracket": 0x2F,
        "rightbrace": 0x30,
        "rightbracket": 0x30,
        "backslash": 0x31,
        "semicolon": 0x33,
        "apostrophe": 0x34,
        "quote": 0x34,
        "grave": 0x35,
        "backtick": 0x35,
        "comma": 0x36,
        "dot": 0x37,
        "period": 0x37,
        "slash": 0x38,
        "capslock": 0x39,
        "printscreen": 0x46,
        "prtsc": 0x46,
        "scrolllock": 0x47,
        "pause": 0x48,
        "insert": 0x49,
        "ins": 0x49,
        "home": 0x4A,
        "pageup": 0x4B,
        "pgup": 0x4B,
        "delete": 0x4C,
        "del": 0x4C,
        "end": 0x4D,
        "pagedown": 0x4E,
        "pgdown": 0x4E,
        "pgdn": 0x4E,
        "right": 0x4F,
        "left": 0x50,
        "down": 0x51,
        "up": 0x52,
        "numlock": 0x53,
        "ctrl": 0xE0,
        "control": 0xE0,
        "lctrl": 0xE0,
        "leftctrl": 0xE0,
        "leftcontrol": 0xE0,
        "shift": 0xE1,
        "lshift": 0xE1,
        "leftshift": 0xE1,
        "alt": 0xE2,
        "option": 0xE2,
        "lalt": 0xE2,
        "leftalt": 0xE2,
        "win": 0xE3,
        "windows": 0xE3,
        "gui": 0xE3,
        "meta": 0xE3,
        "cmd": 0xE3,
        "command": 0xE3,
        "lgui": 0xE3,
        "leftgui": 0xE3,
        "rctrl": 0xE4,
        "rightctrl": 0xE4,
        "rightcontrol": 0xE4,
        "rshift": 0xE5,
        "rightshift": 0xE5,
        "ralt": 0xE6,
        "rightalt": 0xE6,
        "rgui": 0xE7,
        "rightgui": 0xE7,
    }
)
KEY_USAGES.update({f"f{i}": 0x39 + i for i in range(1, 13)})
KEY_USAGES.update({f"f{i}": 0x68 + (i - 13) for i in range(13, 25)})

SYMBOL_USAGES = {
    "-": 0x2D,
    "=": 0x2E,
    "[": 0x2F,
    "]": 0x30,
    "\\": 0x31,
    ";": 0x33,
    "'": 0x34,
    "`": 0x35,
    ",": 0x36,
    ".": 0x37,
    "/": 0x38,
}


def usage_from_text(value: str) -> int:
    """Translate a configured key name into a USB HID keyboard usage."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("key must be a non-empty string")

    candidate = value.strip().lower()
    if candidate in SYMBOL_USAGES:
        return SYMBOL_USAGES[candidate]

    normalized = candidate.replace("-", "").replace("_", "").replace(" ", "")
    if normalized in KEY_USAGES:
        return KEY_USAGES[normalized]

    hex_candidate = normalized[2:] if normalized.startswith("0x") else normalized
    try:
        usage = int(hex_candidate, 16)
    except ValueError as exc:
        raise ValueError(f"unsupported keyboard key: {value!r}") from exc
    if not ((0x04 <= usage <= 0x73) or (0xE0 <= usage <= 0xE7)):
        raise ValueError(f"unsupported HID usage: 0x{usage:02X}")
    return usage


def usage_token(value: str | int) -> str:
    usage = usage_from_text(value) if isinstance(value, str) else value
    if not isinstance(usage, int) or not (
        (0x04 <= usage <= 0x73) or (0xE0 <= usage <= 0xE7)
    ):
        raise ValueError(f"unsupported HID usage: {usage!r}")
    return f"0x{usage:02X}"


class Esp32HidClient:
    """One persistent, reconnecting connection to the ESP32 HID bridge."""

    def __init__(
        self,
        host: str,
        port: int = 3333,
        connect_timeout: float = 1.0,
        request_timeout: float = 2.0,
        heartbeat_interval: float = 1.0,
        reconnect_interval: float = 0.5,
        state_refresh_interval: float = 1.0,
        socket_factory: Callable[..., socket.socket] | None = None,
    ) -> None:
        if not host:
            raise ValueError("ESP32 HID host is required")
        if not 1 <= int(port) <= 65535:
            raise ValueError("ESP32 HID port must be in the range 1..65535")
        if min(connect_timeout, request_timeout) <= 0:
            raise ValueError("ESP32 HID timeouts must be positive")
        if heartbeat_interval < 0 or reconnect_interval < 0:
            raise ValueError("ESP32 HID intervals cannot be negative")
        if state_refresh_interval <= 0:
            raise ValueError("ESP32 HID state refresh interval must be positive")

        self.host = host
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self.request_timeout = float(request_timeout)
        self.heartbeat_interval = float(heartbeat_interval)
        self.reconnect_interval = float(reconnect_interval)
        self.state_refresh_interval = float(state_refresh_interval)
        self._socket_factory = socket_factory or socket.create_connection
        self._socket: socket.socket | None = None
        self._receive_buffer = bytearray()
        self._pressed: set[int] = set()
        self._state_dirty = False
        self._io_lock = threading.RLock()
        self._stop = threading.Event()
        self._closed = False
        self._last_io = time.monotonic()
        self._last_status = 0.0
        self._last_state_write = 0.0
        self._last_connect_attempt = 0.0
        self._heartbeat_thread: threading.Thread | None = None

        with self._io_lock:
            self._connect_locked()

        if self.heartbeat_interval > 0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="esp32-hid-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    @classmethod
    def from_config(cls, cfg: dict) -> "Esp32HidClient":
        section = cfg.get("esp32_hid", {})
        host = os.environ.get("ESP32_HID_HOST", section.get("host", ""))
        port = int(os.environ.get("ESP32_HID_PORT", section.get("port", 3333)))
        return cls(
            host=host,
            port=port,
            connect_timeout=float(section.get("connect_timeout", 1.0)),
            request_timeout=float(section.get("request_timeout", 2.0)),
            heartbeat_interval=float(section.get("heartbeat_interval", 1.0)),
            reconnect_interval=float(section.get("reconnect_interval", 0.5)),
            state_refresh_interval=float(section.get("state_refresh_interval", 1.0)),
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def _make_socket(self) -> socket.socket:
        sock = self._socket_factory(
            (self.host, self.port), timeout=self.connect_timeout
        )
        sock.settimeout(self.request_timeout)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except (AttributeError, OSError):
            pass
        return sock

    @staticmethod
    def _close_socket(sock: socket.socket | None) -> None:
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def _disconnect_locked(self) -> None:
        sock, self._socket = self._socket, None
        self._receive_buffer.clear()
        self._pressed.clear()
        self._state_dirty = True
        self._last_status = 0.0
        self._last_state_write = 0.0
        self._close_socket(sock)

    def _receive_line_from_locked(self, sock: socket.socket) -> str:
        while b"\n" not in self._receive_buffer:
            data = sock.recv(512)
            if not data:
                raise ConnectionError("ESP32 closed the TCP connection")
            self._receive_buffer.extend(data)
            if len(self._receive_buffer) > 4096:
                raise RuntimeError("ESP32 response is too long")
        raw, _, remainder = self._receive_buffer.partition(b"\n")
        self._receive_buffer = bytearray(remainder)
        return raw.rstrip(b"\r").decode("ascii", errors="replace")

    def _raw_request_on_locked(self, sock: socket.socket, command: str) -> str:
        sock.sendall(command.encode("ascii") + b"\n")
        response = self._receive_line_from_locked(sock)
        self._last_io = time.monotonic()
        if command == "STATUS":
            self._last_status = self._last_io
        if response.startswith("ERR "):
            if response in {"ERR BLE_NOT_READY", "ERR HID_SEND_FAILED"}:
                # Both responses mean the firmware has cleared (or cannot
                # guarantee) its keyboard report. Invalidate client-side
                # dedupe immediately so the next control frame reasserts it.
                self._pressed.clear()
                self._state_dirty = True
                self._last_state_write = 0.0
            raise RuntimeError(response)
        return response

    def _connect_locked(self) -> None:
        if self._closed:
            raise RuntimeError("ESP32 HID client is closed")
        if self._socket is not None:
            return

        elapsed = time.monotonic() - self._last_connect_attempt
        if self._last_connect_attempt and elapsed < self.reconnect_interval:
            time.sleep(self.reconnect_interval - elapsed)
        self._last_connect_attempt = time.monotonic()

        sock = self._make_socket()
        self._receive_buffer.clear()
        try:
            status = self._raw_request_on_locked(sock, "STATUS")
            if "WIFI=1" not in status or "BLE_READY=1" not in status:
                raise ConnectionError(f"ESP32 BLE HID is not ready: {status}")
            self._raw_request_on_locked(sock, "RELEASE_ALL")
        except BaseException:
            self._receive_buffer.clear()
            self._close_socket(sock)
            raise

        self._pressed.clear()
        self._state_dirty = False
        self._last_state_write = 0.0
        self._socket = sock

    def _request_locked(self, command: str, retry_safe: bool) -> str:
        attempts = 2 if retry_safe else 1
        last_error: BaseException | None = None
        request_attempted = False
        for attempt in range(attempts):
            try:
                self._connect_locked()
                assert self._socket is not None
                request_attempted = True
                return self._raw_request_on_locked(self._socket, command)
            except (OSError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                self._disconnect_locked()
                if attempt + 1 >= attempts:
                    break
        message = f"ESP32 HID request failed for {command!r}: {last_error}"
        if not retry_safe and request_attempted:
            raise Esp32HidTapUncertainError(message) from last_error
        raise ConnectionError(message) from last_error

    def key_down(self, key: str) -> bool:
        usage = usage_from_text(key)
        with self._io_lock:
            if usage in self._pressed:
                return False
            self._request_locked(f"DOWN {usage_token(usage)}", retry_safe=True)
            self._pressed.add(usage)
            return True

    def key_up(self, key: str) -> bool:
        usage = usage_from_text(key)
        with self._io_lock:
            if usage not in self._pressed:
                return False
            self._request_locked(f"UP {usage_token(usage)}", retry_safe=True)
            self._pressed.discard(usage)
            return True

    def tap(self, key: str, duration_ms: int = 50) -> str:
        usage = usage_from_text(key)
        if not 1 <= int(duration_ms) <= 1000:
            raise ValueError("ESP32 HID tap duration must be in the range 1..1000 ms")
        with self._io_lock:
            if usage in self._pressed:
                # Firmware deliberately treats TAP on an already-held usage as
                # a no-op. Do not report a false successful key edge.
                raise RuntimeError(
                    f"cannot TAP {key!r} while that key is already held"
                )
            try:
                response = self._request_locked(
                    f"TAP {usage_token(usage)} {int(duration_ms)}", retry_safe=False
                )
                expected = f"OK TAP {usage_token(usage)} {int(duration_ms)}ms"
                if response != expected:
                    raise Esp32HidTapUncertainError(
                        f"unexpected TAP response: {response!r}"
                    )
                return response
            except RuntimeError as exc:
                if str(exc) in {"ERR BLE_NOT_READY", "ERR HID_SEND_FAILED"}:
                    raise Esp32HidTapUncertainError(str(exc)) from exc
                if not str(exc).startswith("ERR "):
                    # Framing/decoding failures happen after TAP was sent; its
                    # execution result is unknown and must not be replayed.
                    self._disconnect_locked()
                    raise Esp32HidTapUncertainError(str(exc)) from exc
                raise

    def set_state(self, keys: Iterable[str]) -> bool:
        usages = {usage_from_text(key) for key in keys if key}
        non_modifiers = [usage for usage in usages if usage < 0xE0]
        if len(non_modifiers) > 6:
            raise ValueError("ESP32 HID supports at most six simultaneous non-modifier keys")
        with self._io_lock:
            refresh_due = (
                bool(usages)
                and time.monotonic() - self._last_state_write
                >= self.state_refresh_interval
            )
            if usages == self._pressed and not refresh_due and not self._state_dirty:
                return False
            tokens = " ".join(usage_token(usage) for usage in sorted(usages))
            command = f"STATE {tokens}".rstrip()
            self._request_locked(command, retry_safe=True)
            self._pressed = usages
            self._state_dirty = False
            self._last_state_write = time.monotonic()
            return True

    def release_all(self) -> str:
        with self._io_lock:
            response = self._request_locked("RELEASE_ALL", retry_safe=True)
            self._pressed.clear()
            self._state_dirty = False
            self._last_state_write = 0.0
            return response

    def status(self) -> str:
        with self._io_lock:
            response = self._request_locked("STATUS", retry_safe=True)
            if "BLE_READY=1" not in response:
                # BLE can drop while the Wi-Fi TCP session remains alive. The
                # firmware releases every key on that edge, so invalidate the
                # local dedupe cache and let the controller reassert movement
                # after the keyboard reconnects.
                self._pressed.clear()
                self._state_dirty = True
                self._last_state_write = 0.0
            return response

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval):
            with self._io_lock:
                if self._closed:
                    return
                # Health checks use their own clock. Ordinary input traffic
                # must not suppress STATUS forever while BLE is disconnected.
                if time.monotonic() - self._last_status < self.heartbeat_interval:
                    continue
                try:
                    response = self._request_locked("STATUS", retry_safe=True)
                    if "BLE_READY=1" not in response:
                        self._pressed.clear()
                        self._state_dirty = True
                        self._last_state_write = 0.0
                except (OSError, RuntimeError, ConnectionError):
                    # The next heartbeat or input request will reconnect. A disconnect
                    # clears both the firmware and local pressed-key state.
                    pass

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self.connect_timeout + self.request_timeout + 1.0)

        with self._io_lock:
            if self._closed:
                return
            if self._socket is not None:
                try:
                    self._raw_request_on_locked(self._socket, "RELEASE_ALL")
                except (OSError, RuntimeError, ConnectionError):
                    pass
            self._closed = True
            self._disconnect_locked()

    def __enter__(self) -> "Esp32HidClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
