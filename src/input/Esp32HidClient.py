"""Thread-safe USB serial client for the ESP32 BLE keyboard/mouse bridge."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable

import serial
from serial.tools import list_ports


class Esp32HidTapUncertainError(ConnectionError):
    """A one-shot TAP/click may have executed, so it must not be replayed."""


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

MOUSE_BUTTONS = {
    "left": ("LEFT", 0x01),
    "right": ("RIGHT", 0x02),
    "middle": ("MIDDLE", 0x04),
}
RELATIVE_MOUSE_MAX_DELTA = 32767
ABSOLUTE_MOUSE_MAX_COORDINATE = 32767


def mouse_button_from_text(value: str) -> tuple[str, int]:
    """Return the firmware token and bit mask for a supported mouse button."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("mouse button must be a non-empty string")
    try:
        return MOUSE_BUTTONS[value.strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unsupported mouse button: {value!r}; use left, right, or middle"
        ) from exc


def absolute_mouse_coordinate(value, name: str) -> int:
    """Validate one normalized absolute HID coordinate."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= value <= ABSOLUTE_MOUSE_MAX_COORDINATE:
        raise ValueError(
            f"{name} must be in the range 0..{ABSOLUTE_MOUSE_MAX_COORDINATE}"
        )
    return value


def relative_mouse_delta(value, name: str) -> int:
    """Validate one signed relative HID movement command component."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not -RELATIVE_MOUSE_MAX_DELTA <= value <= RELATIVE_MOUSE_MAX_DELTA:
        raise ValueError(
            f"{name} must be in the range "
            f"-{RELATIVE_MOUSE_MAX_DELTA}..{RELATIVE_MOUSE_MAX_DELTA}"
        )
    return value


def mouse_click_duration(value) -> int:
    """Validate a mouse click duration accepted by the firmware."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("ESP32 HID mouse click duration must be an integer")
    if not 1 <= value <= 1000:
        raise ValueError(
            "ESP32 HID mouse click duration must be in the range 1..1000 ms"
        )
    return value


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


ESPRESSIF_USB_SERIAL_JTAG_VID = 0x303A
ESPRESSIF_USB_SERIAL_JTAG_PID = 0x1001


def resolve_serial_port(
    configured_port: str | None,
    port_lister: Callable[[], Iterable[object]] | None = None,
) -> str:
    """Resolve ``auto`` to the single attached ESP32-S3 USB Serial/JTAG port."""
    candidate = str(configured_port or "auto").strip()
    if candidate and candidate.lower() != "auto":
        return candidate

    devices = list((port_lister or list_ports.comports)())
    matches = [
        device
        for device in devices
        if getattr(device, "vid", None) == ESPRESSIF_USB_SERIAL_JTAG_VID
        and getattr(device, "pid", None) == ESPRESSIF_USB_SERIAL_JTAG_PID
    ]
    if not matches:
        raise ConnectionError(
            "ESP32-S3 USB Serial/JTAG port was not found; connect the board "
            "or set esp32_hid.serial_port explicitly"
        )
    if len(matches) > 1:
        ports = ", ".join(str(getattr(item, "device", item)) for item in matches)
        raise ConnectionError(
            f"multiple ESP32-S3 serial ports found ({ports}); set "
            "esp32_hid.serial_port explicitly"
        )
    return str(getattr(matches[0], "device", matches[0]))


def _open_serial_without_reset(**kwargs):
    """Open pyserial without toggling ESP32 boot/reset control lines."""
    transport = serial.Serial()
    for name, value in kwargs.items():
        setattr(transport, name, value)
    transport.dtr = False
    transport.rts = False
    transport.open()
    return transport


class Esp32HidClient:
    """One persistent, reconnecting USB serial connection to the HID bridge."""

    def __init__(
        self,
        serial_port: str = "auto",
        baudrate: int = 115200,
        connect_timeout: float = 1.0,
        request_timeout: float = 2.0,
        heartbeat_interval: float = 1.0,
        reconnect_interval: float = 0.5,
        state_refresh_interval: float = 1.0,
        serial_factory: Callable[..., object] | None = None,
        port_lister: Callable[[], Iterable[object]] | None = None,
    ) -> None:
        if not str(serial_port or "").strip():
            raise ValueError("ESP32 HID serial port is required")
        if int(baudrate) <= 0:
            raise ValueError("ESP32 HID serial baudrate must be positive")
        if min(connect_timeout, request_timeout) <= 0:
            raise ValueError("ESP32 HID timeouts must be positive")
        if heartbeat_interval < 0 or reconnect_interval < 0:
            raise ValueError("ESP32 HID intervals cannot be negative")
        if state_refresh_interval <= 0:
            raise ValueError("ESP32 HID state refresh interval must be positive")

        self.configured_serial_port = str(serial_port).strip()
        self.serial_port = ""
        self.baudrate = int(baudrate)
        self.connect_timeout = float(connect_timeout)
        self.request_timeout = float(request_timeout)
        self.heartbeat_interval = float(heartbeat_interval)
        self.reconnect_interval = float(reconnect_interval)
        self.state_refresh_interval = float(state_refresh_interval)
        self._serial_factory = serial_factory or _open_serial_without_reset
        self._port_lister = port_lister
        self._serial: object | None = None
        self._pressed: set[int] = set()
        self._state_dirty = False
        # Changes whenever firmware-side held-key continuity is known to have
        # been broken (release, BLE loss, or serial reconnect). Consumers can
        # use this epoch instead of trusting a stale local hold duration.
        self._state_generation = 0
        self._io_lock = threading.RLock()
        self._stop = threading.Event()
        self._closed = False
        self._last_io = time.monotonic()
        self._last_status = 0.0
        self._last_state_write = 0.0
        self._last_connect_attempt = 0.0
        self._supports_absolute_mouse = False
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
        serial_port = os.environ.get(
            "ESP32_HID_SERIAL_PORT", section.get("serial_port", "auto")
        )
        baudrate = int(
            os.environ.get(
                "ESP32_HID_SERIAL_BAUDRATE", section.get("baudrate", 115200)
            )
        )
        return cls(
            serial_port=serial_port,
            baudrate=baudrate,
            connect_timeout=float(section.get("connect_timeout", 1.0)),
            request_timeout=float(section.get("request_timeout", 2.0)),
            heartbeat_interval=float(section.get("heartbeat_interval", 1.0)),
            reconnect_interval=float(section.get("reconnect_interval", 0.5)),
            state_refresh_interval=float(section.get("state_refresh_interval", 1.0)),
        )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def endpoint(self) -> str:
        port = self.serial_port or self.configured_serial_port
        return f"{port}@{self.baudrate}"

    @property
    def state_generation(self) -> int:
        with self._io_lock:
            return self._state_generation

    @property
    def state_continuity_token(self) -> int | None:
        """Return the epoch only while firmware held state is trustworthy."""
        with self._io_lock:
            if self._state_dirty:
                return None
            return self._state_generation

    @property
    def supports_absolute_mouse(self) -> bool:
        with self._io_lock:
            return self._supports_absolute_mouse

    def _make_serial(self):
        selected_port = resolve_serial_port(
            self.configured_serial_port, self._port_lister
        )
        transport = self._serial_factory(
            port=selected_port,
            baudrate=self.baudrate,
            timeout=self.request_timeout,
            write_timeout=self.request_timeout,
        )
        self.serial_port = selected_port
        if hasattr(transport, "reset_input_buffer"):
            transport.reset_input_buffer()
        if hasattr(transport, "reset_output_buffer"):
            transport.reset_output_buffer()
        return transport

    @staticmethod
    def _transport_is_open(transport) -> bool:
        return transport is not None and bool(getattr(transport, "is_open", True))

    @staticmethod
    def _close_serial(transport) -> None:
        if transport is None:
            return
        transport.close()

    def _mark_state_discontinuous_locked(self, *, dirty: bool) -> None:
        self._pressed.clear()
        self._state_dirty = dirty
        self._last_state_write = 0.0
        self._state_generation += 1

    def _disconnect_locked(self) -> None:
        transport, self._serial = self._serial, None
        self._mark_state_discontinuous_locked(dirty=True)
        self._last_status = 0.0
        self._supports_absolute_mouse = False
        self._close_serial(transport)

    @staticmethod
    def _receive_line_from_locked(transport) -> str:
        data = transport.read_until(expected=b"\n", size=4097)
        if not data:
            raise TimeoutError("timed out waiting for ESP32 serial response")
        if len(data) > 4096:
            raise RuntimeError("ESP32 response is too long")
        if b"\n" not in data:
            raise TimeoutError("timed out waiting for a complete ESP32 serial response")
        return data.rstrip(b"\r\n").decode("ascii", errors="replace")

    def _raw_request_on_locked(self, transport, command: str) -> str:
        payload = command.encode("ascii") + b"\n"
        written = transport.write(payload)
        if written != len(payload):
            raise ConnectionError("ESP32 serial write was incomplete")
        if hasattr(transport, "flush"):
            transport.flush()
        response = self._receive_line_from_locked(transport)
        self._last_io = time.monotonic()
        if command == "STATUS":
            self._last_status = self._last_io
            self._supports_absolute_mouse = "MOUSE_ABS=1" in response
        if response.startswith("ERR "):
            if response in {"ERR BLE_NOT_READY", "ERR HID_SEND_FAILED"}:
                # Both responses mean the firmware has cleared (or cannot
                # guarantee) its keyboard report. Invalidate client-side
                # dedupe immediately so the next control frame reasserts it.
                self._mark_state_discontinuous_locked(dirty=True)
            raise RuntimeError(response)
        return response

    def _connect_locked(self) -> None:
        if self._closed:
            raise RuntimeError("ESP32 HID client is closed")
        if self._transport_is_open(self._serial):
            return
        if self._serial is not None:
            self._disconnect_locked()

        elapsed = time.monotonic() - self._last_connect_attempt
        if self._last_connect_attempt and elapsed < self.reconnect_interval:
            time.sleep(self.reconnect_interval - elapsed)
        self._last_connect_attempt = time.monotonic()

        transport = self._make_serial()
        try:
            status = self._raw_request_on_locked(transport, "STATUS")
            if "SERIAL=1" not in status or "BLE_READY=1" not in status:
                raise ConnectionError(f"ESP32 BLE HID is not ready: {status}")
            self._raw_request_on_locked(transport, "RELEASE_ALL")
        except BaseException:
            self._close_serial(transport)
            raise

        # Every connection handshake deliberately sends RELEASE_ALL, so any
        # held-state duration from the prior transport is no longer valid.
        self._mark_state_discontinuous_locked(dirty=False)
        self._serial = transport

    def _request_locked(self, command: str, retry_safe: bool) -> str:
        attempts = 2 if retry_safe else 1
        last_error: BaseException | None = None
        request_attempted = False
        for attempt in range(attempts):
            try:
                self._connect_locked()
                assert self._serial is not None
                request_attempted = True
                return self._raw_request_on_locked(self._serial, command)
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

    def _release_all_after_uncertain_action_locked(self) -> None:
        """Best-effort idempotent cleanup without replaying a one-shot action."""
        try:
            response = self._request_locked("RELEASE_ALL", retry_safe=True)
            if response != "OK RELEASE_ALL":
                raise RuntimeError(
                    f"unexpected RELEASE_ALL response: {response!r}"
                )
            self._mark_state_discontinuous_locked(dirty=False)
        except (OSError, RuntimeError, ConnectionError):
            # Force the next heartbeat/request through the connection handshake,
            # which itself requires a successful RELEASE_ALL before proceeding.
            self._disconnect_locked()

    def _mouse_one_shot_locked(self, command: str, expected: str) -> str:
        """Send one non-replayable mouse action and verify its exact ACK."""
        action = command.split(maxsplit=1)[0]
        try:
            response = self._request_locked(command, retry_safe=False)
            if response != expected:
                raise Esp32HidTapUncertainError(
                    f"unexpected {action} response: {response!r}"
                )
            return response
        except Esp32HidTapUncertainError:
            self._release_all_after_uncertain_action_locked()
            raise
        except RuntimeError as exc:
            if str(exc) in {"ERR BLE_NOT_READY", "ERR HID_SEND_FAILED"}:
                uncertain = Esp32HidTapUncertainError(str(exc))
            elif not str(exc).startswith("ERR "):
                self._disconnect_locked()
                uncertain = Esp32HidTapUncertainError(str(exc))
            else:
                # A well-formed firmware ERR response confirms rejection, so
                # the one-shot action did not execute and is not uncertain.
                raise
            self._release_all_after_uncertain_action_locked()
            raise uncertain from exc

    def mouse_move(self, dx: int, dy: int, wheel: int = 0) -> str:
        """Move the mouse relatively exactly once without replay on ACK loss."""
        dx = relative_mouse_delta(dx, "relative mouse dx")
        dy = relative_mouse_delta(dy, "relative mouse dy")
        wheel = relative_mouse_delta(wheel, "relative mouse wheel")
        command = f"MOUSE_MOVE {dx} {dy} {wheel}"
        expected = f"OK MOUSE_MOVE {dx} {dy} {wheel}"
        with self._io_lock:
            return self._mouse_one_shot_locked(command, expected)

    def mouse_click(
        self,
        button: str = "left",
        duration_ms: int = 50,
    ) -> str:
        """Click at the current pointer position exactly once."""
        button_token, button_mask = mouse_button_from_text(button)
        duration_ms = mouse_click_duration(duration_ms)
        command = f"MOUSE_CLICK {button_token} {duration_ms}"
        expected = f"OK MOUSE_CLICK 0x{button_mask:02X} {duration_ms}ms"
        with self._io_lock:
            return self._mouse_one_shot_locked(command, expected)

    def mouse_click_at(
        self,
        x: int,
        y: int,
        button: str = "left",
        duration_ms: int = 50,
    ) -> str:
        """Move to a normalized absolute point and click exactly once.

        Like ``tap``, the command is deliberately never replayed after an I/O
        failure: the HID host may have received it even when its serial ACK was
        lost.
        """
        x = absolute_mouse_coordinate(x, "absolute mouse x")
        y = absolute_mouse_coordinate(y, "absolute mouse y")
        button_token, button_mask = mouse_button_from_text(button)
        duration_ms = mouse_click_duration(duration_ms)
        command = f"MOUSE_CLICK_AT {x} {y} {button_token} {duration_ms}"
        expected = (
            f"OK MOUSE_CLICK_AT {x} {y} 0x{button_mask:02X} {duration_ms}ms"
        )

        with self._io_lock:
            self._connect_locked()
            if not self._supports_absolute_mouse:
                raise RuntimeError(
                    "ESP32 firmware does not advertise MOUSE_ABS=1"
                )
            return self._mouse_one_shot_locked(command, expected)

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
            self._mark_state_discontinuous_locked(dirty=False)
            return response

    def status(self) -> str:
        with self._io_lock:
            response = self._request_locked("STATUS", retry_safe=True)
            if "BLE_READY=1" not in response:
                # BLE can drop while the USB serial session remains open. The
                # firmware releases every key on that edge, so invalidate the
                # local dedupe cache and let the controller reassert movement
                # after the keyboard reconnects.
                self._mark_state_discontinuous_locked(dirty=True)
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
                        self._mark_state_discontinuous_locked(dirty=True)
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
            if self._transport_is_open(self._serial):
                try:
                    self._raw_request_on_locked(self._serial, "RELEASE_ALL")
                except (OSError, RuntimeError, ConnectionError):
                    pass
            self._closed = True
            self._disconnect_locked()

    def __enter__(self) -> "Esp32HidClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
