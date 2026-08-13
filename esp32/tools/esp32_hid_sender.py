#!/usr/bin/env python3
"""Send keyboard commands to the ESP32-S3 USB serial + BLE HID bridge."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import threading
import time
from typing import Iterable

import serial
from serial.tools import list_ports


KEYS: dict[str, int] = {}
KEYS.update({chr(ord("A") + i): 0x04 + i for i in range(26)})
KEYS.update({str(i): 0x1D + i for i in range(1, 10)})
KEYS["0"] = 0x27
KEYS.update(
    {
        "ENTER": 0x28,
        "RETURN": 0x28,
        "ESC": 0x29,
        "ESCAPE": 0x29,
        "BACKSPACE": 0x2A,
        "TAB": 0x2B,
        "SPACE": 0x2C,
        "MINUS": 0x2D,
        "EQUAL": 0x2E,
        "LEFTBRACE": 0x2F,
        "RIGHTBRACE": 0x30,
        "BACKSLASH": 0x31,
        "SEMICOLON": 0x33,
        "APOSTROPHE": 0x34,
        "GRAVE": 0x35,
        "COMMA": 0x36,
        "DOT": 0x37,
        "PERIOD": 0x37,
        "SLASH": 0x38,
        "CAPSLOCK": 0x39,
        "PRINTSCREEN": 0x46,
        "SCROLLLOCK": 0x47,
        "PAUSE": 0x48,
        "INSERT": 0x49,
        "HOME": 0x4A,
        "PAGEUP": 0x4B,
        "DELETE": 0x4C,
        "END": 0x4D,
        "PAGEDOWN": 0x4E,
        "RIGHT": 0x4F,
        "LEFT": 0x50,
        "DOWN": 0x51,
        "UP": 0x52,
        "NUMLOCK": 0x53,
        "LCTRL": 0xE0,
        "LEFTCTRL": 0xE0,
        "LSHIFT": 0xE1,
        "LEFTSHIFT": 0xE1,
        "LALT": 0xE2,
        "LEFTALT": 0xE2,
        "LGUI": 0xE3,
        "WIN": 0xE3,
        "RCTRL": 0xE4,
        "RIGHTCTRL": 0xE4,
        "RSHIFT": 0xE5,
        "RIGHTSHIFT": 0xE5,
        "RALT": 0xE6,
        "RIGHTALT": 0xE6,
        "RGUI": 0xE7,
    }
)
KEYS.update({f"F{i}": 0x39 + i for i in range(1, 13)})


def usage_from_text(value: str) -> int:
    """Convert a friendly key name or a hexadecimal HID usage into a byte."""
    normalized = value.strip().upper().replace("-", "").replace("_", "")
    if normalized in KEYS:
        return KEYS[normalized]

    candidate = normalized[2:] if normalized.startswith("0X") else normalized
    try:
        usage = int(candidate, 16)
    except ValueError as exc:
        raise ValueError(f"unknown key: {value!r}") from exc
    if not ((0x04 <= usage <= 0x73) or (0xE0 <= usage <= 0xE7)):
        raise ValueError(
            f"unsupported HID usage 0x{usage:02X}; expected 0x04-0x73 or 0xE0-0xE7"
        )
    return usage


def usage_token(value: str) -> str:
    return f"0x{usage_from_text(value):02X}"


class HidClient:
    """One request / one response serial client with a lease heartbeat."""

    def __init__(
        self,
        serial_port: str = "auto",
        baudrate: int = 115200,
        timeout: float = 4.0,
        heartbeat: float = 1.0,
    ) -> None:
        self.serial_port = resolve_serial_port(serial_port)
        self._serial = serial.Serial()
        self._serial.port = self.serial_port
        self._serial.baudrate = baudrate
        self._serial.timeout = timeout
        self._serial.write_timeout = timeout
        self._serial.dtr = False
        self._serial.rts = False
        self._serial.open()
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        self._io_lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._heartbeat_error: BaseException | None = None
        self._heartbeat_interval = heartbeat
        self._heartbeat_thread: threading.Thread | None = None
        status = self.request("STATUS")
        if "SERIAL=1" not in status or "BLE_READY=1" not in status:
            self.close()
            raise ConnectionError(f"ESP32 BLE HID is not ready: {status}")
        self.request("RELEASE_ALL")

        if heartbeat > 0:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="esp32-hid-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread.start()

    def _receive_line_locked(self) -> str:
        data = self._serial.read_until(expected=b"\n", size=4097)
        if not data:
            raise TimeoutError("timed out waiting for ESP32 serial response")
        if len(data) > 4096:
            raise RuntimeError("response from ESP32 is too long")
        if b"\n" not in data:
            raise TimeoutError("timed out waiting for a complete ESP32 serial response")
        return data.rstrip(b"\r\n").decode("ascii", errors="replace")

    def _raw_request_locked(self, command: str) -> str:
        payload = command.encode("ascii") + b"\n"
        if self._serial.write(payload) != len(payload):
            raise ConnectionError("ESP32 serial write was incomplete")
        self._serial.flush()
        return self._receive_line_locked()

    def request(self, command: str) -> str:
        if self._closed:
            raise RuntimeError("client is closed")
        if self._heartbeat_error is not None:
            raise ConnectionError(
                f"heartbeat failed: {self._heartbeat_error}"
            ) from self._heartbeat_error

        with self._io_lock:
            response = self._raw_request_locked(command)
        if response.startswith("ERR "):
            raise RuntimeError(response)
        return response

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_interval):
            try:
                with self._io_lock:
                    response = self._raw_request_locked("PING")
                if response != "PONG":
                    raise RuntimeError(f"unexpected heartbeat response: {response}")
            except (OSError, RuntimeError, ConnectionError) as exc:
                self._heartbeat_error = exc
                self._stop.set()
                return

    def close(self) -> None:
        if self._closed:
            return

        self._stop.set()
        if (
            self._heartbeat_thread is not None
            and self._heartbeat_thread is not threading.current_thread()
        ):
            self._heartbeat_thread.join()

        # Keep the request path alive until RELEASE_ALL has been attempted.
        try:
            with self._io_lock:
                self._raw_request_locked("RELEASE_ALL")
        except (OSError, RuntimeError, ConnectionError):
            pass
        finally:
            self._closed = True
            self._serial.close()

    def __enter__(self) -> "HidClient":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def show_keys() -> None:
    canonical = {
        name: usage
        for name, usage in KEYS.items()
        if name
        not in {
            "RETURN",
            "ESCAPE",
            "PERIOD",
            "LEFTCTRL",
            "LEFTSHIFT",
            "LEFTALT",
            "RIGHTCTRL",
            "RIGHTSHIFT",
            "RIGHTALT",
        }
    }
    width = max(len(name) for name in canonical)
    for name, usage in sorted(canonical.items(), key=lambda item: (item[1], item[0])):
        print(f"{name:<{width}}  0x{usage:02X}")


def interactive(client: HidClient) -> None:
    print("Connected. Commands: tap/down/up/state/release/status/ping/help/keys/quit")
    while True:
        try:
            parts = shlex.split(input("hid> "))
        except EOFError:
            print()
            return
        if not parts:
            continue

        command = parts[0].lower()
        try:
            if command in {"quit", "exit"}:
                return
            if command == "keys":
                show_keys()
                continue
            if command == "ping" and len(parts) == 1:
                print(client.request("PING"))
                continue
            if command == "status" and len(parts) == 1:
                print(client.request("STATUS"))
                continue
            if command in {"help", "server-help"} and len(parts) == 1:
                print(client.request("HELP"))
                continue
            if command in {"release", "release-all"} and len(parts) == 1:
                print(client.request("RELEASE_ALL"))
                continue
            if command in {"down", "up"} and len(parts) == 2:
                print(client.request(f"{command.upper()} {usage_token(parts[1])}"))
                continue
            if command == "tap" and len(parts) in {2, 3}:
                duration = int(parts[2]) if len(parts) == 3 else 60
                print(client.request(f"TAP {usage_token(parts[1])} {duration}"))
                continue
            if command == "state":
                tokens = " ".join(usage_token(key) for key in parts[1:])
                print(client.request(f"STATE {tokens}".rstrip()))
                continue
            print("usage error; type help for the server command list")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)


def resolve_serial_port(configured_port: str | None) -> str:
    candidate = str(configured_port or "auto").strip()
    if candidate and candidate.lower() != "auto":
        return candidate
    matches = [
        item
        for item in list_ports.comports()
        if item.vid == 0x303A and item.pid == 0x1001
    ]
    if len(matches) == 1:
        return matches[0].device
    if not matches:
        raise ConnectionError("ESP32-S3 USB Serial/JTAG port was not found")
    ports = ", ".join(item.device for item in matches)
    raise ConnectionError(
        f"multiple ESP32-S3 ports found ({ports}); pass --serial-port explicitly"
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send keyboard commands over USB serial to the ESP32-S3 BLE HID bridge."
    )
    parser.add_argument(
        "--serial-port",
        default=os.environ.get("ESP32_HID_SERIAL_PORT", "auto"),
        help="Serial port or auto; can also be set with ESP32_HID_SERIAL_PORT",
    )
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=1.0,
        help="PING interval in seconds; use 0 to disable (default: 1.0)",
    )

    commands = parser.add_subparsers(dest="command")
    commands.add_parser("ping")
    commands.add_parser("status")
    commands.add_parser("server-help")
    commands.add_parser("release-all")
    commands.add_parser("keys")

    down = commands.add_parser("down")
    down.add_argument("key")
    down.add_argument(
        "--hold",
        type=float,
        help="release automatically after this many seconds; otherwise wait for Ctrl+C",
    )

    up = commands.add_parser("up")
    up.add_argument("key")

    tap = commands.add_parser("tap")
    tap.add_argument("key")
    tap.add_argument("--ms", type=int, default=60)

    state = commands.add_parser("state")
    state.add_argument("keys", nargs="*")
    return parser


def run_serial_command(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not 0 <= args.heartbeat:
        parser.error("--heartbeat must be zero or positive")

    with HidClient(
        serial_port=args.serial_port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        heartbeat=args.heartbeat,
    ) as client:
        if args.command is None:
            interactive(client)
        elif args.command == "ping":
            print(client.request("PING"))
        elif args.command == "status":
            print(client.request("STATUS"))
        elif args.command == "server-help":
            print(client.request("HELP"))
        elif args.command == "release-all":
            print(client.request("RELEASE_ALL"))
        elif args.command == "up":
            print(client.request(f"UP {usage_token(args.key)}"))
        elif args.command == "tap":
            if not 1 <= args.ms <= 1000:
                parser.error("tap --ms must be in the range 1..1000")
            print(client.request(f"TAP {usage_token(args.key)} {args.ms}"))
        elif args.command == "state":
            tokens = " ".join(usage_token(key) for key in args.keys)
            print(client.request(f"STATE {tokens}".rstrip()))
        elif args.command == "down":
            print(client.request(f"DOWN {usage_token(args.key)}"))
            try:
                if args.hold is None:
                    print("Key is down; press Ctrl+C to release it.")
                    while True:
                        time.sleep(1)
                else:
                    if args.hold < 0:
                        parser.error("down --hold must be zero or positive")
                    time.sleep(args.hold)
            finally:
                try:
                    print(client.request(f"UP {usage_token(args.key)}"))
                except (OSError, RuntimeError, ConnectionError):
                    pass


def main(argv: Iterable[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command == "keys":
        show_keys()
        return 0

    try:
        run_serial_command(parser, args)
    except KeyboardInterrupt:
        print("\nInterrupted; RELEASE_ALL attempted.")
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
