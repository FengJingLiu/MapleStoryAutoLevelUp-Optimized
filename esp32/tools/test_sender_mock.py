#!/usr/bin/env python3
"""Local serial protocol smoke test; no ESP32 is required."""

from __future__ import annotations

from collections import deque
from unittest.mock import patch

import esp32_hid_sender as sender


class FakeSerial:
    def __init__(self):
        self.port = None
        self.baudrate = None
        self.timeout = None
        self.write_timeout = None
        self.dtr = None
        self.rts = None
        self.is_open = False
        self.commands: list[str] = []
        self.responses: deque[bytes] = deque()

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def reset_input_buffer(self):
        self.responses.clear()

    def reset_output_buffer(self):
        pass

    def flush(self):
        pass

    def write(self, payload: bytes) -> int:
        command = payload.decode("ascii").rstrip("\r\n")
        self.commands.append(command)
        if command == "PING":
            response = "PONG\n"
        elif command == "STATUS":
            response = (
                "OK SERIAL=1 BLE_CONNECTED=1 BLE_READY=1 "
                "MOUSE=1 MOUSE_ABS=1\n"
            )
        elif command == "RELEASE_ALL":
            response = "OK RELEASE_ALL\n"
        elif command.startswith("MOUSE_CLICK_AT "):
            _, x, y, button, duration = command.split()
            button_mask = {
                "LEFT": "0x01",
                "RIGHT": "0x02",
                "MIDDLE": "0x04",
            }[button]
            response = (
                f"OK MOUSE_CLICK_AT {x} {y} {button_mask} {duration}ms\n"
            )
        else:
            response = f"OK {command}\n"
        self.responses.append(response.encode("ascii"))
        return len(payload)

    def read_until(self, expected=b"\n", size=None) -> bytes:
        return self.responses.popleft() if self.responses else b""


def main() -> int:
    fake = FakeSerial()
    with (
        patch.object(sender.serial, "Serial", return_value=fake),
        patch.object(sender, "resolve_serial_port", return_value="COM6"),
    ):
        with sender.HidClient(
            serial_port="auto", timeout=2.0, heartbeat=0
        ) as client:
            assert client.request("PING") == "PONG"
            assert client.request("DOWN 0x04") == "OK DOWN 0x04"
            assert client.request("UP 0x04") == "OK UP 0x04"
            assert client.request("MOUSE_MOVE 120 -40 0") == (
                "OK MOUSE_MOVE 120 -40 0"
            )
            assert client.request("MOUSE_CLICK LEFT 60") == (
                "OK MOUSE_CLICK LEFT 60"
            )
            assert client.request("MOUSE_CLICK_AT 16384 16384 LEFT 60") == (
                "OK MOUSE_CLICK_AT 16384 16384 0x01 60ms"
            )

    assert fake.commands == [
        "STATUS",
        "RELEASE_ALL",
        "PING",
        "DOWN 0x04",
        "UP 0x04",
        "MOUSE_MOVE 120 -40 0",
        "MOUSE_CLICK LEFT 60",
        "MOUSE_CLICK_AT 16384 16384 LEFT 60",
        "RELEASE_ALL",
    ], fake.commands
    assert fake.dtr is False
    assert fake.rts is False
    assert sender.usage_from_text("A") == 0x04
    assert sender.usage_from_text("LEFT") == 0x50
    assert sender.usage_from_text("E1") == 0xE1
    assert sender.mouse_button_token("left") == "LEFT"
    assert sender.mouse_delta(-32767) == -32767
    assert sender.absolute_mouse_coordinate(32767) == 32767
    print("sender serial mock test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
