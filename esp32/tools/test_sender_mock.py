#!/usr/bin/env python3
"""Local protocol smoke test for esp32_hid_sender.py; no ESP32 is required."""

from __future__ import annotations

import socket
import threading

import esp32_hid_sender as sender


def mock_server(listener: socket.socket, commands: list[str]) -> None:
    connection, _ = listener.accept()
    with connection, connection.makefile("r", encoding="ascii", newline="\n") as stream:
        for raw_line in stream:
            command = raw_line.rstrip("\r\n")
            commands.append(command)
            if command == "PING":
                response = "PONG\n"
            elif command == "STATUS":
                response = "OK WIFI=1 BLE_CONNECTED=1 BLE_READY=1\n"
            elif command == "RELEASE_ALL":
                response = "OK RELEASE_ALL\n"
                connection.sendall(response.encode("ascii"))
                return
            else:
                response = f"OK {command}\n"
            connection.sendall(response.encode("ascii"))


def main() -> int:
    commands: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        thread = threading.Thread(
            target=mock_server, args=(listener, commands), daemon=True
        )
        thread.start()

        with sender.HidClient(
            "127.0.0.1", port=port, timeout=2.0, heartbeat=0
        ) as client:
            assert client.request("PING") == "PONG"
            assert client.request("STATUS") == "OK WIFI=1 BLE_CONNECTED=1 BLE_READY=1"
            assert client.request("DOWN 0x04") == "OK DOWN 0x04"
            assert client.request("UP 0x04") == "OK UP 0x04"

        thread.join(timeout=2.0)

    assert commands == [
        "PING",
        "STATUS",
        "DOWN 0x04",
        "UP 0x04",
        "RELEASE_ALL",
    ], commands
    assert sender.usage_from_text("A") == 0x04
    assert sender.usage_from_text("LEFT") == 0x50
    assert sender.usage_from_text("E1") == 0xE1
    print("sender mock test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
