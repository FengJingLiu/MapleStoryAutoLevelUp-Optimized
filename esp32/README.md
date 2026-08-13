# ESP32-S3 USB Serial + BLE HID Bridge

This directory implements the remote keyboard path used by the bot:

```text
Computer A --USB Serial/JTAG--> ESP32-S3 --BLE HID--> game computer B
```

Wi-Fi is not started. ESP-IDF logs remain on UART0 while the native ESP32-S3
USB Serial/JTAG port is dedicated to the line-oriented command protocol.

## Build and flash

Use ESP-IDF v6.0.2:

```powershell
cd D:\project\MapleStoryAutoLevelUp-Optimized-serial\esp32\wifi_ble_hid_demo
idf.py set-target esp32s3
idf.py build
idf.py -p COM6 flash
```

Replace `COM6` if Windows assigns another port. Do not run `idf.py monitor` on
the USB command port because the Python client needs exclusive access to it.

Pair `Maple-ESP32-Keyboard` with game computer B before sending commands. BLE
bonding data remains in NVS when firmware is updated without erasing flash.

## Test commands

Install project dependencies, including `pyserial`, then run:

```powershell
py ..\tools\esp32_hid_sender.py --serial-port auto status
py ..\tools\esp32_hid_sender.py --serial-port auto ping
py ..\tools\esp32_hid_sender.py --serial-port auto tap A --ms 60
py ..\tools\esp32_hid_sender.py --serial-port COM6 down LEFT --hold 2
```

Without a subcommand, the sender starts an interactive prompt.

## Protocol and safety

Commands remain compatible with the previous transport:

- `PING`, `STATUS`, `HELP`
- `DOWN <usage>`, `UP <usage>`
- `TAP <usage> <milliseconds>`
- `STATE [usage ...]`, `RELEASE_ALL`

Every command and response is an ASCII line ending in `\n`. The firmware
releases all held keys when no accepted command arrives for three seconds,
when BLE disconnects, and when the client explicitly closes.
