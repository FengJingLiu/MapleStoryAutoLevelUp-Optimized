# ESP32-S3 USB Serial + BLE Keyboard/Mouse HID Bridge

This directory implements the remote keyboard and mouse path used by the bot:

```text
Computer A --USB Serial/JTAG--> ESP32-S3 --BLE keyboard/mouse HID--> game computer B
```

Wi-Fi is not started. ESP-IDF logs remain on UART0 while the native ESP32-S3
USB Serial/JTAG port is dedicated to the line-oriented command protocol.

## Build and flash

Use ESP-IDF v6.0.2:

```powershell
cd D:\project\MapleStoryAutoLevelUp-Optimized\esp32\wifi_ble_hid_demo
idf.py set-target esp32s3
idf.py build
idf.py -p COM6 flash
```

Replace `COM6` if Windows assigns another port. Do not run `idf.py monitor` on
the USB command port because the Python client needs exclusive access to it.

Pair `Maple-ESP32-Keyboard` with game computer B before sending commands. BLE
bonding data remains in NVS when firmware is updated without erasing flash.
Because this firmware adds mouse reports to the HID descriptor, remove and
re-pair the device on computer B if Windows keeps the older keyboard-only
descriptor after flashing.

## Test commands

Install project dependencies, including `pyserial`, then run:

```powershell
py ..\tools\esp32_hid_sender.py --serial-port auto status
py ..\tools\esp32_hid_sender.py --serial-port auto ping
py ..\tools\esp32_hid_sender.py --serial-port auto tap A --ms 60
py ..\tools\esp32_hid_sender.py --serial-port COM6 down LEFT --hold 2
py ..\tools\esp32_hid_sender.py --serial-port COM6 mouse-move 120 -40
py ..\tools\esp32_hid_sender.py --serial-port COM6 mouse-click left
py ..\tools\esp32_hid_sender.py --serial-port COM6 mouse-click-at 16384 16384 left
py ..\tools\esp32_hid_sender.py --serial-port COM6 scroll -3
```

`MOUSE_ABS=1` is required for `MOUSE_CLICK_AT` and for calibrated game-UI
clicks. The current machine selects absolute session recovery in
`config_custom.yaml`. The visual-relative path remains available as an
explicit compatibility fallback. Without a subcommand, the sender starts an
interactive prompt.

## Protocol and safety

Commands remain compatible with the previous transport:

- `PING`, `STATUS`, `HELP`
- `DOWN <usage>`, `UP <usage>`
- `TAP <usage> <milliseconds>`
- `STATE [usage ...]`, `RELEASE_ALL`
- `MOUSE_MOVE <dx> <dy> [wheel]`
- `MOUSE_DOWN <left|right|middle>`, `MOUSE_UP <left|right|middle>`
- `MOUSE_CLICK <left|right|middle> [milliseconds]`
- `MOUSE_CLICK_AT <x:0..32767> <y:0..32767> <left|right|middle> <milliseconds>`

Every command and response is an ASCII line ending in `\n`. The firmware
releases all held keys and mouse buttons when no accepted command arrives for
three seconds, when BLE disconnects, and when the client explicitly closes.
Taps, relative moves, and mouse clicks are one-shot commands: the client never
replays one after an uncertain ACK and sends only the idempotent `RELEASE_ALL`
cleanup instead. Relative movement is re-observed from the next capture frame.

Absolute HID coordinates map to computer B's physical Windows desktop. The
GC573 frame contains Magpie's fullscreen 3840x2160 output, but Magpie forwards
pointer input through the original 1366x768 source client. Recovery therefore
maps capture pixels back into that physical source rectangle before
normalization to `0..32767`:

```yaml
esp32_hid:
  capture_frame_is_desktop: false
  absolute_desktop_rect: [0, 0, 3840, 2160]
  magpie_source_rect: [1235, 721, 1366, 768]
```

Recalibrate `magpie_source_rect` whenever the source game window moves or
resizes. The rectangle belongs to computer B's physical desktop; never derive
it from the PotPlayer window position on computer A.
