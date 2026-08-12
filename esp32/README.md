# ESP32-S3 Wi-Fi + BLE HID Demo

这个目录是一条独立于游戏电脑输入设备的测试链路：

```text
控制端 Python 脚本 --TCP/Wi-Fi--> ESP32-S3 --BLE HID--> Windows 笔记本
```

当前 Demo 用于验证三件事：

1. ESP32-S3 能连接 2.4 GHz Wi-Fi。
2. Windows 能把 ESP32-S3 配对为蓝牙键盘。
3. 控制端能通过 TCP 发送按下、松开、点击和全部松开指令。

## 目录

- `wifi_ble_hid_demo/`：ESP-IDF v6.0.2 固件项目。
- `tools/esp32_hid_sender.py`：Windows/Linux/macOS 均可运行的纯标准库测试客户端。

## 1. 配置并烧录固件

先安装 [ESP-IDF v6.0.2](https://github.com/espressif/esp-idf/releases/tag/v6.0.2)，进入 ESP-IDF PowerShell，然后执行：

```powershell
cd D:\project\MapleStoryAutoLevelUp-Optimized\esp32\wifi_ble_hid_demo
idf.py set-target esp32s3
```

本机实际使用的 `sdkconfig.defaults` 包含 Wi-Fi 凭据，因此已被 Git 忽略。新环境先从无密码模板复制：

```powershell
Copy-Item sdkconfig.defaults.example sdkconfig.defaults
```

然后在 `sdkconfig.defaults` 中填写：

- `CONFIG_HID_WIFI_SSID`
- `CONFIG_HID_WIFI_PASSWORD`

不要提交实际的 `sdkconfig.defaults`；NimBLE、安全等级和 HID 所需的可复现配置保留在 `sdkconfig.defaults.example`。

需要修改蓝牙设备名或 TCP 端口时，可以运行 `idf.py menuconfig`，进入 `Wi-Fi BLE HID Demo` 菜单。

然后构建、烧录并打开串口监视器：

```powershell
idf.py build
idf.py -p COM5 flash monitor
```

把 `COM5` 换成开发板实际串口。串口日志出现 `got ip:` 后记下 IP 地址。

## 2. 在 Windows 配对 BLE 键盘

打开“设置 -> 蓝牙和设备 -> 添加设备 -> 蓝牙”，选择：

```text
Maple-ESP32-Keyboard
```

如果修改了 `Bluetooth HID device name`，选择修改后的名称。首次连接使用无需 PIN 的加密绑定；绑定信息保存在 NVS。

## 3. 发送测试按键

先测试 TCP 连接：

```powershell
py -3 ..\tools\esp32_hid_sender.py --host 192.168.1.123 ping
```

点击一次 A，按住左方向键 2 秒：

```powershell
py -3 ..\tools\esp32_hid_sender.py --host 192.168.1.123 tap A --ms 80
py -3 ..\tools\esp32_hid_sender.py --host 192.168.1.123 down LEFT --hold 2
```

也可以进入交互模式：

```powershell
py -3 ..\tools\esp32_hid_sender.py --host 192.168.1.123
```

交互模式示例：

```text
tap A 80
down LEFT
up LEFT
status
release
quit
```

列出脚本已内置的键名：

```powershell
py -3 ..\tools\esp32_hid_sender.py keys
```

## 防卡键设计

- TCP 客户端会每秒发送心跳。
- ESP32 默认 3 秒未收到有效命令就释放全部按键并断开 TCP。
- TCP 断开、Wi-Fi 掉线或 BLE 断开时，也会清空按键状态。
- 客户端在 Ctrl+C 和正常退出时会尽力发送 `RELEASE_ALL`。

## 注意

- ESP32-S3 仅支持 2.4 GHz Wi-Fi，不能连接只开启 5 GHz 的 SSID。
- 这是局域网功能验证 Demo，TCP 协议没有鉴权或加密；只应在可信网络使用。
- BLE 与 Wi-Fi 共用 2.4 GHz 射频，ESP-IDF 会处理共存调度，但拥挤网络仍可能增加按键延迟。
- 如果重新刷机后 Windows 拒绝配对，先在 Windows 删除旧设备，再重新添加。
