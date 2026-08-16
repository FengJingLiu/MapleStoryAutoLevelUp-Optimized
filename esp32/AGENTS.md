# ESP32 开发与排障约束

本文件适用于 `esp32/` 及其全部子目录。修改 ESP32-S3 串口到 BLE
键盘/鼠标桥接固件前，必须先阅读本文件。

## 不可破坏的行为约束

- `STATE [usage ...]` 只替换键盘状态。空 `STATE` 表示释放全部键盘按键，仍须调用
  `ble_keyboard_set_state(usages, 0)`；不得把它改成 `ble_keyboard_release_all()`。
- `RELEASE_ALL` 才是键盘和鼠标的全局安全释放。没有鼠标按钮待释放时，它不得重新发送
  空闲的相对鼠标报告，也不得重发缓存的绝对坐标。否则主循环正常发送空 `STATE` 时，会把
  光标持续拉回最近一次 `MOUSE_CLICK_AT` 的位置。
- 绝对鼠标报告仅可在新的绝对移动/点击发生时发送，或确实存在绝对鼠标按钮释放待完成时发送。
  不要把“缓存中有绝对坐标”当作“需要再次发送坐标”。
- `CONFIG_ESP_MAIN_TASK_STACK_SIZE` 必须保持至少 `8192`，除非在真实硬件上测得 BLE/HID
  初始化调用链的栈高水位并证明更小的值有足够余量。当前 `3584` 会在 BLE 初始化期间触发
  `A stack overflow in task main has been detected` 并形成启动循环。
- 正式固件的 ESP-IDF 日志保持在 UART0。原生 USB Serial/JTAG 端口供行协议独占，不能让日志
  混入 `PING`、`STATUS`、HID 命令及其响应。
- 默认不得擦除 NVS。NVS 保存 BLE bonding；一旦擦除，Windows/游戏电脑必须删除旧的
  `Maple-ESP32-Keyboard` 后重新配对。

上述约束当前落实在：

- `wifi_ble_hid_demo/main/serial_control.c`
- `wifi_ble_hid_demo/main/ble_keyboard.c`
- `wifi_ble_hid_demo/sdkconfig.defaults.example`

## 构建

项目使用 ESP-IDF v6.0.2，`CMakeLists.txt` 已通过 `SDKCONFIG_DEFAULTS` 加载
`sdkconfig.defaults.example`。标准流程：

```powershell
cd D:\project\MapleStoryAutoLevelUp-Optimized\esp32\wifi_ble_hid_demo
idf.py set-target esp32s3
idf.py build
```

构建前后检查实际生成的 `sdkconfig` 中包含：

```text
CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192
CONFIG_ESP_CONSOLE_UART_DEFAULT=y
```

如果直接使用 Ninja，应用目标名是 `app`，不是输出文件名：

```powershell
ninja -C <build-directory> -j 4 app
```

不要执行 `ninja ... maple_serial_ble_hid_demo.bin`；该名称不是 Ninja target。首次完整构建可能
需要数分钟。不要因为耗时、无关子模块警告或一次瞬时编译器故障就改动业务代码；先重试并确认
真正失败的 target。烧录前核对 `.bin` 的更新时间和 SHA-256，防止把旧产物误当成新构建。

本机 ESP-IDF 导出脚本若因中文用户路径失败，可临时手动设置以下环境；这些 `R:` 路径只适用于
2026-08-16 排障所用机器，不是项目的通用依赖：

```powershell
$env:IDF_PATH='R:\Documents\Codex\2026-08-12\d-project\work\esp-idf-v6.0.2-ref'
$env:IDF_TOOLS_PATH='R:\.espressif'
$env:IDF_PYTHON_ENV_PATH='R:\.espressif\python_env\idf6.0_py3.14_env'
```

## 烧录、复位与配对

正常优先使用：

```powershell
idf.py -p COM6 flash
```

端口可能变化，当前板卡是 ESP32-S3，USB VID:PID 为 `303A:1001`。如果 bootloader 和分区表
未变，可以只把应用镜像烧录到 `0x10000`，从而减少无关变量并保留其余已知正常内容。不要在
没有必要时执行 erase-flash。

若板卡仍停留在 ROM 下载模式，普通 hard reset 可能无法启动应用：

- 进入下载模式：按住 BOOT，点按 RESET/EN，约两秒后松开 BOOT。
- 启动应用：确保 BOOT 已松开，仅点按 RESET/EN。
- 也可用 esptool 的 `--before no-reset --after watchdog-reset run`。复位时 COM 端口会消失，
  pyserial/esptool 因断连以退出码 1 结束可能是预期现象；必须随后用 `STATUS`/`PING` 判断应用
  是否真正启动，不能只根据该退出码判定失败。

固件更新本身不会要求重新配对；只有擦除 NVS、修改 HID descriptor，或 Windows 缓存旧 descriptor
时才需要在游戏电脑上删除设备并重新配对。NVS 被擦除后，`BLE_READY=0` 是正常的未配对状态，
不是启动失败。

## 启动循环排障顺序

不要先假定启动循环由最近的鼠标补丁造成，也不要靠逐段撤销功能猜测。按以下顺序取证：

1. 确认 COM 端口是否反复出现/消失，并记录 reset reason。
2. 从 UART0 捕获完整 panic/backtrace。若当前硬件暂时无法读取 UART0，可仅为诊断把 console
   切到 USB Serial/JTAG，重新构建并获取 panic；定位后必须恢复 UART0，再构建正式固件。
3. 若看到 `rst:0xc (RTC_SW_CPU_RST)` 且重启发生在 BLE/HID 初始化附近，优先检查 main task
   栈，而不是继续修改 BLE 配对或鼠标逻辑。
4. 检查生成的 `sdkconfig`，确保 main task stack 是 `8192`，再进行干净的应用重构建。
5. 烧录后连续发送三次 `STATUS` 和一次 `PING`；只有状态稳定且返回 `PONG` 才算启动通过。

Python 的 `HidClient` 构造阶段要求 `SERIAL=1` 且 `BLE_READY=1`，所以未配对时它可能拒绝启动。
此时应使用 pyserial 直接发送原始 `STATUS\n`、`PING\n` 诊断固件，不要把客户端的拒绝误判成
串口或固件崩溃。

## 修改后的最低验收

每次涉及串口协议、BLE/HID、鼠标或构建配置的修改至少完成：

```powershell
.venv\Scripts\python.exe -m pytest -q tests\test_esp32_hid_client.py tests\test_keyboard_controller_esp32.py
git diff --check -- esp32
```

并在真实硬件上验证：

1. 连续三次 `STATUS` 均返回，且板卡不复位。
2. `PING` 返回 `PONG`。
3. 已配对时确认 `SERIAL=1 BLE_READY=1 MOUSE=1 MOUSE_ABS=1`；刚擦除 NVS 时允许
   `BLE_READY=0`，但必须明确记录需要重新配对。
4. 先执行一次 `MOUSE_CLICK_AT`，再连续发送有键和空的 `STATE`；光标不得被拉回刚才的绝对坐标。
5. 按住鼠标按钮后执行安全释放，按钮必须释放，同时不能产生额外位置跳变。

## 2026-08-16 故障记录

这次排障包含两个互不相同的问题：

- 运行期鼠标被持续操作：Python 主循环会发送空 `STATE` 释放键盘；固件却把空 `STATE` 转成
  `ble_keyboard_release_all()`，后者又重发最后的绝对鼠标坐标，导致光标不断回到最近一次
  `MOUSE_CLICK_AT`。修复为始终用 `ble_keyboard_set_state()` 处理 `STATE`，并让
  `ble_keyboard_release_all()` 只在确有鼠标释放待完成时发送鼠标报告。
- 新固件烧录后启动循环：实际 panic 是 main task stack overflow，与上述鼠标语义修复无关。
  将 `CONFIG_ESP_MAIN_TASK_STACK_SIZE` 从 `3584` 提高到 `8192` 后恢复稳定。

本次还曾擦除 NVS，因此最终验证时出现 `SERIAL=1 BLE_CONNECTED=0 BLE_READY=0 MOUSE=1
MOUSE_ABS=1`；连续 `STATUS` 稳定且 `PING=PONG`，说明固件正常，只需重新进行 BLE 配对。
当时最终应用镜像 SHA-256 为
`4DF35C1E460B5D9AF6306F2EF52FD93D397C029A752F25B4E78EE98E7FAA9BE9`，仅用于识别该次
已验证产物，后续正常修改固件后哈希必然变化。
