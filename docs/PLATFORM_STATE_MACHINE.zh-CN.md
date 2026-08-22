# WZ 平台状态机

树林的底层（WZ `100040110`）默认启用平台状态机。启动时只构建
foothold/绳索/传送点动作图，不再生成整圈 `routes`。识别到 Hero 连续两帧
稳定站在某个平台后，才实时计算当前位置到当前目标平台或巡逻点的临时路径。
临时路径只存在于内存，不写入 `minimaps/forest_floor/route*.png`。

默认循环配置位于 `config/config_default.yaml`：

```yaml
wz_navigation:
  platform_state_machine:
    enable: true
    maps:
      "100040110":
        sequence: [1, 3, 7, 9, 13, 9, 5, 3]
        dwell_seconds: 8.0
        maximum_dwell_seconds: 24.0
        combat_quiet_seconds: 0.8
        travel_combat_budget_seconds: 6.0
```

`sequence` 只写平台编号。比如要验证 `P1 -> P3 -> P4 -> P5 -> P6`，改为：

```yaml
sequence: [1, 3, 4, 5, 6]
```

平台之间的 WALK/JUMP/DROP/CLIMB 由当前 WZ 图实时求最短路径；两个平台没有
直接边时会自动选择可执行的绕路。树林的底层默认禁用 PORTAL 边，避免离开练级
循环。

状态含义：

- `LOCALIZING`：等待稳定的 Hero 落地平台。
- `DWELLING`：在当前平台的安全射击点间巡逻；可攻击怪物优先于移动。
- `TRAVELING`：执行到下一目标平台的临时路径。
- `BLOCKED`：路径暂时不可达，停止输入并定时从实时位置重算。
- `SUSPENDED`：F1 暂停、采集丢失或自动登录期间冻结驻留计时；恢复后重新定位。

为了避免 YOLO 假阳性造成永久空打，单个平台达到
`maximum_dwell_seconds` 后会强制进入下一状态；过渡途中同一检测连续抢占超过
`travel_combat_budget_seconds` 后会先完成到目标平台，再恢复清怪优先。

## 低 FPS 定时起跳

水平 WZ 跳跃不再等待 Hero 的小地图中心恰好覆盖 2–3 像素宽的起跳色块。程序在
预计一个真实采样周期内会到达起跳点时，按以下关系反推独立计时器：

```text
预测当前位置 = 帧内位置 ± 实测速度 × min(采集帧年龄, 同方向按键已持续时间)
起跳延时 = 剩余距离 / 实测速度 - HID延迟 - 匹配残差 / 实测速度
```

计时器在键盘控制线程之外执行一次原子 `方向状态 + Jump TAP`，因此不需要等待
下一次 YOLO/WZ 主循环。只有清怪、状态切换和卡住仲裁后仍保持同方向移动才会
上锁；F1、采集丢失、自动登录、路径更换、方向更换或强制回血都会取消或拒绝
迟到的 Jump。相关参数：

```yaml
route:
  timed_jump_enable: true
  timed_jump_input_lead_ms: 35
  timed_jump_max_lookahead_ms: 1200
  timed_jump_frame_age_limit_ms: 500
  timed_jump_confirmation_timeout_ms: 450
```

绳索起跳继续使用原有的实测动量计时（`rope_climb_runup_ms`），不会叠加第二次
等待。

运行日志中的关键行：

```text
[wz-navigation] Platform state machine ready without a pre-rendered route loop
[wz-navigation] [platform-fsm] Enter P1 ...
[wz-navigation] [platform-fsm] Planned TRANSIT P1->P3 ...
[wz-navigation] Rendered temporary TRANSIT P1->P3 ...
[timed-jump] Armed route2 right: ... speed=21.358px/s ... delay=224ms
```
