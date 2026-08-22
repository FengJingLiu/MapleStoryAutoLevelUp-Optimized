# DirectShow 4K 素材重做清单

所有新素材必须直接来自 `CaptureCardCapturor.get_frame()` 返回的
`(2160, 3840, 3) uint8 BGR` 原帧。固定游戏分辨率、UI 缩放、语言、小地图
展开状态和鼠标主题；禁止使用 PotPlayer/Qt 预览截图，也不要缩放旧图。

## 1. 每张要使用的地图（必做）

- `minimaps/<map>/map.png`
- `minimaps/<map>/route1.png`，以及需要的 `route2.png`、`route3.png`……
- `minimaps/<map>/minimap_geometry.txt`

三类文件必须由新版 `tools/routeRecorder.py` 在同一次 3840×2160 设置下生成；
`map.png` 与全部 `route*.png` 必须完全同尺寸。路线动作颜色、传送门、绳索和
原地跳标记均须重新画，不能复制旧像素坐标。正式地图名以
`config/config_data.yaml` 为准；现有 29 张正式地图若都要使用，就都要重录。

## 2. 中文 RuneSolver（使用符文时重做，共 17 张）

- `rune/arrow_left_1.png`、`rune/arrow_left_2.png`、`rune/arrow_left_3.png`
- `rune/arrow_right_1.png`、`rune/arrow_right_2.png`、`rune/arrow_right_3.png`
- `rune/arrow_up_1.png`、`rune/arrow_up_2.png`、`rune/arrow_up_3.png`
- `rune/arrow_down_1.png`、`rune/arrow_down_2.png`、`rune/arrow_down_3.png`
- `rune/rune_1.png`
- `rune/rune_2_cn.png`
- `rune/rune_3.png`
- `rune/rune_warning_cn.png`
- `rune/rune_enable_cn.png`

箭头、符文分段和 warning 使用精确绿幕 `BGR(0,255,0)`，不需要 alpha；
enable 为灰度匹配。还要用 4K 实测值重校箭头框坐标/间隔/大小、圆半径、
warning/enable ROI 和阈值。当前 `rune_solver.enable: False`，17 张素材与
参数全部完成前不要启用。

## 3. 当前角色定位（启用对应检测器时重做，共 7 张）

- `nametag/liu_muning.png`：角色 ID；精确绿幕。
- `nametag/liu_muning_overhead_smile.png`：头顶笑脸气泡；精确绿幕。
- `nametag/liu_muning_appearance_climb.png`：攀爬姿态；精确绿幕。
- `nametag/liu_muning_appearance_stand_left.png`：站立朝左；精确绿幕。
- `nametag/liu_muning_appearance_stand_right.png`：站立朝右；精确绿幕。
- `nametag/liu_muning_medal.png`：勋章/称号条；自然背景紧裁。
- `nametag/liu_muning_pet.png`：宠物名字；自然背景紧裁。

重做后保留 `nametag.template_reference_size: [2160, 3840]`，并重校
`player_offset`、组件尺寸、搜索半径和阈值。当前 custom 配置已暂时关闭
appearance/overhead/medal/pet，且 `nametag.enable` 仍为 `False`，避免任一
旧角色模板参与控制；7 张全部替换并校准后再启用。

## 4. 掉线自动重登（重新启用前必做，共 7 张）

自然色页面特征（选择稳定、唯一、非动画的小区域，不要截整页）：

- `misc/auto_relogin_disconnect_cn.png`
- `misc/auto_relogin_connect_cn.png`
- `misc/auto_relogin_world_cn.png`
- `misc/auto_relogin_channel_cn.png`
- `misc/auto_relogin_character_cn.png`

鼠标模板（必须为带透明背景的 BGRA 四通道 PNG）：

- `misc/auto_relogin_cursor_cn.png`
- `misc/auto_relogin_cursor_small_cn.png`

同时重校各 OCR `search_region`、两种 cursor hotspot、搜索半径和阈值。世界页
直接点击当前帧中“4.漂漂猪”的 OCR 框中心；频道弹窗出现后，再从当前帧 OCR
结果中选择一个“频道N”框中心双击，不记录 `channel_points`。掉线与连接页使用
`Enter`，不需要鼠标确认坐标。保持
`flow_template_reference_size: [2160, 3840]`；完成前不要把
`auto_relogin.enable` 改回 `True`。

可用下列工具从保存的原生 4K PNG 逐张无缩放裁取页面模板：

```powershell
python -m tools.extract_auto_relogin_templates `
  --page disconnect --frame screenshot/directshow_disconnect.png `
  --box X0 Y0 X1 Y1 --output-dir misc --overwrite
```

## 5. 条件使用的其他模板

- 英文界面：`misc/login_button_eng.png`、
  `misc/party_button_create_enable_eng.png`、
  `misc/party_button_create_disable_eng.png`、
  `rune/rune_2_eng.png`、`rune/rune_warning_eng.png`、
  `rune/rune_enable_eng.png`。
- 旧本地窗口登录/组队：`misc/login_button_cn.png`、
  `misc/party_button_create_enable_cn.png`、
  `misc/party_button_create_disable_cn.png`。当前 DirectShow 远程模式禁止这些
  本地点击流程，因此暂时不用重做。
- AutoDice：`numbers/4.png` 至 `numbers/13.png` 共 10 张灰度字形；还要重校
  四行 ROI、行距、掷骰按钮点和阈值。DirectShow 当前只做识别并拒绝误点。
- 怪物旧模板后端：`monster/<monster_name>/<monster_name>*.png` 共 113 张，
  只有切换到 `monster_detect.backend: template` 才需要全部用精确绿幕重做。
  默认 `backend: yolo` 使用模型文件，不需要重做这些 PNG。

## 6. 不需要重做

- `minimaps/**/route_rest.png`
- `minimaps/empty_house/upper_route*.png`
- `rune/rune.png`
- `maps/**`、`media/**`、`tests/fixtures/**`
