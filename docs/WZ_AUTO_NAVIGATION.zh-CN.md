# WZ 自动地图识别与寻路

这套模式不再读取手绘的 `route*.png`。程序把游戏小地图与 WZ 导出的标准小地图画布匹配，再从 foothold、rope、portal 和怪物出生点生成可执行路线。

## 数据准备

二进制 WZ 的解析边界复用 `SuperHumanMapleStory` 中的 .NET 导出器；本项目只读取其版本化 JSON/PNG 合约，避免同时维护第二套 WZ 解析器。

```powershell
dotnet D:\project\SuperHumanMapleStory\dotnet\MapleWzExporter\bin\Release\net10.0-windows7.0\MapleWzExporter.dll `
  export-all `
  --client-dir D:\games\mxd\冒险岛online `
  --output-dir D:\project\SuperHumanMapleStory\var\all-maps-with-canvases `
  --include-canvases true
```

导出只需在客户端 WZ 更新后重跑。目录中应同时存在 `<mapId>.json` 和 `canvases/<mapId>/*.png`。

## 启用

```yaml
monster_detect:
  backend: yolo
wz_navigation:
  enable: true
  geometry_dir: D:/project/SuperHumanMapleStory/var/all-maps-with-canvases
  projectile_terrain_check: true
  projectile_height_wz: 30
  projectile_clearance_wz: 8
```

UI 中也可以直接选择 `__auto_wz__ (WZ 自动识别)`。选择该虚拟地图会自动启用 WZ 模式；动态地图不能使用固定怪物模板，所以该模式要求 YOLO 怪物检测。

如果仍选择已有的 `forest_floor`，程序会先用它的 `map.png` 快速确定 WZ 地图，之后每帧只重新配准当前小地图。换图后连续三帧失配会释放全部移动键，并在 Hero 停止时扫描新地图；失败扫描最多每 10 秒重试一次。

## 预览和核验

```powershell
python -m tools.wz_navigation_preview `
  --minimap minimaps\forest_floor\map.png `
  --geometry-dir D:\project\SuperHumanMapleStory\var\all-maps-with-canvases `
  --output-dir screenshot\wz_navigation
```

已知地图可加 `--map-id 100040110` 跳过全库扫描。加 `--write-route-images` 可输出每一段自动路线。

预览图中绿色是合并后的可行走平台，青色是绳索/梯子，洋红色是传送门，红色是怪物出生点。巡逻图额外绘制各动作边。摘要 JSON 包含匹配置信度、平台/绳索/传送门数量、动作图规模和 Hero 运动参数。

## 树林的底层验证结果

当前 `forest_floor/map.png` 唯一匹配 WZ 地图 `100040110`：

- 特征匹配为 37/48 个内点，比例 0.771，缩放 2.806，95% 重投影误差 1.21 像素。
- 201 条原始 foothold 合并为 29 个可行走平台，另有 44 面竖直墙。
- 找到 15 条绳索/梯子、13 个传送门和 36 个怪物出生点。
- 动作图包含 170 个节点和 388 条边：282 WALK、25 JUMP、47 DROP、30 CLIMB、4 PORTAL。
- 怪物覆盖巡逻为 109 段：91 WALK、4 JUMP、10 DROP、1 CLIMB、3 PORTAL。

WZ 物理给出的基础参数为：行走速度 125 WZ 单位/秒、跳跃初速度 555、重力 2000。默认 0.82 安全系数下，规划器使用约 63.15 WZ 单位的跳高和 56.89 WZ 单位的水平跳距。

## 规划规则

1. 连续 foothold 先合并为平台；竖直 foothold 作为墙，带 `force` 的特殊平台不参与普通移动。
2. 怪物出生点吸附到最近平台，生成需要覆盖的平台区间。
3. 相邻锚点生成双向 WALK；平台重叠向下生成 DROP；满足弹道、安全余量且轨迹不穿过竖直墙时生成 JUMP。
4. 绳索两端吸附到平台后生成双向 CLIMB。
5. 只有目标仍在当前 mapId 内的成对传送门才生成 PORTAL，避免自动离开练级地图。
6. 以预计耗时和动作风险为权重，用 Dijkstra 连接怪物覆盖点，并闭合成巡逻路线。
7. 运行中从普通行走和跳跃帧估计实际像素速度、跳高和跳距；样本足够且变化超过 8% 时重新生成图。也可在 `wz_navigation.motion` 中直接填写像素测量值。

弓箭等会碰撞地形的横向技能可启用 `projectile_terrain_check`。选怪时会把画面距离换算回 WZ 世界坐标，并剔除弹道与竖墙或弹道高度附近 foothold 相交的目标；调试画面会用橙色标出 `Terrain blocked`。`projectile_height_wz` 是弹道高出脚下平台的 WZ 像素，`projectile_clearance_wz` 是箭体宽度与小地图配准误差余量。穿墙技能应保持关闭。

自动图是保守可执行图，不会假设所有几何上看似可达的平台都能稳定落地。首次在新职业、新移速或新跳跃属性下使用时，建议先以 `--disable_control` 观察匹配与路线预览，再开启输入。
