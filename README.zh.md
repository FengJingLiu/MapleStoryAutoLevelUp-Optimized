# 楓之谷Artale 自動練等腳本

一款用於楓之谷Artale的自動練等腳本機器人

## 優化分支

此儲存庫由 [@Micheal-tcy](https://github.com/Micheal-tcy) 維護，基於
[KenYu910645/MapleStoryAutoLevelUp](https://github.com/KenYu910645/MapleStoryAutoLevelUp)。
原作者、提交紀錄與 MIT 授權資訊均予以保留。

目前修正包含 SQDIFF 偵測排序、偵測框尺寸一致性、關閉除錯視窗時的
穩定性、暫停與重新啟動流程、公開郵件憑證，以及 Windows/Python 3.12
自動化測試，並**支援 MapleStory Worlds 以外的其他程式視窗**（見下方說明）。

## 支援其他程式視窗

腳本不再寫死只能鎖定 `MapleStory Worlds` 視窗，你可以將它指向其他楓之谷
客戶端、私服客戶端或模擬器視窗。

**透過 UI（建議）：** 在「主頁面」使用新的 **🎯 Target Program（目標程式）**
區塊：

- **Window（視窗）**：從下拉選單選擇一個已開啟的視窗（按 **🔄 Refresh** 重新
  掃描），或直接輸入視窗標題的一部分。
- **Exact title match（完全比對）**：要求標題完全相符，而非子字串比對（當有
  多個視窗名稱相近時很有用）。
- **Auto-resize window（自動調整大小）**：啟動時強制調整目標視窗大小。楓之谷
  需要固定解析度，請保持開啟；對於不可被移動／縮放的程式，請關閉此選項。

**透過設定檔：** 編輯設定檔中的 `game_window` 區段：

```yaml
game_window:
  title: "你的視窗標題"        # 目標視窗標題的子字串
  exact_match: False          # True = 需完全相符
  auto_resize: True           # False = 不調整目標視窗大小
  resize_width: 1296          # auto_resize 啟用時使用的視窗寬度
  resize_height: 759          # auto_resize 啟用時使用的視窗高度
```

> [!NOTE]
> 電腦視覺偵測（小地圖、隊伍紅條、怪物、符文）是針對楓之谷調校的。鎖定非楓之谷
> 程式可以擷取並控制該視窗，但內建偵測可能需要另外製作對應的樣板才能正常運作。

> [!WARNING]
> 自動操作可能違反遊戲服務條款並導致帳號風險，使用前請先確認伺服器規則。

<img src="media/intro2.gif" width="100%">

[▶ 在 YouTube 上觀看Demo](https://www.youtube.com/watch?v=QeEXLHO8KN4)

## 下載
[![Latest Release](https://img.shields.io/github/v/release/KenYu910645/MapleStoryAutoLevelUp)](https://github.com/KenYu910645/MapleStoryAutoLevelUp/releases/latest)

📥 **[點此下載最新版](https://github.com/KenYu910645/MapleStoryAutoLevelUp/releases/latest)**

## 執行方式
1. 執行 MapleStory World，並將遊戲設定為視窗模式，且視窗大小縮至最小
2. 開啟遊戲左上角的小地圖
3. 在遊戲中建立隊伍（按下 `P` 並點擊「建立」），確保遊戲角色上方出現紅色血條
4. 將角色移動到想要練功的地圖
5. **[下載最新版本](https://github.com/KenYu910645/MapleStoryAutoLevelUp/releases/latest)**
6. 解壓縮 MapleStoryAutoLevelUp.zip，執行 MapleStoryAutoLevelUp.exe
7. 在 UI 主頁面調整設定
8. 按下 `Start` 按鈕或 `F1` 鍵開始腳本
9. Enjoy!

## 功能介紹
本專案完全以電腦視覺技術實作，無需讀取遊戲記憶體。透過偵測遊戲畫面上的圖像（例如角色的紅色血條與怪物），並模擬鍵盤輸入來控制角色。

✅ 不需讀取遊戲記憶體

✅ 純電腦視覺實作

✅ 模擬真實鍵盤輸入

✅ 友善的使用者介面
| ![Main Tab](media/main_tab.png) | ![Advanced Tab](media/adv_settings_tab.png) |
|:-------------------------------:|:-------------------------------------------:|
| 主頁面 | 進階設定頁面 |

✅ 自動解符文
<img src="media/rune_solve.gif" width="100%">

✅ 除錯視窗

✅ 自動喝 HP/MP 藥水

✅ 自動換頻道

✅ 角色建立自動擲骰

✅ 支援全球與台服 Artale 伺服器

✅ 支援英文與繁體中文

## 環境需求
* Windows11/MacOS
* Python3.12
* OpenCV4.11

注意：本專案不支援虛擬機環境，僅供娛樂與學術用途。

### 郵件測試憑證

實驗性的 `tools/email_test.py` 不再包含憑證。執行前請設定
`MAPLE_BOT_SENDER_EMAIL`、`MAPLE_BOT_EMAIL_PASSWORD` 與
`MAPLE_BOT_RECEIVER_EMAIL` 環境變數，且不要將真實密碼提交至設定檔。

## 支援的 MapleStory 版本
本專案主要在 MapleStory Artale Taiwan與Global伺服器開發與測試。

## 執行方式（開發者用）

### 安裝依賴
```bash
pip install -r requirements.txt
```

本分支預設以 `models/yolo/mob_1024_best.pt` 的 1024 YOLO 模型辨識怪物，
只保留模型中的 `mob` 類別。Windows + NVIDIA 顯示卡建議先安裝 CUDA 版
PyTorch，再安裝其餘依賴：

```powershell
uv pip install --python .venv\Scripts\python.exe torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

可用以下指令確認是否已使用 GPU：

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### GC573 DirectShow 擷取

Windows 預設不再經過 PotPlayer 視窗截圖，而是由 OpenCV DirectShow 直接
讀取 GC573。啟動時會嚴格驗證 RGB24、3840×2160、60 FPS；若驅動退回其他
格式或解析度，程式會安全停止。DirectShow 的媒體子型別是 RGB24，而
OpenCV 提供給既有視覺演算法的陣列仍是 BGR 通道順序。

```yaml
capture:
  source: directshow
capture_card:
  device_index: 0
  device_name: "AVerMedia GC573 1 Capture"
  width: 3840
  height: 2160
  fps: 60
  pixel_format: RGB24
game_window:
  capture_profile: capture_card
```

影像會保持原生 4K，不再裁切播放器外框，也不再縮放至 1296×700。
`device_index: 0` 是目前電腦已實測的 GC573 端點；只有需要舊版視窗擷取時
才把 `capture.source` 改為 `window`。

舊路線和像素範本不會被插值放大；請直接從原生 4K 畫面重錄實際會用的素材。
完整檔名與條件請見 [DirectShow 4K 素材重做清單](DIRECTSHOW_4K_ASSET_CHECKLIST.md)。

鍵盤指令由電腦 A 經 USB 串口傳給 ESP32-S3，再由 ESP32-S3 透過 BLE HID
送到遊戲電腦 B。預設會自動尋找 ESP32-S3 USB Serial/JTAG 裝置；也可固定端口：

```yaml
esp32_hid:
  remote_target: True
  serial_port: "auto"  # 或 "COM6"
  baudrate: 115200
```

環境變數 `ESP32_HID_SERIAL_PORT` 可覆蓋設定檔。固件編譯與測試方式請參閱
`esp32/README.md`。

遠端模式中，組隊視窗、遊戲內換頻道及本機視窗啟用等通用滑鼠
流程仍停用。掉線重新登入是限定用途的例外：每個頁面通過影像確認後，
先在擷取畫面中識別手形游標，再透過 ESP32 `MOUSE_MOVE` 小步相對移動。
點擊前必須先由探針或導向移動證明游標確實依指令方向移動，靜態的相似圖案
不能通過；只有游標連續兩幀進入目標範圍後，才會在
目前位置傳送 `MOUSE_CLICK`，不再要求 `MOUSE_ABS=1`。`auto_relogin` 的五步流程如下：

1. 識別掉線提示框，點擊設定的「確定」座標。
2. 識別連線頁，傳送 `Enter`。
3. 識別世界頁，點擊「漂漂豬」範本中心。
4. 識別頻道頁，從設定的 20 個頻道座標中隨機選擇一個。
5. 識別角色頁，點擊「開始遊戲」範本中心。

只有在連續畫面中確認新的小地圖玩家點後才恢復遊戲控制。每個動作都必須
先命中對應的頁面範本；若下一頁或新遊戲畫面未在限時內出現，功能會安全
停用自動輸入，等待人工恢復。設定座標已改用 DirectShow 4K，並以
`[2160, 3840]` 表示「高、寬」，但倉庫內的 PNG 仍是舊解析度素材。
因此在從原生 4K 畫面重做下列頁面與游標範本、熱點及錨點之前，
`auto_relogin.enable` 預設為 `False`。頻道覆蓋世界頁及掉線模態框會先按
頁面優先級分類；掉線與連線頁使用 `Enter`，世界與角色頁點擊匹配中心，
只有頻道固定點會跟隨本幀頁面範本錨點平移。請保持相同的 UI 配置與游標外觀；其他
客戶端需於 `auto_relogin` 更換頁面範本、游標範本、熱點、座標與頁面錨點。
游標識別模糊、舊畫面、頁面消失、
擷取尺寸改變、移動停滯或逾時時都會安全停止且不點擊。此流程只恢復已驗證的
工作階段，不會輸入帳號密碼，也不處理啟動器、驗證碼、雙重驗證或未預期頁面。

DirectShow 沒有可點擊的桌面視窗，電腦 A 的桌面座標不參與遠端點擊。
遊戲電腦的縮放只會改變單次相對移動的距離，下一幀會依游標實際落點
繼續修正；但游標必須出現在擷取畫面中，且外觀尺寸需符合範本。

### 建議使用 UI 執行
執行以下指令
```bash
python -m src.main
```
按下 `F1` 或 `start` 按鈕開始

調整設定以符合你的角色

進階設定仍在開發中，若需修改請編輯 config_default.yaml

### 不使用 UI 執行
#### 執行腳本
```bash
python -m src.engine.MapleStoryAutoLevelUp
```
#### 使用自訂設定檔
```bash
python -m src.engine.MapleStoryAutoLevelUp --cfg my_config
```
#### 關閉除錯視窗
```bash
python -m src.engine.MapleStoryAutoLevelUp --disable_viz
```
#### 錄製除錯視窗
```bash
python -m src.engine.MapleStoryAutoLevelUp --record
```
#### 透過 config.yaml 選擇地圖
在設定檔中修改：
```yaml
# 於 config/config_custom.yaml
bot:
  map: ""  # 設定地圖名稱，可於 config/config_data.yaml 內查詢
```

#### 弓箭手單側 AOE

方向型角色可以保留普通攻擊，並在左側或右側技能範圍內的怪物數量
達到門檻時改用單側 AOE。請在 `config/config_custom.yaml` 加入類似設定：

```yaml
bot:
  attack: directional
key:
  directional_attack: control  # 普通攻擊鍵
  aoe_skill: shift             # 單側 AOE 鍵；必須與普通攻擊鍵不同
directional_aoe:
  enable: true
  min_monsters: 3              # 單側達到此數量（含）時施放
  range_x: 350
  range_y: 70
  cooldown: 0.9
  attack_recovery_delay: 0.9
```

若左右兩側都達到門檻，會先選怪物較多的一側；數量相同時選最近的一側。
AOE 冷卻期間會等待，不會退回普通攻擊。

#### 弓箭手近身強力擊退

怪物距離太近而無法射箭時，可啟用近身備援技能，並將「強力擊退」綁定為
`S`：

```yaml
key:
  power_knockback: s
power_knockback:
  enable: true
  trigger_distance_x: 100     # 水平中心距離，包含臨界值
  range_y: 70
  cooldown: 0.9
  attack_recovery_delay: 0.9
```

任一近身怪會令同側暫時不可射箭；若另一側有可射的怪物，會優先攻擊另一側。
只有左右都沒有可射目標時才會使用強力擊退。距離以原始 1296 x 700 基準像素
設定，執行時會依目前擷取解析度自動縮放。

* 按下 `F1` 暫停/繼續腳本
* 按下 `F2` 截圖，檔案會存於 screenshot/
* 按下 `F12` 結束腳本

## 支援地圖
請參考 config/config_data.yaml

## 想製作新地圖？→ Route Recorder

可使用 `routeRecorder.py` 來設計自訂路徑地圖。此工具會監聽鍵盤輸入並記錄成路徑圖。

在終端機輸入以下指令開始記錄：
```bash
python -m tools.routeRecorder --new_map <map_directory_name>
```
| 鍵位 | 功能 |
| ---- | ------------------------------------------ |
| `F1` | 暫停或繼續記錄 |
| `F2` | 截圖（儲存於 `screenshot/`） |
| `F3` | 儲存目前路徑並開始新的 |
| `F4` | 將目前地圖存為 map.png |

### 手工繪製傳送門觸發色塊

路線製作器不會產生傳送門色塊。儲存路線後，請手工編輯對應的
`minimaps/<map>/route*.png`，在傳送門觸發位置畫一個橫向實心色塊：

- 顏色：RGB `(127, 255, 255)`，十六進位 `#7FFFFF`；若使用 OpenCV，BGR 為
  `(255, 255, 127)`。
- 建議尺寸約為小地圖座標的 `11 x 5` 像素。色塊應保持為單一連通區域，
  並避免被其他路線顏色覆蓋。
- 只有 Hero 色塊質心進入此區域才會觸發。程式會持續按住 `Up`，同時短按
  左右方向鍵，並在到達該連通區域的左右邊界前反向。
- 偵測到同一地圖內的瞬間位置跳變後會結束操作；若 6 秒內未成功，程式會
  釋放控制，且必須等 Hero 離開色塊後才會再次嘗試。
- RGB `(255, 255, 127)` 仍是原有的 `Down` 路線顏色，不可用作傳送門色塊。

### 手工繪製爬繩引導線

路線製作器維持不變，不會建立爬繩指令。儲存路線後，請手工編輯對應的
`minimaps/<map>/route*.png`，從目前平台的接近路線往繩索畫一條顏色完全相符、
保持連通的線段：

- 顏色：RGB `(0, 127, 255)`，十六進位 `#007FFF`；若使用 OpenCV，BGR 為
  `(255, 127, 0)`。
- 近端放在平台的一般路線能提前遇到的位置，遠端落在繩索的 x 接觸位置。
  程式會提前辨識線段，並把距離 Hero 較遠的端點預判為繩索位置。
- 每條引導線必須是各自獨立的連通區域，不要把兩條繩索接在一起，也不要用
  其他路線顏色蓋住它。
- 程式會自行產生起跳點（預設距繩索 8 個小地圖像素）。若 Hero 已朝繩索
  同向奔跑，方向鍵會持續按住，到起跳點直接跳躍，不會停頓；否則先重新定位，
  再以左／右助跑後按下 `Up + Jump`，不會使用原地上跳。
- 掛上繩索後會持續按住 `Up`，依小地圖向上位移判定進度；若掛繩沒有進度，
  會改從另一側重試。未手工加入此顏色的舊路線維持原有行為。

* 新地圖建立後，請至 config/config_data.yaml 登記怪物
* 若為大型地圖，建議先探索一次地圖再開始記錄路徑
* 按下 `F4` 更新目前掃描地圖至除錯視窗，若滿意即可按 `F3` 重新記錄路徑
* 記錄時可邊打怪，攻擊鍵不會被記錄
* 原始路徑通常效果不好，可利用繪圖軟體(Paint)微調

## 想新增怪物？→ Mob Maker

可於下列網站查詢要加入的怪物名稱：

[Maplestory GMS 65](https://maplestory.wiki/GMS/65/mob)

```bash
python tools/mob_maker.py

>Fetching mobs from: https://maplestory.io/api/GMS/65/mob
>You can find monster names at https://maplestory.wiki/GMS/65/mob
>Enter mob name:Snail
```
下載完成後的圖片會位於 `monster/{MonsterName}` 資料夾。

## 自動擲骰工具
在創角介面自動擲骰，可輸入想要的屬性值交由腳本完成

```bash
python -m tools.AutoDiceRoller --attribute <STR,DEX,INT,LUK>

# 例如建立全智法師角色：
python -m tools.AutoDiceRoller --attribute 4,4,13,4

# 亦可使用問號表示不指定數值
python -m tools.AutoDiceRoller --attribute 4,4,?,?
```
## Discord
歡迎加入 Discord 伺服器討論

https://discord.gg/DqjtJGNEx7

## 舊版本
此專案以前是透過全螢幕截圖來定位相機與規劃路徑，但後來發現直接從左上角小地圖取得位置更簡單可靠。因此重新設計了新的定位方式，並將所有 maps/ 遷移至 minimaps/。若仍想使用舊方法，請執行：
```bash
python -m src.legacy.mapleStoryAutoLevelUp_legacy.py --map <map> --monsters <monster> --attack <skill>

# 範例：
python -m src.legacy.mapleStoryAutoLevelUp_legacy.py --map lost_time_1 --monsters evolved_ghost --attack aoe_skill
```

## ☕ 贊助作者
若覺得本專案對你有幫助，歡迎請作者喝杯咖啡！

> 你可以自由輸入金額，$1、$5 或 $10 均可
> 提醒：贊助金額以 **美元** 計算

[![Buy Me a Coffee](https://img.shields.io/badge/%F0%9F%92%96_Tip_me_$1_or_more-yellow?style=flat-square&logo=buymeacoffee)](https://www.buymeacoffee.com/kenyu910645)
