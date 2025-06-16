import time
import threading
import mss
import cv2
import numpy as np
from logger import logger
from config.config import Config
import Quartz

def get_window_region(window_title):
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID
    )
    for window in window_list:
        if window.get(Quartz.kCGWindowName, '') == window_title:
            bounds = window.get(Quartz.kCGWindowBounds, {})
            return {
                "left": int(bounds.get('X', 0)),
                "top": int(bounds.get('Y', 0)),
                "width": int(bounds.get('Width', 0)),
                "height": int(bounds.get('Height', 0))
            }
    return None

class GameWindowCapturor:
    '''
    GameWindowCapturor for macOS
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame = None
        self.lock = threading.Lock()

        # 使用 mss 來擷取特定螢幕區域
        self.capture = mss.mss()

        # 動態取得 MapleStory 視窗位置
        self.region = get_window_region(self.cfg.game_window_title)
        if self.region is None:
            logger.error(f"找不到視窗: {self.cfg.game_window_title}")
            raise RuntimeError("找不到遊戲視窗")

        # 啟動擷取執行緒
        threading.Thread(target=self.start_capture, daemon=True).start()

        # 等待畫面初始化
        time.sleep(0.1)
        while self.frame is None:
            time.sleep(0.1)

        # 驗證解析度正確
        # if self.frame.shape[:2] != (self.region["height"], self.region["width"]):
        #     logger.error(f"擷取畫面大小錯誤: {self.frame.shape[:2]} (預期 {self.region['height'], self.region['width']})")
        #     raise RuntimeError("畫面擷取失敗，請確認 MapleStory 是否位於指定區域內。")

    def start_capture(self):
        '''
        開始螢幕擷取，並不斷更新 frame。
        '''
        while True:
            self.capture_frame()
            time.sleep(0.033)

    def capture_frame(self):
        '''
        捕捉當前遊戲區域畫面
        '''
        img = self.capture.grab(self.region)
        frame = np.array(img)
        with self.lock:
            self.frame = frame

    def get_frame(self):
        '''
        安全地獲取最新的螢幕畫面
        '''
        with self.lock:
            if self.frame is None:
                return None
            cv2.imwrite("debug_frame.png", self.frame)
            return cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)

    def on_closed(self):
        '''
        捕捉結束後的回調
        '''
        logger.warning("Capture session closed.")
        cv2.destroyAllWindows()
