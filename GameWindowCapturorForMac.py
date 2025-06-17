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
    # Get all exist windows
    all_titles = []
    for window in window_list:
        title = window.get(Quartz.kCGWindowName, '')
        owner = window.get(Quartz.kCGWindowOwnerName, '')
        if title:
            all_titles.append(f"{title} (Owner: {owner})")
    logger.debug(f"all_titles: {all_titles}")
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

        # Get game window region
        self.update_window_region()

        # start game window capture
        threading.Thread(target=self.start_capture, daemon=True).start()

        # Wait frame init
        time.sleep(0.1)
        while self.frame is None:
            time.sleep(0.1)


    def start_capture(self):
        '''
        開始螢幕擷取，並不斷更新 frame。
        '''
        while True:
            # Update self.region
            self.update_window_region()

            # Update self.frame
            self.capture_frame()

            # Cap FPS to 30
            time.sleep(0.033)

    def update_window_region(self):
        '''
        Update window region
        '''
        self.region = get_window_region(self.cfg.game_window_title)
        if self.region is None:
            text = f"Cannot find window: {self.cfg.game_window_title}"
            logger.error(text)
            raise RuntimeError(text)

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
            # cv2.imwrite("debug_frame.png", self.frame)
            return cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)

    def on_closed(self):
        '''
        捕捉結束後的回調
        '''
        logger.warning("Capture session closed.")
        cv2.destroyAllWindows()
