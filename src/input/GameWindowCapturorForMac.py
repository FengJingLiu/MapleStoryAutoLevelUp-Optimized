# Standard import
import time
import threading

# Library import
import mss
import cv2
import numpy as np
import Quartz

# Local import
from src.utils.logger import logger

def get_window_title(token, exact_match=False):
    '''
    Get a window title that matches token.

    Not tied to "MapleStory Worlds" - any program window can be targeted.
    When exact_match is True, only a title equal to token is returned.
    '''
    if not token:
        return None
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID
    )
    # Get all exist windows
    for window in window_list:
        title = window.get(Quartz.kCGWindowName, '')
        if not title:
            continue
        if exact_match:
            if token == title:
                return title
        elif token in title:
            return title
    return None

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
        self.last_frame_time = 0.0
        self.is_closed = False
        self.lock = threading.Lock()
        self.is_terminated = False

        self.window_title = get_window_title(
            cfg["game_window"]["title"],
            exact_match=cfg["game_window"].get("exact_match", False),
        )
        if self.window_title is None:
            logger.error(
                f"[GameWindowCapturor] Unable to find window titles that contain {cfg['game_window']['title']}"
            )
            return -1

        self.fps = 0
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.t_last_run = 0.0

        # 使用 mss 來擷取特定螢幕區域
        self.capture = mss.mss()

        # Get game window region
        self.update_window_region()

        # start game window capture
        threading.Thread(target=self.start_capture, daemon=True).start()

        # Wait frame init
        time.sleep(0.1)
        while self.frame is None:
            self.limit_fps()

    def start_capture(self):
        '''
        開始螢幕擷取，並不斷更新 frame。
        '''
        while not self.is_terminated:
            # Update self.region
            self.update_window_region()

            # Update self.frame
            self.capture_frame()

            # Limit FPS to save systme resources
            self.limit_fps()

    def stop(self):
        '''
        Stop capturing thread
        '''
        self.is_terminated = True
        with self.lock:
            self.frame = None
            self.is_closed = True
        logger.info("[GameWindowCapturor] Terminated")

    def update_window_region(self):
        '''
        Update window region
        '''
        self.region = get_window_region(self.window_title)
        if self.region is None:
            text = f"Cannot find window: {self.window_title}"
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
            self.last_frame_time = time.monotonic()
            self.is_closed = False

    def get_frame_snapshot(self):
        '''
        安全地獲取最新的螢幕畫面
        '''
        with self.lock:
            frame_timeout = float(
                self.cfg.get("game_window", {}).get("frame_timeout", 1.0)
            )
            if self.frame is None or self.is_closed or (
                self.last_frame_time > 0
                and time.monotonic() - self.last_frame_time > frame_timeout
            ):
                return None, None
            # cv2.imwrite("debug_frame.png", self.frame)
            return (
                cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR),
                self.last_frame_time,
            )

    def get_frame(self):
        '''Safely get the latest game window frame.'''
        frame, _ = self.get_frame_snapshot()
        return frame

    def on_closed(self):
        '''
        捕捉結束後的回調
        '''
        with self.lock:
            self.frame = None
            self.is_closed = True
        logger.warning("Capture session closed.")
        cv2.destroyAllWindows()

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")
