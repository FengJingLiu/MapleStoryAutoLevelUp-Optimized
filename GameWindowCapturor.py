'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import threading

from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import cv2

# local import
from logger import logger
from util import is_img_16_to_9

class GameWindowCapturor:
    '''
    GameWindowCapturor
    '''
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.window_title = cfg["game_window"]["title"]
        self.frame = None
        self.lock = threading.Lock()

        self.capture = WindowsCapture(window_name=self.window_title)
        self.capture.event(self.on_frame_arrived)
        self.capture.event(self.on_closed)

        self.fps = 0
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.t_last_run = 0.0

        # Start capturing thread, blocking
        threading.Thread(target=self.capture.start, daemon=True).start()

        # Wait 0.1 second for frame to load
        time.sleep(0.1)

        # Check is game windows size is as expected
        if args.aux:
            # Check is game windows ratio is 16:9
            if not is_img_16_to_9(self.frame, cfg):
                logger.error(f"Invalid window ratio: {self.frame.shape[:2]} (expected 16:9 window)")
                logger.error("Please use windowed mode & smallest resolution.")
                raise RuntimeError(f"Unexpected window ratio: {self.frame.shape[:2]}")
        else:
            if self.frame.shape[:2] != cfg["game_window"]["size"]:
                logger.error(f"Invalid window size: {self.frame.shape[:2]} (expected {cfg['game_window']['size']})")
                logger.error("Please use windowed mode & smallest resolution.")
                raise RuntimeError(f"Unexpected window size: {self.frame.shape[:2]}")

    def on_frame_arrived(self, frame: Frame,
                         capture_control: InternalCaptureControl):
        '''
        Frame arrived callback: store frame into buffer with lock.
        '''
        with self.lock:
            self.frame = frame.frame_buffer
        self.limit_fps()

    def on_closed(self):
        '''
        Capture closed callback.
        '''
        logger.warning("Capture session closed.")
        cv2.destroyAllWindows()

    def get_frame(self):
        '''
        Safely get latest game window frame.
        '''
        with self.lock:
            if self.frame is None:
                return None
            return cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR)

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
