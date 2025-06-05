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

class GameWindowCapturor:
    '''
    GameWindowCapturor
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.window_title = cfg.game_window_title
        self.frame = None
        self.lock = threading.Lock()

        self.capture = WindowsCapture(window_name=self.window_title)
        self.capture.event(self.on_frame_arrived)
        self.capture.event(self.on_closed)

        # Start capturing thread, blocking
        threading.Thread(target=self.capture.start, daemon=True).start()

        # Wait 0.1 second for frame to load
        time.sleep(0.1)

        # Check is game windows size is as expected
        if self.frame.shape[:2] != cfg.window_size:
            logger.error(f"Invalid window size: {self.frame.shape[:2]} (expected {cfg.window_size})")
            logger.error("Please use windowed mode & smallest resolution.")
            raise RuntimeError(f"Unexpected window size: {self.frame.shape[:2]}")

    def on_frame_arrived(self, frame: Frame,
                         capture_control: InternalCaptureControl):
        '''
        Frame arrived callback: store frame into buffer with lock.
        '''
        with self.lock:
            self.frame = frame.frame_buffer
        time.sleep(0.033)  # Cap FPS to ~30

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

