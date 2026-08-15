'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import threading

# Libarary Import
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import cv2

# local import
from src.utils.logger import logger
from src.utils.common import get_game_window_title_by_token, load_image, resize_window
from src.input.CaptureFramePreprocessor import (
    get_capture_resize_size,
    resolve_capture_profile,
)

class GameWindowCapturor:
    '''
    GameWindowCapturor
    '''
    def __init__(self, cfg, test_image_name = None):
        self.cfg = cfg
        self.frame = None
        self.last_frame_time = 0.0
        self.is_closed = False
        self.is_static_frame = test_image_name is not None
        self.lock = threading.Lock()
        self.is_terminated = False
        self.fps = 0
        self.fps_limit = cfg["system"]["fps_limit_window_capturor"]
        self.t_last_run = 0.0
        self.capture_control = None
        self.window_title = ""

        # If use test image as input, disable the whole capture thread
        if test_image_name is not None:
            self.frame = load_image(f"test/{test_image_name}.png")
            return

        # Get target program / game window title
        game_window_cfg = cfg["game_window"]
        self.window_title = get_game_window_title_by_token(
            game_window_cfg["title"],
            exact_match=game_window_cfg.get("exact_match", False),
        )

        if self.window_title is None:
            raise RuntimeError(
                f"[GameWindowCapturor] Unable to find window title containing: {game_window_cfg['title']}"
            )
        else:
            logger.info(f"[GameWindowCapturor] Found target window title: {self.window_title}")

        # Only force-resize when enabled. Non-MapleStory programs often should
        # not be resized, so this is configurable via game_window.auto_resize.
        if game_window_cfg.get("auto_resize", True):
            resize_width, resize_height = get_capture_resize_size(
                game_window_cfg, self.window_title
            )
            actual_size = resize_window(
                self.window_title,
                width=resize_width,
                height=resize_height,
            )
            logger.info(
                "[GameWindowCapturor] "
                f"Resize profile={resolve_capture_profile(game_window_cfg, self.window_title)} "
                f"requested_outer_size={(resize_width, resize_height)} "
                f"actual_outer_size={actual_size}"
            )

        # Create capture handler
        self.capture = WindowsCapture(window_name=self.window_title)
        self.capture.event(self.on_frame_arrived)
        self.capture.event(self.on_closed)

        # Start capturing thread
        self.capture_control = self.capture.start_free_threaded()

        logger.info("[GameWindowCapturor] Init done")

    def on_frame_arrived(self, frame: Frame,
                         capture_control: InternalCaptureControl):
        '''
        Frame arrived callback: store frame into buffer with lock.
        '''
        with self.lock:
            self.frame = frame.frame_buffer
            self.last_frame_time = time.monotonic()
            self.is_closed = False
        self.limit_fps()

    def on_closed(self):
        '''
        Capture closed callback.
        '''
        with self.lock:
            self.frame = None
            self.is_closed = True
        logger.warning("[GameWindowCapturor] closed.")
        cv2.destroyAllWindows()

    def get_frame_snapshot(self):
        '''
        Atomically get the latest frame and its capture timestamp.

        Keeping the image and timestamp under the same lock prevents callers
        from pairing an older image with a newer ``last_frame_time`` value.
        '''
        with self.lock:
            frame_timeout = float(
                self.cfg.get("game_window", {}).get("frame_timeout", 1.0)
            )
            if self.frame is None or self.is_closed or (
                not self.is_static_frame
                and self.last_frame_time > 0
                and time.monotonic() - self.last_frame_time > frame_timeout
            ):
                return None, None
            return (
                cv2.cvtColor(self.frame, cv2.COLOR_BGRA2BGR),
                self.last_frame_time,
            )

    def get_frame(self):
        '''
        Safely get latest game window frame.
        '''
        frame, _ = self.get_frame_snapshot()
        return frame

    def stop(self):
        '''
        Stop capturing thread
        '''
        if self.capture_control is not None:
            self.capture_control.stop()
        with self.lock:
            self.frame = None
            self.is_closed = True
        logger.info("[GameWindowCapturor] Terminated")

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
