# Standard Import
from argparse import Namespace
from copy import deepcopy
import sys
import threading
import time

# Pyside
from PySide6.QtCore import Qt, Signal, QObject

#  Local Import
from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.utils.logger import logger
from src.utils.common import load_yaml, screenshot
from src.input.CaptureFramePreprocessor import preprocess_capture_frame
from src.input.CaptureSource import (
    capture_profile_override,
    create_capture_source,
)
from src.input.KeyBoardListener import KeyBoardListener

class AutoBotController(QObject):
    '''
    AutoBot Controller server as a middleman between engine and UI
    '''
    debug_image_signal = Signal(object)
    route_map_viz_signal = Signal(object)
    start_pause_hotkey_signal = Signal()
    start_finished_signal = Signal(int)
    screenshot_hotkey_signal = Signal()
    record_hotkey_signal = Signal()
    close_hotkey_signal = Signal()

    CAPTURE_IDLE = "idle"
    CAPTURE_SCREENSHOT = "screenshot"
    CAPTURE_STARTING = "starting"
    CAPTURE_STOPPING = "stopping"
    CAPTURE_CLOSING = "closing"

    def __init__(self):
        """
        Init
        """
        super().__init__()
        self.ui = None
        # A GC573 DirectShow session must never be opened twice.  Serialize
        # the F2-only capture path with F1 start/pause and window shutdown.
        self._capture_lifecycle_lock = threading.RLock()
        self._capture_state_lock = threading.Lock()
        self._capture_state = self.CAPTURE_IDLE
        self._start_thread = None
        self._screenshot_thread = None
        self._prestart_capture = None
        self._prestart_release_failed = False
        self._closing = False

        # Init Auto Bot
        try:
            # Fake args to pass to AutoBot
            args = Namespace(
                disable_control=False,
                cfg="default",
                debug=False,
                record=False,
                is_ui=True,
                disable_viz=True,
                test_image='',
                init_state='',
            )
            self.auto_bot = MapleStoryAutoBot(args)
        except Exception as e:
            logger.error(f"MapleStoryAutoBot Init Failed: {e}")
            sys.exit(1)
        else:
            logger.info("MapleStoryAutoBot Init Successfully")

        # Update signal for debug window viz
        self.auto_bot.update_signals(self.debug_image_signal,
                                     self.route_map_viz_signal)

        # Monitor function keys
        self.kb_listener = KeyBoardListener(is_autobot=True)

    def update_signal(self, ui):
        '''
        Only called after UI init
        '''
        self.ui = ui
        self.debug_image_signal.connect(ui.update_debug_canvas)
        self.route_map_viz_signal.connect(ui.update_route_map_canvas)
        # pynput invokes callbacks on its own thread.  Queue every UI action
        # back to Qt's main thread before touching a widget.
        queued = Qt.ConnectionType.QueuedConnection
        self.start_pause_hotkey_signal.connect(
            ui.button_start_pause.click,
            queued,
        )
        self.start_finished_signal.connect(
            ui.finish_start_ui,
            queued,
        )
        self.screenshot_hotkey_signal.connect(
            ui.button_screenshot.click,
            queued,
        )
        self.record_hotkey_signal.connect(
            ui.button_record.click,
            queued,
        )
        self.close_hotkey_signal.connect(ui.request_close.emit, queued)
        # Register Function Key handler
        self.kb_listener.register_func_key_handler(
            'f1', self.start_pause_hotkey_signal.emit
        )
        self.kb_listener.register_func_key_handler(
            'f2', self.screenshot_hotkey_signal.emit
        )
        self.kb_listener.register_func_key_handler(
            'f3', self.record_hotkey_signal.emit
        )
        self.kb_listener.register_func_key_handler(
            'f12', self.close_hotkey_signal.emit
        )

    def _reserve_start(self):
        with self._capture_state_lock:
            self._refresh_stopping_state_locked()
            if self._closing:
                logger.warning("[start_bot] UI is closing")
                return -1
            if self._capture_state != self.CAPTURE_IDLE:
                logger.warning(
                    "[start_bot] Capture is busy: "
                    f"{self._capture_state}"
                )
                return -1
            if self._bot_thread_is_alive():
                if not getattr(self.auto_bot, "is_terminated", False):
                    logger.warning("[start_bot] AutoBot is already running")
                    return -1
                logger.warning(
                    "[start_bot] Previous capture session is still stopping"
                )
                self._capture_state = self.CAPTURE_STOPPING
                return -1
            if self._bot_capture_worker_is_alive():
                logger.warning(
                    "[start_bot] Previous capture session is still stopping"
                )
                self._capture_state = self.CAPTURE_STOPPING
                return -1
            self._capture_state = self.CAPTURE_STARTING
        return 0

    def _start_bot_reserved(self, cfg_path):
        with self._capture_lifecycle_lock:
            result = -1
            try:
                with self._capture_state_lock:
                    if self._closing:
                        logger.warning("[start_bot] Start cancelled during close")
                        return -1
                cfg = load_yaml(cfg_path)
                if self.auto_bot.load_config(cfg) != 0:
                    return -1 # Load fail
                with self._capture_state_lock:
                    if self._closing:
                        logger.warning("[start_bot] Start cancelled during close")
                        return -1
                self.auto_bot.start()
                result = 0
            except Exception as e:
                logger.error(f"[start_bot] {e}")
            finally:
                with self._capture_state_lock:
                    if self._closing:
                        self._capture_state = self.CAPTURE_CLOSING
                    elif (
                        getattr(self.auto_bot, "is_terminated", False)
                        and (
                            self._bot_thread_is_alive()
                            or self._bot_capture_worker_is_alive()
                        )
                    ):
                        self._capture_state = self.CAPTURE_STOPPING
                    else:
                        self._capture_state = self.CAPTURE_IDLE

        return result

    def start_bot(self, cfg_path):
        '''
        Start the bot engine threads synchronously.
        '''
        if self._reserve_start() != 0:
            return -1
        return self._start_bot_reserved(cfg_path)

    def _start_bot_worker(self, cfg_path):
        result = -1
        try:
            result = self._start_bot_reserved(cfg_path)
        finally:
            with self._capture_state_lock:
                self._start_thread = None
        self.start_finished_signal.emit(result)

    def start_bot_async(self, cfg_path):
        '''
        Start the bot without blocking Qt's event loop during model warmup.
        '''
        if self._reserve_start() != 0:
            return -1

        worker = threading.Thread(
            target=self._start_bot_worker,
            args=(cfg_path,),
            name="ui-f1-start",
            daemon=True,
        )
        with self._capture_state_lock:
            self._start_thread = worker
        try:
            worker.start()
        except Exception:
            with self._capture_state_lock:
                self._start_thread = None
                if not self._closing:
                    self._capture_state = self.CAPTURE_IDLE
            raise
        return 0

    def pause_bot(self):
        '''
        Gracefully pause in the engine
        '''
        # A running-bot screenshot writes two large PNGs in its worker.  Let it
        # finish before stopping the frame producer it is reading from.
        while True:
            if not self._wait_for_screenshot_worker():
                logger.error("[pause_bot] Screenshot worker did not stop")
                return -1
            retry = False
            with self._capture_lifecycle_lock:
                with self._capture_state_lock:
                    if self._closing:
                        return -1
                    if self._capture_state == self.CAPTURE_SCREENSHOT:
                        retry = True
                    elif self._capture_state in {
                        self.CAPTURE_STARTING,
                        self.CAPTURE_CLOSING,
                    }:
                        logger.warning(
                            "[pause_bot] Capture transition is busy: "
                            f"{self._capture_state}"
                        )
                        return -1
                    else:
                        self._capture_state = self.CAPTURE_STOPPING

                if retry:
                    continue

                try:
                    self.auto_bot.pause()
                finally:
                    with self._capture_state_lock:
                        if self._closing:
                            self._capture_state = self.CAPTURE_CLOSING
                        elif (
                            self._bot_thread_is_alive()
                            or self._bot_capture_worker_is_alive()
                        ):
                            self._capture_state = self.CAPTURE_STOPPING
                        else:
                            self._capture_state = self.CAPTURE_IDLE
                return 0

    def _bot_thread_is_alive(self):
        thread = getattr(self.auto_bot, "thread_auto_bot", None)
        return thread is not None and thread.is_alive()

    @staticmethod
    def _capture_worker_is_alive(capture):
        if capture is None:
            return False
        for attribute in ("_capture_thread", "thread"):
            worker = getattr(capture, attribute, None)
            is_alive = getattr(worker, "is_alive", None)
            if not callable(is_alive):
                continue
            try:
                result = is_alive()
            except Exception:
                return True
            if isinstance(result, bool) and result:
                return True
        return False

    def _bot_capture_worker_is_alive(self):
        return self._capture_worker_is_alive(
            getattr(self.auto_bot, "capture", None)
        )

    def _refresh_stopping_state_locked(self):
        """Move STOPPING back to IDLE only after every producer has exited."""
        if self._capture_state != self.CAPTURE_STOPPING or self._closing:
            return
        if (
            self._bot_thread_is_alive()
            or self._bot_capture_worker_is_alive()
            or self._prestart_release_failed
            or self._capture_worker_is_alive(self._prestart_capture)
        ):
            return
        self._prestart_capture = None
        self._capture_state = self.CAPTURE_IDLE

    def _wait_for_screenshot_worker(self, timeout=10.0):
        with self._capture_state_lock:
            worker = self._screenshot_thread
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=timeout)
            return not worker.is_alive()
        return True

    @staticmethod
    def _wait_for_capture_frame(capture, cfg):
        capture_card_cfg = cfg.get("capture_card", {})
        if not isinstance(capture_card_cfg, dict):
            capture_card_cfg = {}
        try:
            timeout = max(
                0.0,
                float(capture_card_cfg.get("startup_timeout", 3.0)),
            )
            poll_interval = max(
                0.001,
                min(
                    0.05,
                    float(capture_card_cfg.get("read_retry_interval", 0.01)),
                ),
            )
        except (TypeError, ValueError):
            timeout = 3.0
            poll_interval = 0.01

        deadline = time.monotonic() + timeout
        while True:
            frame = capture.get_frame()
            if frame is not None:
                return frame
            if time.monotonic() >= deadline:
                return None
            time.sleep(poll_interval)

    def _take_screenshot_without_starting_bot(self, cfg):
        """Capture one read-only frame without creating any input objects."""
        capture = None
        release_failed = False
        try:
            capture = create_capture_source(cfg)
            with self._capture_state_lock:
                self._prestart_capture = capture
                self._prestart_release_failed = False
            frame = self._wait_for_capture_frame(capture, cfg)
            if frame is None:
                raise RuntimeError("capture source did not provide a frame")

            img_frame, _ = preprocess_capture_frame(
                frame,
                cfg,
                window_title=getattr(capture, "window_title", ""),
                capture_profile=capture_profile_override(capture),
            )
            screenshot(img_frame, "img_frame")
            screenshot(frame, "frame")
            logger.info(
                "[take_screenshot] Saved a frame while AutoBot control is inactive"
            )
            return 0
        except Exception as exc:
            logger.error(f"[take_screenshot] Failed: {exc}")
            return -1
        finally:
            if capture is not None:
                try:
                    capture.stop()
                except Exception as exc:
                    release_failed = True
                    logger.warning(
                        "[take_screenshot] Failed to release capture source: "
                        f"{exc}"
                    )
                finally:
                    with self._capture_state_lock:
                        self._prestart_release_failed = release_failed
                        if (
                            not release_failed
                            and not self._capture_worker_is_alive(capture)
                        ):
                            self._prestart_capture = None

    def _take_screenshot_worker(self, cfg):
        try:
            with self._capture_lifecycle_lock:
                with self._capture_state_lock:
                    if self._closing:
                        return

                if self._bot_thread_is_alive():
                    if getattr(self.auto_bot, "is_terminated", False):
                        logger.warning(
                            "[take_screenshot] AutoBot capture is still stopping"
                        )
                        return
                    self.auto_bot.screenshot_img_frame()
                    return

                if self._bot_capture_worker_is_alive():
                    logger.warning(
                        "[take_screenshot] Previous capture is still stopping"
                    )
                    return

                if not isinstance(cfg, dict):
                    logger.error(
                        "[take_screenshot] Failed: UI capture config is unavailable"
                    )
                    return
                self._take_screenshot_without_starting_bot(cfg)
        finally:
            with self._capture_state_lock:
                if self._closing:
                    self._capture_state = self.CAPTURE_CLOSING
                elif (
                    self._prestart_release_failed
                    or self._capture_worker_is_alive(self._prestart_capture)
                    or (
                        getattr(self.auto_bot, "is_terminated", False)
                        and (
                            self._bot_thread_is_alive()
                            or self._bot_capture_worker_is_alive()
                        )
                    )
                ):
                    self._capture_state = self.CAPTURE_STOPPING
                else:
                    self._prestart_capture = None
                    self._capture_state = self.CAPTURE_IDLE
                self._screenshot_thread = None

    def take_screenshot(self, cfg=None):
        '''
        Called when user press screenshot button
        '''
        if cfg is None and self.ui is not None:
            cfg = deepcopy(self.ui.cfg)
        elif isinstance(cfg, dict):
            cfg = deepcopy(cfg)

        with self._capture_state_lock:
            self._refresh_stopping_state_locked()
            if self._closing:
                logger.warning("[take_screenshot] UI is closing")
                return -1
            if self._capture_state != self.CAPTURE_IDLE:
                logger.warning(
                    "[take_screenshot] Capture is busy: "
                    f"{self._capture_state}"
                )
                return -1
            if (
                getattr(self.auto_bot, "is_terminated", False)
                and (
                    self._bot_thread_is_alive()
                    or self._bot_capture_worker_is_alive()
                )
            ):
                self._capture_state = self.CAPTURE_STOPPING
                logger.warning(
                    "[take_screenshot] AutoBot capture is still stopping"
                )
                return -1

            self._capture_state = self.CAPTURE_SCREENSHOT
            worker = threading.Thread(
                target=self._take_screenshot_worker,
                args=(cfg,),
                name="ui-f2-screenshot",
                daemon=True,
            )
            self._screenshot_thread = worker
            try:
                worker.start()
            except Exception:
                self._screenshot_thread = None
                self._capture_state = self.CAPTURE_IDLE
                raise
        return 0

    def start_recording(self):
        '''
        Called when user press start record button
        '''
        self.auto_bot.start_record()

    def stop_recording(self):
        '''
        Called when user press stop record button
        '''
        self.auto_bot.stop_record()

    def terminate_bot(self):
        '''
        Called when user stop bot or close UI
        '''
        # Close the admission gate first.  A queued F1/F2 event must not start
        # another capture after shutdown has begun.
        with self._capture_state_lock:
            self._closing = True
            self._capture_state = self.CAPTURE_CLOSING

        listener = getattr(self, "kb_listener", None)
        if listener is not None:
            listener.stop()

        if not self._wait_for_screenshot_worker():
            logger.error(
                "[terminate_bot] Screenshot worker did not stop within 10 seconds"
            )
            # Input safety takes priority during application shutdown.  The
            # screenshot worker is daemonized and the closing gate prevents
            # it from starting another session.
            self.auto_bot.terminate_threads()
            return
        with self._capture_lifecycle_lock:
            # Terminate all bot threads
            self.auto_bot.terminate_threads()
            capture = self._prestart_capture
            if capture is not None:
                try:
                    capture.stop()
                except Exception as exc:
                    logger.warning(
                        "[terminate_bot] Failed to release F2 capture source: "
                        f"{exc}"
                    )

    def enable_bot_viz(self):
        '''
        Called when user switch to viz tab
        '''
        self.auto_bot.enable_viz()

    def disable_bot_viz(self):
        '''
        Called when user switch from viz tab
        '''
        self.auto_bot.disable_viz()
