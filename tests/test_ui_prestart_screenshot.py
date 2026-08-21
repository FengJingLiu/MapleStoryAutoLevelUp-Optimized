import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal, Slot

from src.ui.AutoBotController import AutoBotController
from src.ui.ui import MainWindow


class _Capture:
    capture_profile = "capture_card"
    window_title = "AVerMedia GC573 1 Capture"

    def __init__(self, frame):
        self.frame = frame
        self.stop = Mock()

    def get_frame(self):
        return self.frame


class _Button(QObject):
    clicked = Signal()

    @Slot()
    def click(self):
        self.clicked.emit()


class _FakeUi(QObject):
    request_close = Signal()

    def __init__(self):
        super().__init__()
        self.button_start_pause = _Button()
        self.button_screenshot = _Button()
        self.button_record = _Button()

    @Slot(object)
    def update_debug_canvas(self, _image):
        pass

    @Slot(object)
    def update_route_map_canvas(self, _image):
        pass

    @Slot(int)
    def finish_start_ui(self, _result):
        pass


def _controller(*, running=False):
    controller = AutoBotController.__new__(AutoBotController)
    thread = Mock()
    thread.is_alive.return_value = running
    controller.auto_bot = SimpleNamespace(
        thread_auto_bot=thread if running else None,
        is_terminated=False,
        screenshot_img_frame=Mock(),
        start=Mock(),
        load_config=Mock(return_value=0),
        pause=Mock(),
        terminate_threads=Mock(),
        capture=None,
    )
    controller.ui = None
    controller.kb_listener = Mock()
    controller._capture_lifecycle_lock = threading.RLock()
    controller._capture_state_lock = threading.Lock()
    controller._capture_state = controller.CAPTURE_IDLE
    controller._start_thread = None
    controller._screenshot_thread = None
    controller._prestart_capture = None
    controller._prestart_release_failed = False
    controller._closing = False
    return controller


class AutoBotControllerPrestartScreenshotTests(unittest.TestCase):
    def test_f2_hotkey_is_queued_back_to_qt_main_thread(self):
        app = QCoreApplication.instance() or QCoreApplication([])
        controller = AutoBotController.__new__(AutoBotController)
        QObject.__init__(controller)
        controller.kb_listener = Mock()
        ui = _FakeUi()
        called_on = []
        ui.button_screenshot.clicked.connect(
            lambda: called_on.append(threading.get_ident())
        )

        controller.update_signal(ui)
        handlers = {
            registered.args[0]: registered.args[1]
            for registered in (
                controller.kb_listener.register_func_key_handler.call_args_list
            )
        }
        hotkey_thread = threading.Thread(target=handlers["f2"])
        hotkey_thread.start()
        hotkey_thread.join(timeout=1)

        self.assertEqual(called_on, [])
        app.processEvents()
        self.assertEqual(called_on, [threading.get_ident()])

    def test_f2_before_f1_captures_native_frame_without_starting_bot(self):
        controller = _controller()
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        capture = _Capture(frame)
        cfg = {
            "capture": {"source": "directshow"},
            "capture_card": {"startup_timeout": 0},
            "game_window": {},
        }

        with (
            patch(
                "src.ui.AutoBotController.create_capture_source",
                return_value=capture,
            ) as create_capture,
            patch(
                "src.ui.AutoBotController.preprocess_capture_frame",
                return_value=(frame, {"output_size": frame.shape[:2]}),
            ) as preprocess,
            patch("src.ui.AutoBotController.screenshot") as save_screenshot,
        ):
            result = controller.take_screenshot(cfg)
            controller._wait_for_screenshot_worker()

        self.assertEqual(result, 0)
        create_capture.assert_called_once()
        self.assertEqual(create_capture.call_args.args[0], cfg)
        preprocess.assert_called_once()
        self.assertIs(preprocess.call_args.args[0], frame)
        self.assertEqual(
            save_screenshot.call_args_list,
            [call(frame, "img_frame"), call(frame, "frame")],
        )
        capture.stop.assert_called_once_with()
        controller.auto_bot.start.assert_not_called()
        controller.auto_bot.screenshot_img_frame.assert_not_called()

    def test_running_bot_uses_existing_frame_and_does_not_open_second_card(self):
        controller = _controller(running=True)

        with patch(
            "src.ui.AutoBotController.create_capture_source"
        ) as create_capture:
            result = controller.take_screenshot({"capture": {}})
            controller._wait_for_screenshot_worker()

        self.assertEqual(result, 0)
        controller.auto_bot.screenshot_img_frame.assert_called_once_with()
        create_capture.assert_not_called()

    def test_repeated_start_does_not_mark_healthy_bot_as_stopping(self):
        controller = _controller(running=True)

        result = controller.start_bot("unused.yaml")

        self.assertEqual(result, -1)
        self.assertEqual(controller._capture_state, controller.CAPTURE_IDLE)
        controller.auto_bot.start.assert_not_called()

    def test_async_start_returns_while_model_load_continues(self):
        controller = _controller()
        QObject.__init__(controller)
        load_entered = threading.Event()
        release_load = threading.Event()

        def load_config(_cfg):
            load_entered.set()
            release_load.wait(timeout=2)
            return 0

        controller.auto_bot.load_config.side_effect = load_config

        with patch(
            "src.ui.AutoBotController.load_yaml",
            return_value={"capture": {"source": "directshow"}},
        ):
            self.assertEqual(controller.start_bot_async("unused.yaml"), 0)
            self.assertTrue(load_entered.wait(timeout=1))
            self.assertEqual(
                controller._capture_state,
                controller.CAPTURE_STARTING,
            )
            worker = controller._start_thread
            self.assertTrue(worker.is_alive())
            self.assertEqual(controller.start_bot_async("unused.yaml"), -1)

            release_load.set()
            worker.join(timeout=2)

        controller.auto_bot.start.assert_called_once_with()
        self.assertEqual(controller._capture_state, controller.CAPTURE_IDLE)

    def test_capture_error_releases_device_and_does_not_start_controls(self):
        controller = _controller()
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = _Capture(frame)
        cfg = {
            "capture_card": {"startup_timeout": 0},
            "game_window": {},
        }

        with (
            patch(
                "src.ui.AutoBotController.create_capture_source",
                return_value=capture,
            ),
            patch(
                "src.ui.AutoBotController.preprocess_capture_frame",
                side_effect=ValueError("bad frame"),
            ),
            patch("src.ui.AutoBotController.screenshot") as save_screenshot,
        ):
            result = controller.take_screenshot(cfg)
            controller._wait_for_screenshot_worker()

        self.assertEqual(result, 0)
        capture.stop.assert_called_once_with()
        save_screenshot.assert_not_called()
        controller.auto_bot.start.assert_not_called()

    def test_f1_start_is_rejected_until_prestart_f2_releases_card(self):
        controller = _controller()
        screenshot_entered = threading.Event()
        release_screenshot = threading.Event()
        start_called = threading.Event()

        def take_idle_frame(_cfg):
            screenshot_entered.set()
            release_screenshot.wait(timeout=2)
            return 0

        controller._take_screenshot_without_starting_bot = take_idle_frame
        controller.auto_bot.start.side_effect = start_called.set

        with patch(
            "src.ui.AutoBotController.load_yaml",
            return_value={"capture": {"source": "directshow"}},
        ):
            self.assertEqual(
                controller.take_screenshot(
                    {"capture": {"source": "directshow"}}
                ),
                0,
            )
            self.assertTrue(screenshot_entered.wait(timeout=1))

            self.assertEqual(controller.start_bot("unused.yaml"), -1)
            self.assertFalse(start_called.is_set())

            release_screenshot.set()
            controller._wait_for_screenshot_worker()
            self.assertEqual(controller.start_bot("unused.yaml"), 0)

        self.assertTrue(start_called.is_set())

    def test_stop_timeout_blocks_f1_until_capture_worker_exits(self):
        controller = _controller()
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        capture = _Capture(frame)
        capture_worker = Mock()
        capture_worker.is_alive.return_value = True
        capture._capture_thread = capture_worker
        cfg = {
            "capture_card": {"startup_timeout": 0},
            "game_window": {},
        }

        with (
            patch(
                "src.ui.AutoBotController.create_capture_source",
                return_value=capture,
            ),
            patch(
                "src.ui.AutoBotController.preprocess_capture_frame",
                return_value=(frame, {}),
            ),
            patch("src.ui.AutoBotController.screenshot"),
            patch(
                "src.ui.AutoBotController.load_yaml",
                return_value=cfg,
            ),
        ):
            self.assertEqual(controller.take_screenshot(cfg), 0)
            controller._wait_for_screenshot_worker()
            self.assertEqual(
                controller._capture_state,
                controller.CAPTURE_STOPPING,
            )
            self.assertEqual(controller.start_bot("unused.yaml"), -1)
            controller.auto_bot.start.assert_not_called()

            capture_worker.is_alive.return_value = False
            self.assertEqual(controller.start_bot("unused.yaml"), 0)

        controller.auto_bot.start.assert_called_once_with()

    def test_close_rejects_queued_f1_and_f2_while_screenshot_finishes(self):
        controller = _controller()
        screenshot_entered = threading.Event()
        release_screenshot = threading.Event()

        def take_idle_frame(_cfg):
            screenshot_entered.set()
            release_screenshot.wait(timeout=2)
            return 0

        controller._take_screenshot_without_starting_bot = take_idle_frame
        self.assertEqual(controller.take_screenshot({"capture": {}}), 0)
        self.assertTrue(screenshot_entered.wait(timeout=1))

        terminate_thread = threading.Thread(target=controller.terminate_bot)
        terminate_thread.start()
        for _ in range(100):
            with controller._capture_state_lock:
                if controller._closing:
                    break
            threading.Event().wait(0.01)

        self.assertEqual(controller.start_bot("unused.yaml"), -1)
        self.assertEqual(controller.take_screenshot({"capture": {}}), -1)

        release_screenshot.set()
        terminate_thread.join(timeout=2)
        self.assertFalse(terminate_thread.is_alive())
        controller.auto_bot.start.assert_not_called()

    def test_window_close_stops_global_hotkey_listener(self):
        controller = _controller()

        controller.terminate_bot()

        controller.auto_bot.terminate_threads.assert_called_once_with()
        controller.kb_listener.stop.assert_called_once_with()


class MainWindowPrestartScreenshotTests(unittest.TestCase):
    def test_ui_passes_current_capture_settings_without_mutating_saved_cfg(self):
        window = MainWindow.__new__(MainWindow)
        window.cfg = {
            "capture": {"source": "directshow"},
            "game_window": {
                "title": "old title",
                "exact_match": False,
                "auto_resize": True,
            },
        }
        window.target_window_combo = SimpleNamespace(
            currentText=lambda: "new title"
        )
        window.checkbox_exact_match = SimpleNamespace(isChecked=lambda: True)
        window.checkbox_auto_resize = SimpleNamespace(isChecked=lambda: False)
        window.controller = SimpleNamespace(take_screenshot=Mock())

        window.toggle_screenshot_ui()

        snapshot = window.controller.take_screenshot.call_args.args[0]
        self.assertEqual(snapshot["capture"]["source"], "directshow")
        self.assertEqual(snapshot["game_window"]["title"], "new title")
        self.assertTrue(snapshot["game_window"]["exact_match"])
        self.assertFalse(snapshot["game_window"]["auto_resize"])
        self.assertEqual(window.cfg["game_window"]["title"], "old title")


if __name__ == "__main__":
    unittest.main()
