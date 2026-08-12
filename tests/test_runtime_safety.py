import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.input.GameWindowCapturor import GameWindowCapturor
from src.utils.common import draw_circle, draw_line, draw_rectangle, draw_text


class DebugDrawingTests(unittest.TestCase):
    def test_debug_helpers_ignore_missing_canvas(self):
        draw_rectangle(None, (0, 0), (10, 20), (0, 0, 0), "test")
        draw_circle(None, (0, 0), 5, (0, 0, 0), 1)
        draw_line(None, (0, 0), (1, 1), (0, 0, 0), 1)
        draw_text(None, "test", (0, 0), 0, 1, (0, 0, 0), 1)


class AutoBotLifecycleTests(unittest.TestCase):
    def test_start_rejects_duplicate_main_loop(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.thread_auto_bot = Mock()
        bot.thread_auto_bot.is_alive.return_value = True

        with self.assertRaisesRegex(RuntimeError, "already running"):
            bot.start()

    def test_terminate_stops_components_and_releases_keys(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.is_terminated = False
        bot.thread_auto_bot = None
        bot.kb = SimpleNamespace(
            is_terminated=False,
            release_all_key=Mock(),
        )
        bot.capture = SimpleNamespace(stop=Mock())
        bot.health_monitor = SimpleNamespace(stop=Mock())

        bot.terminate_threads()

        self.assertTrue(bot.is_terminated)
        self.assertTrue(bot.kb.is_terminated)
        bot.kb.release_all_key.assert_called_once_with()
        bot.capture.stop.assert_called_once_with()
        bot.health_monitor.stop.assert_called_once_with()

    def test_capture_failure_does_not_open_esp32_input(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.thread_auto_bot = None
        bot.is_terminated = False
        bot.args = SimpleNamespace(test_image="", init_state="")
        bot.cfg = {}
        bot.is_disable_control = False

        with patch(
            "src.engine.MapleStoryAutoLevelUp.GameWindowCapturor",
            side_effect=RuntimeError("capture unavailable"),
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.KeyBoardController"
        ) as keyboard_controller:
            with self.assertRaisesRegex(RuntimeError, "capture unavailable"):
                bot.start()

        keyboard_controller.assert_not_called()
        self.assertTrue(bot.is_terminated)

    def test_keyboard_failure_stops_the_new_capture_session(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.thread_auto_bot = None
        bot.is_terminated = False
        bot.args = SimpleNamespace(test_image="", init_state="")
        bot.cfg = {}
        bot.is_disable_control = False
        capture = Mock()

        with patch(
            "src.engine.MapleStoryAutoLevelUp.GameWindowCapturor",
            return_value=capture,
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.KeyBoardController",
            side_effect=ConnectionError("ESP32 unavailable"),
        ):
            with self.assertRaisesRegex(ConnectionError, "ESP32 unavailable"):
                bot.start()

        capture.stop.assert_called_once_with()
        self.assertTrue(bot.is_terminated)

    def test_main_loop_failure_stops_esp32_keyboard_and_capture(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.is_terminated = False
        bot.thread_auto_bot = None
        bot.kb = SimpleNamespace(is_terminated=False, stop=Mock())
        bot.capture = SimpleNamespace(stop=Mock())
        bot.health_monitor = SimpleNamespace(stop=Mock())
        bot.run_once = Mock(side_effect=RuntimeError("vision failed"))
        bot.is_frame_done = True

        with patch(
            "src.engine.MapleStoryAutoLevelUp.is_mac", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, "vision failed"):
                bot.loop()

        self.assertTrue(bot.is_terminated)
        bot.kb.stop.assert_called_once_with()
        bot.capture.stop.assert_called_once_with()
        bot.health_monitor.stop.assert_called_once_with()

    def test_remote_mode_skips_party_mouse_workflow_entirely(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "esp32_hid": {"remote_target": True},
            "key": {"party": "p"},
        }

        with patch(
            "src.engine.MapleStoryAutoLevelUp.press_key"
        ) as press, patch.object(bot, "get_img_frame") as get_frame:
            self.assertFalse(bot.ensure_is_in_party())

        press.assert_not_called()
        get_frame.assert_not_called()

    def test_remote_loop_does_not_activate_a_window_or_start_party_workflow(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {"esp32_hid": {"remote_target": True}}
        bot.is_terminated = True
        bot.thread_auto_bot = None
        bot.kb = SimpleNamespace(is_terminated=False)
        bot.capture = SimpleNamespace(window_title="PotPlayer")
        bot.ensure_is_in_party = Mock()
        bot.terminate_threads = Mock()

        with patch(
            "src.engine.MapleStoryAutoLevelUp.is_mac", return_value=False
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.activate_game_window"
        ) as activate:
            bot.loop()

        activate.assert_not_called()
        bot.ensure_is_in_party.assert_not_called()

    def test_capture_loss_releases_once_and_only_that_suspension_auto_resumes(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.kb = SimpleNamespace(
            set_command=Mock(),
            set_capture_available=Mock(),
        )
        bot._input_suspended_for_capture = False

        self.assertTrue(bot.suspend_input_for_capture_loss())
        self.assertFalse(bot.suspend_input_for_capture_loss())
        bot.kb.set_command.assert_called_once_with("none none none")
        bot.kb.set_capture_available.assert_called_once_with(False)

        self.assertTrue(bot.resume_input_after_capture())
        self.assertFalse(bot.resume_input_after_capture())
        self.assertEqual(
            bot.kb.set_capture_available.call_args_list,
            [call(False), call(True)],
        )


class WindowCaptureTests(unittest.TestCase):
    def test_missing_window_is_not_resized(self):
        cfg = {
            "system": {"fps_limit_window_capturor": 15},
            "game_window": {"title": "Missing MapleStory Window"},
        }

        with patch(
            "src.input.GameWindowCapturor.get_game_window_title_by_token",
            return_value=None,
        ), patch("src.input.GameWindowCapturor.resize_window") as resize:
            with self.assertRaisesRegex(RuntimeError, "Unable to find window"):
                GameWindowCapturor(cfg)

        resize.assert_not_called()

    def test_stale_capture_frame_is_not_returned(self):
        capture = GameWindowCapturor.__new__(GameWindowCapturor)
        capture.cfg = {"game_window": {"frame_timeout": 1.0}}
        capture.lock = threading.Lock()
        capture.frame = np.zeros((2, 2, 4), dtype=np.uint8)
        capture.last_frame_time = time.monotonic() - 2.0
        capture.is_closed = False
        capture.is_static_frame = False

        self.assertIsNone(capture.get_frame())


if __name__ == "__main__":
    unittest.main()
