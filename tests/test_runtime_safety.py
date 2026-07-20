import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
