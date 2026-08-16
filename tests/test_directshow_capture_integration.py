import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.input.CaptureSource import DIRECTSHOW_SOURCE, WINDOW_SOURCE
from tools.AutoDiceRoller import AutoDiceRoller
from src.states.hunting import HuntingState
from tools.routeRecorder import RouteRecorder


def capture_card_geometry(height=2160, width=3840):
    return {
        "profile": "capture_card",
        "source_size": (height, width),
        "video_roi": (0, 0, width, height),
        "native_size": (height, width),
        "content_size": (height, width),
        "output_size": (height, width),
        "working_size": (height, width),
        "normalized": False,
    }


class CaptureProfileForwardingTests(unittest.TestCase):
    def test_main_forwards_capture_card_profile_to_preprocessor(self):
        raw = np.zeros((4, 8, 3), dtype=np.uint8)
        geometry = capture_card_geometry(4, 8)
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
            get_frame_snapshot=lambda: (raw, 12.5),
        )
        bot.cfg = {}
        bot.args = SimpleNamespace(test_image="")
        bot._refresh_runtime_frame_config = Mock()

        with patch(
            "src.engine.MapleStoryAutoLevelUp.preprocess_capture_frame",
            return_value=(raw, geometry),
        ) as preprocess:
            output = bot.get_img_frame()

        self.assertIs(output, raw)
        self.assertEqual(
            preprocess.call_args.kwargs["capture_profile"], "capture_card"
        )
        self.assertEqual(
            bot._current_capture_frame_token, ("capture", 12.5)
        )
        bot._refresh_runtime_frame_config.assert_called_once_with((4, 8))
        self.assertIsNone(bot.img_capture_content)

    def test_route_recorder_forwards_capture_card_profile(self):
        raw = np.zeros((4, 8, 3), dtype=np.uint8)
        geometry = capture_card_geometry(4, 8)
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
            get_frame=lambda: raw,
        )
        recorder.cfg = {}
        recorder.update_runtime_config = Mock()

        with patch(
            "tools.routeRecorder.preprocess_capture_frame",
            return_value=(raw, geometry),
        ) as preprocess:
            output = recorder.get_img_frame()

        self.assertIs(output, raw)
        self.assertEqual(
            preprocess.call_args.kwargs["capture_profile"], "capture_card"
        )
        recorder.update_runtime_config.assert_called_once_with((4, 8))
        self.assertIsNone(recorder.img_capture_content)

    def test_auto_dice_forwards_profile_before_recognition(self):
        raw = np.zeros((4, 8, 3), dtype=np.uint8)
        roller = AutoDiceRoller.__new__(AutoDiceRoller)
        roller.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
            get_frame=lambda: raw,
        )
        roller.capture_source = DIRECTSHOW_SOURCE
        roller.cfg = {"ui_coords": {"ui_y_start": 3}}
        roller.kb = SimpleNamespace(is_pressed_func_key=[False] * 12)
        roller.is_enable = False
        roller.is_first_frame = True
        roller.refresh_runtime_geometry = Mock()
        roller.update_img_frame_debug = Mock()

        with patch(
            "tools.AutoDiceRoller.preprocess_capture_frame",
            return_value=(raw, capture_card_geometry(4, 8)),
        ) as preprocess:
            roller.run_once()

        self.assertEqual(
            preprocess.call_args.kwargs["capture_profile"], "capture_card"
        )
        roller.refresh_runtime_geometry.assert_called_once_with((4, 8))
        roller.update_img_frame_debug.assert_called_once_with()


class AutoDiceDirectShowTests(unittest.TestCase):
    def test_scales_legacy_roi_but_never_enlarges_digit_templates(self):
        roller = AutoDiceRoller.__new__(AutoDiceRoller)
        roller.cfg = {
            "game_window": {
                "coordinate_reference_size": [700, 1296]
            },
            "ui_coords": {"ui_y_start": 610},
        }
        roller._runtime_geometry_size = None
        roller.img_numbers = [
            np.zeros((10, 20), dtype=np.uint8)
        ]

        roller.refresh_runtime_geometry((2160, 3840))

        self.assertEqual(roller.loc_dice, (2907, 1373))
        self.assertEqual(roller.loc_first_box, (2637, 1145))
        self.assertEqual(roller.box_size, (68, 110))
        self.assertEqual(roller.box_y_interval, 77)
        self.assertEqual(roller.debug_ui_y_start, 1882)
        self.assertEqual(roller.img_numbers[0].shape, (10, 20))

    def test_coordinate_reference_size_is_configurable_with_legacy_fallback(self):
        source = np.zeros((10, 20), dtype=np.uint8)
        for game_window, expected_loc in (
            ({"coordinate_reference_size": [1400, 2592]}, (1453, 687)),
            ({}, (2907, 1373)),
        ):
            with self.subTest(game_window=game_window):
                roller = AutoDiceRoller.__new__(AutoDiceRoller)
                roller.cfg = {
                    "game_window": game_window,
                    "ui_coords": {"ui_y_start": 610},
                }
                roller._runtime_geometry_size = None
                roller.img_numbers = [source]

                roller.refresh_runtime_geometry((2160, 3840))

                self.assertEqual(roller.loc_dice, expected_loc)

    def test_directshow_click_fails_before_any_local_mouse_event(self):
        for remote_target in (False, True):
            with self.subTest(remote_target=remote_target):
                roller = AutoDiceRoller.__new__(AutoDiceRoller)
                roller.capture_source = DIRECTSHOW_SOURCE
                roller.capture = SimpleNamespace(
                    window_title="DirectShow: GC573"
                )
                roller.cfg = {
                    "esp32_hid": {"remote_target": remote_target}
                }
                roller.loc_dice = (2907, 1373)

                with patch(
                    "tools.AutoDiceRoller.click_in_game_window"
                ) as click:
                    with self.assertRaisesRegex(
                        RuntimeError, "no click was sent"
                    ):
                        roller.click_dice()

                click.assert_not_called()

    def test_window_capture_click_uses_its_actual_window_title(self):
        roller = AutoDiceRoller.__new__(AutoDiceRoller)
        roller.capture_source = WINDOW_SOURCE
        roller.capture = SimpleNamespace(window_title="Actual Window")
        roller.cfg = {"game_window": {"title": "Configured Window"}}
        roller.loc_dice = (981, 445)

        with patch("tools.AutoDiceRoller.click_in_game_window") as click:
            roller.click_dice()

        click.assert_called_once_with("Actual Window", (981, 445))


class RuneDirectShowSafetyTests(unittest.TestCase):
    def test_disabled_rune_solver_never_matches_legacy_templates(self):
        rune_solver = Mock()
        bot = SimpleNamespace(
            cfg={"rune_solver": {"enable": False}},
            rune_solver=rune_solver,
            img_frame_gray=np.zeros((4, 8), dtype=np.uint8),
            img_frame_debug=np.zeros((4, 8, 3), dtype=np.uint8),
        )

        state = HuntingState("hunting", bot)

        self.assertIsNone(state.check_transitions())
        rune_solver.is_rune_enable.assert_not_called()
        rune_solver.is_rune_warning.assert_not_called()


class MainDirectShowSafetyTests(unittest.TestCase):
    def test_static_fixture_overrides_directshow_safety_classification(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {"capture": {"source": "directshow"}}
        bot.capture = SimpleNamespace(window_title="Static fixture")
        bot.args = SimpleNamespace(test_image="fixture.png")

        self.assertFalse(bot.is_capture_card_source())

    def test_capture_card_never_clicks_a_local_window(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "capture": {"source": "directshow"},
            "esp32_hid": {"remote_target": False},
        }
        bot.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
        )
        bot.args = SimpleNamespace(test_image="")

        with patch(
            "src.engine.MapleStoryAutoLevelUp.click_in_game_window"
        ) as click:
            self.assertFalse(bot.click_game_ui((50, 60), "test action"))
            self.assertFalse(bot.channel_change())

        click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
