import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

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

    def test_directshow_local_click_fails_before_any_mouse_event(self):
        roller = AutoDiceRoller.__new__(AutoDiceRoller)
        roller.capture_source = DIRECTSHOW_SOURCE
        roller.capture = SimpleNamespace(window_title="DirectShow: GC573")
        roller.cfg = {"esp32_hid": {"remote_target": False}}
        roller.loc_dice = (2907, 1373)
        roller.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        roller.input_client = None

        with patch("tools.AutoDiceRoller.click_in_game_window") as click:
            with self.assertRaisesRegex(RuntimeError, "no click was sent"):
                roller.click_dice()

        click.assert_not_called()

    def test_directshow_remote_dice_click_uses_calibrated_absolute_hid(self):
        roller = AutoDiceRoller.__new__(AutoDiceRoller)
        roller.capture_source = DIRECTSHOW_SOURCE
        roller.capture = SimpleNamespace(window_title="DirectShow: GC573")
        roller.cfg = {
            "esp32_hid": {
                "remote_target": True,
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 1366, 768],
            }
        }
        roller.loc_dice = (2907, 1373)
        roller.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        roller.input_client = Mock(
            mouse_click_at=Mock(return_value="OK MOUSE_CLICK_AT")
        )

        with patch("tools.AutoDiceRoller.click_in_game_window") as click:
            self.assertEqual(roller.click_dice(), "OK MOUSE_CLICK_AT")

        roller.input_client.mouse_click_at.assert_called_once_with(
            19367,
            18349,
            "left",
            50,
        )
        click.assert_not_called()

    def test_directshow_remote_dice_click_requires_magpie_calibration(self):
        roller = AutoDiceRoller.__new__(AutoDiceRoller)
        roller.capture_source = DIRECTSHOW_SOURCE
        roller.capture = SimpleNamespace(window_title="DirectShow: GC573")
        roller.cfg = {
            "esp32_hid": {
                "remote_target": True,
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": None,
            }
        }
        roller.loc_dice = (2907, 1373)
        roller.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        roller.input_client = Mock()

        with patch("tools.AutoDiceRoller.click_in_game_window") as click:
            with self.assertRaisesRegex(RuntimeError, "no click was sent"):
                roller.click_dice()

        roller.input_client.mouse_click_at.assert_not_called()
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

    def test_remote_window_capture_never_falls_back_to_local_mouse(self):
        roller = AutoDiceRoller.__new__(AutoDiceRoller)
        roller.capture_source = WINDOW_SOURCE
        roller.capture = SimpleNamespace(window_title="Local Window")
        roller.cfg = {
            "esp32_hid": {"remote_target": True},
            "game_window": {"title": "Configured Window"},
        }
        roller.loc_dice = (981, 445)

        with patch("tools.AutoDiceRoller.click_in_game_window") as click:
            with self.assertRaisesRegex(RuntimeError, "no click was sent"):
                roller.click_dice()

        click.assert_not_called()


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

    def test_capture_card_remote_click_routes_capture_point_to_esp32(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "capture": {"source": "directshow"},
            "esp32_hid": {
                "remote_target": True,
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 1366, 768],
            },
            "auto_relogin": {"mouse_click_duration": 0.05},
        }
        bot.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
        )
        bot.args = SimpleNamespace(test_image="")
        bot.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        bot.kb = SimpleNamespace(
            click_game_ui_point=Mock(return_value=True)
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.click_in_game_window"
        ) as local_click:
            self.assertTrue(bot.click_game_ui((2500, 1200), "test action"))

        bot.kb.click_game_ui_point.assert_called_once_with(
            2500,
            1200,
            3840,
            2160,
            button="left",
            duration=0.05,
        )
        local_click.assert_not_called()

    def test_capture_card_remote_click_requires_magpie_calibration(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "capture": {"source": "directshow"},
            "esp32_hid": {
                "remote_target": True,
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": None,
            },
            "auto_relogin": {"mouse_click_duration": 0.05},
        }
        bot.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
        )
        bot.args = SimpleNamespace(test_image="")
        bot.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        bot.kb = SimpleNamespace(click_game_ui_point=Mock())

        with patch(
            "src.engine.MapleStoryAutoLevelUp.click_in_game_window"
        ) as local_click:
            self.assertFalse(bot.click_game_ui((2500, 1200), "test action"))

        bot.kb.click_game_ui_point.assert_not_called()
        local_click.assert_not_called()

    def test_remote_legacy_ui_points_remove_title_bar_before_4k_scaling(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "ui_coords": {
                "menu": (3378, 2253),
                "channel": (3378, 2055),
            }
        }
        bot._base_cfg = {
            "game_window": {
                "coordinate_reference_size": [700, 1296],
                "title_bar_height": 59,
            },
            "ui_coords": {
                "menu": [1140, 730],
                "channel": [1140, 666],
                "random_channel": [877, 161],
                "random_channel_confirm": [585, 420],
                "select_character": [888, 275],
            },
        }
        bot.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        expected = {
            "menu": (3378, 2071),
            "channel": (3378, 1873),
            "random_channel": (2599, 315),
            "random_channel_confirm": (1733, 1114),
            "select_character": (2631, 667),
        }
        for name, point in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    bot._configured_remote_ui_capture_point(name), point
                )

    def test_window_capture_game_ui_helper_preserves_local_coordinate(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "capture": {"source": "window"},
            "esp32_hid": {"remote_target": False},
            "game_window": {"title_bar_height": 34},
        }
        bot.capture = SimpleNamespace(
            capture_profile="window",
            window_title="MapleStory",
        )
        bot.args = SimpleNamespace(test_image="")

        with patch(
            "src.engine.MapleStoryAutoLevelUp.click_in_game_window"
        ) as local_click:
            self.assertTrue(bot.click_game_ui((31, 47), "test action"))

        local_click.assert_called_once_with("MapleStory", (31, 47))

    def test_remote_channel_change_uses_exclusive_absolute_ui_flow(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "capture": {"source": "directshow"},
            "esp32_hid": {
                "remote_target": True,
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 1366, 768],
            },
            "ui_coords": {
                "menu": (10, 20),
                "channel": (30, 40),
                "random_channel": (50, 60),
                "random_channel_confirm": (70, 80),
                "select_character": (90, 100),
            },
        }
        bot.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
        )
        bot.args = SimpleNamespace(test_image="")
        bot.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        bot.is_terminated = False
        bot.kb = SimpleNamespace(
            game_ui_active=False,
            suspend_automation_for_game_ui=Mock(return_value=True),
            resume_automation_after_game_ui=Mock(return_value=True),
            disable=Mock(),
            set_command=Mock(),
            release_all_key=Mock(),
        )
        bot.click_game_ui = Mock(return_value=True)
        bot.get_img_frame = Mock(return_value=bot.img_frame)
        bot.get_login_button_location = Mock(return_value=(110, 120))
        bot._wait_for_channel_gameplay_ready = Mock(return_value=True)

        def ensure_party_before_resume():
            self.assertFalse(
                bot.kb.resume_automation_after_game_ui.called
            )
            return True

        bot.ensure_is_in_party = Mock(side_effect=ensure_party_before_resume)
        bot.fsm = SimpleNamespace(set_init_state=Mock())

        with patch("src.engine.MapleStoryAutoLevelUp.time.sleep"):
            self.assertTrue(bot.channel_change())

        bot.kb.suspend_automation_for_game_ui.assert_called_once_with()
        bot.kb.resume_automation_after_game_ui.assert_called_once_with()
        bot.kb.disable.assert_not_called()
        bot._wait_for_channel_gameplay_ready.assert_called_once_with()
        self.assertEqual(
            bot.click_game_ui.call_args_list,
            [
                call((10, 20), "channel_change_menu"),
                call((30, 40), "channel_change_channel"),
                call(
                    (50, 60), "channel_change_random_channel"
                ),
                call(
                    (70, 80), "channel_change_random_channel_confirm"
                ),
                call((110, 120), "channel_change_login"),
                call(
                    (90, 100), "channel_change_select_character"
                ),
            ],
        )
        bot.ensure_is_in_party.assert_called_once_with()
        bot.fsm.set_init_state.assert_called_once_with("hunting")

    def test_remote_channel_change_times_out_and_keeps_input_disabled(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "capture": {"source": "directshow"},
            "esp32_hid": {
                "remote_target": True,
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 1366, 768],
            },
            "channel_change": {"ui_timeout": 1},
            "ui_coords": {
                "menu": (10, 20),
                "channel": (30, 40),
                "random_channel": (50, 60),
                "random_channel_confirm": (70, 80),
                "select_character": (90, 100),
            },
        }
        bot.capture = SimpleNamespace(
            capture_profile="capture_card",
            window_title="DirectShow: GC573",
        )
        bot.args = SimpleNamespace(test_image="")
        bot.img_frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        bot.is_terminated = False
        bot.kb = SimpleNamespace(
            game_ui_active=False,
            suspend_automation_for_game_ui=Mock(return_value=True),
            resume_automation_after_game_ui=Mock(return_value=True),
            disable=Mock(),
            set_command=Mock(),
            release_all_key=Mock(),
        )
        bot.click_game_ui = Mock(return_value=True)
        bot.get_img_frame = Mock(return_value=bot.img_frame)
        bot.get_login_button_location = Mock(return_value=None)
        bot._wait_for_channel_gameplay_ready = Mock(return_value=True)
        bot.ensure_is_in_party = Mock(return_value=True)
        bot.fsm = SimpleNamespace(set_init_state=Mock())

        with patch("src.engine.MapleStoryAutoLevelUp.time.sleep"), patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            side_effect=[0.0, 2.0],
        ):
            self.assertFalse(bot.channel_change())

        self.assertEqual(bot.click_game_ui.call_count, 4)
        bot.kb.disable.assert_called_once_with()
        bot.kb.resume_automation_after_game_ui.assert_called_once_with()
        bot.kb.set_command.assert_not_called()
        bot._wait_for_channel_gameplay_ready.assert_not_called()
        bot.ensure_is_in_party.assert_not_called()
        bot.fsm.set_init_state.assert_not_called()

    def test_channel_gameplay_ready_requires_consecutive_evidence(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "channel_change": {
                "game_ready_timeout": 10,
                "game_ready_confirm_frames": 2,
                "game_ready_poll_interval": 0.05,
            }
        }
        bot.is_terminated = False
        bot.img_frame = np.zeros((4, 8, 3), dtype=np.uint8)

        frame_tokens = iter((
            ("capture", 1.0),
            ("capture", 2.0),
            ("capture", 3.0),
        ))

        def get_fresh_frame():
            bot._current_capture_frame_token = next(frame_tokens)
            return bot.img_frame

        bot.get_img_frame = Mock(side_effect=get_fresh_frame)
        bot._auto_relogin_current_gameplay_evidence = Mock(
            side_effect=[None, (1, 1), (1, 1)]
        )

        with patch("src.engine.MapleStoryAutoLevelUp.time.sleep"), patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            side_effect=[0.0, 1.0, 2.0, 3.0],
        ):
            self.assertTrue(bot._wait_for_channel_gameplay_ready())

        self.assertEqual(bot.get_img_frame.call_count, 3)
        self.assertEqual(
            bot._auto_relogin_current_gameplay_evidence.call_count, 3
        )

    def test_channel_gameplay_ready_does_not_recount_cached_frame(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "channel_change": {
                "game_ready_timeout": 10,
                "game_ready_confirm_frames": 2,
                "game_ready_poll_interval": 0.05,
            }
        }
        bot.is_terminated = False
        bot.img_frame = np.zeros((4, 8, 3), dtype=np.uint8)
        frame_tokens = iter((
            ("capture", 1.0),
            ("capture", 1.0),
            ("capture", 2.0),
        ))

        def get_frame_with_duplicate():
            bot._current_capture_frame_token = next(frame_tokens)
            return bot.img_frame

        bot.get_img_frame = Mock(side_effect=get_frame_with_duplicate)
        bot._auto_relogin_current_gameplay_evidence = Mock(
            return_value=(1, 1)
        )

        with patch("src.engine.MapleStoryAutoLevelUp.time.sleep"), patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            side_effect=[0.0, 1.0, 2.0, 3.0],
        ):
            self.assertTrue(bot._wait_for_channel_gameplay_ready())

        self.assertEqual(bot.get_img_frame.call_count, 3)
        self.assertEqual(
            bot._auto_relogin_current_gameplay_evidence.call_count, 2
        )

    def test_channel_gameplay_ready_rejects_frame_without_freshness_token(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "channel_change": {
                "game_ready_timeout": 1,
                "game_ready_confirm_frames": 1,
                "game_ready_poll_interval": 0.05,
            }
        }
        bot.is_terminated = False
        bot.img_frame = np.zeros((4, 8, 3), dtype=np.uint8)
        bot.get_img_frame = Mock(return_value=bot.img_frame)
        bot._auto_relogin_current_gameplay_evidence = Mock(
            return_value=(1, 1)
        )

        with patch("src.engine.MapleStoryAutoLevelUp.time.sleep"), patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            side_effect=[0.0, 0.5, 2.0],
        ):
            self.assertFalse(bot._wait_for_channel_gameplay_ready())

        bot.get_img_frame.assert_called_once_with()
        bot._auto_relogin_current_gameplay_evidence.assert_not_called()


if __name__ == "__main__":
    unittest.main()
