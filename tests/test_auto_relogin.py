import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np
import yaml

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.vision.auto_relogin_ocr import OcrTextMatch
from src.vision.cursor_tracker import CursorTracker


class AutoReloginTests(unittest.TestCase):
    """Focused tests for the screenshot-confirmed five-page login flow."""

    FRAME_HEIGHT = 200
    FRAME_WIDTH = 400
    DISCONNECT_CONFIRM_POINT = (100, 120)
    CHANNEL_POINT = (200, 140)
    PAGE_LOCATIONS = {
        "disconnect": (10, 20),
        "connect": (20, 30),
        "world": (30, 40),
        "channel": (50, 60),
        "character": (70, 80),
    }

    @classmethod
    def make_bot(
            cls, *, remote=True, mode="normal", enabled=True,
            mouse_mode="absolute"):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"mode": mode},
            "esp32_hid": {"remote_target": remote},
            "game_window": {"title_bar_height": 34},
            "auto_relogin": {
                "enable": enabled,
                "flow_template_reference_size": [
                    cls.FRAME_HEIGHT,
                    cls.FRAME_WIDTH,
                ],
                "confirm_frames": 2,
                "confirm_seconds": 0.0,
                "cancel_confirm_misses": 2,
                "input_retry_delay": 1.0,
                "retry_cooldown": 3.0,
                "step_timeout": 60.0,
                "game_ready_timeout": 20.0,
                "max_recovery_duration": 300.0,
                "max_step_attempts": 5,
                "game_ready_confirm_frames": 2,
                "mouse_click_duration": 0.05,
                "channel_click_count": 2,
                "channel_double_click_interval": 0.0,
                "remote_mouse_mode": mouse_mode,
                "mouse_move_gain": 0.35,
                "mouse_max_delta": 64,
                "mouse_target_tolerance": [18, 18],
                "mouse_target_confirm_frames": 2,
                "mouse_feedback_delay": 0.2,
                "mouse_feedback_frames": 1,
                "mouse_pointer_timeout": 20.0,
                "mouse_max_moves": 40,
                "mouse_cursor_miss_limit": 5,
                "mouse_page_miss_limit": 3,
                "mouse_stall_limit": 4,
                "mouse_cursor_rescue_deltas": [[0, -64]],
                "remote_confirm_key": "enter",
                "disconnect_confirm_point": list(
                    cls.DISCONNECT_CONFIRM_POINT
                ),
                "channel_points": [list(cls.CHANNEL_POINT)],
                "page_anchor_points": {
                    "disconnect": list(cls.PAGE_LOCATIONS["disconnect"]),
                    "channel": list(cls.PAGE_LOCATIONS["channel"]),
                },
                "ocr": {
                    "enable": True,
                    "idle_scan_interval": 1.0,
                    "min_score": 0.85,
                    "box_threshold": 0.3,
                    "confirm_frames": 1,
                    "max_center_drift": [24, 24],
                    # Most state-machine tests use small integer frame tokens
                    # and an independent mocked monotonic clock. Freshness is
                    # covered separately in the OCR-focused test module.
                    "max_frame_age": 10000.0,
                    "targets": {
                        "disconnect": {
                            "texts": ["与服务器连接发生错误"],
                            "region_source": "configured",
                            "search_region": [0, 0, 400, 200],
                            "match_mode": "contains",
                            "action": "enter",
                        },
                        "connect": {
                            "texts": ["连接"],
                            "region_source": "configured",
                            "search_region": [0, 0, 400, 200],
                            "match_mode": "exact",
                            "action": "click",
                        },
                        "world": {
                            "texts": ["4.漂漂猪"],
                            "region_source": "configured",
                            "search_region": [0, 0, 400, 200],
                            "match_mode": "exact",
                            "action": "click",
                        },
                        "channel": {
                            "texts": ["漂漂猪"],
                            "region_source": "configured",
                            "search_region": [0, 0, 400, 200],
                            "match_mode": "exact",
                            "action": "fixed_click",
                        },
                        "character": {
                            "texts": ["开始游戏"],
                            "region_source": "configured",
                            "search_region": [0, 0, 400, 200],
                            "match_mode": "exact",
                            "action": "click",
                        },
                    },
                },
            },
            "health_monitor": {"enable": True},
        }
        bot.is_terminated = False
        bot.is_disable_control = False
        bot.img_frame = np.zeros(
            (cls.FRAME_HEIGHT, cls.FRAME_WIDTH, 3), dtype=np.uint8
        )
        bot.img_frame_debug = None
        bot.capture = SimpleNamespace(
            last_frame_time=1.0,
            window_title="MapleStory",
            is_static_frame=False,
        )
        bot._current_capture_frame_token = None

        bot._visible_page = None

        def match_visible_page(page):
            if page == bot._visible_page:
                return cls.PAGE_LOCATIONS[page]
            return None

        bot._match_auto_relogin_page = Mock(side_effect=match_visible_page)
        bot._auto_relogin_current_gameplay_evidence = Mock(return_value=None)
        bot.click_game_ui = Mock(return_value=True)
        bot._reset_ladder_route_hold = Mock()
        bot._reset_stationary_jump_proximity = Mock()
        bot._reset_portal_sweep = Mock()
        bot.kb = SimpleNamespace(
            is_enable=True,
            is_terminated=False,
            is_need_force_heal=False,
            set_command=Mock(),
            release_all_key=Mock(),
            suspend_automation_for_session_recovery=Mock(),
            resume_automation_after_session_recovery=Mock(),
            press_session_recovery_key=Mock(return_value=True),
            focus_next_window_and_press_session_recovery_key=Mock(
                return_value=True
            ),
            move_session_recovery_mouse=Mock(return_value=True),
            click_session_recovery_mouse=Mock(return_value=True),
            click_session_recovery_point=Mock(return_value=True),
        )
        bot._locate_auto_relogin_cursor = Mock(return_value=(20, 20))
        target_pages = {
            tuple(target["texts"]): page
            for page, target in bot.cfg["auto_relogin"]["ocr"][
                "targets"
            ].items()
        }

        def locate_ocr_target(_frame, _region, targets, **_kwargs):
            page = target_pages[tuple(targets)]
            x, y = cls.PAGE_LOCATIONS[page]
            text = targets[0]
            return OcrTextMatch(
                text=text,
                normalized_text=text,
                score=0.99,
                box=(
                    (x - 2, y - 2), (x + 2, y - 2),
                    (x + 2, y + 2), (x - 2, y + 2),
                ),
                center=(x, y),
            )

        bot._auto_relogin_ocr_locator = SimpleNamespace(
            locate=Mock(side_effect=locate_ocr_target)
        )
        bot._auto_relogin_ocr_gate = None
        bot._auto_relogin_ocr_gate_signature = None
        bot.health_monitor = SimpleNamespace(
            enabled=True,
            disable=Mock(),
            enable=Mock(),
        )
        bot.fsm = SimpleNamespace(set_init_state=Mock())
        bot._reset_auto_relogin_runtime()
        return bot

    @staticmethod
    def check_at(bot, now):
        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            return_value=now,
        ):
            return bot._check_auto_relogin_screen()

    @classmethod
    def show_and_check(cls, bot, page, *, frame_token, now):
        bot._visible_page = page
        bot.capture.last_frame_time = float(frame_token)
        bot._current_capture_frame_token = (
            "capture", float(frame_token)
        )
        return cls.check_at(bot, now)

    def test_any_known_login_page_can_start_recovery_from_idle(self):
        bot = self.make_bot()
        self.assertFalse(
            self.show_and_check(bot, None, frame_token=1, now=10.0)
        )
        self.assertEqual(bot._auto_relogin_state, "idle")
        bot.kb.suspend_automation_for_session_recovery.assert_not_called()

        for visible_page in (
                "disconnect", "connect", "world", "channel", "character"):
            with self.subTest(visible_page=visible_page):
                bot = self.make_bot()

                self.assertTrue(
                    self.show_and_check(
                        bot, visible_page, frame_token=1, now=10.0
                    )
                )

                self.assertEqual(bot._auto_relogin_state, "confirming")
                self.assertEqual(bot._auto_relogin_pending_page, visible_page)
                bot.kb.suspend_automation_for_session_recovery.assert_called_once_with()
                bot.kb.press_session_recovery_key.assert_not_called()
                bot.kb.click_session_recovery_point.assert_not_called()

    def test_bound_frame_token_is_not_replaced_by_newer_live_timestamp(self):
        bot = self.make_bot()
        bot._current_capture_frame_token = ("capture", 10.0)
        bot.capture.last_frame_time = 11.0

        self.assertEqual(
            bot._auto_relogin_frame_token(999.0),
            ("capture", 10.0),
        )

    def test_disconnect_modal_wins_over_a_visible_background_page(self):
        bot = self.make_bot()
        visible = {
            "disconnect": self.PAGE_LOCATIONS["disconnect"],
            "world": self.PAGE_LOCATIONS["world"],
        }
        bot._match_auto_relogin_page.side_effect = visible.get

        self.assertTrue(self.show_and_check(
            bot, None, frame_token=10.0, now=10.0
        ))

        self.assertEqual(bot._auto_relogin_pending_page, "disconnect")

    def test_expected_world_step_precedes_combined_channel_marker(self):
        bot = self.make_bot()
        bot._auto_relogin_state = "waiting_page"
        bot._auto_relogin_expected_page = "world"
        bot._auto_relogin_last_action_page = "connect"
        bot._auto_relogin_step_started_at = 20.0
        bot._auto_relogin_started_at = 20.0
        visible = {
            "channel": self.PAGE_LOCATIONS["channel"],
            "world": self.PAGE_LOCATIONS["world"],
        }
        bot._match_auto_relogin_page.side_effect = visible.get

        self.assertTrue(self.show_and_check(
            bot, None, frame_token=20.1, now=20.1
        ))

        self.assertEqual(bot._auto_relogin_state, "confirming")
        self.assertEqual(bot._auto_relogin_pending_page, "world")
        bot.kb.move_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_fixed_click_point_tracks_the_current_page_anchor(self):
        bot = self.make_bot()
        base = self.CHANNEL_POINT
        recorded_anchor = self.PAGE_LOCATIONS["channel"]

        adjusted = bot._auto_relogin_anchor_adjusted_point(
            "channel",
            base,
            (recorded_anchor[0] + 17, recorded_anchor[1] - 9),
        )

        self.assertEqual(adjusted, (base[0] + 17, base[1] - 9))

    def test_five_pages_require_fresh_frames_and_send_actions_in_order(self):
        bot = self.make_bot(remote=True)
        events = []
        bot.kb.press_session_recovery_key.side_effect = (
            lambda key: events.append(("key", key)) or True
        )
        bot.kb.click_session_recovery_point.side_effect = (
            lambda x, y, frame_width, frame_height, **kwargs:
            events.append((
                "click",
                x,
                y,
                frame_width,
                frame_height,
                kwargs["button"],
                kwargs["duration"],
            )) or True
        )

        token = 1

        def confirm_page(page, start_time):
            nonlocal token
            event_count = len(events)
            self.assertTrue(self.show_and_check(
                bot, page, frame_token=token, now=start_time
            ))
            self.assertEqual(bot._auto_relogin_confirm_count, 1)

            # Reprocessing the same capture cannot satisfy a second frame.
            self.assertTrue(self.show_and_check(
                bot, page, frame_token=token, now=start_time + 0.05
            ))
            self.assertEqual(bot._auto_relogin_confirm_count, 1)
            self.assertEqual(len(events), event_count)

            token += 1
            self.assertTrue(self.show_and_check(
                bot, page, frame_token=token, now=start_time + 0.1
            ))
            action_count = 2 if page == "channel" else 1
            self.assertEqual(len(events), event_count + action_count)
            token += 1

        confirm_page("disconnect", 100.0)
        self.assertEqual(bot._auto_relogin_expected_page, "connect")

        # An unrecognized transition frame owns no input action.
        event_count = len(events)
        self.assertTrue(self.show_and_check(
            bot, None, frame_token=token, now=101.0
        ))
        self.assertEqual(len(events), event_count)
        token += 1

        confirm_page("connect", 104.0)
        self.assertEqual(bot._auto_relogin_expected_page, "world")
        confirm_page("world", 108.0)
        self.assertEqual(bot._auto_relogin_expected_page, "channel")
        confirm_page("channel", 112.0)
        self.assertEqual(bot._auto_relogin_expected_page, "character")
        confirm_page("character", 116.0)

        self.assertEqual(bot._auto_relogin_state, "waiting_game")
        self.assertEqual(
            events,
            [
                ("key", "enter"),
                (
                    "click",
                    *self.PAGE_LOCATIONS["connect"],
                    self.FRAME_WIDTH,
                    self.FRAME_HEIGHT,
                    "left",
                    0.05,
                ),
                (
                    "click",
                    *self.PAGE_LOCATIONS["world"],
                    self.FRAME_WIDTH,
                    self.FRAME_HEIGHT,
                    "left",
                    0.05,
                ),
                (
                    "click",
                    *self.CHANNEL_POINT,
                    self.FRAME_WIDTH,
                    self.FRAME_HEIGHT,
                    "left",
                    0.05,
                ),
                (
                    "click",
                    *self.CHANNEL_POINT,
                    self.FRAME_WIDTH,
                    self.FRAME_HEIGHT,
                    "left",
                    0.05,
                ),
                (
                    "click",
                    *self.PAGE_LOCATIONS["character"],
                    self.FRAME_WIDTH,
                    self.FRAME_HEIGHT,
                    "left",
                    0.05,
                ),
            ],
        )
        bot.click_game_ui.assert_not_called()

    def test_connect_page_can_use_ocr_authorized_enter(self):
        bot = self.make_bot(remote=True)
        bot.cfg["auto_relogin"]["ocr"]["targets"]["connect"][
            "action"
        ] = "enter"

        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=1, now=120.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=2, now=120.1
        ))

        bot.kb.press_session_recovery_key.assert_called_once_with("enter")
        bot.kb.click_session_recovery_point.assert_not_called()
        self.assertEqual(bot._auto_relogin_expected_page, "world")

    def test_connect_page_repeats_enter_until_world_page_appears(self):
        bot = self.make_bot(remote=True)
        bot.cfg["auto_relogin"]["ocr"]["targets"]["connect"][
            "action"
        ] = "enter"
        bot.cfg["auto_relogin"].update({
            "connect_enter_retry_delay": 1.0,
            "connect_enter_max_attempts": 30,
        })

        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=1, now=120.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=2, now=120.1
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)

        # A still-visible connection page is re-authorized from fresh OCR
        # frames, but the next Enter cannot fire before its retry delay.
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=3, now=120.2
        ))
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=4, now=120.3
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=5, now=121.2
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 2)

        # Once the world page appears, the connection-page Enter loop stops.
        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=6, now=121.3
        ))
        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=7, now=121.4
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 2)
        bot.kb.focus_next_window_and_press_session_recovery_key.assert_not_called()
        self.assertEqual(bot._auto_relogin_expected_page, "world")

    def test_connect_page_can_focus_launcher_then_send_enter(self):
        bot = self.make_bot(remote=True)
        bot.cfg["auto_relogin"]["ocr"]["targets"]["connect"][
            "action"
        ] = "focus_next_enter"
        bot.cfg["auto_relogin"].update({
            "focus_switch_keys": ["alt", "tab"],
            "focus_switch_hold": 0.10,
            "focus_switch_settle_delay": 0.50,
            "focus_enter_duration": 0.10,
        })

        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=1, now=120.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=2, now=120.1
        ))

        sender = bot.kb.focus_next_window_and_press_session_recovery_key
        sender.assert_called_once_with(
            "enter",
            focus_keys=["alt", "tab"],
            focus_hold=0.10,
            settle_delay=0.50,
            duration=0.10,
        )
        bot.kb.press_session_recovery_key.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()
        self.assertEqual(bot._auto_relogin_expected_page, "world")
        self.assertEqual(bot._auto_relogin_next_action_at, 180.1)

    def test_unknown_transition_frames_never_send_blind_input(self):
        bot = self.make_bot(remote=True)
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=1, now=200.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=2, now=200.1
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)

        for frame_token, now in ((3, 201.0), (4, 202.0), (5, 203.0)):
            self.assertTrue(self.show_and_check(
                bot, None, frame_token=frame_token, now=now
            ))

        self.assertEqual(bot._auto_relogin_state, "waiting_page")
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)
        bot.kb.click_session_recovery_point.assert_not_called()

        # A minimap-like false positive during an earlier transition cannot
        # bypass the remaining pages; only Start Game authorizes that handoff.
        bot._auto_relogin_current_gameplay_evidence.return_value = (1, 1)
        self.assertTrue(self.show_and_check(
            bot, None, frame_token=6, now=204.0
        ))
        self.assertEqual(bot._auto_relogin_state, "waiting_page")

    def test_waiting_page_accepts_only_expected_last_or_disconnect(self):
        def waiting_bot():
            bot = self.make_bot(remote=True)
            bot._auto_relogin_state = "waiting_page"
            bot._auto_relogin_expected_page = "world"
            bot._auto_relogin_last_action_page = "connect"
            bot._auto_relogin_step_started_at = 250.0
            bot._auto_relogin_started_at = 250.0
            return bot

        bot = waiting_bot()
        for frame_token, unexpected_page in enumerate(
                ("channel", "character"), start=1):
            self.assertTrue(self.show_and_check(
                bot,
                unexpected_page,
                frame_token=frame_token,
                now=250.0 + frame_token,
            ))

        self.assertEqual(bot._auto_relogin_state, "failed")
        self.assertIsNone(bot._auto_relogin_pending_page)
        bot.kb.press_session_recovery_key.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

        # Each explicitly allowed page can independently enter confirmation
        # and perform only its own action.
        for allowed_page in ("world", "connect", "disconnect"):
            with self.subTest(allowed_page=allowed_page):
                bot = waiting_bot()
                self.assertTrue(self.show_and_check(
                    bot, allowed_page, frame_token=10, now=253.0
                ))
                self.assertTrue(self.show_and_check(
                    bot, allowed_page, frame_token=11, now=253.1
                ))
                if allowed_page == "disconnect":
                    bot.kb.press_session_recovery_key.assert_called_once_with(
                        "enter"
                    )
                    bot.kb.click_session_recovery_point.assert_not_called()
                else:
                    bot.kb.click_session_recovery_point.assert_called_once()
                    bot.kb.press_session_recovery_key.assert_not_called()

    def test_uncertain_ack_can_advance_to_successor_without_replaying_old_action(self):
        bot = self.make_bot(remote=True)
        bot.kb.press_session_recovery_key.side_effect = [False, True]

        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=1, now=275.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=2, now=275.1
        ))

        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)
        self.assertEqual(bot._auto_relogin_state, "confirming")
        self.assertEqual(bot._auto_relogin_last_action_page, "disconnect")
        self.assertEqual(bot._auto_relogin_expected_page, "connect")
        self.assertTrue(bot._auto_relogin_has_attempted_input)

        # The next captured page is the configured successor, which proves the
        # uncertain key may have executed. Missing the old page must never
        # cause an immediate duplicate key press.
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=3, now=275.2
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)
        bot.kb.click_session_recovery_point.assert_not_called()

        # After the old-page confirmation is dismissed, fresh confirmation of
        # the successor advances normally.
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=4, now=275.3
        ))
        self.assertEqual(bot._auto_relogin_state, "confirming")
        self.assertTrue(self.show_and_check(
            bot, "connect", frame_token=5, now=278.3
        ))

        self.assertEqual(
            bot.kb.press_session_recovery_key.call_args_list,
            [call("enter")],
        )
        bot.kb.click_session_recovery_point.assert_called_once_with(
            *self.PAGE_LOCATIONS["connect"],
            self.FRAME_WIDTH,
            self.FRAME_HEIGHT,
            button="left",
            duration=0.05,
        )
        self.assertEqual(bot._auto_relogin_state, "waiting_page")
        self.assertEqual(bot._auto_relogin_expected_page, "world")

    def test_same_page_can_retry_only_after_cooldown(self):
        bot = self.make_bot(remote=True)
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=1, now=300.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=2, now=300.1
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)

        # The stale page can be confirmed, but no duplicate key is sent
        # before retry_cooldown expires.
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=3, now=301.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=4, now=301.1
        ))
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=5, now=303.0
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)

        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=6, now=303.2
        ))
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 2)
        self.assertEqual(bot._auto_relogin_expected_page, "connect")

    def test_step_timeout_fails_closed_without_blind_retry(self):
        bot = self.make_bot(remote=True)
        bot.cfg["auto_relogin"]["step_timeout"] = 5.0
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=1, now=400.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=2, now=400.1
        ))

        self.assertTrue(self.show_and_check(
            bot, None, frame_token=3, now=405.2
        ))

        self.assertEqual(bot._auto_relogin_state, "failed")
        self.assertEqual(bot.kb.press_session_recovery_key.call_count, 1)
        bot.kb.click_session_recovery_point.assert_not_called()
        self.assertTrue(bot._gate_auto_relogin_until_game_ready(None))

    def test_waiting_game_timeout_fails_closed(self):
        bot = self.make_bot(remote=True)
        bot.cfg["auto_relogin"]["game_ready_timeout"] = 5.0
        bot._auto_relogin_state = "waiting_game"
        bot._auto_relogin_waiting_game_started_at = 500.0

        self.assertTrue(self.show_and_check(
            bot, None, frame_token=1, now=505.1
        ))

        self.assertEqual(bot._auto_relogin_state, "failed")
        bot.kb.press_session_recovery_key.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()
        self.assertTrue(bot._gate_auto_relogin_until_game_ready(None))

    def test_false_positive_is_cancelled_after_distinct_misses_and_resumes(self):
        bot = self.make_bot(remote=True)
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=1, now=600.0
        ))
        bot.health_monitor.disable.assert_called_once_with()

        self.assertTrue(self.show_and_check(
            bot, None, frame_token=2, now=600.1
        ))
        self.assertEqual(bot._auto_relogin_confirm_miss_count, 1)

        # A duplicate processing pass over that miss does not count twice.
        self.assertTrue(self.show_and_check(
            bot, None, frame_token=2, now=600.2
        ))
        self.assertEqual(bot._auto_relogin_confirm_miss_count, 1)

        self.assertFalse(self.show_and_check(
            bot, None, frame_token=3, now=600.3
        ))

        self.assertEqual(bot._auto_relogin_state, "idle")
        bot.health_monitor.enable.assert_called_once_with()
        bot.kb.resume_automation_after_session_recovery.assert_called_once_with()
        bot.kb.press_session_recovery_key.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_confirmation_match_count_restarts_after_a_missed_frame(self):
        bot = self.make_bot(remote=True)
        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=1, now=650.0
        ))
        self.assertTrue(self.show_and_check(
            bot, None, frame_token=2, now=651.0
        ))
        self.assertEqual(bot._auto_relogin_confirm_count, 0)

        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=3, now=652.0
        ))
        self.assertEqual(bot._auto_relogin_confirm_count, 1)
        bot.kb.click_session_recovery_point.assert_not_called()

        self.assertTrue(self.show_and_check(
            bot, "disconnect", frame_token=4, now=652.1
        ))
        bot.kb.press_session_recovery_key.assert_called_once_with("enter")
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_waiting_game_requires_consecutive_fresh_gameplay_evidence(self):
        expected_states = {
            "normal": "hunting",
            "aux": "aux",
            "patrol": "patrol",
        }
        for mode, expected_state in expected_states.items():
            with self.subTest(mode=mode):
                bot = self.make_bot(mode=mode)
                bot._auto_relogin_state = "waiting_game"
                bot._auto_relogin_health_was_enabled = True

                self.assertTrue(bot._gate_auto_relogin_until_game_ready(None))
                bot.capture.last_frame_time = 10.0
                bot._current_capture_frame_token = ("capture", 10.0)
                self.assertTrue(
                    bot._gate_auto_relogin_until_game_ready((100, 200))
                )
                self.assertEqual(bot._auto_relogin_ready_count, 1)

                # A cached capture is not a second gameplay confirmation.
                self.assertTrue(
                    bot._gate_auto_relogin_until_game_ready((101, 201))
                )
                self.assertEqual(bot._auto_relogin_ready_count, 1)

                # A missing dot breaks the consecutive run.
                self.assertTrue(bot._gate_auto_relogin_until_game_ready(None))
                bot.capture.last_frame_time = 11.0
                bot._current_capture_frame_token = ("capture", 11.0)
                self.assertTrue(
                    bot._gate_auto_relogin_until_game_ready((102, 202))
                )
                bot.fsm.set_init_state.assert_not_called()

                bot.capture.last_frame_time = 12.0
                bot._current_capture_frame_token = ("capture", 12.0)
                self.assertTrue(
                    bot._gate_auto_relogin_until_game_ready((103, 203))
                )

                self.assertEqual(bot._auto_relogin_state, "idle")
                bot.fsm.set_init_state.assert_called_once_with(expected_state)
                bot.health_monitor.enable.assert_called_once_with()
                bot.kb.resume_automation_after_session_recovery.assert_called_once_with()
                self.assertEqual(bot.health_monitor.hp_percent, 100)
                self.assertEqual(bot.health_monitor.mp_percent, 100)
                self.assertFalse(bot.kb.is_need_force_heal)

    def test_remote_absolute_click_uses_frame_geometry_and_duration(self):
        bot = self.make_bot(remote=True)

        self.assertTrue(bot._send_auto_relogin_click((31, 47), "test_click"))

        bot.kb.click_session_recovery_point.assert_called_once_with(
            31,
            47,
            self.FRAME_WIDTH,
            self.FRAME_HEIGHT,
            button="left",
            duration=0.05,
        )
        bot.click_game_ui.assert_not_called()
        bot.kb.move_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_mouse.assert_not_called()

    def test_visual_relative_click_waits_for_fresh_feedback_and_alignment(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")
        bot._locate_auto_relogin_cursor.side_effect = (
            (20, 20),
            (25, 36),
            self.PAGE_LOCATIONS["world"],
        )

        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.0, now=800.0
        ))
        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.1, now=800.1
        ))
        self.assertEqual(bot._auto_relogin_state, "aiming")

        # The confirmation image cannot also drive the first movement.
        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.1, now=800.2
        ))
        bot.kb.move_session_recovery_mouse.assert_not_called()

        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.4, now=800.4
        ))
        bot.kb.move_session_recovery_mouse.assert_called_once_with(0, 7)

        # Neither a duplicate capture nor a fresh image that arrived before
        # the configured feedback delay may trigger another nudge or click.
        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.4, now=800.45
        ))
        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.5, now=800.5
        ))
        bot.kb.move_session_recovery_mouse.assert_called_once()
        bot.kb.click_session_recovery_mouse.assert_not_called()

        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.7, now=800.7
        ))
        self.assertEqual(bot._auto_relogin_pointer_aligned_count, 1)
        bot.kb.click_session_recovery_mouse.assert_not_called()

        self.assertTrue(self.show_and_check(
            bot, "world", frame_token=800.8, now=800.8
        ))
        bot.kb.click_session_recovery_mouse.assert_called_once_with(
            button="left", duration=0.05
        )
        bot.kb.click_session_recovery_point.assert_not_called()
        self.assertEqual(bot._auto_relogin_state, "waiting_page")
        self.assertEqual(bot._auto_relogin_expected_page, "channel")

    def test_hovered_world_button_is_click_ready_without_center_alignment(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")

        self.show_and_check(bot, "world", frame_token=810.0, now=810.0)
        self.show_and_check(bot, "world", frame_token=810.1, now=810.1)
        self.assertEqual(bot._auto_relogin_state, "aiming")

        # The target center is (30, 40). This hotspot is well outside the
        # ordinary 18-pixel center tolerance, but still lies inside the
        # 100x40 highlighted button rectangle.
        bot._visible_page = None
        bot._auto_relogin_templates = {
            "world": np.zeros((40, 100, 3), dtype=np.uint8),
        }
        bot._match_auto_relogin_hovered_target = Mock(
            return_value=self.PAGE_LOCATIONS["world"]
        )
        bot._locate_auto_relogin_cursor.return_value = (75, 40)
        bot._auto_relogin_pointer_motion_verified = True

        self.show_and_check(bot, None, frame_token=810.4, now=810.4)
        self.assertEqual(bot._auto_relogin_pointer_aligned_count, 1)
        bot.kb.click_session_recovery_mouse.assert_not_called()

        self.show_and_check(bot, None, frame_token=810.5, now=810.5)
        bot.kb.click_session_recovery_mouse.assert_called_once_with(
            button="left", duration=0.05
        )
        self.assertEqual(bot._auto_relogin_expected_page, "channel")
        bot.kb.move_session_recovery_mouse.assert_not_called()

    def test_visual_relative_five_page_flow_never_uses_absolute_mouse(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")
        events = []
        def located_cursor():
            target = bot._auto_relogin_pointer_target
            if bot._auto_relogin_pointer_move_origin is not None:
                return target
            if bot._auto_relogin_pointer_last_cursor is not None:
                return bot._auto_relogin_pointer_last_cursor
            return (target[0] - 30, target[1])

        bot._locate_auto_relogin_cursor.side_effect = located_cursor
        bot.kb.press_session_recovery_key.side_effect = (
            lambda key: events.append(("key", key)) or True
        )
        bot.kb.click_session_recovery_mouse.side_effect = (
            lambda **kwargs: events.append(("click", kwargs)) or True
        )
        def finish_page(page, now):
            self.show_and_check(
                bot, page, frame_token=now, now=now
            )
            self.show_and_check(
                bot, page, frame_token=now + 0.1, now=now + 0.1
            )
            if page != "disconnect":
                self.assertEqual(bot._auto_relogin_state, "aiming")
                self.show_and_check(
                    bot, page, frame_token=now + 0.2, now=now + 0.2
                )
                self.show_and_check(
                    bot, page, frame_token=now + 0.5, now=now + 0.5
                )
                self.show_and_check(
                    bot, page, frame_token=now + 0.6, now=now + 0.6
                )

        finish_page("disconnect", 860.0)
        finish_page("connect", 864.0)
        finish_page("world", 868.0)
        finish_page("channel", 872.0)
        finish_page("character", 876.0)

        self.assertEqual(
            events,
            [
                ("key", "enter"),
                ("click", {"button": "left", "duration": 0.05}),
                ("click", {"button": "left", "duration": 0.05}),
                ("click", {"button": "left", "duration": 0.05}),
                ("click", {"button": "left", "duration": 0.05}),
                ("click", {"button": "left", "duration": 0.05}),
            ],
        )
        self.assertEqual(bot._auto_relogin_state, "waiting_game")
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_visual_relative_cursor_miss_only_sends_safe_no_click_rescue(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")
        bot._locate_auto_relogin_cursor.return_value = None

        self.show_and_check(bot, "world", frame_token=1, now=820.0)
        self.show_and_check(bot, "world", frame_token=2, now=820.1)
        self.show_and_check(bot, "world", frame_token=3, now=820.4)

        bot.kb.move_session_recovery_mouse.assert_called_once_with(0, -64)
        bot.kb.click_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()
        self.assertEqual(bot._auto_relogin_state, "aiming")

    def test_cursor_rescue_finishes_long_path_before_miss_limit(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")
        bot.cfg["auto_relogin"].update({
            "mouse_cursor_miss_limit": 1,
            "mouse_cursor_rescue_deltas": [
                [4096, 0],
                [0, -256],
                [0, 512],
            ],
        })
        bot._locate_auto_relogin_cursor.return_value = None

        self.show_and_check(
            bot, "world", frame_token=825.0, now=825.0
        )
        self.show_and_check(
            bot, "world", frame_token=825.1, now=825.1
        )
        self.show_and_check(
            bot, "world", frame_token=825.4, now=825.4
        )
        self.show_and_check(
            bot, "world", frame_token=825.7, now=825.7
        )
        self.show_and_check(
            bot, "world", frame_token=826.0, now=826.0
        )

        self.assertEqual(
            bot.kb.move_session_recovery_mouse.call_args_list,
            [call(4096, 0), call(0, -256), call(0, 512)],
        )
        self.assertEqual(bot._auto_relogin_state, "aiming")
        bot.kb.click_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

        self.show_and_check(
            bot, "world", frame_token=826.3, now=826.3
        )
        self.assertEqual(bot._auto_relogin_state, "failed")
        bot.kb.click_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_static_cursor_like_candidate_never_becomes_clickable(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")
        bot._locate_auto_relogin_cursor.side_effect = lambda: (
            bot._auto_relogin_pointer_target
        )

        self.show_and_check(
            bot, "world", frame_token=900.0, now=900.0
        )
        self.show_and_check(
            bot, "world", frame_token=900.1, now=900.1
        )
        for captured_at in (900.2, 900.5, 900.8, 901.1, 901.4):
            self.show_and_check(
                bot,
                "world",
                frame_token=captured_at,
                now=captured_at,
            )

        self.assertEqual(bot._auto_relogin_state, "failed")
        self.assertGreaterEqual(
            bot.kb.move_session_recovery_mouse.call_count, 1
        )
        bot.kb.click_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_cursor_loss_discards_an_earlier_motion_verification(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")
        bot._locate_auto_relogin_cursor.side_effect = (
            (20, 20),
            self.PAGE_LOCATIONS["world"],
            None,
            self.PAGE_LOCATIONS["world"],
            self.PAGE_LOCATIONS["world"],
        )

        for captured_at in (
                950.0, 950.1, 950.2, 950.5, 950.6, 950.9, 951.2):
            self.show_and_check(
                bot,
                "world",
                frame_token=captured_at,
                now=captured_at,
            )

        self.assertFalse(bot._auto_relogin_pointer_motion_verified)
        self.assertEqual(bot._auto_relogin_state, "aiming")
        bot.kb.click_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_visual_relative_page_loss_and_geometry_change_fail_closed(self):
        bot = self.make_bot(remote=True, mouse_mode="visual_relative")
        self.show_and_check(bot, "world", frame_token=1, now=840.0)
        self.show_and_check(bot, "world", frame_token=2, now=840.1)
        for token in (3, 4, 5):
            self.show_and_check(
                bot, None, frame_token=token, now=840.1 + token * 0.1
            )
        self.assertEqual(bot._auto_relogin_state, "failed")
        bot.kb.move_session_recovery_mouse.assert_not_called()
        bot.kb.click_session_recovery_mouse.assert_not_called()

        resized = self.make_bot(
            remote=True, mouse_mode="visual_relative"
        )
        self.show_and_check(
            resized, "world", frame_token=1, now=850.0
        )
        self.show_and_check(
            resized, "world", frame_token=2, now=850.1
        )
        resized.img_frame = np.zeros((201, 400, 3), dtype=np.uint8)
        self.show_and_check(
            resized, "world", frame_token=3, now=850.4
        )
        self.assertEqual(resized._auto_relogin_state, "failed")
        resized.kb.move_session_recovery_mouse.assert_not_called()
        resized.kb.click_session_recovery_mouse.assert_not_called()

    def test_local_click_adds_title_bar_offset(self):
        bot = self.make_bot(remote=False)

        self.assertTrue(bot._send_auto_relogin_click((31, 47), "test_click"))

        bot.click_game_ui.assert_called_once_with((31, 81), "test_click")
        bot.kb.click_session_recovery_point.assert_not_called()

    def test_debug_disabled_and_terminated_modes_do_not_emit_input(self):
        disabled_control = self.make_bot()
        disabled_control.is_disable_control = True
        cases = (
            self.make_bot(mode="debug"),
            self.make_bot(enabled=False),
            disabled_control,
        )
        for bot in cases:
            with self.subTest(
                mode=bot.cfg["bot"]["mode"],
                enabled=bot.cfg["auto_relogin"]["enable"],
            ):
                self.assertFalse(self.show_and_check(
                    bot, "disconnect", frame_token=1, now=700.0
                ))
                self.assertEqual(bot._auto_relogin_state, "idle")
                bot.kb.suspend_automation_for_session_recovery.assert_not_called()
                bot.kb.press_session_recovery_key.assert_not_called()
                bot.kb.click_session_recovery_point.assert_not_called()

        terminated = self.make_bot()
        terminated.is_terminated = True
        self.assertTrue(self.show_and_check(
            terminated, "disconnect", frame_token=1, now=701.0
        ))
        terminated._match_auto_relogin_page.assert_not_called()
        terminated.kb.press_session_recovery_key.assert_not_called()
        terminated.kb.click_session_recovery_point.assert_not_called()

    def test_recorded_templates_classify_each_page_without_crossing(self):
        root = Path(__file__).resolve().parents[1]
        page_screenshots = {
            "disconnect": "2026-08-15_11-58-57_img_frame.png",
            "connect": "2026-08-15_11-59-40_img_frame.png",
            "world": "2026-08-15_11-59-51_img_frame.png",
            "channel": "2026-08-15_11-59-55_img_frame.png",
            "character": "2026-08-15_11-59-59_img_frame.png",
        }
        page_regions = {
            "disconnect": [1250, 600, 2300, 1150],
            "connect": [1700, 850, 2200, 1150],
            "world": [1800, 300, 2450, 550],
            "channel": [1200, 700, 1950, 1050],
            "character": [2300, 500, 2950, 950],
        }
        required_screenshots = [
            root / "screenshot" / name
            for name in page_screenshots.values()
        ]
        required_screenshots.append(
            root / "screenshot" / "2026-08-14_02-30-12_img_frame.png"
        )
        if not all(path.exists() for path in required_screenshots):
            self.skipTest("local recorded login screenshots are unavailable")
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "auto_relogin": {
                "flow_template_reference_size": [2013, 3579],
                "template_threshold": 0.03,
                "page_search_regions": page_regions,
            }
        }
        bot._auto_relogin_templates = {}
        for page in page_screenshots:
            template_path = root / "misc" / f"auto_relogin_{page}_cn.png"
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(template, str(template_path))
            bot._auto_relogin_templates[page] = template

        for expected_page, screenshot_name in page_screenshots.items():
            with self.subTest(expected_page=expected_page):
                screenshot_path = root / "screenshot" / screenshot_name
                frame = cv2.imread(str(screenshot_path), cv2.IMREAD_COLOR)
                self.assertIsNotNone(frame, str(screenshot_path))
                self.assertEqual(frame.shape[:2], (2013, 3579))
                bot.img_frame = frame

                self.assertIsNotNone(
                    bot._match_auto_relogin_page(expected_page)
                )
                classified_page, _ = bot._find_known_auto_relogin_page()
                self.assertEqual(classified_page, expected_page)
                for other_page in page_screenshots:
                    if other_page != expected_page:
                        # The channel overlay intentionally leaves the selected
                        # world's label visible behind it.  Its exact world
                        # crop therefore remains present, but classifier
                        # priority must still select the channel page above.
                        if (expected_page, other_page) == (
                                "channel", "world"):
                            self.assertIsNotNone(
                                bot._match_auto_relogin_page(other_page)
                            )
                            continue
                        self.assertIsNone(
                            bot._match_auto_relogin_page(other_page),
                            f"{other_page} falsely matched {expected_page}",
                        )

        gameplay_path = (
            root / "screenshot" / "2026-08-14_02-30-12_img_frame.png"
        )
        bot.img_frame = cv2.imread(str(gameplay_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(bot.img_frame, str(gameplay_path))
        for page in page_screenshots:
            with self.subTest(gameplay_false_positive=page):
                self.assertIsNone(bot._match_auto_relogin_page(page))

    def test_recorded_hovered_world_target_is_structurally_verified(self):
        root = Path(__file__).resolve().parents[1]
        frame_path = (
            root / "screenshot" / "2026-08-16_02-41-40_img_frame.png"
        )
        template_path = root / "misc" / "auto_relogin_world_cn.png"
        if not frame_path.exists() or not template_path.exists():
            self.skipTest("local hovered-world screenshot is unavailable")

        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "auto_relogin": {
                "flow_template_reference_size": [2013, 3579],
                "template_threshold": 0.03,
                "mouse_hover_template_correlation": 0.55,
                "mouse_target_drift": [50, 50],
                "page_search_regions": {
                    "world": [1800, 300, 2450, 550],
                },
            },
        }
        bot.img_frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        bot._auto_relogin_templates = {
            "world": cv2.imread(str(template_path), cv2.IMREAD_COLOR),
        }
        self.assertIsNotNone(bot.img_frame, str(frame_path))
        self.assertIsNotNone(
            bot._auto_relogin_templates["world"], str(template_path)
        )

        target = (2127, 430)
        self.assertIsNone(bot._match_auto_relogin_page("world"))
        hovered = bot._match_auto_relogin_hovered_target("world", target)
        self.assertIsNotNone(hovered)
        self.assertLessEqual(abs(hovered[0] - 2127), 2)
        self.assertLessEqual(abs(hovered[1] - 429), 2)

    def test_page_template_scaling_always_uses_original_sources(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "auto_relogin": {
                "flow_template_reference_size": [100, 200],
            }
        }
        bot._base_cfg = bot.cfg
        disconnect_source = np.arange(
            10 * 20 * 3, dtype=np.uint8
        ).reshape((10, 20, 3))
        world_source = np.full((6, 8, 3), 137, dtype=np.uint8)
        bot._auto_relogin_template_sources = {
            "disconnect": disconnect_source.copy(),
            "world": world_source.copy(),
        }
        bot._auto_relogin_templates = {}
        bot._last_auto_relogin_template_geometry = None

        bot._refresh_auto_relogin_templates((200, 400))
        self.assertEqual(
            bot._auto_relogin_templates["disconnect"].shape,
            (20, 40, 3),
        )
        self.assertEqual(
            bot._auto_relogin_templates["world"].shape,
            (12, 16, 3),
        )

        # A second, smaller output must be derived from the immutable sources,
        # not from the previously enlarged runtime templates.
        bot._refresh_auto_relogin_templates((50, 100))
        self.assertEqual(
            bot._auto_relogin_templates["disconnect"].shape,
            (5, 10, 3),
        )
        self.assertEqual(
            bot._auto_relogin_templates["world"].shape,
            (3, 4, 3),
        )
        self.assertEqual(
            bot._auto_relogin_template_sources["disconnect"].shape,
            (10, 20, 3),
        )
        self.assertTrue(np.array_equal(
            bot._auto_relogin_template_sources["disconnect"],
            disconnect_source,
        ))

        bot._refresh_auto_relogin_templates((100, 200))
        self.assertEqual(
            bot._auto_relogin_templates["disconnect"].shape,
            disconnect_source.shape,
        )
        self.assertTrue(np.array_equal(
            bot._auto_relogin_templates["disconnect"],
            disconnect_source,
        ))
        self.assertIsNot(
            bot._auto_relogin_templates["disconnect"],
            bot._auto_relogin_template_sources["disconnect"],
        )

    def test_disconnect_page_uses_dedicated_cursor_tracker(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {"auto_relogin": {}}
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot._auto_relogin_pointer_last_cursor = None
        bot._auto_relogin_pointer_rescue_index = 0
        default_tracker = Mock()
        disconnect_tracker = Mock()
        bot._auto_relogin_cursor_tracker = default_tracker
        bot._auto_relogin_disconnect_cursor_tracker = disconnect_tracker

        disconnect_match = SimpleNamespace(
            hotspot=(30, 40), score=0.97, uniqueness=0.08
        )
        disconnect_tracker.locate.return_value = disconnect_match
        default_tracker.locate.return_value = None
        bot._auto_relogin_pointer_page = "disconnect"

        self.assertEqual(bot._locate_auto_relogin_cursor(), (30, 40))
        disconnect_tracker.locate.assert_called_once_with(
            bot.img_frame,
            previous_hotspot=None,
            local_radius=None,
            search_region=None,
        )
        default_tracker.locate.assert_called_once_with(
            bot.img_frame,
            previous_hotspot=None,
            local_radius=None,
            search_region=None,
        )

        world_match = SimpleNamespace(
            hotspot=(70, 80), score=0.96, uniqueness=0.07
        )
        default_tracker.reset_mock()
        default_tracker.locate.return_value = world_match
        bot._auto_relogin_pointer_page = "world"

        self.assertEqual(bot._locate_auto_relogin_cursor(), (70, 80))
        default_tracker.locate.assert_called_once_with(
            bot.img_frame,
            previous_hotspot=None,
            local_radius=None,
            search_region=None,
        )

    def test_disconnect_page_accepts_large_cursor_when_compact_misses(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {"auto_relogin": {}}
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot._auto_relogin_pointer_last_cursor = None
        bot._auto_relogin_pointer_rescue_index = 0
        bot._auto_relogin_pointer_page = "disconnect"
        bot._auto_relogin_disconnect_cursor_tracker = Mock()
        bot._auto_relogin_disconnect_cursor_tracker.locate.return_value = None
        large_match = SimpleNamespace(
            hotspot=(45, 55), score=0.96, uniqueness=0.07
        )
        bot._auto_relogin_cursor_tracker = Mock()
        bot._auto_relogin_cursor_tracker.locate.return_value = large_match

        self.assertEqual(bot._locate_auto_relogin_cursor(), (45, 55))
        bot._auto_relogin_disconnect_cursor_tracker.locate.assert_called_once()
        bot._auto_relogin_cursor_tracker.locate.assert_called_once()

    def test_disconnect_page_rejects_disagreeing_cursor_sizes(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "auto_relogin": {
                "flow_template_reference_size": [100, 200],
                "mouse_target_tolerance": [5, 5],
            }
        }
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot._auto_relogin_pointer_last_cursor = None
        bot._auto_relogin_pointer_rescue_index = 0
        bot._auto_relogin_pointer_page = "disconnect"
        bot._auto_relogin_disconnect_cursor_tracker = Mock()
        bot._auto_relogin_disconnect_cursor_tracker.locate.return_value = \
            SimpleNamespace(
                hotspot=(40, 50), score=0.96, uniqueness=0.07
            )
        bot._auto_relogin_cursor_tracker = Mock()
        bot._auto_relogin_cursor_tracker.locate.return_value = \
            SimpleNamespace(
                hotspot=(80, 90), score=0.97, uniqueness=0.08
            )

        self.assertIsNone(bot._locate_auto_relogin_cursor())

    def test_cursor_size_change_resets_click_qualification(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {"auto_relogin": {}}
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot._auto_relogin_pointer_last_cursor = None
        bot._auto_relogin_pointer_rescue_index = 0
        bot._auto_relogin_pointer_page = "disconnect"
        bot._auto_relogin_pointer_cursor_variant = "compact"
        bot._auto_relogin_pointer_motion_verified = True
        bot._auto_relogin_pointer_aligned_count = 2
        bot._auto_relogin_disconnect_cursor_tracker = Mock()
        bot._auto_relogin_disconnect_cursor_tracker.locate.return_value = None
        bot._auto_relogin_cursor_tracker = Mock()
        bot._auto_relogin_cursor_tracker.locate.return_value = \
            SimpleNamespace(
                hotspot=(45, 55), score=0.96, uniqueness=0.07
            )

        self.assertEqual(bot._locate_auto_relogin_cursor(), (45, 55))
        self.assertEqual(bot._auto_relogin_pointer_cursor_variant, "large")
        self.assertFalse(bot._auto_relogin_pointer_motion_verified)
        self.assertEqual(bot._auto_relogin_pointer_aligned_count, 0)

    def test_cursor_tracker_selection_fails_closed_for_unknown_pages(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {"auto_relogin": {}}
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot._auto_relogin_pointer_last_cursor = None
        bot._auto_relogin_pointer_rescue_index = 0
        bot._auto_relogin_cursor_tracker = Mock()
        bot._auto_relogin_disconnect_cursor_tracker = Mock()

        for page in (None, "connect", "typo"):
            with self.subTest(page=page):
                bot._auto_relogin_pointer_page = page
                self.assertIsNone(bot._locate_auto_relogin_cursor())

        bot._auto_relogin_cursor_tracker.locate.assert_not_called()
        bot._auto_relogin_disconnect_cursor_tracker.locate.assert_not_called()

    def test_both_cursor_templates_scale_from_original_sources(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "auto_relogin": {
                "flow_template_reference_size": [100, 200],
                "cursor_hotspot": [2, 1],
                "disconnect_cursor_hotspot": [1, 1],
                "cursor_min_score": 0.8,
                "cursor_uniqueness_margin": 0.02,
                "cursor_min_visible_fraction": 0.25,
                "cursor_min_visible_pixels": 1,
            }
        }
        bot._base_cfg = bot.cfg
        bot._auto_relogin_template_sources = {}
        default_source = np.zeros((10, 20, 4), dtype=np.uint8)
        default_source[:, :, 3] = 255
        disconnect_source = np.zeros((5, 8, 4), dtype=np.uint8)
        disconnect_source[:, :, 3] = 255
        bot._auto_relogin_cursor_template_source = default_source
        bot._auto_relogin_disconnect_cursor_template_source = \
            disconnect_source
        bot._auto_relogin_cursor_tracker = None
        bot._auto_relogin_disconnect_cursor_tracker = None
        bot._last_auto_relogin_template_geometry = None

        bot._refresh_auto_relogin_templates((200, 400))

        self.assertEqual(
            bot._auto_relogin_cursor_tracker._template.shape[:2],
            (20, 40),
        )
        self.assertEqual(bot._auto_relogin_cursor_tracker.hotspot, (4, 2))
        self.assertEqual(
            bot._auto_relogin_disconnect_cursor_tracker._template.shape[:2],
            (10, 16),
        )
        self.assertEqual(
            bot._auto_relogin_disconnect_cursor_tracker.hotspot,
            (2, 2),
        )

        self.assertEqual(default_source.shape, (10, 20, 4))
        self.assertEqual(disconnect_source.shape, (5, 8, 4))

    def test_shipped_cursor_templates_are_masked_and_hotspots_valid(self):
        root = Path(__file__).resolve().parents[1]
        templates = (
            ("auto_relogin_cursor_cn.png", (13, 6)),
            ("auto_relogin_cursor_small_cn.png", (9, 4)),
        )
        for filename, hotspot in templates:
            with self.subTest(filename=filename):
                path = root / "misc" / filename
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(image, str(path))
                self.assertEqual(image.ndim, 3)
                self.assertEqual(image.shape[2], 4)
                self.assertGreater(cv2.countNonZero(image[:, :, 3]), 0)
                tracker = CursorTracker(image, hotspot=hotspot)
                self.assertEqual(tracker.hotspot, hotspot)

    def test_shipped_config_uses_chinese_ocr_for_all_five_pages(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "config" / "config_default.yaml").open(
                encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)

        auto_relogin = cfg["auto_relogin"]
        self.assertTrue(auto_relogin["enable"])
        self.assertTrue(auto_relogin["ocr"]["enable"])
        self.assertNotIn("page_templates", auto_relogin)
        self.assertNotIn("page_search_regions", auto_relogin)
        targets = auto_relogin["ocr"]["targets"]
        self.assertEqual(
            set(targets),
            {"disconnect", "connect", "world", "channel", "character"},
        )
        disconnect = auto_relogin["ocr"]["targets"]["disconnect"]
        self.assertEqual(disconnect["action"], "enter")
        self.assertEqual(disconnect["match_mode"], "contains")
        self.assertEqual(
            disconnect["texts"], ["与服务器连接发生错误"]
        )
        self.assertEqual(targets["connect"]["action"], "enter")
        self.assertEqual(auto_relogin["connect_enter_retry_delay"], 1.0)
        self.assertEqual(auto_relogin["connect_enter_max_attempts"], 30)
        self.assertNotIn("focus_switch_keys", auto_relogin)
        self.assertEqual(targets["world"]["texts"], ["4.漂漂猪"])
        self.assertEqual(targets["channel"]["action"], "fixed_click")
        self.assertEqual(targets["channel"]["texts"], ["漂漂猪"])
        self.assertEqual(targets["channel"]["match_mode"], "exact")
        self.assertEqual(auto_relogin["channel_click_count"], 2)
        self.assertEqual(len(auto_relogin["channel_points"]), 20)
        self.assertEqual(targets["character"]["action"], "click")

    def test_invalid_relogin_config_fails_during_load(self):
        valid_section = {
            "enable": True,
            "confirm_frames": 2,
            "cancel_confirm_misses": 2,
            "flow_template_reference_size": [100, 200],
            "channel_points": [[50, 50]],
            "ocr": {
                "enable": True,
                "idle_scan_interval": 1.0,
                "min_score": 0.85,
                "box_threshold": 0.3,
                "confirm_frames": 2,
                "max_center_drift": [10, 10],
                "max_frame_age": 1.0,
                "targets": {
                    "disconnect": {
                        "texts": ["与服务器连接发生错误"],
                        "region_source": "configured",
                        "search_region": [0, 0, 100, 100],
                        "match_mode": "contains",
                        "action": "enter",
                    },
                    "connect": {
                        "texts": ["连接"],
                        "region_source": "configured",
                        "search_region": [0, 0, 100, 100],
                        "match_mode": "exact",
                        "action": "click",
                    },
                    "world": {
                        "texts": ["4.漂漂猪"],
                        "region_source": "configured",
                        "search_region": [0, 0, 100, 100],
                        "match_mode": "exact",
                        "action": "click",
                    },
                    "channel": {
                        "texts": ["漂漂猪"],
                        "region_source": "configured",
                        "search_region": [0, 0, 100, 100],
                        "match_mode": "exact",
                        "action": "fixed_click",
                    },
                    "character": {
                        "texts": ["开始游戏"],
                        "region_source": "configured",
                        "search_region": [0, 0, 100, 100],
                        "match_mode": "exact",
                        "action": "click",
                    },
                },
            },
            "remote_confirm_key": "enter",
        }
        invalid_sections = (
            {"enable": "False"},
            {"enable": True, "confirm_frames": 1.5},
            {"enable": True, "cancel_confirm_misses": 0},
            {
                "enable": True,
                "flow_template_reference_size": [0, 3579],
            },
            {"enable": True, "channel_points": []},
            {"enable": True, "channel_click_count": 0},
            {"enable": True, "channel_click_count": 3},
            {"enable": True, "connect_enter_retry_delay": -0.1},
            {"enable": True, "connect_enter_max_attempts": 0},
            {"enable": True, "channel_double_click_interval": -0.01},
            {"enable": True, "remote_confirm_key": "nonsense"},
            {"enable": True, "remote_mouse_mode": "desktop_absolute"},
            {"enable": True, "remote_mouse_mode": "visual_relative"},
            {"enable": True, "mouse_max_delta": 128},
            {"enable": True, "mouse_probe_delta": 128},
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template":
                    "misc/auto_relogin_cursor_small_cn.png",
                "cursor_hotspot": [-1, 6],
            },
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template": "",
            },
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template":
                    "misc/auto_relogin_cursor_small_cn.png",
                "disconnect_cursor_hotspot": [-1, 4],
            },
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template":
                    "misc/auto_relogin_cursor_small_cn.png",
                "cursor_min_score": 0,
            },
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template":
                    "misc/auto_relogin_cursor_small_cn.png",
                "cursor_uniqueness_margin": 0,
            },
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template":
                    "misc/auto_relogin_cursor_small_cn.png",
                "cursor_mask_erode_pixels": -1,
            },
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template":
                    "misc/auto_relogin_cursor_small_cn.png",
                "mouse_cursor_rescue_deltas": [[0, 0]],
            },
            {
                "enable": True,
                "remote_mouse_mode": "visual_relative",
                "cursor_template": "misc/auto_relogin_cursor_cn.png",
                "disconnect_cursor_template":
                    "misc/auto_relogin_cursor_small_cn.png",
                "mouse_cursor_rescue_deltas": [[32768, 0]],
            },
            {"enable": True, "ocr": {"enable": "yes"}},
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "min_score": 0,
                },
            },
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "box_threshold": 2,
                },
            },
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "confirm_frames": 0,
                },
            },
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "max_frame_age": 0,
                },
            },
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "max_center_drift": [-1, 10],
                },
            },
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "targets": {},
                },
            },
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "targets": {
                        "disconnect": {
                            **valid_section["ocr"]["targets"]["disconnect"],
                            "action": "keypress",
                        },
                    },
                },
            },
            {
                "enable": True,
                "ocr": {
                    **valid_section["ocr"],
                    "targets": {
                        "world": {
                            "texts": ["漂漂猪"],
                            "region_source": "configured",
                            "search_region": [0, 0, 100, 100],
                            "match_mode": "contains",
                            "action": "enter",
                        },
                    },
                },
            },
        )
        for override in invalid_sections:
            with self.subTest(override=override):
                bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
                section = deepcopy(valid_section)
                section.update(override)
                cfg = {
                    "bot": {"mode": "normal"},
                    "monster_detect": {},
                    "auto_relogin": section,
                }
                self.assertEqual(bot.load_config(cfg), -1)

    def test_invalid_absolute_mouse_geometry_fails_during_load(self):
        invalid_sections = (
            {
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 0, 768],
            },
            {
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [3000, 721, 1366, 768],
            },
            {
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, True, 768],
            },
            {
                "magpie_source_rect": [1235, 721, 1366, 768],
            },
        )
        for section in invalid_sections:
            with self.subTest(section=section):
                bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
                cfg = {
                    "bot": {"mode": "normal"},
                    "monster_detect": {},
                    "esp32_hid": section,
                }

                self.assertEqual(bot.load_config(cfg), -1)

    def test_run_once_checks_relogin_before_saved_minimap_geometry(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.profiler = Mock()
        bot.is_need_show_debug_window = False
        bot.get_img_frame = Mock(
            return_value=np.zeros((40, 60, 3), dtype=np.uint8)
        )
        call_order = []
        bot._auto_relogin_state = "idle"
        bot.resume_input_after_capture = Mock(
            side_effect=lambda: call_order.append("resume")
        )
        bot._check_auto_relogin_screen = Mock(
            side_effect=lambda: call_order.append("relogin") or True
        )
        bot.apply_saved_minimap_geometry = Mock(return_value=True)

        self.assertEqual(bot.run_once(), -1)

        bot._check_auto_relogin_screen.assert_called_once_with()
        self.assertEqual(call_order, ["relogin", "resume"])
        bot.apply_saved_minimap_geometry.assert_not_called()
        bot.profiler.mark.assert_called_once_with("Auto Relogin")

    def test_saved_minimap_recovery_requires_matching_detected_structure(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.minimap_geometry = {"minimap_rect": (10, 20, 60, 30)}
        bot.loc_minimap = (10, 20)
        bot.img_minimap_screen = np.zeros((30, 60, 3), dtype=np.uint8)

        # Dynamic detection includes a one-pixel border around saved geometry.
        self.assertTrue(
            bot._auto_relogin_minimap_structure_valid((9, 19, 62, 32))
        )
        self.assertFalse(
            bot._auto_relogin_minimap_structure_valid((100, 100, 62, 32))
        )
        self.assertFalse(bot._auto_relogin_minimap_structure_valid(None))


if __name__ == "__main__":
    unittest.main()
