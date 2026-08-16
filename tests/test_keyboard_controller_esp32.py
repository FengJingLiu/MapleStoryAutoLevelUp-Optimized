import unittest
from unittest.mock import Mock, call, patch

from src.input import KeyBoardController as controller_module
from src.input.Esp32HidClient import Esp32HidTapUncertainError


class KeyboardControllerEsp32RoutingTests(unittest.TestCase):
    def setUp(self):
        self.previous_client = controller_module._input_client
        self.previous_allowed = controller_module._input_allowed.is_set()
        self.previous_recovery_until = controller_module._input_recovery_until
        controller_module._input_client = Mock()
        controller_module._input_allowed.clear()
        controller_module.clear_input_recovery()

    def tearDown(self):
        controller_module._input_client = self.previous_client
        controller_module._input_recovery_until = self.previous_recovery_until
        if self.previous_allowed:
            controller_module._input_allowed.set()
        else:
            controller_module._input_allowed.clear()

    def test_tap_is_blocked_without_foreground_permission(self):
        self.assertFalse(controller_module.press_key("space", 0.075))
        controller_module._input_client.tap.assert_not_called()

    def test_tap_routes_to_esp32_when_foreground_is_allowed(self):
        controller_module._input_allowed.set()

        controller_module.press_key("space", 0.075)

        controller_module._input_client.tap.assert_called_once_with(
            "space", 75
        )

    def test_blank_tap_is_disabled_without_network_input(self):
        controller_module._input_allowed.set()

        self.assertFalse(controller_module.press_key("   "))

        controller_module._input_client.tap.assert_not_called()

    def test_uncertain_tap_is_consumed_without_high_level_replay(self):
        controller_module._input_allowed.set()
        controller_module._input_client.tap.side_effect = (
            Esp32HidTapUncertainError("response lost")
        )

        self.assertTrue(controller_module.press_key("space"))

        controller_module._input_client.tap.assert_called_once_with("space", 50)

    def test_definitely_rejected_tap_remains_retryable(self):
        controller_module._input_allowed.set()
        controller_module._input_client.tap.side_effect = RuntimeError(
            "ERR INVALID_KEY_OR_ARGUMENT"
        )

        self.assertFalse(controller_module.press_key("space"))

        controller_module._input_client.tap.assert_called_once_with("space", 50)

    def test_release_all_bypasses_foreground_guard(self):
        controller_module.release_all_keys()

        controller_module._input_client.release_all.assert_called_once_with()

    @staticmethod
    def make_session_recovery_controller():
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.session_recovery_active = False
        controller.game_ui_active = False
        controller.cfg = {}
        controller.is_need_force_heal = True
        controller.cmd_left_right = "left"
        controller.cmd_up_down = "up"
        controller.cmd_action = "attack"
        controller._last_source_action = "attack"
        controller.cached_facing = "left"
        controller.cmd_left_right_last = "left"
        controller.cmd_up_down_last = "up"
        controller.direction_held_since = 1.0
        controller.is_game_window_active = Mock(return_value=True)
        return controller

    def test_session_recovery_gates_worker_but_allows_explicit_enter(self):
        controller_module._input_allowed.set()
        controller = self.make_session_recovery_controller()

        self.assertTrue(
            controller.suspend_automation_for_session_recovery()
        )

        self.assertTrue(controller.session_recovery_active)
        self.assertFalse(controller_module._input_allowed.is_set())
        self.assertEqual(
            (
                controller.cmd_left_right,
                controller.cmd_up_down,
                controller.cmd_action,
            ),
            ("none", "none", "none"),
        )
        self.assertFalse(controller.is_need_force_heal)
        controller_module._input_client.release_all.assert_called_once_with()

        # Normal automatic paths remain closed, including a stale movement
        # snapshot captured by the worker before the suspension.
        self.assertFalse(controller_module.press_key("ins"))
        self.assertFalse(controller.update_movement_state("left", "up"))
        controller_module._input_client.set_state.assert_not_called()

        self.assertTrue(controller.press_session_recovery_key("enter"))
        controller_module._input_client.tap.assert_called_once_with(
            "enter", 50
        )

        self.assertTrue(
            controller.resume_automation_after_session_recovery()
        )
        self.assertFalse(controller.session_recovery_active)
        self.assertTrue(controller_module._input_allowed.is_set())

    def test_paused_user_gate_blocks_explicit_session_recovery_key(self):
        controller = self.make_session_recovery_controller()
        controller.suspend_automation_for_session_recovery()
        controller.is_enable = False

        self.assertFalse(controller.press_session_recovery_key("enter"))

        controller_module._input_client.tap.assert_not_called()

    def test_session_recovery_focuses_next_window_before_enter(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True

        with patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(
                controller.focus_next_window_and_press_session_recovery_key(
                    "enter",
                    focus_keys=["alt", "tab"],
                    focus_hold=0.10,
                    settle_delay=0.50,
                    duration=0.10,
                )
            )

        controller_module._input_client.set_state.assert_called_once_with(
            ("alt", "tab")
        )
        controller_module._input_client.release_all.assert_called_once_with()
        controller_module._input_client.tap.assert_called_once_with(
            "enter", 100
        )
        self.assertEqual(sleep.call_args_list, [call(0.10), call(0.50)])

    def test_failed_focus_switch_releases_keys_and_never_sends_enter(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
        controller_module._input_client.set_state.side_effect = RuntimeError(
            "ERR INVALID_KEY_OR_ARGUMENT"
        )

        self.assertFalse(
            controller.focus_next_window_and_press_session_recovery_key(
                "enter"
            )
        )

        controller_module._input_client.release_all.assert_called_once_with()
        controller_module._input_client.tap.assert_not_called()

    def test_session_recovery_click_normalizes_capture_point(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
        controller_module._input_client.mouse_click_at.return_value = (
            "OK MOUSE_CLICK_AT 16384 16384 0x01 50ms"
        )

        self.assertTrue(
            controller.click_session_recovery_point(
                1789,
                1006,
                3579,
                2013,
                button="left",
                duration=0.05,
            )
        )

        controller_module._input_client.mouse_click_at.assert_called_once_with(
            16384,
            16384,
            "left",
            50,
        )

    def test_session_recovery_click_applies_calibrated_magpie_geometry(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
        controller.cfg = {
            "esp32_hid": {
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 1366, 768],
            }
        }
        controller_module._input_client.mouse_click_at.return_value = (
            "OK MOUSE_CLICK_AT 18129 17408 0x01 50ms"
        )

        self.assertTrue(controller.click_session_recovery_point(
            2500, 1200, 3840, 2160
        ))

        controller_module._input_client.mouse_click_at.assert_called_once_with(
            18129,
            17408,
            "left",
            50,
        )

    def test_calibrated_absolute_geometry_preserves_frame_endpoints(self):
        cfg = {
            "esp32_hid": {
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 1366, 768],
            }
        }

        self.assertEqual(
            controller_module.capture_point_to_absolute_hid(
                cfg, 0, 0, 3840, 2160
            ),
            (10541, 10943),
        )
        self.assertEqual(
            controller_module.capture_point_to_absolute_hid(
                cfg, 3839, 2159, 3840, 2160
            ),
            (22192, 22583),
        )
        self.assertEqual(
            controller_module.capture_point_to_absolute_hid(
                cfg, 2200, 800, 3840, 2160
            ),
            (17216, 15253),
        )
        self.assertEqual(
            controller_module.capture_point_to_absolute_hid(
                cfg, 3000, 1500, 3840, 2160
            ),
            (19648, 19032),
        )

    def test_invalid_magpie_geometry_fails_closed_before_hid_send(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
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
                controller.cfg = {"esp32_hid": section}
                controller_module._input_client.reset_mock()

                self.assertFalse(controller.click_session_recovery_point(
                    2500, 1200, 3840, 2160
                ))

                controller_module._input_client.mouse_click_at.assert_not_called()

    def test_session_recovery_click_is_consumed_when_ack_is_uncertain(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
        controller_module._input_client.mouse_click_at.side_effect = (
            Esp32HidTapUncertainError("response lost")
        )

        self.assertTrue(
            controller.click_session_recovery_point(100, 200, 3579, 2013)
        )

        controller_module._input_client.mouse_click_at.assert_called_once()

    def test_session_recovery_click_checks_every_safety_gate(self):
        cases = {
            "session": lambda controller: setattr(
                controller, "session_recovery_active", False
            ),
            "user_pause": lambda controller: setattr(
                controller, "is_enable", False
            ),
            "capture": lambda controller: setattr(
                controller, "capture_available", False
            ),
            "terminate": lambda controller: setattr(
                controller, "is_terminated", True
            ),
            "foreground": lambda controller: (
                controller.is_game_window_active.reset_mock(),
                setattr(
                    controller.is_game_window_active,
                    "return_value",
                    False,
                ),
            ),
        }
        for name, close_gate in cases.items():
            with self.subTest(gate=name):
                controller = self.make_session_recovery_controller()
                controller.session_recovery_active = True
                close_gate(controller)
                controller_module._input_client.reset_mock()

                self.assertFalse(
                    controller.click_session_recovery_point(
                        100, 200, 3579, 2013
                    )
                )

                controller_module._input_client.mouse_click_at.assert_not_called()

    def test_session_recovery_click_rejects_invalid_frame_points(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
        invalid_points = (
            (-1, 0, 3579, 2013),
            (3579, 0, 3579, 2013),
            (0, 2013, 3579, 2013),
            (0, 0, 1, 2013),
            (float("nan"), 0, 3579, 2013),
        )
        for point in invalid_points:
            with self.subTest(point=point):
                self.assertFalse(
                    controller.click_session_recovery_point(*point)
                )

        controller_module._input_client.mouse_click_at.assert_not_called()

    def test_session_recovery_relative_mouse_move_and_current_click(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
        controller_module._input_client.mouse_move.return_value = (
            "OK MOUSE_MOVE 24 -11 0"
        )
        controller_module._input_client.mouse_click.return_value = (
            "OK MOUSE_CLICK 0x01 50ms"
        )

        self.assertTrue(controller.move_session_recovery_mouse(24, -11))
        self.assertTrue(
            controller.click_session_recovery_mouse(
                button="left", duration=0.05
            )
        )

        controller_module._input_client.mouse_move.assert_called_once_with(
            24, -11, 0
        )
        controller_module._input_client.mouse_click.assert_called_once_with(
            "left", 50
        )
        controller_module._input_client.mouse_click_at.assert_not_called()

    def test_uncertain_relative_move_is_consumed_for_visual_recheck(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = True
        controller_module._input_client.mouse_move.side_effect = (
            Esp32HidTapUncertainError("response lost")
        )

        self.assertTrue(controller.move_session_recovery_mouse(10, 5))

        controller_module._input_client.mouse_move.assert_called_once_with(
            10, 5, 0
        )

    def test_relative_recovery_mouse_checks_session_gate(self):
        controller = self.make_session_recovery_controller()
        controller.session_recovery_active = False

        self.assertFalse(controller.move_session_recovery_mouse(10, 5))
        self.assertFalse(controller.click_session_recovery_mouse())

        controller_module._input_client.mouse_move.assert_not_called()
        controller_module._input_client.mouse_click.assert_not_called()

    def test_exclusive_game_ui_gate_allows_only_explicit_absolute_input(self):
        controller_module._input_allowed.set()
        controller = self.make_session_recovery_controller()
        controller.cfg = {
            "esp32_hid": {
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": [1235, 721, 1366, 768],
            }
        }
        controller_module._input_client.mouse_click_at.return_value = (
            "OK MOUSE_CLICK_AT 18129 17408 0x01 50ms"
        )

        self.assertTrue(controller.suspend_automation_for_game_ui())
        self.assertTrue(controller.game_ui_active)
        self.assertFalse(controller_module._input_allowed.is_set())
        self.assertFalse(controller_module.press_key("left"))
        self.assertTrue(controller.press_game_ui_key("p"))
        self.assertTrue(controller.click_game_ui_point(
            2500, 1200, 3840, 2160
        ))

        controller_module._input_client.tap.assert_called_once_with("p", 50)
        controller_module._input_client.mouse_click_at.assert_called_once_with(
            18129, 17408, "left", 50
        )
        self.assertTrue(controller.resume_automation_after_game_ui())
        self.assertFalse(controller.game_ui_active)
        self.assertTrue(controller_module._input_allowed.is_set())

    def test_game_ui_absolute_click_requires_exclusive_gate(self):
        controller = self.make_session_recovery_controller()

        self.assertFalse(controller.click_game_ui_point(
            100, 200, 3840, 2160
        ))
        self.assertFalse(controller.press_game_ui_key("p"))

        controller_module._input_client.mouse_click_at.assert_not_called()
        controller_module._input_client.tap.assert_not_called()

    def test_uncertain_game_ui_ack_stops_fixed_sequence(self):
        controller = self.make_session_recovery_controller()
        controller.game_ui_active = True
        controller_module._input_client.tap.side_effect = (
            Esp32HidTapUncertainError("tap response lost")
        )
        controller_module._input_client.mouse_click_at.side_effect = (
            Esp32HidTapUncertainError("click response lost")
        )

        self.assertFalse(controller.press_game_ui_key("p"))
        self.assertFalse(controller.click_game_ui_point(
            100, 200, 3840, 2160
        ))

        controller_module._input_client.tap.assert_called_once_with("p", 50)
        controller_module._input_client.mouse_click_at.assert_called_once_with(
            854,
            3035,
            "left",
            50,
        )

    def test_remote_game_ui_click_requires_magpie_calibration(self):
        controller = self.make_session_recovery_controller()
        controller.game_ui_active = True
        controller.cfg = {
            "esp32_hid": {
                "remote_target": True,
                "absolute_desktop_rect": [0, 0, 3840, 2160],
                "magpie_source_rect": None,
            }
        }

        self.assertFalse(controller.click_game_ui_point(
            100, 200, 3840, 2160
        ))

        controller_module._input_client.mouse_click_at.assert_not_called()

    def test_failed_game_ui_suspension_stays_paused_without_hidden_latch(self):
        controller_module._input_allowed.set()
        controller = self.make_session_recovery_controller()
        controller_module._input_client.release_all.side_effect = RuntimeError(
            "serial unavailable"
        )

        self.assertFalse(controller.suspend_automation_for_game_ui())

        self.assertFalse(controller.is_enable)
        self.assertFalse(controller.game_ui_active)
        self.assertFalse(controller_module._input_allowed.is_set())

    def test_movement_axes_are_sent_as_one_atomic_state(self):
        controller_module._input_allowed.set()
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cmd_left_right = "left"
        controller.cmd_up_down = "up"
        controller.cmd_action = "none"
        controller.cmd_left_right_last = "none"
        controller.cmd_up_down_last = "none"

        controller.update_movement_state()

        controller_module._input_client.set_state.assert_called_once_with(
            ["left", "up"]
        )
        self.assertEqual(controller.cached_facing, "left")

    @staticmethod
    def make_directional_controller(facing=None):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.command_lock = controller_module.threading.RLock()
        controller.cached_facing = facing
        controller.cmd_left_right = "none"
        controller.cmd_up_down = "none"
        controller.cmd_action = "attack"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.character_turn_delay = 0.08
        controller.attack_recovery_delay = 0.90
        controller.attack_key = "control"
        controller.t_last_skill = 0.0
        return controller

    def test_directional_attack_turns_then_attacks_as_one_transaction(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="right")

        calls_seen_at_turn_delay = []

        def record_turn_delay(duration):
            calls_seen_at_turn_delay.extend(
                controller_module._input_client.method_calls
            )
            self.assertEqual(duration, 0.08)

        with patch.object(
            controller_module.time,
            "sleep",
            side_effect=record_turn_delay,
        ) as sleep:
            self.assertTrue(controller.perform_directional_attack("left"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["left"]),
                call.set_state([]),
                call.tap("control", 50),
            ],
        )
        sleep.assert_called_once_with(0.08)
        self.assertEqual(
            calls_seen_at_turn_delay,
            [call.set_state(["left"])],
        )
        self.assertEqual(controller.cached_facing, "left")
        self.assertGreater(controller_module.input_recovery_remaining(), 0.80)

    def test_same_facing_attack_does_not_send_direction_key(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="left")

        with patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(controller.perform_directional_attack("left"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state([]), call.tap("control", 50)],
        )
        sleep.assert_not_called()

    def test_directional_aoe_uses_its_own_key_and_recovery_delay(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="right")

        with patch.object(controller_module.time, "sleep"):
            self.assertTrue(controller.perform_directional_attack(
                "left",
                attack_key="shift",
                recovery_delay=1.5,
            ))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["left"]),
                call.set_state([]),
                call.tap("shift", 50),
            ],
        )
        self.assertGreater(controller_module.input_recovery_remaining(), 1.4)

    def test_power_knockback_uses_s_and_its_own_recovery_delay(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="right")

        with patch.object(controller_module.time, "sleep"):
            self.assertTrue(controller.perform_directional_attack(
                "left",
                attack_key="s",
                recovery_delay=1.1,
            ))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["left"]),
                call.set_state([]),
                call.tap("s", 50),
            ],
        )
        self.assertGreater(controller_module.input_recovery_remaining(), 1.0)

    def test_other_input_cannot_enter_directional_attack_transaction(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="right")
        turn_started = controller_module.threading.Event()
        finish_turn = controller_module.threading.Event()
        other_started = controller_module.threading.Event()
        results = {}

        def hold_turn(_duration):
            turn_started.set()
            self.assertTrue(finish_turn.wait(timeout=1.0))

        def attack():
            results["attack"] = controller.perform_directional_attack("left")

        def other_input():
            other_started.set()
            results["other"] = controller_module.press_key("space")

        with patch.object(controller_module.time, "sleep", side_effect=hold_turn):
            attack_thread = controller_module.threading.Thread(target=attack)
            attack_thread.start()
            self.assertTrue(turn_started.wait(timeout=1.0))

            other_thread = controller_module.threading.Thread(target=other_input)
            other_thread.start()
            self.assertTrue(other_started.wait(timeout=1.0))
            self.assertTrue(other_thread.is_alive())

            finish_turn.set()
            attack_thread.join(timeout=1.0)
            other_thread.join(timeout=1.0)

        self.assertFalse(attack_thread.is_alive())
        self.assertFalse(other_thread.is_alive())
        self.assertTrue(results["attack"])
        self.assertFalse(results["other"])
        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["left"]),
                call.set_state([]),
                call.tap("control", 50),
            ],
        )

    def test_pause_during_same_facing_stop_prevents_late_attack(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="left")

        def pause_during_stop(_keys):
            controller_module._input_allowed.clear()
            return True

        controller_module._input_client.set_state.side_effect = pause_during_stop

        self.assertFalse(controller.perform_directional_attack("left"))

        controller_module._input_client.set_state.assert_called_once_with([])
        controller_module._input_client.tap.assert_not_called()
        self.assertIsNone(controller.cached_facing)
        self.assertEqual(controller_module.input_recovery_remaining(), 0.0)

    def test_transport_failure_invalidates_facing_cache(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="left")
        controller_module._input_client.set_state.side_effect = RuntimeError(
            "serial disconnected"
        )

        self.assertFalse(controller.perform_directional_attack("left"))

        self.assertIsNone(controller.cached_facing)
        controller_module._input_client.tap.assert_not_called()
        self.assertEqual(controller_module.input_recovery_remaining(), 0.0)

    def test_attack_recovery_blocks_every_regular_hid_command(self):
        controller_module._input_allowed.set()
        controller_module._set_input_recovery(0.90)

        self.assertFalse(controller_module.key_down("left"))
        self.assertFalse(controller_module.key_up("left"))
        self.assertFalse(controller_module.set_key_state(["right"]))
        self.assertFalse(controller_module.press_key("space"))
        self.assertEqual(controller_module._input_client.method_calls, [])

    def test_safety_release_bypasses_but_preserves_attack_recovery(self):
        controller_module._input_allowed.set()
        controller_module._set_input_recovery(0.90)

        self.assertTrue(controller_module.release_all_keys())

        controller_module._input_client.release_all.assert_called_once_with()
        self.assertGreater(controller_module.input_recovery_remaining(), 0.80)
        self.assertFalse(controller_module.press_key("space"))

    def test_stale_client_close_does_not_clear_new_session_recovery(self):
        controller_module._input_allowed.set()
        controller_module._set_input_recovery(0.90)
        active_client = controller_module._input_client
        stale_client = Mock()

        controller_module.close_esp32_input(stale_client)

        self.assertIs(controller_module._input_client, active_client)
        self.assertTrue(controller_module._input_allowed.is_set())
        self.assertGreater(controller_module.input_recovery_remaining(), 0.80)
        stale_client.close.assert_not_called()

    def test_uncertain_directional_attack_still_starts_full_recovery(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_controller(facing="left")
        controller_module._input_client.tap.side_effect = (
            Esp32HidTapUncertainError("response lost")
        )

        self.assertTrue(controller.perform_directional_attack("left"))

        self.assertGreater(controller_module.input_recovery_remaining(), 0.80)
        self.assertFalse(controller_module.press_key("space"))

    @staticmethod
    def make_directional_jump_controller(
        direction_last="none", held_since=None
    ):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.command_lock = controller_module.threading.RLock()
        controller.cached_facing = (
            direction_last if direction_last in {"left", "right"} else None
        )
        controller.cmd_left_right_last = direction_last
        controller.cmd_up_down_last = "none"
        controller.direction_held_since = held_since
        controller.directional_jump_runup_ms = 180
        controller.cfg = {"key": {"jump": "space"}}
        return controller

    def test_directional_jump_from_rest_builds_full_runup_before_tap(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_jump_controller()
        calls_seen_while_running_up = []

        def record_runup(duration):
            self.assertAlmostEqual(duration, 0.18)
            calls_seen_while_running_up.extend(
                controller_module._input_client.method_calls
            )

        with patch.object(
            controller_module.time, "monotonic", return_value=10.0
        ), patch.object(
            controller_module.time, "sleep", side_effect=record_runup
        ) as sleep:
            self.assertTrue(controller.perform_directional_jump("left"))

        sleep.assert_called_once()
        self.assertEqual(
            calls_seen_while_running_up,
            [call.set_state(["left"])],
        )
        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state(["left"]), call.tap("space", 50)],
        )
        self.assertEqual(controller.cmd_left_right_last, "left")
        self.assertEqual(controller.direction_held_since, 10.0)

    def test_directional_jump_only_waits_for_missing_runup(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_jump_controller(
            direction_last="right", held_since=9.9
        )

        with patch.object(
            controller_module.time, "monotonic", return_value=10.0
        ), patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(controller.perform_directional_jump("right"))

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.08)
        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state(["right"]), call.tap("space", 50)],
        )

    def test_directional_jump_with_existing_momentum_does_not_wait(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_jump_controller(
            direction_last="left", held_since=9.0
        )

        with patch.object(
            controller_module.time, "monotonic", return_value=10.0
        ), patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(controller.perform_directional_jump("left"))

        sleep.assert_not_called()
        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state(["left"]), call.tap("space", 50)],
        )

    def test_pause_during_directional_jump_runup_prevents_late_tap(self):
        controller_module._input_allowed.set()
        controller = self.make_directional_jump_controller()

        def pause_during_runup(_duration):
            controller_module._input_allowed.clear()

        with patch.object(
            controller_module.time, "monotonic", return_value=10.0
        ), patch.object(
            controller_module.time, "sleep", side_effect=pause_during_runup
        ):
            self.assertFalse(controller.perform_directional_jump("right"))

        controller_module._input_client.set_state.assert_called_once_with(
            ["right"]
        )
        controller_module._input_client.tap.assert_not_called()
        self.assertIsNone(controller.cached_facing)
        self.assertIsNone(controller.direction_held_since)

    @staticmethod
    def make_stationary_jump_controller():
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.command_lock = controller_module.threading.RLock()
        controller.cached_facing = "right"
        controller.cmd_left_right_last = "right"
        controller.cmd_up_down_last = "none"
        controller.jump_up_settle_delay = 0.15
        controller.cfg = {"key": {"jump": "space"}}
        return controller

    @staticmethod
    def make_rope_controller():
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.command_lock = controller_module.threading.RLock()
        controller.cached_facing = "right"
        controller.cmd_left_right_last = "right"
        controller.cmd_up_down_last = "none"
        controller.direction_held_since = None
        controller.direction_held_generation = None
        controller.rope_climb_runup_ms = 180
        controller.rope_climb_align_nudge_ms = 30
        controller.cfg = {"key": {"jump": "space"}}
        return controller

    def test_stationary_jump_stops_and_settles_before_tap(self):
        controller_module._input_allowed.set()
        controller = self.make_stationary_jump_controller()
        calls_seen_while_settling = []

        def record_settle_delay(duration):
            self.assertEqual(duration, 0.15)
            calls_seen_while_settling.extend(
                controller_module._input_client.method_calls
            )

        with patch.object(
            controller_module.time,
            "sleep",
            side_effect=record_settle_delay,
        ) as sleep:
            self.assertTrue(controller.perform_stationary_jump())

        sleep.assert_called_once_with(0.15)
        self.assertEqual(
            calls_seen_while_settling,
            [call.set_state([])],
        )
        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state([]), call.tap("space", 50)],
        )
        # Releasing movement does not change the in-game facing direction.
        self.assertEqual(controller.cached_facing, "right")

    def test_pause_while_stationary_jump_settles_prevents_late_tap(self):
        controller_module._input_allowed.set()
        controller = self.make_stationary_jump_controller()

        def pause_during_settle(_duration):
            controller_module._input_allowed.clear()

        with patch.object(
            controller_module.time,
            "sleep",
            side_effect=pause_during_settle,
        ):
            self.assertFalse(controller.perform_stationary_jump())

        controller_module._input_client.set_state.assert_called_once_with([])
        controller_module._input_client.tap.assert_not_called()
        self.assertIsNone(controller.cached_facing)

    def test_other_input_cannot_enter_stationary_jump_settle(self):
        controller_module._input_allowed.set()
        controller = self.make_stationary_jump_controller()
        settle_started = controller_module.threading.Event()
        finish_settle = controller_module.threading.Event()
        other_started = controller_module.threading.Event()
        results = {}

        def hold_settle(_duration):
            settle_started.set()
            self.assertTrue(finish_settle.wait(timeout=1.0))

        def jump():
            results["jump"] = controller.perform_stationary_jump()

        def other_input():
            other_started.set()
            results["other"] = controller_module.press_key("q")

        with patch.object(
            controller_module.time,
            "sleep",
            side_effect=hold_settle,
        ):
            jump_thread = controller_module.threading.Thread(target=jump)
            jump_thread.start()
            self.assertTrue(settle_started.wait(timeout=1.0))

            other_thread = controller_module.threading.Thread(
                target=other_input
            )
            other_thread.start()
            self.assertTrue(other_started.wait(timeout=1.0))
            self.assertTrue(other_thread.is_alive())

            finish_settle.set()
            jump_thread.join(timeout=1.0)
            other_thread.join(timeout=1.0)

        self.assertFalse(jump_thread.is_alive())
        self.assertFalse(other_thread.is_alive())
        self.assertTrue(results["jump"])
        self.assertTrue(results["other"])
        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state([]),
                call.tap("space", 50),
                call.tap("q", 50),
            ],
        )

    def test_stationary_jump_stop_failure_does_not_wait_or_tap(self):
        controller_module._input_allowed.set()
        controller = self.make_stationary_jump_controller()
        controller_module._input_client.set_state.side_effect = RuntimeError(
            "serial disconnected"
        )

        with patch.object(controller_module.time, "sleep") as sleep:
            self.assertFalse(controller.perform_stationary_jump())

        sleep.assert_not_called()
        controller_module._input_client.tap.assert_not_called()
        self.assertIsNone(controller.cached_facing)

    def test_only_directionless_up_jump_uses_settle_transaction(self):
        predicate = controller_module.KeyBoardController \
            .is_stationary_jump_command

        self.assertTrue(predicate("none", "none", "jump"))
        self.assertTrue(predicate("stop", "stop", "jump"))
        self.assertFalse(predicate("left", "none", "jump"))
        self.assertFalse(predicate("right", "none", "jump"))
        self.assertFalse(predicate("none", "down", "jump"))
        self.assertFalse(predicate("none", "none", "teleport"))

    def test_only_horizontal_jump_uses_runup_transaction(self):
        predicate = controller_module.KeyBoardController \
            .is_directional_jump_command

        self.assertTrue(predicate("left", "none", "jump"))
        self.assertTrue(predicate("right", "stop", "jump"))
        self.assertFalse(predicate("none", "none", "jump"))
        self.assertFalse(predicate("left", "down", "jump"))
        self.assertFalse(predicate("left", "none", "teleport"))

    def test_repeated_source_action_is_edge_triggered(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cmd_action = "none"
        controller._last_source_action = "none"

        controller.set_command("left none jump")
        self.assertEqual(controller.cmd_action, "jump")

        controller.cmd_action = "none"  # consumed by the input loop
        controller.set_command("left none jump")
        self.assertEqual(controller.cmd_action, "none")

        controller.set_command("left none none")
        controller.set_command("left none jump")
        self.assertEqual(controller.cmd_action, "jump")

    def test_consumed_portal_sweep_action_can_be_queued_again(self):
        for action in ("portal_sweep_left", "portal_sweep_right"):
            with self.subTest(action=action):
                controller = controller_module.KeyBoardController.__new__(
                    controller_module.KeyBoardController
                )
                controller.cmd_action = "none"
                controller._last_source_action = "none"

                controller.set_command(f"none up {action}")
                controller._consume_action(action)
                controller.set_command(f"none up {action}")

                self.assertEqual(controller.cmd_action, action)

    def test_consumed_rope_alignment_action_can_be_queued_again(self):
        for action in ("rope_align_left", "rope_align_right"):
            with self.subTest(action=action):
                controller = controller_module.KeyBoardController.__new__(
                    controller_module.KeyBoardController
                )
                controller.cmd_action = "none"
                controller._last_source_action = "none"

                controller.set_command(f"none none {action}")
                controller._consume_action(action)
                controller.set_command(f"none none {action}")

                self.assertEqual(controller.cmd_action, action)

    def test_rope_alignment_is_one_safe_short_tap_transaction(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()
        controller.rope_climb_align_nudge_ms = 34

        self.assertTrue(controller.perform_rope_alignment_nudge("left"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state([]), call.tap("left", 34)],
        )
        self.assertEqual(controller.cmd_left_right_last, "none")
        self.assertEqual(controller.cmd_up_down_last, "none")
        self.assertEqual(controller.cached_facing, "left")

    def test_pause_after_rope_alignment_state_prevents_late_tap(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()

        def pause_after_state(_keys):
            controller_module._input_allowed.clear()

        controller_module._input_client.set_state.side_effect = \
            pause_after_state

        self.assertFalse(controller.perform_rope_alignment_nudge("right"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state([])],
        )
        self.assertIsNone(controller.cached_facing)

    def test_rope_mount_runs_then_adds_up_and_jumps(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()
        calls_seen_during_runup = []

        def record_runup(duration):
            self.assertEqual(duration, 0.18)
            calls_seen_during_runup.extend(
                controller_module._input_client.method_calls
            )

        with patch.object(
            controller_module.time,
            "sleep",
            side_effect=record_runup,
        ) as sleep:
            self.assertTrue(controller.perform_rope_mount("left"))

        sleep.assert_called_once_with(0.18)
        self.assertEqual(calls_seen_during_runup, [call.set_state(["left"])])
        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["left"]),
                call.set_state(["left", "up"]),
                call.tap("space", 50),
                call.set_state(["up"]),
            ],
        )
        self.assertEqual(controller.cmd_left_right_last, "none")
        self.assertEqual(controller.cmd_up_down_last, "up")
        self.assertEqual(controller.cached_facing, "left")

    def test_rope_mount_keeps_existing_same_direction_run_without_pause(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()
        controller.cmd_left_right_last = "right"
        controller.direction_held_since = 99.0

        with patch.object(
            controller_module.time, "monotonic", return_value=100.0
        ), patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(controller.perform_rope_mount("right"))

        sleep.assert_not_called()
        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["right"]),
                call.set_state(["right", "up"]),
                call.tap("space", 50),
                call.set_state(["up"]),
            ],
        )
        self.assertIsNone(controller.direction_held_since)

    def test_rope_mount_only_builds_missing_same_direction_runup(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()
        controller.cmd_left_right_last = "left"
        controller.direction_held_since = 99.9

        with patch.object(
            controller_module.time, "monotonic", return_value=100.0
        ), patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(controller.perform_rope_mount("left"))

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.08)

    def test_movement_tracking_feeds_same_direction_rope_runup(self):
        controller_module._input_allowed.set()
        controller_module._input_client.state_continuity_token = None
        controller = self.make_rope_controller()
        controller.cmd_left_right_last = "none"

        with patch.object(
            controller_module.time, "monotonic", return_value=100.0
        ):
            self.assertTrue(
                controller.update_movement_state("right", "none")
            )
        self.assertEqual(controller.direction_held_since, 100.0)

        with patch.object(
            controller_module.time, "monotonic", return_value=100.1
        ), patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(controller.perform_rope_mount("right"))

        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.08)

    def test_rope_mount_discards_hold_time_after_state_discontinuity(self):
        controller_module._input_allowed.set()
        controller_module._input_client.state_continuity_token = 8
        controller = self.make_rope_controller()
        controller.cmd_left_right_last = "right"
        controller.direction_held_since = 99.0
        controller.direction_held_generation = 7

        with patch.object(
            controller_module.time, "monotonic", return_value=100.0
        ), patch.object(controller_module.time, "sleep") as sleep:
            self.assertTrue(controller.perform_rope_mount("right"))

        sleep.assert_called_once_with(0.18)

    def test_pause_during_rope_jump_prevents_final_up_state(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()

        def pause_during_jump(_key, _duration):
            controller_module._input_allowed.clear()

        controller_module._input_client.tap.side_effect = pause_during_jump

        with patch.object(controller_module.time, "sleep"):
            self.assertFalse(controller.perform_rope_mount("right"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["right"]),
                call.set_state(["right", "up"]),
                call.tap("space", 50),
            ],
        )
        self.assertIsNone(controller.cached_facing)
        self.assertIsNone(controller.direction_held_since)
        self.assertIsNone(controller.direction_held_generation)

    def test_pause_during_rope_runup_prevents_late_state_or_tap(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()

        def pause_during_runup(_duration):
            controller_module._input_allowed.clear()

        with patch.object(
            controller_module.time,
            "sleep",
            side_effect=pause_during_runup,
        ):
            self.assertFalse(controller.perform_rope_mount("right"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state(["right"])],
        )
        self.assertIsNone(controller.cached_facing)

    def test_pause_during_rope_up_state_prevents_late_jump(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()

        def pause_on_up_state(keys):
            if keys == ["left", "up"]:
                controller_module._input_allowed.clear()

        controller_module._input_client.set_state.side_effect = \
            pause_on_up_state

        with patch.object(controller_module.time, "sleep"):
            self.assertFalse(controller.perform_rope_mount("left"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["left"]),
                call.set_state(["left", "up"]),
            ],
        )
        controller_module._input_client.tap.assert_not_called()
        self.assertIsNone(controller.cached_facing)

    def test_rope_mount_invalidates_when_final_up_state_fails(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()

        def fail_final_up_state(keys):
            if keys == ["up"]:
                raise RuntimeError("serial disconnected")

        controller_module._input_client.set_state.side_effect = \
            fail_final_up_state

        with patch.object(controller_module.time, "sleep"):
            self.assertFalse(controller.perform_rope_mount("right"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [
                call.set_state(["right"]),
                call.set_state(["right", "up"]),
                call.tap("space", 50),
                call.set_state(["up"]),
            ],
        )
        self.assertIsNone(controller.cached_facing)
        self.assertEqual(controller.cmd_left_right_last, "")
        self.assertEqual(controller.cmd_up_down_last, "")

    def test_rope_actions_reject_unknown_horizontal_direction(self):
        controller_module._input_allowed.set()
        controller = self.make_rope_controller()

        self.assertFalse(controller.perform_rope_alignment_nudge("none"))
        self.assertFalse(controller.perform_rope_mount("up"))

        self.assertEqual(controller_module._input_client.method_calls, [])

    def test_portal_sweep_holds_up_and_taps_horizontal_direction(self):
        controller_module._input_allowed.set()
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.command_lock = controller_module.threading.RLock()
        controller.cached_facing = "right"
        controller.cmd_left_right_last = "right"
        controller.cmd_up_down_last = "none"
        controller.portal_sweep_nudge_ms = 37

        self.assertTrue(controller.perform_portal_sweep_step("left"))

        self.assertEqual(
            controller_module._input_client.method_calls,
            [call.set_state(["up"]), call.tap("left", 37)],
        )
        self.assertEqual(controller.cmd_left_right_last, "none")
        self.assertEqual(controller.cmd_up_down_last, "up")
        self.assertEqual(controller.cached_facing, "left")

    def test_portal_sweep_rejects_unknown_horizontal_direction(self):
        controller_module._input_allowed.set()
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )

        self.assertFalse(controller.perform_portal_sweep_step("none"))

        controller_module._input_client.assert_not_called()

    def test_consumed_combat_action_can_be_queued_again(self):
        for action in ("attack", "directional_aoe", "power_knockback"):
            with self.subTest(action=action):
                controller = controller_module.KeyBoardController.__new__(
                    controller_module.KeyBoardController
                )
                controller.cmd_action = "none"
                controller._last_source_action = "none"

                controller.set_command(f"left none {action}")
                controller._consume_action(action)
                controller.set_command(f"left none {action}")

                self.assertEqual(controller.cmd_action, action)

    def test_disabled_control_does_not_require_an_esp32_connection(self):
        cfg = {
            "game_window": {"title": "Test Window"},
            "buff_skill": {"keys": []},
            "system": {
                "key_debounce_interval": 1,
                "fps_limit_keyboard_controller": 30,
            },
            "bot": {"attack": "directional"},
            "key": {"directional_attack": "w"},
        }
        with patch.object(
            controller_module, "configure_esp32_input"
        ) as configure, patch.object(
            controller_module, "close_esp32_input"
        ) as close, patch.object(
            controller_module.threading, "Thread"
        ) as thread_class:
            controller = controller_module.KeyBoardController(
                cfg, connect_input=False
            )

        configure.assert_not_called()
        close.assert_called_once_with()
        thread_class.return_value.start.assert_called_once_with()
        self.assertIsNone(controller.input_client)
        self.assertFalse(controller.is_enable)
        self.assertEqual(controller.rope_climb_runup_ms, 180)
        self.assertEqual(controller.rope_climb_align_nudge_ms, 30)
        self.assertEqual(controller.portal_sweep_nudge_ms, 30)

    def test_enabled_directional_aoe_requires_an_aoe_key(self):
        cfg = {
            "game_window": {"title": "Test Window"},
            "buff_skill": {"keys": []},
            "system": {
                "key_debounce_interval": 1,
                "fps_limit_keyboard_controller": 30,
            },
            "bot": {"attack": "directional"},
            "directional_aoe": {
                "enable": True,
                "attack_recovery_delay": 1.0,
            },
            "key": {
                "directional_attack": "control",
                "aoe_skill": "",
            },
        }

        with self.assertRaisesRegex(ValueError, "key.aoe_skill"):
            controller_module.KeyBoardController(cfg, connect_input=False)

    def test_directional_aoe_key_must_differ_from_normal_attack(self):
        cfg = {
            "game_window": {"title": "Test Window"},
            "buff_skill": {"keys": []},
            "system": {
                "key_debounce_interval": 1,
                "fps_limit_keyboard_controller": 30,
            },
            "bot": {"attack": "directional"},
            "directional_aoe": {
                "enable": True,
                "attack_recovery_delay": 1.0,
            },
            "key": {
                "directional_attack": "ctrl",
                "aoe_skill": "control",
            },
        }

        with self.assertRaisesRegex(ValueError, "must differ"):
            controller_module.KeyBoardController(cfg, connect_input=False)

    def test_enabled_power_knockback_requires_a_key(self):
        cfg = {
            "game_window": {"title": "Test Window"},
            "buff_skill": {"keys": []},
            "system": {
                "key_debounce_interval": 1,
                "fps_limit_keyboard_controller": 30,
            },
            "bot": {"attack": "directional"},
            "power_knockback": {
                "enable": True,
                "attack_recovery_delay": 0.9,
            },
            "key": {
                "directional_attack": "control",
                "power_knockback": "",
            },
        }

        with self.assertRaisesRegex(ValueError, "key.power_knockback"):
            controller_module.KeyBoardController(cfg, connect_input=False)

    def test_power_knockback_key_must_differ_from_normal_attack(self):
        cfg = {
            "game_window": {"title": "Test Window"},
            "buff_skill": {"keys": []},
            "system": {
                "key_debounce_interval": 1,
                "fps_limit_keyboard_controller": 30,
            },
            "bot": {"attack": "directional"},
            "power_knockback": {
                "enable": True,
                "attack_recovery_delay": 0.9,
            },
            "key": {
                "directional_attack": "s",
                "power_knockback": "s",
            },
        }

        with self.assertRaisesRegex(ValueError, "must differ"):
            controller_module.KeyBoardController(cfg, connect_input=False)

    def test_run_routes_horizontal_jump_to_runup_transaction(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "bot": {"attack": "directional"},
        }
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "left"
        controller.cmd_up_down = "none"
        controller.cmd_action = "jump"
        controller.input_client = controller_module._input_client
        controller.perform_directional_jump = Mock(return_value=True)
        controller.release_all_key = Mock()

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        controller.run()

        controller.perform_directional_jump.assert_called_once_with("left")
        self.assertEqual(controller.cmd_action, "none")

    def test_run_routes_directional_aoe_to_the_aoe_key(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "bot": {"attack": "directional"},
            "buff_skill": {"keys": [], "cooldown": [], "action_cooldown": 1},
            "key": {"aoe_skill": "shift"},
            "system": {"fps_limit_keyboard_controller": 30},
        }
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "left"
        controller.cmd_up_down = "none"
        controller.cmd_action = "directional_aoe"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.directional_aoe_key = "shift"
        controller.directional_aoe_recovery_delay = 1.2
        controller.attack_recovery_delay = 0.9
        controller.t_last_buff_cast = []
        controller.t_last_skill = 0.0
        controller.input_client = controller_module._input_client
        controller.perform_directional_attack = Mock(return_value=True)
        controller.release_all_key = Mock()

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        controller.run()

        controller.perform_directional_attack.assert_called_once_with(
            "left",
            attack_key="shift",
            recovery_delay=1.2,
        )
        self.assertEqual(controller.cmd_action, "none")

    def test_run_routes_power_knockback_to_s(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "bot": {"attack": "directional"},
            "buff_skill": {"keys": [], "cooldown": [], "action_cooldown": 1},
            "key": {"power_knockback": "s"},
            "system": {"fps_limit_keyboard_controller": 30},
        }
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "right"
        controller.cmd_up_down = "none"
        controller.cmd_action = "power_knockback"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.power_knockback_key = "s"
        controller.power_knockback_recovery_delay = 1.1
        controller.attack_recovery_delay = 0.9
        controller.t_last_buff_cast = []
        controller.t_last_skill = 0.0
        controller.input_client = controller_module._input_client
        controller.perform_directional_attack = Mock(return_value=True)
        controller.release_all_key = Mock()

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        controller.run()

        controller.perform_directional_attack.assert_called_once_with(
            "right",
            attack_key="s",
            recovery_delay=1.1,
        )
        self.assertEqual(controller.cmd_action, "none")

    def test_run_routes_portal_sweep_to_atomic_portal_step(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "bot": {"attack": "directional"},
            "buff_skill": {"keys": [], "cooldown": [], "action_cooldown": 1},
            "system": {"fps_limit_keyboard_controller": 30},
        }
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "none"
        controller.cmd_up_down = "up"
        controller.cmd_action = "portal_sweep_right"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.t_last_buff_cast = []
        controller.t_last_skill = 0.0
        controller.input_client = controller_module._input_client
        controller.perform_portal_sweep_step = Mock(return_value=True)
        controller.release_all_key = Mock()

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        controller.run()

        controller.perform_portal_sweep_step.assert_called_once_with("right")
        self.assertEqual(controller.cmd_action, "none")

    def test_run_routes_rope_alignment_before_ordinary_movement(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "bot": {"attack": "directional"},
            "buff_skill": {"keys": [], "cooldown": [], "action_cooldown": 1},
            "system": {"fps_limit_keyboard_controller": 30},
        }
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "none"
        controller.cmd_up_down = "none"
        controller.cmd_action = "rope_align_left"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.t_last_buff_cast = []
        controller.t_last_skill = 0.0
        controller.input_client = controller_module._input_client
        controller.perform_rope_alignment_nudge = Mock(return_value=True)
        controller.update_movement_state = Mock(return_value=True)
        controller.release_all_key = Mock()

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        controller.run()

        controller.perform_rope_alignment_nudge.assert_called_once_with(
            "left"
        )
        controller.update_movement_state.assert_not_called()
        self.assertEqual(controller.cmd_action, "none")

    def test_run_routes_rope_mount_before_ordinary_movement(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "bot": {"attack": "directional"},
            "buff_skill": {"keys": [], "cooldown": [], "action_cooldown": 1},
            "system": {"fps_limit_keyboard_controller": 30},
        }
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "right"
        controller.cmd_up_down = "none"
        controller.cmd_action = "rope_mount_right"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.t_last_buff_cast = []
        controller.t_last_skill = 0.0
        controller.input_client = controller_module._input_client
        controller.perform_rope_mount = Mock(return_value=True)
        controller.update_movement_state = Mock(return_value=True)
        controller.release_all_key = Mock()

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        controller.run()

        controller.perform_rope_mount.assert_called_once_with("right")
        controller.update_movement_state.assert_not_called()
        self.assertEqual(controller.cmd_action, "none")

    def test_run_rope_hold_owns_frame_without_consuming_action(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "bot": {"attack": "directional"},
            "buff_skill": {
                "keys": ["b"],
                "cooldown": [0],
                "action_cooldown": 0,
            },
            "system": {"fps_limit_keyboard_controller": 30},
        }
        controller.command_lock = controller_module.threading.RLock()
        controller.is_enable = True
        controller.capture_available = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "none"
        controller.cmd_up_down = "up"
        controller.cmd_action = "rope_hold"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.t_last_buff_cast = [0.0]
        controller.t_last_skill = 0.0
        controller.input_client = controller_module._input_client
        controller.update_movement_state = Mock(return_value=True)
        controller.release_all_key = Mock()

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        with patch.object(controller_module, "press_key") as press:
            controller.run()

        controller.update_movement_state.assert_called_once_with("none", "up")
        press.assert_not_called()
        self.assertEqual(controller.cmd_action, "rope_hold")

    def test_remote_target_does_not_depend_on_computer_a_focus(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {"esp32_hid": {"remote_target": True}}

        with patch.object(controller_module.gw, "getActiveWindow") as active:
            self.assertTrue(controller.is_game_window_active())

        active.assert_not_called()

    def test_capture_gate_does_not_override_user_pause(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.capture_available = True
        controller.is_enable = False
        controller.release_all_key = Mock()
        controller.is_game_window_active = Mock(return_value=True)

        self.assertTrue(controller.set_capture_available(False))
        self.assertTrue(controller.set_capture_available(True))

        self.assertFalse(controller_module._input_allowed.is_set())
        controller.release_all_key.assert_called_once_with()

    def test_blank_potion_mapping_clears_pending_action(self):
        controller = controller_module.KeyBoardController.__new__(
            controller_module.KeyBoardController
        )
        controller.cfg = {
            "esp32_hid": {"remote_target": True},
            "buff_skill": {"keys": [], "cooldown": [], "action_cooldown": 1},
            "key": {"add_hp": "   ", "add_mp": "   "},
            "system": {"fps_limit_keyboard_controller": 30},
        }
        controller.is_enable = True
        controller.is_terminated = False
        controller.is_need_force_heal = False
        controller.cmd_left_right = "none"
        controller.cmd_up_down = "none"
        controller.cmd_action = "add_hp"
        controller.cmd_left_right_last = ""
        controller.cmd_up_down_last = ""
        controller.t_last_buff_cast = []
        controller.t_last_skill = 0.0
        controller.input_client = controller_module._input_client

        def stop_after_one_frame():
            controller.is_terminated = True

        controller.limit_fps = stop_after_one_frame
        controller.run()

        self.assertEqual(controller.cmd_action, "none")


if __name__ == "__main__":
    unittest.main()
