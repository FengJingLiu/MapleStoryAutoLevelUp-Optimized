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
