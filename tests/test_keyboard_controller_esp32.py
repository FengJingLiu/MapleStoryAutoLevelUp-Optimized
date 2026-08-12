import unittest
from unittest.mock import Mock, patch

from src.input import KeyBoardController as controller_module
from src.input.Esp32HidClient import Esp32HidTapUncertainError


class KeyboardControllerEsp32RoutingTests(unittest.TestCase):
    def setUp(self):
        self.previous_client = controller_module._input_client
        self.previous_allowed = controller_module._input_allowed.is_set()
        controller_module._input_client = Mock()
        controller_module._input_allowed.clear()

    def tearDown(self):
        controller_module._input_client = self.previous_client
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
        controller.cmd_left_right_last = "none"
        controller.cmd_up_down_last = "none"

        controller.update_movement_state()

        controller_module._input_client.set_state.assert_called_once_with(
            ["left", "up"]
        )

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
