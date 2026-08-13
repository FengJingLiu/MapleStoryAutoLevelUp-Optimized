import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from pynput import keyboard

from src.input.KeyBoardListener import KeyBoardListener, normalize_key_name
from tools.routeRecorder import RouteRecorder, route_action_from_pressed_keys


class RouteRecorderKeyTests(unittest.TestCase):
    def test_modifier_names_are_normalized(self):
        self.assertEqual(normalize_key_name("alt_l"), "alt")
        self.assertEqual(normalize_key_name("Right Alt"), "alt")
        self.assertEqual(normalize_key_name("control"), "ctrl")
        self.assertEqual(normalize_key_name("shift_r"), "shift")

    def test_listener_tracks_modifier_press_and_release(self):
        listener = KeyBoardListener.__new__(KeyBoardListener)
        listener.key_pressing = []
        listener.func_keys = {}
        listener.movement_keys = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
            keyboard.Key.space: "space",
        }

        listener.on_press(keyboard.Key.alt_l)
        self.assertEqual(listener.key_pressing, ["alt"])

        listener.on_release(keyboard.Key.alt_l)
        self.assertEqual(listener.key_pressing, [])

    def test_custom_jump_key_replaces_hard_coded_space(self):
        key_cfg = {"jump": "alt", "teleport": "t"}

        self.assertEqual(
            route_action_from_pressed_keys(["left", "alt_l"], key_cfg),
            ("left none jump", True),
        )
        self.assertEqual(
            route_action_from_pressed_keys(["space"], key_cfg),
            ("", False),
        )

    def test_custom_teleport_key_is_used_with_direction(self):
        key_cfg = {"jump": "j", "teleport": "t"}

        self.assertEqual(
            route_action_from_pressed_keys(["up", "t"], key_cfg),
            ("none up teleport", True),
        )
        self.assertEqual(
            route_action_from_pressed_keys(["down", "t"], key_cfg),
            ("none down teleport", True),
        )

    def test_empty_teleport_key_does_not_create_teleport_action(self):
        key_cfg = {"jump": "alt", "teleport": ""}

        self.assertEqual(
            route_action_from_pressed_keys(["up"], key_cfg),
            ("none up none", False),
        )


class RouteRecorderGeometryTests(unittest.TestCase):
    def test_map_expansion_keeps_route_canvas_and_coordinates_aligned(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.cfg = {"route_recoder": {"map_padding": 3}}
        recorder.img_map = np.full((4, 6, 3), 11, dtype=np.uint8)
        recorder.img_route = np.full((4, 6, 3), 22, dtype=np.uint8)
        recorder.loc_minimap_global = (0, 0)
        recorder.loc_player_global_last = (2, 3)

        offset = recorder.ensure_img_map_capacity(0, 0, 4, 6)

        self.assertEqual(offset, (3, 3))
        self.assertEqual(recorder.img_map.shape, (10, 12, 3))
        self.assertEqual(recorder.img_route.shape, recorder.img_map.shape)
        self.assertTrue(np.all(recorder.img_map[3:7, 3:9] == 11))
        self.assertTrue(np.all(recorder.img_route[3:7, 3:9] == 22))
        self.assertEqual(recorder.loc_minimap_global, (3, 3))
        self.assertEqual(recorder.loc_player_global_last, (5, 6))

    def test_player_global_location_reuses_masked_minimap_match(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.loc_minimap_global = (7, 9)
        recorder.loc_player_minimap = (4, 5)
        recorder.minimap_match_score = 0.125
        recorder.img_minimap = np.zeros((10, 12, 3), dtype=np.uint8)
        recorder.img_route_debug = np.zeros((40, 40, 3), dtype=np.uint8)

        self.assertEqual(recorder.get_player_location_on_global_map(), (11, 14))

    def test_out_of_bounds_route_preview_is_skipped(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.t_last_frame = 1.0
        recorder.kb = SimpleNamespace(t_func_key=[0.0, 0.0, 0.0, 0.0])
        recorder.is_enable = True
        recorder.img_frame_debug = np.zeros((700, 1296, 3), dtype=np.uint8)
        recorder.loc_minimap = (0, 0)
        recorder.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8)
        recorder.img_route_debug = np.zeros((20, 20, 3), dtype=np.uint8)
        recorder.loc_player_global = (100, 100)

        with patch("tools.routeRecorder.cv2.resize") as resize:
            recorder.update_info_on_img_frame_debug()

        resize.assert_not_called()

    def test_capture_warmup_accepts_second_frame(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.capture = Mock()
        recorder.capture.get_frame.side_effect = [
            None,
            np.zeros((2, 2, 3), dtype=np.uint8),
        ]

        self.assertTrue(
            recorder.wait_for_initial_capture_frame(
                timeout=1.0,
                poll_interval=0.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
