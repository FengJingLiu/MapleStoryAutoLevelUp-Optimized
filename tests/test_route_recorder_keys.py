import unittest
import os
import threading
import time
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2
from pynput import keyboard

from src.input.KeyBoardListener import KeyBoardListener, normalize_key_name
from src.input.Esp32KeyForwarder import Esp32KeyForwarder
from src.utils.common import find_pattern_sqdiff, mask_route_colors
from tools.routeRecorder import (
    RouteRecorder,
    fill_empty_canvas_pixels,
    fit_debug_preview,
    prepare_minimap_for_alignment,
    prepare_route_output_directory,
    route_action_from_pressed_keys,
    route_forward_keys_from_config,
    select_stable_minimap_match,
)


class RouteRecorderKeyTests(unittest.TestCase):
    def test_existing_route_directory_keeps_map_and_clears_only_routes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            map_dir = Path(temp_dir) / "wisdom_forest"
            map_dir.mkdir()
            map_data = b"saved stitched map"
            (map_dir / "map.png").write_bytes(map_data)
            (map_dir / "route1.png").write_bytes(b"route one")
            (map_dir / "route-old.PNG").write_bytes(b"route old")
            (map_dir / "notes.txt").write_text("keep", encoding="utf-8")

            self.assertTrue(
                prepare_route_output_directory(map_dir, confirm=lambda _: "y")
            )

            self.assertEqual((map_dir / "map.png").read_bytes(), map_data)
            self.assertFalse((map_dir / "route1.png").exists())
            self.assertFalse((map_dir / "route-old.PNG").exists())
            self.assertEqual(
                (map_dir / "notes.txt").read_text(encoding="utf-8"),
                "keep",
            )

    def test_declining_route_cleanup_leaves_directory_untouched(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            map_dir = Path(temp_dir) / "wisdom_forest"
            map_dir.mkdir()
            route_path = map_dir / "route1.png"
            route_path.write_bytes(b"route")

            self.assertFalse(
                prepare_route_output_directory(map_dir, confirm=lambda _: "n")
            )
            self.assertEqual(route_path.read_bytes(), b"route")

    def test_esp32_forwarder_preserves_press_and_release_states(self):
        class FakeEsp32Client:
            endpoint = "COM6@115200"
            connect_timeout = 0.01
            request_timeout = 0.01

            def __init__(self):
                self.states = []
                self.state_event = threading.Event()
                self.closed = False

            def set_state(self, keys):
                self.states.append(tuple(keys))
                self.state_event.set()
                return True

            def close(self):
                self.closed = True

        client = FakeEsp32Client()
        forwarder = Esp32KeyForwarder(
            {},
            {"left", "alt"},
            client=client,
        )
        try:
            self.assertFalse(forwarder.handle_key_event("f3", True))
            self.assertTrue(forwarder.handle_key_event("left", True))
            self.assertTrue(forwarder.handle_key_event("alt", True))
            self.assertTrue(forwarder.handle_key_event("left", False))

            deadline = time.monotonic() + 1.0
            while len(client.states) < 3 and time.monotonic() < deadline:
                client.state_event.wait(0.01)
                client.state_event.clear()

            self.assertEqual(
                client.states[:3],
                [("left",), ("alt", "left"), ("alt",)],
            )
        finally:
            forwarder.close()

        self.assertTrue(client.closed)

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
        listener.key_event_handler = Mock()

        listener.on_press(keyboard.Key.alt_l)
        self.assertEqual(listener.key_pressing, ["alt"])
        listener.key_event_handler.assert_called_once_with("alt", True)

        listener.on_release(keyboard.Key.alt_l)
        self.assertEqual(listener.key_pressing, [])
        self.assertEqual(
            listener.key_event_handler.call_args_list[-1].args,
            ("alt", False),
        )

    def test_route_forward_keys_include_gameplay_but_exclude_local_controls(self):
        cfg = {
            "key": {
                "jump": "alt_l",
                "teleport": "E",
                "directional_attack": "control",
            },
            "buff_skill": {"keys": ["5"]},
            "route_recoder": {"forward_keys": ["f3", "insert"]},
        }

        keys = route_forward_keys_from_config(cfg)

        self.assertTrue(
            {"left", "right", "up", "down", "alt", "e", "ctrl", "5", "insert"}
            <= keys
        )
        self.assertNotIn("f3", keys)

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
    @staticmethod
    def make_draw_recorder():
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.cfg = {
            "route_recoder": {"blob_cooldown": 0.7, "map_padding": 3}
        }
        recorder.color_code = {
            (255, 0, 0): "left none none",
            (0, 255, 0): "right none none",
            (0, 0, 255): "none none jump",
            (255, 255, 0): "none none goal",
        }
        recorder.img_route = np.zeros((20, 20, 3), dtype=np.uint8)
        recorder.is_enable = True
        recorder.loc_player_global_last = None
        recorder._last_route_action = None
        recorder.t_last_draw_blob = time.time()
        return recorder

    def test_new_map_background_is_added_without_erasing_route(self):
        canvas = np.zeros((10, 12, 3), dtype=np.uint8)
        source = np.full((4, 5, 3), 80, dtype=np.uint8)
        canvas[7, 6] = (1, 2, 3)  # Existing route/action pixel.

        self.assertTrue(fill_empty_canvas_pixels(canvas, source, (4, 5)))

        self.assertTrue(np.all(canvas[5:9, 4:9][0, 0] == 80))
        self.assertTrue(np.all(canvas[7, 6] == (1, 2, 3)))

    def test_route_debug_preview_fits_tall_map_without_cropping(self):
        img = np.zeros((500, 240, 3), dtype=np.uint8)
        img[-1, -1] = (0, 255, 255)

        preview, scale = fit_debug_preview(
            img,
            preferred_scale=4,
            max_size=(1200, 900),
        )

        self.assertAlmostEqual(scale, 1.8)
        self.assertEqual(preview.shape[:2], (900, 432))
        self.assertTrue(np.any(preview[-2:, -2:] != 0))

    def test_stable_match_prefers_local_when_repeated_platforms_tie(self):
        tile = np.full((30, 30, 3), 80, dtype=np.uint8)
        cv2.line(tile, (2, 8), (27, 8), (25, 110, 40), 2)
        stitched_map = np.zeros((30, 120, 3), dtype=np.uint8)
        stitched_map[:, 0:30] = tile
        stitched_map[:, 90:120] = tile

        location, _, accepted = select_stable_minimap_match(
            stitched_map,
            tile,
            last_result=(90, 0),
            mask=np.full((30, 30), 255, dtype=np.uint8),
            local_search_radius=20,
            local_accept_threshold=0.01,
            global_accept_threshold=0.01,
        )

        self.assertEqual(location, (90, 0))
        self.assertTrue(accepted)

    def test_initial_match_rejects_an_unrelated_existing_map(self):
        stitched_map = np.zeros((30, 40, 3), dtype=np.uint8)
        current_minimap = np.full((20, 25, 3), 180, dtype=np.uint8)

        _, score, accepted = select_stable_minimap_match(
            stitched_map,
            current_minimap,
            last_result=None,
            mask=np.full((20, 25), 255, dtype=np.uint8),
            global_accept_threshold=0.05,
        )

        self.assertGreater(score, 0.05)
        self.assertFalse(accepted)

    def test_stable_match_accepts_clearly_better_distant_portal(self):
        pattern = np.full((20, 20, 3), 80, dtype=np.uint8)
        cv2.circle(pattern, (10, 10), 5, (20, 150, 40), -1)
        degraded_local = pattern.copy()
        degraded_local[5:15, 5:15] = (120, 80, 30)
        stitched_map = np.zeros((20, 120, 3), dtype=np.uint8)
        stitched_map[:, 0:20] = degraded_local
        stitched_map[:, 100:120] = pattern

        location, score, accepted = select_stable_minimap_match(
            stitched_map,
            pattern,
            last_result=(0, 0),
            mask=np.full((20, 20), 255, dtype=np.uint8),
            local_search_radius=20,
            local_accept_threshold=0.5,
            global_accept_threshold=0.1,
            teleport_score_margin=0.03,
        )

        self.assertEqual(location, (100, 0))
        self.assertLess(score, 0.01)
        self.assertTrue(accepted)

    def test_moving_player_dot_cannot_choose_a_repeated_platform(self):
        tile = np.full((30, 30, 3), 80, dtype=np.uint8)
        cv2.line(tile, (2, 8), (27, 8), (25, 110, 40), 2)
        cv2.rectangle(tile, (4, 17), (10, 25), (35, 90, 45), -1)

        current = tile.copy()
        current[14:17, 15] = (98, 243, 213)
        stitched_map = np.zeros((30, 70, 3), dtype=np.uint8)
        stitched_map[:, 0:30] = tile
        stitched_map[:, 40:70] = tile
        stitched_map[14:17, 55] = (98, 243, 213)

        unmasked_location, _, _ = find_pattern_sqdiff(
            stitched_map,
            current,
            mask=np.full((30, 30), 255, dtype=np.uint8),
        )
        alignment, mask = prepare_minimap_for_alignment(
            current,
            player_location=(15, 15),
        )
        masked_location, _, _ = find_pattern_sqdiff(
            stitched_map,
            alignment,
            mask=mask,
        )

        self.assertEqual(unmasked_location, (40, 0))
        self.assertEqual(masked_location, (0, 0))

    def test_alignment_excludes_dynamic_minimap_markers(self):
        minimap = np.full((40, 60, 3), 80, dtype=np.uint8)
        minimap[19:22, 29] = (98, 243, 213)  # HDMI-scaled player dot
        minimap[10:12, 45:47] = (0, 26, 161)  # red other-player dot
        minimap[30:32, 8:10] = (220, 130, 20)  # blue party dot

        alignment, mask = prepare_minimap_for_alignment(
            minimap,
            player_location=(29, 20),
        )

        self.assertTrue(np.all(alignment[20, 29] == 0))
        self.assertEqual(mask[20, 29], 0)
        self.assertTrue(np.all(alignment[10, 45] == 0))
        self.assertEqual(mask[10, 45], 0)
        self.assertTrue(np.all(alignment[30, 8] == 0))
        self.assertEqual(mask[30, 8], 0)
        self.assertTrue(np.all(alignment[20, 15] == 80))
        self.assertEqual(mask[20, 15], 255)

    def test_native_alignment_scales_dynamic_marker_exclusion(self):
        minimap = np.full((293, 350, 3), 80, dtype=np.uint8)
        minimap[130:145, 200:215] = (98, 243, 213)
        minimap[60:75, 290:305] = (0, 0, 200)

        alignment, mask = prepare_minimap_for_alignment(
            minimap,
            player_location=(207, 137),
        )

        self.assertTrue(np.all(alignment[137, 207] == 0))
        self.assertEqual(mask[137, 207], 0)
        # A 15x15 native marker (225 pixels) exceeded the old fixed area=100.
        self.assertTrue(np.all(alignment[67, 297] == 0))
        self.assertEqual(mask[67, 297], 0)

    def test_alignment_keeps_large_colored_minimap_artwork(self):
        minimap = np.full((40, 60, 3), 80, dtype=np.uint8)
        minimap[5:20, 5:20] = (0, 0, 200)

        alignment, mask = prepare_minimap_for_alignment(
            minimap,
            player_location=None,
            max_colored_marker_area=100,
        )

        self.assertTrue(np.all(alignment[10, 10] == (0, 0, 200)))
        self.assertEqual(mask[10, 10], 255)

    def test_map_expansion_keeps_route_canvas_and_coordinates_aligned(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.cfg = {"route_recoder": {"map_padding": 3}}
        recorder.img_map = np.full((4, 6, 3), 11, dtype=np.uint8)
        recorder.img_route = np.full((4, 6, 3), 22, dtype=np.uint8)
        recorder.completed_routes = [np.full((4, 6, 3), 33, dtype=np.uint8)]
        recorder.loc_minimap_global = (0, 0)
        recorder.loc_player_global_last = (2, 3)

        offset = recorder.ensure_img_map_capacity(0, 0, 4, 6)

        self.assertEqual(offset, (3, 3))
        self.assertEqual(recorder.img_map.shape, (10, 12, 3))
        self.assertEqual(recorder.img_route.shape, recorder.img_map.shape)
        self.assertTrue(np.all(recorder.img_map[3:7, 3:9] == 11))
        self.assertTrue(np.all(recorder.img_route[3:7, 3:9] == 22))
        self.assertEqual(recorder.completed_routes[0].shape, recorder.img_map.shape)
        self.assertTrue(
            np.all(recorder.completed_routes[0][3:7, 3:9] == 33)
        )
        self.assertEqual(recorder.loc_minimap_global, (3, 3))
        self.assertEqual(recorder.loc_player_global_last, (5, 6))

    def test_key_release_breaks_route_segment(self):
        recorder = self.make_draw_recorder()
        recorder.loc_player_global = (1, 5)
        recorder.record_route_sample("left none none", False)
        recorder.loc_player_global = (5, 5)
        recorder.record_route_sample("left none none", False)

        recorder.loc_player_global = (7, 5)
        recorder.record_route_sample("", False)
        recorder.loc_player_global = (9, 5)
        recorder.record_route_sample("left none none", False)

        red_bgr = (0, 0, 255)
        self.assertTrue(np.all(recorder.img_route[5, 3] == red_bgr))
        self.assertTrue(np.all(recorder.img_route[5, 7] == 0))
        self.assertTrue(np.all(recorder.img_route[5, 9] == red_bgr))

    def test_action_change_starts_a_new_segment(self):
        recorder = self.make_draw_recorder()
        recorder.loc_player_global = (5, 1)
        recorder.record_route_sample("left none none", False)
        recorder.loc_player_global = (5, 5)
        recorder.record_route_sample("left none none", False)

        recorder.loc_player_global = (9, 9)
        recorder.record_route_sample("right none none", False)

        green_bgr = (0, 255, 0)
        self.assertTrue(np.all(recorder.img_route[9, 9] == green_bgr))
        self.assertTrue(np.all(recorder.img_route[7, 7] == 0))

    def test_blob_cooldown_still_breaks_route_segment(self):
        recorder = self.make_draw_recorder()
        recorder.loc_player_global = (1, 5)
        recorder.record_route_sample("left none none", False)
        recorder.loc_player_global = (5, 5)
        recorder.record_route_sample("left none none", False)

        recorder.loc_player_global = (6, 5)
        self.assertFalse(
            recorder.record_route_sample("none none jump", True)
        )
        recorder.loc_player_global = (9, 5)
        recorder.record_route_sample("left none none", False)

        self.assertTrue(np.all(recorder.img_route[5, 7] == 0))

    def test_save_bundle_keeps_map_and_every_route_at_one_size(self):
        recorder = self.make_draw_recorder()
        recorder.img_map = np.full((4, 6, 3), 11, dtype=np.uint8)
        recorder.img_route = np.full((4, 6, 3), 22, dtype=np.uint8)
        recorder.completed_routes = [
            np.full((4, 6, 3), 33, dtype=np.uint8)
        ]
        recorder.loc_minimap_global = (0, 0)
        recorder.loc_player_global_last = None
        recorder._locked_minimap_rect = (1, 1, 3, 2)
        recorder._minimap_lock_frame_size = (10, 12)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder.map_dir = temp_dir
            self.assertTrue(
                recorder.save_recording_bundle(include_current_route=True)
            )

            def read_png(name):
                data = np.frombuffer(
                    (Path(temp_dir) / name).read_bytes(), dtype=np.uint8
                )
                return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

            saved_map = read_png("map.png")
            saved_route1 = read_png("route1.png")
            saved_route2 = read_png("route2.png")
            self.assertEqual(saved_map.shape, (4, 6, 3))
            self.assertEqual(saved_route1.shape, saved_map.shape)
            self.assertEqual(saved_route2.shape, saved_map.shape)

            recorder.completed_routes.append(recorder.img_route.copy())
            recorder.ensure_img_map_capacity(0, 0, 4, 6)
            self.assertTrue(
                recorder.save_recording_bundle(include_current_route=False)
            )
            rewritten_route1 = read_png("route1.png")
            rewritten_route2 = read_png("route2.png")
            self.assertEqual(rewritten_route1.shape, (10, 12, 3))
            self.assertEqual(rewritten_route2.shape, (10, 12, 3))
            self.assertTrue(np.all(rewritten_route1[3:7, 3:9] == 33))

    def test_save_bundle_rejects_a_mismatched_completed_route(self):
        recorder = self.make_draw_recorder()
        recorder.img_map = np.zeros((4, 6, 3), dtype=np.uint8)
        recorder.completed_routes = [np.zeros((5, 6, 3), dtype=np.uint8)]
        recorder._locked_minimap_rect = (1, 1, 3, 2)
        recorder._minimap_lock_frame_size = (10, 12)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder.map_dir = temp_dir
            self.assertFalse(recorder.save_recording_bundle())
            self.assertFalse((Path(temp_dir) / "map.png").exists())

    def test_geometry_stage_failure_keeps_existing_bundle_unchanged(self):
        recorder = self.make_draw_recorder()
        recorder.img_map = np.full((4, 6, 3), 11, dtype=np.uint8)
        recorder.completed_routes = [
            np.full((4, 6, 3), 33, dtype=np.uint8)
        ]
        recorder._locked_minimap_rect = (1, 1, 3, 2)
        recorder._minimap_lock_frame_size = (10, 12)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder.map_dir = temp_dir
            map_path = Path(temp_dir) / "map.png"
            route_path = Path(temp_dir) / "route1.png"
            map_path.write_bytes(b"old map")
            route_path.write_bytes(b"old route")

            with patch(
                "tools.routeRecorder.stage_text_write",
                side_effect=OSError("geometry stage failed"),
            ):
                self.assertFalse(recorder.save_recording_bundle())

            self.assertEqual(map_path.read_bytes(), b"old map")
            self.assertEqual(route_path.read_bytes(), b"old route")
            self.assertFalse((Path(temp_dir) / "minimap_geometry.txt").exists())

    def test_replace_failure_rolls_back_every_bundle_target(self):
        recorder = self.make_draw_recorder()
        recorder.img_map = np.full((4, 6, 3), 11, dtype=np.uint8)
        recorder.completed_routes = [
            np.full((4, 6, 3), 33, dtype=np.uint8)
        ]
        recorder._locked_minimap_rect = (1, 1, 3, 2)
        recorder._minimap_lock_frame_size = (10, 12)

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder.map_dir = temp_dir
            map_path = Path(temp_dir) / "map.png"
            route_path = Path(temp_dir) / "route1.png"
            map_path.write_bytes(b"old map")
            route_path.write_bytes(b"old route")
            real_replace = os.replace

            def fail_route_commit(source, destination):
                if (
                    Path(source).name == ".route1.recording.tmp.png"
                    and Path(destination).name == "route1.png"
                ):
                    raise OSError("route replace failed")
                return real_replace(source, destination)

            with patch(
                "tools.routeRecorder.os.replace",
                side_effect=fail_route_commit,
            ):
                self.assertFalse(recorder.save_recording_bundle())

            self.assertEqual(map_path.read_bytes(), b"old map")
            self.assertEqual(route_path.read_bytes(), b"old route")
            self.assertFalse((Path(temp_dir) / "minimap_geometry.txt").exists())
            leftovers = [
                path.name for path in Path(temp_dir).iterdir()
                if ".recording." in path.name
            ]
            self.assertEqual(leftovers, [])

    def test_route_loader_rejects_canvas_resize(self):
        img_map = np.zeros((4, 6, 3), dtype=np.uint8)
        img_route = np.zeros((5, 6, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "canvas mismatch"):
            mask_route_colors(
                img_map,
                img_route,
                {"255,0,0": "left none none"},
            )

    def test_expansion_mismatch_does_not_partially_resize_the_map(self):
        recorder = self.make_draw_recorder()
        recorder.img_map = np.full((4, 6, 3), 11, dtype=np.uint8)
        recorder.img_route = np.full((5, 6, 3), 22, dtype=np.uint8)
        recorder.completed_routes = []
        recorder.loc_minimap_global = (0, 0)

        with self.assertRaises(RuntimeError):
            recorder.ensure_img_map_capacity(0, 0, 4, 6)

        self.assertEqual(recorder.img_map.shape, (4, 6, 3))
        self.assertTrue(np.all(recorder.img_map == 11))

    def test_player_global_location_tracks_centroid_pixel_changes_directly(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.loc_minimap_global = (7, 9)
        recorder.loc_player_minimap = (4, 5)
        recorder.minimap_match_score = 0.125
        recorder.img_minimap = np.zeros((10, 12, 3), dtype=np.uint8)
        recorder.img_route_debug = np.zeros((40, 40, 3), dtype=np.uint8)

        self.assertEqual(recorder.get_player_location_on_global_map(), (11, 14))

        recorder.loc_player_minimap = (5, 6)

        self.assertEqual(recorder.get_player_location_on_global_map(), (12, 15))

    def test_out_of_bounds_route_preview_is_skipped(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.t_last_frame = 1.0
        recorder.kb = SimpleNamespace(t_func_key=[0.0, 0.0, 0.0, 0.0])
        recorder.is_enable = True
        recorder.img_frame_debug = np.zeros((700, 1296, 3), dtype=np.uint8)
        recorder.loc_minimap = (0, 0)
        recorder.img_minimap_screen = np.zeros((10, 10, 3), dtype=np.uint8)
        recorder.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8)
        recorder.img_route_debug = np.zeros((20, 20, 3), dtype=np.uint8)
        recorder.loc_player_global = (100, 100)

        with patch("tools.routeRecorder.cv2.resize") as resize:
            recorder.update_info_on_img_frame_debug()

        resize.assert_not_called()

    def test_debug_minimap_box_uses_screen_coordinate_size(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.t_last_frame = 1.0
        recorder.kb = SimpleNamespace(t_func_key=[0.0, 0.0, 0.0, 0.0])
        recorder.is_enable = True
        recorder.img_frame_debug = np.zeros((700, 1296, 3), dtype=np.uint8)
        recorder.loc_minimap = (6, 45)
        recorder.img_minimap_screen = np.zeros((67, 83, 3), dtype=np.uint8)
        recorder.img_minimap = np.zeros((149, 180, 3), dtype=np.uint8)
        recorder.img_route_debug = np.zeros((80, 80, 3), dtype=np.uint8)
        recorder.loc_player_global = (40, 40)
        recorder.minimap_match_held = False

        with patch("tools.routeRecorder.draw_rectangle") as draw:
            recorder.update_info_on_img_frame_debug()

        self.assertEqual(draw.call_args.args[2], (67, 83))

    def test_debug_shortcut_text_fits_inside_visible_game_crop(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.t_last_frame = 1.0
        recorder.kb = SimpleNamespace(t_func_key=[0.0, 0.0, 0.0, 0.0])
        recorder.is_enable = True
        recorder.cfg = {"ui_coords": {"ui_y_start": 610}}
        recorder.img_frame_debug = np.zeros((700, 1296, 3), dtype=np.uint8)
        recorder.loc_minimap = (6, 45)
        recorder.img_minimap_screen = np.zeros((67, 83, 3), dtype=np.uint8)
        recorder.img_minimap = np.zeros((149, 180, 3), dtype=np.uint8)
        recorder.img_route_debug = np.zeros((80, 80, 3), dtype=np.uint8)
        recorder.loc_player_global = (40, 40)

        with patch("tools.routeRecorder.cv2.putText") as put_text:
            recorder.update_info_on_img_frame_debug()

        shortcut_calls = [
            call for call in put_text.call_args_list
            if call.args[1].startswith(("FPS:", "Press "))
        ]
        baselines = [call.args[2][1] for call in shortcut_calls]
        _, text_baseline = cv2.getTextSize(
            "Ag", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        self.assertEqual(len(baselines), 5)
        self.assertEqual(baselines, sorted(baselines))
        self.assertLessEqual(max(baselines) + text_baseline + 4, 610)

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
