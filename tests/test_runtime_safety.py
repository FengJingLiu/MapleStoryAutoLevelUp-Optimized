import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import cv2
import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.engine.RuneSolver import RuneSolver
from src.input.CaptureFramePreprocessor import (
    get_capture_resize_size,
    preprocess_capture_frame,
)
from src.input.GameWindowCapturor import GameWindowCapturor
from src.states.debug import DebugState
from src.utils.common import (
    draw_circle,
    draw_line,
    draw_rectangle,
    draw_text,
    get_minimap_loc_size,
    get_player_location_on_minimap,
    resize_window,
    copy_minimap_native_raster, copy_minimap_native_location,
    route_map_can_fit_minimap,
)


class DebugDrawingTests(unittest.TestCase):
    def test_debug_helpers_ignore_missing_canvas(self):
        draw_rectangle(None, (0, 0), (10, 20), (0, 0, 0), "test")
        draw_circle(None, (0, 0), 5, (0, 0, 0), 1)
        draw_line(None, (0, 0), (1, 1), (0, 0, 0), 1)
        draw_text(None, "test", (0, 0), 0, 1, (0, 0, 0), 1)

    def test_aux_mode_emits_game_viz_without_a_route_canvas(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {"ui_coords": {"ui_y_start": 6}}
        bot.is_show_debug_window = True
        bot.is_ui = True
        bot.img_frame_debug = np.full((10, 20, 3), 123, dtype=np.uint8)
        bot.img_route_debug = None
        bot.image_debug_signal = Mock()
        bot.route_map_viz_signal = Mock()

        bot._emit_debug_images()

        bot.image_debug_signal.emit.assert_called_once()
        emitted_frame = bot.image_debug_signal.emit.call_args.args[0]
        self.assertEqual(emitted_frame.shape, (10, 20, 3))
        self.assertTrue(np.all(emitted_frame[6:] == 123))
        self.assertFalse(np.shares_memory(emitted_frame, bot.img_frame_debug))
        bot.route_map_viz_signal.emit.assert_not_called()

    def test_native_fullscreen_viz_is_bounded_before_qt_signal(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "system": {
                "fps_limit_ui_viz": 0,
                "ui_viz_max_size": (720, 1280),
            }
        }
        bot.is_show_debug_window = True
        bot.is_ui = True
        bot.img_frame_debug = np.zeros((2013, 3579, 3), dtype=np.uint8)
        bot.img_route_debug = None
        bot.image_debug_signal = Mock()
        bot.route_map_viz_signal = Mock()

        bot._emit_debug_images()

        emitted_frame = bot.image_debug_signal.emit.call_args.args[0]
        self.assertEqual(emitted_frame.shape, (720, 1280, 3))
        self.assertTrue(emitted_frame.flags.c_contiguous)
        self.assertFalse(np.shares_memory(emitted_frame, bot.img_frame_debug))

    def test_ui_viz_signal_is_rate_limited(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "system": {
                "fps_limit_ui_viz": 5,
                "ui_viz_max_size": (720, 1280),
            }
        }
        bot.is_show_debug_window = True
        bot.is_ui = True
        bot.img_frame_debug = np.zeros((10, 20, 3), dtype=np.uint8)
        bot.img_route_debug = None
        bot.image_debug_signal = Mock()
        bot.route_map_viz_signal = Mock()
        bot._last_ui_viz_emit_time = 0.0

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            side_effect=(10.0, 10.05, 10.21),
        ):
            bot._emit_debug_images()
            bot._emit_debug_images()
            bot._emit_debug_images()

        self.assertEqual(bot.image_debug_signal.emit.call_count, 2)

    def test_native_debug_style_matches_legacy_after_preview_resize(self):
        def draw_metrics(frame_shape):
            bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
            bot.cfg = {
                "game_window": {
                    "coordinate_reference_size": (700, 1296),
                },
                "system": {"ui_viz_max_size": (720, 1280)},
            }
            bot._base_cfg = bot.cfg
            bot.img_frame_debug = np.zeros(
                (*frame_shape, 3), dtype=np.uint8
            )
            with patch(
                "src.engine.MapleStoryAutoLevelUp.cv2.putText"
            ) as put_text:
                bot._draw_debug_text(
                    "hint",
                    (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                    reference_position=True,
                )
            call_args = put_text.call_args.args
            height, width = frame_shape
            preview_scale = min(1.0, 1280 / width, 720 / height)
            next_y = bot.scale_debug_reference_point((10, 483))[1]
            return {
                "visual_scale": bot.get_frame_visual_scale(),
                "x": call_args[2][0] * preview_scale,
                "y": call_args[2][1] * preview_scale,
                "font": call_args[4] * preview_scale,
                "thickness": call_args[6] * preview_scale,
                "interval": (
                    next_y - call_args[2][1]
                ) * preview_scale,
            }

        legacy = draw_metrics((700, 1296))
        native = draw_metrics((2013, 3579))

        self.assertAlmostEqual(legacy["visual_scale"], 1.0)
        self.assertAlmostEqual(
            native["visual_scale"],
            min(2013 / 700, 3579 / 1296),
        )
        for key, delta in (
            ("x", 1.0),
            ("y", 1.0),
            ("font", 0.02),
            ("thickness", 0.25),
            ("interval", 1.0),
        ):
            self.assertAlmostEqual(legacy[key], native[key], delta=delta)

    def test_native_status_hints_scale_font_anchor_and_line_spacing(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "game_window": {
                "coordinate_reference_size": (700, 1296),
            },
            "bot": {"mode": "debug", "attack": "directional"},
        }
        bot._base_cfg = bot.cfg
        bot.img_frame_debug = np.zeros((2013, 3579, 3), dtype=np.uint8)
        bot.frame = SimpleNamespace(shape=(2013, 3579, 3))
        bot.t_last_frame = 1.0
        bot.kb = SimpleNamespace(t_last_screenshot=0.0, is_enable=True)
        bot.fsm = SimpleNamespace(state=SimpleNamespace(name="debug"))
        bot.screen_player_location_valid = False
        bot.loc_minimap = (10, 10)
        bot.img_minimap_screen = np.zeros((92, 203, 3), dtype=np.uint8)

        with (
            patch("src.engine.MapleStoryAutoLevelUp.time.time", return_value=2.0),
            patch("src.engine.MapleStoryAutoLevelUp.cv2.putText") as put_text,
        ):
            bot.update_info_on_img_frame_debug()

        hint_calls = [
            item for item in put_text.call_args_list
            if item.args[1].startswith(("FPS:", "State:", "Resolution:", "Press "))
        ]
        self.assertEqual(len(hint_calls), 6)
        visual_scale = min(2013 / 700, 3579 / 1296)
        self.assertAlmostEqual(hint_calls[0].args[4], 0.7 * visual_scale)
        self.assertEqual(hint_calls[0].args[6], 6)
        self.assertEqual(hint_calls[0].args[2][0], 28)
        baselines = [item.args[2][1] for item in hint_calls]
        self.assertEqual(
            [b - a for a, b in zip(baselines, baselines[1:])],
            [64] * 5,
        )

    def test_scaled_rectangle_preserves_fill_and_scales_label_gap(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        with (
            patch("src.utils.common.cv2.rectangle") as rectangle,
            patch("src.utils.common.cv2.putText") as put_text,
        ):
            draw_rectangle(
                image,
                (40, 50),
                (20, 30),
                (0, 0, 255),
                "box",
                thickness=-1,
                text_height=0.7,
                visual_scale=3,
            )

        self.assertEqual(rectangle.call_args.args[-1], -1)
        self.assertEqual(put_text.call_args.args[2], (40, 20))
        self.assertAlmostEqual(put_text.call_args.args[4], 2.1)
        self.assertEqual(put_text.call_args.args[6], 3)

    def test_rune_debug_scales_strokes_but_not_hough_radius(self):
        solver = RuneSolver.__new__(RuneSolver)
        solver.cfg = {
            "game_window": {
                "coordinate_reference_size": (700, 1296),
            }
        }
        image = np.zeros((2013, 3579, 3), dtype=np.uint8)
        with patch("src.engine.RuneSolver.draw_circle") as circle:
            solver._draw_debug_circle(
                image, (100, 100), 32, (0, 255, 0), 2
            )
            solver._draw_debug_circle(
                image,
                (100, 100),
                5,
                (0, 255, 255),
                -1,
                scale_radius=True,
            )

        hough_args = circle.call_args_list[0].args
        marker_args = circle.call_args_list[1].args
        self.assertEqual(hough_args[2], 32)
        self.assertEqual(hough_args[4], 6)
        self.assertEqual(marker_args[2], 14)
        self.assertEqual(marker_args[4], -1)


class MinimapDetectionTests(unittest.TestCase):
    def test_wide_shallow_minimap_survives_potplayer_normalization(self):
        frame = np.zeros((700, 1296, 3), dtype=np.uint8)
        # Henesys East Field becomes a roughly 101x47 outer border after the
        # 2768x1557 PotPlayer viewport is normalized to the working raster.
        frame[47, 7:108] = 235
        frame[93, 7:108] = 235
        frame[47:94, 7] = 235
        frame[47:94, 107] = 235
        frame[48:93, 8:107] = 40

        self.assertEqual(get_minimap_loc_size(frame), (8, 48, 99, 45))

    def test_near_white_capture_card_border_is_detected(self):
        frame = np.zeros((700, 1296, 3), dtype=np.uint8)
        # Simulate an antialiased border that no longer contains pure white.
        frame[43, 3:129] = 235
        frame[117, 3:129] = 235
        frame[43:118, 3] = 235
        frame[43:118, 128] = 235
        frame[44:117, 4:128] = 40

        self.assertEqual(get_minimap_loc_size(frame), (4, 44, 124, 73))

    def test_partial_white_ui_lines_are_rejected(self):
        frame = np.zeros((700, 1296, 3), dtype=np.uint8)
        frame[30, 20:220] = 255
        frame[120, 20:220] = 255
        frame[30:121, 20] = 255

        self.assertIsNone(get_minimap_loc_size(frame))

    def test_solid_white_ui_block_is_rejected(self):
        frame = np.zeros((700, 1296, 3), dtype=np.uint8)
        frame[20:100, 10:210] = 255

        self.assertIsNone(get_minimap_loc_size(frame))

    def test_thick_capture_border_is_fully_removed(self):
        frame = np.zeros((700, 1296, 3), dtype=np.uint8)
        frame[40:42, 5:131] = 235
        frame[114:117, 5:131] = 235
        frame[40:117, 5:7] = 235
        frame[40:117, 129:131] = 235
        frame[42:114, 7:129] = 40

        self.assertEqual(get_minimap_loc_size(frame), (7, 42, 122, 72))

    def test_antialiased_pixel_before_border_is_ignored(self):
        frame = np.zeros((700, 1296, 3), dtype=np.uint8)
        frame[39, 5] = 235
        frame[40:42, 5:131] = 235
        frame[114:117, 5:131] = 235
        frame[39:117, 5:7] = 235
        frame[39:117, 129:131] = 235
        frame[42:114, 7:129] = 40

        self.assertEqual(get_minimap_loc_size(frame), (7, 42, 122, 72))

    def test_minimap_native_raster_is_not_resized(self):
        minimap = np.zeros((70, 120, 3), dtype=np.uint8)

        native = copy_minimap_native_raster(minimap)

        self.assertEqual(native.shape, (70, 120, 3))
        self.assertTrue(np.array_equal(native, minimap))
        self.assertIsNot(native, minimap)

    def test_minimap_player_location_is_not_rescaled_or_rounded_again(self):
        self.assertEqual(
            copy_minimap_native_location((211, 137)),
            (211, 137),
        )

    def test_legacy_route_map_smaller_than_native_minimap_is_rejected(self):
        legacy_map = np.zeros((322, 240, 3), dtype=np.uint8)
        native_minimap = np.zeros((293, 350, 3), dtype=np.uint8)

        self.assertFalse(
            route_map_can_fit_minimap(legacy_map, native_minimap)
        )

    def test_native_route_map_can_contain_native_minimap(self):
        native_map = np.zeros((410, 480, 3), dtype=np.uint8)
        native_minimap = np.zeros((293, 350, 3), dtype=np.uint8)

        self.assertTrue(
            route_map_can_fit_minimap(native_map, native_minimap)
        )

    def test_player_dot_allows_capture_card_color_drift(self):
        minimap = np.zeros((50, 80, 3), dtype=np.uint8)
        minimap[20:22, 30:32] = (125, 198, 237)
        minimap[5, 70] = (125, 198, 237)

        location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=(136, 255, 255),
            color_tolerance=65,
        )

        self.assertEqual(location, (30, 20))

    def test_player_dot_prefers_color_accuracy_over_larger_terrain_blob(self):
        minimap = np.zeros((50, 80, 3), dtype=np.uint8)
        minimap[20:22, 30:32] = (130, 245, 250)
        minimap[3:6, 68:72] = (190, 200, 200)

        location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=(136, 255, 255),
            color_tolerance=65,
        )

        self.assertEqual(location, (30, 20))

    def test_three_pixel_player_dot_survives_hdmi_downscaling(self):
        minimap = np.zeros((50, 80, 3), dtype=np.uint8)
        minimap[20, 30:33] = (104, 241, 233)

        location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=(136, 255, 255),
            color_tolerance=65,
            min_component_area=3,
        )

        self.assertEqual(location, (31, 20))

    def test_two_pixel_player_dot_survives_hdmi_downscaling(self):
        minimap = np.zeros((50, 80, 3), dtype=np.uint8)
        minimap[20, 30:32] = (98, 243, 213)

        location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=(136, 255, 255),
            color_tolerance=65,
            min_component_area=2,
        )

        self.assertEqual(location, (30, 20))

    def test_player_dot_on_rope_uses_compact_hsv_fallback(self):
        minimap = np.zeros((149, 180, 3), dtype=np.uint8)
        # Real capture-card colors from the marker while the player hangs on a
        # green rope.  Every pixel is outside the 65-point BGR cube.
        minimap[52, 104] = (42, 253, 222)
        minimap[52, 105] = (15, 224, 189)
        minimap[53, 103] = (52, 248, 221)
        minimap[53, 104] = (49, 255, 230)
        minimap[53, 105] = (42, 254, 221)
        minimap[53, 106] = (139, 197, 187)
        minimap[54, 104] = (43, 253, 223)
        minimap[54, 105] = (17, 226, 192)

        location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=(136, 255, 255),
            color_tolerance=65,
            min_component_area=2,
        )

        self.assertEqual(location, (104, 53))

    def test_hsv_fallback_rejects_large_yellow_map_artwork(self):
        minimap = np.zeros((149, 180, 3), dtype=np.uint8)
        minimap[30:50, 60:80] = (49, 255, 230)

        location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=(136, 255, 255),
            color_tolerance=65,
            min_component_area=2,
        )

        self.assertIsNone(location)


class AutoBotLifecycleTests(unittest.TestCase):
    @staticmethod
    def _route_color_bot():
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.loc_player_global = (10, 10)
        bot.img_route = np.zeros((21, 21, 3), dtype=np.uint8)
        bot.img_route_debug = None
        bot.img_frame_debug = None
        bot.idx_routes = 0
        bot.color_code = {
            (255, 0, 0): "left none none",
            (0, 0, 255): "right none none",
            (255, 0, 255): "none none jump",
            (255, 0, 127): "none up teleport",
            (255, 255, 0): "none none goal",
            (0, 255, 127): "stop stop stop",
        }
        bot.color_code_up_down = {
            (127, 127, 127): "none up none",
        }
        bot.cfg = {
            "route": {"search_range": 10},
            "teleport": {
                "is_use_teleport_to_walk": False,
                "cooldown": 1,
            },
            "key": {"teleport": "e"},
        }
        bot.img_routes = [bot.img_route]
        bot.is_on_ladder = False
        bot.cmd_move_x = "none"
        bot.cmd_move_y = "none"
        bot.cmd_action = "none"
        bot.t_last_teleport = time.time()
        bot.is_near_edge = Mock(return_value=False)
        return bot

    def test_route_point_actions_are_ignored_until_player_pixel_overlaps(self):
        bot = self._route_color_bot()
        bot.img_route[10, 9] = (255, 0, 255)

        color_code, color_code_up_down = bot.get_nearest_color_code()

        self.assertIsNone(color_code)
        self.assertIsNone(color_code_up_down)

    def test_route_point_actions_trigger_at_player_center_pixel(self):
        action_cases = {
            (255, 0, 255): "none none jump",
            (255, 0, 127): "none up teleport",
            (255, 255, 0): "none none goal",
            (0, 255, 127): "stop stop stop",
        }
        for color, command in action_cases.items():
            with self.subTest(command=command):
                bot = self._route_color_bot()
                bot.img_route[10, 10] = color

                color_code, _ = bot.get_nearest_color_code()

                self.assertIsNotNone(color_code)
                self.assertEqual(color_code["pixel"], (10, 10))
                self.assertEqual(color_code["command"], command)
                self.assertEqual(color_code["distance"], 0)
                self.assertTrue(color_code["exact_action"])

    def test_route_movement_colors_keep_search_range_tolerance(self):
        bot = self._route_color_bot()
        bot.img_route[10, 4] = (255, 0, 0)
        bot.img_route[6, 10] = (127, 127, 127)

        color_code, color_code_up_down = bot.get_nearest_color_code()

        self.assertEqual(color_code["command"], "left none none")
        self.assertEqual(color_code["distance"], 6)
        self.assertFalse(color_code["exact_action"])
        self.assertEqual(color_code_up_down["command"], "none up none")
        self.assertEqual(color_code_up_down["distance"], 4)

    def test_exact_action_has_priority_over_ladder_complement(self):
        bot = self._route_color_bot()
        bot.img_route[10, 10] = (255, 0, 255)
        bot.img_route[9, 10] = (127, 127, 127)
        bot.is_on_ladder = True

        bot.update_cmd_by_route()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("none", "none", "jump"),
        )

    def test_nearby_action_does_not_override_movement(self):
        bot = self._route_color_bot()
        bot.img_route[10, 6] = (255, 0, 255)
        bot.img_route[10, 10] = (255, 0, 0)

        bot.update_cmd_by_route()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "none"),
        )

    @staticmethod
    def _route_recovery_bot(route_count=4, active_index=0):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.img_routes = [
            np.full((4, 4, 3), index, dtype=np.uint8)
            for index in range(route_count)
        ]
        bot.idx_routes = active_index
        bot.img_route = bot.img_routes[active_index]
        bot.img_route_debug = None
        bot.is_show_debug_window = False
        bot.is_on_ladder = False
        bot.cmd_move_x = "left"
        bot.cmd_move_y = "up"
        bot.cmd_action = "jump"
        bot.cfg = {
            "teleport": {
                "is_use_teleport_to_walk": False,
                "cooldown": 1,
            },
            "key": {"teleport": ""},
        }
        bot.t_last_teleport = time.time()
        bot.is_near_edge = Mock(return_value=False)
        return bot

    def test_missing_route_action_scans_forward_and_executes_same_frame(self):
        bot = self._route_recovery_bot(route_count=4, active_index=0)

        def find_action_for_active_route():
            if bot.idx_routes == 2:
                return ({
                    "command": "right none jump",
                    "distance": 0,
                }, None)
            return (None, None)

        bot.get_nearest_color_code = Mock(
            side_effect=find_action_for_active_route
        )

        bot.update_cmd_by_route()

        self.assertEqual(bot.idx_routes, 2)
        self.assertIs(bot.img_route, bot.img_routes[2])
        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("right", "none", "jump"),
        )
        self.assertEqual(bot.get_nearest_color_code.call_count, 3)

    def test_route_recovery_wraps_from_last_route_to_first(self):
        bot = self._route_recovery_bot(route_count=3, active_index=2)

        def find_action_for_active_route():
            if bot.idx_routes == 0:
                return ({
                    "command": "left none none",
                    "distance": 0,
                }, None)
            return (None, None)

        bot.get_nearest_color_code = Mock(
            side_effect=find_action_for_active_route
        )

        bot.update_cmd_by_route()

        self.assertEqual(bot.idx_routes, 0)
        self.assertEqual(bot.cmd_move_x, "left")
        self.assertEqual(bot.get_nearest_color_code.call_count, 2)

    def test_no_route_action_restores_route_and_releases_stale_command(self):
        bot = self._route_recovery_bot(route_count=3, active_index=1)
        bot.get_nearest_color_code = Mock(return_value=(None, None))

        bot.update_cmd_by_route()

        self.assertEqual(bot.idx_routes, 1)
        self.assertIs(bot.img_route, bot.img_routes[1])
        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("none", "none", "none"),
        )
        self.assertEqual(bot.get_nearest_color_code.call_count, 3)

    def _make_real_nametag_anchor_bot(self, frame):
        project_root = Path(__file__).resolve().parents[1]
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "mode": "grayscale",
                "split_width": 30,
                "global_diff_thres": 0.2,
                "cache_accept_thres": 0.12,
                "diff_thres": 0.15,
                "max_stale_frames": 2,
                "jump_confirm_distance": 40,
                "jump_confirm_radius": 12,
                "offset": (0, 30),
                "medal": {
                    "enable": True,
                    "diff_thres": 0.16,
                    "assisted_id_diff_thres": 0.24,
                    "id_fragment_width": 30,
                    "id_fragment_stride": 15,
                    "center_offset_x": 3,
                    "vertical_gap": 0,
                    "search_tolerance": (18, 6),
                    "bottom_min_visible_ratio": 0.5,
                    "bottom_partial_diff_thres": 0.16,
                    "bottom_partial_id_diff_thres": 0.18,
                },
                "pet": {
                    "enable": True,
                    "diff_thres": 0.16,
                    "medal_offset": (37, 17),
                    "medal_search_tolerance": (28, 10),
                },
            },
            "ui_coords": {"ui_y_start": frame.shape[0]},
        }
        bot.img_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bot.img_frame_debug = None

        # Keep these regression tests independent from the user's live hero
        # resources.  The production templates can be recaptured at a new
        # PotPlayer resolution, while this fixture intentionally remains at
        # the legacy geometry represented by nametag_anchors.png.
        pristine = cv2.imread(str(
            project_root / "tests" / "fixtures" / "nametag_anchors.png"
        ))
        self.assertIsNotNone(pristine)
        bot.img_nametag = pristine[71:91, 98:143].copy()
        bot.img_nametag_gray = cv2.cvtColor(
            bot.img_nametag, cv2.COLOR_BGR2GRAY
        )
        bot.img_nametag_medal = pristine[91:105, 75:171].copy()
        bot.img_nametag_medal_gray = cv2.cvtColor(
            bot.img_nametag_medal, cv2.COLOR_BGR2GRAY
        )
        bot.img_nametag_pet = pristine[74:88, 38:91].copy()
        bot.img_nametag_pet_gray = cv2.cvtColor(
            bot.img_nametag_pet, cv2.COLOR_BGR2GRAY
        )
        bot.loc_nametag = (0, 0)
        bot.has_valid_nametag_location = False
        bot.nametag_miss_count = 0
        bot.pending_nametag_location = None
        return bot

    def _make_appearance_bot(self, fixture_name, *, confirm_frames=1):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root / "tests" / "fixtures" / fixture_name
        ))
        self.assertIsNotNone(frame)
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        templates = (
            ("liu_muning_appearance_climb", "appearance_climb.png",
             "climbing", (20, 38), (35, 9, 44, 30)),
            ("liu_muning_appearance_stand_left", "appearance_stand_left.png",
             "standing", (18, 41), (10, 10, 45, 44)),
            ("liu_muning_appearance_stand_right", "appearance_stand_right.png",
             "standing", (25, 39), (45, 13, 46, 44)),
        )
        bot.cfg = {
            "nametag": {
                "offset": (0, 30),
                "appearance": {
                    "enable": True,
                    "diff_thres": 0.16,
                    "climb_diff_thres": 0.12,
                    "local_search_radius": 90,
                    "validation_distance": 30,
                    "climb_validation_distance": 80,
                    "global_confirm_frames": confirm_frames,
                    "global_confirm_radius": 12,
                },
            },
            "ui_coords": {"ui_y_start": frame.shape[0]},
        }
        bot.img_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bot.img_frame_debug = frame.copy()
        bot.img_nametag = cv2.imread(str(
            project_root / "nametag" / "liu_muning.png"
        ))
        bot.nametag_appearance_templates = []
        for name, fixture, pose, player_offset, crop in templates:
            fixture_image = cv2.imread(str(
                project_root / "tests" / "fixtures" / fixture
            ))
            self.assertIsNotNone(fixture_image)
            x, y, width, height = crop
            image = fixture_image[y:y+height, x:x+width].copy()
            bot.nametag_appearance_templates.append({
                "name": name,
                "pose": pose,
                "image": image,
                "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                "mask": np.where(
                    np.all(image == (0, 255, 0), axis=2), 0, 255
                ).astype(np.uint8),
                "player_offset": player_offset,
            })
        bot.is_on_ladder = False
        bot.has_valid_nametag_location = False
        bot.loc_nametag = (0, 0)
        bot.pending_appearance_location = None
        bot.pending_appearance_count = 0
        bot.has_valid_appearance_location = False
        bot.last_appearance_match = None
        return bot

    def test_id_and_medal_detect_real_capture(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        bot = self._make_real_nametag_anchor_bot(frame)

        location = bot.get_player_location_by_nametag()

        self.assertEqual(location, (120, 41))
        self.assertEqual(bot.loc_nametag, (98, 71))

    def test_color_templates_are_normalized_before_nametag_matching(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        bot = self._make_real_nametag_anchor_bot(frame)

        # Reproduce the UI failure where resources carrying the *_gray name
        # were unexpectedly still BGR images after a config reload.
        bot.img_nametag_gray = bot.img_nametag.copy()
        bot.img_nametag_medal_gray = bot.img_nametag_medal.copy()
        bot.img_nametag_pet_gray = bot.img_nametag_pet.copy()

        location = bot.get_player_location_by_nametag()

        self.assertEqual(location, (120, 41))
        self.assertEqual(bot.loc_nametag, (98, 71))

    def test_weak_id_without_matching_medal_is_rejected(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        no_medal = frame.copy()
        no_medal[89:107, 68:178] = no_medal[108:126, 68:178]
        bot = self._make_real_nametag_anchor_bot(no_medal)

        location = bot.get_player_location_by_nametag()

        self.assertIsNone(location)

    def test_pet_and_medal_recover_when_entire_id_is_covered(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        covered = frame.copy()
        covered[69:91, 95:147] = covered[40:62, 95:147]
        bot = self._make_real_nametag_anchor_bot(covered)

        location = bot.get_player_location_by_nametag()

        self.assertEqual(location, (120, 41))
        self.assertEqual(bot.loc_nametag, (98, 71))

    def test_bottom_clipped_medal_still_validates_id(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        # The ID ends at y=91 and the 14-row medal starts there. Keep exactly
        # its upper 7 rows, reproducing the game/UI boundary on the last floor.
        clipped = frame[:98].copy()
        bot = self._make_real_nametag_anchor_bot(clipped)

        location = bot.get_player_location_by_nametag()

        self.assertEqual(location, (120, 41))
        self.assertEqual(bot.loc_nametag, (98, 71))

    def test_bottom_medal_with_less_than_half_visible_is_rejected(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        # Six of fourteen rows is below the configured 50% minimum.
        clipped = frame[:97].copy()
        bot = self._make_real_nametag_anchor_bot(clipped)

        self.assertIsNone(bot.get_player_location_by_nametag())

    def test_bottom_partial_medal_does_not_accept_background(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        no_medal = frame.copy()
        no_medal[91:98, 57:183] = no_medal[108:115, 57:183]
        clipped = no_medal[:98].copy()
        bot = self._make_real_nametag_anchor_bot(clipped)

        self.assertIsNone(bot.get_player_location_by_nametag())

    def test_pet_and_bottom_clipped_medal_recover_covered_id(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "nametag_anchors.png"
        ))
        self.assertIsNotNone(frame)
        covered = frame.copy()
        covered[69:91, 95:147] = covered[40:62, 95:147]
        clipped = covered[:98].copy()
        bot = self._make_real_nametag_anchor_bot(clipped)

        location = bot.get_player_location_by_nametag()

        self.assertEqual(location, (120, 41))
        self.assertEqual(bot.loc_nametag, (98, 71))

    def test_standing_head_left_recovers_player_location(self):
        bot = self._make_appearance_bot("appearance_stand_left.png")

        location = bot.get_player_location_by_appearance()

        self.assertEqual(location, (28, 51))
        self.assertEqual(bot.last_appearance_match["pose"], "standing")

    def test_standing_head_right_recovers_player_location(self):
        bot = self._make_appearance_bot("appearance_stand_right.png")

        location = bot.get_player_location_by_appearance()

        self.assertEqual(location, (70, 52))
        self.assertEqual(bot.last_appearance_match["pose"], "standing")

    def test_global_appearance_recovery_requires_consistent_frames(self):
        bot = self._make_appearance_bot(
            "appearance_stand_right.png", confirm_frames=2
        )

        self.assertIsNone(bot.get_player_location_by_appearance())
        self.assertEqual(bot.pending_appearance_count, 1)
        self.assertEqual(bot.get_player_location_by_appearance(), (70, 52))
        self.assertEqual(bot.get_player_location_by_appearance(), (70, 52))

    def test_climbing_hood_is_not_used_for_unrestricted_global_search(self):
        bot = self._make_appearance_bot("appearance_climb.png")

        self.assertIsNone(bot.get_player_location_by_appearance())

    def test_climbing_hood_matches_near_expected_player(self):
        bot = self._make_appearance_bot("appearance_climb.png")

        location = bot.get_player_location_by_appearance(
            expected_player=(55, 47),
            allow_global=False,
        )

        self.assertEqual(location, (55, 47))
        self.assertEqual(bot.last_appearance_match["pose"], "climbing")

    def test_smile_anchored_climbing_hood_enters_ladder_state(self):
        bot = self._make_appearance_bot("appearance_climb.png")

        state = bot._update_ladder_state_from_smile_pose((55, 47))

        self.assertTrue(state)
        self.assertTrue(bot.is_on_ladder)

    def test_smile_anchored_standing_head_leaves_ladder_state(self):
        bot = self._make_appearance_bot("appearance_stand_left.png")
        bot.is_on_ladder = True

        state = bot._update_ladder_state_from_smile_pose((28, 51))

        self.assertFalse(state)
        self.assertFalse(bot.is_on_ladder)

    def test_inconclusive_smile_anchored_pose_keeps_ladder_state(self):
        bot = self._make_appearance_bot("appearance_climb.png")
        bot.img_frame_gray[:] = 0
        bot.is_on_ladder = True

        state = bot._update_ladder_state_from_smile_pose((55, 47))

        self.assertIsNone(state)
        self.assertTrue(bot.is_on_ladder)

    def test_nametag_entry_recovers_from_head_when_all_text_is_hidden(self):
        project_root = Path(__file__).resolve().parents[1]
        frame = cv2.imread(str(
            project_root
            / "tests"
            / "fixtures"
            / "appearance_stand_right.png"
        ))
        self.assertIsNotNone(frame)
        frame[75:, :] = 0
        bot = self._make_real_nametag_anchor_bot(frame)
        appearance_bot = self._make_appearance_bot(
            "appearance_stand_right.png", confirm_frames=2
        )
        bot.cfg["nametag"]["appearance"] = appearance_bot.cfg[
            "nametag"
        ]["appearance"]
        bot.nametag_appearance_templates = (
            appearance_bot.nametag_appearance_templates
        )
        bot.pending_appearance_location = None
        bot.pending_appearance_count = 0
        bot.has_valid_appearance_location = False
        bot.last_appearance_match = None
        bot.loc_appearance_player = (0, 0)
        bot.is_on_ladder = False

        self.assertIsNone(bot.get_player_location_by_nametag())
        self.assertEqual(bot.get_player_location_by_nametag(), (70, 52))
        self.assertEqual(bot.last_appearance_match["pose"], "standing")

    def test_nametag_first_miss_does_not_return_fake_player_location(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "mode": "grayscale",
                "split_width": 30,
                "global_diff_thres": 0.2,
                "diff_thres": 0.2,
                "offset": (0, 30),
            },
            "ui_coords": {"ui_y_start": 90},
        }
        bot.img_frame_gray = np.zeros((100, 160), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.img_nametag = np.full((12, 40, 3), (0, 255, 0), dtype=np.uint8)
        bot.img_nametag[:, 4:36] = 255
        bot.img_nametag_gray = np.full((12, 40), 255, dtype=np.uint8)
        bot.loc_nametag = (0, 0)
        bot.has_valid_nametag_location = False
        bot.is_first_frame = True

        with patch(
            "src.engine.MapleStoryAutoLevelUp.find_pattern_sqdiff",
            return_value=((50, 30), 0.8, False),
        ):
            location = bot.get_player_location_by_nametag()

        self.assertIsNone(location)
        self.assertEqual(bot.loc_nametag, (0, 0))
        self.assertFalse(bot.has_valid_nametag_location)

    def test_ambiguous_cached_nametag_yields_to_better_global_match(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "mode": "grayscale",
                "split_width": 30,
                "global_diff_thres": 0.2,
                "cache_accept_thres": 0.12,
                "diff_thres": 0.15,
                "max_stale_frames": 2,
                "offset": (0, 30),
            },
            "ui_coords": {"ui_y_start": 90},
        }
        bot.img_frame_gray = np.zeros((100, 160), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.img_nametag = np.full((12, 40, 3), (0, 255, 0), dtype=np.uint8)
        bot.img_nametag[:, 4:36] = 255
        bot.img_nametag_gray = np.full((12, 40), 255, dtype=np.uint8)
        bot.loc_nametag = (10, 10)
        bot.has_valid_nametag_location = True
        bot.nametag_miss_count = 0
        bot.pending_nametag_location = None

        with patch(
            "src.engine.MapleStoryAutoLevelUp.find_pattern_sqdiff",
            side_effect=[
                ((60, 40), 0.19, True),
                ((90, 55), 0.09, False),
            ],
        ) as matcher:
            location = bot.get_player_location_by_nametag()

        self.assertEqual(matcher.call_count, 2)
        self.assertEqual(bot.loc_nametag, (50, 43))
        self.assertEqual(location, (70, 13))

    def test_strong_cached_nametag_keeps_fast_path(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "mode": "grayscale",
                "split_width": 30,
                "global_diff_thres": 0.2,
                "cache_accept_thres": 0.12,
                "diff_thres": 0.15,
                "max_stale_frames": 2,
                "offset": (0, 30),
            },
            "ui_coords": {"ui_y_start": 90},
        }
        bot.img_frame_gray = np.zeros((100, 160), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.img_nametag = np.full((12, 40, 3), (0, 255, 0), dtype=np.uint8)
        bot.img_nametag[:, 4:36] = 255
        bot.img_nametag_gray = np.full((12, 40), 255, dtype=np.uint8)
        bot.loc_nametag = (10, 10)
        bot.has_valid_nametag_location = True
        bot.nametag_miss_count = 0
        bot.pending_nametag_location = None

        with patch(
            "src.engine.MapleStoryAutoLevelUp.find_pattern_sqdiff",
            return_value=((60, 40), 0.08, True),
        ) as matcher:
            bot.get_player_location_by_nametag()

        matcher.assert_called_once()

    def test_rejected_nametag_expires_after_short_stale_window(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "mode": "grayscale",
                "split_width": 30,
                "global_diff_thres": 0.2,
                "cache_accept_thres": 0.12,
                "diff_thres": 0.15,
                "max_stale_frames": 2,
                "offset": (0, 30),
            },
            "ui_coords": {"ui_y_start": 90},
        }
        bot.img_frame_gray = np.zeros((100, 160), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.img_nametag = np.full((12, 40, 3), (0, 255, 0), dtype=np.uint8)
        bot.img_nametag[:, 4:36] = 255
        bot.img_nametag_gray = np.full((12, 40), 255, dtype=np.uint8)
        bot.loc_nametag = (25, 35)
        bot.has_valid_nametag_location = True
        bot.nametag_miss_count = 0
        bot.pending_nametag_location = None

        with patch(
            "src.engine.MapleStoryAutoLevelUp.find_pattern_sqdiff",
            return_value=((70, 60), 0.3, False),
        ):
            self.assertIsNone(bot.get_player_location_by_nametag())
            self.assertIsNone(bot.get_player_location_by_nametag())
            self.assertIsNone(bot.get_player_location_by_nametag())

        self.assertFalse(bot.has_valid_nametag_location)
        self.assertEqual(bot.nametag_miss_count, 3)

    def test_large_nametag_jump_requires_two_consistent_frames(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "mode": "grayscale",
                "split_width": 30,
                "global_diff_thres": 0.2,
                "cache_accept_thres": 0.12,
                "diff_thres": 0.15,
                "max_stale_frames": 2,
                "jump_confirm_distance": 40,
                "jump_confirm_radius": 12,
                "offset": (0, 30),
            },
            "ui_coords": {"ui_y_start": 150},
        }
        bot.img_frame_gray = np.zeros((160, 240), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.img_nametag = np.full((12, 40, 3), (0, 255, 0), dtype=np.uint8)
        bot.img_nametag[:, 4:36] = 255
        bot.img_nametag_gray = np.full((12, 40), 255, dtype=np.uint8)
        bot.loc_nametag = (10, 60)
        bot.has_valid_nametag_location = True
        bot.nametag_miss_count = 0
        bot.pending_nametag_location = None

        # Returned matcher coordinates include pad_x=40 and pad_y=12.
        with patch(
            "src.engine.MapleStoryAutoLevelUp.find_pattern_sqdiff",
            side_effect=[
                ((160, 72), 0.08, False),
                ((55, 72), 0.08, True),
                ((162, 73), 0.07, False),
            ],
        ):
            self.assertIsNone(bot.get_player_location_by_nametag())
            self.assertEqual(bot.loc_nametag, (10, 60))
            self.assertEqual(bot.pending_nametag_location, (120, 60))
            self.assertEqual(bot.get_player_location_by_nametag(), (142, 31))

        self.assertEqual(bot.loc_nametag, (122, 61))
        self.assertIsNone(bot.pending_nametag_location)

    def test_invalid_player_location_suppresses_combat_detection(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.screen_player_location_valid = False
        bot.monsters = [{"name": "stale"}]
        bot.cmd_action = "attack"
        bot.get_monsters_in_range = Mock()

        bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.monsters, [])
        self.assertEqual(bot.cmd_action, "none")
        bot.get_monsters_in_range.assert_not_called()

    def test_debug_state_detects_full_camera_without_keyboard_commands(self):
        bot = SimpleNamespace(
            img_frame=np.zeros((700, 1296, 3), dtype=np.uint8),
            cfg={
                "ui_coords": {"ui_y_start": 610},
                "debug": {
                    "monster_diff_thres": 0.25,
                    "scan_interval_frames": 5,
                },
            },
            monsters=[],
            get_debug_monsters_in_range=Mock(
                return_value=[{"name": "fire_boar"}]
            ),
            kb=Mock(),
        )

        DebugState("debug", bot).on_frame()

        bot.get_debug_monsters_in_range.assert_called_once_with(
            (0, 0), (1296, 610), score_thres=0.25
        )
        self.assertEqual(bot.monsters, [{"name": "fire_boar"}])
        bot.kb.set_command.assert_not_called()
        bot.kb.release_all_key.assert_not_called()

    def test_debug_state_reuses_detections_between_expensive_scans(self):
        bot = SimpleNamespace(
            img_frame=np.zeros((700, 1296, 3), dtype=np.uint8),
            cfg={
                "ui_coords": {"ui_y_start": 610},
                "debug": {
                    "monster_diff_thres": 0.25,
                    "scan_interval_frames": 5,
                },
            },
            monsters=[],
            get_debug_monsters_in_range=Mock(
                return_value=[{"name": "fire_boar"}]
            ),
            draw_monster_detections=Mock(),
        )
        state = DebugState("debug", bot)

        state.on_frame()
        state.on_frame()

        bot.get_debug_monsters_in_range.assert_called_once()
        bot.draw_monster_detections.assert_called_once_with(
            [{"name": "fire_boar"}], (0, 0), (1296, 610)
        )

    def test_debug_color_detection_draws_search_and_monster_boxes(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        template = np.full((12, 16, 3), (0, 255, 0), dtype=np.uint8)
        template[2:10, 3:13] = (20, 80, 180)
        mask = np.zeros((12, 16), dtype=np.uint8)
        mask[2:10, 3:13] = 255
        bot.cfg = {
            "bot": {"mode": "debug"},
            "monster_detect": {
                "mode": "color",
                "diff_thres": 0.8,
                "with_enemy_hp_bar": False,
            },
            "character": {"width": 10, "height": 10},
        }
        bot.img_frame = np.zeros((80, 120, 3), dtype=np.uint8)
        bot.img_frame[30:42, 50:66] = template
        bot.img_frame_debug = bot.img_frame.copy()
        bot.loc_player = (10, 10)
        bot.monsters_info = {"fire_boar": [(template, mask)]}

        with patch(
            "src.engine.MapleStoryAutoLevelUp.draw_rectangle"
        ) as draw:
            detections = bot.get_monsters_in_range(
                (0, 0), (120, 80), diff_thres=0.01
            )

        self.assertTrue(
            any(
                item["name"] == "fire_boar"
                and item["position"] == (50, 30)
                for item in detections
            )
        )
        self.assertTrue(
            any(
                call_item.args[-1] == "Mob Detection Box"
                for call_item in draw.call_args_list
            )
        )
        self.assertTrue(
            any(
                call_item.args[1] == (50, 30)
                for call_item in draw.call_args_list
            )
        )
        self.assertTrue(
            any(
                call_item.args[-1].startswith("fire_boar:")
                for call_item in draw.call_args_list
            )
        )

    def test_debug_full_frame_matching_limits_each_template_to_top_k(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        template = np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8)
        template[2:6, 2:6] = (20, 80, 180)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        bot.cfg = {
            "bot": {"mode": "debug"},
            "debug": {
                "template_top_k": 1,
                "local_peak_radius": 2,
                "monster_diff_thres": 0.1,
                "verify_color": False,
            },
            "monster_detect": {
                "mode": "color",
                "diff_thres": 0.8,
                "with_enemy_hp_bar": False,
            },
            "character": {"width": 10, "height": 10},
        }
        bot.img_frame = np.zeros((80, 120, 3), dtype=np.uint8)
        bot.img_frame[10:18, 10:18] = template
        bot.img_frame[50:58, 80:88] = template
        bot.img_frame_debug = None
        bot.loc_player = (0, 0)
        bot.monsters_info = {"fire_boar": [(template, mask)]}

        detections = bot.get_debug_monsters_in_range(
            (0, 0), (120, 80), score_thres=0.1
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["name"], "fire_boar")
        detected_x, detected_y = detections[0]["position"]
        self.assertTrue(
            any(
                abs(detected_x - expected_x) <= 2
                and abs(detected_y - expected_y) <= 2
                for expected_x, expected_y in ((10, 10), (80, 50))
            )
        )

    def test_normal_debug_pipeline_reuses_debug_matcher_in_combat_roi(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"mode": "normal"},
            "monster_detect": {"mode": "debug_pipeline"},
        }
        expected = [{"name": "slime"}]
        bot.get_debug_monsters_in_range = Mock(return_value=expected)

        detections = bot.get_monsters_in_range(
            (100, 200), (500, 350), diff_thres=0.31
        )

        self.assertIs(detections, expected)
        bot.get_debug_monsters_in_range.assert_called_once_with(
            (100, 200), (500, 350), score_thres=0.31
        )

    def test_normal_debug_pipeline_uses_debug_threshold_by_default(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"mode": "normal"},
            "monster_detect": {"mode": "debug_pipeline"},
        }
        bot.get_debug_monsters_in_range = Mock(return_value=[])

        bot.get_monsters_in_range((10, 20), (80, 60))

        bot.get_debug_monsters_in_range.assert_called_once_with(
            (10, 20), (80, 60), score_thres=None
        )

    def test_normal_matching_bounds_candidates_before_nms(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        rng = np.random.default_rng(7)
        template = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        mask = np.full((8, 8), 255, dtype=np.uint8)
        bot.cfg = {
            "bot": {"mode": "normal"},
            "monster_detect": {
                "mode": "color",
                # Accept every finite result to recreate the dense-candidate
                # failure without requiring a large screenshot fixture.
                "diff_thres": 1.0,
                "local_min_radius": 2,
                "max_candidates_per_template": 5,
                "with_enemy_hp_bar": False,
            },
            "character": {"width": 10, "height": 10},
        }
        bot.img_frame = rng.integers(
            0, 256, (80, 120, 3), dtype=np.uint8
        )
        bot.img_frame_debug = None
        bot.loc_player = (60, 40)
        bot.monsters_info = {"green_mushroom": [(template, mask)]}

        with patch(
            "src.engine.MapleStoryAutoLevelUp.nms",
            side_effect=lambda detections, iou_threshold: detections,
        ) as nms_mock:
            detections = bot.get_monsters_in_range((0, 0), (120, 80))

        candidates_before_nms = nms_mock.call_args.args[0]
        self.assertLessEqual(len(candidates_before_nms), 5)
        self.assertLessEqual(len(detections), len(candidates_before_nms))
        self.assertTrue(
            all(item in candidates_before_nms for item in detections)
        )

    def test_normal_matching_uses_per_monster_threshold(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        template = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.full((8, 8), 255, dtype=np.uint8)
        bot.cfg = {
            "bot": {"mode": "normal"},
            "monster_detect": {
                "mode": "color",
                "diff_thres": 0.35,
                "diff_thres_by_monster": {"snail": 0.28},
                "local_min_radius": 2,
                "max_candidates_per_template": 5,
                "with_enemy_hp_bar": False,
            },
            "character": {"width": 10, "height": 10},
        }
        bot.img_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.loc_player = (30, 20)
        bot.monsters_info = {
            "snail": [(template, mask)],
            "slime": [(template, mask)],
        }

        synthetic_result = np.full((33, 53), 0.30, dtype=np.float32)
        with patch(
            "src.engine.MapleStoryAutoLevelUp.cv2.matchTemplate",
            return_value=synthetic_result,
        ):
            detections = bot.get_monsters_in_range((0, 0), (60, 40))

        self.assertFalse(any(item["name"] == "snail" for item in detections))
        self.assertTrue(any(item["name"] == "slime" for item in detections))

    def test_debug_matching_uses_per_monster_threshold(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        template = np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.full((8, 8), 255, dtype=np.uint8)
        bot.cfg = {
            "bot": {"mode": "debug"},
            "debug": {
                "monster_diff_thres": 0.30,
                "monster_diff_thres_by_monster": {"snail": 0.45},
                "template_top_k": 1,
                "local_peak_radius": 2,
                "verify_color": False,
            },
        }
        bot.img_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.monsters_info = {
            "snail": [(template, mask)],
            "slime": [(template, mask)],
        }

        synthetic_result = np.full((33, 53), 0.35, dtype=np.float32)
        with patch(
            "src.engine.MapleStoryAutoLevelUp.cv2.matchTemplate",
            return_value=synthetic_result,
        ):
            detections = bot.get_debug_monsters_in_range(
                (0, 0), (60, 40)
            )

        self.assertFalse(any(item["name"] == "snail" for item in detections))
        self.assertTrue(any(item["name"] == "slime" for item in detections))

    def test_debug_edge_candidate_requires_color_verification(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        template = np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8)
        template[2:6, 2:6] = (20, 80, 180)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        bot.cfg = {
            "bot": {"mode": "debug"},
            "debug": {
                "monster_diff_thres": 0.30,
                "template_top_k": 1,
                "local_peak_radius": 2,
                "verify_color": True,
                "color_verify_candidates": 1,
            },
            "monster_detect": {
                "diff_thres": 0.8,
                "diff_thres_by_monster": {"snail": 0.32},
            },
        }
        bot.img_frame = np.zeros((40, 60, 3), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.monsters_info = {"snail": [(template, mask)]}

        edge_result = np.full((33, 53), 0.10, dtype=np.float32)
        edge_result[5, 7] = 0.40
        with patch(
            "src.engine.MapleStoryAutoLevelUp.cv2.matchTemplate",
            side_effect=[edge_result, np.array([[0.35]], dtype=np.float32)],
        ):
            detections = bot.get_debug_monsters_in_range(
                (0, 0), (60, 40)
            )

        self.assertEqual(detections, [])

    def test_debug_loads_selected_map_monsters_without_routes(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.data = {
            "map_mobs_mapping": {
                "fire_land_1": ["black_axe_stump", "fire_boar"]
            }
        }
        bot.img_map = np.ones((2, 2, 3), dtype=np.uint8)
        bot.img_route = np.ones((2, 2, 3), dtype=np.uint8)
        bot.img_route_debug = np.ones((2, 2, 3), dtype=np.uint8)
        bot.img_routes = [np.ones((2, 2, 3), dtype=np.uint8)]
        bot.monsters_info = {"stale_map_monster": []}
        cfg = {
            "bot": {"mode": "debug", "map": "fire_land_1"},
            "esp32_hid": {"remote_target": False},
            "route": {"color_code": {}, "color_code_up_down": {}},
            "nametag": {"enable": False},
            "system": {"language": "cn"},
            "rune_warning_cn": {"top_left": (0, 0), "bottom_right": (1, 1)},
            "rune_warning_eng": {"top_left": (0, 0), "bottom_right": (1, 1)},
            "rune_enable_msg_cn": {"top_left": (0, 0), "bottom_right": (1, 1)},
            "rune_enable_msg_eng": {"top_left": (0, 0), "bottom_right": (1, 1)},
            "rune_solver": {"arrow_box_coord": (0, 0)},
            "ui_coords": {
                "login_button_top_left": (0, 0),
                "login_button_bottom_right": (1, 1),
            },
            "game_window": {"size": (693, 1282)},
        }
        monster_image = np.zeros((8, 10, 3), dtype=np.uint8)

        def fake_glob(pattern):
            if pattern.startswith("monster/black_axe_stump/"):
                return ["monster/black_axe_stump/black_axe_stump_1.png"]
            if pattern.startswith("monster/fire_boar/"):
                return ["monster/fire_boar/fire_boar_1.png"]
            raise AssertionError(f"Debug mode unexpectedly requested routes: {pattern}")

        def fake_load(path, flags=None):
            if path.startswith("monster/"):
                return monster_image.copy()
            if path.startswith("minimaps/"):
                raise AssertionError("Debug mode must not load map or route images")
            return np.zeros((2, 2, 3), dtype=np.uint8)

        with patch(
            "src.engine.MapleStoryAutoLevelUp.glob.glob",
            side_effect=fake_glob,
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.load_image",
            side_effect=fake_load,
        ):
            self.assertEqual(bot.load_config(cfg), 0)

        self.assertEqual(
            set(bot.monsters_info), {"black_axe_stump", "fire_boar"}
        )
        self.assertEqual(len(bot.monsters_info["black_axe_stump"]), 2)
        self.assertEqual(len(bot.monsters_info["fire_boar"]), 2)
        self.assertIsNone(bot.img_map)
        self.assertIsNone(bot.img_route)
        self.assertIsNone(bot.img_route_debug)
        self.assertEqual(bot.img_routes, [])

    def test_debug_start_disables_esp32_and_health_monitor(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.thread_auto_bot = None
        bot.is_terminated = False
        bot.is_disable_control = False
        bot.args = SimpleNamespace(test_image="", init_state="")
        bot.cfg = {
            "bot": {"mode": "debug"},
            "health_monitor": {"enable": True},
            "esp32_hid": {"remote_target": True},
        }
        bot.fsm = Mock()
        bot.loop = Mock()
        bot._shutdown_lock = threading.Lock()
        bot._components_stopped = True
        capture = Mock()
        keyboard = Mock()
        keyboard.is_terminated = False
        health = Mock()

        with patch(
            "src.engine.MapleStoryAutoLevelUp.GameWindowCapturor",
            return_value=capture,
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.KeyBoardController",
            return_value=keyboard,
        ) as keyboard_cls, patch(
            "src.engine.MapleStoryAutoLevelUp.HealthMonitor",
            return_value=health,
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.Profiler",
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.RuneSolver",
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.threading.Thread"
        ) as thread_cls:
            bot.start()

        keyboard_cls.assert_called_once_with(
            bot.cfg,
            connect_input=False,
            capture_available=False,
        )
        health.start.assert_not_called()
        bot.fsm.set_init_state.assert_called_once_with("debug")
        thread_cls.return_value.start.assert_called_once_with()

    def test_debug_loop_skips_local_window_activation_and_party_setup(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"mode": "debug"},
            "esp32_hid": {"remote_target": False},
            "system": {"fps_limit_main": 10},
        }
        bot.capture = SimpleNamespace(window_title="PotPlayer")
        bot.is_terminated = True
        bot.kb = SimpleNamespace(is_terminated=False)
        bot.thread_auto_bot = None
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
        bot.terminate_threads.assert_called_once_with()

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

    def test_party_red_bar_mask_uses_screen_minimap_size(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.img_frame = np.full((200, 400, 3), 255, dtype=np.uint8)
        bot.loc_minimap = (10, 20)
        bot.img_minimap_screen = np.zeros((30, 60, 3), dtype=np.uint8)
        bot.img_minimap = np.zeros((142, 259, 3), dtype=np.uint8)
        bot.cfg = {
            "ui_coords": {"ui_y_start": 190},
            "party_red_bar": {
                "lower_red": (0, 100, 100),
                "upper_red": (0, 100, 100),
                "offset": (0, 0),
            },
        }

        with patch(
            "src.engine.MapleStoryAutoLevelUp.cv2.cvtColor",
            side_effect=RuntimeError("inspect frame"),
        ) as convert:
            with self.assertRaisesRegex(RuntimeError, "inspect frame"):
                bot.get_player_location_by_party_red_bar()

        masked_frame = convert.call_args.args[0]
        self.assertTrue(np.all(masked_frame[20:50, 10:70] == 0))
        self.assertTrue(np.all(masked_frame[60:100, 80:180] == 255))

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
    @staticmethod
    def make_frame_bot(
        frame,
        *,
        title_bar_height=59,
        window_title="MapleStory Worlds",
        capture_profile="auto",
    ):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.capture = SimpleNamespace(
            get_frame=Mock(return_value=frame),
            window_title=window_title,
        )
        bot.args = SimpleNamespace(test_image="")
        bot.cfg = {
            "bot": {"mode": "normal"},
            "game_window": {
                "capture_profile": capture_profile,
                "title_bar_height": title_bar_height,
                "size": (693, 1282),
                "potplayer_chrome_top": 34,
                "potplayer_chrome_bottom": 65,
                "potplayer_video_aspect_ratio": (16, 9),
            },
        }
        return bot

    def test_potplayer_chrome_and_black_bars_are_removed(self):
        # Reproduce the measured 2501x1440 PotPlayer geometry using distinct
        # colors for chrome, side bars, and the HDMI video itself.
        raw = np.full((1440, 2501, 3), 10, dtype=np.uint8)
        raw[34:1375, :] = 20
        raw[34:1375, 58:2442] = 200
        cfg = {
            "bot": {"mode": "normal"},
            "game_window": {
                "capture_profile": "potplayer",
                "size": (693, 1282),
                "potplayer_chrome_top": 34,
                "potplayer_chrome_bottom": 65,
                "potplayer_video_aspect_ratio": (16, 9),
            },
        }

        frame, geometry = preprocess_capture_frame(
            raw, cfg, window_title="TV/CAM/Device - PotPlayer"
        )

        self.assertEqual(frame.shape, (700, 1296, 3))
        self.assertTrue(np.all(frame == 200))
        self.assertEqual(geometry["video_roi"], (58, 34, 2442, 1375))

    def test_maximized_4k_potplayer_frame_is_normalized(self):
        # Live geometry measured from the current PotPlayer window. The
        # capture pipeline must remain independent of its outer dimensions.
        raw = np.full((2112, 3840, 3), 10, dtype=np.uint8)
        raw[34:2047, 130:3709] = 200
        cfg = {
            "bot": {"mode": "normal"},
            "game_window": {
                "capture_profile": "potplayer",
                "size": (693, 1282),
                "potplayer_chrome_top": 34,
                "potplayer_chrome_bottom": 65,
                "potplayer_video_aspect_ratio": (16, 9),
            },
        }

        frame, geometry = preprocess_capture_frame(
            raw, cfg, window_title="TV/CAM/Device - PotPlayer"
        )

        self.assertEqual(frame.shape, (700, 1296, 3))
        self.assertTrue(np.all(frame == 200))
        self.assertEqual(geometry["source_size"], (2112, 3840))
        self.assertEqual(geometry["video_roi"], (130, 34, 3709, 2047))

    @patch("src.utils.common.time.sleep")
    @patch("src.utils.common.win32gui")
    def test_resize_restores_maximized_window_and_verifies_size(
        self, win32gui, sleep
    ):
        win32gui.FindWindow.return_value = 42
        win32gui.GetWindowPlacement.return_value = (
            0,
            3,
            (0, 0),
            (0, 0),
            (0, 0, 3840, 2112),
        )
        win32gui.GetWindowRect.side_effect = [
            (0, 0, 3840, 2112),
            (0, 0, 2768, 1656),
        ]
        win32gui.MoveWindow.return_value = True

        actual_size = resize_window("PotPlayer", 2768, 1656)

        self.assertEqual(actual_size, (2768, 1656))
        win32gui.ShowWindow.assert_called_once()
        win32gui.MoveWindow.assert_called_once_with(
            42, 0, 0, 2768, 1656, True
        )
        sleep.assert_called_once()

    def test_potplayer_auto_profile_is_used_by_bot(self):
        raw = np.full((828, 1296, 3), 7, dtype=np.uint8)
        raw[34:763, :] = 180
        bot = self.make_frame_bot(
            raw,
            window_title="TV/CAM/Device - PotPlayer",
        )

        frame = bot.get_img_frame()

        self.assertEqual(frame.shape, (700, 1296, 3))
        self.assertTrue(np.all(frame == 180))
        self.assertEqual(bot.img_capture_content.shape, (729, 1296, 3))

    def test_legacy_game_size_is_still_accepted_after_title_crop(self):
        bot = self.make_frame_bot(np.zeros((752, 1282, 3), dtype=np.uint8))

        frame = bot.get_img_frame()

        self.assertEqual(frame.shape, (700, 1296, 3))

    def test_unsupported_capture_size_is_rejected(self):
        bot = self.make_frame_bot(np.zeros((759, 1200, 3), dtype=np.uint8))

        self.assertIsNone(bot.get_img_frame())

    def test_potplayer_and_direct_profiles_use_different_outer_sizes(self):
        cfg = {
            "capture_profile": "auto",
            "resize_width": 1296,
            "resize_height": 759,
            "potplayer_resize_width": 2768,
            "potplayer_resize_height": 1656,
        }

        self.assertEqual(
            get_capture_resize_size(cfg, "TV/CAM/Device - PotPlayer"),
            (2768, 1656),
        )
        self.assertEqual(
            get_capture_resize_size(cfg, "MapleStory Worlds"),
            (1296, 759),
        )

    def test_invalid_title_crop_is_rejected(self):
        bot = self.make_frame_bot(
            np.zeros((59, 1296, 3), dtype=np.uint8),
            title_bar_height=59,
        )

        self.assertIsNone(bot.get_img_frame())

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
