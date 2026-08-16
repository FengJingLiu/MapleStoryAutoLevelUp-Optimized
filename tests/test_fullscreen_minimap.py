import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.utils.common import get_minimap_loc_size
from tools.routeRecorder import RouteRecorder, get_debug_preview_max_size


def make_bordered_frame(
        *,
        frame_size=(1200, 1600),
        map_top_left=(50, 210),
        map_bottom_right=(470, 390),
        expanded_panel=True,
    ):
    """Build a synthetic PotPlayer frame with an expanded minimap panel."""
    frame = np.zeros((*frame_size, 3), dtype=np.uint8)
    if expanded_panel:
        cv2.rectangle(frame, (35, 25), (490, 420), (235, 235, 235), 2)

    cv2.rectangle(
        frame,
        map_top_left,
        map_bottom_right,
        (235, 235, 235),
        2,
    )
    x0, y0 = map_top_left
    x1, y1 = map_bottom_right
    frame[y0 + 10:y1 - 10, x0 + 10:x1 - 10] = (35, 50, 45)
    return frame


class FullscreenMinimapTests(unittest.TestCase):
    def test_expanded_panel_selects_nested_map_raster(self):
        frame = make_bordered_frame()

        location = get_minimap_loc_size(frame)

        # Each two-pixel synthetic border is stripped by border-depth logic.
        # The result must be the lower nested map, not the outer panel whose
        # top starts at y=25 and includes title/icon content.
        self.assertEqual(location, (52, 212, 417, 177))

    def test_route_recorder_locks_median_minimap_rect(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.cfg = {"route_recoder": {"minimap_lock_frames": 5}}
        recorder.img_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        detections = [
            (30, 80, 220, 100),
            (31, 81, 220, 100),
            (30, 80, 221, 101),
            (90, 140, 280, 160),  # One bad border detection is rejected.
            (29, 79, 219, 99),
        ]

        with patch(
            "tools.routeRecorder.get_minimap_loc_size",
            side_effect=detections,
        ) as detect:
            for _ in range(4):
                self.assertFalse(recorder.update_minimap_from_current_frame())
            self.assertTrue(recorder.update_minimap_from_current_frame())
            self.assertEqual(recorder.loc_minimap, (31, 81))
            self.assertEqual(recorder.minimap_screen_size, (98, 218))

            # Once calibrated, even a later frame uses the same crop and does
            # not ask the noisy border detector for another rectangle.
            recorder.img_frame = np.ones((400, 600, 3), dtype=np.uint8)
            self.assertTrue(recorder.update_minimap_from_current_frame())
            self.assertEqual(recorder.loc_minimap, (31, 81))
            self.assertEqual(recorder.img_minimap_source.shape[:2], (98, 218))
            self.assertEqual(detect.call_count, 5)

    def test_route_recorder_pauses_if_frame_size_changes_after_lock(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.cfg = {"route_recoder": {"minimap_lock_frames": 1}}
        recorder.is_enable = True
        recorder.img_frame = np.zeros((400, 600, 3), dtype=np.uint8)

        with patch(
            "tools.routeRecorder.get_minimap_loc_size",
            return_value=(30, 80, 220, 100),
        ):
            self.assertTrue(recorder.update_minimap_from_current_frame())

        recorder.loc_player_global_last = (10, 20)
        recorder.img_frame = np.zeros((401, 600, 3), dtype=np.uint8)
        self.assertFalse(recorder.update_minimap_from_current_frame())
        self.assertFalse(recorder.is_enable)
        self.assertIsNone(recorder.loc_player_global_last)

    def test_route_recorder_reuses_preserved_minimap_geometry(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.cfg = {"route_recoder": {"minimap_lock_frames": 5}}
        recorder.is_enable = True
        recorder.img_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        recorder._saved_minimap_geometry = {
            "frame_size": (400, 600),
            "minimap_rect": (31, 81, 218, 98),
        }

        with patch("tools.routeRecorder.get_minimap_loc_size") as detect:
            self.assertTrue(recorder.update_minimap_from_current_frame())

        detect.assert_not_called()
        self.assertEqual(recorder.loc_minimap, (31, 81))
        self.assertEqual(recorder.img_minimap_source.shape[:2], (98, 218))

    def test_runtime_pixel_config_is_always_derived_from_unscaled_base(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        base_cfg = {
            "game_window": {"coordinate_reference_size": [700, 1296]},
            "ui_coords": {"ui_y_start": 610, "menu": [1140, 730]},
            "route_recoder": {
                "map_padding": 20,
                "local_search_radius": 35,
            },
            "minimap": {
                "offset": [3, 7],
                "debug_window_max_screen_ratio": 0.85,
            },
        }
        recorder._cfg_reference = deepcopy(base_cfg)
        recorder.cfg = deepcopy(base_cfg)
        recorder._runtime_output_size = None

        recorder.update_runtime_config((1400, 2592))

        self.assertEqual(recorder.cfg["ui_coords"]["ui_y_start"], 1220)
        self.assertEqual(recorder.cfg["ui_coords"]["menu"], [2280, 1460])
        # Route/minimap coordinates belong to their own rasters and must not
        # be multiplied with the game-frame x/y scale.
        self.assertEqual(recorder.cfg["route_recoder"]["map_padding"], 20)
        self.assertEqual(recorder.cfg["route_recoder"]["local_search_radius"], 35)
        self.assertEqual(recorder.cfg["minimap"]["offset"], [3, 7])

        # Returning to legacy output must not compound the previous 2x scale.
        recorder.update_runtime_config((700, 1296))
        self.assertEqual(recorder.cfg["ui_coords"]["ui_y_start"], 610)
        self.assertEqual(recorder.cfg["ui_coords"]["menu"], [1140, 730])
        self.assertEqual(recorder._cfg_reference, base_cfg)

    def test_capture_output_size_drives_native_runtime_config(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        base_cfg = {
            "game_window": {
                "capture_profile": "potplayer",
                "preserve_native_resolution": True,
                "coordinate_reference_size": [700, 1296],
                "size": [693, 1282],
                "potplayer_chrome_top": 34,
                "potplayer_chrome_bottom": 65,
                "potplayer_chrome_left": 0,
                "potplayer_chrome_right": 0,
                "potplayer_video_aspect_ratio": [16, 9],
            },
            "ui_coords": {"ui_y_start": 610},
            "minimap": {"offset": [3, 7]},
            "route_recoder": {"map_padding": 20},
        }
        recorder._cfg_reference = deepcopy(base_cfg)
        recorder.cfg = deepcopy(base_cfg)
        recorder._runtime_output_size = None
        recorder.capture = Mock(
            window_title="TV/CAM/device - PotPlayer",
        )
        recorder.capture.get_frame.return_value = np.zeros(
            (2112, 3840, 3),
            dtype=np.uint8,
        )

        frame = recorder.get_img_frame()

        self.assertEqual(frame.shape[:2], (2013, 3579))
        self.assertEqual(recorder._runtime_output_size, (2013, 3579))
        self.assertEqual(recorder.cfg["ui_coords"]["ui_y_start"], 1754)
        self.assertEqual(recorder.cfg["minimap"]["offset"], [3, 7])
        self.assertEqual(recorder.cfg["route_recoder"]["map_padding"], 20)
        self.assertEqual(recorder._cfg_reference, base_cfg)

    def test_native_debug_text_and_inset_scale_with_frame_and_stay_visible(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder._cfg_reference = {
            "game_window": {"coordinate_reference_size": [700, 1296]},
        }
        recorder.cfg = {"ui_coords": {"ui_y_start": 1754}}
        recorder.t_last_frame = 1.0
        recorder.kb = SimpleNamespace(t_func_key=[0.0, 0.0, 0.0, 0.0])
        recorder.is_enable = True
        recorder.img_frame_debug = np.zeros((2013, 3579, 3), dtype=np.uint8)
        recorder.loc_minimap = (15, 189)
        recorder.img_minimap_screen = np.zeros((184, 397, 3), dtype=np.uint8)
        recorder.img_minimap = np.zeros((92, 203, 3), dtype=np.uint8)
        recorder.img_route_debug = np.zeros((200, 240, 3), dtype=np.uint8)
        recorder.loc_player_global = (120, 100)

        with patch("tools.routeRecorder.cv2.putText") as put_text:
            recorder.update_info_on_img_frame_debug()

        shortcut_calls = [
            call for call in put_text.call_args_list
            if call.args[1].startswith(("FPS:", "Press "))
        ]
        self.assertEqual(len(shortcut_calls), 5)
        baselines = [call.args[2][1] for call in shortcut_calls]
        self.assertEqual(baselines, sorted(baselines))
        native_font_scale = shortcut_calls[0].args[4]
        native_thickness = shortcut_calls[0].args[6]
        _, baseline = cv2.getTextSize(
            "Ag",
            cv2.FONT_HERSHEY_SIMPLEX,
            native_font_scale,
            native_thickness,
        )
        self.assertGreater(native_font_scale, 1.8)
        self.assertLess(max(baselines) + baseline, 1754)

        # The enlarged route inset is bounded and entirely inside the camera
        # area; this checks the bottom/right paste boundary after scaling.
        self.assertTrue(np.any(recorder.img_frame_debug[:1754] != 0))

    def test_game_debug_preview_fits_monitor_without_dropping_bottom(self):
        recorder = RouteRecorder.__new__(RouteRecorder)
        recorder.cfg = {
            "ui_coords": {"ui_y_start": 1754},
            "minimap": {"debug_window_max_screen_ratio": 0.85},
        }
        recorder.img_frame_debug = np.zeros((2013, 3579, 3), dtype=np.uint8)
        recorder.img_frame_debug[1734:1754, :] = (0, 0, 255)
        recorder.t_last_frame = 0.0

        with (
            patch(
                "tools.routeRecorder.get_debug_monitor_work_size",
                return_value=(1920, 1080),
            ),
            patch("tools.routeRecorder.cv2.imshow") as imshow,
        ):
            recorder.update_img_frame_debug()

        preview = imshow.call_args.args[1]
        self.assertLessEqual(preview.shape[1], 1632)
        self.assertLessEqual(preview.shape[0], 918)
        self.assertTrue(np.any(preview[-12:, :, 2] > 0))

    def test_debug_preview_limit_is_optional_for_headless_tests(self):
        cfg = {"minimap": {"debug_window_max_screen_ratio": 0.85}}

        with patch(
            "tools.routeRecorder.get_debug_monitor_work_size",
            return_value=None,
        ):
            self.assertIsNone(get_debug_preview_max_size(cfg))


if __name__ == "__main__":
    unittest.main()
