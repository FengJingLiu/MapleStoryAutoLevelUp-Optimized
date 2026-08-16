from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.utils.frame_geometry import scale_runtime_pixel_config


def _config():
    return {
        "game_window": {"coordinate_reference_size": [700, 1296]},
        "ui_coords": {
            "ui_y_start": 610,
            "menu": [1140, 730],
            "login_button_top_left": (838, 317),
            "login_button_bottom_right": [940, 373],
            "login_button_thres": 0.05,
        },
        "rune_warning_cn": {
            "top_left": [513, 137],
            "bottom_right": [768, 177],
            "diff_thres": 0.15,
        },
        "rune_enable_msg_eng": {
            "top_left": [460, 60],
            "bottom_right": [702, 83],
        },
        "rune_detect": {"box_width": 350, "box_height": 150},
        "rune_find": {
            "rune_trigger_distance_x": 20,
            "rune_trigger_distance_y": 200,
            "rune_trigger_cooldown": 0.3,
        },
        "rune_solver": {
            "arrow_box_coord": [355, 296],
            "arrow_box_interval": 170,
            "arrow_box_size": 80,
            "arrow_low_hsv": [340, 40, 40],
        },
        "aoe_skill": {"range_x": 400, "range_y": 170, "cooldown": 0.05},
        "directional_attack": {
            "range_x": 350,
            "range_y": 70,
            "cooldown": 0.9,
        },
        "directional_aoe": {
            "enable": True,
            "min_monsters": 3,
            "range_x": 420,
            "range_y": 110,
            "cooldown": 1.2,
            "attack_recovery_delay": 1.0,
        },
        "power_knockback": {
            "enable": True,
            "trigger_distance_x": 100,
            "range_y": 70,
            "cooldown": 0.9,
            "attack_recovery_delay": 0.95,
            "hp_bar_supplement": {
                "enable": True,
                "lower_hsv": [50, 120, 80],
                "upper_hsv": [75, 255, 255],
                "search_above_y": 90,
                "search_below_y": 10,
                "min_width": 6,
                "max_width": 30,
                "min_height": 1,
                "max_height": 4,
                "min_area": 10,
                "min_fill_rate": 0.75,
                "min_aspect_ratio": 3.0,
            },
        },
        "monster_detect": {
            "search_box_margin": 50,
            "max_mob_area_trigger": 1500,
            "min_box_width": 24,
            "min_box_height": 20,
            "confidence": 0.5,
        },
        "character": {"width": 100, "height": 150},
        "edge_teleport": {
            "trigger_box_width": 20,
            "trigger_box_height": 10,
            "color_code": [255, 127, 127],
        },
        "party_red_bar": {"offset": [20, 66], "lower_red": [0, 60, 60]},
        "nametag": {
            "offset": [-50, 30],
            "split_width": 30,
            "jump_confirm_distance": 40,
            "jump_confirm_radius": 12,
            "diff_thres": 0.15,
            "overhead_marker": {
                "player_offset": [18, 71],
                "component_width": 37,
                "component_height": 30,
                "match_search_tolerance": 2,
                "local_search_radius": 90,
                "diff_thres": 0.02,
            },
            "medal": {
                "id_fragment_width": 30,
                "id_fragment_stride": 15,
                "center_offset_x": 3,
                "vertical_gap": 4,
                "search_tolerance": [18, 6],
            },
            "pet": {
                "medal_offset": [37, 17],
                "medal_search_tolerance": [28, 10],
                "yolo_name_vertical_gap": 3,
                "yolo_name_search_tolerance": [14, 8],
                "yolo_name_max_gap": 12,
            },
            "appearance": {
                "local_search_radius": 90,
                "validation_distance": 30,
                "climb_validation_distance": 80,
                "global_confirm_radius": 12,
                "global_confirm_frames": 2,
                # Match load_yaml(), which normalizes YAML sequences to tuples.
                "templates": (
                    {
                        "suffix": "appearance_climb",
                        "pose": "climbing",
                        "player_offset": [20, 38],
                    },
                    {
                        "suffix": "appearance_stand_left",
                        "pose": "standing",
                        "player_offset": (18, 41),
                    },
                ),
            },
        },
        "route": {
            "search_range": 10,
            "jump_down_cooldown": 3.0,
            "jump_up_settle_delay": 0.6,
            "color_code": {"255,0,255": "none jump none"},
        },
        "route_recoder": {"local_search_radius": 35, "map_padding": 30},
        "minimap": {
            "offset": [0, 0],
            "player_min_component_area": 2,
            "reference_size_by_map": {"fire_land_1": [142, 259]},
        },
    }


def test_scales_full_frame_pixel_settings_without_mutating_source():
    cfg = _config()
    original = deepcopy(cfg)

    # x3 and y2 deliberately verify independent axis scaling.
    scaled = scale_runtime_pixel_config(cfg, [1400, 3888])

    assert cfg == original
    assert scaled is not cfg
    assert scaled["ui_coords"]["ui_y_start"] == 1220
    assert scaled["ui_coords"]["menu"] == [3420, 1460]
    assert scaled["ui_coords"]["login_button_top_left"] == (2514, 634)
    assert scaled["ui_coords"]["login_button_thres"] == 0.05

    assert scaled["rune_warning_cn"]["top_left"] == [1539, 274]
    assert scaled["rune_warning_cn"]["bottom_right"] == [2304, 354]
    assert scaled["rune_enable_msg_eng"]["top_left"] == [1380, 120]
    assert scaled["rune_detect"] == {"box_width": 1050, "box_height": 300}
    assert scaled["rune_find"]["rune_trigger_distance_x"] == 60
    assert scaled["rune_find"]["rune_trigger_distance_y"] == 400
    assert scaled["rune_find"]["rune_trigger_cooldown"] == 0.3
    assert scaled["rune_solver"]["arrow_box_coord"] == [1065, 592]
    assert scaled["rune_solver"]["arrow_box_interval"] == 510
    assert scaled["rune_solver"]["arrow_box_size"] == 240
    assert scaled["rune_solver"]["arrow_low_hsv"] == [340, 40, 40]

    assert scaled["aoe_skill"]["range_x"] == 1200
    assert scaled["aoe_skill"]["range_y"] == 340
    assert scaled["directional_attack"]["range_x"] == 1050
    assert scaled["directional_attack"]["range_y"] == 140
    assert scaled["directional_aoe"] == {
        "enable": True,
        "min_monsters": 3,
        "range_x": 1260,
        "range_y": 220,
        "cooldown": 1.2,
        "attack_recovery_delay": 1.0,
    }
    assert scaled["power_knockback"] == {
        "enable": True,
        "trigger_distance_x": 300,
        "range_y": 140,
        "cooldown": 0.9,
        "attack_recovery_delay": 0.95,
        "hp_bar_supplement": {
            "enable": True,
            "lower_hsv": [50, 120, 80],
            "upper_hsv": [75, 255, 255],
            "search_above_y": 180,
            "search_below_y": 20,
            "min_width": 18,
            "max_width": 90,
            "min_height": 2,
            "max_height": 8,
            "min_area": 60,
            "min_fill_rate": 0.75,
            "min_aspect_ratio": 3.0,
        },
    }
    assert scaled["monster_detect"]["search_box_margin"] == 150
    assert scaled["monster_detect"]["max_mob_area_trigger"] == 9000
    assert scaled["monster_detect"]["min_box_width"] == 72
    assert scaled["monster_detect"]["min_box_height"] == 40
    assert scaled["character"] == {"width": 300, "height": 300}
    assert scaled["edge_teleport"]["trigger_box_width"] == 60
    assert scaled["edge_teleport"]["trigger_box_height"] == 20
    assert scaled["edge_teleport"]["color_code"] == [255, 127, 127]
    assert scaled["party_red_bar"]["offset"] == [60, 132]
    assert scaled["party_red_bar"]["lower_red"] == [0, 60, 60]


def test_scales_nametag_anchors_and_template_offsets():
    scaled = scale_runtime_pixel_config(_config(), [1400, 3888])
    nametag = scaled["nametag"]

    assert nametag["offset"] == [-150, 60]
    assert nametag["split_width"] == 90
    # Isotropic distances/radii use max(x3, y2).
    assert nametag["jump_confirm_distance"] == 120
    assert nametag["jump_confirm_radius"] == 36
    assert nametag["overhead_marker"]["player_offset"] == [54, 142]
    assert nametag["overhead_marker"]["component_width"] == 111
    assert nametag["overhead_marker"]["component_height"] == 60
    assert nametag["overhead_marker"]["match_search_tolerance"] == 6
    assert nametag["overhead_marker"]["local_search_radius"] == 270
    assert nametag["overhead_marker"]["diff_thres"] == 0.02
    assert nametag["medal"]["id_fragment_width"] == 90
    assert nametag["medal"]["id_fragment_stride"] == 45
    assert nametag["medal"]["center_offset_x"] == 9
    assert nametag["medal"]["vertical_gap"] == 8
    assert nametag["medal"]["search_tolerance"] == [54, 12]
    assert nametag["pet"]["medal_offset"] == [111, 34]
    assert nametag["pet"]["medal_search_tolerance"] == [84, 20]
    assert nametag["pet"]["yolo_name_vertical_gap"] == 6
    assert nametag["pet"]["yolo_name_search_tolerance"] == [42, 16]
    assert nametag["pet"]["yolo_name_max_gap"] == 24
    assert nametag["appearance"]["local_search_radius"] == 270
    assert nametag["appearance"]["validation_distance"] == 90
    assert nametag["appearance"]["climb_validation_distance"] == 240
    assert nametag["appearance"]["global_confirm_radius"] == 36
    assert nametag["appearance"]["global_confirm_frames"] == 2
    assert nametag["appearance"]["templates"][0]["player_offset"] == [60, 76]
    assert nametag["appearance"]["templates"][1]["player_offset"] == (54, 82)


def test_keeps_minimap_route_and_route_jump_logic_in_map_time_coordinates():
    cfg = _config()
    scaled = scale_runtime_pixel_config(cfg, [1400, 3888])

    assert scaled["route"] == cfg["route"]
    assert scaled["route_recoder"] == cfg["route_recoder"]
    assert scaled["minimap"] == cfg["minimap"]


def test_keeps_absolute_mouse_rectangles_in_remote_desktop_coordinates():
    cfg = {
        "game_window": {"coordinate_reference_size": [700, 1296]},
        "esp32_hid": {
            "absolute_desktop_rect": [0, 0, 3840, 2160],
            "magpie_source_rect": [1235, 721, 1366, 768],
        },
    }

    scaled = scale_runtime_pixel_config(cfg, [1400, 3888])

    assert scaled["esp32_hid"] == cfg["esp32_hid"]


def test_uses_legacy_reference_by_default_and_accepts_partial_config():
    cfg = {"aoe_skill": {"range_x": 1296, "range_y": 700}}

    scaled = scale_runtime_pixel_config(cfg, (350, 648, 3))

    assert scaled == {"aoe_skill": {"range_x": 648, "range_y": 350}}


def test_rejects_invalid_inputs():
    cases = [
        ([], [700, 1296], TypeError),
        ({}, [0, 1296], ValueError),
        ({"game_window": {"coordinate_reference_size": [700, 0]}},
         [700, 1296], ValueError),
    ]
    for cfg, output_size, expected_exception in cases:
        try:
            scale_runtime_pixel_config(cfg, output_size)
        except expected_exception:
            continue
        raise AssertionError(
            f"expected {expected_exception.__name__} for "
            f"cfg={cfg!r}, output_size={output_size!r}"
        )


def _green_template(height=4, width=6):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (0, 255, 0)
    image[1:-1, 1:-1] = (30, 80, 150)
    return image


def test_bot_runtime_refresh_is_idempotent_and_syncs_native_templates():
    base_cfg = {
        "game_window": {"coordinate_reference_size": [700, 1296]},
        "ui_coords": {"ui_y_start": 610},
        "nametag": {
            "enable": True,
            "template_reference_size": [100, 200],
            "offset": [-50, 30],
            "appearance": {
                "templates": [{"player_offset": [20, 38]}],
            },
        },
    }
    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    bot.cfg = deepcopy(base_cfg)
    bot._base_cfg = deepcopy(base_cfg)
    bot._last_runtime_output_size = None
    bot._last_nametag_template_geometry = None
    bot._img_nametag_source = _green_template()
    bot._img_nametag_medal_source = _green_template(3, 7)
    bot._img_nametag_pet_source = _green_template(5, 4)
    bot.img_nametag = bot._img_nametag_source.copy()
    bot.img_nametag_medal = bot._img_nametag_medal_source.copy()
    bot.img_nametag_pet = bot._img_nametag_pet_source.copy()
    bot.nametag_appearance_templates = [{
        "image": _green_template(8, 5),
        "source_image": _green_template(8, 5),
        "config_index": 0,
        "player_offset": (20, 38),
    }]
    bot.capture = SimpleNamespace(cfg=None)
    bot.kb = SimpleNamespace(cfg=None)
    bot.health_monitor = SimpleNamespace(cfg=None)
    bot.rune_solver = SimpleNamespace(cfg=None)

    bot._refresh_runtime_frame_config((200, 600))

    runtime_cfg = bot.cfg
    assert runtime_cfg["ui_coords"]["ui_y_start"] == 174
    assert runtime_cfg["nametag"]["offset"] == [-23, 9]
    assert bot.img_nametag.shape == (8, 18, 3)
    assert bot.img_nametag_medal.shape == (6, 21, 3)
    assert bot.img_nametag_pet.shape == (10, 12, 3)
    appearance = bot.nametag_appearance_templates[0]
    assert appearance["image"].shape == (16, 15, 3)
    assert appearance["gray"].shape == (16, 15)
    assert appearance["mask"].shape == (16, 15)
    assert appearance["player_offset"] == (9, 11)
    for template in (
        bot.img_nametag,
        bot.img_nametag_medal,
        bot.img_nametag_pet,
        appearance["image"],
    ):
        assert np.all(template[0, 0] == (0, 255, 0))
        green = np.all(template == (0, 255, 0), axis=2)
        assert np.any(green)
        assert np.all(template[green] == (0, 255, 0))
    for component in (
        bot.capture, bot.kb, bot.health_monitor, bot.rune_solver
    ):
        assert component.cfg is runtime_cfg

    # Returning to the reference raster rebuilds from source/config baselines,
    # rather than applying an inverse scale to the prior runtime objects.
    bot._refresh_runtime_frame_config((100, 200))
    assert bot.img_nametag.shape == (4, 6, 3)
    assert np.array_equal(bot.img_nametag, bot._img_nametag_source)
    assert bot.cfg["ui_coords"]["ui_y_start"] == 87
    assert bot._base_cfg == base_cfg


def test_native_get_frame_does_not_keep_duplicate_minimap_content():
    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    bot.frame = None
    bot.capture = SimpleNamespace(
        get_frame=Mock(return_value=np.zeros((20, 30, 3), dtype=np.uint8)),
        window_title="TV/CAM/device - PotPlayer",
    )
    bot.args = SimpleNamespace(test_image="")
    bot.cfg = {
        "game_window": {"coordinate_reference_size": [700, 1296]},
        "ui_coords": {"ui_y_start": 610},
        "nametag": {"enable": False},
    }
    bot._base_cfg = deepcopy(bot.cfg)
    bot._last_runtime_output_size = None
    bot._last_nametag_template_geometry = None
    bot._last_capture_geometry = None
    bot._last_capture_error = None
    bot.img_capture_content = np.ones((2, 2, 3), dtype=np.uint8)
    output = np.zeros((10, 15, 3), dtype=np.uint8)
    geometry = {
        "profile": "potplayer",
        "source_size": (20, 30),
        "video_roi": (1, 2, 16, 12),
        "content_size": (10, 15),
        "working_size": (10, 15),
        "output_size": (10, 15),
        "normalized": False,
    }

    with patch(
        "src.engine.MapleStoryAutoLevelUp.preprocess_capture_frame",
        return_value=(output, geometry),
    ):
        result = bot.get_img_frame()

    assert result is output
    assert bot.img_capture_content is None
    assert bot._last_runtime_output_size == (10, 15)


def test_video_writer_uses_current_frame_width_and_height():
    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    bot._video_record_path = "video/native.mp4"
    bot._video_record_size = None
    bot.video_writer = None
    frame = np.zeros((2013, 3579, 3), dtype=np.uint8)
    writer = Mock()

    with patch(
        "src.engine.MapleStoryAutoLevelUp.cv2.VideoWriter_fourcc",
        return_value=123,
    ), patch(
        "src.engine.MapleStoryAutoLevelUp.cv2.VideoWriter",
        return_value=writer,
    ) as video_writer:
        bot._open_video_writer_for_frame(frame)

    video_writer.assert_called_once_with(
        "video/native.mp4", 123, 10, (3579, 2013)
    )
    assert bot.video_writer is writer
    assert bot._video_record_size == (3579, 2013)


def test_start_record_uses_unprocessed_capture_frame_without_enabling_viz():
    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    raw = np.zeros((2160, 3840, 3), dtype=np.uint8)
    bot.frame = raw
    bot.img_frame = np.ones((700, 1296, 3), dtype=np.uint8)
    bot.img_frame_debug = np.full((700, 1296, 3), 2, dtype=np.uint8)
    bot.enable_viz = Mock()
    bot._open_video_writer_for_frame = Mock()

    with patch("src.engine.MapleStoryAutoLevelUp.os.makedirs"):
        bot.start_record()

    bot.enable_viz.assert_not_called()
    bot._open_video_writer_for_frame.assert_called_once_with(raw)
    assert bot._video_record_path.endswith("_raw.mp4")


def test_raw_video_writer_receives_capture_frame_unchanged():
    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    raw = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    writer = Mock()
    bot._video_record_path = "video/raw.mp4"
    bot._video_record_size = (8, 4)
    bot.video_writer = writer

    assert bot._write_raw_video_frame(raw)

    writer.write.assert_called_once_with(raw)
