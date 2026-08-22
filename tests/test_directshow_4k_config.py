from pathlib import Path

import yaml

from src.input.KeyBoardController import capture_point_to_absolute_hid
from src.utils.common import override_cfg


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name):
    with (ROOT / "config" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_default_capture_source_is_gc573_directshow_rgb24_4k60():
    cfg = load_yaml("config_default.yaml")

    assert cfg["capture"]["source"] == "directshow"
    assert cfg["game_window"]["capture_profile"] == "capture_card"
    assert cfg["capture_card"] == {
        "device_index": 0,
        "device_name": "AVerMedia GC573 1 Capture",
        "width": 3840,
        "height": 2160,
        "fps": 60,
        "pixel_format": "RGB24",
        "frame_timeout": 1.0,
        "startup_timeout": 3.0,
        "shutdown_timeout": 2.0,
        "read_retry_interval": 0.01,
    }
    assert cfg["esp32_hid"]["absolute_desktop_rect"] == [0, 0, 3840, 2160]
    assert cfg["esp32_hid"]["capture_frame_is_desktop"] is False
    assert cfg["esp32_hid"]["magpie_source_rect"] == [1235, 710, 1366, 768]
    assert cfg["auto_relogin"]["remote_mouse_mode"] == "absolute"


def test_native_relogin_uses_ocr_only_by_default():
    default_cfg = load_yaml("config_default.yaml")
    cfg = default_cfg["auto_relogin"]
    height, width = cfg["flow_template_reference_size"]

    assert cfg["enable"] is True
    assert default_cfg["rune_solver"]["enable"] is False
    assert (height, width) == (2160, 3840)
    assert cfg["cursor_search_region"] == [0, 0, width, height]
    assert cfg["mouse_cursor_rescue_search_width"] == width
    assert "page_templates" not in cfg
    assert "page_search_regions" not in cfg

    targets = cfg["ocr"]["targets"]
    assert set(targets) == {
        "disconnect", "connect", "world", "channel", "queue", "character",
    }
    for target in targets.values():
        x0, y0, x1, y1 = target["search_region"]
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
        assert target["region_source"] == "configured"
    assert cfg["channel_click_count"] == 2
    assert cfg["connect_enter_retry_delay"] == 1.0
    assert cfg["connect_enter_max_attempts"] == 30
    assert targets["connect"]["action"] == "enter"
    assert "focus_switch_keys" not in cfg
    assert targets["world"]["match_mode"] == "partial"
    assert targets["channel"]["texts"] == ["漂漂猪"]
    assert targets["channel"]["match_mode"] == "exact"
    assert "channel_points" not in cfg


def test_default_uses_shared_hero_and_mob_yolo():
    cfg = load_yaml("config_default.yaml")
    marker = cfg["nametag"]["overhead_marker"]

    assert marker["enable"] is True
    assert marker["backend"] == "yolo"
    assert marker["yolo"]["class_name"] == "hero"
    assert marker["yolo"]["confidence"] == 0.85
    assert marker["max_stale_frames"] == -1
    assert cfg["monster_detect"]["model_path"].endswith(
        "yolov8n_1024_rect_hero_mob_level_ge10_all_pets_2860_best.pt"
    )
    assert cfg["monster_detect"]["preprocess_size"] == [768, 1366]
    assert cfg["monster_detect"]["imgsz"] == 1280
    assert cfg["monster_detect"]["confidence"] == 0.55
    assert cfg["monster_detect"]["iou"] == 0.70
    assert cfg["monster_detect"]["max_det"] == 100
    assert cfg["monster_detect"]["min_box_width"] == 20
    assert cfg["monster_detect"]["min_box_height"] == 20


def test_custom_capture_and_absolute_mouse_use_4k_reference():
    cfg = override_cfg(
        load_yaml("config_default.yaml"),
        load_yaml("config_custom.yaml"),
    )

    assert cfg["capture"]["source"] == "directshow"
    assert cfg["game_window"]["capture_profile"] == "capture_card"
    assert cfg["nametag"]["template_reference_size"] == [2160, 3840]
    assert cfg["nametag"]["medal"]["enable"] is False
    assert cfg["nametag"]["appearance"]["enable"] == "auto"
    assert cfg["nametag"]["overhead_marker"]["enable"] is True
    assert cfg["nametag"]["overhead_marker"]["backend"] == "yolo"
    assert cfg["monster_detect"]["model_path"] == \
        "models/yolo/yolov8n_1024_rect_hero_mob_level_ge10_all_pets_2860_best.pt"
    assert cfg["monster_detect"]["class_name"] == "mob"
    assert cfg["monster_detect"]["min_box_width"] == 40
    assert cfg["monster_detect"]["min_box_height"] == 40
    assert cfg["monster_detect"]["preprocess_size"] == [768, 1366]
    assert cfg["monster_detect"]["imgsz"] == 1280
    assert cfg["monster_detect"]["confidence"] == 0.7
    assert cfg["monster_detect"]["iou"] == 0.70
    assert cfg["monster_detect"]["max_det"] == 100
    assert cfg["nametag"]["pet"]["enable"] is True
    assert cfg["nametag"]["pet"]["yolo_ocr_text"] == "花蘑菇仔"
    assert cfg["esp32_hid"]["absolute_desktop_rect"] == [0, 0, 3840, 2160]
    # The capture card sees Magpie's scaled output, so points must map back to
    # the physical source client before Windows receives the absolute report.
    assert cfg["esp32_hid"]["capture_frame_is_desktop"] is False
    assert capture_point_to_absolute_hid(
        cfg, 2100, 1120, 3840, 2160
    ) == (16917, 16816)
    # Live Magpie calibration: the OCR center of 4.漂漂猪 at (2265, 462)
    # rendered its cursor hotspot at (2265, 463), within one capture pixel.
    assert capture_point_to_absolute_hid(
        cfg, 2265, 462, 3840, 2160
    ) == (17412, 13265)
    assert cfg["auto_relogin"]["enable"] is True
    assert cfg["auto_relogin"]["remote_mouse_mode"] == "absolute"


def test_only_rope_mount_has_a_configured_runup():
    cfg = load_yaml("config_default.yaml")

    assert "directional_jump_runup_ms" not in cfg["route"]
    assert cfg["route"]["rope_climb_runup_ms"] == 180
    assert cfg["wz_navigation"]["rope_mount_calibration"] == {
        "launch_window_px": [8, 14],
        "staging_offset_px": 15,
    }
