from pathlib import Path

import yaml


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


def test_unrecalibrated_template_features_are_disabled_by_default():
    default_cfg = load_yaml("config_default.yaml")
    cfg = default_cfg["auto_relogin"]
    height, width = cfg["flow_template_reference_size"]

    assert cfg["enable"] is False
    assert default_cfg["rune_solver"]["enable"] is False
    assert (height, width) == (2160, 3840)
    assert cfg["cursor_search_region"] == [0, 0, width, height]
    assert cfg["mouse_cursor_rescue_search_width"] == width

    for x0, y0, x1, y1 in cfg["page_search_regions"].values():
        assert 0 <= x0 < x1 <= width
        assert 0 <= y0 < y1 <= height
    for point in (
        list(cfg["page_anchor_points"].values())
        + [cfg["disconnect_confirm_point"]]
        + cfg["channel_points"]
    ):
        assert 0 <= point[0] < width
        assert 0 <= point[1] < height


def test_custom_capture_and_native_templates_use_4k_reference():
    cfg = load_yaml("config_custom.yaml")

    assert cfg["capture"]["source"] == "directshow"
    assert cfg["game_window"]["capture_profile"] == "capture_card"
    assert cfg["nametag"]["template_reference_size"] == [2160, 3840]
    assert cfg["nametag"]["medal"]["enable"] is False
    assert cfg["nametag"]["appearance"]["enable"] is False
    assert cfg["nametag"]["overhead_marker"]["enable"] is False
    assert cfg["nametag"]["pet"]["enable"] is False
