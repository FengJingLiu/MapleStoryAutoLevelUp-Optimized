import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.vision.YoloMonsterDetector import (
    YoloMonsterDetector,
    resolve_model_path,
)


class FakeYoloModel:
    names = {
        0: "character",
        1: "environment",
        2: "item",
        3: "mob",
        4: "npc",
        5: "ui",
    }

    def __init__(self, boxes=None):
        self.boxes = boxes or []
        self.predict_calls = []

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        boxes = SimpleNamespace(
            xyxy=np.asarray([item[0] for item in self.boxes], dtype=np.float32).reshape(-1, 4),
            conf=np.asarray([item[1] for item in self.boxes], dtype=np.float32),
            cls=np.asarray([item[2] for item in self.boxes], dtype=np.float32),
        )
        return [SimpleNamespace(boxes=boxes)]


class YoloMonsterDetectorTests(unittest.TestCase):
    def build_detector(self, boxes=None, **kwargs):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        model_path = Path(temp_dir.name) / "best.pt"
        model_path.write_bytes(b"test")
        model = FakeYoloModel(boxes)
        detector = YoloMonsterDetector(
            model_path,
            model=model,
            device="cpu",
            **kwargs,
        )
        return detector, model

    def test_detect_selects_mob_class_and_preserves_box_convention(self):
        detector, model = self.build_detector(
            boxes=[
                ((10.2, 20.4, 50.4, 80.6), 0.82, 3),
                ((2, 3, 8, 9), 0.95, 0),
            ],
            imgsz=1024,
            confidence=0.5,
            half=True,
        )

        detections = detector.detect(np.zeros((100, 120, 3), dtype=np.uint8))

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["name"], "mob")
        self.assertEqual(detections[0]["class_id"], 3)
        self.assertEqual(detections[0]["position"], (10, 20))
        self.assertEqual(detections[0]["size"], (61, 40))
        self.assertAlmostEqual(detections[0]["confidence"], 0.82, places=5)
        self.assertAlmostEqual(detections[0]["score"], 0.18, places=5)
        self.assertEqual(model.predict_calls[0]["classes"], [3])
        self.assertEqual(model.predict_calls[0]["imgsz"], 1024)
        self.assertEqual(model.predict_calls[0]["device"], "cpu")
        self.assertFalse(model.predict_calls[0]["half"])

    def test_detect_runs_full_frame_then_filters_boxes_touching_roi(self):
        detector, model = self.build_detector(
            boxes=[
                ((5, 5, 25, 25), 0.7, 3),
                ((50, 20, 80, 50), 0.8, 3),
                ((90, 60, 110, 90), 0.9, 3),
            ]
        )
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        detections = detector.detect(frame, roi=(30, 10, 90, 60))

        self.assertEqual([item["position"] for item in detections], [(50, 20)])
        self.assertIs(model.predict_calls[0]["source"], frame)
        self.assertEqual(model.predict_calls[0]["conf"], 0.4)

    def test_warmup_uses_runtime_shape_and_only_runs_once(self):
        detector, model = self.build_detector()

        detector.warmup(frame_size=(1296, 700))
        detector.warmup(frame_size=(1296, 700))

        self.assertEqual(len(model.predict_calls), 1)
        self.assertEqual(
            model.predict_calls[0]["source"].shape,
            (700, 1296, 3),
        )

    def test_missing_mob_class_is_rejected(self):
        detector_path = Path(tempfile.gettempdir()) / "missing-mob-class.pt"
        detector_path.write_bytes(b"test")
        self.addCleanup(detector_path.unlink, missing_ok=True)
        model = FakeYoloModel()
        model.names = {0: "character"}

        with self.assertRaisesRegex(ValueError, "mob"):
            YoloMonsterDetector(detector_path, model=model, device="cpu")

    def test_packaged_model_prefers_file_beside_executable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exe_dir = Path(temp_dir)
            model_path = exe_dir / "models" / "yolo" / "best.pt"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"test")
            with patch(
                "src.vision.YoloMonsterDetector.sys.frozen",
                True,
                create=True,
            ), patch(
                "src.vision.YoloMonsterDetector.sys.executable",
                str(exe_dir / "MapleStoryAutoLevelUp.exe"),
            ):
                resolved = resolve_model_path("models/yolo/best.pt")

        self.assertEqual(resolved, model_path.resolve())


class YoloMonsterEngineIntegrationTests(unittest.TestCase):
    @staticmethod
    def _make_pet_filter_bot(frame, pet_template):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "enable": True,
                "pet": {
                    "enable": True,
                    "filter_yolo_mob": True,
                    "yolo_name_diff_thres": 0.10,
                    "yolo_name_vertical_gap": 3,
                    "yolo_name_search_tolerance": [14, 8],
                    "yolo_name_max_gap": 12,
                },
            },
            "bot": {"mode": "normal"},
        }
        bot.img_frame = frame
        bot.img_frame_gray = frame[:, :, 0]
        bot.img_frame_debug = frame.copy()
        bot.img_nametag_pet = cv2.cvtColor(
            pet_template, cv2.COLOR_GRAY2BGR
        )
        bot.img_nametag_pet_gray = pet_template
        return bot

    def test_pet_name_directly_below_yolo_box_filters_that_mob(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        pet_name = np.full((10, 42), 180, dtype=np.uint8)
        pet_name[2:8, 5:37] = 240
        # Detection center x=90; centered pet name starts at x=69.
        frame[73:83, 69:111] = pet_name[:, :, None]
        bot = self._make_pet_filter_bot(frame, pet_name)
        pet_detection = {
            "name": "mob",
            "position": (70, 30),
            "size": (40, 40),
            "confidence": 0.8,
            "score": 0.2,
        }

        filtered = bot.filter_pet_yolo_detections([pet_detection])

        self.assertEqual(filtered, [])

    def test_same_pet_name_elsewhere_does_not_filter_real_mob(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        pet_name = np.full((10, 42), 180, dtype=np.uint8)
        pet_name[2:8, 5:37] = 240
        frame[90:100, 10:52] = pet_name[:, :, None]
        bot = self._make_pet_filter_bot(frame, pet_name)
        real_mob = {
            "name": "mob",
            "position": (70, 30),
            "size": (40, 40),
            "confidence": 0.8,
            "score": 0.2,
        }

        filtered = bot.filter_pet_yolo_detections([real_mob])

        self.assertEqual(filtered, [real_mob])

    def test_pet_filter_is_inactive_without_a_loaded_pet_name(self):
        frame = np.zeros((120, 180, 3), dtype=np.uint8)
        bot = self._make_pet_filter_bot(
            frame, np.full((10, 42), 180, dtype=np.uint8)
        )
        bot.img_nametag_pet = None
        mob = {
            "name": "mob",
            "position": (70, 30),
            "size": (40, 40),
            "confidence": 0.8,
            "score": 0.2,
        }

        self.assertEqual(bot.filter_pet_yolo_detections([mob]), [mob])

    def test_normal_detection_uses_full_camera_and_attack_roi(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "monster_detect": {"backend": "yolo"},
            "ui_coords": {"ui_y_start": 80},
            "bot": {"mode": "normal"},
        }
        bot.img_frame = np.zeros((100, 120, 3), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.yolo_monster_detector = Mock(
            detect=Mock(
                return_value=[
                    {
                        "name": "mob",
                        "position": (40, 20),
                        "size": (20, 30),
                        "confidence": 0.8,
                        "score": 0.2,
                    }
                ]
            )
        )

        detections = bot.get_monsters_in_range((30, 10), (90, 70))

        self.assertEqual(len(detections), 1)
        source = bot.yolo_monster_detector.detect.call_args.args[0]
        self.assertEqual(source.shape, (100, 120, 3))
        self.assertEqual(
            bot.yolo_monster_detector.detect.call_args.kwargs["roi"],
            (30, 10, 90, 70),
        )

    def test_yolo_box_size_filter_removes_too_narrow_or_short_boxes(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "monster_detect": {
                "min_box_width": 20,
                "min_box_height": 15,
            }
        }
        exact_threshold = {
            "name": "mob",
            "position": (0, 0),
            "size": (15, 20),
        }
        large = {
            "name": "mob",
            "position": (30, 0),
            "size": (30, 40),
        }
        too_narrow = {
            "name": "mob",
            "position": (60, 0),
            "size": (30, 19),
        }
        too_short = {
            "name": "mob",
            "position": (90, 0),
            "size": (14, 40),
        }

        filtered = bot.filter_yolo_detections_by_box_size([
            exact_threshold,
            large,
            too_narrow,
            too_short,
        ])

        self.assertEqual(filtered, [exact_threshold, large])

    def test_yolo_box_area_replaces_template_area_for_attack_overlap(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"attack": "directional"},
            "directional_attack": {"range_x": 100, "range_y": 100},
            "monster_detect": {"max_mob_area_trigger": 1500},
        }
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot.loc_player = (100, 50)
        bot.monsters_info = {}
        bot.monsters = [
            {
                "name": "mob",
                "position": (120, 30),
                "size": (40, 40),
                "confidence": 0.9,
                "score": 0.1,
            }
        ]

        self.assertIsNone(bot.get_nearest_monster(is_left=True))
        self.assertEqual(
            bot.get_nearest_monster(is_left=False)["position"],
            (120, 30),
        )

    def test_template_backend_keeps_smallest_template_overlap_threshold(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"attack": "directional"},
            "directional_attack": {"range_x": 100, "range_y": 100},
            "monster_detect": {
                "backend": "template",
                "max_mob_area_trigger": 1500,
            },
        }
        bot.img_frame = np.zeros((100, 220, 3), dtype=np.uint8)
        bot.loc_player = (100, 50)
        bot.monsters_info = {
            "small": [(np.zeros((10, 10, 3), dtype=np.uint8), None)],
            "large": [(np.zeros((50, 50, 3), dtype=np.uint8), None)],
        }
        bot.monsters = [
            {
                "name": "large",
                "position": (190, 30),
                "size": (40, 40),
                "score": 0.1,
            }
        ]

        self.assertEqual(
            bot.get_nearest_monster(is_left=False)["position"],
            (190, 30),
        )

    def test_yolo_load_does_not_require_map_templates(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.data = {"map_mobs_mapping": {}}
        bot.img_map = np.ones((2, 2, 3), dtype=np.uint8)
        bot.img_route = np.ones((2, 2, 3), dtype=np.uint8)
        bot.img_route_debug = np.ones((2, 2, 3), dtype=np.uint8)
        bot.img_routes = [np.ones((2, 2, 3), dtype=np.uint8)]
        bot.monsters_info = {"stale": []}
        bot.yolo_monster_detector = None
        cfg = {
            "bot": {"mode": "debug", "map": "unregistered_map"},
            "monster_detect": {
                "backend": "yolo",
                "model_path": "models/yolo/mob_1024_best.pt",
            },
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
        detector = Mock(
            config_signature=YoloMonsterDetector.signature_from_config(
                cfg["monster_detect"]
            ),
            model_path="model.pt",
            imgsz=1024,
            confidence=0.4,
            device="cpu",
            class_name="mob",
            warmup=Mock(return_value=0.0),
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.YoloMonsterDetector.from_config",
            return_value=detector,
        ) as load_detector, patch(
            "src.engine.MapleStoryAutoLevelUp.glob.glob"
        ) as glob_mock, patch(
            "src.engine.MapleStoryAutoLevelUp.load_image",
            return_value=np.zeros((2, 2, 3), dtype=np.uint8),
        ):
            self.assertEqual(bot.load_config(cfg), 0)

        load_detector.assert_called_once_with(cfg["monster_detect"])
        glob_mock.assert_not_called()
        self.assertEqual(bot.monsters_info, {})
        self.assertIsNone(bot.img_map)
        self.assertEqual(bot.img_routes, [])


if __name__ == "__main__":
    unittest.main()
