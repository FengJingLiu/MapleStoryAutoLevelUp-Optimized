import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.navigation.wz_geometry import Point
from src.vision.YoloMonsterDetector import (
    YoloMonsterDetector,
    resolve_model_path,
)
from src.vision.auto_relogin_ocr import RapidOcrError


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

    def test_preprocess_resizes_for_inference_and_maps_boxes_to_source(self):
        detector, model = self.build_detector(
            boxes=[
                ((10, 10, 30, 40), 0.8, 3),
            ],
            preprocess_size=(50, 60),
        )
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        detections = detector.detect(
            frame,
            roi=(15, 15, 61, 81),
        )

        self.assertEqual(model.predict_calls[0]["source"].shape, (50, 60, 3))
        self.assertIsNot(model.predict_calls[0]["source"], frame)
        self.assertEqual(detections[0]["position"], (20, 20))
        self.assertEqual(detections[0]["size"], (60, 40))

    def test_preprocess_size_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "preprocess_size"):
            self.build_detector(preprocess_size=(768, 0))

    def test_warmup_uses_runtime_shape_and_only_runs_once(self):
        detector, model = self.build_detector()

        detector.warmup(frame_size=(1296, 700))
        detector.warmup(frame_size=(1296, 700))

        self.assertEqual(len(model.predict_calls), 1)
        self.assertEqual(
            model.predict_calls[0]["source"].shape,
            (700, 1296, 3),
        )

    def test_warmup_uses_configured_preprocess_shape(self):
        detector, model = self.build_detector(
            preprocess_size=(768, 1366),
        )

        detector.warmup(frame_size=(3840, 2160))

        self.assertEqual(
            model.predict_calls[0]["source"].shape,
            (768, 1366, 3),
        )

    def test_one_prediction_cache_serves_hero_and_mob_classes(self):
        detector, model = self.build_detector(
            boxes=[
                ((10, 20, 50, 80), 0.82, 3),
                ((60, 30, 100, 90), 0.91, 0),
            ]
        )
        model.names[0] = "hero"
        detector.names[0] = "hero"
        detector.class_ids["hero"] = 0
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        heroes = detector.detect(
            frame,
            confidence=0.85,
            class_name="hero",
            inference_class_names=("mob", "hero"),
            inference_confidence=0.4,
            cache_key=123,
        )
        mobs = detector.detect(
            frame,
            class_name="mob",
            inference_class_names=("mob", "hero"),
            inference_confidence=0.4,
            cache_key=123,
        )

        self.assertEqual(len(model.predict_calls), 1)
        self.assertEqual(model.predict_calls[0]["classes"], [0, 3])
        self.assertEqual(model.predict_calls[0]["conf"], 0.4)
        self.assertEqual(heroes[0]["name"], "hero")
        self.assertEqual(mobs[0]["name"], "mob")

    def test_class_threshold_filters_cached_lower_confidence_output(self):
        detector, model = self.build_detector(
            boxes=[
                ((10, 20, 50, 80), 0.82, 3),
                ((60, 30, 100, 90), 0.80, 0),
            ]
        )
        model.names[0] = "hero"
        detector.names[0] = "hero"
        detector.class_ids["hero"] = 0
        frame = np.zeros((100, 120, 3), dtype=np.uint8)

        heroes = detector.detect(
            frame,
            confidence=0.85,
            class_name="hero",
            inference_class_names=("mob", "hero"),
            inference_confidence=0.4,
            cache_key=456,
        )
        mobs = detector.detect(
            frame,
            confidence=0.4,
            class_name="mob",
            inference_class_names=("mob", "hero"),
            inference_confidence=0.4,
            cache_key=456,
        )

        self.assertEqual(heroes, [])
        self.assertEqual(len(mobs), 1)
        self.assertEqual(len(model.predict_calls), 1)

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
    def _make_yolo_hero_bot(detections):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "enable": False,
                "overhead_marker": {
                    "enable": True,
                    "backend": "yolo",
                    "local_search_radius": 90,
                    "global_confirm_frames": 2,
                    "global_confirm_radius": 24,
                    "max_stale_frames": 1,
                    "require_minimap_player": False,
                    "require_unique_global": True,
                    "yolo": {
                        "class_name": "hero",
                        "confidence": 0.4,
                        "player_anchor": [0.5, 0.5],
                        "player_offset": [0, 0],
                    },
                },
                "appearance": {"enable": False},
            },
            "ui_coords": {"ui_y_start": 90},
        }
        bot.img_frame = np.zeros((100, 160, 3), dtype=np.uint8)
        bot.img_frame_debug = None
        bot.yolo_hero_detector = Mock(detect=Mock(return_value=detections))
        bot.yolo_monster_detector = None
        bot.yolo_inference_class_names = ("hero",)
        bot._current_capture_frame_token = 1
        bot.has_valid_overhead_marker_location = False
        bot.loc_overhead_marker_player = (0, 0)
        bot.overhead_marker_miss_count = 0
        bot.pending_overhead_marker_location = None
        bot.pending_overhead_marker_count = 0
        bot.last_overhead_marker_match = None
        bot.nametag_appearance_templates = []
        bot.is_on_ladder = False
        return bot

    @staticmethod
    def _hero(position, confidence=0.8):
        return {
            "name": "hero",
            "class_id": 1,
            "position": position,
            "size": (40, 20),
            "confidence": confidence,
            "score": 1.0 - confidence,
        }

    def test_yolo_hero_rejects_ambiguous_cold_global_frame(self):
        bot = self._make_yolo_hero_bot([
            self._hero((20, 20), 0.9),
            self._hero((100, 20), 0.8),
        ])

        self.assertIsNone(bot.get_player_location_by_yolo())
        self.assertEqual(bot.last_overhead_marker_match["status"], "ambiguous")
        self.assertFalse(bot.has_valid_overhead_marker_location)

    def test_yolo_hero_does_not_infer_without_own_minimap_dot(self):
        bot = self._make_yolo_hero_bot([self._hero((40, 20), 0.9)])
        bot.cfg["nametag"]["overhead_marker"][
            "require_minimap_player"
        ] = True
        bot.cfg["minimap"] = {
            "player_color": [136, 255, 255],
            "player_color_tolerance": 10,
            "player_min_component_area": 2,
        }
        bot.img_minimap_source = np.zeros((30, 40, 3), dtype=np.uint8)

        self.assertIsNone(bot.get_player_location_by_yolo())
        bot.yolo_hero_detector.detect.assert_not_called()
        self.assertEqual(
            bot.last_overhead_marker_match["status"], "minimap-unavailable"
        )

    def test_yolo_hero_requires_stable_cold_confirmation(self):
        bot = self._make_yolo_hero_bot([self._hero((40, 20), 0.9)])

        self.assertIsNone(bot.get_player_location_by_yolo())
        self.assertEqual(bot.pending_overhead_marker_count, 1)
        self.assertEqual(bot.get_player_location_by_yolo(), (50, 40))
        self.assertTrue(bot.has_valid_overhead_marker_location)

    def test_yolo_hero_uses_previous_single_box_for_crowded_frame(self):
        single = self._hero((40, 20), 0.90)
        near_previous_single = self._hero((45, 23), 0.86)
        remote_higher_score = self._hero((120, 20), 0.99)
        bot = self._make_yolo_hero_bot([single])
        bot.yolo_hero_detector.detect.side_effect = [
            [single],
            [remote_higher_score, near_previous_single],
        ]

        self.assertIsNone(bot.get_player_location_by_yolo())
        bot._current_capture_frame_token = 2
        self.assertEqual(bot.get_player_location_by_yolo(), (55, 43))
        self.assertEqual(
            bot.last_overhead_marker_match["status"], "single-history"
        )

    def test_yolo_hero_tracks_nearest_box_instead_of_remote_higher_score(self):
        bot = self._make_yolo_hero_bot([
            self._hero((43, 22), 0.7),
            self._hero((120, 20), 0.99),
        ])
        bot.has_valid_overhead_marker_location = True
        bot.loc_overhead_marker_player = (50, 40)

        self.assertEqual(bot.get_player_location_by_yolo(), (53, 42))
        self.assertEqual(bot.last_overhead_marker_match["status"], "local")

    def test_wz_rope_uses_hero_anchored_climbing_template(self):
        bot = self._make_yolo_hero_bot([self._hero((40, 20), 0.9)])
        bot.cfg["nametag"]["overhead_marker"]["global_confirm_frames"] = 1
        bot._wz_navigation_enabled = True
        bot._rope_climb_active = True
        bot._update_ladder_state_from_smile_pose = Mock(return_value=True)

        self.assertEqual(bot.get_player_location_by_yolo(), (50, 40))
        bot._update_ladder_state_from_smile_pose.assert_called_once_with(
            (50, 40)
        )

    def test_wz_navigation_skips_climbing_template_outside_rope_action(self):
        bot = self._make_yolo_hero_bot([self._hero((40, 20), 0.9)])
        bot.cfg["nametag"]["overhead_marker"]["global_confirm_frames"] = 1
        bot._wz_navigation_enabled = True
        bot._rope_climb_active = False
        bot._update_ladder_state_from_smile_pose = Mock(return_value=False)

        self.assertEqual(bot.get_player_location_by_yolo(), (50, 40))
        bot._update_ladder_state_from_smile_pose.assert_not_called()

    def test_wz_rope_requests_grayscale_for_fresh_yolo_pose_frame(self):
        bot = self._make_yolo_hero_bot([])
        bot.cfg["nametag"]["appearance"]["enable"] = True
        bot._wz_navigation_enabled = True
        bot._rope_climb_active = True
        bot._current_vision_snapshot = SimpleNamespace(generation=7)
        bot._last_yolo_hero_snapshot_generation = 6

        self.assertTrue(bot._frame_grayscale_required())

        bot._rope_climb_active = False
        self.assertFalse(bot._frame_grayscale_required())

    def test_yolo_hero_unlimited_cache_survives_repeated_misses(self):
        hero = self._hero((40, 20), 0.9)
        bot = self._make_yolo_hero_bot([hero])
        bot.cfg["nametag"]["overhead_marker"]["max_stale_frames"] = -1
        bot.yolo_hero_detector.detect.side_effect = [
            [hero],
            [hero],
            *([[]] * 20),
        ]

        self.assertIsNone(bot.get_player_location_by_yolo())
        self.assertEqual(bot.get_player_location_by_yolo(), (50, 40))
        for _ in range(20):
            self.assertEqual(bot.get_player_location_by_yolo(), (50, 40))

        self.assertTrue(bot.has_valid_overhead_marker_location)
        self.assertEqual(bot.overhead_marker_miss_count, 20)
        self.assertEqual(
            bot.last_overhead_marker_match["status"], "not-found,cached"
        )

    def test_yolo_hero_reacquires_stable_far_single_without_cache_gap(self):
        original = self._hero((20, 20), 0.9)
        far_first = self._hero((120, 20), 0.91)
        far_second = self._hero((122, 21), 0.92)
        bot = self._make_yolo_hero_bot([original])
        bot.cfg["nametag"]["overhead_marker"]["max_stale_frames"] = -1
        bot.yolo_hero_detector.detect.side_effect = [
            [original],
            [original],
            [far_first],
            [far_second],
        ]

        self.assertIsNone(bot.get_player_location_by_yolo())
        self.assertEqual(bot.get_player_location_by_yolo(), (30, 40))
        bot._current_capture_frame_token = 2
        self.assertEqual(bot.get_player_location_by_yolo(), (30, 40))
        self.assertEqual(
            bot.last_overhead_marker_match["status"],
            "reacquire-pending,cached",
        )
        self.assertEqual(bot.pending_overhead_marker_count, 1)
        bot._current_capture_frame_token = 3
        self.assertEqual(bot.get_player_location_by_yolo(), (132, 41))
        self.assertEqual(
            bot.last_overhead_marker_match["status"], "reacquired"
        )

    def test_yolo_hero_does_not_reacquire_multiple_far_candidates(self):
        original = self._hero((20, 20), 0.9)
        far_left = self._hero((120, 20), 0.91)
        far_right = self._hero((140, 20), 0.92)
        bot = self._make_yolo_hero_bot([original])
        bot.cfg["nametag"]["overhead_marker"]["max_stale_frames"] = -1
        bot.yolo_hero_detector.detect.side_effect = [
            [original],
            [original],
            [far_left, far_right],
        ]

        self.assertIsNone(bot.get_player_location_by_yolo())
        self.assertEqual(bot.get_player_location_by_yolo(), (30, 40))
        bot._current_capture_frame_token = 2
        self.assertEqual(bot.get_player_location_by_yolo(), (30, 40))
        self.assertEqual(
            bot.last_overhead_marker_match["status"],
            "not-found-local,cached",
        )
        self.assertEqual(bot.pending_overhead_marker_count, 0)

    @staticmethod
    def _make_pet_filter_bot(frame):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "enable": False,
                "pet": {
                    "enable": True,
                    "filter_yolo_mob": True,
                    "yolo_ocr_text": "花蘑菇仔",
                    "yolo_ocr_min_score": 0.8,
                    "yolo_ocr_box_threshold": 0.2,
                    "yolo_ocr_max_box_size": [45, 40],
                    "yolo_ocr_max_hero_distance": [100, 60],
                },
            },
            "game_window": {"coordinate_reference_size": [700, 1296]},
        }
        bot.img_frame = frame
        bot.img_frame_debug = frame.copy()
        bot.loc_player = (100, 60)
        bot.screen_player_location_valid = True
        bot._pet_mob_ocr_locator = Mock()
        bot._last_pet_mob_ocr_error = None
        return bot

    @staticmethod
    def _small_nearby_mob():
        return {
            "name": "mob",
            "position": (115, 45),
            "size": (30, 35),
            "confidence": 0.8,
            "score": 0.2,
        }

    def test_exact_pet_ocr_below_small_nearby_box_filters_that_mob(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        bot = self._make_pet_filter_bot(frame)
        bot._pet_mob_ocr_locator.locate.return_value = SimpleNamespace(
            score=0.91,
            box=((105, 74), (165, 74), (165, 88), (105, 88)),
        )
        pet_detection = self._small_nearby_mob()

        filtered = bot.filter_pet_yolo_detections([pet_detection])

        self.assertEqual(filtered, [])
        args, kwargs = bot._pet_mob_ocr_locator.locate.call_args
        self.assertIs(args[0], frame)
        self.assertEqual(args[2], ("花蘑菇仔",))
        self.assertEqual(kwargs["match_mode"], "exact")
        self.assertEqual(kwargs["min_score"], 0.8)

    def test_pet_ocr_runs_only_for_small_boxes_near_hero(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        bot = self._make_pet_filter_bot(frame)
        bot._pet_mob_ocr_locator.locate.return_value = SimpleNamespace(
            score=0.91,
            box=((105, 74), (165, 74), (165, 88), (105, 88)),
        )
        nearby_small = self._small_nearby_mob()
        far_small = {**nearby_small, "position": (260, 45)}
        nearby_large = {**nearby_small, "size": (50, 55)}

        filtered = bot.filter_pet_yolo_detections([
            nearby_small,
            far_small,
            nearby_large,
        ])

        self.assertEqual(filtered, [far_small, nearby_large])
        self.assertEqual(bot._pet_mob_ocr_locator.locate.call_count, 1)

    def test_small_nearby_real_mob_is_kept_without_exact_pet_ocr(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        bot = self._make_pet_filter_bot(frame)
        bot._pet_mob_ocr_locator.locate.return_value = None
        real_mob = self._small_nearby_mob()

        self.assertEqual(
            bot.filter_pet_yolo_detections([real_mob]),
            [real_mob],
        )

    def test_pet_ocr_failure_keeps_small_nearby_mob(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        bot = self._make_pet_filter_bot(frame)
        bot._pet_mob_ocr_locator.locate.side_effect = RapidOcrError(
            "engine unavailable"
        )
        mob = self._small_nearby_mob()

        self.assertEqual(bot.filter_pet_yolo_detections([mob]), [mob])

    def test_pet_ocr_is_inactive_without_a_valid_hero_location(self):
        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        bot = self._make_pet_filter_bot(frame)
        bot.screen_player_location_valid = False
        mob = self._small_nearby_mob()

        self.assertEqual(bot.filter_pet_yolo_detections([mob]), [mob])
        bot._pet_mob_ocr_locator.locate.assert_not_called()

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

    def test_fresh_hero_box_filters_overlapping_cross_class_mob(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.yolo_monster_detector = Mock()
        bot.yolo_hero_detector = bot.yolo_monster_detector
        bot._current_capture_frame_token = 123
        bot.last_yolo_hero_detection = {
            "position": (100, 80),
            "size": (100, 80),
            "confidence": 0.92,
            "frame_token": 123,
        }
        bot.img_frame_debug = None
        duplicate_mob = {
            "name": "mob",
            "position": (95, 95),
            "size": (100, 90),
            "confidence": 0.70,
        }
        nearby_real_mob = {
            "name": "mob",
            "position": (170, 95),
            "size": (100, 90),
            "confidence": 0.90,
        }

        filtered = bot.filter_yolo_hero_class_conflicts([
            duplicate_mob,
            nearby_real_mob,
        ])

        self.assertEqual(filtered, [nearby_real_mob])

    def test_cached_hero_box_does_not_filter_current_mob(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.yolo_monster_detector = Mock()
        bot.yolo_hero_detector = bot.yolo_monster_detector
        bot._current_capture_frame_token = 124
        bot.last_yolo_hero_detection = {
            "position": (100, 80),
            "size": (100, 80),
            "confidence": 0.92,
            "frame_token": 123,
        }
        mob = {
            "name": "mob",
            "position": (100, 80),
            "size": (100, 80),
            "confidence": 0.90,
        }

        self.assertEqual(
            bot.filter_yolo_hero_class_conflicts([mob]),
            [mob],
        )

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

    def test_directional_targets_behind_wz_terrain_are_filtered(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"attack": "directional"},
            "directional_attack": {"range_x": 120, "range_y": 100},
            "monster_detect": {
                "backend": "yolo",
                "max_mob_area_trigger": 100,
            },
        }
        bot.img_frame = np.zeros((100, 240, 3), dtype=np.uint8)
        bot.loc_player = (100, 50)
        bot.monsters_info = {}
        blocked = {
            "name": "mob",
            "position": (130, 30),
            "size": (40, 40),
            "confidence": 0.9,
        }
        visible = {
            "name": "mob",
            "position": (180, 30),
            "size": (40, 40),
            "confidence": 0.9,
        }
        bot.monsters = [blocked, visible]
        bot._projectile_terrain_blocker = Mock(
            side_effect=lambda monster: (
                object() if monster is blocked else None
            )
        )

        self.assertEqual(
            bot.get_monsters_in_attack_range(is_left=False),
            [visible],
        )
        self.assertEqual(
            bot._projectile_terrain_blocker.call_args_list,
            [unittest.mock.call(blocked), unittest.mock.call(visible)],
        )

    def test_projectile_ray_converts_capture_pixels_with_wz_registration_scale(
            self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        wz_map = SimpleNamespace(
            first_horizontal_projectile_blocker=Mock(return_value=None)
        )
        bot.cfg = {
            "wz_navigation": {
                "projectile_terrain_check": True,
                "projectile_height_wz": 30,
                "projectile_clearance_wz": 8,
            }
        }
        bot.wz_navigation = SimpleNamespace(
            active=True,
            jump_active=False,
            projection=SimpleNamespace(
                navigation_to_world=Mock(return_value=Point(100, 200))
            ),
            registration=SimpleNamespace(scale_x=2.8, scale_y=2.7),
            wz_map=wz_map,
            motion_profile=SimpleNamespace(character_half_width_wz=15),
        )
        bot.is_on_ladder = False
        bot.loc_player_global = (50, 60)
        bot.loc_player = (1000, 500)
        bot.img_frame_debug = None
        monster = {
            "name": "mob",
            "position": (1270, 450),
            "size": (100, 100),
        }

        self.assertIsNone(bot._projectile_terrain_blocker(monster))

        args, kwargs = wz_map.first_horizontal_projectile_blocker.call_args
        self.assertEqual(args[0], Point(100, 170))
        self.assertAlmostEqual(args[1], 100 + 320 / 2.8)
        self.assertEqual(kwargs, {"clearance": 8.0, "origin_margin": 15.0})

    def test_forest_floor_p1_tree_walls_do_not_block_projectiles(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        decorative_tree = SimpleNamespace(
            kind="wall",
            geometry_id="wall:1:1:104",
            point=Point(388, 60),
        )
        real_wall = SimpleNamespace(
            kind="wall",
            geometry_id="wall:9:9:999",
            point=Point(450, 60),
        )
        wz_map = SimpleNamespace(
            first_horizontal_projectile_blocker=Mock(
                side_effect=(decorative_tree, real_wall)
            )
        )
        bot.cfg = {
            "wz_navigation": {
                "projectile_terrain_check": True,
                "projectile_height_wz": 30,
                "projectile_clearance_wz": 8,
            }
        }
        bot.wz_navigation = SimpleNamespace(
            map_id="100040110",
            active=True,
            jump_active=False,
            projection=SimpleNamespace(
                navigation_to_world=Mock(return_value=Point(453, 98))
            ),
            registration=SimpleNamespace(scale_x=2.8, scale_y=2.8),
            wz_map=wz_map,
            motion_profile=SimpleNamespace(character_half_width_wz=15),
        )
        bot.is_on_ladder = False
        bot.loc_player_global = (67, 333)
        bot.loc_player = (1000, 500)
        bot.img_frame_debug = None
        monster = {
            "name": "mob",
            "position": (700, 450),
            "size": (100, 100),
        }

        self.assertIsNone(bot._projectile_terrain_blocker(monster))
        self.assertIs(
            bot._projectile_terrain_blocker(monster),
            real_wall,
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
            preprocess_size=None,
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
