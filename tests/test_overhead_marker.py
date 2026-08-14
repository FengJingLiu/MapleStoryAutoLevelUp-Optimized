import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.utils.common import get_mask


class OverheadMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project_root = Path(__file__).resolve().parents[1]
        cls.template = cv2.imread(str(
            project_root / "nametag" / "liu_muning_overhead_smile.png"
        ))
        if cls.template is None:
            raise RuntimeError("Missing overhead smile test template")

    def _make_bot(self, marker_locations=(), *, confirm_frames=1):
        frame = np.full((360, 640, 3), (45, 65, 85), dtype=np.uint8)
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "nametag": {
                "enable": True,
                "overhead_marker": {
                    "enable": True,
                    "diff_thres": 0.02,
                    "white_lower": (185, 185, 185),
                    "component_width": 1,
                    "component_height": 1,
                    "component_size_tolerance": 0.20,
                    "min_fill_rate": 0.60,
                    "max_fill_rate": 0.90,
                    "match_search_tolerance": 2,
                    "player_offset": (18, 71),
                    "local_search_radius": 90,
                    "global_confirm_frames": confirm_frames,
                    "lost_timeout_s": 2.0,
                    "require_unique_local": True,
                    "require_unique_global": True,
                },
            },
            "ui_coords": {"ui_y_start": frame.shape[0]},
        }
        bot.img_frame = frame
        self._set_marker_locations(bot, marker_locations)
        bot.img_frame_debug = None
        bot.img_overhead_marker = self.template.copy()
        bot.img_overhead_marker_gray = None
        bot.img_overhead_marker_mask = None
        bot.overhead_marker_component_bbox = None
        bot._update_overhead_marker_template_metadata()
        _, _, component_w, component_h = \
            bot.overhead_marker_component_bbox
        marker_cfg = bot.cfg["nametag"]["overhead_marker"]
        marker_cfg["component_width"] = component_w
        marker_cfg["component_height"] = component_h
        bot.loc_overhead_marker_player = (0, 0)
        bot.has_valid_overhead_marker_location = False
        bot.overhead_marker_miss_count = 0
        bot.pending_overhead_marker_location = None
        bot.pending_overhead_marker_count = 0
        bot.t_last_overhead_marker_detected = None
        bot.last_overhead_marker_match = None
        bot.screen_player_location_valid = False
        bot.loc_player = (0, 0)
        return bot

    def _set_marker_locations(self, bot, marker_locations):
        bot.img_frame[:] = (45, 65, 85)
        template_mask = get_mask(self.template, (0, 255, 0))
        template_h, template_w = self.template.shape[:2]
        for x, y in marker_locations:
            roi = bot.img_frame[y:y+template_h, x:x+template_w]
            roi[template_mask > 0] = self.template[template_mask > 0]
        bot.img_frame_gray = cv2.cvtColor(
            bot.img_frame, cv2.COLOR_BGR2GRAY
        )

    def _expected_player(self, bot, template_location):
        component_x, component_y, _, _ = \
            bot.overhead_marker_component_bbox
        offset_x, offset_y = bot.cfg["nametag"][
            "overhead_marker"
        ]["player_offset"]
        return (
            template_location[0] + component_x + offset_x,
            template_location[1] + component_y + offset_y,
        )

    def test_cold_global_marker_requires_consecutive_confirmation(self):
        marker_location = (220, 80)
        bot = self._make_bot((marker_location,), confirm_frames=2)
        expected = self._expected_player(bot, marker_location)

        self.assertIsNone(bot.get_player_location_by_overhead_marker())
        self.assertEqual(bot.pending_overhead_marker_location, expected)
        self.assertEqual(
            bot.get_player_location_by_overhead_marker(), expected
        )
        self.assertTrue(bot.has_valid_overhead_marker_location)

    def test_local_marker_is_accepted_immediately(self):
        marker_location = (220, 80)
        bot = self._make_bot((marker_location,), confirm_frames=2)
        expected = self._expected_player(bot, marker_location)

        self.assertEqual(
            bot.get_player_location_by_overhead_marker(
                expected_player=expected
            ),
            expected,
        )
        self.assertEqual(bot.last_overhead_marker_match["status"], "local")

    def test_multiple_global_smiles_are_rejected_without_history(self):
        bot = self._make_bot(((80, 70), (410, 150)), confirm_frames=1)

        self.assertIsNone(bot.get_player_location_by_overhead_marker())
        self.assertEqual(
            bot.last_overhead_marker_match["status"], "ambiguous"
        )

    def test_local_history_disambiguates_multiple_smiles(self):
        marker_locations = ((80, 70), (410, 150))
        bot = self._make_bot(marker_locations, confirm_frames=2)
        expected = self._expected_player(bot, marker_locations[1])

        self.assertEqual(
            bot.get_player_location_by_overhead_marker(
                expected_player=expected
            ),
            expected,
        )

    def test_two_nearby_local_smiles_are_rejected(self):
        marker_locations = ((220, 80), (350, 80))
        bot = self._make_bot(marker_locations, confirm_frames=1)
        bot.cfg["nametag"]["overhead_marker"]["local_search_radius"] = 160
        expected = self._expected_player(bot, marker_locations[0])

        self.assertIsNone(bot.get_player_location_by_overhead_marker(
            expected_player=expected
        ))
        self.assertEqual(
            bot.last_overhead_marker_match["status"], "ambiguous-local"
        )

    def test_global_confirmation_allows_large_position_change(self):
        first_location = (70, 80)
        second_location = (450, 190)
        bot = self._make_bot((first_location,), confirm_frames=2)

        self.assertIsNone(bot.get_player_location_by_overhead_marker())
        self._set_marker_locations(bot, (second_location,))
        self.assertEqual(
            bot.get_player_location_by_overhead_marker(),
            self._expected_player(bot, second_location),
        )

    def test_last_smile_position_expires_only_after_two_seconds(self):
        marker_location = (220, 80)
        bot = self._make_bot((marker_location,), confirm_frames=1)
        expected = self._expected_player(bot, marker_location)
        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            return_value=100.0,
        ):
            self.assertEqual(
                bot.get_player_location_by_overhead_marker(), expected
            )

        bot.img_frame[:] = (45, 65, 85)
        bot.img_frame_gray = cv2.cvtColor(
            bot.img_frame, cv2.COLOR_BGR2GRAY
        )
        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            return_value=102.0,
        ):
            self.assertEqual(
                bot.get_player_location_by_overhead_marker(), expected
            )
            self.assertTrue(bot.has_valid_overhead_marker_location)

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            return_value=102.001,
        ):
            self.assertIsNone(
                bot.get_player_location_by_overhead_marker()
            )
            self.assertFalse(bot.has_valid_overhead_marker_location)

    def test_enabled_smile_does_not_fall_back_to_nametag(self):
        bot = self._make_bot((), confirm_frames=1)
        bot.get_player_location_by_overhead_marker = Mock(return_value=None)
        bot.get_player_location_by_nametag = Mock(return_value=(123, 234))

        self.assertEqual(bot.get_player_location_on_screen(), (None, None))
        bot.get_player_location_by_nametag.assert_not_called()

    def test_enabled_smile_does_not_fall_back_to_party_bar(self):
        bot = self._make_bot((), confirm_frames=1)
        bot.cfg["nametag"]["enable"] = False
        bot.get_player_location_by_overhead_marker = Mock(return_value=None)
        bot.get_player_location_by_party_red_bar = Mock(
            return_value=((123, 234), (100, 200))
        )

        self.assertEqual(bot.get_player_location_on_screen(), (None, None))
        bot.get_player_location_by_party_red_bar.assert_not_called()

    def test_screen_locator_prefers_smile_over_nametag(self):
        bot = self._make_bot((), confirm_frames=1)
        bot.get_player_location_by_overhead_marker = Mock(
            return_value=(321, 210)
        )
        bot.get_player_location_by_nametag = Mock(return_value=(123, 234))

        self.assertEqual(
            bot.get_player_location_on_screen(), ((321, 210), None)
        )
        bot.get_player_location_by_nametag.assert_not_called()

if __name__ == "__main__":
    unittest.main()
