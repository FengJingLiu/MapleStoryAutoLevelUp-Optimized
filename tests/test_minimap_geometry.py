import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.utils.minimap_geometry import (
    MINIMAP_GEOMETRY_FILENAME,
    build_minimap_geometry,
    load_minimap_geometry,
    save_minimap_geometry,
    scale_minimap_rect,
)


class MinimapGeometryFileTests(unittest.TestCase):
    def test_round_trip_keeps_arbitrary_minimap_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = save_minimap_geometry(
                directory,
                (2013, 3579),
                (15, 189, 286, 340),
            )
            loaded = load_minimap_geometry(directory)

            self.assertEqual(loaded, saved)
            text = (
                Path(directory) / MINIMAP_GEOMETRY_FILENAME
            ).read_text(encoding="utf-8")
            self.assertIn("x=15", text)
            self.assertIn("width=286", text)
            self.assertIn("height=340", text)

    def test_rectangle_scales_by_saved_frame_geometry(self):
        geometry = build_minimap_geometry(
            (700, 1296),
            (10, 20, 200, 100),
        )
        self.assertEqual(
            scale_minimap_rect(geometry, (1400, 2592)),
            (20, 40, 400, 200),
        )


class MainMinimapGeometryTests(unittest.TestCase):
    def test_main_crops_saved_geometry_without_detection(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.minimap_geometry = build_minimap_geometry(
            (100, 200),
            (10, 20, 40, 30),
        )
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot.img_capture_content = None

        self.assertTrue(bot.apply_saved_minimap_geometry())
        self.assertEqual(bot.loc_minimap, (10, 20))
        self.assertEqual(bot.img_minimap_screen.shape, (30, 40, 3))
        self.assertIs(bot.img_minimap_source, bot.img_minimap_screen)

    def test_native_content_uses_same_saved_geometry_at_native_scale(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.minimap_geometry = build_minimap_geometry(
            (100, 200),
            (10, 20, 40, 30),
        )
        bot.img_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        bot.img_capture_content = np.zeros((200, 400, 3), dtype=np.uint8)

        self.assertTrue(bot.apply_saved_minimap_geometry())
        self.assertEqual(bot.img_minimap_screen.shape, (30, 40, 3))
        self.assertEqual(bot.img_minimap_source.shape, (60, 80, 3))


if __name__ == "__main__":
    unittest.main()
