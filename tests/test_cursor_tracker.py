import unittest
from pathlib import Path

import cv2
import numpy as np

from src.vision.cursor_tracker import CursorTracker


ROOT = Path(__file__).resolve().parents[1]


def make_template():
    template = np.zeros((17, 15, 4), dtype=np.uint8)
    points = np.array([[3, 2], [11, 9], [8, 10], [11, 15], [7, 16], [4, 11], [1, 13]])
    cv2.fillPoly(template, [points], (235, 245, 250, 255))
    cv2.polylines(template, [points], True, (20, 30, 40, 255), 1)
    return template


def composite(frame, template, origin):
    x, y = origin
    height, width = frame.shape[:2]
    template_height, template_width = template.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + template_width), min(height, y + template_height)
    if x1 >= x2 or y1 >= y2:
        return
    tx1, ty1 = x1 - x, y1 - y
    tx2, ty2 = tx1 + x2 - x1, ty1 + y2 - y1
    source = template[ty1:ty2, tx1:tx2]
    alpha = source[:, :, 3] > 0
    target = frame[y1:y2, x1:x2]
    target[alpha] = source[:, :, :3][alpha]


class CursorTrackerSyntheticTests(unittest.TestCase):
    def setUp(self):
        self.template = make_template()
        self.tracker = CursorTracker(
            self.template,
            hotspot=(3, 2),
            min_score=0.99,
            uniqueness_margin=0.05,
            min_visible_fraction=0.25,
            min_visible_pixels=8,
        )

    def test_finds_unique_hotspot(self):
        frame = np.full((90, 120, 3), (70, 90, 110), dtype=np.uint8)
        composite(frame, self.template, (42, 31))

        match = self.tracker.locate(frame)

        self.assertIsNotNone(match)
        self.assertEqual(match.hotspot, (45, 33))
        self.assertEqual(match.template_origin, (42, 31))
        self.assertAlmostEqual(match.score, 1.0, places=5)

    def test_duplicate_matches_return_none(self):
        frame = np.full((90, 140, 3), (70, 90, 110), dtype=np.uint8)
        composite(frame, self.template, (20, 31))
        composite(frame, self.template, (95, 31))

        self.assertIsNone(self.tracker.locate(frame))

    def test_local_search_resolves_global_duplicate(self):
        frame = np.full((90, 140, 3), (70, 90, 110), dtype=np.uint8)
        composite(frame, self.template, (20, 31))
        composite(frame, self.template, (95, 31))

        match = self.tracker.locate(
            frame,
            previous_hotspot=(24, 34),
            local_radius=18,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.hotspot, (23, 33))

    def test_finds_cursor_partially_clipped_by_bottom_edge(self):
        frame = np.full((50, 100, 3), (70, 90, 110), dtype=np.uint8)
        composite(frame, self.template, (42, 40))

        match = self.tracker.locate(frame)

        self.assertIsNotNone(match)
        self.assertEqual(match.hotspot, (45, 42))
        self.assertLess(match.visible_fraction, 1.0)

    def test_insufficient_edge_evidence_returns_none(self):
        frame = np.full((50, 100, 3), (70, 90, 110), dtype=np.uint8)
        composite(frame, self.template, (42, 47))

        self.assertIsNone(self.tracker.locate(frame))

    def test_mask_erosion_removes_background_blended_edge_pixels(self):
        eroded = CursorTracker(
            self.template,
            hotspot=(3, 2),
            min_score=0.99,
            uniqueness_margin=0.05,
            min_visible_fraction=0.25,
            min_visible_pixels=8,
            mask_erode_pixels=1,
        )

        self.assertLess(eroded._mask_pixels, self.tracker._mask_pixels)
        self.assertGreater(eroded._mask_pixels, 0)

    def test_invalid_mask_erosion_is_rejected(self):
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CursorTracker(
                        self.template,
                        hotspot=(3, 2),
                        mask_erode_pixels=value,
                    )

    def test_full_sprite_cannot_hide_a_second_edge_clipped_cursor(self):
        frame = np.full((50, 140, 3), (70, 90, 110), dtype=np.uint8)
        composite(frame, self.template, (20, 20))
        composite(frame, self.template, (100, 40))

        # Both candidates are exact and spatially distinct, so selecting the
        # fully visible one would be an unsafe guess.
        self.assertIsNone(self.tracker.locate(frame))


class CursorTrackerRealImageTests(unittest.TestCase):
    def test_rgba_asset_matches_cursor_in_another_login_frame(self):
        template_path = ROOT / "misc" / "auto_relogin_cursor_cn.png"
        frame_path = ROOT / "screenshot" / "2026-08-15_11-58-57_img_frame.png"
        if not frame_path.exists():
            self.skipTest("local recorded login screenshot is not available")
        tracker = CursorTracker.from_file(
            template_path,
            hotspot=(13, 6),
            min_score=0.90,
            uniqueness_margin=0.02,
        )
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

        match = tracker.locate(
            frame,
            previous_hotspot=(2670, 1070),
            local_radius=140,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.hotspot, (2673, 1067))
        self.assertGreater(match.score, 0.97)

    def test_asset_is_transparent_and_not_from_red_annotation(self):
        template = cv2.imread(
            str(ROOT / "misc" / "auto_relogin_cursor_cn.png"),
            cv2.IMREAD_UNCHANGED,
        )
        self.assertEqual(template.shape[2], 4)
        alpha = template[:, :, 3] > 0
        self.assertGreater(np.count_nonzero(alpha), 3000)
        bgr = template[:, :, :3]
        annotation_red = (
            (bgr[:, :, 2] > 180)
            & (bgr[:, :, 2] > bgr[:, :, 1] * 1.5)
            & (bgr[:, :, 2] > bgr[:, :, 0] * 1.5)
            & alpha
        )
        self.assertEqual(np.count_nonzero(annotation_red), 0)


if __name__ == "__main__":
    unittest.main()
