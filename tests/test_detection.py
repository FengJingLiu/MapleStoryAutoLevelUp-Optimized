import unittest

from src.utils.detection import (
    detection_center,
    detection_to_box,
    get_iou,
    intersection_area,
    nms,
)


class DetectionGeometryTests(unittest.TestCase):
    def test_height_width_order_is_preserved(self):
        detection = {
            "position": (10, 20),
            "size": (30, 80),
            "score": 0.1,
        }

        self.assertEqual(detection_to_box(detection), (10, 20, 90, 50))
        self.assertEqual(detection_center(detection), (50, 35))

    def test_intersection_area_handles_non_square_boxes(self):
        self.assertEqual(
            intersection_area((0, 0, 80, 30), (60, 10, 100, 50)),
            400,
        )

    def test_iou_returns_zero_for_disjoint_boxes(self):
        self.assertEqual(get_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)


class NonMaximumSuppressionTests(unittest.TestCase):
    def test_nms_keeps_lowest_sqdiff_score(self):
        worse = {
            "name": "mob",
            "position": (10, 10),
            "size": (20, 40),
            "score": 0.7,
        }
        better = {
            "name": "mob",
            "position": (12, 11),
            "size": (20, 40),
            "score": 0.1,
        }

        self.assertEqual(nms([worse, better], iou_threshold=0.4), [better])

    def test_nms_keeps_non_overlapping_detections(self):
        first = {
            "position": (0, 0),
            "size": (10, 20),
            "score": 0.2,
        }
        second = {
            "position": (100, 100),
            "size": (10, 20),
            "score": 0.3,
        }

        self.assertEqual(nms([second, first], iou_threshold=0.4), [first, second])


if __name__ == "__main__":
    unittest.main()
