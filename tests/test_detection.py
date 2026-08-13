import unittest

from src.utils.detection import (
    detection_center,
    detection_to_box,
    get_iou,
    intersection_area,
    nms,
    suppress_nearby_same_class,
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

    def test_nearby_same_class_animation_hits_keep_best_score(self):
        worse = {
            "name": "green_mushroom",
            "position": (15, 13),
            "size": (30, 18),
            "score": 0.28,
        }
        better = {
            "name": "green_mushroom",
            "position": (10, 10),
            "size": (20, 30),
            "score": 0.17,
        }

        self.assertEqual(
            suppress_nearby_same_class(
                [worse, better], center_distance=18
            ),
            [better],
        )

    def test_nearby_different_classes_are_preserved(self):
        mushroom = {
            "name": "green_mushroom",
            "position": (10, 10),
            "size": (20, 30),
            "score": 0.17,
        }
        slime = {
            "name": "slime",
            "position": (12, 11),
            "size": (20, 30),
            "score": 0.20,
        }

        self.assertEqual(
            suppress_nearby_same_class(
                [slime, mushroom], center_distance=18
            ),
            [mushroom, slime],
        )


if __name__ == "__main__":
    unittest.main()
