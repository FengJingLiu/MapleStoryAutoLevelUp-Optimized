import unittest

import numpy as np

from src.input.CaptureFramePreprocessor import (
    CAPTURE_CARD_PROFILE,
    preprocess_capture_frame,
    resolve_capture_profile,
)


def capture_card_config(width=3840, height=2160):
    return {
        "capture_card": {"width": width, "height": height},
        "game_window": {
            "capture_profile": "capture_card",
            "size": [693, 1282],
            "title_bar_height": 59,
        },
    }


class CaptureCardPreprocessorTests(unittest.TestCase):
    def test_capture_card_rgb24_frame_is_native_4k_passthrough(self):
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

        output, geometry = preprocess_capture_frame(
            frame,
            capture_card_config(),
        )

        self.assertIs(output, frame)
        self.assertEqual(output.shape, (2160, 3840, 3))
        self.assertEqual(geometry, {
            "profile": CAPTURE_CARD_PROFILE,
            "source_size": (2160, 3840),
            "video_roi": (0, 0, 3840, 2160),
            "native_size": (2160, 3840),
            "content_size": (2160, 3840),
            "output_size": (2160, 3840),
            "working_size": (2160, 3840),
            "normalized": False,
        })

    def test_capture_card_rejects_driver_fallback_resolution(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        with self.assertRaisesRegex(
            ValueError, "Unexpected capture-card frame size"
        ):
            preprocess_capture_frame(frame, capture_card_config())

    def test_directshow_profile_alias_and_capture_override_are_supported(self):
        cfg = capture_card_config(width=8, height=4)
        cfg["game_window"]["capture_profile"] = "direct"
        frame = np.zeros((4, 8, 3), dtype=np.uint8)

        output, geometry = preprocess_capture_frame(
            frame,
            cfg,
            capture_profile="directshow",
        )

        self.assertEqual(
            resolve_capture_profile({"capture_profile": "directshow"}),
            CAPTURE_CARD_PROFILE,
        )
        self.assertIs(output, frame)
        self.assertEqual(geometry["profile"], CAPTURE_CARD_PROFILE)


if __name__ == "__main__":
    unittest.main()
