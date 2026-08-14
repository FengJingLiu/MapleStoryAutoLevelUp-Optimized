import unittest

import numpy as np

from src.input.CaptureFramePreprocessor import preprocess_capture_frame


class FullscreenNativeCaptureTests(unittest.TestCase):
    def test_fullscreen_potplayer_preserves_native_video_resolution(self):
        raw = np.full((2112, 3840, 3), 10, dtype=np.uint8)
        raw[34:2047, 130:3709] = 200
        cfg = {
            "game_window": {
                "capture_profile": "potplayer",
                "size": (693, 1282),
                "potplayer_chrome_top": 34,
                "potplayer_chrome_bottom": 65,
                "potplayer_chrome_left": 0,
                "potplayer_chrome_right": 0,
                "potplayer_video_aspect_ratio": (16, 9),
                "preserve_native_resolution": True,
            }
        }

        frame, geometry = preprocess_capture_frame(
            raw,
            cfg,
            window_title="TV/CAM/Device - PotPlayer",
        )

        self.assertEqual(geometry["video_roi"], (130, 34, 3709, 2047))
        self.assertEqual(frame.shape, (2013, 3579, 3))
        self.assertEqual(geometry["native_size"], (2013, 3579))
        self.assertEqual(geometry["output_size"], (2013, 3579))
        self.assertEqual(geometry["working_size"], (2013, 3579))
        self.assertFalse(geometry["normalized"])
        self.assertTrue(np.all(frame == 200))

    def test_fullscreen_potplayer_keeps_legacy_normalization_by_default(self):
        raw = np.full((2112, 3840, 3), 10, dtype=np.uint8)
        raw[34:2047, 130:3709] = 200
        cfg = {
            "game_window": {
                "capture_profile": "potplayer",
                "size": (693, 1282),
                "potplayer_chrome_top": 34,
                "potplayer_chrome_bottom": 65,
                "potplayer_chrome_left": 0,
                "potplayer_chrome_right": 0,
                "potplayer_video_aspect_ratio": (16, 9),
            }
        }

        frame, geometry = preprocess_capture_frame(
            raw,
            cfg,
            window_title="TV/CAM/Device - PotPlayer",
        )

        self.assertEqual(frame.shape, (700, 1296, 3))
        self.assertEqual(geometry["native_size"], (2013, 3579))
        self.assertEqual(geometry["content_size"], (693, 1282))
        self.assertEqual(geometry["output_size"], (700, 1296))
        self.assertTrue(geometry["normalized"])
        self.assertTrue(np.all(frame == 200))


if __name__ == "__main__":
    unittest.main()
