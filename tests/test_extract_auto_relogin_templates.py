from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from tools.extract_auto_relogin_templates import extract_template


def write_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if ok:
        encoded.tofile(path)
    return ok


def read_png(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


class ExtractAutoReloginTemplateTests(unittest.TestCase):
    def test_extracts_exact_crop_and_refuses_implicit_overwrite(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
            frame_path = root / "directshow.png"
            self.assertTrue(write_png(frame_path, frame))

            output = extract_template(
                "disconnect",
                frame_path,
                (2, 1, 7, 4),
                root / "misc",
                expected_size=(4, 8),
            )

            actual = read_png(output)
            self.assertTrue(np.array_equal(actual, frame[1:4, 2:7]))
            with self.assertRaises(FileExistsError):
                extract_template(
                    "disconnect",
                    frame_path,
                    (2, 1, 7, 4),
                    root / "misc",
                    expected_size=(4, 8),
                )

    def test_rejects_non_4k_source_by_default(self):
        with TemporaryDirectory() as temporary:
            frame_path = Path(temporary) / "legacy.png"
            self.assertTrue(write_png(
                frame_path, np.zeros((4, 8, 3), dtype=np.uint8)
            ))

            with self.assertRaisesRegex(ValueError, "expected .*2160.*3840"):
                extract_template(
                    "world",
                    frame_path,
                    (0, 0, 4, 4),
                    Path(temporary) / "misc",
                )


if __name__ == "__main__":
    unittest.main()
