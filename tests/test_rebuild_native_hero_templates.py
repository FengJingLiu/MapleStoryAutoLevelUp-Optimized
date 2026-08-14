import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.rebuild_native_hero_templates import (
    GREEN,
    MatchResult,
    TemplateSpec,
    extract_matched_template,
    run,
    template_mask,
    write_image,
)


def make_masked_template() -> np.ndarray:
    template = np.full((8, 10, 3), GREEN, dtype=np.uint8)
    template[2:6, 2:8] = (25, 25, 25)
    template[2:6, 3] = (235, 235, 235)
    template[3, 4:7] = (80, 150, 220)
    template[5, 5:8] = (210, 80, 35)
    return template


class NativeHeroTemplateTests(unittest.TestCase):
    def test_masked_extraction_fills_only_mask_exterior_with_green(self):
        old_template = make_masked_template()
        old_mask = template_mask(old_template)
        self.assertIsNotNone(old_mask)

        frame = np.full((40, 50, 3), (11, 22, 33), dtype=np.uint8)
        new_foreground = cv2.resize(
            old_template, (20, 16), interpolation=cv2.INTER_NEAREST
        )
        new_mask = cv2.resize(
            old_mask, (20, 16), interpolation=cv2.INTER_NEAREST
        )
        frame[9:25, 14:34][new_mask > 0] = new_foreground[new_mask > 0]
        match = MatchResult(
            spec=TemplateSpec("appearance", "hero.png", True),
            location=(14, 9),
            size=(20, 16),
            scale_x=2.0,
            scale_y=2.0,
            score=0.0,
            source_template=old_template,
            source_mask=old_mask,
        )

        extracted = extract_matched_template(frame, match)

        self.assertTrue(np.all(extracted[new_mask == 0] == GREEN))
        np.testing.assert_array_equal(
            extracted[new_mask > 0], new_foreground[new_mask > 0]
        )

    def test_dry_run_creates_preview_without_replacing_template(self):
        with tempfile.TemporaryDirectory() as temp_dir_value:
            temp_dir = Path(temp_dir_value)
            source_dir = temp_dir / "nametag"
            source_dir.mkdir()
            template_path = source_dir / "tester_appearance_stand_right.png"
            old_template = make_masked_template()
            write_image(template_path, old_template)
            original_bytes = template_path.read_bytes()

            frame = np.full((80, 100, 3), (70, 90, 110), dtype=np.uint8)
            enlarged = cv2.resize(
                old_template, (20, 16), interpolation=cv2.INTER_NEAREST
            )
            enlarged_mask = cv2.resize(
                template_mask(old_template),
                (20, 16),
                interpolation=cv2.INTER_NEAREST,
            )
            target = frame[31:47, 43:63]
            target[enlarged_mask > 0] = enlarged[enlarged_mask > 0]
            frame_path = temp_dir / "native.png"
            write_image(frame_path, frame)

            result = run([
                "--frame", str(frame_path),
                "--pose", "stand_right",
                "--name", "tester",
                "--source-dir", str(source_dir),
                "--dry-run",
                "--scale-min", "1.8",
                "--scale-max", "2.2",
                "--coarse-step", "0.1",
                "--refine-radius", "0.15",
                "--refine-step", "0.05",
                "--coarse-max-dimension", "200",
                "--max-score", "0.2",
            ])

            self.assertEqual(result, 0)
            self.assertEqual(template_path.read_bytes(), original_bytes)
            self.assertFalse((source_dir / "backups").exists())
            previews = list((source_dir / "previews").glob("*_preview.jpg"))
            self.assertEqual(len(previews), 1)
            self.assertGreater(previews[0].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
