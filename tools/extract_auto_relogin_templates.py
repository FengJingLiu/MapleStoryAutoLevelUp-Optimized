"""Extract deterministic login-flow templates from captured game frames.

The source screenshots are intentionally not committed.  This helper records
the exact crops used by ``auto_relogin`` so they can be regenerated when the
login UI or capture geometry changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


REFERENCE_SIZE = (2013, 3579)  # height, width

TEMPLATE_SPECS = {
    "disconnect": (
        "2026-08-15_11-58-57_img_frame.png",
        (1400, 740, 2075, 975),
    ),
    "connect": (
        "2026-08-15_11-59-40_img_frame.png",
        # Keep the static Connect button and its frame, excluding the dynamic
        # account row and explanatory text above/below it.
        (1870, 980, 2040, 1070),
    ),
    "world": (
        "2026-08-15_11-59-51_img_frame.png",
        (1985, 390, 2270, 470),
    ),
    "channel": (
        "2026-08-15_11-59-55_img_frame.png",
        (1350, 800, 1780, 950),
    ),
    "character": (
        "2026-08-15_11-59-59_img_frame.png",
        (2490, 650, 2790, 795),
    ),
}


def extract_templates(screenshot_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for page, (source_name, box) in TEMPLATE_SPECS.items():
        source_path = screenshot_dir / source_name
        frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"Unable to read source frame: {source_path}")
        if frame.shape[:2] != REFERENCE_SIZE:
            raise ValueError(
                f"{source_path} is {frame.shape[:2]}, expected {REFERENCE_SIZE}"
            )

        x0, y0, x1, y1 = box
        template = frame[y0:y1, x0:x1]
        if template.size == 0:
            raise ValueError(f"Empty crop for {page}: {box}")

        output_path = output_dir / f"auto_relogin_{page}_cn.png"
        if not cv2.imwrite(str(output_path), template):
            raise OSError(f"Unable to write template: {output_path}")
        written.append(output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screenshot-dir", type=Path, default=Path("screenshot"))
    parser.add_argument("--output-dir", type=Path, default=Path("misc"))
    args = parser.parse_args()
    for path in extract_templates(args.screenshot_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
