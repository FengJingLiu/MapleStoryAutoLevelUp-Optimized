"""Crop one auto-relogin page template from a native DirectShow frame.

Crop boxes are deliberately supplied on the command line.  The old PotPlayer
boxes are not reused because every 3840x2160 login page must be calibrated
again from an actual GC573 frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


REFERENCE_SIZE = (2160, 3840)  # height, width
PAGES = ("disconnect", "connect", "world", "channel", "character")


def _read_color(path: Path) -> np.ndarray | None:
    """Read through an encoded buffer so Windows Unicode paths work."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _write_png(path: Path, image: np.ndarray) -> bool:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
    except OSError:
        return False
    return True


def extract_template(
    page: str,
    frame_path: Path,
    box: Sequence[int],
    output_dir: Path,
    *,
    expected_size: tuple[int, int] = REFERENCE_SIZE,
    overwrite: bool = False,
) -> Path:
    """Write a lossless page crop after validating its frame and bounds."""
    page = str(page).strip().lower()
    if page not in PAGES:
        raise ValueError(f"page must be one of: {', '.join(PAGES)}")
    if len(box) != 4:
        raise ValueError("box must be x0 y0 x1 y1")

    frame_path = Path(frame_path)
    frame = _read_color(frame_path)
    if frame is None:
        raise FileNotFoundError(f"Unable to read source frame: {frame_path}")
    if frame.shape[:2] != tuple(expected_size):
        raise ValueError(
            f"{frame_path} is {frame.shape[:2]}, expected {expected_size}"
        )

    x0, y0, x1, y1 = map(int, box)
    frame_h, frame_w = frame.shape[:2]
    if not (0 <= x0 < x1 <= frame_w and 0 <= y0 < y1 <= frame_h):
        raise ValueError(
            f"Crop {(x0, y0, x1, y1)} lies outside "
            f"{(frame_h, frame_w)}"
        )

    output_dir = Path(output_dir)
    output_path = output_dir / f"auto_relogin_{page}_cn.png"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --overwrite after QA"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    template = frame[y0:y1, x0:x1]
    if not _write_png(output_path, template):
        raise OSError(f"Unable to write template: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop one page template from a 3840x2160 DirectShow PNG"
    )
    parser.add_argument("--page", required=True, choices=PAGES)
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument(
        "--box",
        required=True,
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("misc"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(extract_template(
        args.page,
        args.frame,
        args.box,
        args.output_dir,
        overwrite=args.overwrite,
    ))


if __name__ == "__main__":
    main()
