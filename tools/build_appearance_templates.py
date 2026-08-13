"""Build masked player-head templates from the supplied capture crops.

Usage:
    python -m tools.build_appearance_templates <climb.png> <stand-left.png> <stand-right.png>

The screenshots are UI-enlarged crops (about 2.1x the normalized game frame).
This script extracts the tiger-hood contour, keeps enclosed hair/face pixels,
and writes green-backed templates understood by the existing masked matcher.
"""

from pathlib import Path
import argparse

import cv2
import numpy as np


SPECS = (
    # Stop above the pet's green leaf; this template is deliberately hood-only.
    ("liu_muning_appearance_climb.png", (70, 20, 170, 82)),
    ("liu_muning_appearance_stand_left.png", (16, 18, 119, 126)),
    ("liu_muning_appearance_stand_right.png", (94, 26, 199, 137)),
)


def read_image(path):
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    encoded.tofile(path)


def extract_head(image, crop_rect, display_scale=2.1):
    x0, y0, x1, y1 = crop_rect
    crop = image[y0:y1, x0:x1].copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow_or_orange = cv2.inRange(
        hsv,
        np.array((4, 90, 65), dtype=np.uint8),
        np.array((42, 255, 255), dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        yellow_or_orange, 8
    )
    if count <= 1:
        raise ValueError("Tiger hood was not found in configured crop")

    center_x = crop.shape[1] / 2
    component = max(
        range(1, count),
        key=lambda index: (
            stats[index, cv2.CC_STAT_AREA]
            - 2 * abs(
                stats[index, cv2.CC_STAT_LEFT]
                + stats[index, cv2.CC_STAT_WIDTH] / 2
                - center_x
            )
        ),
    )
    hood = (labels == component).astype(np.uint8) * 255
    hood = cv2.morphologyEx(
        hood, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(
        hood, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(hood.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8))

    x, y, width, height = cv2.boundingRect(mask)
    padding = 2
    x = max(0, x - padding)
    y = max(0, y - padding)
    x1 = min(crop.shape[1], x + width + 2 * padding)
    y1 = min(crop.shape[0], y + height + 2 * padding)
    crop = crop[y:y1, x:x1]
    mask = mask[y:y1, x:x1]

    size = (
        max(1, round(crop.shape[1] / display_scale)),
        max(1, round(crop.shape[0] / display_scale)),
    )
    crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA) >= 128
    result = np.full_like(crop, (0, 255, 0))
    result[mask] = crop[mask]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("climb")
    parser.add_argument("stand_left")
    parser.add_argument("stand_right")
    parser.add_argument("--output", default="nametag")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    for source, (name, crop_rect) in zip(
        (args.climb, args.stand_left, args.stand_right), SPECS
    ):
        template = extract_head(read_image(source), crop_rect)
        destination = output / name
        write_image(destination, template)
        print(f"Saved {destination}: {template.shape[1]}x{template.shape[0]}")


if __name__ == "__main__":
    main()
