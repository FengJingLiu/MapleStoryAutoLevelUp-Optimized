"""Persist and reuse the screen-space minimap rectangle for each route map."""

from __future__ import annotations

import os
from pathlib import Path

from src.utils.logger import logger


MINIMAP_GEOMETRY_FILENAME = "minimap_geometry.txt"
MINIMAP_GEOMETRY_VERSION = 1


def build_minimap_geometry(frame_size, minimap_rect):
    """Return validated minimap geometry using ``[h, w]`` and ``[x, y, w, h]``."""
    try:
        frame_h, frame_w = map(int, frame_size)
        x, y, width, height = map(int, minimap_rect)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid minimap geometry values") from exc

    if frame_h <= 0 or frame_w <= 0:
        raise ValueError(f"Invalid minimap frame size: {(frame_h, frame_w)}")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid minimap rectangle: {(x, y, width, height)}")
    if x + width > frame_w or y + height > frame_h:
        raise ValueError(
            "Minimap rectangle lies outside its recorded frame: "
            f"rect={(x, y, width, height)}, frame={(frame_h, frame_w)}"
        )

    return {
        "version": MINIMAP_GEOMETRY_VERSION,
        "frame_size": (frame_h, frame_w),
        "minimap_rect": (x, y, width, height),
    }


def minimap_geometry_path(map_dir):
    """Return the text metadata path inside one ``minimaps/<map>`` folder."""
    return Path(map_dir) / MINIMAP_GEOMETRY_FILENAME


def serialize_minimap_geometry(geometry):
    """Return the canonical text representation of validated geometry."""
    geometry = build_minimap_geometry(
        geometry["frame_size"],
        geometry["minimap_rect"],
    )
    frame_h, frame_w = geometry["frame_size"]
    x, y, width, height = geometry["minimap_rect"]
    return (
        "# MapleStoryAutoLevelUp minimap capture geometry\n"
        "# minimap_rect is the border-free interior: x, y, width, height\n"
        f"version={MINIMAP_GEOMETRY_VERSION}\n"
        f"frame_height={frame_h}\n"
        f"frame_width={frame_w}\n"
        f"x={x}\n"
        f"y={y}\n"
        f"width={width}\n"
        f"height={height}\n"
    )


def save_minimap_geometry(map_dir, frame_size, minimap_rect):
    """Save one human-readable minimap rectangle text file atomically."""
    geometry = build_minimap_geometry(frame_size, minimap_rect)
    path = minimap_geometry_path(map_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = serialize_minimap_geometry(geometry)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, path)
    logger.info(f"Saved minimap geometry: {path}")
    return geometry


def load_minimap_geometry(map_dir):
    """Load and validate a route map's minimap rectangle, or return ``None``."""
    path = minimap_geometry_path(map_dir)
    if not path.is_file():
        return None

    values = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError(f"Expected key=value line: {raw_line!r}")
            values[key.strip()] = int(value.strip())

        version = values.get("version")
        if version != MINIMAP_GEOMETRY_VERSION:
            raise ValueError(f"Unsupported minimap geometry version: {version}")
        geometry = build_minimap_geometry(
            (values["frame_height"], values["frame_width"]),
            (
                values["x"],
                values["y"],
                values["width"],
                values["height"],
            ),
        )
    except (KeyError, OSError, ValueError) as exc:
        logger.error(f"Invalid minimap geometry file {path}: {exc}")
        return None

    logger.info(
        "Loaded minimap geometry: "
        f"{path}, frame={geometry['frame_size']}, "
        f"rect={geometry['minimap_rect']}"
    )
    return geometry


def scale_minimap_rect(geometry, frame_size):
    """Scale saved rectangle edges to the current working-frame dimensions."""
    recorded_h, recorded_w = geometry["frame_size"]
    x, y, width, height = geometry["minimap_rect"]
    current_h, current_w = map(int, frame_size)
    if current_h <= 0 or current_w <= 0:
        raise ValueError(f"Invalid current frame size: {(current_h, current_w)}")

    scale_x = current_w / recorded_w
    scale_y = current_h / recorded_h
    x0 = int(round(x * scale_x))
    y0 = int(round(y * scale_y))
    x1 = int(round((x + width) * scale_x))
    y1 = int(round((y + height) * scale_y))
    x0 = min(max(0, x0), current_w)
    y0 = min(max(0, y0), current_h)
    x1 = min(max(x0, x1), current_w)
    y1 = min(max(y0, y1), current_h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "Saved minimap rectangle is empty at the current frame size: "
            f"frame={(current_h, current_w)}, rect={(x0, y0, x1, y1)}"
        )
    return x0, y0, x1 - x0, y1 - y0
