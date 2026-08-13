"""Normalize direct-game and PotPlayer capture frames for vision processing."""

from __future__ import annotations

from typing import Any

import cv2

from src.utils.global_var import WINDOW_WORKING_SIZE


POTPLAYER_PROFILE = "potplayer"
DIRECT_PROFILE = "direct"


def resolve_capture_profile(
    game_window_cfg: dict[str, Any], window_title: str = ""
) -> str:
    """Resolve ``auto`` to either the direct-window or PotPlayer pipeline."""
    configured = str(game_window_cfg.get("capture_profile", "auto")).strip().lower()
    if configured == "auto":
        title = window_title or str(game_window_cfg.get("title", ""))
        return POTPLAYER_PROFILE if "potplayer" in title.casefold() else DIRECT_PROFILE
    if configured not in {DIRECT_PROFILE, POTPLAYER_PROFILE}:
        raise ValueError(
            "game_window.capture_profile must be one of: auto, direct, potplayer"
        )
    return configured


def get_capture_resize_size(
    game_window_cfg: dict[str, Any], window_title: str = ""
) -> tuple[int, int]:
    """Return the outer window size appropriate for the selected capture source."""
    profile = resolve_capture_profile(game_window_cfg, window_title)
    if profile == POTPLAYER_PROFILE:
        width = int(game_window_cfg.get("potplayer_resize_width", 1296))
        height = int(game_window_cfg.get("potplayer_resize_height", 828))
    else:
        width = int(game_window_cfg.get("resize_width", 1296))
        height = int(game_window_cfg.get("resize_height", 759))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid capture resize dimensions: {(width, height)}")
    return width, height


def _positive_pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers")
    first, second = int(value[0]), int(value[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be greater than zero")
    return first, second


def _potplayer_video_roi(
    frame, game_window_cfg: dict[str, Any]
) -> tuple[Any, tuple[int, int, int, int]]:
    """Remove PotPlayer chrome and its centered letterbox/pillarbox area."""
    frame_h, frame_w = frame.shape[:2]
    top = int(game_window_cfg.get("potplayer_chrome_top", 34))
    bottom = int(game_window_cfg.get("potplayer_chrome_bottom", 65))
    left = int(game_window_cfg.get("potplayer_chrome_left", 0))
    right = int(game_window_cfg.get("potplayer_chrome_right", 0))
    if min(top, bottom, left, right) < 0:
        raise ValueError("PotPlayer chrome crop values cannot be negative")

    x0, x1 = left, frame_w - right
    y0, y1 = top, frame_h - bottom
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"PotPlayer chrome crop {(top, bottom, left, right)} is invalid "
            f"for captured frame {(frame_h, frame_w)}"
        )

    aspect_w, aspect_h = _positive_pair(
        game_window_cfg.get("potplayer_video_aspect_ratio", (16, 9)),
        "game_window.potplayer_video_aspect_ratio",
    )
    target_ratio = aspect_w / aspect_h
    available_w, available_h = x1 - x0, y1 - y0
    available_ratio = available_w / available_h

    if available_ratio > target_ratio:
        video_w = min(available_w, max(1, round(available_h * target_ratio)))
        x0 += (available_w - video_w) // 2
        x1 = x0 + video_w
    elif available_ratio < target_ratio:
        video_h = min(available_h, max(1, round(available_w / target_ratio)))
        y0 += (available_h - video_h) // 2
        y1 = y0 + video_h

    return frame[y0:y1, x0:x1], (x0, y0, x1, y1)


def preprocess_capture_frame(
    frame,
    cfg: dict[str, Any],
    *,
    window_title: str = "",
    skip_direct_size_check: bool = False,
):
    """Extract pure game content and return the project's standard working frame.

    PotPlayer frames first remove the configured skin chrome and centered black
    bars, then pass through the legacy game-content raster. This preserves the
    coordinate/template geometry used by the original direct-window pipeline.
    """
    if frame is None or getattr(frame, "ndim", 0) < 2 or frame.size == 0:
        raise ValueError("Captured frame is empty")

    game_window_cfg = cfg["game_window"]
    expected_h, expected_w = _positive_pair(
        game_window_cfg["size"], "game_window.size"
    )
    profile = resolve_capture_profile(game_window_cfg, window_title)

    if profile == POTPLAYER_PROFILE:
        content, roi = _potplayer_video_roi(frame, game_window_cfg)
        if content.shape[:2] != (expected_h, expected_w):
            downscaling = content.shape[0] >= expected_h and content.shape[1] >= expected_w
            interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_LINEAR
            content = cv2.resize(
                content,
                (expected_w, expected_h),
                interpolation=interpolation,
            )
    else:
        title_bar_height = int(game_window_cfg.get("title_bar_height", 0))
        if title_bar_height < 0 or title_bar_height >= frame.shape[0]:
            raise ValueError(
                f"Invalid title bar height: {title_bar_height} for captured "
                f"frame {frame.shape[:2]}"
            )
        content = frame[title_bar_height:, :]
        roi = (0, title_bar_height, frame.shape[1], frame.shape[0])

        if not skip_direct_size_check:
            if cfg.get("bot", {}).get("mode") in {"aux", "debug"}:
                tolerance = float(game_window_cfg.get("ratio_tolerance", 0.08))
                actual_ratio = content.shape[1] / content.shape[0]
                if abs(actual_ratio - 16 / 9) > tolerance:
                    raise ValueError(
                        f"Unexpected direct capture size: {content.shape[:2]} "
                        "(expected approximately 16:9 in aux/debug mode)"
                    )
            elif content.shape[:2] != (expected_h, expected_w):
                raise ValueError(
                    f"Unexpected direct capture size: {content.shape[:2]} "
                    f"(expected {(expected_h, expected_w)})"
                )

    working_frame = cv2.resize(
        content,
        WINDOW_WORKING_SIZE,
        interpolation=cv2.INTER_NEAREST,
    )
    metadata = {
        "profile": profile,
        "source_size": tuple(frame.shape[:2]),
        "video_roi": roi,
        "content_size": tuple(content.shape[:2]),
        "working_size": tuple(working_frame.shape[:2]),
    }
    return working_frame, metadata
