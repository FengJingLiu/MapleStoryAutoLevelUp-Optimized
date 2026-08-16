"""Extract game content from window and capture-card frames."""

from __future__ import annotations

from typing import Any

import cv2

from src.utils.global_var import WINDOW_WORKING_SIZE


POTPLAYER_PROFILE = "potplayer"
DIRECT_PROFILE = "direct"
CAPTURE_CARD_PROFILE = "capture_card"


def _normalize_profile_name(value: Any) -> str:
    """Return the canonical preprocessing profile name."""
    profile = str(value).strip().lower()
    if profile in {"directshow", "direct_show", "capture-card"}:
        return CAPTURE_CARD_PROFILE
    return profile


def resolve_capture_profile(
    game_window_cfg: dict[str, Any], window_title: str = ""
) -> str:
    """Resolve ``auto`` to either the direct-window or PotPlayer pipeline."""
    configured = _normalize_profile_name(
        game_window_cfg.get("capture_profile", "auto")
    )
    if configured == "auto":
        # Capture implementations and lifecycle tests may expose a non-string
        # diagnostic attribute (for example ``Mock.window_title``).  Only a
        # real title can participate in the PotPlayer heuristic.
        title = window_title if isinstance(window_title, str) else ""
        if not title:
            configured_title = game_window_cfg.get("title", "")
            title = configured_title if isinstance(configured_title, str) else ""
        return POTPLAYER_PROFILE if "potplayer" in title.casefold() else DIRECT_PROFILE
    if configured not in {
        DIRECT_PROFILE,
        POTPLAYER_PROFILE,
        CAPTURE_CARD_PROFILE,
    }:
        raise ValueError(
            "game_window.capture_profile must be one of: "
            "auto, direct, potplayer, capture_card"
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
    capture_profile: str | None = None,
    skip_direct_size_check: bool = False,
):
    """Extract pure game content and return the configured working frame.

    PotPlayer frames first remove the configured skin chrome and centered black
    bars. By default they then pass through the legacy game-content raster to
    preserve the original coordinate/template geometry. When
    ``preserve_native_resolution`` is enabled, the cropped PotPlayer video is
    returned at its native resolution instead.
    """
    if frame is None or getattr(frame, "ndim", 0) < 2 or frame.size == 0:
        raise ValueError("Captured frame is empty")

    game_window_cfg = cfg["game_window"]
    if capture_profile is None:
        profile = resolve_capture_profile(game_window_cfg, window_title)
    else:
        profile = _normalize_profile_name(capture_profile)
        if profile not in {
            DIRECT_PROFILE,
            POTPLAYER_PROFILE,
            CAPTURE_CARD_PROFILE,
        }:
            raise ValueError(f"Unsupported capture profile: {capture_profile}")
    preserve_native_resolution = bool(
        game_window_cfg.get("preserve_native_resolution", False)
    )

    if profile == CAPTURE_CARD_PROFILE:
        capture_card_cfg = cfg.get("capture_card", {})
        expected_w = int(capture_card_cfg.get("width", 3840))
        expected_h = int(capture_card_cfg.get("height", 2160))
        if expected_w <= 0 or expected_h <= 0:
            raise ValueError(
                "capture_card.width and capture_card.height must be positive"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "DirectShow capture must provide a three-channel RGB24 frame; "
                f"received shape {frame.shape}"
            )
        if frame.shape[:2] != (expected_h, expected_w):
            raise ValueError(
                f"Unexpected capture-card frame size: {frame.shape[:2]} "
                f"(expected {(expected_h, expected_w)})"
            )
        content = frame
        native_content = frame
        roi = (0, 0, frame.shape[1], frame.shape[0])
    elif profile == POTPLAYER_PROFILE:
        expected_h, expected_w = _positive_pair(
            game_window_cfg["size"], "game_window.size"
        )
        native_content, roi = _potplayer_video_roi(frame, game_window_cfg)
        content = native_content
        if not preserve_native_resolution and content.shape[:2] != (
            expected_h,
            expected_w,
        ):
            downscaling = (
                content.shape[0] >= expected_h
                and content.shape[1] >= expected_w
            )
            interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_LINEAR
            content = cv2.resize(
                content,
                (expected_w, expected_h),
                interpolation=interpolation,
            )
    else:
        expected_h, expected_w = _positive_pair(
            game_window_cfg["size"], "game_window.size"
        )
        title_bar_height = int(game_window_cfg.get("title_bar_height", 0))
        if title_bar_height < 0 or title_bar_height >= frame.shape[0]:
            raise ValueError(
                f"Invalid title bar height: {title_bar_height} for captured "
                f"frame {frame.shape[:2]}"
            )
        content = frame[title_bar_height:, :]
        native_content = content
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

    normalized = profile != CAPTURE_CARD_PROFILE and not (
        profile == POTPLAYER_PROFILE and preserve_native_resolution
    )
    if normalized:
        output_frame = cv2.resize(
            content,
            WINDOW_WORKING_SIZE,
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        output_frame = content
    metadata = {
        "profile": profile,
        "source_size": tuple(frame.shape[:2]),
        "video_roi": roi,
        "native_size": tuple(native_content.shape[:2]),
        "content_size": tuple(content.shape[:2]),
        "output_size": tuple(output_frame.shape[:2]),
        "working_size": tuple(output_frame.shape[:2]),
        "normalized": normalized,
    }
    return output_frame, metadata
