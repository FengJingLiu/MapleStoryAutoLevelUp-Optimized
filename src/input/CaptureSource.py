"""Select the configured frame source behind one small capture interface.

The automation pipeline consumes only the latest frame and its timestamp.  A
factory keeps that contract independent from whether frames come from a local
window or a DirectShow capture card.
"""

from __future__ import annotations

import platform
from typing import Any, Callable, Protocol, runtime_checkable


WINDOW_SOURCE = "window"
DIRECTSHOW_SOURCE = "directshow"
CAPTURE_CARD_SOURCE = "capture_card"


@runtime_checkable
class CaptureSource(Protocol):
    """Structural interface shared by all capture implementations."""

    capture_profile: str | None
    is_static_frame: bool
    last_frame_time: float
    window_title: str

    def get_frame_snapshot(self) -> tuple[Any | None, float | None]:
        """Return the latest frame and its monotonic capture timestamp."""

    def get_frame(self) -> Any | None:
        """Return the latest frame, or ``None`` when no fresh frame exists."""

    def stop(self) -> None:
        """Release the capture session and its worker resources."""


def resolve_capture_source(cfg: dict[str, Any], test_image_name: str | None = None) -> str:
    """Return the normalized source name, preserving legacy window defaults.

    Static test images are implemented by the existing window capturer and
    deliberately override ``capture.source``.  This keeps offline tests usable
    on machines without a capture card.
    """

    if test_image_name:
        return WINDOW_SOURCE

    capture_cfg = cfg.get("capture", {})
    if capture_cfg is None:
        capture_cfg = {}
    if not isinstance(capture_cfg, dict):
        raise ValueError("capture must be a mapping")

    source = str(capture_cfg.get("source", WINDOW_SOURCE)).strip().lower()
    if source == WINDOW_SOURCE:
        return WINDOW_SOURCE
    if source in {DIRECTSHOW_SOURCE, CAPTURE_CARD_SOURCE}:
        # DirectShow does not exist on macOS/Linux. Shared/custom configs may
        # still select the Windows default, so preserve those platforms' local
        # window capture path instead of failing during application startup.
        if platform.system() != "Windows":
            return WINDOW_SOURCE
        return DIRECTSHOW_SOURCE
    raise ValueError(
        "capture.source must be one of: window, directshow, capture_card"
    )


def _window_capture_class():
    if platform.system() == "Darwin":
        from src.input.GameWindowCapturorForMac import GameWindowCapturor
    else:
        from src.input.GameWindowCapturor import GameWindowCapturor
    return GameWindowCapturor


def _directshow_capture_class():
    if platform.system() != "Windows":
        raise RuntimeError("DirectShow capture is available only on Windows")
    from src.input.CaptureCardCapturor import CaptureCardCapturor
    return CaptureCardCapturor


def _set_window_capture_profile(
    capture: CaptureSource,
    cfg: dict[str, Any],
    *,
    is_static_frame: bool,
) -> CaptureSource:
    """Bind window captures to a non-card preprocessing profile.

    The DirectShow-ready default configuration intentionally selects the
    ``capture_card`` profile.  If a user changes only ``capture.source`` back
    to ``window``, that default must not make a desktop frame look like a raw
    4K card frame.  Static fixtures are always ordinary direct-window rasters.
    """

    from src.input.CaptureFramePreprocessor import (
        CAPTURE_CARD_PROFILE,
        DIRECT_PROFILE,
        resolve_capture_profile,
    )

    if is_static_frame:
        profile = DIRECT_PROFILE
    else:
        game_window_cfg = cfg.get("game_window", {})
        if not isinstance(game_window_cfg, dict):
            raise ValueError("game_window must be a mapping")
        window_title = getattr(capture, "window_title", "")
        if not isinstance(window_title, str):
            # Lightweight lifecycle tests and third-party adapters may expose
            # dynamically generated mock attributes instead of a title.
            window_title = ""
        profile = resolve_capture_profile(game_window_cfg, window_title)
        if profile == CAPTURE_CARD_PROFILE:
            # Treat a card-only value (including its DirectShow aliases) as
            # ``auto`` when the selected source is a desktop window.
            window_cfg = dict(game_window_cfg)
            window_cfg["capture_profile"] = "auto"
            profile = resolve_capture_profile(window_cfg, window_title)

    capture.capture_profile = profile
    return capture


def create_capture_source(
    cfg: dict[str, Any],
    test_image_name: str | None = None,
    *,
    window_capture_cls: Callable[..., CaptureSource] | None = None,
    directshow_capture_cls: Callable[..., CaptureSource] | None = None,
) -> CaptureSource:
    """Construct the configured capture implementation.

    ``capture.source`` is intentionally opt-in.  Missing configuration retains
    the historical window-capture behavior, while ``directshow`` and
    ``capture_card`` both select the GC573/OpenCV backend.
    """

    source = resolve_capture_source(cfg, test_image_name)
    if source == DIRECTSHOW_SOURCE:
        capture_class = directshow_capture_cls or _directshow_capture_class()
        return capture_class(cfg)

    window_capture = window_capture_cls or _window_capture_class()
    if test_image_name:
        # The Windows implementation owns the existing static-image fixture
        # path.  Preserve its constructor contract exactly.
        if platform.system() == "Darwin":
            raise RuntimeError("Static test-image capture is unavailable on macOS")
        capture = window_capture(cfg, test_image_name)
        return _set_window_capture_profile(
            capture,
            cfg,
            is_static_frame=True,
        )
    capture = window_capture(cfg)
    return _set_window_capture_profile(
        capture,
        cfg,
        is_static_frame=False,
    )


def capture_profile_override(capture: Any) -> str | None:
    """Return a real profile string without leaking attributes from mocks."""

    profile = getattr(capture, "capture_profile", None)
    if not isinstance(profile, str):
        return None
    profile = profile.strip()
    return profile or None


__all__ = [
    "CAPTURE_CARD_SOURCE",
    "CaptureSource",
    "DIRECTSHOW_SOURCE",
    "WINDOW_SOURCE",
    "capture_profile_override",
    "create_capture_source",
    "resolve_capture_source",
]
