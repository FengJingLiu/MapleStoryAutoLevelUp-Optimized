"""One-time dynamic minimap viewport discovery for WZ navigation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from src.utils.common import get_minimap_loc_size


@dataclass(frozen=True, slots=True)
class MinimapViewport:
    frame_size: tuple[int, int]
    rect: tuple[int, int, int, int]
    generation: int
    located_at: float

    def scaled_rect(self, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
        source_h, source_w = self.frame_size
        target_h, target_w = map(int, frame_size)
        x, y, width, height = self.rect
        x0 = int(round(x * target_w / source_w))
        y0 = int(round(y * target_h / source_h))
        x1 = int(round((x + width) * target_w / source_w))
        y1 = int(round((y + height) * target_h / source_h))
        x0 = min(max(0, x0), target_w)
        y0 = min(max(0, y0), target_h)
        x1 = min(max(x0, x1), target_w)
        y1 = min(max(y0, y1), target_h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("dynamic minimap viewport became empty")
        return x0, y0, x1 - x0, y1 - y0


class DynamicMinimapLocator:
    """Locate once, then return zero-copy views until explicitly invalidated."""

    def __init__(
        self,
        detector: Callable[[np.ndarray], tuple[int, int, int, int] | None]
        = get_minimap_loc_size,
        *,
        border_inset: int = 1,
    ):
        self.detector = detector
        self.border_inset = max(0, int(border_inset))
        self.viewport: MinimapViewport | None = None
        self._generation = 0

    def invalidate(self) -> None:
        self.viewport = None

    def acquire(
        self, frame: np.ndarray, *, force: bool = False
    ) -> MinimapViewport | None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        frame_size = tuple(map(int, frame.shape[:2]))
        if not force and self.viewport is not None and \
                self.viewport.frame_size == frame_size:
            return self.viewport

        detected = self.detector(frame)
        if detected is None:
            if force:
                self.viewport = None
            return None
        x, y, width, height = map(int, detected)
        inset = self.border_inset
        x += inset
        y += inset
        width -= inset * 2
        height -= inset * 2
        frame_h, frame_w = frame_size
        if width <= 0 or height <= 0 or x < 0 or y < 0 or \
                x + width > frame_w or y + height > frame_h:
            raise ValueError(
                "detected minimap viewport is outside the frame: "
                f"frame={frame_size}, rect={(x, y, width, height)}"
            )
        rect = (x, y, width, height)
        if self.viewport is not None and \
                self.viewport.frame_size == frame_size and \
                self.viewport.rect == rect:
            return self.viewport
        self._generation += 1
        self.viewport = MinimapViewport(
            frame_size=frame_size,
            rect=rect,
            generation=self._generation,
            located_at=time.monotonic(),
        )
        return self.viewport

    def crop(
        self, frame: np.ndarray, *, force: bool = False
    ) -> tuple[MinimapViewport, np.ndarray] | None:
        viewport = self.acquire(frame, force=force)
        if viewport is None:
            return None
        x, y, width, height = viewport.rect
        return viewport, frame[y:y + height, x:x + width]
