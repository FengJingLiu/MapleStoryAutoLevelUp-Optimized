"""Conservative template-based mouse cursor localization.

The tracker intentionally returns ``None`` when the best match is weak or is
not spatially unique.  Cursor motion can then be retried after a harmless
relative move instead of clicking at a guessed location.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class CursorMatch:
    """A reliable cursor-template match in frame coordinates."""

    hotspot: Point
    score: float
    uniqueness: float
    second_score: Optional[float]
    visible_fraction: float
    template_origin: Point


@dataclass(frozen=True)
class _Candidate:
    origin: Point
    score: float
    visible_fraction: float


class CursorTracker:
    """Find a cursor hotspot using an RGBA (or BGR) template.

    An RGBA template is preferred: transparent pixels are excluded from
    matching, so the background from which the cursor was sampled is not part
    of the template.  ``TM_SQDIFF_NORMED`` is used internally and exposed as a
    higher-is-better score in the range 0..1.
    """

    def __init__(
        self,
        template: np.ndarray,
        *,
        hotspot: Point,
        min_score: float = 0.90,
        uniqueness_margin: float = 0.02,
        min_visible_fraction: float = 0.25,
        min_visible_pixels: int = 32,
        mask_erode_pixels: int = 0,
        suppression_radius: Optional[int] = None,
    ) -> None:
        if template is None or template.size == 0:
            raise ValueError("cursor template must not be empty")
        if template.ndim == 2:
            color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            mask = np.full(template.shape, 255, dtype=np.uint8)
        elif template.ndim == 3 and template.shape[2] == 4:
            color = np.ascontiguousarray(template[:, :, :3])
            mask = np.ascontiguousarray(template[:, :, 3])
        elif template.ndim == 3 and template.shape[2] == 3:
            color = np.ascontiguousarray(template)
            mask = np.full(template.shape[:2], 255, dtype=np.uint8)
        else:
            raise ValueError("cursor template must be grayscale, BGR, or BGRA")

        height, width = color.shape[:2]
        hotspot_x, hotspot_y = (int(hotspot[0]), int(hotspot[1]))
        if not (0 <= hotspot_x < width and 0 <= hotspot_y < height):
            raise ValueError("cursor hotspot must lie inside the template")
        if cv2.countNonZero(mask) == 0:
            raise ValueError("cursor template alpha mask must not be empty")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be in the range 0..1")
        if not 0.0 <= uniqueness_margin <= 1.0:
            raise ValueError("uniqueness_margin must be in the range 0..1")
        if not 0.0 < min_visible_fraction <= 1.0:
            raise ValueError("min_visible_fraction must be in the range (0, 1]")
        if min_visible_pixels <= 0:
            raise ValueError("min_visible_pixels must be positive")
        if isinstance(mask_erode_pixels, bool) or not isinstance(
                mask_erode_pixels, int) or mask_erode_pixels < 0:
            raise ValueError("mask_erode_pixels must be a non-negative integer")

        self._template = color
        binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        if mask_erode_pixels:
            binary_mask = cv2.erode(
                binary_mask,
                np.ones((3, 3), dtype=np.uint8),
                iterations=mask_erode_pixels,
            )
        if cv2.countNonZero(binary_mask) == 0:
            raise ValueError("cursor mask erosion removed every template pixel")
        self._mask = binary_mask
        self.hotspot = (hotspot_x, hotspot_y)
        self.min_score = float(min_score)
        self.uniqueness_margin = float(uniqueness_margin)
        self.min_visible_fraction = float(min_visible_fraction)
        self.min_visible_pixels = int(min_visible_pixels)
        self.mask_erode_pixels = mask_erode_pixels
        self._mask_pixels = int(cv2.countNonZero(self._mask))
        self.suppression_radius = int(
            suppression_radius
            if suppression_radius is not None
            else max(3, min(width, height) // 4)
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        hotspot: Point,
        **kwargs,
    ) -> "CursorTracker":
        """Load a cursor template without losing its alpha channel."""

        template = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if template is None:
            raise ValueError(f"unable to read cursor template: {path}")
        return cls(template, hotspot=hotspot, **kwargs)

    def locate(
        self,
        frame: np.ndarray,
        *,
        previous_hotspot: Optional[Point] = None,
        local_radius: Optional[int | Sequence[int]] = None,
        search_region: Optional[Box] = None,
    ) -> Optional[CursorMatch]:
        """Return a unique cursor match, or ``None`` rather than guessing.

        ``previous_hotspot`` and ``local_radius`` optionally constrain the
        candidate hotspot to a local rectangle.  ``search_region`` uses the
        half-open frame-coordinate form ``(x1, y1, x2, y2)``.  If both are
        supplied their intersection is searched.
        """

        image = self._as_bgr(frame)
        frame_height, frame_width = image.shape[:2]
        if frame_height == 0 or frame_width == 0:
            return None

        region = self._normalize_region(
            frame_width,
            frame_height,
            search_region,
            previous_hotspot,
            local_radius,
        )
        if region is None:
            return None

        # Compare complete and edge-clipped candidates in one global ranking.
        # Otherwise a cursor-like full sprite in the UI could hide the real
        # pointer while it is clipped at a screen edge.
        full_candidates = self._match_fully_visible(image, region)
        edge_candidates = self._match_clipped_edges(image, region)
        return self._select_reliable(full_candidates + edge_candidates)

    @staticmethod
    def _as_bgr(frame: np.ndarray) -> np.ndarray:
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("frame must be a numpy image")
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return np.ascontiguousarray(frame)
        raise ValueError("frame must be grayscale, BGR, or BGRA")

    def _normalize_region(
        self,
        width: int,
        height: int,
        search_region: Optional[Box],
        previous_hotspot: Optional[Point],
        local_radius: Optional[int | Sequence[int]],
    ) -> Optional[Box]:
        if search_region is None:
            x1, y1, x2, y2 = 0, 0, width, height
        else:
            x1, y1, x2, y2 = (int(value) for value in search_region)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)

        if previous_hotspot is not None and local_radius is not None:
            if isinstance(local_radius, Sequence) and not isinstance(
                local_radius, (str, bytes)
            ):
                if len(local_radius) != 2:
                    raise ValueError("local_radius sequence must contain x and y")
                radius_x, radius_y = int(local_radius[0]), int(local_radius[1])
            else:
                radius_x = radius_y = int(local_radius)
            if radius_x < 0 or radius_y < 0:
                raise ValueError("local_radius must not be negative")
            center_x, center_y = int(previous_hotspot[0]), int(previous_hotspot[1])
            x1, x2 = max(x1, center_x - radius_x), min(
                x2, center_x + radius_x + 1
            )
            y1, y2 = max(y1, center_y - radius_y), min(
                y2, center_y + radius_y + 1
            )

        return None if x1 >= x2 or y1 >= y2 else (x1, y1, x2, y2)

    def _match_fully_visible(
        self, image: np.ndarray, region: Box
    ) -> list[_Candidate]:
        frame_height, frame_width = image.shape[:2]
        template_height, template_width = self._template.shape[:2]
        hotspot_x, hotspot_y = self.hotspot
        x1, y1, x2, y2 = region

        origin_x1 = max(0, x1 - hotspot_x)
        origin_y1 = max(0, y1 - hotspot_y)
        origin_x2 = min(frame_width - template_width, x2 - 1 - hotspot_x)
        origin_y2 = min(frame_height - template_height, y2 - 1 - hotspot_y)
        if origin_x1 > origin_x2 or origin_y1 > origin_y2:
            return []

        source = image[
            origin_y1 : origin_y2 + template_height,
            origin_x1 : origin_x2 + template_width,
        ]
        distances = cv2.matchTemplate(
            source,
            self._template,
            cv2.TM_SQDIFF_NORMED,
            mask=self._mask,
        )
        return self._candidates_from_distances(
            distances,
            offset=(origin_x1, origin_y1),
            visible_fraction=1.0,
            limit=6,
        )

    def _match_clipped_edges(
        self, image: np.ndarray, region: Box
    ) -> list[_Candidate]:
        """Match templates clipped by exactly one frame edge.

        Corner-clipped cursors are deliberately not inferred: their visible
        evidence is usually too small for a safe click.  A harmless relative
        move can bring such a cursor back into view before retrying.
        """

        frame_height, frame_width = image.shape[:2]
        template_height, template_width = self._template.shape[:2]
        hotspot_x, hotspot_y = self.hotspot
        x1, y1, x2, y2 = region
        candidates: list[_Candidate] = []

        interior_x1 = max(0, x1 - hotspot_x)
        interior_x2 = min(frame_width - template_width, x2 - 1 - hotspot_x)
        if interior_x1 <= interior_x2:
            vertical_origins = list(
                range(max(-hotspot_y, y1 - hotspot_y), min(-1, y2 - 1 - hotspot_y) + 1)
            )
            vertical_origins += list(
                range(
                    max(frame_height - template_height + 1, y1 - hotspot_y),
                    min(frame_height - 1 - hotspot_y, y2 - 1 - hotspot_y) + 1,
                )
            )
            for origin_y in vertical_origins:
                template_y1 = max(0, -origin_y)
                template_y2 = min(template_height, frame_height - origin_y)
                if not self._visible_enough(
                    self._mask[template_y1:template_y2, :]
                ):
                    continue
                source_y1 = origin_y + template_y1
                source_y2 = origin_y + template_y2
                distances = cv2.matchTemplate(
                    image[
                        source_y1:source_y2,
                        interior_x1 : interior_x2 + template_width,
                    ],
                    self._template[template_y1:template_y2, :],
                    cv2.TM_SQDIFF_NORMED,
                    mask=self._mask[template_y1:template_y2, :],
                )
                visible_fraction = self._visible_fraction(
                    self._mask[template_y1:template_y2, :]
                )
                candidates.extend(
                    self._candidates_from_distances(
                        distances,
                        offset=(interior_x1, origin_y),
                        visible_fraction=visible_fraction,
                        limit=3,
                    )
                )

        interior_y1 = max(0, y1 - hotspot_y)
        interior_y2 = min(frame_height - template_height, y2 - 1 - hotspot_y)
        if interior_y1 <= interior_y2:
            horizontal_origins = list(
                range(max(-hotspot_x, x1 - hotspot_x), min(-1, x2 - 1 - hotspot_x) + 1)
            )
            horizontal_origins += list(
                range(
                    max(frame_width - template_width + 1, x1 - hotspot_x),
                    min(frame_width - 1 - hotspot_x, x2 - 1 - hotspot_x) + 1,
                )
            )
            for origin_x in horizontal_origins:
                template_x1 = max(0, -origin_x)
                template_x2 = min(template_width, frame_width - origin_x)
                if not self._visible_enough(
                    self._mask[:, template_x1:template_x2]
                ):
                    continue
                source_x1 = origin_x + template_x1
                source_x2 = origin_x + template_x2
                distances = cv2.matchTemplate(
                    image[
                        interior_y1 : interior_y2 + template_height,
                        source_x1:source_x2,
                    ],
                    self._template[:, template_x1:template_x2],
                    cv2.TM_SQDIFF_NORMED,
                    mask=self._mask[:, template_x1:template_x2],
                )
                visible_fraction = self._visible_fraction(
                    self._mask[:, template_x1:template_x2]
                )
                candidates.extend(
                    self._candidates_from_distances(
                        distances,
                        offset=(origin_x, interior_y1),
                        visible_fraction=visible_fraction,
                        limit=3,
                    )
                )

        return candidates

    def _visible_fraction(self, mask: np.ndarray) -> float:
        return cv2.countNonZero(mask) / self._mask_pixels

    def _visible_enough(self, mask: np.ndarray) -> bool:
        visible_pixels = int(cv2.countNonZero(mask))
        return (
            visible_pixels >= self.min_visible_pixels
            and visible_pixels / self._mask_pixels >= self.min_visible_fraction
        )

    def _candidates_from_distances(
        self,
        distances: np.ndarray,
        *,
        offset: Point,
        visible_fraction: float,
        limit: int,
    ) -> list[_Candidate]:
        values = np.asarray(distances, dtype=np.float32).copy()
        values[~np.isfinite(values)] = np.inf
        candidates: list[_Candidate] = []
        for _ in range(limit):
            min_value, _, min_location, _ = cv2.minMaxLoc(values)
            if not np.isfinite(min_value):
                break
            origin_x = offset[0] + min_location[0]
            origin_y = offset[1] + min_location[1]
            candidates.append(
                _Candidate(
                    origin=(origin_x, origin_y),
                    score=float(np.clip(1.0 - min_value, 0.0, 1.0)),
                    visible_fraction=float(visible_fraction),
                )
            )
            radius = self.suppression_radius
            x1 = max(0, min_location[0] - radius)
            x2 = min(values.shape[1], min_location[0] + radius + 1)
            y1 = max(0, min_location[1] - radius)
            y2 = min(values.shape[0], min_location[1] + radius + 1)
            values[y1:y2, x1:x2] = np.inf
        return candidates

    def _select_reliable(
        self, candidates: list[_Candidate]
    ) -> Optional[CursorMatch]:
        if not candidates:
            return None

        distinct: list[_Candidate] = []
        for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
            if all(
                abs(candidate.origin[0] - kept.origin[0])
                > self.suppression_radius
                or abs(candidate.origin[1] - kept.origin[1])
                > self.suppression_radius
                for kept in distinct
            ):
                distinct.append(candidate)

        best = distinct[0]
        second = distinct[1] if len(distinct) > 1 else None
        uniqueness = best.score - second.score if second is not None else 1.0
        if best.score < self.min_score or uniqueness < self.uniqueness_margin:
            return None

        hotspot = (
            best.origin[0] + self.hotspot[0],
            best.origin[1] + self.hotspot[1],
        )
        return CursorMatch(
            hotspot=hotspot,
            score=best.score,
            uniqueness=uniqueness,
            second_score=second.score if second is not None else None,
            visible_fraction=best.visible_fraction,
            template_origin=best.origin,
        )


def locate_cursor_hotspot(
    frame: np.ndarray,
    template: np.ndarray,
    *,
    hotspot: Point,
    previous_hotspot: Optional[Point] = None,
    local_radius: Optional[int | Sequence[int]] = None,
    search_region: Optional[Box] = None,
    **tracker_options,
) -> Optional[CursorMatch]:
    """Pure-function convenience wrapper around :class:`CursorTracker`."""

    return CursorTracker(
        template,
        hotspot=hotspot,
        **tracker_options,
    ).locate(
        frame,
        previous_hotspot=previous_hotspot,
        local_radius=local_radius,
        search_region=search_region,
    )
