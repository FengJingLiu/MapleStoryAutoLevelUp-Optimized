"""Frame-driven launch points derived from live minimap Hero motion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SpatialLaunchPoint:
    """One speed-adjusted launch point expressed in navigation pixels."""

    trigger_x: float
    current_speed_px_per_sec: float
    cruise_speed_px_per_sec: float
    speed_ratio: float


class HeroVelocityTracker:
    """Estimate current horizontal speed from a short minimap sample window."""

    def __init__(
        self,
        *,
        window_seconds: float = 0.18,
        minimum_span_seconds: float = 0.05,
        maximum_vertical_delta_px: float = 2.5,
    ):
        self.window_seconds = float(window_seconds)
        self.minimum_span_seconds = float(minimum_span_seconds)
        self.maximum_vertical_delta_px = float(maximum_vertical_delta_px)
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("velocity window must be positive")
        if not math.isfinite(self.minimum_span_seconds) or not (
                0 < self.minimum_span_seconds <= self.window_seconds):
            raise ValueError("minimum velocity span must fit inside the window")
        if not math.isfinite(self.maximum_vertical_delta_px) or \
                self.maximum_vertical_delta_px < 0:
            raise ValueError("maximum vertical delta must be non-negative")
        self._samples: deque[tuple[float, float, float]] = deque(maxlen=32)
        self._direction: str | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._direction = None

    def observe(
        self,
        point: tuple[float, float],
        timestamp: float,
        *,
        direction: str,
        vertical_command: str,
        action: str,
    ) -> float:
        """Record one raw Hero point and return current directional speed."""
        x, y = map(float, point[:2])
        timestamp = float(timestamp)
        valid = (
            direction in {"left", "right"}
            and vertical_command in {"none", "stop"}
            and action in {"none", "stop"}
            and all(math.isfinite(value) for value in (x, y, timestamp))
        )
        if not valid:
            self.reset()
            return 0.0
        if self._direction != direction:
            self.reset()
            self._direction = direction
        if self._samples and (
            timestamp <= self._samples[-1][0]
            or timestamp - self._samples[-1][0] > 0.35
            or abs(y - self._samples[-1][2]) > self.maximum_vertical_delta_px
        ):
            self.reset()
            self._direction = direction

        self._samples.append((timestamp, x, y))
        cutoff = timestamp - self.window_seconds
        while len(self._samples) > 2 and self._samples[1][0] < cutoff:
            self._samples.popleft()
        return self.speed(direction)

    def speed(self, direction: str | None = None) -> float:
        if direction is not None and direction != self._direction:
            return 0.0
        if len(self._samples) < 2:
            return 0.0
        span = self._samples[-1][0] - self._samples[0][0]
        if span < self.minimum_span_seconds:
            return 0.0

        times = [sample[0] for sample in self._samples]
        xs = [sample[1] for sample in self._samples]
        mean_t = sum(times) / len(times)
        mean_x = sum(xs) / len(xs)
        denominator = sum((value - mean_t) ** 2 for value in times)
        if denominator <= 0:
            return 0.0
        slope = sum(
            (sample_t - mean_t) * (sample_x - mean_x)
            for sample_t, sample_x in zip(times, xs)
        ) / denominator
        directional_speed = slope if self._direction == "right" else -slope
        return max(0.0, float(directional_speed))


def speed_ratio(current_speed: float, cruise_speed: float) -> float:
    current_speed = float(current_speed)
    cruise_speed = float(cruise_speed)
    if not all(math.isfinite(value) for value in (
            current_speed, cruise_speed)) or cruise_speed <= 0:
        return 0.0
    return min(1.0, max(0.0, current_speed / cruise_speed))


def platform_launch_point(
    bounds: tuple[float, float, float, float],
    direction: str,
    current_speed_px_per_sec: float,
    cruise_speed_px_per_sec: float,
) -> SpatialLaunchPoint:
    """Move the trigger from the outer edge toward the interior as speed rises."""
    if direction not in {"left", "right"}:
        raise ValueError("launch direction must be left or right")
    left, _, right, _ = map(float, bounds)
    left, right = min(left, right), max(left, right)
    ratio = speed_ratio(current_speed_px_per_sec, cruise_speed_px_per_sec)
    trigger_x = (
        right - (right - left) * ratio
        if direction == "right"
        else left + (right - left) * ratio
    )
    return SpatialLaunchPoint(
        trigger_x=trigger_x,
        current_speed_px_per_sec=max(0.0, float(current_speed_px_per_sec)),
        cruise_speed_px_per_sec=max(0.0, float(cruise_speed_px_per_sec)),
        speed_ratio=ratio,
    )


def rope_launch_distance(
    late_distance_px: float,
    early_distance_px: float,
    current_speed_px_per_sec: float,
    cruise_speed_px_per_sec: float,
) -> SpatialLaunchPoint:
    """Choose a farther-from-rope launch point for a faster current run."""
    late = max(0.0, float(late_distance_px))
    early = max(late, float(early_distance_px))
    ratio = speed_ratio(current_speed_px_per_sec, cruise_speed_px_per_sec)
    distance = late + (early - late) * ratio
    return SpatialLaunchPoint(
        trigger_x=distance,
        current_speed_px_per_sec=max(0.0, float(current_speed_px_per_sec)),
        cruise_speed_px_per_sec=max(0.0, float(cruise_speed_px_per_sec)),
        speed_ratio=ratio,
    )
