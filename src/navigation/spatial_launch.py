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


@dataclass(frozen=True, slots=True)
class TimedPlatformJump:
    """One latency-compensated directional Jump schedule."""

    send_at: float
    delay_seconds: float
    takeoff_x: float
    edge_x: float
    remaining_px: float
    speed_px_per_sec: float
    input_latency_seconds: float
    latency_compensation_px: float


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


def timed_platform_jump(
    bounds: tuple[float, float, float, float],
    direction: str,
    player_x: float,
    sampled_at: float,
    now: float,
    current_speed_px_per_sec: float,
    cruise_speed_px_per_sec: float,
    input_latency_seconds: float,
    takeoff_edge_margin_px: float,
    *,
    minimum_speed_ratio: float = 0.75,
) -> TimedPlatformJump | None:
    """Schedule Alt so game-side takeoff occurs just inside an edge.

    ``input_latency_seconds`` is the measured end-to-end interval from the
    host sending Alt until the capture pipeline first observes upward Hero
    motion.  Subtracting it from the predicted edge-arrival time compensates
    both remote HID/game input and capture feedback latency.
    """
    if direction not in {"left", "right"}:
        raise ValueError("jump direction must be left or right")
    values = (
        player_x,
        sampled_at,
        now,
        current_speed_px_per_sec,
        cruise_speed_px_per_sec,
        input_latency_seconds,
        takeoff_edge_margin_px,
        minimum_speed_ratio,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("timed jump values must be finite")
    if cruise_speed_px_per_sec <= 0 or input_latency_seconds < 0 or \
            takeoff_edge_margin_px < 0 or not 0 < minimum_speed_ratio <= 1:
        raise ValueError("invalid timed jump calibration")

    ratio = speed_ratio(current_speed_px_per_sec, cruise_speed_px_per_sec)
    if ratio < minimum_speed_ratio:
        return None
    speed = min(
        float(cruise_speed_px_per_sec),
        max(0.0, float(current_speed_px_per_sec)),
    )
    if speed <= 0:
        return None

    left, _, right, _ = map(float, bounds)
    left, right = min(left, right), max(left, right)
    edge_x = right if direction == "right" else left
    platform_width = max(0.0, right - left)
    margin = min(float(takeoff_edge_margin_px), platform_width)
    takeoff_x = edge_x - margin if direction == "right" else edge_x + margin
    remaining = (
        edge_x - float(player_x)
        if direction == "right"
        else float(player_x) - edge_x
    )
    distance_to_takeoff = (
        takeoff_x - float(player_x)
        if direction == "right"
        else float(player_x) - takeoff_x
    )
    travel_seconds = max(0.0, distance_to_takeoff / speed)
    send_at = (
        float(sampled_at)
        + travel_seconds
        - float(input_latency_seconds)
    )
    return TimedPlatformJump(
        send_at=send_at,
        delay_seconds=max(0.0, send_at - float(now)),
        takeoff_x=takeoff_x,
        edge_x=edge_x,
        remaining_px=remaining,
        speed_px_per_sec=speed,
        input_latency_seconds=float(input_latency_seconds),
        latency_compensation_px=speed * float(input_latency_seconds),
    )


def reachable_takeoff_edge_margin(
    jump_source: tuple[float, float],
    target: tuple[float, float],
    jump_height_px: float,
    jump_distance_px: float,
    requested_margin_px: float,
    *,
    landing_reserve_px: float = 1.0,
) -> float:
    """Cap an interior takeoff margin by the descending jump arc.

    ``jump_distance_px`` is the same-height airborne distance.  A higher
    destination intersects the descending arc sooner, so consuming too much
    of that distance inside the source platform can make an otherwise valid
    edge unreachable.  The small landing reserve keeps the Hero center past
    the destination edge instead of merely touching it.
    """
    source_x, source_y = map(float, jump_source[:2])
    target_x, target_y = map(float, target[:2])
    values = (
        source_x,
        source_y,
        target_x,
        target_y,
        jump_height_px,
        jump_distance_px,
        requested_margin_px,
        landing_reserve_px,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("jump reach values must be finite")
    jump_height_px = float(jump_height_px)
    jump_distance_px = float(jump_distance_px)
    requested_margin_px = float(requested_margin_px)
    landing_reserve_px = float(landing_reserve_px)
    if jump_height_px <= 0 or jump_distance_px <= 0 or \
            requested_margin_px < 0 or landing_reserve_px < 0:
        raise ValueError("invalid jump reach calibration")

    rise_px = source_y - target_y
    if rise_px >= jump_height_px:
        return 0.0
    descending_fraction = (
        1.0 + math.sqrt(max(0.0, 1.0 - rise_px / jump_height_px))
    ) / 2.0
    reachable_distance = jump_distance_px * descending_fraction
    edge_gap = abs(target_x - source_x)
    reachable_margin = max(
        0.0,
        reachable_distance - edge_gap - landing_reserve_px,
    )
    return min(requested_margin_px, reachable_margin)


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
