"""Estimate Hero movement in the stable WZ navigation pixel space."""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MotionObservation:
    walk_speed_px_per_sec: float | None
    jump_height_px: float | None
    jump_distance_px: float | None
    walk_samples: int
    jump_samples: int
    jump_runup_distance_px: float | None = None


@dataclass(slots=True)
class _PendingJump:
    command_origin: tuple[float, float]
    last_ground_point: tuple[float, float]
    started_at: float


@dataclass(slots=True)
class _AirborneJump:
    origin: tuple[float, float]
    minimum_y: float
    maximum_dx: float
    runup_distance: float
    started_at: float
    descent_seen: bool = False
    landing_stable_frames: int = 0
    landing_candidate_distance: float | None = None


class HeroMotionEstimator:
    """Collect conservative medians without issuing calibration-only input."""

    TAKEOFF_MIN_RISE_PX = 1.5
    PENDING_JUMP_TIMEOUT_SEC = 0.8
    LANDING_MIN_DESCENT_PX = 3.0
    LANDING_MAX_DELTA_Y_PX = 2.5
    LANDING_CONFIRM_FRAMES = 2
    MINIMUM_FLIGHT_TIME_SEC = 0.20
    MAXIMUM_FLIGHT_TIME_SEC = 2.0

    def __init__(
        self,
        *,
        minimum_walk_samples: int = 12,
        minimum_jump_samples: int = 2,
        maximum_samples: int = 80,
    ):
        self.minimum_walk_samples = max(3, int(minimum_walk_samples))
        self.minimum_jump_samples = max(1, int(minimum_jump_samples))
        self.walk_speeds: deque[float] = deque(maxlen=max(10, maximum_samples))
        self.jump_heights: deque[float] = deque(maxlen=max(4, maximum_samples // 4))
        self.jump_distances: deque[float] = deque(maxlen=max(4, maximum_samples // 4))
        self.jump_runup_distances: deque[float] = deque(
            maxlen=max(4, maximum_samples // 4)
        )
        self._previous_point: tuple[float, float] | None = None
        self._previous_time: float | None = None
        self._pending_jump: _PendingJump | None = None
        self._jump: _AirborneJump | None = None

    @staticmethod
    def _command(value: str | tuple[str, str, str]) -> tuple[str, str, str]:
        if isinstance(value, tuple) and len(value) == 3:
            return tuple(str(item) for item in value)
        parts = str(value).split()
        return tuple(parts) if len(parts) == 3 else ("none", "none", "none")

    @property
    def jump_active(self) -> bool:
        return self._pending_jump is not None or self._jump is not None

    @classmethod
    def is_jump_command(cls, value: str | tuple[str, str, str]) -> bool:
        command = cls._command(value)
        return command[2] == "jump" and command[1] in {"none", "stop"}

    def observe(
        self,
        point: tuple[float, float],
        timestamp: float,
        previous_command: str | tuple[str, str, str],
        *,
        on_ladder: bool,
    ) -> MotionObservation:
        command = self._command(previous_command)
        previous = self._previous_point
        previous_time = self._previous_time
        has_recent_previous = previous is not None and \
            previous_time is not None and \
            0.01 <= timestamp - previous_time <= 0.35

        if on_ladder:
            self._pending_jump = None
            self._jump = None
        elif self.is_jump_command(command) and not self.jump_active:
            command_origin = previous if has_recent_previous else point
            started_at = previous_time if has_recent_previous else timestamp
            assert command_origin is not None and started_at is not None
            anchor = tuple(map(float, command_origin))
            self._pending_jump = _PendingJump(
                command_origin=anchor,
                last_ground_point=anchor,
                started_at=float(started_at),
            )

        if previous is not None and previous_time is not None:
            elapsed = timestamp - previous_time
            delta_x = point[0] - previous[0]
            delta_y = point[1] - previous[1]
            # ``previous_command`` is the command that was active during the
            # interval ending at this point, so attribute this displacement to
            # it directly instead of lagging the samples by another frame.
            move_x, move_y, action = command
            if 0.01 <= elapsed <= 0.35 and not on_ladder and \
                    not self.jump_active and move_x in {"left", "right"} and \
                    move_y in {"none", "stop"} and action in {
                        "none", "attack", "directional_aoe", "power_knockback"
                    } and abs(delta_y) <= 2.5:
                speed = abs(delta_x) / elapsed
                if 1.0 <= speed <= 500.0:
                    self.walk_speeds.append(speed)

            if self._jump is not None:
                jump = self._jump
                jump.minimum_y = min(jump.minimum_y, float(point[1]))
                jump.maximum_dx = max(
                    jump.maximum_dx,
                    abs(float(point[0]) - jump.origin[0]),
                )
                descent = float(point[1]) - jump.minimum_y
                if descent >= self.LANDING_MIN_DESCENT_PX:
                    jump.descent_seen = True

                age = timestamp - jump.started_at
                stable = jump.descent_seen and \
                    age >= self.MINIMUM_FLIGHT_TIME_SEC and \
                    abs(delta_y) <= self.LANDING_MAX_DELTA_Y_PX
                if stable:
                    if jump.landing_stable_frames == 0:
                        # Keep the first grounded distance. The second frame is
                        # confirmation only and may already include ground run.
                        jump.landing_candidate_distance = jump.maximum_dx
                    jump.landing_stable_frames += 1
                else:
                    jump.landing_stable_frames = 0
                    jump.landing_candidate_distance = None

                if jump.landing_stable_frames >= self.LANDING_CONFIRM_FRAMES:
                    height = jump.origin[1] - jump.minimum_y
                    distance = jump.landing_candidate_distance
                    if height >= 2.0 and distance is not None:
                        self.jump_heights.append(height)
                        self.jump_distances.append(max(0.0, distance))
                        self.jump_runup_distances.append(
                            max(0.0, jump.runup_distance)
                        )
                    self._jump = None
                elif age > self.MAXIMUM_FLIGHT_TIME_SEC:
                    self._jump = None

            pending = self._pending_jump
            if pending is not None:
                pending_age = timestamp - pending.started_at
                if pending_age > self.PENDING_JUMP_TIMEOUT_SEC:
                    self._pending_jump = None
                elif delta_y <= -self.TAKEOFF_MIN_RISE_PX:
                    origin = pending.last_ground_point
                    point_float = tuple(map(float, point))
                    self._jump = _AirborneJump(
                        origin=origin,
                        minimum_y=min(origin[1], point_float[1]),
                        maximum_dx=abs(point_float[0] - origin[0]),
                        runup_distance=abs(
                            origin[0] - pending.command_origin[0]
                        ),
                        started_at=float(previous_time),
                    )
                    self._pending_jump = None
                else:
                    pending.last_ground_point = tuple(map(float, point))

        pending = self._pending_jump
        if pending is not None and \
                timestamp - pending.started_at > self.PENDING_JUMP_TIMEOUT_SEC:
            self._pending_jump = None

        self._previous_point = tuple(map(float, point))
        self._previous_time = float(timestamp)
        return self.snapshot()

    def snapshot(self) -> MotionObservation:
        walk = (
            statistics.median(self.walk_speeds)
            if len(self.walk_speeds) >= self.minimum_walk_samples
            else None
        )
        jump_height = (
            statistics.median(self.jump_heights)
            if len(self.jump_heights) >= self.minimum_jump_samples
            else None
        )
        jump_distance = (
            statistics.median(self.jump_distances)
            if len(self.jump_distances) >= self.minimum_jump_samples
            and any(value > 0 for value in self.jump_distances)
            else None
        )
        jump_runup = (
            statistics.median(self.jump_runup_distances)
            if len(self.jump_runup_distances) >= self.minimum_jump_samples
            else None
        )
        values = (walk, jump_height, jump_distance, jump_runup)
        if any(value is not None and not math.isfinite(value) for value in values):
            return MotionObservation(
                None,
                None,
                None,
                len(self.walk_speeds),
                len(self.jump_heights),
                None,
            )
        return MotionObservation(
            walk_speed_px_per_sec=walk,
            jump_height_px=jump_height,
            jump_distance_px=jump_distance,
            walk_samples=len(self.walk_speeds),
            jump_samples=len(self.jump_heights),
            jump_runup_distance_px=jump_runup,
        )
