"""Latency-compensated timing for generated horizontal WZ jumps."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TimedJumpPrediction:
    """One timer decision expressed entirely in navigation pixels/seconds."""

    direction: str
    sample_x: float
    predicted_x: float
    launch_x: float
    speed_px_per_second: float
    capture_age_seconds: float
    held_motion_seconds: float
    distance_to_launch_px: float
    time_to_launch_seconds: float
    input_lead_seconds: float
    delay_seconds: float
    lookahead_seconds: float


def predict_directional_jump(
    *,
    sample_x: float,
    launch_x: float,
    direction: str,
    speed_px_per_second: float,
    capture_age_seconds: float,
    direction_held_seconds: float,
    input_latency_seconds: float,
    position_uncertainty_px: float,
    lookahead_seconds: float,
    maximum_frame_age_seconds: float,
    overshoot_tolerance_px: float = 1.0,
) -> TimedJumpPrediction | None:
    """Predict when a continuously moving Hero reaches one WZ launch point.

    The observed minimap position belongs to an older captured frame.  Only
    the portion of that age for which the same physical direction was held is
    projected forward.  HID latency and registration uncertainty are then
    converted into an earlier timer deadline.
    """
    if direction not in {"left", "right"}:
        raise ValueError("direction must be left or right")
    values = (
        sample_x,
        launch_x,
        speed_px_per_second,
        capture_age_seconds,
        direction_held_seconds,
        input_latency_seconds,
        position_uncertainty_px,
        lookahead_seconds,
        maximum_frame_age_seconds,
        overshoot_tolerance_px,
    )
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timed-jump values must be finite numbers") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("timed-jump values must be finite numbers")

    (
        sample_x,
        launch_x,
        speed,
        frame_age,
        held_seconds,
        input_latency,
        uncertainty,
        lookahead,
        maximum_frame_age,
        overshoot_tolerance,
    ) = values
    if speed <= 0:
        return None
    if min(
        frame_age,
        held_seconds,
        input_latency,
        uncertainty,
        lookahead,
        maximum_frame_age,
        overshoot_tolerance,
    ) < 0:
        raise ValueError("timed-jump durations and tolerances cannot be negative")
    if frame_age > maximum_frame_age:
        return None

    sign = 1.0 if direction == "right" else -1.0
    held_motion = min(frame_age, held_seconds)
    predicted_x = sample_x + sign * speed * held_motion
    distance = sign * (launch_x - predicted_x)
    if distance < -overshoot_tolerance:
        return None

    time_to_launch = max(0.0, distance / speed)
    # Registration residual is a spatial error.  Converting it with the same
    # measured speed makes the advance scale correctly when minimap pixels are
    # small instead of adding another arbitrary pixel margin.
    effective_lead = input_latency + uncertainty / speed
    delay = max(0.0, time_to_launch - effective_lead)
    if delay > lookahead:
        return None

    return TimedJumpPrediction(
        direction=direction,
        sample_x=sample_x,
        predicted_x=predicted_x,
        launch_x=launch_x,
        speed_px_per_second=speed,
        capture_age_seconds=frame_age,
        held_motion_seconds=held_motion,
        distance_to_launch_px=max(0.0, distance),
        time_to_launch_seconds=time_to_launch,
        input_lead_seconds=effective_lead,
        delay_seconds=delay,
        lookahead_seconds=lookahead,
    )
