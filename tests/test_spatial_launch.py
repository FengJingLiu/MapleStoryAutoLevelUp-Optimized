import pytest

from src.navigation.spatial_launch import (
    HeroVelocityTracker,
    platform_launch_point,
    rope_launch_distance,
)


def test_velocity_tracker_uses_a_short_window_and_resets_on_attack():
    tracker = HeroVelocityTracker(
        window_seconds=0.20,
        minimum_span_seconds=0.05,
    )
    for index, x in enumerate((10, 10, 11, 11, 12, 12, 13)):
        speed = tracker.observe(
            (x, 20),
            index / 60.0,
            direction="right",
            vertical_command="none",
            action="none",
        )
    assert speed == pytest.approx(31.07, rel=0.08)

    assert tracker.observe(
        (13, 20),
        7 / 60.0,
        direction="right",
        vertical_command="none",
        action="attack",
    ) == 0.0
    assert tracker.speed("right") == 0.0


def test_platform_trigger_is_later_when_slow_and_earlier_when_fast():
    slow = platform_launch_point((100, 20, 106, 24), "right", 0, 20)
    half = platform_launch_point((100, 20, 106, 24), "right", 10, 20)
    fast = platform_launch_point((100, 20, 106, 24), "right", 20, 20)
    assert (slow.trigger_x, half.trigger_x, fast.trigger_x) == (106, 103, 100)

    slow_left = platform_launch_point((100, 20, 106, 24), "left", 0, 20)
    fast_left = platform_launch_point((100, 20, 106, 24), "left", 20, 20)
    assert (slow_left.trigger_x, fast_left.trigger_x) == (100, 106)


def test_rope_trigger_moves_farther_from_rope_as_speed_rises():
    slow = rope_launch_distance(1, 6, 0, 20)
    half = rope_launch_distance(1, 6, 10, 20)
    fast = rope_launch_distance(1, 6, 20, 20)
    assert slow.trigger_x == 1
    assert half.trigger_x == 3.5
    assert fast.trigger_x == 6
