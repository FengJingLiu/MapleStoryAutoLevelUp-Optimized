import pytest

from src.navigation.spatial_launch import (
    HeroVelocityTracker,
    platform_launch_point,
    reachable_takeoff_edge_margin,
    rope_launch_distance,
    timed_platform_jump,
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


def test_timed_jump_subtracts_measured_alt_feedback_latency():
    decision = timed_platform_jump(
        (50, 18, 55, 22),
        "right",
        player_x=49.5,
        sampled_at=100.0,
        now=100.0,
        current_speed_px_per_sec=21.3575,
        cruise_speed_px_per_sec=21.3575,
        input_latency_seconds=0.157,
        takeoff_edge_margin_px=1.0,
    )

    assert decision is not None
    assert decision.takeoff_x == 54
    assert decision.latency_compensation_px == pytest.approx(3.3531, rel=1e-4)
    assert decision.send_at == pytest.approx(
        100.0 + (54.0 - 49.5) / 21.3575 - 0.157
    )
    assert decision.delay_seconds == pytest.approx(
        (54.0 - 49.5) / 21.3575 - 0.157
    )


def test_timed_jump_is_symmetric_and_waits_for_running_speed():
    assert timed_platform_jump(
        (61, 73, 66, 77),
        "left",
        player_x=70,
        sampled_at=10.0,
        now=10.0,
        current_speed_px_per_sec=10.0,
        cruise_speed_px_per_sec=20.0,
        input_latency_seconds=0.157,
        takeoff_edge_margin_px=1.0,
    ) is None

    decision = timed_platform_jump(
        (61, 73, 66, 77),
        "left",
        player_x=68,
        sampled_at=10.0,
        now=10.0,
        current_speed_px_per_sec=20.0,
        cruise_speed_px_per_sec=20.0,
        input_latency_seconds=0.157,
        takeoff_edge_margin_px=1.0,
    )

    assert decision is not None
    assert decision.edge_x == 61
    assert decision.takeoff_x == 62
    assert decision.remaining_px == 7


def test_takeoff_margin_is_capped_by_higher_platform_landing_arc():
    p15_to_p16 = reachable_takeoff_edge_margin(
        (87, 75),
        (92, 64),
        13.9683,
        11.0251,
        2.22,
    )
    p17_to_p16 = reachable_takeoff_edge_margin(
        (124, 75),
        (118, 64),
        13.9683,
        11.0251,
        2.22,
    )

    assert p15_to_p16 == pytest.approx(2.0537, abs=1e-4)
    assert p17_to_p16 == pytest.approx(1.0537, abs=1e-4)
