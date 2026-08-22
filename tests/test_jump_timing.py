import pytest

from src.navigation.jump_timing import predict_directional_jump


def test_prediction_projects_only_confirmed_same_direction_frame_age():
    prediction = predict_directional_jump(
        sample_x=100,
        launch_x=110,
        direction="right",
        speed_px_per_second=20,
        capture_age_seconds=0.30,
        direction_held_seconds=0.05,
        input_latency_seconds=0.035,
        position_uncertainty_px=1.0,
        lookahead_seconds=1.0,
        maximum_frame_age_seconds=0.5,
    )

    assert prediction is not None
    assert prediction.held_motion_seconds == pytest.approx(0.05)
    assert prediction.predicted_x == pytest.approx(101.0)
    assert prediction.input_lead_seconds == pytest.approx(0.085)
    assert prediction.delay_seconds == pytest.approx(0.365)


def test_prediction_arms_before_next_low_fps_sample_would_cross_launch():
    prediction = predict_directional_jump(
        sample_x=138,
        launch_x=149,
        direction="right",
        speed_px_per_second=21.3575,
        capture_age_seconds=0.20,
        direction_held_seconds=1.0,
        input_latency_seconds=0.035,
        position_uncertainty_px=1.208,
        lookahead_seconds=0.675,
        maximum_frame_age_seconds=0.5,
    )

    assert prediction is not None
    assert prediction.predicted_x == pytest.approx(142.2715)
    assert 0 < prediction.delay_seconds < 0.5
    assert prediction.delay_seconds == pytest.approx(0.2241, abs=0.001)


def test_prediction_is_symmetric_for_left_jump():
    prediction = predict_directional_jump(
        sample_x=104,
        launch_x=93,
        direction="left",
        speed_px_per_second=21.3575,
        capture_age_seconds=0.20,
        direction_held_seconds=1.0,
        input_latency_seconds=0.035,
        position_uncertainty_px=1.208,
        lookahead_seconds=0.675,
        maximum_frame_age_seconds=0.5,
    )

    assert prediction is not None
    assert prediction.predicted_x == pytest.approx(99.7285)
    assert prediction.delay_seconds == pytest.approx(0.2241, abs=0.001)


def test_prediction_rejects_stale_frame_and_far_target():
    common = dict(
        sample_x=100,
        launch_x=120,
        direction="right",
        speed_px_per_second=20,
        direction_held_seconds=1.0,
        input_latency_seconds=0.035,
        position_uncertainty_px=1.0,
        maximum_frame_age_seconds=0.5,
    )
    assert predict_directional_jump(
        **common,
        capture_age_seconds=0.6,
        lookahead_seconds=1.0,
    ) is None
    assert predict_directional_jump(
        **common,
        capture_age_seconds=0.0,
        lookahead_seconds=0.5,
    ) is None


def test_prediction_rejects_a_launch_point_already_missed():
    assert predict_directional_jump(
        sample_x=112,
        launch_x=110,
        direction="right",
        speed_px_per_second=20,
        capture_age_seconds=0.0,
        direction_held_seconds=0.0,
        input_latency_seconds=0.035,
        position_uncertainty_px=1.0,
        lookahead_seconds=1.0,
        maximum_frame_age_seconds=0.5,
        overshoot_tolerance_px=1.0,
    ) is None
