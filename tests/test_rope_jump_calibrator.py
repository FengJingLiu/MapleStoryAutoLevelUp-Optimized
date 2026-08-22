import unittest

from tools.rope_jump_calibrator import (
    build_parser,
    infer_trial_metrics,
    interpolate_position,
    summarize_trials,
)


def make_trial(label, jump_x, rope_x=102, direction="right", trial_id=1):
    sign = 1 if direction == "right" else -1
    keys = [direction, "up", "alt"]
    samples = [
        {"t": 0.40, "x": jump_x - sign * 8, "y": 333, "keys": [direction]},
        {"t": 0.80, "x": jump_x - sign * 2, "y": 333, "keys": [direction]},
        {"t": 0.90, "x": jump_x, "y": 333, "keys": keys},
        {"t": 1.10, "x": jump_x + sign * 4, "y": 329, "keys": keys},
        {"t": 1.30, "x": rope_x, "y": 314, "keys": ["up"]},
        {"t": 1.60, "x": rope_x, "y": 300, "keys": ["up"]},
    ]
    if label == "failure":
        samples = samples[:3] + [
            {"t": 1.10, "x": jump_x + sign * 4, "y": 329, "keys": keys},
            {"t": 1.50, "x": jump_x + sign * 7, "y": 333, "keys": []},
        ]
    return {
        "trial": trial_id,
        "label": label,
        "jump_event": {"t": 0.90, "keys": keys},
        "label_time": 1.70,
        "samples": samples,
    }


class RopeJumpCalibratorTests(unittest.TestCase):
    def test_calibration_defaults_to_capture_card_frame_rate(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.fps, 60)

    def test_interpolates_hero_position_at_jump_key_edge(self):
        position = interpolate_position(
            [
                {"t": 1.0, "x": 80, "y": 333},
                {"t": 1.2, "x": 84, "y": 331},
            ],
            1.1,
        )

        self.assertEqual(position, (82.0, 332.0))

    def test_success_infers_moving_launch_distance_from_rope_ascent(self):
        metrics = infer_trial_metrics(make_trial("success", jump_x=87))

        self.assertEqual(metrics["approach_direction"], "right")
        self.assertEqual(metrics["jump_x"], 87.0)
        self.assertGreater(metrics["approach_speed_px_s"], 0)
        self.assertTrue(metrics["moving_toward_rope"])
        self.assertEqual(metrics["rope_x_estimate"], 102.0)
        self.assertEqual(metrics["lead_distance_px"], 15.0)

    def test_failure_uses_successful_rope_x_for_boundary_distance(self):
        trials = [
            make_trial("success", 86, trial_id=1),
            make_trial("success", 87, trial_id=2),
            make_trial("success", 88, trial_id=3),
            make_trial("success", 89, trial_id=4),
            make_trial("success", 90, trial_id=5),
            make_trial("failure", 93, trial_id=6),
            make_trial("failure", 82, trial_id=7),
        ]

        summary = summarize_trials(trials)
        approach = summary["approaches"]["right"]

        self.assertEqual(summary["rope_x_estimate"], 102.0)
        self.assertEqual(approach["successful_attempts"], 5)
        self.assertEqual(approach["failed_attempts"], 2)
        self.assertEqual(approach["observed_success_window_px"], [12.0, 16.0])
        self.assertEqual(approach["failed_lead_distances_px"], [9.0, 20.0])
        self.assertTrue(approach["ready"])


if __name__ == "__main__":
    unittest.main()
