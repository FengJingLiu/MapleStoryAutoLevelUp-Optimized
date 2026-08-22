"""Record labelled manual rope-mount attempts from the route recorder.

The normal route recorder is reused for minimap alignment, Hero tracking and
ESP32 input forwarding.  This tool adds exact Jump key-edge timestamps and
manual outcome labels so a moving launch window can be learned from real
attempts instead of tuning one fixed x coordinate by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from statistics import median

import cv2

from src.input.KeyBoardListener import normalize_key_name
from src.input.CaptureFramePreprocessor import preprocess_capture_frame
from src.input.CaptureSource import capture_profile_override
from src.utils.common import get_player_location_on_minimap
from src.utils.logger import logger
from tools.routeRecorder import RouteRecorder


SUCCESS_KEY = "f5"
FAILURE_KEY = "f6"
DISCARD_KEY = "f7"


def _percentile(values, percentile):
    """Return a linearly interpolated percentile without another dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def interpolate_position(samples, event_time):
    """Interpolate Hero coordinates at an asynchronous keyboard event time."""
    valid = [
        sample for sample in samples
        if sample.get("x") is not None and sample.get("y") is not None
    ]
    if not valid:
        return None
    valid.sort(key=lambda sample: float(sample["t"]))
    event_time = float(event_time)
    if event_time <= float(valid[0]["t"]):
        return (float(valid[0]["x"]), float(valid[0]["y"]))
    if event_time >= float(valid[-1]["t"]):
        return (float(valid[-1]["x"]), float(valid[-1]["y"]))

    for previous, current in zip(valid, valid[1:]):
        previous_t = float(previous["t"])
        current_t = float(current["t"])
        if previous_t <= event_time <= current_t:
            duration = current_t - previous_t
            if duration <= 0:
                return (float(current["x"]), float(current["y"]))
            ratio = (event_time - previous_t) / duration
            x = float(previous["x"]) + \
                (float(current["x"]) - float(previous["x"])) * ratio
            y = float(previous["y"]) + \
                (float(current["y"]) - float(previous["y"])) * ratio
            return (x, y)
    return (float(valid[-1]["x"]), float(valid[-1]["y"]))


def infer_trial_metrics(trial, min_climb_progress_px=3.0):
    """Derive approach speed and rope-relative launch distance for one trial."""
    samples = list(trial.get("samples", ()))
    jump_event = trial.get("jump_event", {})
    jump_time = float(jump_event.get("t", 0.0))
    jump_position = interpolate_position(samples, jump_time)
    if jump_position is None:
        return {
            "approach_direction": "unknown",
            "jump_x": None,
            "jump_y": None,
            "approach_speed_px_s": None,
            "rope_x_estimate": None,
            "lead_distance_px": None,
        }

    jump_x, jump_y = jump_position
    event_keys = {
        normalize_key_name(key) for key in jump_event.get("keys", ())
    }
    if "right" in event_keys and "left" not in event_keys:
        direction = "right"
    elif "left" in event_keys and "right" not in event_keys:
        direction = "left"
    else:
        direction = "unknown"

    # Use only the final 0.65 seconds before Jump.  This captures actual
    # approach momentum and excludes the earlier staging/repositioning path.
    pre_jump = [
        sample for sample in samples
        if jump_time - 0.65 <= float(sample["t"]) <= jump_time
    ]
    speed = None
    if len(pre_jump) >= 2:
        first = pre_jump[0]
        last = pre_jump[-1]
        duration = float(last["t"]) - float(first["t"])
        if duration > 0:
            speed = (float(last["x"]) - float(first["x"])) / duration
            if direction == "unknown" and abs(speed) >= 0.5:
                direction = "right" if speed > 0 else "left"

    frame_intervals = []
    for previous, current in zip(samples, samples[1:]):
        interval = float(current["t"]) - float(previous["t"])
        if interval > 0:
            frame_intervals.append(interval)
    sample_fps = (
        None if not frame_intervals
        else 1.0 / float(median(frame_intervals))
    )
    toward_speed = None
    if speed is not None:
        if direction == "right":
            toward_speed = speed
        elif direction == "left":
            toward_speed = -speed
    moving_toward_rope = (
        toward_speed is not None and toward_speed >= 3.0
    )

    rope_x = None
    lead_distance = None
    if trial.get("label") == "success":
        post_jump = [
            sample for sample in samples
            if float(sample["t"]) >= jump_time
        ]
        if post_jump:
            highest_y = min(float(sample["y"]) for sample in post_jump)
            climb_progress = jump_y - highest_y
            if climb_progress >= float(min_climb_progress_px):
                # The late/high section of a successful ascent is on the rope;
                # early airborne samples are still moving horizontally.
                high_progress = max(
                    float(min_climb_progress_px),
                    climb_progress * 0.60,
                )
                rope_samples = [
                    sample for sample in post_jump
                    if float(sample["y"]) <= jump_y - high_progress
                ]
                if rope_samples:
                    rope_x = float(median(
                        float(sample["x"]) for sample in rope_samples
                    ))

        if rope_x is not None:
            if direction == "right":
                lead_distance = rope_x - jump_x
            elif direction == "left":
                lead_distance = jump_x - rope_x

    return {
        "approach_direction": direction,
        "jump_x": round(jump_x, 3),
        "jump_y": round(jump_y, 3),
        "approach_speed_px_s": (
            None if speed is None else round(speed, 3)
        ),
        "actual_sample_fps": (
            None if sample_fps is None else round(sample_fps, 3)
        ),
        "toward_rope_speed_px_s": (
            None if toward_speed is None else round(toward_speed, 3)
        ),
        "moving_toward_rope": moving_toward_rope,
        "rope_x_estimate": (
            None if rope_x is None else round(rope_x, 3)
        ),
        "lead_distance_px": (
            None if lead_distance is None else round(lead_distance, 3)
        ),
    }


def summarize_trials(trials):
    """Summarize successful launch windows and nearby labelled failures."""
    enriched = []
    for trial in trials:
        metrics = dict(trial.get("metrics") or infer_trial_metrics(trial))
        enriched.append((trial, metrics))

    successful_rope_x = [
        metrics["rope_x_estimate"]
        for trial, metrics in enriched
        if trial.get("label") == "success"
        and metrics.get("rope_x_estimate") is not None
    ]
    session_rope_x = (
        None if not successful_rope_x else float(median(successful_rope_x))
    )

    groups = {}
    for direction in ("right", "left"):
        successes = []
        failures = []
        excluded_not_moving = 0
        for trial, metrics in enriched:
            if metrics.get("approach_direction") != direction:
                continue
            if trial.get("label") in {"success", "failure"} and not \
                    metrics.get("moving_toward_rope", False):
                excluded_not_moving += 1
                continue
            jump_x = metrics.get("jump_x")
            lead = metrics.get("lead_distance_px")
            if lead is None and session_rope_x is not None and jump_x is not None:
                lead = (
                    session_rope_x - float(jump_x)
                    if direction == "right"
                    else float(jump_x) - session_rope_x
                )
            if lead is None:
                continue
            if trial.get("label") == "success":
                successes.append(float(lead))
            elif trial.get("label") == "failure":
                failures.append(float(lead))

        if not successes and not failures:
            continue
        observed_window = None
        recommended_window = None
        if successes:
            observed_window = [min(successes), max(successes)]
            if len(successes) >= 5:
                recommended_window = [
                    _percentile(successes, 15),
                    _percentile(successes, 85),
                ]
        groups[direction] = {
            "successful_attempts": len(successes),
            "failed_attempts": len(failures),
            "excluded_not_moving_attempts": excluded_not_moving,
            "observed_success_window_px": (
                None if observed_window is None
                else [round(value, 3) for value in observed_window]
            ),
            "recommended_window_px": (
                None if recommended_window is None
                else [round(value, 3) for value in recommended_window]
            ),
            "failed_lead_distances_px": [
                round(value, 3) for value in sorted(failures)
            ],
            "ready": len(successes) >= 5 and len(failures) >= 2,
        }

    return {
        "total_attempts": sum(
            trial.get("label") in {"success", "failure"}
            for trial in trials
        ),
        "successes": sum(
            trial.get("label") == "success" for trial in trials
        ),
        "failures": sum(
            trial.get("label") == "failure" for trial in trials
        ),
        "rope_x_estimate": (
            None if session_rope_x is None else round(session_rope_x, 3)
        ),
        "approaches": groups,
    }


class RopeJumpCalibrator(RouteRecorder):
    """RouteRecorder with labelled, structured rope-jump trials."""

    def __init__(self, args):
        self._session_started_at = time.monotonic()
        self._event_lock = threading.Lock()
        self._sample_lock = threading.Lock()
        self._pending_events = deque()
        self._sample_history = deque(maxlen=1800)
        self._sample_timestamps = deque(maxlen=120)
        self._active_jump_event = None
        self._trials = []
        self._stopped = False
        self._sampler_stop = threading.Event()
        self._sampler_thread = None
        self._last_sampler_error_at = 0.0
        self._pre_roll_s = max(0.25, float(args.pre_roll))
        self._requested_output = args.output
        super().__init__(args)

        # Full 4K route rendering is intentionally kept at its normal low
        # refresh rate.  A separate lightweight thread below reads each unique
        # capture frame and detects only the small minimap Hero marker.
        self.sample_fps_limit = min(60, max(10, int(args.fps)))
        self._jump_key = normalize_key_name(self.cfg["key"].get("jump", ""))
        if not self._jump_key:
            self.stop()
            raise ValueError("Configured Jump key is empty")

        self.output_path = (
            Path(self._requested_output)
            if self._requested_output
            else Path(self.map_dir) / "rope_jump_trials.jsonl"
        )
        self.summary_path = self.output_path.with_name("rope_jump_summary.json")
        self.preview_dir = self.output_path.parent / "rope_jump_previews"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            self.stop()
            raise FileExistsError(
                f"Calibration output already exists: {self.output_path}"
            )

        self._forward_key_event = self.kb.key_event_handler
        self.kb.key_event_handler = self._handle_key_event
        self.kb.register_func_key_handler(
            SUCCESS_KEY, lambda: self._queue_label("success")
        )
        self.kb.register_func_key_handler(
            FAILURE_KEY, lambda: self._queue_label("failure")
        )
        self.kb.register_func_key_handler(
            DISCARD_KEY, lambda: self._queue_label("discarded")
        )
        self._sampler_thread = threading.Thread(
            target=self._sample_capture_frames,
            name="rope-calibration-hero-sampler",
            daemon=True,
        )
        self._sampler_thread.start()
        logger.info(
            "[rope-calibration] Ready: F1 pause/resume, "
            "F5 success, F6 failure, F7 discard; "
            f"Jump={self._jump_key}, sample_FPS={self.sample_fps_limit}, "
            f"preview_FPS={self.fps_limit}, "
            f"output={self.output_path}"
        )

    def _relative_time(self, absolute_time):
        return round(float(absolute_time) - self._session_started_at, 6)

    def _queue_event(self, event):
        with self._event_lock:
            self._pending_events.append(event)

    def _handle_key_event(self, key, pressed):
        if self._forward_key_event is not None:
            self._forward_key_event(key, pressed)
        normalized = normalize_key_name(key)
        if pressed and normalized == self._jump_key:
            self._queue_event({
                "type": "jump",
                "absolute_t": time.monotonic(),
                "recording": bool(self.is_enable),
                "keys": sorted({
                    normalize_key_name(value)
                    for value in tuple(self.kb.key_pressing)
                    if normalize_key_name(value)
                }),
            })

    def _queue_label(self, label):
        self._queue_event({
            "type": "label",
            "label": label,
            "absolute_t": time.monotonic(),
        })

    def _drain_events(self):
        with self._event_lock:
            events = list(self._pending_events)
            self._pending_events.clear()
        events.sort(key=lambda event: event["absolute_t"])
        for event in events:
            if event["type"] == "jump":
                if not event.get("recording", False):
                    continue
                if self._active_jump_event is not None:
                    logger.warning(
                        "[rope-calibration] Label the current attempt before "
                        "starting another Jump"
                    )
                    continue
                event = dict(event)
                event["t"] = self._relative_time(event.pop("absolute_t"))
                self._active_jump_event = event
                logger.info(
                    "[rope-calibration] Jump captured; press F5 on success "
                    "or F6 on failure"
                )
            elif event["type"] == "label":
                if self._active_jump_event is None:
                    logger.warning(
                        "[rope-calibration] No active Jump to label"
                    )
                    continue
                self._finish_trial(
                    event["label"],
                    self._relative_time(event["absolute_t"]),
                )

        for index in (4, 5, 6):
            if self.kb is not None:
                self.kb.is_pressed_func_key[index] = False

    def _current_sample(self, now, player_global):
        return {
            "t": self._relative_time(now),
            "x": int(player_global[0]),
            "y": int(player_global[1]),
            "keys": sorted({
                normalize_key_name(key)
                for key in tuple(self.kb.key_pressing)
                if normalize_key_name(key)
            }),
            "minimap_match_score": round(float(self.minimap_match_score), 6),
        }

    def _sample_capture_frames(self):
        """Track the Hero from every unique capture frame, independent of UI."""
        last_capture_time = None
        last_kept_time = None
        while not self._sampler_stop.is_set():
            capture = getattr(self, "capture", None)
            if capture is None:
                self._sampler_stop.wait(0.005)
                continue
            frame, capture_time = capture.get_frame_snapshot()
            if frame is None or capture_time is None \
                    or capture_time == last_capture_time:
                self._sampler_stop.wait(0.001)
                continue
            last_capture_time = capture_time

            # Respect an explicit lower target while never inventing duplicate
            # samples above the capture device's native frame rate.
            minimum_interval = 1.0 / float(self.sample_fps_limit)
            if last_kept_time is not None and \
                    capture_time - last_kept_time < minimum_interval * 0.80:
                continue

            minimap_rect = getattr(self, "_locked_minimap_rect", None)
            frame_size = getattr(self, "_minimap_lock_frame_size", None)
            if self.is_first_frame or minimap_rect is None or frame_size is None:
                continue

            try:
                working_frame, geometry = preprocess_capture_frame(
                    frame,
                    getattr(self, "_cfg_reference", self.cfg),
                    window_title=getattr(capture, "window_title", ""),
                    capture_profile=capture_profile_override(capture),
                )
                if tuple(geometry["output_size"]) != tuple(frame_size):
                    raise ValueError(
                        "capture frame geometry changed during calibration: "
                        f"{geometry['output_size']} != {frame_size}"
                    )
                x, y, width, height = map(int, minimap_rect)
                minimap = working_frame[y:y + height, x:x + width]
                minimap_cfg = self.cfg["minimap"]
                player_local = get_player_location_on_minimap(
                    minimap,
                    minimap_player_color=minimap_cfg["player_color"],
                    color_tolerance=minimap_cfg.get(
                        "player_color_tolerance", 0
                    ),
                    min_component_area=minimap_cfg.get(
                        "player_min_component_area", 4
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                now = time.monotonic()
                if now - self._last_sampler_error_at >= 2.0:
                    logger.warning(
                        f"[rope-calibration] Hero sampler skipped frame: {exc}"
                    )
                    self._last_sampler_error_at = now
                continue

            if player_local is None:
                continue
            map_offset = tuple(map(int, self.loc_minimap_global))
            player_global = (
                map_offset[0] + int(player_local[0]),
                map_offset[1] + int(player_local[1]),
            )
            last_kept_time = capture_time
            if not self.is_enable:
                continue
            sample = self._current_sample(capture_time, player_global)
            with self._sample_lock:
                self._sample_history.append(sample)
                self._sample_timestamps.append(float(capture_time))

    def _actual_sample_fps(self):
        with self._sample_lock:
            timestamps = list(self._sample_timestamps)
        intervals = [
            current - previous
            for previous, current in zip(timestamps, timestamps[1:])
            if current > previous
        ]
        if not intervals:
            return 0.0
        return 1.0 / float(median(intervals))

    def _finish_trial(self, label, label_time):
        jump_event = dict(self._active_jump_event)
        jump_time = float(jump_event["t"])
        with self._sample_lock:
            samples = [
                dict(sample) for sample in self._sample_history
                if jump_time - self._pre_roll_s
                <= float(sample["t"]) <= label_time
            ]
        trial = {
            "trial": len(self._trials) + 1,
            "label": label,
            "jump_event": jump_event,
            "label_time": round(float(label_time), 6),
            "samples": samples,
        }
        trial["metrics"] = infer_trial_metrics(trial)
        self._trials.append(trial)

        with self.output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(
                json.dumps(trial, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
        summary = summarize_trials(self._trials)
        temporary_summary = self.summary_path.with_suffix(".json.tmp")
        temporary_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_summary.replace(self.summary_path)

        if self.img_route is not None:
            preview_path = self.preview_dir / (
                f"trial_{trial['trial']:02d}_{label}.png"
            )
            if not cv2.imwrite(str(preview_path), self.img_route):
                logger.warning(
                    f"[rope-calibration] Unable to save {preview_path}"
                )
            self.img_route = self.remove_color_code_pixels(self.img_map.copy())
            self.break_route_segment()

        metrics = trial["metrics"]
        logger.info(
            f"[rope-calibration] Trial {trial['trial']}={label}; "
            f"direction={metrics['approach_direction']}, "
            f"jump=({metrics['jump_x']}, {metrics['jump_y']}), "
            f"lead={metrics['lead_distance_px']}px, "
            f"sample_FPS={metrics['actual_sample_fps']}"
        )
        self._active_jump_event = None
        with self._sample_lock:
            self._sample_history.clear()

    def update_info_on_img_frame_debug(self):
        super().update_info_on_img_frame_debug()
        active = "WAIT LABEL" if self._active_jump_event is not None else "READY"
        lines = [
            f"ROPE CALIBRATION: {active}",
            "F5=SUCCESS  F6=FAILURE  F7=DISCARD",
            f"HERO SAMPLE FPS: {self._actual_sample_fps():.1f}",
            f"LABELLED TRIALS: {len(self._trials)}",
        ]
        for index, line in enumerate(lines):
            cv2.putText(
                self.img_frame_debug,
                line,
                (10, 35 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def run_once(self):
        result = super().run_once()
        if not self.is_enable and self._active_jump_event is None:
            with self._sample_lock:
                self._sample_history.clear()
        self._drain_events()
        return result

    def stop(self):
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        if getattr(self, "_active_jump_event", None) is not None:
            logger.warning(
                "[rope-calibration] Unlabelled final attempt was not saved"
            )
        sampler_stop = getattr(self, "_sampler_stop", None)
        if sampler_stop is not None:
            sampler_stop.set()
        sampler_thread = getattr(self, "_sampler_thread", None)
        if sampler_thread is not None \
                and sampler_thread is not threading.current_thread():
            sampler_thread.join(timeout=2.0)
        super().stop()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Record labelled manual rope-jump calibration trials"
    )
    parser.add_argument(
        "--new_map",
        default="",
        help="fresh output map/session name (auto-generated when omitted)",
    )
    parser.add_argument(
        "--cfg",
        default="custom",
        help="custom config suffix from config/config_<name>.yaml",
    )
    parser.add_argument(
        "--map",
        default="minimaps/forest_floor/map.png",
        help="existing clean map used for minimap alignment",
    )
    parser.add_argument(
        "--output",
        default="",
        help="optional JSONL output path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Hero trajectory sampling target FPS",
    )
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=1.5,
        help="seconds of approach trajectory kept before Jump",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.new_map:
        source_map_name = Path(args.map).parent.name or "map"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.new_map = f"{source_map_name}_rope_calibration_{timestamp}"

    recorder = None
    try:
        recorder = RopeJumpCalibrator(args)
        while True:
            started_at = time.time()
            recorder.run_once()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            frame_duration = time.time() - started_at
            target_duration = 1.0 / recorder.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)
    except KeyboardInterrupt:
        logger.info("[rope-calibration] Interrupted by user")
    except Exception as exc:
        logger.error(f"[rope-calibration] Failed: {exc}")
        raise
    finally:
        if recorder is not None:
            recorder.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
