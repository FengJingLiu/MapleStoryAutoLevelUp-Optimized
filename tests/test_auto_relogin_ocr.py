import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.vision.auto_relogin_ocr import (
    OcrTextMatch,
    RapidOcrTextLocator,
    StableOcrTargetGate,
    is_chinese_ocr_target,
)


FRAME_HEIGHT = 2160
FRAME_WIDTH = 3840


def _ocr_output(boxes, texts, scores):
    return SimpleNamespace(boxes=boxes, txts=texts, scores=scores)


def _match(center, *, text="连接", score=0.98):
    x, y = center
    return OcrTextMatch(
        text=text,
        normalized_text=text,
        score=score,
        box=(
            (x - 10, y - 5),
            (x + 10, y - 5),
            (x + 10, y + 5),
            (x - 10, y + 5),
        ),
        center=(x, y),
    )


def _remote_bot(frame):
    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    bot.cfg = {
        "esp32_hid": {"remote_target": True},
        "auto_relogin": {"mouse_click_duration": 0.05},
    }
    bot.img_frame = frame
    bot.is_terminated = False
    bot.kb = SimpleNamespace(
        click_session_recovery_point=Mock(return_value=True)
    )
    return bot


class TestRapidOcrTextLocator(unittest.TestCase):
    def test_engine_is_explicitly_pinned_to_chinese_models(self):
        engine = Mock()
        with patch("rapidocr.RapidOCR", return_value=engine) as factory:
            locator = RapidOcrTextLocator()
            assert locator._get_engine() is engine

        factory.assert_called_once_with(params={
            "Det.lang_type": "ch",
            "Cls.lang_type": "ch",
            "Rec.lang_type": "ch",
            "Global.log_level": "warning",
        })

    def test_only_chinese_targets_without_hangul_or_kana_are_allowed(self):
        assert is_chinese_ocr_target("连接")
        assert is_chinese_ocr_target("4.漂漂猪")
        assert not is_chinese_ocr_target("연결")
        assert not is_chinese_ocr_target("せつぞく")
        assert not is_chinese_ocr_target("Connect")

        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        engine = Mock()
        with self.assertRaisesRegex(ValueError, "Chinese Han"):
            RapidOcrTextLocator(engine=engine).locate(
                frame,
                (0, 0, 200, 100),
                ("연결",),
                min_score=0.85,
            )
        engine.assert_not_called()

    def test_roi_box_is_returned_in_full_capture_coordinates(self):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        region = (2000, 900, 3000, 1600)
        # RapidOCR coordinates are relative to the cropped ROI.  This box has
        # a full-frame center of exactly (2500, 1200).
        local_box = np.array([
            [400.0, 200.0],
            [600.0, 200.0],
            [600.0, 400.0],
            [400.0, 400.0],
        ])
        engine = Mock(return_value=_ocr_output(
            boxes=np.array([local_box]),
            texts=(" 连 接 ",),
            scores=(0.98,),
        ))

        match = RapidOcrTextLocator(engine=engine).locate(
            frame,
            region,
            ("连接",),
            min_score=0.85,
        )

        assert match is not None
        assert match.normalized_text == "连接"
        assert match.box == (
            (2400, 1100),
            (2600, 1100),
            (2600, 1300),
            (2400, 1300),
        )
        assert match.center == (2500, 1200)
        submitted_roi = engine.call_args.args[0]
        assert submitted_roi.shape == (700, 1000, 3)
        assert submitted_roi.flags.c_contiguous
        assert engine.call_args.kwargs == {
            "use_cls": False,
            "return_word_box": False,
            "text_score": 0.85,
            "box_thresh": 0.3,
        }

    def test_duplicate_matching_labels_are_rejected_instead_of_ranked(self):
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)
        boxes = np.array([
            [[10, 10], [110, 10], [110, 50], [10, 50]],
            [[300, 200], [400, 200], [400, 240], [300, 240]],
        ], dtype=np.float64)
        engine = Mock(return_value=_ocr_output(
            boxes=boxes,
            texts=("开始游戏", "开始游戏"),
            scores=(0.99, 0.98),
        ))

        match = RapidOcrTextLocator(engine=engine).locate(
            frame,
            (0, 0, 1000, 600),
            ("开始游戏",),
            min_score=0.85,
        )

        assert match is None

    def test_low_confidence_target_is_rejected(self):
        frame = np.zeros((600, 1000, 3), dtype=np.uint8)
        box = np.array([
            [10, 10], [110, 10], [110, 50], [10, 50]
        ], dtype=np.float64)
        engine = Mock(return_value=_ocr_output(
            boxes=np.array([box]),
            texts=("连接",),
            scores=(0.849,),
        ))

        match = RapidOcrTextLocator(engine=engine).locate(
            frame,
            (0, 0, 1000, 600),
            ("连接",),
            min_score=0.85,
        )

        assert match is None

    def test_invalid_backend_boxes_fail_closed(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        invalid_boxes = (
            # Non-finite coordinate.
            [[10, 10], [30, 10], [30, np.nan], [10, 30]],
            # Extends outside the submitted ROI.
            [[10, 10], [200, 10], [200, 30], [10, 30]],
            # Zero-width quadrilateral.
            [[10, 10], [10, 20], [10, 30], [10, 40]],
            # Not a four-corner OCR box.
            [[10, 10], [30, 10], [30, 30]],
        )

        for invalid_box in invalid_boxes:
            engine = Mock(return_value=_ocr_output(
                boxes=[invalid_box],
                texts=("连接",),
                scores=(0.99,),
            ))

            match = RapidOcrTextLocator(engine=engine).locate(
                frame,
                (0, 0, 200, 100),
                ("连接",),
                min_score=0.85,
            )

            assert match is None


class TestStableOcrTargetGate(unittest.TestCase):
    def test_same_capture_token_neither_confirms_nor_changes_candidate(self):
        gate = StableOcrTargetGate(
            confirm_frames=2,
            max_center_drift=(24, 24),
        )
        first_token = ("capture", 10.0)

        assert gate.observe("connect", first_token, _match((100, 100))) is None
        # If this duplicate-frame observation changed the candidate, the next
        # legitimate point would exceed the drift threshold and be reset.
        assert gate.observe("connect", first_token, _match((500, 500))) is None
        assert gate.observe(
            "connect", ("capture", 10.1), _match((110, 105))
        ) == (110, 105)

    def test_same_token_missing_result_revokes_candidate(self):
        gate = StableOcrTargetGate(confirm_frames=2)
        first_token = ("capture", 20.0)

        assert gate.observe(
            "connect", first_token, _match((100, 100))
        ) is None
        assert gate.observe("connect", first_token, None) is None
        assert gate.observe(
            "connect", ("capture", 20.1), _match((101, 100))
        ) is None
        assert gate.observe(
            "connect", ("capture", 20.2), _match((102, 101))
        ) == (102, 101)

    def test_missing_match_breaks_stability_and_does_not_click(self):
        gate = StableOcrTargetGate(confirm_frames=2)
        click = Mock()
        observations = (
            (("capture", 1.0), _match((100, 100))),
            (("capture", 2.0), None),
            (("capture", 3.0), _match((101, 99))),
        )

        for token, match in observations:
            point = gate.observe("connect", token, match)
            if point is not None:
                click(point)

        click.assert_not_called()

    def test_page_change_breaks_stability_and_does_not_click(self):
        gate = StableOcrTargetGate(confirm_frames=2)
        click = Mock()

        for page, token in (
            ("connect", ("capture", 1.0)),
            ("world", ("capture", 2.0)),
        ):
            point = gate.observe(page, token, _match((100, 100)))
            if point is not None:
                click(point)

        click.assert_not_called()

    def test_excessive_center_drift_restarts_confirmation(self):
        gate = StableOcrTargetGate(
            confirm_frames=2,
            max_center_drift=(24, 24),
        )

        assert gate.observe(
            "connect", ("capture", 1.0), _match((100, 100))
        ) is None
        assert gate.observe(
            "connect", ("capture", 2.0), _match((125, 100))
        ) is None
        assert gate.observe(
            "connect", ("capture", 3.0), _match((130, 103))
        ) == (130, 103)

    def test_two_fresh_stable_frames_release_current_center_once(self):
        gate = StableOcrTargetGate(
            confirm_frames=2,
            max_center_drift=(24, 24),
        )

        assert gate.observe(
            "character", ("capture", 1.0), _match((2500, 1200))
        ) is None
        assert gate.observe(
            "character", ("capture", 2.0), _match((2504, 1197))
        ) == (2504, 1197)


class TestAutoReloginOcrClickCoordinates(unittest.TestCase):
    def test_page_classification_and_action_share_one_ocr_scan_per_frame(self):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        token = ("capture", time.monotonic())
        locator = SimpleNamespace(
            recognize=Mock(return_value=(
                _match((2100, 1120), text="\u8fde\u63a5"),
            )),
            locate=Mock(side_effect=AssertionError(
                "the action must reuse the page-classification OCR result"
            )),
        )
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "auto_relogin": {
                "flow_template_reference_size": [
                    FRAME_HEIGHT, FRAME_WIDTH
                ],
                "ocr": {
                    "enable": True,
                    "min_score": 0.85,
                    "box_threshold": 0.3,
                    "confirm_frames": 1,
                    "max_center_drift": [24, 24],
                    "max_frame_age": 5.0,
                    "targets": {
                        "connect": {
                            "texts": ["\u8fde\u63a5"],
                            "region_source": "configured",
                            "search_region": [1824, 912, 2360, 1234],
                            "match_mode": "exact",
                            "action": "click",
                        },
                    },
                },
            },
        }
        bot.img_frame = frame
        bot._current_capture_frame_token = token
        bot._auto_relogin_state = "waiting_page"
        bot._auto_relogin_expected_page = "connect"
        bot._auto_relogin_last_action_page = "disconnect"
        bot._auto_relogin_ocr_locator = locator
        bot._auto_relogin_ocr_gate = None
        bot._auto_relogin_ocr_gate_signature = None
        bot._auto_relogin_ocr_page_scan_token = None
        bot._auto_relogin_ocr_page_scan_matches = None
        bot._auto_relogin_ocr_page_matches = {}

        assert bot._find_known_auto_relogin_page() == (
            "connect", (2100, 1120)
        )
        # Reclassifying a page on the same capture must be cache-only.
        assert bot._find_known_auto_relogin_page() == (
            "connect", (2100, 1120)
        )
        assert bot._locate_stable_auto_relogin_ocr_target(
            "connect", (2100, 1120), token
        ) == (2100, 1120)
        locator.recognize.assert_called_once()
        locator.locate.assert_not_called()

    def test_ocr_bbox_center_reaches_click_sender_without_second_scaling(self):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        region = (2000, 900, 3000, 1600)
        local_box = np.array([
            [400, 200], [600, 200], [600, 400], [400, 400]
        ], dtype=np.float64)
        engine = Mock(return_value=_ocr_output(
            boxes=np.array([local_box]),
            texts=("开始游戏",),
            scores=(0.99,),
        ))
        locator = RapidOcrTextLocator(engine=engine)
        gate = StableOcrTargetGate(confirm_frames=2)
        bot = _remote_bot(frame)

        first_match = locator.locate(
            frame, region, ("开始游戏",), min_score=0.85
        )
        second_match = locator.locate(
            frame, region, ("开始游戏",), min_score=0.85
        )
        assert gate.observe(
            "character", ("capture", 1.0), first_match
        ) is None
        point = gate.observe(
            "character", ("capture", 2.0), second_match
        )

        assert point == (2500, 1200)
        assert bot._send_auto_relogin_click(
            point, "auto_relogin_start_game"
        )
        bot.kb.click_session_recovery_point.assert_called_once_with(
            2500,
            1200,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            button="left",
            duration=0.05,
        )
