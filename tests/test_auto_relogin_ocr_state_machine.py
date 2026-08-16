import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.vision.auto_relogin_ocr import OcrTextMatch, RapidOcrError


FRAME_HEIGHT = 200
FRAME_WIDTH = 400
PAGE_LOCATIONS = {
    "disconnect": (10, 20),
    "connect": (20, 30),
    "world": (30, 40),
    "channel": (50, 60),
    "character": (70, 80),
}
OCR_TARGET = (210, 110)
CHANNEL_POINT = (200, 140)


def make_match(center=OCR_TARGET):
    x, y = center
    return OcrTextMatch(
        text="target",
        normalized_text="target",
        score=0.99,
        box=(
            (x - 10, y - 5),
            (x + 10, y - 5),
            (x + 10, y + 5),
            (x - 10, y + 5),
        ),
        center=center,
    )


def make_bot(*, click_results=True):
    bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
    target_region = [0, 0, FRAME_WIDTH, FRAME_HEIGHT]
    bot.cfg = {
        "bot": {"mode": "normal"},
        "esp32_hid": {"remote_target": True},
        "game_window": {"title_bar_height": 34},
        "auto_relogin": {
            "enable": True,
            "flow_template_reference_size": [FRAME_HEIGHT, FRAME_WIDTH],
            "template_threshold": 0.03,
            "confirm_frames": 2,
            "confirm_seconds": 0.0,
            "cancel_confirm_misses": 2,
            "input_retry_delay": 0.0,
            "retry_cooldown": 0.0,
            "step_timeout": 60.0,
            "game_ready_timeout": 20.0,
            "max_recovery_duration": 300.0,
            "max_step_attempts": 5,
            "game_ready_confirm_frames": 2,
            "mouse_click_duration": 0.05,
            "channel_click_count": 2,
            "channel_double_click_interval": 0.0,
            "remote_mouse_mode": "absolute",
            "remote_confirm_key": "enter",
            "channel_points": [list(CHANNEL_POINT)],
            "page_anchor_points": {
                "disconnect": list(PAGE_LOCATIONS["disconnect"]),
                "channel": list(PAGE_LOCATIONS["channel"]),
            },
            "ocr": {
                "enable": True,
                "min_score": 0.85,
                "box_threshold": 0.30,
                "confirm_frames": 2,
                "max_center_drift": [24, 24],
                "max_frame_age": 1.0,
                "targets": {
                    "disconnect": {
                        "texts": ["disconnect-target"],
                        "search_region": target_region,
                        "match_mode": "contains",
                        "action": "enter",
                    },
                    "connect": {
                        "texts": ["connect-target"],
                        "search_region": target_region,
                        "match_mode": "exact",
                        "action": "click",
                    },
                    "world": {
                        "texts": ["4.漂漂猪"],
                        "search_region": target_region,
                        "match_mode": "exact",
                        "action": "click",
                    },
                    "channel": {
                        "texts": ["漂漂猪"],
                        "search_region": target_region,
                        "match_mode": "exact",
                        "action": "fixed_click",
                    },
                    "character": {
                        "texts": ["开始游戏"],
                        "search_region": target_region,
                        "match_mode": "exact",
                        "action": "click",
                    },
                },
            },
        },
        "health_monitor": {"enable": True},
    }
    bot.is_terminated = False
    bot.is_disable_control = False
    bot.img_frame = np.zeros(
        (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )
    bot.img_frame_debug = None
    bot.capture = SimpleNamespace(
        last_frame_time=1.0,
        window_title="MapleStory",
        is_static_frame=False,
    )
    bot._current_capture_frame_token = None
    bot._visible_page = None

    def match_visible_page(page):
        if page == bot._visible_page:
            return PAGE_LOCATIONS[page]
        return None

    bot._match_auto_relogin_page = Mock(side_effect=match_visible_page)
    bot._auto_relogin_current_gameplay_evidence = Mock(return_value=None)
    bot.click_game_ui = Mock(return_value=True)
    bot._reset_ladder_route_hold = Mock()
    bot._reset_stationary_jump_proximity = Mock()
    bot._reset_rope_climb = Mock()
    bot._reset_portal_sweep = Mock()
    click_session_recovery_point = Mock()
    if isinstance(click_results, (list, tuple)):
        click_session_recovery_point.side_effect = click_results
    else:
        click_session_recovery_point.return_value = click_results
    bot.kb = SimpleNamespace(
        is_enable=True,
        is_terminated=False,
        is_need_force_heal=False,
        set_command=Mock(),
        release_all_key=Mock(),
        suspend_automation_for_session_recovery=Mock(),
        resume_automation_after_session_recovery=Mock(),
        press_session_recovery_key=Mock(return_value=True),
        move_session_recovery_mouse=Mock(return_value=True),
        click_session_recovery_mouse=Mock(return_value=True),
        click_session_recovery_point=click_session_recovery_point,
    )
    bot.health_monitor = SimpleNamespace(
        enabled=True,
        disable=Mock(),
        enable=Mock(),
    )
    bot.fsm = SimpleNamespace(set_init_state=Mock())
    bot._auto_relogin_ocr_locator = Mock()
    bot._auto_relogin_ocr_gate = None
    bot._auto_relogin_ocr_gate_signature = None
    bot._reset_auto_relogin_runtime()
    return bot


def show_and_check(bot, page, *, captured_at, now=None):
    captured_at = float(captured_at)
    now = captured_at if now is None else float(now)
    bot._visible_page = page
    bot.capture.last_frame_time = captured_at
    bot._current_capture_frame_token = ("capture", captured_at)
    with patch(
        "src.engine.MapleStoryAutoLevelUp.time.monotonic",
        return_value=now,
    ):
        return bot._check_auto_relogin_screen()


def test_ocr_pages_require_two_fresh_stable_ocr_frames_after_page_detection():
    cases = (
        ("connect", "waiting_page", "world"),
        ("world", "waiting_page", "channel"),
        ("character", "waiting_game", None),
    )
    for page, expected_state, expected_next_page in cases:
        bot = make_bot()
        bot._auto_relogin_ocr_locator.locate.side_effect = (
            make_match((210, 110)),
            make_match((212, 111)),
        )

        # The first page-template frame only starts page confirmation.
        assert show_and_check(bot, page, captured_at=10.0)
        bot._auto_relogin_ocr_locator.locate.assert_not_called()
        bot.kb.click_session_recovery_point.assert_not_called()

        # The page is now confirmed, but this is only OCR target frame one.
        assert show_and_check(bot, page, captured_at=10.1)
        assert bot._auto_relogin_ocr_locator.locate.call_count == 1
        bot.kb.click_session_recovery_point.assert_not_called()

        # A second distinct, stable OCR frame authorizes one absolute click.
        assert show_and_check(bot, page, captured_at=10.2)
        bot.kb.click_session_recovery_point.assert_called_once_with(
            212,
            111,
            FRAME_WIDTH,
            FRAME_HEIGHT,
            button="left",
            duration=0.05,
        )
        bot.kb.press_session_recovery_key.assert_not_called()
        assert bot._auto_relogin_state == expected_state
        assert bot._auto_relogin_expected_page == expected_next_page


def test_disconnect_requires_stable_ocr_then_sends_enter_only():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match((210, 110)),
        make_match((212, 111)),
    )

    assert show_and_check(bot, "disconnect", captured_at=20.0)
    bot._auto_relogin_ocr_locator.locate.assert_not_called()
    bot.kb.press_session_recovery_key.assert_not_called()

    assert show_and_check(bot, "disconnect", captured_at=20.1)
    bot.kb.press_session_recovery_key.assert_not_called()

    assert show_and_check(bot, "disconnect", captured_at=20.2)

    bot.kb.press_session_recovery_key.assert_called_once_with("enter")
    bot.kb.click_session_recovery_point.assert_not_called()
    bot.click_game_ui.assert_not_called()
    assert bot._auto_relogin_ocr_locator.locate.call_count == 2
    assert bot._auto_relogin_expected_page == "connect"


def test_disconnect_ocr_miss_requires_two_new_matches_before_enter():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match(),
        None,
        make_match((211, 110)),
        make_match((212, 111)),
    )

    assert show_and_check(bot, "disconnect", captured_at=25.0)
    assert show_and_check(bot, "disconnect", captured_at=25.1)
    assert show_and_check(bot, "disconnect", captured_at=25.2)
    assert show_and_check(bot, "disconnect", captured_at=25.3)
    bot.kb.press_session_recovery_key.assert_not_called()

    assert show_and_check(bot, "disconnect", captured_at=25.4)
    bot.kb.press_session_recovery_key.assert_called_once_with("enter")
    bot.kb.click_session_recovery_point.assert_not_called()


def test_stale_disconnect_ocr_match_never_authorizes_enter():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match(),
        make_match((211, 110)),
        make_match((212, 111)),
        make_match((213, 111)),
    )

    assert show_and_check(bot, "disconnect", captured_at=26.0)
    assert show_and_check(bot, "disconnect", captured_at=26.1)
    assert show_and_check(
        bot, "disconnect", captured_at=26.2, now=27.3
    )
    bot.kb.press_session_recovery_key.assert_not_called()

    assert show_and_check(bot, "disconnect", captured_at=27.4)
    bot.kb.press_session_recovery_key.assert_not_called()
    assert show_and_check(bot, "disconnect", captured_at=27.5)
    bot.kb.press_session_recovery_key.assert_called_once_with("enter")
    bot.kb.click_session_recovery_point.assert_not_called()


def test_channel_requires_stable_ocr_then_uses_its_configured_point():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match((210, 110)),
        make_match((212, 111)),
    )

    assert show_and_check(bot, "channel", captured_at=30.0)
    assert show_and_check(bot, "channel", captured_at=30.1)
    bot.kb.click_session_recovery_point.assert_not_called()

    assert show_and_check(bot, "channel", captured_at=30.2)

    expected_click = call(
        CHANNEL_POINT[0],
        CHANNEL_POINT[1],
        FRAME_WIDTH,
        FRAME_HEIGHT,
        button="left",
        duration=0.05,
    )
    assert bot.kb.click_session_recovery_point.call_args_list == [
        expected_click,
        expected_click,
    ]
    bot.kb.press_session_recovery_key.assert_not_called()
    assert bot._auto_relogin_ocr_locator.locate.call_count == 2
    assert bot._auto_relogin_expected_page == "character"


def test_ocr_miss_breaks_stability_and_requires_two_new_matches():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match(),
        None,
        make_match((211, 110)),
        make_match((212, 111)),
    )

    assert show_and_check(bot, "connect", captured_at=40.0)
    assert show_and_check(bot, "connect", captured_at=40.1)
    assert show_and_check(bot, "connect", captured_at=40.2)
    assert show_and_check(bot, "connect", captured_at=40.3)
    bot.kb.click_session_recovery_point.assert_not_called()

    assert show_and_check(bot, "connect", captured_at=40.4)
    assert bot.kb.click_session_recovery_point.call_count == 1


def test_repeated_capture_token_cannot_supply_the_second_ocr_frame():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match(),
        make_match((211, 110)),
        make_match((212, 111)),
    )

    assert show_and_check(bot, "world", captured_at=50.0)
    assert show_and_check(bot, "world", captured_at=50.1)
    # Reprocessing the same captured image must not authorize a click.
    assert show_and_check(bot, "world", captured_at=50.1, now=50.2)
    bot.kb.click_session_recovery_point.assert_not_called()

    assert show_and_check(bot, "world", captured_at=50.3)
    assert bot.kb.click_session_recovery_point.call_count == 1


def test_expired_ocr_authorization_is_not_clicked_and_is_reset():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match(),
        make_match((211, 110)),
        make_match((212, 111)),
        make_match((213, 111)),
    )

    assert show_and_check(bot, "character", captured_at=60.0)
    assert show_and_check(bot, "character", captured_at=60.1)
    # This stable second OCR observation is already older than max_frame_age.
    assert show_and_check(
        bot, "character", captured_at=60.2, now=61.3
    )
    bot.kb.click_session_recovery_point.assert_not_called()

    # The stale authorization was consumed/reset, so one fresh frame is not
    # enough; two new fresh OCR frames are required.
    assert show_and_check(bot, "character", captured_at=61.4)
    bot.kb.click_session_recovery_point.assert_not_called()
    assert show_and_check(bot, "character", captured_at=61.5)
    assert bot.kb.click_session_recovery_point.call_count == 1


def test_a_stale_first_ocr_frame_cannot_count_toward_confirmation():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match(),
        make_match((211, 110)),
        make_match((212, 111)),
    )

    assert show_and_check(bot, "connect", captured_at=70.0)
    # The first OCR observation is stale and must not seed the stable gate.
    assert show_and_check(bot, "connect", captured_at=70.1, now=71.2)
    assert show_and_check(bot, "connect", captured_at=71.3)
    bot.kb.click_session_recovery_point.assert_not_called()

    assert show_and_check(bot, "connect", captured_at=71.4)
    assert bot.kb.click_session_recovery_point.call_count == 1


def test_failed_absolute_send_must_reacquire_two_fresh_ocr_frames():
    bot = make_bot(click_results=[False, True])
    bot._auto_relogin_ocr_locator.locate.side_effect = (
        make_match(),
        make_match((211, 110)),
        make_match((212, 111)),
        make_match((213, 111)),
    )

    assert show_and_check(bot, "connect", captured_at=80.0)
    assert show_and_check(bot, "connect", captured_at=80.1)
    assert show_and_check(bot, "connect", captured_at=80.2)
    assert bot.kb.click_session_recovery_point.call_count == 1

    # Sending failed, and the dispatch consumed the old OCR authorization.
    assert show_and_check(bot, "connect", captured_at=80.3)
    assert bot.kb.click_session_recovery_point.call_count == 1

    assert show_and_check(bot, "connect", captured_at=80.4)
    assert bot.kb.click_session_recovery_point.call_count == 2
    assert bot._auto_relogin_state == "waiting_page"
    assert bot._auto_relogin_expected_page == "world"


def test_ocr_backend_failure_stops_recovery_without_clicking():
    bot = make_bot()
    bot._auto_relogin_ocr_locator.locate.side_effect = RapidOcrError(
        "model unavailable"
    )

    assert show_and_check(bot, "connect", captured_at=90.0)
    assert show_and_check(bot, "connect", captured_at=90.1)

    assert bot._auto_relogin_state == "failed"
    bot.kb.click_session_recovery_point.assert_not_called()
    bot.kb.press_session_recovery_key.assert_not_called()


def load_tests(loader, standard_tests, pattern):
    """Expose the focused function tests to the repository's unittest CI."""
    del loader, standard_tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(
                value, description=name
            ))
    return suite
