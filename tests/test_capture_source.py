import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.input.CaptureSource import (
    DIRECTSHOW_SOURCE,
    WINDOW_SOURCE,
    capture_profile_override,
    create_capture_source,
    resolve_capture_source,
)


class CaptureSourceFactoryTests(unittest.TestCase):
    def test_missing_capture_config_preserves_window_capture_default(self):
        capture = SimpleNamespace(window_title="MapleStory Worlds")
        window_capture = Mock(return_value=capture)

        actual = create_capture_source(
            {},
            window_capture_cls=window_capture,
        )

        self.assertIs(actual, capture)
        window_capture.assert_called_once_with({})
        self.assertEqual(resolve_capture_source({}), WINDOW_SOURCE)
        self.assertEqual(capture.capture_profile, "direct")

    def test_directshow_and_capture_card_alias_select_gc573_backend(self):
        for source in ("directshow", "capture_card"):
            with self.subTest(source=source):
                cfg = {"capture": {"source": source}}
                capture = SimpleNamespace()
                window_capture = Mock()
                directshow_capture = Mock(return_value=capture)

                with patch(
                    "src.input.CaptureSource.platform.system",
                    return_value="Windows",
                ):
                    actual = create_capture_source(
                        cfg,
                        window_capture_cls=window_capture,
                        directshow_capture_cls=directshow_capture,
                    )
                    resolved = resolve_capture_source(cfg)

                self.assertIs(actual, capture)
                self.assertEqual(resolved, DIRECTSHOW_SOURCE)
                directshow_capture.assert_called_once_with(cfg)
                window_capture.assert_not_called()

    def test_static_test_image_forces_legacy_window_fixture(self):
        cfg = {"capture": {"source": "directshow"}}
        capture = SimpleNamespace(window_title="PotPlayer")
        window_capture = Mock(return_value=capture)
        directshow_capture = Mock()

        with patch("src.input.CaptureSource.platform.system", return_value="Windows"):
            actual = create_capture_source(
                cfg,
                test_image_name="fixture.png",
                window_capture_cls=window_capture,
                directshow_capture_cls=directshow_capture,
            )

        self.assertIs(actual, capture)
        self.assertEqual(
            resolve_capture_source(cfg, "fixture.png"), WINDOW_SOURCE
        )
        window_capture.assert_called_once_with(cfg, "fixture.png")
        directshow_capture.assert_not_called()
        self.assertEqual(capture.capture_profile, "direct")

    def test_non_windows_platform_falls_back_to_window_capture(self):
        cfg = {
            "capture": {"source": "directshow"},
            "game_window": {"capture_profile": "capture_card"},
        }
        capture = SimpleNamespace(window_title="MapleStory Worlds")
        window_capture = Mock(return_value=capture)
        directshow_capture = Mock()

        with patch(
            "src.input.CaptureSource.platform.system",
            return_value="Darwin",
        ):
            actual = create_capture_source(
                cfg,
                window_capture_cls=window_capture,
                directshow_capture_cls=directshow_capture,
            )
            source = resolve_capture_source(cfg)

        self.assertIs(actual, capture)
        self.assertEqual(source, WINDOW_SOURCE)
        self.assertEqual(capture.capture_profile, "direct")
        window_capture.assert_called_once_with(cfg)
        directshow_capture.assert_not_called()

    def test_window_source_replaces_capture_card_profile_using_title(self):
        for configured, title, expected in (
            ("capture_card", "PotPlayer 64 bit", "potplayer"),
            ("directshow", "MapleStory Worlds", "direct"),
            ("capture-card", "PotPlayer Preview", "potplayer"),
        ):
            with self.subTest(configured=configured, title=title):
                cfg = {
                    "capture": {"source": "window"},
                    "game_window": {"capture_profile": configured},
                }
                capture = SimpleNamespace(window_title=title)

                create_capture_source(
                    cfg,
                    window_capture_cls=Mock(return_value=capture),
                )

                self.assertEqual(capture.capture_profile, expected)

    def test_window_source_keeps_explicit_non_card_profile(self):
        cfg = {
            "capture": {"source": "window"},
            "game_window": {"capture_profile": "potplayer"},
        }
        capture = SimpleNamespace(window_title="MapleStory Worlds")

        create_capture_source(
            cfg,
            window_capture_cls=Mock(return_value=capture),
        )

        self.assertEqual(capture.capture_profile, "potplayer")

    def test_dynamic_mock_title_is_not_treated_as_text(self):
        capture = Mock()

        create_capture_source(
            {},
            window_capture_cls=Mock(return_value=capture),
        )

        self.assertEqual(capture.capture_profile, "direct")

    def test_invalid_source_fails_with_actionable_message(self):
        with self.assertRaisesRegex(ValueError, "capture.source"):
            resolve_capture_source({"capture": {"source": "potplayer"}})

    def test_capture_profile_override_accepts_only_real_strings(self):
        self.assertEqual(
            capture_profile_override(
                SimpleNamespace(capture_profile=" capture_card ")
            ),
            "capture_card",
        )
        self.assertIsNone(capture_profile_override(Mock()))
        self.assertIsNone(
            capture_profile_override(
                SimpleNamespace(capture_profile="   ")
            )
        )


if __name__ == "__main__":
    unittest.main()
