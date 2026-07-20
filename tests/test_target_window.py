import unittest
from unittest.mock import patch

from src.utils import common
from src.utils.common import (
    get_game_window_title_by_token,
    list_visible_window_titles,
)


def _make_enum(titles, visible=None):
    '''
    Build a fake win32gui.EnumWindows that yields the provided titles.
    `visible` optionally maps hwnd index -> bool (defaults to all visible).
    '''
    def fake_enum(callback, extra):
        for hwnd, title in enumerate(titles):
            callback(hwnd, extra)

    def fake_get_text(hwnd):
        return titles[hwnd]

    def fake_is_visible(hwnd):
        if visible is None:
            return True
        return visible.get(hwnd, True)

    return fake_enum, fake_get_text, fake_is_visible


class GetWindowTitleTests(unittest.TestCase):
    def test_substring_match(self):
        titles = ["Some App", "MapleStory Worlds-Artale", "Notepad"]
        enum, get_text, is_vis = _make_enum(titles)
        with patch.object(common.win32gui, "EnumWindows", enum), \
             patch.object(common.win32gui, "GetWindowText", get_text), \
             patch.object(common.win32gui, "IsWindowVisible", is_vis):
            self.assertEqual(
                get_game_window_title_by_token("artale"),
                "MapleStory Worlds-Artale",
            )

    def test_exact_match_required(self):
        titles = ["MyGame - Level 5", "MyGame"]
        enum, get_text, is_vis = _make_enum(titles)
        with patch.object(common.win32gui, "EnumWindows", enum), \
             patch.object(common.win32gui, "GetWindowText", get_text), \
             patch.object(common.win32gui, "IsWindowVisible", is_vis):
            # Exact match returns only the exact one
            self.assertEqual(
                get_game_window_title_by_token("MyGame", exact_match=True),
                "MyGame",
            )
            # Exact match with no exact candidate returns None
            self.assertIsNone(
                get_game_window_title_by_token("Level", exact_match=True)
            )

    def test_exact_match_preferred_in_substring_mode(self):
        titles = ["Editor Pro", "Editor"]
        enum, get_text, is_vis = _make_enum(titles)
        with patch.object(common.win32gui, "EnumWindows", enum), \
             patch.object(common.win32gui, "GetWindowText", get_text), \
             patch.object(common.win32gui, "IsWindowVisible", is_vis):
            # Even in substring mode, an exact match wins over a mere substring
            self.assertEqual(get_game_window_title_by_token("Editor"), "Editor")

    def test_invisible_windows_skipped(self):
        titles = ["Hidden Game", "Visible Game"]
        enum, get_text, is_vis = _make_enum(titles, visible={0: False, 1: True})
        with patch.object(common.win32gui, "EnumWindows", enum), \
             patch.object(common.win32gui, "GetWindowText", get_text), \
             patch.object(common.win32gui, "IsWindowVisible", is_vis):
            self.assertEqual(
                get_game_window_title_by_token("Game"), "Visible Game"
            )

    def test_empty_token_returns_none(self):
        self.assertIsNone(get_game_window_title_by_token(""))
        self.assertIsNone(get_game_window_title_by_token(None))


class ListWindowsTests(unittest.TestCase):
    def test_list_dedups_and_sorts(self):
        titles = ["Zeta", "alpha", "Zeta", "Beta"]
        enum, get_text, is_vis = _make_enum(titles)
        with patch.object(common, "is_mac", return_value=False), \
             patch.object(common.win32gui, "EnumWindows", enum), \
             patch.object(common.win32gui, "GetWindowText", get_text), \
             patch.object(common.win32gui, "IsWindowVisible", is_vis):
            result = list_visible_window_titles()
        self.assertEqual(result, ["alpha", "Beta", "Zeta"])


if __name__ == "__main__":
    unittest.main()
