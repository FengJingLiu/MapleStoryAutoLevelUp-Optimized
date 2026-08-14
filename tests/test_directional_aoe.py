import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.engine.MapleStoryAutoLevelUp import MapleStoryAutoBot
from src.states.patrol import PatrolState


def make_monster(center_x, center_y=100, width=20, height=20):
    return {
        "name": "mob",
        "position": (
            int(center_x - width // 2),
            int(center_y - height // 2),
        ),
        "size": (height, width),
        "confidence": 0.9,
        "score": 0.1,
    }


class DirectionalAoeDecisionTests(unittest.TestCase):
    @staticmethod
    def make_bot(
            monsters, *, threshold=3, aoe_cooldown=1.5,
            knockback_enabled=False, knockback_distance=40,
            knockback_cooldown=1.0):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cfg = {
            "bot": {"attack": "directional"},
            "directional_attack": {
                "range_x": 100,
                "range_y": 80,
                "cooldown": 0.5,
            },
            "directional_aoe": {
                "enable": True,
                "min_monsters": threshold,
                "range_x": 140,
                "range_y": 100,
                "cooldown": aoe_cooldown,
                "attack_recovery_delay": 1.0,
            },
            "power_knockback": {
                "enable": knockback_enabled,
                "trigger_distance_x": knockback_distance,
                "range_y": 80,
                "cooldown": knockback_cooldown,
                "attack_recovery_delay": 0.9,
            },
            "monster_detect": {
                "backend": "yolo",
                "search_box_margin": 10,
                "max_mob_area_trigger": 400,
            },
        }
        bot.img_frame = np.zeros((200, 400, 3), dtype=np.uint8)
        bot.loc_player = (200, 100)
        bot.screen_player_location_valid = True
        bot.monsters_info = {}
        bot.monsters = []
        bot.cmd_move_x = "none"
        bot.cmd_move_y = "none"
        bot.cmd_action = "none"
        bot.t_last_attack = 0.0
        bot.t_last_directional_aoe = 0.0
        bot.t_last_power_knockback = 0.0
        bot.kb = SimpleNamespace(
            cached_facing="right",
            is_attack_recovering=lambda: False,
        )
        bot.get_monsters_in_range = Mock(return_value=list(monsters))
        return bot

    def test_exact_threshold_uses_aoe_on_the_crowded_side(self):
        bot = self.make_bot([
            make_monster(90),
            make_monster(120),
            make_monster(150),
            make_monster(215),
        ])

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "directional_aoe")
        self.assertEqual(bot.cmd_move_x, "left")
        self.assertEqual(bot.t_last_directional_aoe, 10.0)
        self.assertEqual(bot.t_last_attack, 10.0)

    def test_below_threshold_keeps_normal_directional_attack(self):
        bot = self.make_bot([
            make_monster(150),
            make_monster(170),
        ])

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_both_sides_qualify_and_more_monsters_wins(self):
        bot = self.make_bot([
            make_monster(90),
            make_monster(120),
            make_monster(150),
            make_monster(235),
            make_monster(260),
            make_monster(285),
            make_monster(315),
        ])

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "directional_aoe")
        self.assertEqual(bot.cmd_move_x, "right")

    def test_aoe_cooldown_never_falls_back_to_normal_attack(self):
        bot = self.make_bot([
            make_monster(90),
            make_monster(120),
            make_monster(150),
        ])
        bot.t_last_attack = 8.0
        bot.t_last_directional_aoe = 9.5
        bot.cmd_action = "attack"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "none")
        self.assertEqual(bot.t_last_attack, 8.0)
        self.assertEqual(bot.t_last_directional_aoe, 9.5)

    def test_margin_only_monsters_do_not_count_toward_threshold(self):
        bot = self.make_bot([
            make_monster(55),
            make_monster(60),
            make_monster(65),
        ])

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "none")

    def test_search_roi_covers_the_larger_directional_aoe_range(self):
        bot = self.make_bot([])

        bot.update_cmd_by_mob_detection()

        bot.get_monsters_in_range.assert_called_once_with(
            (50, 0),
            (350, 200),
        )

    def test_crossing_box_is_counted_only_on_its_center_side(self):
        bot = self.make_bot([], threshold=1)
        crossing = make_monster(205, width=30)
        bot.monsters = [crossing]

        left = bot.get_monsters_in_attack_range(
            is_left=True,
            attack_type="directional_aoe",
        )
        right = bot.get_monsters_in_attack_range(
            is_left=False,
            attack_type="directional_aoe",
        )

        self.assertEqual(left, [])
        self.assertEqual(right, [crossing])

    def test_zero_overlap_is_never_counted_even_with_zero_threshold(self):
        bot = self.make_bot([], threshold=1)
        bot.cfg["monster_detect"]["max_mob_area_trigger"] = 0
        bot.monsters = [make_monster(45, width=10)]

        left = bot.get_monsters_in_attack_range(
            is_left=True,
            attack_type="directional_aoe",
        )

        self.assertEqual(left, [])

    def test_legacy_health_bar_duplicate_counts_as_one_monster(self):
        bot = self.make_bot([], threshold=2)
        bot.cfg["monster_detect"].update({
            "backend": "template",
            "with_enemy_hp_bar": True,
        })
        bot.monsters_info = {
            "mob": [(np.zeros((26, 37, 3), dtype=np.uint8), None)],
        }
        visual_monster = make_monster(150, width=37, height=26)
        health_bar_monster = {
            **make_monster(177, width=70, height=26),
            "name": "Health Bar",
            "score": 1.0,
        }
        bot.monsters = [
            visual_monster,
            health_bar_monster,
        ]

        left = bot.get_monsters_in_attack_range(
            is_left=True,
            attack_type="directional_aoe",
        )

        self.assertEqual(len(left), 1)

    def test_crossing_box_cannot_hide_a_valid_normal_attack_target(self):
        crossing = make_monster(200, width=80)
        valid_left = make_monster(150)
        bot = self.make_bot([crossing, valid_left])
        bot.cfg["directional_aoe"]["enable"] = False

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_fractional_directional_aoe_range_is_rejected(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        cfg = {
            "bot": {
                "mode": "normal",
                "attack": "directional",
            },
            "monster_detect": {"backend": "yolo"},
            "key": {"aoe_skill": "shift"},
            "directional_aoe": {
                "enable": True,
                "min_monsters": 3,
                "range_x": 140.5,
                "range_y": 100,
                "cooldown": 1.0,
                "attack_recovery_delay": 1.0,
            },
        }

        self.assertEqual(bot.load_config(cfg), -1)

    def test_disabled_feature_preserves_normal_attack_behavior(self):
        bot = self.make_bot([
            make_monster(150),
            make_monster(160),
            make_monster(170),
        ])
        bot.cfg["directional_aoe"]["enable"] = False

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_monster_at_close_threshold_uses_power_knockback(self):
        bot = self.make_bot(
            [make_monster(160)],
            knockback_enabled=True,
            knockback_distance=40,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "power_knockback")
        self.assertEqual(bot.cmd_move_x, "left")
        self.assertEqual(bot.t_last_power_knockback, 10.0)
        self.assertEqual(bot.t_last_attack, 10.0)

    def test_monster_one_pixel_beyond_close_threshold_uses_bow(self):
        bot = self.make_bot(
            [make_monster(159)],
            knockback_enabled=True,
            knockback_distance=40,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_close_left_monster_makes_shootable_right_side_win(self):
        bot = self.make_bot(
            [make_monster(170), make_monster(260)],
            knockback_enabled=True,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "right")

    def test_close_right_monster_makes_shootable_left_side_win(self):
        bot = self.make_bot(
            [make_monster(150), make_monster(230)],
            knockback_enabled=True,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_close_monster_blocks_farther_monster_on_the_same_side(self):
        bot = self.make_bot(
            [make_monster(170), make_monster(120)],
            knockback_enabled=True,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "power_knockback")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_equal_close_monsters_use_cached_facing(self):
        bot = self.make_bot(
            [make_monster(170), make_monster(230)],
            knockback_enabled=True,
        )
        bot.kb.cached_facing = "right"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "power_knockback")
        self.assertEqual(bot.cmd_move_x, "right")

    def test_vertically_distant_monster_does_not_trigger_knockback(self):
        bot = self.make_bot(
            [make_monster(170, center_y=141)],
            knockback_enabled=True,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "none")

    def test_knockback_cooldown_never_falls_back_to_bow(self):
        bot = self.make_bot(
            [make_monster(170)],
            knockback_enabled=True,
            knockback_cooldown=1.0,
        )
        bot.t_last_power_knockback = 9.5
        bot.t_last_attack = 8.0
        bot.cmd_action = "attack"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "none")
        self.assertEqual(bot.t_last_power_knockback, 9.5)
        self.assertEqual(bot.t_last_attack, 8.0)

    def test_blocked_crowded_side_does_not_steal_aoe(self):
        bot = self.make_bot(
            [
                make_monster(170),
                make_monster(120),
                make_monster(90),
                make_monster(260),
            ],
            knockback_enabled=True,
            threshold=3,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "right")

    def test_shootable_side_aoe_wins_over_opposite_close_monster(self):
        bot = self.make_bot(
            [
                make_monster(170),
                make_monster(245),
                make_monster(270),
                make_monster(295),
            ],
            knockback_enabled=True,
            threshold=3,
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "directional_aoe")
        self.assertEqual(bot.cmd_move_x, "right")

    def test_search_roi_covers_larger_knockback_distance(self):
        bot = self.make_bot(
            [],
            knockback_enabled=True,
            knockback_distance=180,
        )

        bot.update_cmd_by_mob_detection()

        bot.get_monsters_in_range.assert_called_once_with(
            (10, 0),
            (390, 200),
        )

    def test_fractional_knockback_distance_is_rejected(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        cfg = {
            "bot": {
                "mode": "normal",
                "attack": "directional",
            },
            "monster_detect": {"backend": "yolo"},
            "key": {"power_knockback": "s"},
            "power_knockback": {
                "enable": True,
                "trigger_distance_x": 40.5,
                "range_y": 80,
                "cooldown": 1.0,
                "attack_recovery_delay": 0.9,
            },
        }

        self.assertEqual(bot.load_config(cfg), -1)

    def test_patrol_periodic_attack_cannot_overwrite_close_range_decision(self):
        bot = SimpleNamespace(
            loc_player=(200, 100),
            img_frame=np.zeros((200, 400, 3), dtype=np.uint8),
            cfg={
                "patrol": {
                    "range": [0.1, 0.9],
                    "turn_point_thres": 3,
                    "patrol_attack_interval": 0.1,
                },
            },
            cmd_move_x="none",
            cmd_move_y="none",
            cmd_action="none",
            t_last_attack=0.0,
            kb=SimpleNamespace(set_command=Mock()),
            is_player_stuck=Mock(return_value=False),
            update_cmd_by_random=Mock(),
        )

        def choose_knockback():
            bot.cmd_action = "power_knockback"
            bot.cmd_move_x = "right"
            bot._suppress_periodic_attack = True

        bot.update_cmd_by_mob_detection = Mock(side_effect=choose_knockback)

        with patch("src.states.patrol.time.time", return_value=10.0):
            PatrolState("patrol", bot).on_frame()

        self.assertEqual(bot.cmd_action, "power_knockback")
        self.assertEqual(bot.t_last_attack, 0.0)
        bot.kb.set_command.assert_called_once_with(
            "right none power_knockback"
        )

    def test_patrol_does_not_replay_prior_knockback_after_targets_vanish(self):
        bot = SimpleNamespace(
            loc_player=(200, 100),
            img_frame=np.zeros((200, 400, 3), dtype=np.uint8),
            cfg={
                "patrol": {
                    "range": [0.1, 0.9],
                    "turn_point_thres": 3,
                    "patrol_attack_interval": 0.9,
                },
            },
            cmd_move_x="right",
            cmd_move_y="none",
            cmd_action="power_knockback",
            t_last_attack=10.0,
            _suppress_periodic_attack=False,
            kb=SimpleNamespace(set_command=Mock()),
            is_player_stuck=Mock(return_value=False),
            update_cmd_by_random=Mock(),
            update_cmd_by_mob_detection=Mock(),
        )

        with patch("src.states.patrol.time.time", return_value=10.1):
            PatrolState("patrol", bot).on_frame()

        self.assertEqual(bot.cmd_action, "none")
        bot.kb.set_command.assert_called_once_with("left none none")


if __name__ == "__main__":
    unittest.main()
