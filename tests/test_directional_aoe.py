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
            },
            "directional_aoe": {
                "enable": True,
                "min_monsters": threshold,
                "range_x": 140,
                "range_y": 100,
                "cooldown": aoe_cooldown,
            },
            "power_knockback": {
                "enable": knockback_enabled,
                "trigger_distance_x": knockback_distance,
                "range_y": 80,
                "cooldown": knockback_cooldown,
            },
            "monster_detect": {
                "backend": "yolo",
                "search_box_margin": 10,
                "max_mob_area_trigger": 400,
            },
            "ui_coords": {"ui_y_start": 180},
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

    def test_visible_bow_target_attacks_without_normal_attack_cooldown(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.wz_navigation = SimpleNamespace(
            patrol_strategy="ranged_safe_platforms"
        )
        bot.cmd_move_x = "right"
        bot.t_last_attack = 9.8

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_ranged_wz_patrol_reissues_normal_attack_without_delay(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.wz_navigation = SimpleNamespace(
            patrol_strategy="ranged_safe_platforms"
        )
        bot.cmd_move_x = "right"
        bot.t_last_attack = 9.99

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_clear_before_move_plan_attacks_immediately(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.wz_navigation = SimpleNamespace(
            patrol_strategy="ranged_safe_platforms",
            plan=SimpleNamespace(combat_checkpoints=(object(),)),
        )
        bot.cmd_move_x = "right"
        bot.t_last_attack = 9.8

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_reserved_wz_jump_yields_to_attackable_monster(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.wz_navigation = SimpleNamespace(
            patrol_strategy="ranged_safe_platforms",
            plan=SimpleNamespace(combat_checkpoints=(object(),)),
        )
        bot.idx_routes = 3
        bot.cmd_move_x = "right"
        bot.cmd_move_y = "none"
        bot.cmd_action = "jump"
        bot._wz_route_jump_atomic_pending = True
        bot._wz_route_jump_atomic_route_index = 3

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )
        self.assertTrue(bot._suppress_periodic_attack)
        self.assertFalse(bot._wz_route_jump_atomic_pending)
        bot.get_monsters_in_range.assert_called_once()

    def test_reserved_stationary_wz_jump_yields_to_attackable_monster(self):
        bot = self.make_bot([make_monster(150)])
        bot.cmd_move_x = "none"
        bot.cmd_move_y = "none"
        bot.cmd_action = "jump"
        bot._wz_route_jump_atomic_pending = True
        bot._wz_route_jump_atomic_route_index = 4
        bot.idx_routes = 4

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )
        self.assertTrue(bot._suppress_periodic_attack)
        self.assertFalse(bot._wz_route_jump_atomic_pending)
        bot.get_monsters_in_range.assert_called_once()

    def test_both_sided_p1_checkpoint_attacks_monster_on_its_left(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        checkpoint = SimpleNamespace(
            facing="both",
            label="P1 clear right outside",
        )
        bot.wz_navigation = SimpleNamespace(
            patrol_strategy="ranged_safe_platforms",
            plan=SimpleNamespace(combat_checkpoints=(checkpoint,)),
            route_legs=(SimpleNamespace(combat_checkpoint=checkpoint),),
        )
        bot.idx_routes = 0
        bot._wz_combat_checkpoint_route_index = 0
        bot._wz_combat_checkpoint_clear_since = None
        bot._wz_combat_checkpoint_cleared_route_index = None
        bot.kb.cached_facing = "right"

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_wall_recovery_attacks_before_returning_to_safe_corridor(self):
        bot = self.make_bot([make_monster(150)])
        bot.wz_navigation = SimpleNamespace(
            patrol_strategy="ranged_safe_platforms",
            plan=SimpleNamespace(combat_checkpoints=(object(),)),
            route_legs=(SimpleNamespace(recovery_path=0),),
        )
        bot.idx_routes = 0
        bot.cmd_move_x = "right"

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )
        self.assertTrue(bot._suppress_periodic_attack)

    def test_wall_recovery_resumes_inward_walk_after_monsters_clear(self):
        bot = self.make_bot([])
        bot.wz_navigation = SimpleNamespace(
            patrol_strategy="ranged_safe_platforms",
            plan=SimpleNamespace(combat_checkpoints=(object(),)),
            route_legs=(SimpleNamespace(recovery_path=0),),
        )
        bot.idx_routes = 0
        bot.cmd_move_x = "right"

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("right", "none", "none"),
        )

    def test_combat_checkpoint_waits_for_quiet_window_before_advancing(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        checkpoint = SimpleNamespace(
            facing="left",
            label="P3 clear P2",
        )
        bot.cfg = {
            "wz_navigation": {"combat_clear_quiet_seconds": 0.8}
        }
        bot.idx_routes = 0
        bot.img_routes = [object(), object()]
        bot.wz_navigation = SimpleNamespace(route_legs=(
            SimpleNamespace(combat_checkpoint=checkpoint),
            SimpleNamespace(combat_checkpoint=None),
        ))
        bot.cmd_move_x = "right"
        bot.cmd_move_y = "none"
        bot.cmd_action = "goal"
        bot.is_show_debug_window = False
        bot._wz_combat_checkpoint_route_index = None
        bot._wz_combat_checkpoint_clear_since = None
        bot._wz_combat_checkpoint_cleared_route_index = None
        bot._reset_rope_climb = Mock()
        bot._reset_ladder_route_hold = Mock()
        bot._route_has_local_continuation = Mock(return_value=True)
        bot._wz_navigation_enabled = True

        bot.check_reach_goal()

        self.assertEqual(bot.idx_routes, 0)
        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("none", "none", "none"),
        )

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            side_effect=(10.0, 10.81),
        ):
            bot._observe_wz_combat_checkpoint(False)
            bot._observe_wz_combat_checkpoint(False)

        bot.cmd_action = "goal"
        bot.check_reach_goal()

        self.assertEqual(bot.idx_routes, 1)
        self.assertIsNone(bot._wz_combat_checkpoint_route_index)

    def test_checkpoint_aligns_to_safe_position_before_horizontal_jump(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        checkpoint = SimpleNamespace(
            facing="right",
            label="P3 clear P4",
        )
        bot.cfg = {
            "wz_navigation": {"combat_clear_quiet_seconds": 0.8}
        }
        bot.idx_routes = 0
        bot.wz_navigation = SimpleNamespace(route_legs=(
            SimpleNamespace(
                action="WALK",
                source=(96, 291),
                target=(146, 291),
                combat_checkpoint=checkpoint,
            ),
            SimpleNamespace(
                action="JUMP",
                source=(149, 291),
                target=(158, 291),
                combat_checkpoint=None,
            ),
        ))
        bot.loc_player_global = (149, 291)
        bot.cmd_move_x = "right"
        bot.cmd_move_y = "none"
        bot.cmd_action = "goal"
        bot._wz_combat_checkpoint_route_index = None
        bot._wz_combat_checkpoint_clear_since = None
        bot._wz_combat_checkpoint_cleared_route_index = None

        self.assertTrue(bot._hold_at_wz_combat_checkpoint())

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("none", "none", "jump_align_left"),
        )
        bot._observe_wz_combat_checkpoint(False)
        self.assertIsNone(bot._wz_combat_checkpoint_clear_since)
        self.assertIsNone(bot._wz_combat_checkpoint_cleared_route_index)

        bot.loc_player_global = (146, 291)
        bot.cmd_action = "goal"
        self.assertTrue(bot._hold_at_wz_combat_checkpoint())
        self.assertEqual(bot.cmd_action, "none")
        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            side_effect=(10.0, 10.81),
        ):
            bot._observe_wz_combat_checkpoint(False)
            bot._observe_wz_combat_checkpoint(False)

        self.assertEqual(bot._wz_combat_checkpoint_cleared_route_index, 0)

    def test_wz_buff_gate_blocks_only_jump_transactions(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot._wz_navigation_enabled = True
        bot._stationary_jump_proximity_active = False
        bot._rope_climb_active = False
        bot._portal_sweep_active = False
        bot.is_on_ladder = False
        bot.wz_navigation = SimpleNamespace(
            active=True,
            jump_active=False,
        )

        self.assertTrue(bot._scheduled_buff_allowed())

        bot.wz_navigation.jump_active = True
        self.assertFalse(bot._scheduled_buff_allowed())
        bot.wz_navigation.jump_active = False

        bot._rope_climb_active = True
        bot.is_on_ladder = True
        self.assertTrue(bot._scheduled_buff_allowed())
        bot._rope_climb_active = False
        bot.is_on_ladder = False

        bot._stationary_jump_proximity_active = True
        self.assertFalse(bot._scheduled_buff_allowed())
        bot._stationary_jump_proximity_active = False

        bot._wz_timed_jump_owns_input = Mock(return_value=True)
        self.assertFalse(bot._scheduled_buff_allowed())

    def test_buff_recovery_freezes_engine_route_state(self):
        bot = MapleStoryAutoBot.__new__(MapleStoryAutoBot)
        bot.cmd_move_x = "right"
        bot.cmd_move_y = "up"
        bot.cmd_action = "jump"
        bot.kb = SimpleNamespace(
            is_buff_recovery_active=Mock(return_value=True),
            set_command=Mock(),
        )

        self.assertTrue(bot._hold_for_scheduled_buff_recovery())

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("none", "none", "none"),
        )
        bot.kb.set_command.assert_called_once_with("none none none")

    def test_light_blue_rope_approach_yields_to_attackable_monster(self):
        bot = self.make_bot([make_monster(150)], threshold=3)
        bot._rope_climb_active = True
        bot._rope_climb_state = {
            "phase": "position",
            "started_at": 1.0,
            "position_last_progress_at": 1.0,
        }
        bot.cmd_move_x = "right"
        bot.cmd_action = "rope_align_right"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ), patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")
        self.assertEqual(bot.cmd_move_y, "none")
        self.assertTrue(bot._rope_climb_combat_deferred)

    def test_rope_mount_request_yields_before_hid_dispatch(self):
        bot = self.make_bot([make_monster(150)], threshold=3)
        bot.cfg["directional_aoe"]["enable"] = False
        bot.loc_player_global = (10, 10)
        bot._rope_climb_active = True
        bot._rope_climb_state = {
            "phase": "mount_request",
            "started_at": 1.0,
            "position_last_progress_at": 1.0,
        }
        bot.wz_navigation = SimpleNamespace(
            platform_state_machine_active=True,
            platform_combat_priority=False,
        )
        bot.cmd_move_x = "left"
        bot.cmd_move_y = "up"
        bot.cmd_action = "rope_mount_left"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )
        self.assertTrue(bot._rope_climb_combat_deferred)
        bot.get_monsters_in_range.assert_called_once()

    def test_light_blue_rope_approach_rearms_after_monsters_disappear(self):
        bot = self.make_bot([], threshold=3)
        bot.loc_player_global = (10, 10)
        bot._rope_climb_active = True
        bot._rope_climb_combat_deferred = True
        bot._rope_climb_combat_deferred_at = 5.0
        bot._rope_climb_state = {
            "phase": "position",
            "started_at": 1.0,
            "position_last_progress_at": 2.0,
        }

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.monotonic",
            return_value=8.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertFalse(bot._rope_climb_combat_deferred)
        self.assertIsNone(bot._rope_climb_combat_deferred_at)
        self.assertEqual(bot._rope_climb_state["started_at"], 4.0)
        self.assertEqual(
            bot._rope_climb_state["position_last_progress_at"], 5.0
        )

    def test_ladder_route_skips_all_combat_and_preserves_up(self):
        bot = self.make_bot(
            [make_monster(190)],
            knockback_enabled=True,
        )
        bot.is_on_ladder = True
        bot.cmd_move_x = "right"
        bot.cmd_move_y = "up"
        bot.get_monsters_in_range = Mock(
            side_effect=AssertionError("YOLO should not run on a ladder")
        )
        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("right", "up", "none"),
        )
        self.assertTrue(bot._suppress_periodic_attack)
        self.assertEqual(bot.monsters, [])
        bot.get_monsters_in_range.assert_not_called()

    def test_initial_vertical_route_does_not_suppress_watchdog(self):
        bot = self.make_bot([make_monster(190)])
        bot.is_on_ladder = False
        bot.cmd_move_y = "up"
        bot.get_monsters_in_range = Mock(
            side_effect=AssertionError("combat must not interrupt route Up")
        )

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("none", "up", "none"),
        )
        self.assertFalse(bot._suppress_periodic_attack)
        bot.get_monsters_in_range.assert_not_called()

    def test_stationary_jump_yields_to_attackable_monster(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot._stationary_jump_proximity_active = True
        bot.cmd_action = "jump"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_stationary_jump_yields_without_normal_attack_cooldown(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot._stationary_jump_proximity_active = True
        bot.cmd_action = "jump"
        bot.t_last_attack = 9.8

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_stationary_jump_two_sided_tie_uses_cached_facing(self):
        bot = self.make_bot([make_monster(150), make_monster(250)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot._stationary_jump_proximity_active = True
        bot.cmd_action = "jump"
        bot.kb.cached_facing = "right"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "right")

    def test_directional_jump_yields_to_attackable_monster(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.cmd_move_x = "right"
        bot.cmd_move_y = "none"
        bot.cmd_action = "jump"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "left")

    def test_directional_jump_yields_without_normal_attack_cooldown(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.cmd_move_x = "right"
        bot.cmd_move_y = "none"
        bot.cmd_action = "jump"
        bot.t_last_attack = 9.8

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_vertical_drop_jump_yields_to_attackable_monster(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.cmd_move_x = "none"
        bot.cmd_move_y = "down"
        bot.cmd_action = "jump"

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_timed_jump_candidate_ignores_expired_travel_combat_budget(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.wz_navigation = SimpleNamespace(
            platform_state_machine_active=True,
            platform_combat_priority=False,
        )
        bot._wz_timed_jump_candidate = {"direction": "right"}
        bot.cmd_move_x = "right"

        bot.update_cmd_by_mob_detection()

        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_armed_but_unfired_timed_jump_is_cancelled_for_monster(self):
        bot = self.make_bot([make_monster(150)])
        bot.cfg["directional_aoe"]["enable"] = False
        token = (8, 0, "jump:p3:p4")
        cancel = Mock(return_value=True)
        bot.kb = SimpleNamespace(
            cached_facing="right",
            scheduled_directional_jump_status=Mock(
                return_value={"state": "pending"}
            ),
            cancel_scheduled_directional_jump=cancel,
        )
        bot._wz_timed_jump_token = token
        bot._wz_timed_jump_route_index = 0
        bot._wz_timed_jump_direction = "right"
        bot.cmd_move_x = "right"

        bot.update_cmd_by_mob_detection()

        cancel.assert_called_once_with(token)
        self.assertIsNone(bot._wz_timed_jump_token)
        self.assertEqual(
            (bot.cmd_move_x, bot.cmd_move_y, bot.cmd_action),
            ("left", "none", "attack"),
        )

    def test_directional_jump_two_sided_tie_uses_cached_facing(self):
        bot = self.make_bot([make_monster(150), make_monster(250)])
        bot.cfg["directional_aoe"]["enable"] = False
        bot.cmd_move_x = "left"
        bot.cmd_move_y = "none"
        bot.cmd_action = "jump"
        bot.kb.cached_facing = "right"

        with patch(
            "src.engine.MapleStoryAutoLevelUp.time.time",
            return_value=10.0,
        ):
            bot.update_cmd_by_mob_detection()

        self.assertEqual(bot.cmd_action, "attack")
        self.assertEqual(bot.cmd_move_x, "right")

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
