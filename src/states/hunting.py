from src.states.base_state import State

class HuntingState(State):
    def on_enter(self):
        pass

    def on_exit(self):
        pass

    def check_transitions(self):
        # Legacy rune PNGs are intentionally not enlarged for DirectShow 4K.
        # Keep the solver inert until native templates and geometry are ready.
        if self.bot.cfg.get("rune_solver", {}).get("enable", False) is not True:
            return None
        if self.bot.rune_solver.is_rune_enable(
            self.bot.img_frame_gray, self.bot.img_frame_debug) or \
            self.bot.rune_solver.is_rune_warning(
            self.bot.img_frame_gray, self.bot.img_frame_debug):
            # When "Rune enable" message appears on screen
            self.bot.screenshot_img_frame()

            return "finding_rune"

        else:
            return None

    def on_frame(self):
        # Get commend from route map
        self.bot.update_cmd_by_route()

        # Check if reach goal on route map
        self.bot.check_reach_goal()

        # Get attack commend by detecting mobs near players
        self.bot.update_cmd_by_mob_detection()

        # Platform dwell/quiet transitions are evaluated only after this
        # frame's final monster-range and terrain arbitration.
        platform_combat_update = getattr(
            self.bot, "update_wz_platform_combat_state", None
        )
        if callable(platform_combat_update) and platform_combat_update():
            # The runtime has replaced the temporary path. Do not emit one
            # last command from the retired patrol leg before installation on
            # the next minimap update.
            self.bot.cmd_move_x = "none"
            self.bot.cmd_move_y = "none"
            self.bot.cmd_action = "none"

        # If player stuck for too long, perform a random command
        if self.bot.is_player_stuck():
            deterministic_recovery = getattr(
                self.bot, "recover_wz_platform_navigation", None
            )
            recovered = bool(
                callable(deterministic_recovery)
                and deterministic_recovery()
            )
            if not recovered:
                self.bot.update_cmd_by_random()

        # A rope launch is committed only after the current mob snapshot has
        # kept the 60 Hz spatial trigger.
        finalize_spatial_rope = getattr(
            self.bot, "finalize_rope_spatial_mount", None
        )
        if callable(finalize_spatial_rope):
            finalize_spatial_rope()

        # Horizontal platform jumps use the measured Alt/capture latency and
        # are handed to the keyboard worker only after monster arbitration.
        finalize_timed_jump = getattr(
            self.bot, "finalize_wz_timed_directional_jump", None
        )
        if callable(finalize_timed_jump):
            finalize_timed_jump()

        # Send the one fully arbitrated frame command to the keyboard worker.
        self.bot.kb.set_command(self.bot.cmd_move_x + ' ' + \
                                self.bot.cmd_move_y + ' ' + \
                                self.bot.cmd_action)
