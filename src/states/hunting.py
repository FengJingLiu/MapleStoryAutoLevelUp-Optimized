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

        # Arm a latency-compensated WZ Jump only after monster arbitration and
        # stuck recovery have kept the route's horizontal approach command.
        # Its one-shot timer then runs independently of the next vision frame.
        finalize_timed_jump = getattr(
            self.bot, "finalize_wz_timed_directional_jump", None
        )
        if callable(finalize_timed_jump):
            finalize_timed_jump()

        # A WZ rope approach uses the same two-phase arbitration: route
        # planning reserves a timer candidate, monsters may still cancel it,
        # and only this point arms the one-shot independently of vision FPS.
        finalize_timed_rope = getattr(
            self.bot, "finalize_rope_timed_mount", None
        )
        rope_timer_armed = bool(
            callable(finalize_timed_rope) and finalize_timed_rope()
        )
        timed_rope_owner = getattr(
            self.bot, "_rope_timed_mount_owns_input", None
        )
        rope_timer_owns_input = bool(
            callable(timed_rope_owner) and timed_rope_owner()
        )

        # send command to keyboard controller
        # schedule_rope_mount already installs the arbitrated horizontal
        # state before starting its timer. A zero-delay callback may already
        # have replaced it with Up, so do not overwrite that result here.
        if not (rope_timer_armed or rope_timer_owns_input):
            self.bot.kb.set_command(self.bot.cmd_move_x + ' ' + \
                                    self.bot.cmd_move_y + ' ' + \
                                    self.bot.cmd_action)
