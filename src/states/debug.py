"""Vision-only state used to inspect detections without game input."""

from src.states.base_state import State


class DebugState(State):
    """Run full-camera monster detection and never issue a command."""

    def __init__(self, name, bot):
        super().__init__(name, bot)
        self.frame_index = 0

    def on_enter(self):
        self.frame_index = 0
        self.bot.monsters = []

    def on_exit(self):
        pass

    def check_transitions(self):
        return None

    def on_frame(self):
        interval = max(
            1,
            int(
                self.bot.cfg.get("debug", {}).get(
                    "scan_interval_frames", 5
                )
            ),
        )
        should_scan = self.frame_index % interval == 0
        self.frame_index += 1

        frame_h, frame_w = self.bot.img_frame.shape[:2]
        camera_h = min(frame_h, self.bot.cfg["ui_coords"]["ui_y_start"])
        if should_scan:
            threshold = self.bot.cfg.get("debug", {}).get(
                "monster_diff_thres"
            )
            self.bot.monsters = self.bot.get_debug_monsters_in_range(
                (0, 0),
                (frame_w, camera_h),
                score_thres=threshold,
            )
        else:
            self.bot.draw_monster_detections(
                self.bot.monsters,
                (0, 0),
                (frame_w, camera_h),
            )
