'''
Execute this script:
python mapleStoryAutoLevelUp.py --map cloud_balcony --monster brown_windup_bear,pink_windup_bear
'''
# Standard import
import time
import random
import argparse
import glob
import sys
import logging
import os
import datetime
import threading
import math
from copy import deepcopy

# Library import
import numpy as np
import cv2
import yaml

# Local import
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.logger import logger
from src.utils.common import (find_pattern_sqdiff, draw_rectangle, draw_circle,
    draw_line, draw_text, screenshot, nms,
    load_image, get_mask, get_minimap_loc_size, get_player_location_on_minimap,
    copy_minimap_native_raster, copy_minimap_native_location,
    route_map_can_fit_minimap,
    is_mac, override_cfg, load_yaml, get_all_other_player_locations_on_minimap,
    click_in_game_window, mask_route_colors, to_opencv_hsv, debug_minimap_colors,
    activate_game_window, normalize_pixel_coordinate, resize_window
)
from src.utils.detection import (
    detection_center,
    detection_to_box,
    intersection_area,
    suppress_nearby_same_class,
)
from src.utils.frame_geometry import scale_runtime_pixel_config
from src.utils.minimap_geometry import (
    load_minimap_geometry,
    scale_minimap_rect,
)
from src.input.CaptureFramePreprocessor import preprocess_capture_frame
from src.input.CaptureSource import (
    DIRECTSHOW_SOURCE,
    capture_profile_override,
    create_capture_source,
    resolve_capture_source,
)
from src.input.KeyBoardController import (
    KeyBoardController,
    has_calibrated_absolute_mouse,
    press_key,
    validate_absolute_mouse_config,
)
from src.input.Esp32HidClient import (
    RELATIVE_MOUSE_MAX_DELTA,
    usage_from_text,
)
from src.input.KeyBoardListener import KeyBoardListener
if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
else:
    from src.input.GameWindowCapturor import GameWindowCapturor
from src.engine.HealthMonitor import HealthMonitor
from src.engine.Profiler import Profiler
from src.engine.RuneSolver import RuneSolver
from src.engine.FiniteStateMachine import FiniteStateMachine
from src.vision.YoloMonsterDetector import YoloMonsterDetector
from src.vision.auto_relogin_ocr import (
    is_chinese_ocr_target,
    matches_ocr_target,
    normalize_ocr_text,
    RapidOcrError,
    RapidOcrTextLocator,
    StableOcrTargetGate,
)
from src.vision.cursor_tracker import CursorTracker
from src.navigation import WzNavigationRuntime
from src.navigation.jump_timing import predict_directional_jump
from src.navigation.wz_catalog import WzMapRecognitionError
from src.navigation.wz_geometry import Point
from src.states.hunting import HuntingState
from src.states.finding_rune import FindingRuneState
from src.states.near_rune import NearRuneState
from src.states.solving_rune import SolvingRuneState
from src.states.auxiliary import AuxiliaryState
from src.states.patrol import PatrolState
from src.states.debug import DebugState


FOREST_FLOOR_MAP_ID = "100040110"
FOREST_FLOOR_P1_DECORATIVE_WALL_IDS = frozenset({
    "wall:1:1:102",
    "wall:1:1:104",
    "wall:1:2:141",
    "wall:1:2:143",
    "wall:1:3:144",
    "wall:1:3:146",
    "wall:1:4:138",
    "wall:1:4:140",
})


def _resolve_wz_navigation_request(mode, map_name, wz_navigation_cfg):
    """Return whether this selection requests WZ mode and its bound map ID."""
    binding_map_id = None
    bindings = wz_navigation_cfg.get("map_bindings", {})
    if isinstance(bindings, dict) and map_name != "__auto_wz__":
        raw_binding = bindings.get(map_name)
        if isinstance(raw_binding, (str, int)) and not isinstance(
                raw_binding, bool):
            normalized_binding = str(raw_binding).strip()
            if normalized_binding:
                binding_map_id = normalized_binding

    enabled = mode == "normal" and (
        wz_navigation_cfg.get("enable", False)
        or map_name == "__auto_wz__"
        or binding_map_id is not None
    )
    return enabled, binding_map_id


class MapleStoryAutoBot:
    '''
    MapleStoryAutoBot
    '''
    def __init__(self, args):
        '''
        Init MapleStoryAutoBot
        '''
        self.args = args # User args
        self.cfg = None # Configuration
        self.idx_routes = 0 # Index of route map
        self.monsters_info = {} # monster information
        self.yolo_monster_detector = None
        self.yolo_hero_detector = None
        self.yolo_inference_class_names = ("mob",)
        self.monsters = [] # monster detected in current frame
        self.fps = 0 # Frame per second
        self.red_dot_center_prev = None # previous other player location in minimap
        # F3 writes the source capture and the rendered diagnostic canvas to
        # separate files.  ``video_writer`` keeps its historical name and is
        # the raw-stream writer; the second writer owns annotated frames.
        self.video_writer = None # Raw video recording writer
        self._video_record_path = None
        self._video_record_size = None
        self.annotated_video_writer = None
        self._annotated_video_record_path = None
        self._annotated_video_record_size = None
        self._annotated_record_frame_pending = False
        self._video_record_lock = threading.RLock()
        self.color_code = {} # For color code instruction
        self.color_code_up_down = {} # Color code only contain 'up' and 'down'
        self._ladder_route_move_y = None
        self._ladder_route_exit_confirmed_at = None
        self._route_loop_goal_key = None
        self._route_loop_goal_warned_keys = set()
        self._stationary_jump_targets_by_route = []
        self._rope_climb_targets_by_route = []
        self._rope_climb_state = None
        self._rope_climb_active = False
        self._rope_timed_mount_candidate = None
        self._rope_climb_combat_deferred = False
        self._rope_climb_combat_deferred_at = None
        self._rope_climb_completed_key = None
        self._rope_climb_failed_key = None
        self._rope_climb_completed_position = None
        self._rope_climb_failed_position = None
        self._rope_climb_failed_at = None
        self._stationary_jump_proximity_active = False
        self._portal_sweep_active = False
        self._portal_sweep_key = None
        self._portal_sweep_region = None
        self._portal_sweep_direction = None
        self._portal_sweep_origin = None
        self._portal_sweep_started_at = None
        self._portal_sweep_last_observed_position = None
        self._portal_sweep_last_nudge_position = None
        self._portal_sweep_last_nudge_direction = None
        self._portal_sweep_last_nudge_time = 0.0
        self._portal_sweep_failed_key = None
        self._portal_sweep_failed_region = None
        self._auto_relogin_state = "idle"
        self._auto_relogin_pending_page = None
        self._auto_relogin_confirmation_return_state = None
        self._auto_relogin_expected_page = None
        self._auto_relogin_pending_location = None
        self._auto_relogin_confirm_count = 0
        self._auto_relogin_confirm_miss_count = 0
        self._auto_relogin_ready_count = 0
        self._auto_relogin_next_action_at = 0.0
        self._auto_relogin_page_next_action_at = {}
        self._auto_relogin_last_action_at = None
        self._auto_relogin_last_action_page = None
        self._auto_relogin_has_attempted_input = False
        self._auto_relogin_action_attempts = {}
        self._auto_relogin_step_started_at = None
        self._auto_relogin_health_was_enabled = False
        self._auto_relogin_confirm_started_at = None
        self._auto_relogin_last_confirm_frame_token = None
        self._auto_relogin_last_ready_frame_token = None
        self._auto_relogin_waiting_game_started_at = None
        self._auto_relogin_failure_logged = False
        self._auto_relogin_fallback_frame_counter = 0
        self._auto_relogin_started_at = None
        self._auto_relogin_next_ocr_scan_at = 0.0
        self._auto_relogin_ocr_page_scan_token = None
        self._auto_relogin_ocr_page_scan_matches = None
        self._auto_relogin_ocr_page_matches = {}
        self._reset_auto_relogin_ocr_gate()
        self._reset_auto_relogin_pointer_runtime()
        self.thread_auto_bot = None # thread for running autobot
        self.cmd_move_x = "none" # "left" "right"
        self.cmd_move_y = "none" # "up" "down"
        self.cmd_action = "none" # "jump" "attack" ....
        # Signals (for UI)
        self.image_debug_signal = None
        self.route_map_viz_signal = None
        self._last_ui_viz_emit_time = 0.0
        # Flags
        self.is_first_frame = True # first frame flag
        self.is_terminated = False # Close all object and thread if True
        self.is_on_ladder = False # Character is on ladder or not
        self.is_show_debug_window = not args.disable_viz #
        self.is_need_show_debug_window = not args.disable_viz #
        self.is_disable_control = args.disable_control
        self.is_ui = args.is_ui # Whether is using UI framework to invoke engine
        self.is_frame_done = False #
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on game screen
        self.has_valid_nametag_location = False
        self.nametag_miss_count = 0
        self.pending_nametag_location = None
        self.loc_overhead_marker_player = (0, 0)
        self.has_valid_overhead_marker_location = False
        self.overhead_marker_miss_count = 0
        self.pending_overhead_marker_location = None
        self.pending_overhead_marker_count = 0
        self.t_last_overhead_marker_detected = None
        self.last_overhead_marker_match = None
        self.last_single_yolo_hero_location = None
        self.last_single_yolo_hero_frame_token = None
        self.last_yolo_hero_detection = None
        self.screen_player_location_valid = False
        self.loc_party_red_bar = (0, 0) # party red bar location on game screen
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_watch_dog = (0, 0) # watch dog location on global map
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        # Bound atomically to ``frame`` by ``get_img_frame``. Recovery must
        # never read the capturor's live timestamp after copying an older image.
        self._current_capture_frame_token = None
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        self.img_minimap_screen = self.img_minimap # unscaled minimap in screen coordinates
        self.img_minimap_source = self.img_minimap # native capture-card minimap
        self._native_minimap_size = None
        self._last_native_minimap_error = None
        self.img_capture_content = None
        self.minimap_geometry = None
        self._last_route_map_size_error = None
        self.wz_navigation = None
        self._wz_navigation_enabled = False
        self._wz_navigation_resource_generation = -1
        self._wz_navigation_map_id = None
        self._wz_jump_move_x = "none"
        self._wz_combat_checkpoint_route_index = None
        self._wz_combat_checkpoint_clear_since = None
        self._wz_combat_checkpoint_cleared_route_index = None
        self._wz_combat_checkpoint_jump_alignment_direction = None
        self._wz_route_jump_atomic_pending = False
        self._wz_route_jump_atomic_route_index = None
        self._wz_edge_jump_alignment_route_index = None
        self._wz_timed_jump_candidate = None
        self._wz_timed_jump_token = None
        self._wz_timed_jump_route_index = None
        self._wz_timed_jump_direction = None
        self._wz_navigation_sample_at = None
        self._wz_navigation_sample_position = None
        self._wz_navigation_sample_interval_seconds = None
        self._wz_navigation_sample_step_x = 0.0
        self._wz_attackable_monster_this_frame = False
        self._wz_combat_observation_valid = False
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
        self.t_last_directional_aoe = time.time() # Last single-sided AoE timer
        self.t_last_power_knockback = time.time() # Last close-range knockback timer
        self.t_last_minimap_update = time.time()
        self.t_to_change_channel = time.time()
        # Images
        self.img_map = None
        self.img_routes = []
        self.img_nametag = None
        self.img_nametag_gray = None
        self.img_nametag_medal = None
        self.img_nametag_medal_gray = None
        self.img_nametag_pet = None
        self.img_nametag_pet_gray = None
        self.nametag_appearance_templates = []
        self._img_nametag_source = None
        self._img_nametag_medal_source = None
        self._img_nametag_pet_source = None
        self.img_overhead_marker = None
        self.img_overhead_marker_gray = None
        self.img_overhead_marker_mask = None
        self._img_overhead_marker_source = None
        self.overhead_marker_component_bbox = None
        self._last_nametag_template_geometry = None
        self.loc_appearance_player = (0, 0)
        self.has_valid_appearance_location = False
        self.pending_appearance_location = None
        self.pending_appearance_count = 0
        self.last_appearance_match = None
        self.img_create_party_enable = None
        self.img_create_party_disable = None
        self.img_login_button = None
        self._img_login_button_source = None
        self._last_login_template_geometry = None
        self._auto_relogin_template_sources = {}
        self._auto_relogin_templates = {}
        self._auto_relogin_cursor_template_source = None
        self._auto_relogin_cursor_tracker = None
        self._auto_relogin_disconnect_cursor_template_source = None
        self._auto_relogin_disconnect_cursor_tracker = None
        self._last_auto_relogin_template_geometry = None
        self._auto_relogin_ocr_locator = None
        self._auto_relogin_ocr_gate = None
        self._auto_relogin_ocr_gate_signature = None
        self._pet_mob_ocr_locator = None
        self._last_pet_mob_ocr_error = None

        # Database
        self.data = load_yaml("config/config_data.yaml")
        # Threads & Objects
        self.kb = None # Keyboard controller
        self.capture = None # Game window capturor
        self.health_monitor = None # Health monitor
        self.profiler = None # Profiler, for performance issue debugging
        self.rune_solver = None # Rune solver
        self._shutdown_lock = threading.Lock()
        self._components_stopped = True
        self._input_suspended_for_capture = False
        self._last_capture_geometry = None
        self._last_capture_error = None
        self._base_cfg = None
        self._last_runtime_output_size = None

        # Finite State Machine
        self.fsm = FiniteStateMachine()
        self.fsm.add_state(HuntingState    ("hunting"     , self))
        self.fsm.add_state(FindingRuneState("finding_rune", self))
        self.fsm.add_state(NearRuneState   ("near_rune"   , self))
        self.fsm.add_state(SolvingRuneState("solving_rune", self))
        self.fsm.add_state(AuxiliaryState  ("aux"         , self))
        self.fsm.add_state(PatrolState     ("patrol"      , self))
        self.fsm.add_state(DebugState      ("debug"       , self))
        self.fsm.add_transition("hunting", "finding_rune") # When saw a "Rune has created" messgae
        self.fsm.add_transition("finding_rune", "hunting") # After finding rune timeout
        self.fsm.add_transition("finding_rune", "near_rune") # When detect a nearby rune
        self.fsm.add_transition("finding_rune", "solving_rune") # When enter the arrow minimap
        self.fsm.add_transition("near_rune", "finding_rune") # After rune solving timeout
        self.fsm.add_transition("near_rune", "solving_rune") # When enter the arrow minimap
        self.fsm.add_transition("solving_rune", "hunting") # After rune solving
        self.fsm.set_init_state("hunting")

    def update_signals(self, image_debug_signal, route_map_viz_signal):
        '''
        Update signal from UI framework.
        For debug window viz
        '''
        self.image_debug_signal = image_debug_signal
        self.route_map_viz_signal = route_map_viz_signal

    def _emit_debug_images(self):
        """Emit whichever visualization canvases are available for this mode."""
        if not self.is_show_debug_window or not self.is_ui:
            return

        has_frame = (
            self.img_frame_debug is not None
            and self.image_debug_signal is not None
        )
        has_route = (
            self.img_route_debug is not None
            and self.route_map_viz_signal is not None
        )
        if not has_frame and not has_route:
            return

        # A native 4K capture is about 24 MB at 3840x2160. Sending a
        # fresh full-size copy through Qt ten times per second floods the GUI
        # event queue and makes tab changes appear to hang. Recognition and
        # screenshots continue to use the native image; only the UI preview is
        # rate-limited and reduced here.
        system_cfg = self.cfg.get("system", {}) if self.cfg else {}
        try:
            preview_fps = float(system_cfg.get("fps_limit_ui_viz", 5))
        except (TypeError, ValueError):
            preview_fps = 5.0
        now = time.monotonic()
        last_emit = getattr(self, "_last_ui_viz_emit_time", 0.0)
        if preview_fps > 0 and last_emit > 0 and \
                now - last_emit < 1.0 / preview_fps:
            return
        self._last_ui_viz_emit_time = now

        if has_frame:
            # Visualization should show the complete normalized game frame.
            # ui_y_start separates the camera and HUD for recognition; using
            # it here hid the bottom HUD and clipped a bottom-floor nametag.
            img_frame_debug_emit = self._prepare_ui_preview(
                self.img_frame_debug
            )
            self.image_debug_signal.emit(img_frame_debug_emit)

        # Auxiliary mode has no route map. Keep the game-window visualization
        # alive instead of copying a missing route canvas.
        if has_route:
            self.route_map_viz_signal.emit(
                self._prepare_ui_preview(self.img_route_debug)
            )

    def _prepare_ui_preview(self, image):
        """Return an owned, bounded-size BGR preview for the Qt UI only."""
        system_cfg = self.cfg.get("system", {}) if self.cfg else {}
        max_size = system_cfg.get("ui_viz_max_size", (720, 1280))
        try:
            max_height, max_width = (int(max_size[0]), int(max_size[1]))
        except (TypeError, ValueError, IndexError):
            max_height, max_width = (720, 1280)
        if max_height <= 0 or max_width <= 0:
            max_height, max_width = (720, 1280)

        height, width = image.shape[:2]
        scale = min(1.0, max_width / width, max_height / height)
        if scale < 1.0:
            preview = cv2.resize(
                image,
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            return np.ascontiguousarray(preview)
        return np.ascontiguousarray(image.copy())

    def _debug_reference_size(self):
        """Return the legacy frame geometry used to author debug overlays."""
        base_cfg = getattr(self, "_base_cfg", None)
        if not isinstance(base_cfg, dict):
            base_cfg = self.cfg if isinstance(self.cfg, dict) else {}
        reference = base_cfg.get("game_window", {}).get(
            "coordinate_reference_size", (700, 1296)
        )
        try:
            ref_h, ref_w = map(float, reference[:2])
            if ref_h <= 0 or ref_w <= 0:
                raise ValueError
        except (TypeError, ValueError, IndexError):
            ref_h, ref_w = (700.0, 1296.0)
        return ref_h, ref_w

    def get_frame_visual_scale(self):
        """Scale debug styling so a fitted preview matches the legacy UI."""
        image = getattr(self, "img_frame_debug", None)
        if image is None:
            return 1.0
        frame_h, frame_w = image.shape[:2]
        ref_h, ref_w = self._debug_reference_size()
        return max(0.1, min(frame_h / ref_h, frame_w / ref_w))

    def scale_debug_reference_point(self, point):
        """Map a hard-coded legacy debug anchor to the current native frame."""
        visual_scale = self.get_frame_visual_scale()
        return (
            int(round(float(point[0]) * visual_scale)),
            int(round(float(point[1]) * visual_scale)),
        )

    def _draw_debug_rectangle(
        self,
        top_left,
        size,
        color,
        text,
        thickness=2,
        text_height=0.7,
    ):
        """Draw a native-frame box with preview-stable label styling."""
        draw_rectangle(
            self.img_frame_debug,
            top_left,
            size,
            color,
            text,
            thickness=thickness,
            text_height=text_height,
            visual_scale=self.get_frame_visual_scale(),
        )

    def _draw_debug_text(
        self,
        text,
        origin,
        font_face,
        font_scale,
        color,
        thickness=1,
        line_type=cv2.LINE_8,
        *,
        reference_position=False,
    ):
        """Draw native-frame text at a dynamic or legacy reference anchor."""
        if getattr(self, "img_frame_debug", None) is None:
            return
        if reference_position:
            origin = self.scale_debug_reference_point(origin)
        visual_scale = self.get_frame_visual_scale()
        cv2.putText(
            self.img_frame_debug,
            text,
            tuple(map(int, origin)),
            font_face,
            float(font_scale) * visual_scale,
            color,
            max(1, int(round(float(thickness) * visual_scale))),
            line_type,
        )

    def _draw_debug_circle(self, center, radius, color, thickness=1):
        """Draw a player marker that remains visible in the fitted preview."""
        if getattr(self, "img_frame_debug", None) is None:
            return
        visual_scale = self.get_frame_visual_scale()
        scaled_thickness = (
            -1
            if thickness < 0
            else max(1, int(round(thickness * visual_scale)))
        )
        cv2.circle(
            self.img_frame_debug,
            tuple(map(int, center)),
            max(1, int(round(radius * visual_scale))),
            color,
            scaled_thickness,
        )

    def remote_keyboard_target(self):
        """Return whether HID input goes to game computer B, not this PC."""
        return self.cfg.get("esp32_hid", {}).get("remote_target", False)

    def remote_absolute_mouse_calibrated(self):
        """Return whether remote absolute clicks have explicit geometry."""
        try:
            return has_calibrated_absolute_mouse(self.cfg)
        except (TypeError, ValueError):
            return False

    def is_capture_card_source(self):
        """Return whether frames have no corresponding local target window."""
        capture = getattr(self, "capture", None)
        if capture_profile_override(capture) == "capture_card":
            return True
        args = getattr(self, "args", None)
        test_image_name = getattr(args, "test_image", None)
        try:
            return resolve_capture_source(
                self.cfg,
                test_image_name=test_image_name,
            ) == DIRECTSHOW_SOURCE
        except (AttributeError, TypeError, ValueError):
            return False

    def is_debug_mode(self):
        """Return whether this run is the vision-only debug mode."""
        cfg = getattr(self, "cfg", None) or {}
        return cfg.get("bot", {}).get("mode") == "debug"

    def click_game_ui(self, coord, action):
        """Click one current capture-frame coordinate on the game computer."""
        if self.remote_keyboard_target():
            if not self.remote_absolute_mouse_calibrated():
                logger.error(
                    f"[{action}] Remote absolute mouse is not calibrated"
                )
                return False
            frame = getattr(self, "img_frame", None)
            keyboard_controller = getattr(self, "kb", None)
            if frame is None or keyboard_controller is None or not hasattr(
                    keyboard_controller, "click_game_ui_point"):
                logger.error(
                    f"[{action}] Remote absolute game-UI mouse is unavailable"
                )
                return False
            frame_h, frame_w = frame.shape[:2]
            duration = self._auto_relogin_number(
                "mouse_click_duration", 0.05, minimum=0.001
            )
            return bool(keyboard_controller.click_game_ui_point(
                int(coord[0]),
                int(coord[1]),
                frame_w,
                frame_h,
                button="left",
                duration=duration,
            ))

        if self.is_capture_card_source():
            logger.warning(
                f"[{action}] Skipped local mouse click: DirectShow capture "
                "has no corresponding local game window"
            )
            return False
        click_in_game_window(
            self.capture.window_title,
            (int(coord[0]), int(coord[1])),
        )
        return True

    def _configured_remote_ui_capture_point(self, name):
        """Convert one legacy raw-window UI point to capture content space.

        The legacy menu/channel coordinates include the Windows title bar,
        while DirectShow frames and Magpie's scaled output contain only game
        client pixels. Runtime scaling cannot distinguish those two domains,
        so remote workflows derive these fixed points from the unscaled base
        configuration and remove the authored title-bar offset first.
        """
        point = self.cfg.get("ui_coords", {}).get(name)
        base_cfg = getattr(self, "_base_cfg", None)
        frame = getattr(self, "img_frame", None)
        if base_cfg is None or frame is None:
            return tuple(map(int, point)) if point is not None else None

        base_point = base_cfg.get("ui_coords", {}).get(name)
        reference = base_cfg.get("game_window", {}).get(
            "coordinate_reference_size", (700, 1296)
        )
        try:
            raw_x, raw_y = map(float, base_point)
            reference_h, reference_w = map(float, reference)
            title_bar_height = float(base_cfg.get(
                "game_window", {}
            ).get("title_bar_height", 0))
            frame_h, frame_w = frame.shape[:2]
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                f"invalid configured UI coordinate: {name}"
            ) from exc
        content_y = raw_y - title_bar_height
        if reference_h <= 0 or reference_w <= 0 or raw_x < 0 or \
                content_y < 0:
            raise ValueError(
                f"configured UI coordinate {name} lies outside game content"
            )
        return (
            int(round(raw_x * frame_w / reference_w)),
            int(round(content_y * frame_h / reference_h)),
        )

    def _auto_relogin_config(self):
        """Return the optional session-recovery configuration."""
        cfg = (getattr(self, "cfg", None) or {}).get("auto_relogin", {})
        return cfg if isinstance(cfg, dict) else {}

    def _auto_relogin_remote_mouse_mode(self):
        """Return the configured remote click strategy.

        Missing configuration and shipped profiles select the calibrated
        absolute path. ``visual_relative`` remains an explicit compatibility
        option for integrations that intentionally opt into cursor feedback.
        """
        return str(
            self._auto_relogin_config().get(
                "remote_mouse_mode", "absolute"
            )
        ).strip().lower()

    def _auto_relogin_number(self, name, default, minimum=0.0):
        value = self._auto_relogin_config().get(name, default)
        try:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError
            return max(float(minimum), number)
        except (TypeError, ValueError):
            logger.warning(
                f"[auto_relogin] Invalid {name}={value!r}; using {default}"
            )
            return max(float(minimum), float(default))

    def _auto_relogin_ocr_config(self):
        cfg = self._auto_relogin_config().get("ocr", {})
        return cfg if isinstance(cfg, dict) else {}

    def _auto_relogin_ocr_target_config(self, page):
        """Return a target config supported by the current input path."""
        ocr_cfg = self._auto_relogin_ocr_config()
        if ocr_cfg.get("enable", False) is not True:
            return None
        targets = ocr_cfg.get("targets", {})
        target_cfg = targets.get(page) if isinstance(targets, dict) else None
        if not isinstance(target_cfg, dict):
            return None
        # OCR supplies semantic authorization in every mouse mode. Shipped
        # remote profiles use calibrated absolute clicks; visual-relative is
        # retained only as an explicit compatibility transport.
        return target_cfg

    def _reset_auto_relogin_ocr_gate(self):
        gate = getattr(self, "_auto_relogin_ocr_gate", None)
        if gate is not None:
            gate.reset()

    def _auto_relogin_ocr_components(self):
        """Build the lightweight gate and lazily backed OCR locator."""
        locator = getattr(self, "_auto_relogin_ocr_locator", None)
        if locator is None:
            locator = RapidOcrTextLocator()
            self._auto_relogin_ocr_locator = locator

        ocr_cfg = self._auto_relogin_ocr_config()
        confirm_frames = max(1, int(ocr_cfg.get("confirm_frames", 2)))
        drift = self._auto_relogin_scale_vector(
            ocr_cfg.get("max_center_drift", (24, 24)), minimum=0
        )
        if drift is None:
            raise ValueError("invalid OCR center-drift configuration")
        signature = (confirm_frames, tuple(drift))
        gate = getattr(self, "_auto_relogin_ocr_gate", None)
        if gate is None or signature != getattr(
                self, "_auto_relogin_ocr_gate_signature", None):
            gate = StableOcrTargetGate(
                confirm_frames=confirm_frames,
                max_center_drift=drift,
            )
            self._auto_relogin_ocr_gate = gate
            self._auto_relogin_ocr_gate_signature = signature
        return locator, gate

    def _auto_relogin_ocr_frame_fresh(self, frame_token, now):
        """Reject OCR coordinates derived from a stale or synthetic frame."""
        if not isinstance(frame_token, tuple) or len(frame_token) != 2 or \
                frame_token[0] != "capture":
            return False
        try:
            captured_at = float(frame_token[1])
            max_age = float(
                self._auto_relogin_ocr_config().get("max_frame_age", 1.0)
            )
        except (TypeError, ValueError):
            return False
        return np.isfinite(captured_at) and np.isfinite(max_age) and \
            max_age > 0 and 0 <= now - captured_at <= max_age

    def _auto_relogin_ocr_search_region(
            self, page, target_cfg, page_location):
        """Resolve the OCR ROI in current full-frame coordinates."""
        source = str(
            target_cfg.get("region_source", "configured")
        ).strip().lower()
        if source != "configured":
            return None
        return self._auto_relogin_scale_region(
            target_cfg.get("search_region")
        )

    def _locate_stable_auto_relogin_ocr_target(
            self, page, page_location, frame_token):
        """Return one fresh stable OCR center in current frame coordinates."""
        target_cfg = self._auto_relogin_ocr_target_config(page)
        if target_cfg is None:
            return None
        frame = getattr(self, "img_frame", None)
        region = self._auto_relogin_ocr_search_region(
            page, target_cfg, page_location
        )
        if frame is None or region is None:
            self._fail_auto_relogin(
                f"{page} OCR target geometry is unavailable"
            )
            return None
        try:
            locator, gate = self._auto_relogin_ocr_components()
            scan_token = getattr(
                self, "_auto_relogin_ocr_page_scan_token", None
            )
            scan_matches = getattr(
                self, "_auto_relogin_ocr_page_matches", {}
            )
            current_capture_token = getattr(
                self, "_current_capture_frame_token", None
            )
            scan_is_current = scan_token == frame_token or (
                current_capture_token is None
                and scan_token == ("frame", id(frame))
            )
            match = scan_matches.get(page) if (
                scan_is_current and isinstance(scan_matches, dict)
            ) else None
            if match is None:
                match = locator.locate(
                    frame,
                    region,
                    target_cfg.get("texts", ()),
                    min_score=float(
                        self._auto_relogin_ocr_config().get(
                            "min_score", 0.85
                        )
                    ),
                    match_mode=target_cfg.get("match_mode", "exact"),
                    box_threshold=float(
                        self._auto_relogin_ocr_config().get(
                            "box_threshold", 0.3
                        )
                    ),
                )
        except (RapidOcrError, TypeError, ValueError) as exc:
            logger.error(f"[auto_relogin] {page} RapidOCR failed: {exc}")
            self._fail_auto_relogin(f"{page} RapidOCR target detection failed")
            return None

        if not self._auto_relogin_ocr_frame_fresh(
                frame_token, time.monotonic()):
            gate.reset()
            logger.warning(
                f"[auto_relogin] Ignored stale {page} OCR target frame"
            )
            return None
        point = gate.observe(page, frame_token, match)
        if match is None:
            logger.debug(
                f"[auto_relogin] No unique {page} OCR target in {region}"
            )
            return None
        logger.debug(
            f"[auto_relogin] {page} OCR text={match.text!r}, "
            f"score={match.score:.4f}, center={match.center}"
        )
        if point is None:
            return None
        return point

    def _reset_auto_relogin_pointer_runtime(self):
        """Clear one in-progress visual relative-pointer click session."""
        self._auto_relogin_pointer_page = None
        self._auto_relogin_pointer_action = None
        self._auto_relogin_pointer_target = None
        self._auto_relogin_pointer_frame_shape = None
        self._auto_relogin_pointer_started_at = None
        self._auto_relogin_pointer_last_frame_token = None
        self._auto_relogin_pointer_last_cursor = None
        self._auto_relogin_pointer_cursor_variant = None
        self._auto_relogin_pointer_move_origin = None
        self._auto_relogin_pointer_last_command = None
        self._auto_relogin_pointer_last_move_at = None
        self._auto_relogin_pointer_feedback_frames = 0
        self._auto_relogin_pointer_move_count = 0
        self._auto_relogin_pointer_cursor_misses = 0
        self._auto_relogin_pointer_page_misses = 0
        self._auto_relogin_pointer_aligned_count = 0
        self._auto_relogin_pointer_stall_count = 0
        self._auto_relogin_pointer_motion_verified = False
        self._auto_relogin_pointer_click_failures = 0
        self._auto_relogin_pointer_rescue_index = 0
        self._auto_relogin_pointer_next_input_at = 0.0

    def _reset_auto_relogin_runtime(self):
        """Reset transient recovery state without changing user input gates."""
        self._auto_relogin_state = "idle"
        self._auto_relogin_entry_page = None
        self._auto_relogin_entry_location = None
        self._auto_relogin_entry_count = 0
        self._auto_relogin_entry_started_at = None
        self._auto_relogin_entry_last_frame_token = None
        self._auto_relogin_pending_page = None
        self._auto_relogin_confirmation_return_state = None
        self._auto_relogin_expected_page = None
        self._auto_relogin_pending_location = None
        self._auto_relogin_confirm_count = 0
        self._auto_relogin_confirm_miss_count = 0
        self._auto_relogin_ready_count = 0
        self._auto_relogin_next_action_at = 0.0
        self._auto_relogin_page_next_action_at = {}
        self._auto_relogin_last_action_at = None
        self._auto_relogin_last_action_page = None
        self._auto_relogin_has_attempted_input = False
        self._auto_relogin_action_attempts = {}
        self._auto_relogin_step_started_at = None
        self._auto_relogin_health_was_enabled = False
        self._auto_relogin_confirm_started_at = None
        self._auto_relogin_last_confirm_frame_token = None
        self._auto_relogin_last_ready_frame_token = None
        self._auto_relogin_waiting_game_started_at = None
        self._auto_relogin_failure_logged = False
        self._auto_relogin_fallback_frame_counter = 0
        self._auto_relogin_started_at = None
        self._auto_relogin_next_ocr_scan_at = 0.0
        self._auto_relogin_ocr_page_scan_token = None
        self._auto_relogin_ocr_page_scan_matches = None
        self._auto_relogin_ocr_page_matches = {}
        self._reset_auto_relogin_pointer_runtime()

    def _auto_relogin_enabled(self):
        return self._auto_relogin_config().get("enable", False) is True and \
            not self.is_debug_mode() and \
            not getattr(self, "is_disable_control", False) and \
            not getattr(getattr(self, "capture", None), "is_static_frame", False)

    def _auto_relogin_reactive_pages(self):
        """Return whether recovery reacts to any current OCR page.

        This is the shipped behavior.  The ordered state machine is retained
        only as a compatibility path for older integrations and focused tests.
        """
        return self._auto_relogin_config().get("reactive_pages", True) is True

    def _auto_relogin_frame_token(self, fallback):
        """Identify the capture source frame so cached frames do not confirm."""
        bound_token = getattr(self, "_current_capture_frame_token", None)
        if isinstance(bound_token, tuple) and len(bound_token) == 2 and \
                bound_token[0] == "capture":
            return bound_token

        # Compatibility for lightweight integrations/tests that call recovery
        # directly without going through ``get_img_frame``. A real bot always
        # owns the bound-token attribute and never reads the mutable timestamp.
        token = None
        if not hasattr(self, "_current_capture_frame_token"):
            token = getattr(
                getattr(self, "capture", None), "last_frame_time", None
            )
        try:
            token = float(token)
        except (TypeError, ValueError):
            token = 0.0
        if not np.isfinite(token) or token <= 0:
            self._auto_relogin_fallback_frame_counter = getattr(
                self, "_auto_relogin_fallback_frame_counter", 0
            ) + 1
            return ("loop", self._auto_relogin_fallback_frame_counter)
        return ("capture", token)

    def _pause_gameplay_for_auto_relogin(self):
        """Quiesce gameplay producers while preserving the recovery key path."""
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        self._reset_ladder_route_hold()
        self._reset_stationary_jump_proximity()
        self._reset_rope_climb(clear_locks=True)
        self._reset_portal_sweep()
        self._cancel_wz_timed_directional_jump()
        navigator = getattr(self, "wz_navigation", None)
        suspend_platform = getattr(
            navigator, "suspend_platform_navigation", None
        )
        if callable(suspend_platform):
            suspend_platform()
        self._auto_relogin_ready_count = 0
        self._auto_relogin_last_ready_frame_token = None
        self._auto_relogin_waiting_game_started_at = None
        self._auto_relogin_pending_page = None
        self._auto_relogin_expected_page = None
        self._auto_relogin_pending_location = None
        self._auto_relogin_action_attempts = {}
        self._auto_relogin_last_action_at = None
        self._auto_relogin_last_action_page = None
        self._auto_relogin_has_attempted_input = False
        self._auto_relogin_step_started_at = time.monotonic()
        self._auto_relogin_failure_logged = False
        self._auto_relogin_started_at = time.monotonic()
        self._reset_auto_relogin_ocr_gate()
        self._reset_auto_relogin_pointer_runtime()

        health_monitor = getattr(self, "health_monitor", None)
        health_cfg = (getattr(self, "cfg", None) or {}).get(
            "health_monitor", {}
        )
        self._auto_relogin_health_was_enabled = bool(
            health_monitor is not None
            and health_cfg.get("enable", False)
            and getattr(health_monitor, "enabled", True)
        )
        if health_monitor is not None and hasattr(health_monitor, "disable"):
            health_monitor.disable()

        # Stop the independent health producer before the atomic release so a
        # potion request cannot race in immediately after keys are cleared.
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is not None:
            if hasattr(
                    keyboard_controller,
                    "suspend_automation_for_session_recovery"):
                keyboard_controller.suspend_automation_for_session_recovery()
            else:
                keyboard_controller.set_command("none none none")
                keyboard_controller.release_all_key()

        logger.warning(
            "[auto_relogin] Minimap unavailable; gameplay input suspended "
            "and continuous OCR recovery started"
        )

    def _restore_health_after_auto_relogin(self):
        health_monitor = getattr(self, "health_monitor", None)
        should_enable = getattr(
            self,
            "_auto_relogin_health_was_enabled",
            bool(
                (getattr(self, "cfg", None) or {})
                .get("health_monitor", {})
                .get("enable", False)
            ),
        )
        if should_enable and health_monitor is not None and \
                hasattr(health_monitor, "enable"):
            health_monitor.enable()
        self._auto_relogin_health_was_enabled = False

    def _resume_keyboard_after_auto_relogin(self):
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is not None and hasattr(
                keyboard_controller,
                "resume_automation_after_session_recovery"):
            keyboard_controller.resume_automation_after_session_recovery()

    def _cancel_auto_relogin_confirmation(self):
        """Undo a one-frame false positive before any login action was sent."""
        self._restore_health_after_auto_relogin()
        self._resume_keyboard_after_auto_relogin()
        self._reset_auto_relogin_runtime()

    def _auto_relogin_control_available(self):
        keyboard_controller = getattr(self, "kb", None)
        return not getattr(self, "is_disable_control", False) and \
            keyboard_controller is not None and \
            not getattr(keyboard_controller, "is_terminated", False) and \
            getattr(keyboard_controller, "is_enable", True)

    def _auto_relogin_scale_point(self, point):
        """Scale a point from the recorded login-flow frame to this frame."""
        frame = getattr(self, "img_frame", None)
        if frame is None or not isinstance(point, (list, tuple)) or \
                len(point) != 2:
            return None
        reference = self._auto_relogin_config().get(
            "flow_template_reference_size", (2160, 3840)
        )
        try:
            reference_h, reference_w = map(float, reference[:2])
            x, y = map(float, point)
        except (TypeError, ValueError, IndexError):
            return None
        if min(reference_h, reference_w) <= 0:
            return None
        frame_h, frame_w = frame.shape[:2]
        return (
            int(round(x * frame_w / reference_w)),
            int(round(y * frame_h / reference_h)),
        )

    def _auto_relogin_scale_vector(self, vector, minimum=0):
        """Scale an [x, y] distance from the recorded flow geometry."""
        frame = getattr(self, "img_frame", None)
        if frame is None or not isinstance(vector, (list, tuple)) or \
                len(vector) != 2:
            return None
        reference = self._auto_relogin_config().get(
            "flow_template_reference_size", (2160, 3840)
        )
        try:
            reference_h, reference_w = map(float, reference[:2])
            value_x, value_y = map(float, vector)
        except (TypeError, ValueError, IndexError):
            return None
        if min(reference_h, reference_w, value_x, value_y) < 0 or \
                min(reference_h, reference_w) <= 0:
            return None
        frame_h, frame_w = frame.shape[:2]
        return (
            max(int(minimum), int(round(value_x * frame_w / reference_w))),
            max(int(minimum), int(round(value_y * frame_h / reference_h))),
        )

    def _auto_relogin_scale_region(self, region):
        if not isinstance(region, (list, tuple)) or len(region) != 4:
            return None
        top_left = self._auto_relogin_scale_point(region[:2])
        bottom_right = self._auto_relogin_scale_point(region[2:])
        if top_left is None or bottom_right is None:
            return None
        return (*top_left, *bottom_right)

    def _auto_relogin_ocr_page_scan(self):
        """Run at most one Chinese OCR inference for the current frame."""
        frame = getattr(self, "img_frame", None)
        if frame is None:
            return None
        frame_token = getattr(self, "_current_capture_frame_token", None)
        if not isinstance(frame_token, tuple) or len(frame_token) != 2:
            frame_token = ("frame", id(frame))
        if frame_token == getattr(
                self, "_auto_relogin_ocr_page_scan_token", None):
            return getattr(
                self, "_auto_relogin_ocr_page_scan_matches", None
            )

        target_configs = self._auto_relogin_ocr_config().get("targets", {})
        if not isinstance(target_configs, dict):
            return None
        regions = []
        for target_cfg in target_configs.values():
            if not isinstance(target_cfg, dict):
                continue
            region = self._auto_relogin_ocr_search_region(
                "", target_cfg, None
            )
            if region is not None:
                regions.append(region)
        if not regions:
            return None
        scan_region = (
            min(region[0] for region in regions),
            min(region[1] for region in regions),
            max(region[2] for region in regions),
            max(region[3] for region in regions),
        )

        self._auto_relogin_ocr_page_scan_token = frame_token
        self._auto_relogin_ocr_page_matches = {}
        try:
            locator, _ = self._auto_relogin_ocr_components()
            matches = locator.recognize(
                frame,
                scan_region,
                min_score=float(
                    self._auto_relogin_ocr_config().get("min_score", 0.85)
                ),
                box_threshold=float(
                    self._auto_relogin_ocr_config().get(
                        "box_threshold", 0.3
                    )
                ),
            )
        except (RapidOcrError, TypeError, ValueError) as exc:
            logger.error(f"[auto_relogin] Chinese page OCR failed: {exc}")
            matches = None
        self._auto_relogin_ocr_page_scan_matches = matches
        return matches

    def _auto_relogin_ocr_target_match(self, page):
        """Return one unique configured OCR target from the current scan."""
        frame = getattr(self, "img_frame", None)
        target_cfg = self._auto_relogin_ocr_target_config(page)
        if frame is None or target_cfg is None:
            return None
        region = self._auto_relogin_ocr_search_region(
            page, target_cfg, None
        )
        if region is None:
            return None
        matches = self._auto_relogin_ocr_page_scan()
        if matches is None:
            return None
        targets = tuple(
            normalize_ocr_text(text)
            for text in target_cfg.get("texts", ())
        )
        match_mode = str(
            target_cfg.get("match_mode", "exact")
        ).strip().lower()
        x0, y0, x1, y1 = region
        candidates = []
        for match in matches:
            center_x, center_y = match.center
            if not (x0 <= center_x < x1 and y0 <= center_y < y1):
                continue
            accepted = matches_ocr_target(
                match.normalized_text, targets, match_mode
            )
            if accepted:
                candidates.append(match)
        if len(candidates) != 1:
            return None
        return candidates[0]

    def _match_auto_relogin_page(self, page):
        """Return one unique Chinese OCR marker for a recovery page."""
        match = self._auto_relogin_ocr_target_match(page)
        if match is None:
            return None
        self._auto_relogin_ocr_page_matches[page] = match
        logger.debug(
            f"[auto_relogin] OCR classified {page}: text={match.text!r}, "
            f"score={match.score:.4f}, center={match.center}"
        )
        return match.center

    def _auto_relogin_page_order(self):
        """Return OCR-page priority for the current recovery mode."""
        if self._auto_relogin_reactive_pages():
            # These are specificity/overlay priorities, not login-flow order.
            # The numbered world label remains visible after its channel grid
            # opens, and the queue dialog leaves that channel marker visible
            # behind it. More-specific overlays must win when both exist.
            return (
                "disconnect", "queue", "character", "channel", "world",
                "connect"
            )

        # Compatibility mode preserves the former expected-page preference.
        state = getattr(self, "_auto_relogin_state", "idle")
        preferred = []
        if state == "waiting_page":
            expected = getattr(self, "_auto_relogin_expected_page", None)
            preferred.extend(("disconnect", "queue", "character"))
            preferred.extend((
                expected,
                getattr(self, "_auto_relogin_last_action_page", None),
            ))
        elif state == "confirming":
            pending = getattr(self, "_auto_relogin_pending_page", None)
            preferred.extend(("disconnect", "queue", "character"))
            preferred.extend((
                pending,
                getattr(self, "_auto_relogin_expected_page", None),
            ))
        elif state == "waiting_game":
            preferred.extend(("queue", "character", "disconnect"))

        # More-specific overlays stay ahead of labels visible behind them.
        preferred.extend((
            "disconnect", "queue", "character", "channel", "world",
            "connect"
        ))
        ordered = []
        for page in preferred:
            if page is not None and page not in ordered:
                ordered.append(page)
        return tuple(ordered)

    def _find_known_auto_relogin_page(self):
        """Classify the current recovery page using Chinese OCR only."""
        for page in self._auto_relogin_page_order():
            location = self._match_auto_relogin_page(page)
            if location is not None:
                return page, location
        return None, None

    def _begin_auto_relogin_confirmation(
            self, page, location, now, frame_token):
        if getattr(self, "_auto_relogin_pending_page", None) != page:
            self._reset_auto_relogin_ocr_gate()
            current_state = getattr(self, "_auto_relogin_state", "idle")
            if current_state != "confirming":
                self._auto_relogin_confirmation_return_state = current_state
            self._auto_relogin_pending_page = page
            self._auto_relogin_pending_location = location
            self._auto_relogin_confirm_count = 1
            self._auto_relogin_confirm_miss_count = 0
            self._auto_relogin_confirm_started_at = now
            self._auto_relogin_last_confirm_frame_token = frame_token
        else:
            self._auto_relogin_pending_location = location
            if frame_token != getattr(
                    self, "_auto_relogin_last_confirm_frame_token", None):
                prior_count = getattr(self, "_auto_relogin_confirm_count", 0)
                self._auto_relogin_confirm_count = max(1, prior_count + 1)
                if prior_count <= 0:
                    self._auto_relogin_confirm_started_at = now
                self._auto_relogin_last_confirm_frame_token = frame_token
                self._auto_relogin_confirm_miss_count = 0
        self._auto_relogin_state = "confirming"

    def _auto_relogin_confirmation_ready(self, now):
        required_frames = max(
            1,
            int(round(self._auto_relogin_number(
                "confirm_frames", 2, minimum=1
            ))),
        )
        required_seconds = self._auto_relogin_number(
            "confirm_seconds", 0.5, minimum=0.0
        )
        started_at = getattr(self, "_auto_relogin_confirm_started_at", now)
        if started_at is None:
            return False
        return getattr(self, "_auto_relogin_confirm_count", 0) >= \
            required_frames and now - started_at >= required_seconds

    def _send_auto_relogin_key(self):
        key = str(
            self._auto_relogin_config().get("remote_confirm_key", "enter")
        ).strip()
        if not key:
            return False
        keyboard_controller = self.kb
        if hasattr(keyboard_controller, "press_session_recovery_key"):
            return bool(keyboard_controller.press_session_recovery_key(key))
        return bool(press_key(key))

    def _send_auto_relogin_focus_next_key(self):
        """Focus the next remote window before sending the confirmed key."""
        cfg = self._auto_relogin_config()
        key = str(cfg.get("remote_confirm_key", "enter")).strip()
        sender = getattr(
            self.kb,
            "focus_next_window_and_press_session_recovery_key",
            None,
        )
        if not key or not callable(sender):
            logger.error(
                "[auto_relogin] Remote launcher focus switching is unavailable"
            )
            return False
        focus_keys = cfg.get("focus_switch_keys", ("alt", "tab"))
        logger.info(
            "[auto_relogin] Switching focus to the launcher before sending "
            f"{key!r}"
        )
        return bool(sender(
            key,
            focus_keys=focus_keys,
            focus_hold=self._auto_relogin_number(
                "focus_switch_hold", 0.10, minimum=0.001
            ),
            settle_delay=self._auto_relogin_number(
                "focus_switch_settle_delay", 0.50, minimum=0.0
            ),
            duration=self._auto_relogin_number(
                "focus_enter_duration", 0.10, minimum=0.001
            ),
        ))

    def _send_auto_relogin_click(
            self, point, action, *, click_count=1, click_interval=0.0):
        """Send one bounded click sequence to a capture-frame point."""
        if point is None or getattr(self, "is_terminated", False):
            return False
        if isinstance(click_count, bool) or not isinstance(click_count, int) or \
                click_count < 1:
            return False
        try:
            click_interval = max(0.0, float(click_interval))
        except (TypeError, ValueError, OverflowError):
            return False
        if self.remote_keyboard_target():
            keyboard_controller = self.kb
            frame = getattr(self, "img_frame", None)
            if frame is None or not hasattr(
                    keyboard_controller, "click_session_recovery_point"):
                logger.error(
                    f"[{action}] Remote absolute mouse HID is unavailable"
                )
                return False
            frame_h, frame_w = frame.shape[:2]
            duration = self._auto_relogin_number(
                "mouse_click_duration", 0.05, minimum=0.001
            )
            for click_index in range(click_count):
                if click_index and click_interval:
                    time.sleep(click_interval)
                if getattr(self, "is_terminated", False) or not bool(
                        keyboard_controller.click_session_recovery_point(
                            int(point[0]),
                            int(point[1]),
                            frame_w,
                            frame_h,
                            button="left",
                            duration=duration,
                        )):
                    return False
            return True

        title_bar_height = int(
            self.cfg.get("game_window", {}).get("title_bar_height", 0)
        )
        local_point = (
            int(point[0]), int(point[1]) + title_bar_height
        )
        for click_index in range(click_count):
            if click_index and click_interval:
                time.sleep(click_interval)
            if getattr(self, "is_terminated", False) or not bool(
                    self.click_game_ui(local_point, action)):
                return False
        return True

    def _auto_relogin_channel_click_point(self):
        """Return one recognized channel-label center from the current popup.

        Channel geometry is deliberately not recorded. RapidOCR finds all
        visible ``频道N`` labels in the same scan that confirmed the popup,
        and the highest visible channel number is preferred deterministically.
        """
        matches = self._auto_relogin_ocr_page_scan()
        if matches is None:
            return None
        candidates = {}
        duplicate_numbers = set()
        for match in matches:
            text = normalize_ocr_text(match.normalized_text)
            if not text.startswith("频道"):
                continue
            number_text = text[2:]
            if not number_text.isascii() or not number_text.isdecimal():
                continue
            number = int(number_text)
            if not 1 <= number <= 20 or number in duplicate_numbers:
                continue
            if number in candidates:
                duplicate_numbers.add(number)
                del candidates[number]
                continue
            candidates[number] = match
        if not candidates:
            logger.warning(
                "[auto_relogin] Channel popup was confirmed but no unique "
                "频道N OCR target was available"
            )
            return None
        number = max(candidates)
        match = candidates[number]
        logger.info(
            f"[auto_relogin] Dynamic channel target text={match.text!r}, "
            f"score={match.score:.4f}, capture={match.center}"
        )
        return tuple(map(int, match.center))

    def _begin_auto_relogin_pointer_action(
            self, page, target, action, now):
        """Start one cross-frame, visual relative-pointer click session."""
        frame = getattr(self, "img_frame", None)
        if frame is None or target is None:
            self._fail_auto_relogin(
                f"{page} visual click has no current frame or target"
            )
            return False
        try:
            target = (int(target[0]), int(target[1]))
        except (TypeError, ValueError, IndexError, OverflowError):
            self._fail_auto_relogin(f"{page} visual click target is invalid")
            return False
        frame_h, frame_w = frame.shape[:2]
        if not 0 <= target[0] < frame_w or not 0 <= target[1] < frame_h:
            self._fail_auto_relogin(
                f"{page} visual click target is outside the capture frame"
            )
            return False

        self._reset_auto_relogin_pointer_runtime()
        self._auto_relogin_pointer_page = page
        self._auto_relogin_pointer_action = action
        self._auto_relogin_pointer_target = target
        self._auto_relogin_pointer_frame_shape = (frame_h, frame_w)
        self._auto_relogin_pointer_started_at = now
        # The confirmation frame was already consumed to start this session;
        # never use a second processing pass over that cached image to move.
        self._auto_relogin_pointer_last_frame_token = getattr(
            self, "_auto_relogin_last_confirm_frame_token", None
        )
        self._auto_relogin_state = "aiming"
        logger.info(
            f"[auto_relogin] Aiming remote pointer for {page} at {target}"
        )
        return True

    def _match_auto_relogin_hovered_target(self, page, target):
        """Verify a cached button structurally when hover changes its colors."""
        if page not in {"world", "character"}:
            return None
        frame = getattr(self, "img_frame", None)
        template = getattr(self, "_auto_relogin_templates", {}).get(page)
        if frame is None or template is None or target is None:
            return None

        template_h, template_w = template.shape[:2]
        margin = self._auto_relogin_scale_vector((40, 40), minimum=1)
        if margin is None:
            return None
        margin_x, margin_y = margin
        expected_x0 = int(target[0]) - template_w // 2
        expected_y0 = int(target[1]) - template_h // 2
        frame_h, frame_w = frame.shape[:2]
        x0 = max(0, expected_x0 - margin_x)
        y0 = max(0, expected_y0 - margin_y)
        x1 = min(frame_w, expected_x0 + template_w + margin_x)
        y1 = min(frame_h, expected_y0 + template_h + margin_y)
        if x1 - x0 < template_w or y1 - y0 < template_h:
            return None

        roi_gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(
            roi_gray, template_gray, cv2.TM_CCOEFF_NORMED
        )
        _, score, _, location = cv2.minMaxLoc(result)
        threshold = self._auto_relogin_number(
            "mouse_hover_template_correlation", 0.55, minimum=0.0
        )
        if not np.isfinite(score) or score < threshold:
            return None
        center = (
            x0 + int(location[0]) + template_w // 2,
            y0 + int(location[1]) + template_h // 2,
        )
        drift = self._auto_relogin_scale_vector(
            self._auto_relogin_config().get(
                "mouse_target_drift", (50, 50)
            ),
            minimum=1,
        )
        if drift is None or abs(center[0] - int(target[0])) > drift[0] or \
                abs(center[1] - int(target[1])) > drift[1]:
            return None
        logger.debug(
            f"[auto_relogin] Structurally verified hovered {page} "
            f"target at {center} with correlation {score:.4f}"
        )
        return center

    def _auto_relogin_pointer_page_evidence(self, page, target):
        """Return ``(safe_target, hovered)`` for the current pointer page."""
        classified_page, location = self._find_known_auto_relogin_page()
        if classified_page is not None and classified_page != page:
            return None, False
        hovered = False
        if classified_page is None:
            location = self._match_auto_relogin_hovered_target(page, target)
            hovered = location is not None
        if location is None:
            return None, False

        # Fixed-grid pages use their anchored target, while button-template
        # pages follow the current match center so a shifted viewport is safe.
        if page not in {"world", "character"}:
            return tuple(map(int, target)), False
        drift = self._auto_relogin_scale_vector(
            self._auto_relogin_config().get(
                "mouse_target_drift", (50, 50)
            ),
            minimum=1,
        )
        if drift is None or \
                abs(location[0] - int(target[0])) > drift[0] or \
                abs(location[1] - int(target[1])) > drift[1]:
            return None, False
        return tuple(map(int, location)), hovered

    def _auto_relogin_pointer_page_verified(self, page, target):
        """Return a current safe click target only for the classified page."""
        location, _ = self._auto_relogin_pointer_page_evidence(page, target)
        return location

    def _auto_relogin_cursor_over_target(self, page, cursor, target):
        """Return whether the cursor hotspot is inside a hovered button."""
        if page not in {"world", "character"} or cursor is None or \
                target is None:
            return False
        template = getattr(self, "_auto_relogin_templates", {}).get(page)
        if template is None or not hasattr(template, "shape") or \
                len(template.shape) < 2:
            return False
        template_h, template_w = map(int, template.shape[:2])
        if min(template_h, template_w) <= 0:
            return False
        left = int(target[0]) - template_w // 2
        top = int(target[1]) - template_h // 2
        return (
            left <= int(cursor[0]) < left + template_w
            and top <= int(cursor[1]) < top + template_h
        )

    def _auto_relogin_pointer_delta(self, error, tolerance):
        """Convert capture-space error to one bounded relative HID nudge."""
        gain = self._auto_relogin_number(
            "mouse_move_gain", 0.35, minimum=0.001
        )
        maximum = min(
            127,
            max(
                1,
                int(round(self._auto_relogin_number(
                    "mouse_max_delta", 64, minimum=1
                ))),
            ),
        )

        def axis_delta(axis_error, axis_tolerance):
            if abs(axis_error) <= axis_tolerance:
                return 0
            magnitude = max(
                1,
                min(maximum, int(round(abs(axis_error) * gain))),
            )
            return magnitude if axis_error > 0 else -magnitude

        return (
            axis_delta(float(error[0]), float(tolerance[0])),
            axis_delta(float(error[1]), float(tolerance[1])),
        )

    def _locate_auto_relogin_cursor(self):
        """Return one uniquely verified cursor hotspot in capture coordinates."""
        pointer_page = getattr(self, "_auto_relogin_pointer_page", None)
        if pointer_page == "disconnect":
            # The title client has been observed switching between compact and
            # large hand cursors on the same disconnect page. Validate both;
            # the shared strict score/uniqueness gates reject the wrong size.
            tracker_candidates = (
                (
                    "compact",
                    getattr(
                        self,
                        "_auto_relogin_disconnect_cursor_tracker",
                        None,
                    ),
                ),
                (
                    "large",
                    getattr(self, "_auto_relogin_cursor_tracker", None),
                ),
            )
        elif pointer_page in {"world", "channel", "character"}:
            tracker_candidates = ((
                "large",
                getattr(self, "_auto_relogin_cursor_tracker", None),
            ),)
        else:
            # Connect is keyboard-only, and an unknown page must never inherit
            # a cursor model that could authorize a click on the wrong screen.
            return None
        tracker_candidates = tuple(
            (label, tracker)
            for label, tracker in tracker_candidates
            if tracker is not None
        )
        frame = getattr(self, "img_frame", None)
        if not tracker_candidates or frame is None:
            return None

        previous = getattr(
            self, "_auto_relogin_pointer_last_cursor", None
        )
        local_radius = None
        if previous is not None:
            local_radius = self._auto_relogin_scale_vector(
                self._auto_relogin_config().get(
                    "cursor_local_search_radius", (450, 450)
                ),
                minimum=1,
            )
        configured_region = self._auto_relogin_config().get(
            "cursor_search_region"
        )
        search_region = self._auto_relogin_scale_region(configured_region) \
            if configured_region is not None else None
        # After the first rescue command, use the configured right-edge strip.
        # The shipped remote profile deliberately makes this the full frame so
        # a hand cursor seen anywhere during the vertical scan is not missed.
        # Once found, normal local/global tracking resumes.
        if previous is None and getattr(
                self, "_auto_relogin_pointer_rescue_index", 0) > 0:
            rescue_width = self._auto_relogin_scale_vector((
                int(round(self._auto_relogin_number(
                    "mouse_cursor_rescue_search_width", 1200, minimum=1
                ))),
                1,
            ), minimum=1)
            if rescue_width is not None:
                frame_h, frame_w = frame.shape[:2]
                rescue_region = (
                    max(0, frame_w - rescue_width[0]),
                    0,
                    frame_w,
                    frame_h,
                )
                if search_region is None:
                    search_region = rescue_region
                else:
                    search_region = (
                        max(search_region[0], rescue_region[0]),
                        max(search_region[1], rescue_region[1]),
                        min(search_region[2], rescue_region[2]),
                        min(search_region[3], rescue_region[3]),
                    )
        matches = []
        for label, tracker in tracker_candidates:
            try:
                match = tracker.locate(
                    frame,
                    previous_hotspot=previous,
                    local_radius=local_radius,
                    search_region=search_region,
                )
                # A nonlinear Windows/Magpie step can occasionally leave the
                # conservative local window. Fall back to one globally unique
                # match before declaring the cursor missing or nudging it.
                if match is None and previous is not None:
                    match = tracker.locate(
                        frame,
                        search_region=search_region,
                    )
            except (cv2.error, TypeError, ValueError) as exc:
                logger.warning(
                    "[auto_relogin] Cursor localization failed for "
                    f"{label} template: {exc}"
                )
                continue
            if match is not None:
                matches.append((label, tracker, match))

        if not matches:
            return None
        if len(matches) > 1:
            tolerance = self._auto_relogin_scale_vector(
                self._auto_relogin_config().get(
                    "mouse_target_tolerance", (18, 18)
                ),
                minimum=1,
            ) or (1, 1)
            hotspots = [match.hotspot for _, _, match in matches]
            if any(
                    abs(hotspot[0] - hotspots[0][0]) > tolerance[0]
                    or abs(hotspot[1] - hotspots[0][1]) > tolerance[1]
                    for hotspot in hotspots[1:]):
                logger.warning(
                    "[auto_relogin] Compact and large cursor templates "
                    f"disagreed: {hotspots}"
                )
                return None

        def reliability(item):
            """Prefer the match with the strongest weakest safety margin."""
            _, tracker, candidate = item
            min_score = getattr(tracker, "min_score", 0.90)
            uniqueness_margin = getattr(
                tracker, "uniqueness_margin", 0.02
            )
            if not isinstance(min_score, (int, float)):
                min_score = 0.90
            if not isinstance(uniqueness_margin, (int, float)):
                uniqueness_margin = 0.02
            return (
                min(
                    candidate.score - float(min_score),
                    candidate.uniqueness - float(uniqueness_margin),
                ),
                candidate.score,
            )

        label, _, match = max(matches, key=reliability)
        previous_variant = getattr(
            self, "_auto_relogin_pointer_cursor_variant", None
        )
        if previous_variant is not None and label != previous_variant:
            # A different template cannot inherit a prior candidate's click
            # qualification, even when both estimate nearly the same hotspot.
            self._auto_relogin_pointer_motion_verified = False
            self._auto_relogin_pointer_aligned_count = 0
            logger.info(
                "[auto_relogin] Cursor size changed from "
                f"{previous_variant} to {label}; motion proof reset"
            )
        self._auto_relogin_pointer_cursor_variant = label
        logger.debug(
            f"[auto_relogin] Cursor hotspot ({label}) "
            f"{match.hotspot}, score={match.score:.4f}, "
            f"uniqueness={match.uniqueness:.4f}"
        )
        return tuple(map(int, match.hotspot))

    def _auto_relogin_dispatch_time(self, *, pointer=False):
        """Return a fresh dispatch time only while every deadline is valid."""
        dispatch_at = time.monotonic()
        if getattr(self, "is_terminated", False) or not \
                self._auto_relogin_control_available():
            self._fail_auto_relogin("recovery input became unavailable")
            return None

        started_at = getattr(self, "_auto_relogin_started_at", None)
        if started_at is not None and dispatch_at - started_at >= \
                self._auto_relogin_number(
                    "max_recovery_duration", 300.0, minimum=1.0
                ):
            self._fail_auto_relogin("the recovery time limit was exceeded")
            return None
        step_started_at = getattr(
            self, "_auto_relogin_step_started_at", None
        )
        if step_started_at is not None and dispatch_at - step_started_at >= \
                self._auto_relogin_number(
                    "step_timeout", 60.0, minimum=1.0
                ):
            self._fail_auto_relogin(
                "the current login page did not advance in time"
            )
            return None
        if pointer:
            pointer_started_at = getattr(
                self, "_auto_relogin_pointer_started_at", None
            )
            if pointer_started_at is None or \
                    dispatch_at - pointer_started_at >= \
                    self._auto_relogin_number(
                        "mouse_pointer_timeout", 20.0, minimum=1.0
                    ):
                self._fail_auto_relogin(
                    "visual pointer positioning timed out"
                )
                return None
        return dispatch_at

    def _auto_relogin_pointer_feedback_ready(self, frame_token):
        """Accept only capture frames acquired after the last HID movement."""
        moved_at = getattr(
            self, "_auto_relogin_pointer_last_move_at", None
        )
        if moved_at is None:
            return True
        try:
            captured_at = float(frame_token[1])
        except (TypeError, ValueError, IndexError):
            return False
        if not np.isfinite(captured_at) or captured_at <= moved_at:
            return False
        self._auto_relogin_pointer_feedback_frames = getattr(
            self, "_auto_relogin_pointer_feedback_frames", 0
        ) + 1
        required_frames = max(
            1,
            int(round(self._auto_relogin_number(
                "mouse_feedback_frames", 1, minimum=1
            ))),
        )
        required_delay = self._auto_relogin_number(
            "mouse_feedback_delay", 0.20, minimum=0.0
        )
        return self._auto_relogin_pointer_feedback_frames >= \
            required_frames and captured_at - moved_at >= required_delay

    def _auto_relogin_pointer_response(self, origin, cursor, command):
        """Return observed motion and whether it proves cursor identity."""
        observed = (
            int(cursor[0]) - int(origin[0]),
            int(cursor[1]) - int(origin[1]),
        )
        observed_norm = float(np.hypot(*observed))
        command_norm = float(np.hypot(*command))
        minimum = self._auto_relogin_number(
            "mouse_response_min_pixels", 2.0, minimum=1.0
        )
        if observed_norm < minimum or command_norm <= 0:
            return observed, False
        cosine = (
            observed[0] * command[0] + observed[1] * command[1]
        ) / (observed_norm * command_norm)
        required_cosine = min(
            1.0,
            self._auto_relogin_number(
                "mouse_response_min_cosine", 0.50, minimum=0.001
            ),
        )
        return observed, bool(
            np.isfinite(cosine) and cosine >= required_cosine
        )

    def _auto_relogin_pointer_probe_delta(self, cursor):
        """Choose a small inward movement that can identify the real cursor."""
        frame = getattr(self, "img_frame", None)
        if frame is None:
            return None
        amount = min(
            127,
            max(
                2,
                int(round(self._auto_relogin_number(
                    "mouse_probe_delta", 8, minimum=2
                ))),
            ),
        )
        frame_h, frame_w = frame.shape[:2]
        if frame_w >= frame_h or frame_h <= amount * 2:
            return (amount if cursor[0] < frame_w // 2 else -amount, 0)
        return (0, amount if cursor[1] < frame_h // 2 else -amount)

    def _send_auto_relogin_pointer_move(self, cursor, delta, reason):
        """Send one bounded relative move and bind its actual dispatch time."""
        if cursor is None or delta is None or tuple(delta) == (0, 0):
            return False
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is None or not hasattr(
                keyboard_controller, "move_session_recovery_mouse"):
            self._fail_auto_relogin(
                "remote relative mouse movement is unavailable"
            )
            return False
        if self._auto_relogin_dispatch_time(pointer=True) is None:
            return False
        sent = bool(keyboard_controller.move_session_recovery_mouse(*delta))
        dispatched_at = time.monotonic()
        if not sent:
            self._auto_relogin_pointer_next_input_at = dispatched_at + \
                self._auto_relogin_number("input_retry_delay", 1.0)
            return False

        self._auto_relogin_pointer_move_origin = tuple(map(int, cursor))
        self._auto_relogin_pointer_last_command = tuple(map(int, delta))
        self._auto_relogin_pointer_last_move_at = dispatched_at
        self._auto_relogin_pointer_feedback_frames = 0
        self._auto_relogin_pointer_move_count = getattr(
            self, "_auto_relogin_pointer_move_count", 0
        ) + 1
        logger.debug(
            f"[auto_relogin] {reason}: cursor {cursor}, relative move {delta}"
        )
        return True

    def _auto_relogin_cursor_rescue_deltas(self):
        """Return the configured no-click, multi-monitor rescue path."""
        configured = self._auto_relogin_config().get(
            "mouse_cursor_rescue_deltas",
            (
                (4096, 0),
                (4096, 0),
                (-127, 0),
                (0, -64),
                (0, 128),
                (0, -192),
                (0, 256),
                (0, -320),
                (0, 384),
                (0, -448),
                (0, 512),
            ),
        )
        return configured if isinstance(configured, (list, tuple)) else ()

    def _auto_relogin_cursor_rescue_available(self):
        configured = self._auto_relogin_cursor_rescue_deltas()
        index = getattr(self, "_auto_relogin_pointer_rescue_index", 0)
        return index < len(configured)

    def _send_auto_relogin_cursor_rescue(self, now):
        """Home an unseen pointer onto the main display without clicking."""
        configured = self._auto_relogin_cursor_rescue_deltas()
        index = getattr(self, "_auto_relogin_pointer_rescue_index", 0)
        if index >= len(configured):
            return False
        try:
            dx, dy = configured[index]
            dx, dy = int(dx), int(dy)
        except (TypeError, ValueError, IndexError, OverflowError):
            return False
        dx = max(
            -RELATIVE_MOUSE_MAX_DELTA,
            min(RELATIVE_MOUSE_MAX_DELTA, dx),
        )
        dy = max(
            -RELATIVE_MOUSE_MAX_DELTA,
            min(RELATIVE_MOUSE_MAX_DELTA, dy),
        )
        if dx == 0 and dy == 0:
            return False
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is None or not hasattr(
                keyboard_controller, "move_session_recovery_mouse"):
            return False
        if self._auto_relogin_dispatch_time(pointer=True) is None:
            return False
        sent = bool(keyboard_controller.move_session_recovery_mouse(dx, dy))
        dispatched_at = time.monotonic()
        # A rescue acts on an unseen pointer, so no previously verified cursor
        # identity can survive it. The next visible candidate must be probed.
        self._auto_relogin_pointer_motion_verified = False
        self._auto_relogin_pointer_aligned_count = 0
        if not sent:
            self._auto_relogin_pointer_next_input_at = dispatched_at + \
                self._auto_relogin_number("input_retry_delay", 1.0)
            return False
        self._auto_relogin_pointer_rescue_index = index + 1
        self._auto_relogin_pointer_last_command = (dx, dy)
        self._auto_relogin_pointer_move_origin = None
        self._auto_relogin_pointer_last_move_at = dispatched_at
        self._auto_relogin_pointer_feedback_frames = 0
        self._auto_relogin_pointer_move_count = getattr(
            self, "_auto_relogin_pointer_move_count", 0
        ) + 1
        logger.info(
            f"[auto_relogin] Cursor was not visible; sent safe rescue "
            f"movement {(dx, dy)} "
            f"({index + 1}/{len(configured)})"
        )
        return True

    def _advance_auto_relogin_pointer_action(self, now, frame_token):
        """Advance one fresh-frame step of visual relative mouse aiming."""
        if not isinstance(frame_token, tuple) or not frame_token or \
                frame_token[0] != "capture":
            self._fail_auto_relogin(
                "visual pointer control has no fresh capture-frame token"
            )
            return False
        if frame_token == getattr(
                self, "_auto_relogin_pointer_last_frame_token", None):
            return False
        self._auto_relogin_pointer_last_frame_token = frame_token

        page = getattr(self, "_auto_relogin_pointer_page", None)
        target = getattr(self, "_auto_relogin_pointer_target", None)
        started_at = getattr(
            self, "_auto_relogin_pointer_started_at", None
        )
        frame = getattr(self, "img_frame", None)
        if page is None or target is None or started_at is None or frame is None:
            self._fail_auto_relogin("visual pointer state is incomplete")
            return False
        if tuple(frame.shape[:2]) != getattr(
                self, "_auto_relogin_pointer_frame_shape", None):
            self._fail_auto_relogin(
                "capture geometry changed during visual pointer control"
            )
            return False
        if now - started_at >= self._auto_relogin_number(
                "mouse_pointer_timeout", 20.0, minimum=1.0):
            self._fail_auto_relogin("visual pointer positioning timed out")
            return False

        verified_target, target_hovered = \
            self._auto_relogin_pointer_page_evidence(
            page, target
        )
        if verified_target is None:
            classified_page, _ = self._find_known_auto_relogin_page()
            next_page = {
                "disconnect": "connect",
                "connect": "world",
                "world": "channel",
                "channel": "character",
            }.get(page)
            if getattr(
                    self, "_auto_relogin_has_attempted_input", False
                    ) and getattr(
                        self, "_auto_relogin_last_action_page", None
                    ) == page and classified_page == next_page:
                # The one-shot click may have executed even if its serial ACK
                # was lost. A classified successor is the only safe proof.
                return self._complete_auto_relogin_page_action(
                    page, time.monotonic(), next_page
                )
            if page == "character" and getattr(
                    self, "_auto_relogin_has_attempted_input", False
                    ) and self._auto_relogin_current_gameplay_evidence() \
                    is not None:
                return self._complete_auto_relogin_page_action(
                    page, time.monotonic(), None
                )
            if classified_page is not None:
                self._fail_auto_relogin(
                    f"classified {classified_page} while aiming for {page}"
                )
                return False
            self._auto_relogin_pointer_page_misses = getattr(
                self, "_auto_relogin_pointer_page_misses", 0
            ) + 1
            self._auto_relogin_pointer_aligned_count = 0
            miss_limit = max(
                1,
                int(round(self._auto_relogin_number(
                    "mouse_page_miss_limit", 3, minimum=1
                ))),
            )
            if self._auto_relogin_pointer_page_misses >= miss_limit:
                self._fail_auto_relogin(
                    f"{page} page disappeared while positioning the pointer"
                )
            return False
        self._auto_relogin_pointer_page_misses = 0
        verified_target = tuple(map(int, verified_target))
        if verified_target != tuple(map(int, target)):
            self._auto_relogin_pointer_target = verified_target
            self._auto_relogin_pointer_aligned_count = 0
            target = verified_target

        feedback_ready = self._auto_relogin_pointer_feedback_ready(frame_token)
        if not feedback_ready:
            return False

        cursor = self._locate_auto_relogin_cursor()
        if cursor is None:
            self._auto_relogin_pointer_motion_verified = False
            self._auto_relogin_pointer_cursor_misses = getattr(
                self, "_auto_relogin_pointer_cursor_misses", 0
            ) + 1
            self._auto_relogin_pointer_aligned_count = 0
            miss_limit = max(
                1,
                int(round(self._auto_relogin_number(
                    "mouse_cursor_miss_limit", 5, minimum=1
                ))),
            )
            if now >= getattr(
                    self, "_auto_relogin_pointer_next_input_at", 0.0):
                if self._send_auto_relogin_cursor_rescue(now):
                    # The scripted right-edge homing/vertical scan must be
                    # allowed to finish even when the ordinary miss limit is
                    # smaller than the rescue path.  Once the path is
                    # exhausted, misses are bounded again below.
                    self._auto_relogin_pointer_cursor_misses = 0
                    return False
            if self._auto_relogin_cursor_rescue_available():
                return False
            if self._auto_relogin_pointer_cursor_misses >= miss_limit:
                self._fail_auto_relogin(
                    "the mouse cursor could not be located reliably"
                )
            return False
        self._auto_relogin_pointer_cursor_misses = 0

        move_origin = getattr(
            self, "_auto_relogin_pointer_move_origin", None
        )
        last_command = getattr(
            self, "_auto_relogin_pointer_last_command", None
        )
        if move_origin is not None and last_command is not None:
            observed, response_valid = self._auto_relogin_pointer_response(
                move_origin, cursor, last_command
            )
            if response_valid:
                self._auto_relogin_pointer_stall_count = 0
                self._auto_relogin_pointer_motion_verified = True
            else:
                self._auto_relogin_pointer_stall_count = getattr(
                    self, "_auto_relogin_pointer_stall_count", 0
                ) + 1
                self._auto_relogin_pointer_motion_verified = False
                self._auto_relogin_pointer_aligned_count = 0
                logger.warning(
                    "[auto_relogin] Cursor candidate did not follow relative "
                    f"move {last_command}; observed {observed}"
                )
            stall_limit = max(
                1,
                int(round(self._auto_relogin_number(
                    "mouse_stall_limit", 4, minimum=1
                ))),
            )
            if self._auto_relogin_pointer_stall_count >= stall_limit:
                self._fail_auto_relogin(
                    "visible cursor motion did not match relative HID commands"
                )
                return False
        elif last_command is None:
            previous_cursor = getattr(
                self, "_auto_relogin_pointer_last_cursor", None
            )
            jump_tolerance = self._auto_relogin_scale_vector(
                self._auto_relogin_config().get(
                    "mouse_uncommanded_jump_tolerance", (6, 6)
                ),
                minimum=1,
            )
            if previous_cursor is not None and (
                    jump_tolerance is None
                    or abs(int(cursor[0]) - int(previous_cursor[0]))
                    > jump_tolerance[0]
                    or abs(int(cursor[1]) - int(previous_cursor[1]))
                    > jump_tolerance[1]
                    ):
                self._auto_relogin_pointer_motion_verified = False
                self._auto_relogin_pointer_aligned_count = 0
                logger.warning(
                    "[auto_relogin] Cursor candidate jumped without a HID "
                    "move; identity probe required again"
                )

        self._auto_relogin_pointer_last_move_at = None
        self._auto_relogin_pointer_feedback_frames = 0
        self._auto_relogin_pointer_move_origin = None
        self._auto_relogin_pointer_last_command = None
        self._auto_relogin_pointer_last_cursor = cursor

        tolerance = self._auto_relogin_scale_vector(
            self._auto_relogin_config().get(
                "mouse_target_tolerance", (18, 18)
            ),
            minimum=1,
        )
        if tolerance is None:
            self._fail_auto_relogin("mouse target tolerance is invalid")
            return False
        error = (
            int(target[0]) - int(cursor[0]),
            int(target[1]) - int(cursor[1]),
        )
        center_aligned = abs(error[0]) <= tolerance[0] and \
            abs(error[1]) <= tolerance[1]
        hover_aligned = target_hovered and \
            self._auto_relogin_cursor_over_target(page, cursor, target)
        aligned = center_aligned or hover_aligned
        max_moves = max(
            1,
            int(round(self._auto_relogin_number(
                "mouse_max_moves", 40, minimum=1
            ))),
        )

        if aligned:
            if not getattr(
                    self, "_auto_relogin_pointer_motion_verified", False):
                # Never click a template candidate until the same candidate has
                # visibly followed a known relative HID command.
                self._auto_relogin_pointer_aligned_count = 0
                if getattr(
                        self, "_auto_relogin_pointer_move_count", 0
                        ) >= max_moves:
                    self._fail_auto_relogin(
                        "visual pointer movement limit was exceeded"
                    )
                    return False
                if time.monotonic() < getattr(
                        self, "_auto_relogin_pointer_next_input_at", 0.0):
                    return False
                probe = self._auto_relogin_pointer_probe_delta(cursor)
                return self._send_auto_relogin_pointer_move(
                    cursor, probe, "cursor identity probe"
                )

            self._auto_relogin_pointer_aligned_count = getattr(
                self, "_auto_relogin_pointer_aligned_count", 0
            ) + 1
            required = max(
                1,
                int(round(self._auto_relogin_number(
                    "mouse_target_confirm_frames", 2, minimum=1
                ))),
            )
            if self._auto_relogin_pointer_aligned_count < required:
                return False
            final_target, final_hovered = \
                self._auto_relogin_pointer_page_evidence(
                page, target
            )
            if final_target is None:
                self._auto_relogin_pointer_aligned_count = 0
                return False
            final_target = tuple(map(int, final_target))
            if final_target != tuple(map(int, target)):
                self._auto_relogin_pointer_target = final_target
                self._auto_relogin_pointer_aligned_count = 0
                return False
            final_error = (
                int(final_target[0]) - int(cursor[0]),
                int(final_target[1]) - int(cursor[1]),
            )
            final_ready = (
                abs(final_error[0]) <= tolerance[0]
                and abs(final_error[1]) <= tolerance[1]
            ) or (
                final_hovered
                and self._auto_relogin_cursor_over_target(
                    page, cursor, final_target
                )
            )
            if not final_ready:
                self._auto_relogin_pointer_aligned_count = 0
                return False
            keyboard_controller = getattr(self, "kb", None)
            if keyboard_controller is None or not hasattr(
                    keyboard_controller, "click_session_recovery_mouse"):
                self._fail_auto_relogin(
                    "remote current-position mouse click is unavailable"
                )
                return False
            if time.monotonic() < getattr(
                    self, "_auto_relogin_pointer_next_input_at", 0.0):
                return False

            next_page = {
                "disconnect": "connect",
                "connect": "world",
                "world": "channel",
                "channel": "character",
            }.get(page)
            duration = self._auto_relogin_number(
                "mouse_click_duration", 0.05, minimum=0.001
            )
            click_count = 1
            click_interval = 0.0
            if page == "channel":
                click_count = max(
                    1,
                    int(round(self._auto_relogin_number(
                        "channel_click_count", 2, minimum=1
                    ))),
                )
                click_interval = self._auto_relogin_number(
                    "channel_double_click_interval", 0.08, minimum=0.0
                )
            dispatch_at = self._auto_relogin_dispatch_time(pointer=True)
            if dispatch_at is None:
                return False
            try:
                captured_at = float(frame_token[1])
            except (TypeError, ValueError, IndexError):
                captured_at = 0.0
            max_frame_age = self._auto_relogin_number(
                "mouse_click_frame_max_age", 1.0, minimum=0.05
            )
            if not np.isfinite(captured_at) or captured_at <= 0 or \
                    captured_at > dispatch_at or \
                    dispatch_at - captured_at > max_frame_age:
                self._auto_relogin_pointer_aligned_count = 0
                return False

            # Mark the one-shot as consumed before dispatch. If the ACK is
            # lost, only a classified successor may prove it executed; the
            # same click is never blindly replayed.
            self._auto_relogin_has_attempted_input = True
            self._auto_relogin_last_action_page = page
            self._auto_relogin_expected_page = next_page
            sent = True
            for click_index in range(click_count):
                if click_index and click_interval:
                    time.sleep(click_interval)
                sent = bool(
                    keyboard_controller.click_session_recovery_mouse(
                        button="left", duration=duration
                    )
                )
                if not sent:
                    break
            completed_at = time.monotonic()
            if not sent:
                self._fail_auto_relogin(
                    f"{page} click acknowledgement was uncertain"
                )
                return False
            return self._complete_auto_relogin_page_action(
                page, completed_at, next_page
            )

        self._auto_relogin_pointer_aligned_count = 0
        if getattr(self, "_auto_relogin_pointer_move_count", 0) >= max_moves:
            self._fail_auto_relogin(
                "visual pointer movement limit was exceeded"
            )
            return False
        if time.monotonic() < getattr(
                self, "_auto_relogin_pointer_next_input_at", 0.0):
            return False
        delta = self._auto_relogin_pointer_delta(error, tolerance)
        return self._send_auto_relogin_pointer_move(
            cursor,
            delta,
            f"target {target}, error {error}",
        )

    def _begin_auto_relogin_queue_wait(self, now):
        """Hold recovery without input until the queue overlay disappears."""
        self._auto_relogin_state = "waiting_queue"
        self._auto_relogin_pending_page = None
        self._auto_relogin_pending_location = None
        self._auto_relogin_confirmation_return_state = None
        self._auto_relogin_confirm_count = 0
        self._auto_relogin_confirm_miss_count = 0
        self._auto_relogin_confirm_started_at = None
        self._auto_relogin_last_confirm_frame_token = None
        self._auto_relogin_step_started_at = now
        self._reset_auto_relogin_ocr_gate()
        self._reset_auto_relogin_pointer_runtime()
        logger.info(
            "[auto_relogin] Login queue confirmed; waiting without input "
            "until the character page appears"
        )
        return True

    def _complete_auto_relogin_page_action(self, page, now, next_page):
        """Record one consumed page action and continue recovery scanning."""
        attempts = getattr(self, "_auto_relogin_action_attempts", {})
        attempts[page] = attempts.get(page, 0) + 1
        self._auto_relogin_action_attempts = attempts
        self._auto_relogin_last_action_page = page
        self._auto_relogin_last_action_at = now
        retry_delay = self._auto_relogin_number("retry_cooldown", 3.0)
        target_cfg = self._auto_relogin_ocr_target_config(page)
        target_action = str(
            target_cfg.get("action", "")
        ).strip().lower() if isinstance(target_cfg, dict) else ""
        if page == "connect" and target_action == "enter":
            retry_delay = self._auto_relogin_number(
                "connect_enter_retry_delay", 1.0, minimum=0.0
            )
        elif target_action == "focus_next_enter":
            # Repeating Alt+Tab would toggle back to the prior window. Treat
            # the acknowledged launcher handoff as a one-shot and wait for the
            # successor until the normal step timeout fails closed.
            retry_delay = self._auto_relogin_number(
                "step_timeout", 60.0, minimum=1.0
            )
        if self._auto_relogin_reactive_pages():
            page_cooldowns = getattr(
                self, "_auto_relogin_page_next_action_at", {}
            )
            page_cooldowns[page] = now + retry_delay
            self._auto_relogin_page_next_action_at = page_cooldowns
            # A new page can act immediately; only this same page is cooled
            # down.  The legacy global value remains clear in reactive mode.
            self._auto_relogin_next_action_at = 0.0
        else:
            self._auto_relogin_next_action_at = now + retry_delay
        self._auto_relogin_pending_page = None
        self._auto_relogin_confirmation_return_state = None
        self._auto_relogin_pending_location = None
        self._auto_relogin_confirm_count = 0
        self._auto_relogin_confirm_miss_count = 0
        self._auto_relogin_confirm_started_at = None
        self._auto_relogin_last_confirm_frame_token = None
        self._auto_relogin_step_started_at = now
        self._reset_auto_relogin_ocr_gate()
        self._reset_auto_relogin_pointer_runtime()

        if self._auto_relogin_reactive_pages():
            self._auto_relogin_state = "waiting_page"
            self._auto_relogin_expected_page = None
            self._auto_relogin_waiting_game_started_at = None
            logger.info(
                f"[auto_relogin] {page} action sent; continuing OCR page "
                "detection until the minimap returns"
            )
        elif page == "character":
            self._auto_relogin_state = "waiting_game"
            self._auto_relogin_expected_page = None
            self._auto_relogin_waiting_game_started_at = now
            self._auto_relogin_ready_count = 0
            self._auto_relogin_last_ready_frame_token = None
            logger.info(
                "[auto_relogin] Start Game clicked; waiting for gameplay evidence"
            )
        else:
            self._auto_relogin_state = "waiting_page"
            self._auto_relogin_expected_page = next_page
            logger.info(
                f"[auto_relogin] {page} action sent; waiting for {next_page} page"
            )
        return True

    def _execute_auto_relogin_page_action(
            self, page, location, now, frame_token=None):
        """Act only on a visually confirmed page, then wait for its successor."""
        if not self._auto_relogin_control_available():
            logger.warning(
                "[auto_relogin] Recovery input skipped while control is paused"
            )
            return False
        reactive_pages = self._auto_relogin_reactive_pages()
        next_action_at = (
            getattr(self, "_auto_relogin_page_next_action_at", {}).get(
                page, 0.0
            )
            if reactive_pages
            else getattr(self, "_auto_relogin_next_action_at", 0.0)
        )
        if now < next_action_at:
            return False

        attempts = getattr(self, "_auto_relogin_action_attempts", {})
        attempt_limit_name = "max_step_attempts"
        if page == "connect":
            connect_target = self._auto_relogin_ocr_target_config(page)
            connect_action = str(
                connect_target.get("action", "")
            ).strip().lower() if isinstance(connect_target, dict) else ""
            if connect_action == "enter":
                attempt_limit_name = "connect_enter_max_attempts"
        max_attempts = max(
            1,
            int(round(self._auto_relogin_number(
                attempt_limit_name,
                30 if attempt_limit_name == "connect_enter_max_attempts" else 5,
                minimum=1,
            ))),
        )
        if not reactive_pages and attempts.get(page, 0) >= max_attempts:
            self._fail_auto_relogin(f"{page} page attempts were exhausted")
            return False

        next_page = {
            "disconnect": "connect",
            "connect": "world",
            "world": "channel",
            "channel": "character",
        }.get(page)
        point = None
        action = None
        click_count = 1
        click_interval = 0.0
        ocr_target_cfg = self._auto_relogin_ocr_target_config(page)
        uses_ocr_target = ocr_target_cfg is not None
        ocr_action = str(
            ocr_target_cfg.get("action", "click")
        ).strip().lower() if uses_ocr_target else None
        uses_ocr_click = uses_ocr_target and ocr_action == "click"
        uses_ocr_enter = uses_ocr_target and ocr_action == "enter"
        uses_ocr_focus_next_enter = uses_ocr_target and \
            ocr_action == "focus_next_enter"
        uses_ocr_wait = uses_ocr_target and ocr_action == "wait"
        if not uses_ocr_target:
            self._fail_auto_relogin(
                f"{page} has no configured Chinese OCR target"
            )
            return False

        semantic_point = self._locate_stable_auto_relogin_ocr_target(
            page, location, frame_token
        )
        if semantic_point is None:
            return False
        logger.info(
            f"[auto_relogin] {page} OCR target confirmed at "
            f"capture={semantic_point}"
        )
        if uses_ocr_wait:
            return self._begin_auto_relogin_queue_wait(now)
        if uses_ocr_click:
            point = semantic_point
            if page == "channel":
                point = self._auto_relogin_channel_click_point()
                if point is None:
                    return False
            action = {
                "connect": "auto_relogin_connect",
                "world": "auto_relogin_select_world",
                "channel": "auto_relogin_select_channel",
                "character": "auto_relogin_start_game",
            }.get(page, f"auto_relogin_{page}")
        elif not uses_ocr_enter and not uses_ocr_focus_next_enter:
            return False

        if page == "channel" and uses_ocr_click:
            click_count = max(
                1,
                int(round(self._auto_relogin_number(
                    "channel_click_count", 2, minimum=1
                ))),
            )
            click_interval = self._auto_relogin_number(
                "channel_double_click_interval", 0.08, minimum=0.0
            )

        if point is not None:
            click_label = "double-click" if click_count == 2 else \
                f"{click_count}-click"
            logger.info(
                f"[auto_relogin] {page} action point={point}, "
                f"input={click_label}"
            )

        if not uses_ocr_enter and not uses_ocr_focus_next_enter and \
                self.remote_keyboard_target() and \
                self._auto_relogin_remote_mouse_mode() == "visual_relative":
            return self._begin_auto_relogin_pointer_action(
                page, point, action, now
            )

        # A serial timeout can mean the HID command executed but its ACK was
        # lost. Remember dispatch before calling the transport so a page change
        # can never be mistaken for a harmless one-frame false positive.
        # Consume the two-frame OCR authorization before touching the
        # transport. A missing ACK must be retried from fresh OCR frames.
        self._reset_auto_relogin_ocr_gate()
        if self._auto_relogin_dispatch_time(pointer=False) is None:
            return False
        self._auto_relogin_has_attempted_input = True
        if uses_ocr_enter:
            sent = self._send_auto_relogin_key()
        elif uses_ocr_focus_next_enter:
            sent = self._send_auto_relogin_focus_next_key()
        else:
            sent = self._send_auto_relogin_click(
                point,
                action,
                click_count=click_count,
                click_interval=click_interval,
            )
        completed_at = time.monotonic()

        if not sent:
            # Preserve both possible outcomes of an uncertain transport result:
            # retry this same visually confirmed page if it remains, or accept
            # only its configured successor if the command actually executed.
            self._auto_relogin_last_action_page = page
            self._auto_relogin_expected_page = next_page
            retry_at = completed_at + self._auto_relogin_number(
                "input_retry_delay", 1.0
            )
            if reactive_pages:
                page_cooldowns = getattr(
                    self, "_auto_relogin_page_next_action_at", {}
                )
                page_cooldowns[page] = retry_at
                self._auto_relogin_page_next_action_at = page_cooldowns
            else:
                self._auto_relogin_next_action_at = retry_at
            logger.warning(f"[auto_relogin] {page} action was not sent")
            return False
        return self._complete_auto_relogin_page_action(
            page, completed_at, next_page
        )

    def _fail_auto_relogin(self, reason):
        """Reset one recovery step, or fail the legacy ordered flow."""
        if self._auto_relogin_reactive_pages():
            self._auto_relogin_state = "waiting_page"
            self._auto_relogin_pending_page = None
            self._auto_relogin_pending_location = None
            self._auto_relogin_confirmation_return_state = None
            self._auto_relogin_confirm_count = 0
            self._auto_relogin_confirm_miss_count = 0
            self._auto_relogin_confirm_started_at = None
            self._auto_relogin_last_confirm_frame_token = None
            self._reset_auto_relogin_ocr_gate()
            self._reset_auto_relogin_pointer_runtime()
            logger.warning(
                f"[auto_relogin] Recovery step reset: {reason}; continuing "
                "OCR page detection until the minimap returns"
            )
            return

        self._auto_relogin_state = "failed"
        self._reset_auto_relogin_ocr_gate()
        self._reset_auto_relogin_pointer_runtime()
        self._auto_relogin_ready_count = 0
        self._auto_relogin_last_ready_frame_token = None
        if not getattr(self, "_auto_relogin_failure_logged", False):
            logger.error(
                f"[auto_relogin] Automatic recovery stopped: {reason}. "
                "Gameplay input remains suspended; recover manually or "
                "restart the bot."
            )
            self._auto_relogin_failure_logged = True

    def _reset_health_monitor_after_auto_relogin(self):
        """Discard stale health/watchdog state before its worker is resumed."""
        health_monitor = getattr(self, "health_monitor", None)
        now = time.time()
        if health_monitor is not None:
            health_monitor.hp_percent = 100
            health_monitor.mp_percent = 100
            health_monitor.exp_percent = 100
            health_monitor.t_hp_watch_dog = now
            health_monitor.t_last_hp_reduce = now
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is not None:
            keyboard_controller.is_need_force_heal = False

    def _auto_relogin_current_minimap_structure(self):
        """Return the current detected minimap, or ``None`` when absent."""
        frame = getattr(self, "img_frame", None)
        if frame is None:
            return None
        detected_result = get_minimap_loc_size(frame)
        if self._auto_relogin_minimap_structure_valid(detected_result):
            return detected_result

        # The four-edge detector is deliberately strict and can miss a valid
        # capture-card minimap when one border is blended or covered.  A route
        # with persisted geometry has a stronger second source of truth: crop
        # that exact raster and require the current Hero dot inside it.  Merely
        # being able to crop the rectangle is not gameplay evidence because the
        # same screen coordinates also exist on login pages.
        geometry = getattr(self, "minimap_geometry", None)
        if geometry is None:
            return None
        try:
            x, y, width, height = scale_minimap_rect(
                geometry,
                frame.shape[:2],
            )
        except (TypeError, ValueError, KeyError):
            return None
        minimap = frame[y:y+height, x:x+width]
        if minimap.size == 0:
            return None

        minimap_cfg = self.cfg.get("minimap", {})
        player_color = minimap_cfg.get("player_color")
        if player_color is None:
            return None
        player_location = get_player_location_on_minimap(
            minimap,
            minimap_player_color=player_color,
            color_tolerance=minimap_cfg.get(
                "player_color_tolerance", 0
            ),
            min_component_area=minimap_cfg.get(
                "player_min_component_area", 4
            ),
        )
        if player_location is None:
            return None

        if not getattr(
                self, "_auto_relogin_saved_minimap_fallback_logged", False):
            logger.info(
                "[auto_relogin] Minimap border scan missed; accepted saved "
                "geometry with a current Hero dot"
            )
            self._auto_relogin_saved_minimap_fallback_logged = True

        # Match get_minimap_loc_size(): it returns the rectangle including the
        # one-pixel border, while persisted geometry stores the inner raster.
        return (x - 1, y - 1, width + 2, height + 2)

    def _auto_relogin_current_gameplay_evidence(self):
        """Return a current player dot if the game already finished loading."""
        frame = getattr(self, "img_frame", None)
        detected_result = self._auto_relogin_current_minimap_structure()
        if frame is None or detected_result is None:
            return None

        x, y, width, height = map(int, detected_result)
        x += 1
        y += 1
        width -= 2
        height -= 2
        if width <= 0 or height <= 0:
            return None
        minimap = frame[y:y+height, x:x+width]
        if minimap.size == 0:
            return None

        minimap_cfg = self.cfg.get("minimap", {})
        player_color = minimap_cfg.get("player_color")
        if player_color is None:
            return None
        return get_player_location_on_minimap(
            minimap,
            minimap_player_color=player_color,
            color_tolerance=minimap_cfg.get("player_color_tolerance", 0),
            min_component_area=minimap_cfg.get("player_min_component_area", 4),
        )

    def _reset_reactive_auto_relogin_confirmation(self):
        """Return to passive OCR scanning without leaving recovery mode."""
        self._auto_relogin_state = "waiting_page"
        self._auto_relogin_pending_page = None
        self._auto_relogin_pending_location = None
        self._auto_relogin_confirmation_return_state = None
        self._auto_relogin_confirm_count = 0
        self._auto_relogin_confirm_miss_count = 0
        self._auto_relogin_confirm_started_at = None
        self._auto_relogin_last_confirm_frame_token = None
        self._reset_auto_relogin_ocr_gate()

    def _reset_auto_relogin_entry_confirmation(self):
        self._auto_relogin_entry_page = None
        self._auto_relogin_entry_location = None
        self._auto_relogin_entry_count = 0
        self._auto_relogin_entry_started_at = None
        self._auto_relogin_entry_last_frame_token = None

    def _observe_auto_relogin_entry_page(
            self, page, location, now, frame_token):
        """Require one stable OCR page before suspending the game session."""
        if page is None or location is None:
            self._reset_auto_relogin_entry_confirmation()
            return False
        if getattr(self, "_auto_relogin_entry_page", None) != page:
            self._auto_relogin_entry_page = page
            self._auto_relogin_entry_location = location
            self._auto_relogin_entry_count = 1
            self._auto_relogin_entry_started_at = now
            self._auto_relogin_entry_last_frame_token = frame_token
        else:
            self._auto_relogin_entry_location = location
            if frame_token != getattr(
                    self, "_auto_relogin_entry_last_frame_token", None):
                self._auto_relogin_entry_count = getattr(
                    self, "_auto_relogin_entry_count", 0
                ) + 1
                self._auto_relogin_entry_last_frame_token = frame_token

        required_frames = max(
            1,
            int(round(self._auto_relogin_number(
                "confirm_frames", 2, minimum=1
            ))),
        )
        required_seconds = self._auto_relogin_number(
            "confirm_seconds", 0.5, minimum=0.0
        )
        started_at = getattr(self, "_auto_relogin_entry_started_at", None)
        return bool(
            started_at is not None
            and getattr(self, "_auto_relogin_entry_count", 0)
            >= required_frames
            and now - started_at >= required_seconds
        )

    def _hold_gameplay_for_missing_minimap(self):
        """Release movement for this frame without resetting route/FSM state."""
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is not None:
            keyboard_controller.set_command("none none none")
            if hasattr(keyboard_controller, "set_minimap_available"):
                keyboard_controller.set_minimap_available(False)

    def _restore_gameplay_minimap_input_gate(self):
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is not None and hasattr(
                keyboard_controller, "set_minimap_available"):
            keyboard_controller.set_minimap_available(True)

    def _check_auto_relogin_reactive_screen(self):
        """React to a stable OCR login page until the minimap reappears."""
        if not self._auto_relogin_enabled():
            return False
        if getattr(self, "is_terminated", False):
            return True

        state = getattr(self, "_auto_relogin_state", "idle")
        minimap_result = self._auto_relogin_current_minimap_structure()
        if minimap_result is not None:
            if state == "idle":
                self._reset_auto_relogin_entry_confirmation()
                self._restore_gameplay_minimap_input_gate()
                return False
            self._resume_gameplay_after_auto_relogin()
            # Resume ordinary processing from a completely fresh frame.
            return True

        now = time.monotonic()
        frame_token = self._auto_relogin_frame_token(now)
        classified_page = None
        location = None
        if state == "idle":
            classified_page, location = self._find_known_auto_relogin_page()
            if classified_page is None:
                # Border detection alone must not own the frame. A configured
                # saved crop (processed later in run_once) may still contain a
                # perfectly valid minimap, as is common with native 4K capture.
                # The normal WZ update safely emits ``none`` if that crop also
                # lacks a current Hero dot.
                self._reset_auto_relogin_entry_confirmation()
                self._restore_gameplay_minimap_input_gate()
                return False

            # A recognized login page stops gameplay immediately, while stable
            # OCR confirmation is still required before any recovery input.
            self._hold_gameplay_for_missing_minimap()
            if not self._observe_auto_relogin_entry_page(
                    classified_page, location, now, frame_token):
                return True

            entry_count = self._auto_relogin_entry_count
            entry_started_at = self._auto_relogin_entry_started_at
            self._pause_gameplay_for_auto_relogin()
            self._begin_auto_relogin_confirmation(
                classified_page, location, now, frame_token
            )
            # Entry already supplied the configured consecutive-frame/time
            # proof. Preserve it so the confirmed page may act immediately
            # after the normal-input gate has closed.
            self._auto_relogin_confirm_count = entry_count
            self._auto_relogin_confirm_started_at = entry_started_at
            self._reset_auto_relogin_entry_confirmation()
            state = "confirming"

        # A pointer transaction may span several capture frames. If the UI was
        # changed manually or loaded another recognizable page, abandon that
        # pointer target and immediately start confirming the current page.
        if state == "aiming":
            pointer_page = getattr(self, "_auto_relogin_pointer_page", None)
            classified_page, location = self._find_known_auto_relogin_page()
            if classified_page is not None and \
                    classified_page != pointer_page:
                self._reset_auto_relogin_pointer_runtime()
                self._reset_reactive_auto_relogin_confirmation()
                self._begin_auto_relogin_confirmation(
                    classified_page, location, now, frame_token
                )
                return True
            self._advance_auto_relogin_pointer_action(now, frame_token)
            return True

        if state == "waiting_queue":
            classified_page, location = self._find_known_auto_relogin_page()
            if classified_page in {None, "queue"}:
                # The queue can briefly blank while its count refreshes. Keep
                # ownership and, critically, emit no click or key in either
                # case. A newly visible character page resumes confirmation.
                return True
            self._begin_auto_relogin_confirmation(
                classified_page, location, now, frame_token
            )
            return True

        # Legacy terminal/wait states are non-terminal in reactive mode.
        if state not in {"waiting_page", "confirming"}:
            self._reset_reactive_auto_relogin_confirmation()
            state = "waiting_page"

        if classified_page is None:
            classified_page, location = self._find_known_auto_relogin_page()
        if state == "waiting_page":
            if classified_page is not None:
                self._begin_auto_relogin_confirmation(
                    classified_page, location, now, frame_token
                )
            return True

        pending_page = getattr(self, "_auto_relogin_pending_page", None)
        if classified_page is not None and classified_page != pending_page:
            # Page transitions and ambiguous world/channel text are expected.
            # Switch confirmation targets instead of treating either as an
            # invalid sequence or a terminal recovery failure.
            logger.debug(
                "[auto_relogin] OCR page changed while confirming: "
                f"{pending_page} -> {classified_page}"
            )
            self._begin_auto_relogin_confirmation(
                classified_page, location, now, frame_token
            )
            return True

        if classified_page is None:
            if frame_token != getattr(
                    self, "_auto_relogin_last_confirm_frame_token", None):
                self._reset_auto_relogin_ocr_gate()
                self._auto_relogin_confirm_miss_count = getattr(
                    self, "_auto_relogin_confirm_miss_count", 0
                ) + 1
                self._auto_relogin_confirm_count = 0
                self._auto_relogin_confirm_started_at = None
                self._auto_relogin_last_confirm_frame_token = frame_token
            allowed_misses = max(
                1,
                int(round(self._auto_relogin_number(
                    "cancel_confirm_misses", 2, minimum=1
                ))),
            )
            if getattr(
                    self, "_auto_relogin_confirm_miss_count", 0
                    ) >= allowed_misses:
                self._reset_reactive_auto_relogin_confirmation()
            return True

        self._begin_auto_relogin_confirmation(
            pending_page, location, now, frame_token
        )
        if not self._auto_relogin_confirmation_ready(now):
            return True

        self._execute_auto_relogin_page_action(
            pending_page,
            getattr(self, "_auto_relogin_pending_location", location),
            now,
            frame_token,
        )
        return True

    def _check_auto_relogin_screen(self):
        if self._auto_relogin_reactive_pages():
            return self._check_auto_relogin_reactive_screen()
        return self._check_auto_relogin_ordered_screen()

    def _check_auto_relogin_ordered_screen(self):
        """Advance the ordered OCR recovery flow, including queue waiting.

        Returning ``True`` means recovery owns the frame, so gameplay systems
        must remain stopped. Only ``waiting_game`` yields unknown frames to the
        regular minimap detector; the second gate still blocks gameplay until
        fresh player evidence has been confirmed on consecutive frames.
        """
        if not self._auto_relogin_enabled():
            return False
        if getattr(self, "is_terminated", False):
            return True

        state = getattr(self, "_auto_relogin_state", "idle")
        now = time.monotonic()
        frame_token = self._auto_relogin_frame_token(now)

        if state == "idle":
            next_scan_at = getattr(
                self, "_auto_relogin_next_ocr_scan_at", 0.0
            )
            if now < next_scan_at:
                return False
            scan_interval = self._auto_relogin_ocr_config().get(
                "idle_scan_interval", 1.0
            )
            try:
                scan_interval = max(0.1, float(scan_interval))
            except (TypeError, ValueError):
                scan_interval = 1.0
            self._auto_relogin_next_ocr_scan_at = now + scan_interval
            # Avoid running OCR continuously during ordinary gameplay.  A
            # missing live minimap is only a cheap trigger; the Chinese OCR
            # target still provides all recovery-page authorization.
            if self._auto_relogin_current_gameplay_evidence() is not None:
                return False
            page, location = self._find_known_auto_relogin_page()
            if page is None or location is None:
                return False
            self._pause_gameplay_for_auto_relogin()
            self._begin_auto_relogin_confirmation(
                page, location, now, frame_token
            )
            return True

        if state == "failed":
            # Automatic input stays stopped after a bounded failure. Known
            # login pages remain owned by recovery, while an unknown frame is
            # allowed through so manual recovery can prove gameplay readiness.
            next_scan_at = getattr(
                self, "_auto_relogin_next_ocr_scan_at", 0.0
            )
            if now < next_scan_at:
                return True
            self._auto_relogin_next_ocr_scan_at = now + max(
                0.1,
                float(self._auto_relogin_ocr_config().get(
                    "idle_scan_interval", 1.0
                )),
            )
            page, _ = self._find_known_auto_relogin_page()
            return page is not None

        started_at = getattr(self, "_auto_relogin_started_at", None)
        if started_at is not None and now - started_at >= \
                self._auto_relogin_number(
                    "max_recovery_duration", 300.0, minimum=1.0
                ):
            self._fail_auto_relogin("the recovery time limit was exceeded")
            return True

        if state in {"waiting_page", "confirming", "aiming"}:
            step_started_at = getattr(
                self, "_auto_relogin_step_started_at", None
            )
            if step_started_at is not None and now - step_started_at >= \
                    self._auto_relogin_number(
                        "step_timeout", 60.0, minimum=1.0
                ):
                self._fail_auto_relogin(
                    "the current login page did not advance in time"
                )
                return True

        if state == "aiming":
            self._advance_auto_relogin_pointer_action(now, frame_token)
            return True

        if state == "waiting_queue":
            page, location = self._find_known_auto_relogin_page()
            if page in {None, "queue"}:
                return True
            self._begin_auto_relogin_confirmation(
                page, location, now, frame_token
            )
            state = "confirming"

        if state == "waiting_game":
            page, location = self._find_known_auto_relogin_page()
            if page is None:
                game_started_at = getattr(
                    self, "_auto_relogin_waiting_game_started_at", now
                )
                if game_started_at is None:
                    game_started_at = now
                    self._auto_relogin_waiting_game_started_at = now
                if now - game_started_at >= self._auto_relogin_number(
                        "game_ready_timeout", 60.0, minimum=1.0):
                    self._fail_auto_relogin(
                        "gameplay evidence timed out after Start Game"
                    )
                    return True
                return False
            self._begin_auto_relogin_confirmation(
                page, location, now, frame_token
            )
            state = "confirming"

        elif state == "waiting_page":
            page, location = self._find_known_auto_relogin_page()
            allowed_pages = {
                getattr(self, "_auto_relogin_expected_page", None),
                getattr(self, "_auto_relogin_last_action_page", None),
                "disconnect",
            }
            allowed_pages.discard(None)
            if page is not None and page not in allowed_pages:
                self._fail_auto_relogin(
                    f"unexpected {page} page while waiting for "
                    f"{getattr(self, '_auto_relogin_expected_page', None)}"
                )
                return True

            if page is None:
                # Only a Start Game dispatch may transition directly to play.
                # Earlier blank/loading pages must never bypass the remaining
                # visually confirmed login steps on a minimap false positive.
                start_game_attempted = getattr(
                    self, "_auto_relogin_last_action_page", None
                ) == "character"
                if start_game_attempted and \
                        self._auto_relogin_current_gameplay_evidence() is not None:
                    self._auto_relogin_state = "waiting_game"
                    self._auto_relogin_waiting_game_started_at = now
                    self._auto_relogin_ready_count = 0
                    self._auto_relogin_last_ready_frame_token = None
                    logger.info(
                        "[auto_relogin] Gameplay appeared during a page "
                        "transition; waiting for consecutive-frame evidence"
                    )
                    return False
                return True

            self._begin_auto_relogin_confirmation(
                page, location, now, frame_token
            )
            state = "confirming"

        if state != "confirming":
            self._fail_auto_relogin(f"unexpected recovery state {state!r}")
            return True

        page = getattr(self, "_auto_relogin_pending_page", None)
        classified_page, classified_location = \
            self._find_known_auto_relogin_page()
        location = classified_location if classified_page == page else None
        if classified_page is not None and classified_page != page:
            expected_page = getattr(
                self, "_auto_relogin_expected_page", None
            )
            if getattr(
                    self, "_auto_relogin_has_attempted_input", False
                    ) and getattr(
                        self, "_auto_relogin_last_action_page", None
                    ) == page and classified_page == expected_page:
                self._complete_auto_relogin_page_action(
                    page, now, expected_page
                )
                return True
            self._fail_auto_relogin(
                f"classified {classified_page} while confirming {page}"
            )
            return True
        if location is None:
            if frame_token != getattr(
                    self, "_auto_relogin_last_confirm_frame_token", None):
                self._reset_auto_relogin_ocr_gate()
                self._auto_relogin_confirm_miss_count = getattr(
                    self, "_auto_relogin_confirm_miss_count", 0
                ) + 1
                # Confirmation frames must be consecutive. A recovered match
                # starts both the frame count and dwell timer from scratch.
                self._auto_relogin_confirm_count = 0
                self._auto_relogin_confirm_started_at = None
                self._auto_relogin_last_confirm_frame_token = frame_token
            allowed_misses = max(
                1,
                int(round(self._auto_relogin_number(
                    "cancel_confirm_misses", 2, minimum=1
                ))),
            )
            if getattr(
                    self, "_auto_relogin_confirm_miss_count", 0
                    ) < allowed_misses:
                return True

            return_state = getattr(
                self, "_auto_relogin_confirmation_return_state", "waiting_page"
            )
            self._auto_relogin_pending_page = None
            self._auto_relogin_pending_location = None
            self._auto_relogin_confirmation_return_state = None
            self._auto_relogin_confirm_count = 0
            self._auto_relogin_confirm_miss_count = 0
            self._auto_relogin_confirm_started_at = None
            self._auto_relogin_last_confirm_frame_token = None
            if return_state == "idle" and not getattr(
                    self, "_auto_relogin_has_attempted_input", False):
                self._cancel_auto_relogin_confirmation()
                return False
            if return_state == "waiting_game":
                self._auto_relogin_state = "waiting_game"
                return False
            if return_state == "waiting_queue":
                self._auto_relogin_state = "waiting_queue"
                return True
            self._auto_relogin_state = "waiting_page"
            return True

        self._begin_auto_relogin_confirmation(
            page, location, now, frame_token
        )
        if not self._auto_relogin_confirmation_ready(now):
            return True

        self._execute_auto_relogin_page_action(
            page,
            getattr(self, "_auto_relogin_pending_location", location),
            now,
            frame_token,
        )
        return True

    def _resume_gameplay_after_auto_relogin(self):
        """Restore gameplay producers after current minimap detection."""
        mode = self.cfg.get("bot", {}).get("mode", "normal")
        resume_state = {
            "normal": "hunting",
            "patrol": "patrol",
            "aux": "aux",
        }.get(mode, "hunting")
        self.fsm.set_init_state(resume_state)

        now = time.time()
        self.t_last_attack = now
        self.t_watch_dog = now
        self.t_last_minimap_update = now
        self.red_dot_center_prev = None
        self.is_first_frame = True
        self.screen_player_location_valid = False
        self._reset_health_monitor_after_auto_relogin()
        self._restore_health_after_auto_relogin()
        self._reset_auto_relogin_runtime()
        self._restore_gameplay_minimap_input_gate()
        self._resume_keyboard_after_auto_relogin()
        logger.info(
            f"[auto_relogin] Minimap restored; resumed {resume_state} state"
        )

    def _gate_auto_relogin_until_game_ready(self, player_location):
        """Keep gameplay stopped until consecutive fresh player dots appear."""
        if getattr(self, "_auto_relogin_state", "idle") not in {
                "waiting_game", "failed"}:
            return False

        if player_location is None:
            self._auto_relogin_ready_count = 0
            self._auto_relogin_last_ready_frame_token = None
            return True

        frame_token = self._auto_relogin_frame_token(time.monotonic())
        if frame_token == getattr(
                self, "_auto_relogin_last_ready_frame_token", None):
            return True
        self._auto_relogin_last_ready_frame_token = frame_token
        self._auto_relogin_ready_count = \
            getattr(self, "_auto_relogin_ready_count", 0) + 1
        required_frames = max(
            1,
            int(round(self._auto_relogin_number(
                "game_ready_confirm_frames", 2, minimum=1
            ))),
        )
        if self._auto_relogin_ready_count < required_frames:
            return True

        self._resume_gameplay_after_auto_relogin()
        # Start gameplay on the next completely fresh frame rather than using
        # screen/player fields that were collected during the recovery gate.
        return True

    def _auto_relogin_minimap_structure_valid(self, detected_result):
        """Confirm the detected minimap occupies the saved/expected region."""
        if detected_result is None:
            return False
        geometry = getattr(self, "minimap_geometry", None)
        if geometry is None:
            return True

        # Recovery runs before apply_saved_minimap_geometry() in run_once.
        # On the first frame img_minimap_screen is therefore still the 10x10
        # constructor placeholder. Derive the expected rectangle directly from
        # persisted geometry so a valid startup minimap can break the recovery
        # gate instead of deadlocking before that crop is initialized.
        frame = getattr(self, "img_frame", None)
        if frame is None:
            return False
        try:
            expected_x, expected_y, expected_w, expected_h = \
                scale_minimap_rect(geometry, frame.shape[:2])
        except (TypeError, ValueError, KeyError):
            return False

        detected_x, detected_y, detected_w, detected_h = map(
            int, detected_result
        )
        # The legacy detector includes the one-pixel white border while saved
        # geometry stores the border-free raster used by route matching.
        detected_x += 1
        detected_y += 1
        detected_w -= 2
        detected_h -= 2
        if min(detected_w, detected_h) <= 0:
            return False

        tolerance_x = max(4, int(round(expected_w * 0.10)))
        tolerance_y = max(4, int(round(expected_h * 0.10)))
        return (
            abs(detected_x - expected_x) <= tolerance_x
            and abs(detected_y - expected_y) <= tolerance_y
            and abs(detected_w - expected_w) <= tolerance_x
            and abs(detected_h - expected_h) <= tolerance_y
        )

    def suspend_input_for_capture_loss(self):
        """Fail closed once when the capture stream becomes stale."""
        if self.kb is None or getattr(self, "_input_suspended_for_capture", False):
            return False
        self._reset_ladder_route_hold()
        self._reset_stationary_jump_proximity()
        self._reset_rope_climb(clear_locks=True)
        self._reset_portal_sweep()
        self._cancel_wz_timed_directional_jump()
        navigator = getattr(self, "wz_navigation", None)
        suspend_platform = getattr(
            navigator, "suspend_platform_navigation", None
        )
        if callable(suspend_platform):
            suspend_platform()
        self.kb.set_command("none none none")
        if hasattr(self.kb, "set_capture_available"):
            self.kb.set_capture_available(False)
        else:
            self.kb.disable()
        self._input_suspended_for_capture = True
        logger.warning("[capture] Input suspended because video frames stopped")
        return True

    def resume_input_after_capture(self):
        """Resume only a suspension caused by capture loss, not a user pause."""
        if self.kb is None or not getattr(
            self, "_input_suspended_for_capture", False
        ):
            return False
        if hasattr(self.kb, "set_capture_available"):
            self.kb.set_capture_available(True)
        else:
            self.kb.enable()
        self._input_suspended_for_capture = False
        logger.info("[capture] Fresh video frames restored; input resumed")
        return True

    def _install_wz_navigation_resources(self):
        """Atomically replace hand-painted routes with generated WZ routes."""
        navigator = getattr(self, "wz_navigation", None)
        platform_fsm = bool(getattr(
            navigator, "platform_state_machine_active", False
        ))
        if navigator is None or navigator.map_bgr is None or (
                not navigator.routes_rgb and not platform_fsm):
            return False
        generation = int(navigator.resource_generation)
        if generation == getattr(
                self, "_wz_navigation_resource_generation", -1):
            return True

        previous_map_id = getattr(self, "_wz_navigation_map_id", None)
        self.img_map = navigator.map_bgr
        self.img_routes = list(navigator.routes_rgb)
        self.idx_routes = 0
        self.img_route = self.img_routes[0] if self.img_routes else None
        self.img_route_debug = None
        self._stationary_jump_targets_by_route = [
            self._find_stationary_jump_targets(route)
            for route in self.img_routes
        ]
        self._rope_climb_targets_by_route = [
            self._find_rope_climb_targets(route, navigator.route_config)
            for route in self.img_routes
        ]
        self._reset_ladder_route_hold()
        self._ladder_route_exit_confirmed_at = None
        self._route_loop_goal_key = None
        self._route_loop_goal_warned_keys = set()
        self._reset_stationary_jump_proximity()
        self._reset_rope_climb(clear_locks=True)
        self._reset_portal_sweep()
        self._reset_wz_combat_checkpoint()
        self._wz_navigation_resource_generation = generation
        self._wz_navigation_map_id = navigator.map_id
        logger.info(
            "[wz-navigation] Installed generated route resources: "
            f"map={navigator.map_id}, routes={len(self.img_routes)}, "
            f"generation={generation}, platform_fsm={platform_fsm}"
        )
        if previous_map_id is not None and previous_map_id != navigator.map_id:
            self._native_minimap_size = None
        return True

    def _select_initial_wz_route(self, player_navigation):
        """Start a new generated route set at the leg nearest to Hero."""
        navigator = getattr(self, "wz_navigation", None)
        if navigator is None:
            return False
        generation = int(navigator.resource_generation)
        if generation == getattr(
                self, "_wz_navigation_route_selection_generation", -1):
            return True
        if bool(getattr(
                navigator, "platform_state_machine_active", False)):
            if not self.img_routes:
                self._wz_navigation_route_selection_generation = generation
                return True
            # The temporary path starts at a virtual node placed at Hero's
            # current grounded x. Never skip ahead to an overlapping leg.
            route_index = 0
        else:
            route_index = navigator.nearest_route_index(player_navigation)
        if route_index is None:
            return False
        self.idx_routes = int(route_index)
        self.img_route = self.img_routes[self.idx_routes]
        if getattr(self, "is_show_debug_window", False):
            self.img_route_debug = cv2.cvtColor(
                self.img_route, cv2.COLOR_RGB2BGR
            )
        self._wz_navigation_route_selection_generation = generation
        logger.info(
            "[wz-navigation] Selected initial "
            f"route{self.idx_routes + 1} near Hero at "
            f"({int(round(player_navigation[0]))}, "
            f"{int(round(player_navigation[1]))})"
        )
        return True

    def _record_wz_navigation_sample(self, player_navigation, sampled_at):
        """Retain actual navigation cadence and displacement for edge braking."""
        try:
            point = tuple(float(value) for value in player_navigation[:2])
            sampled_at = float(sampled_at)
        except (TypeError, ValueError):
            return
        if len(point) < 2 or not all(math.isfinite(value) for value in point) \
                or not math.isfinite(sampled_at):
            return

        previous_at = getattr(self, "_wz_navigation_sample_at", None)
        previous = getattr(self, "_wz_navigation_sample_position", None)
        interval = None
        step_x = 0.0
        if previous_at is not None and isinstance(
                previous, (tuple, list)) and len(previous) >= 2:
            elapsed = sampled_at - float(previous_at)
            if elapsed > 0:
                # Capture loss releases all gameplay input. Capping one valid
                # interval at the measured 1.2 s run distance prevents a long
                # pause from turning the whole platform into a launch band.
                interval = min(1.2, elapsed)
                if abs(point[1] - float(previous[1])) <= 3.0:
                    step_x = abs(point[0] - float(previous[0]))

        self._wz_navigation_sample_at = sampled_at
        self._wz_navigation_sample_position = point
        self._wz_navigation_sample_interval_seconds = interval
        self._wz_navigation_sample_step_x = step_x

    def _update_wz_navigation_from_minimap(self, player_minimap):
        """Register one live minimap and update the stable route coordinate."""
        navigator = getattr(self, "wz_navigation", None)
        if navigator is None:
            return False
        # A catalog scan can take several seconds. Stop before entering it, and
        # also after any failed registration, so stale movement can never span
        # a map transition or an ambiguous match.
        if not navigator.active and self.kb is not None:
            self.kb.set_command("none none none")
        previous_command = (
            self.cmd_move_x,
            self.cmd_move_y,
            self.cmd_action,
        )
        rope_state = getattr(self, "_rope_climb_state", None)
        rope_ascent_tracking = bool(
            getattr(self, "_rope_climb_active", False)
            and isinstance(rope_state, dict)
            and rope_state.get("phase") in {
                "mount_request", "mounting", "climbing"
            }
        )
        navigation_on_ladder = bool(
            getattr(self, "is_on_ladder", False)
            or rope_ascent_tracking
        )
        allow_motion_rebuild = not (
            getattr(self, "_rope_climb_active", False)
            or getattr(self, "_portal_sweep_active", False)
            or getattr(self, "is_on_ladder", False)
        )
        navigation_sample_at = time.monotonic()
        update = navigator.update(
            self.img_minimap,
            player_minimap,
            # A rope mount must use raw minimap motion from the jump request
            # through the ascent. Ground snapping can otherwise move Hero to
            # the upper foothold before the sprite physically reaches it and
            # make the climb state release Up halfway along the rope.
            on_ladder=navigation_on_ladder,
            previous_command=previous_command,
            allow_motion_rebuild=allow_motion_rebuild,
            timestamp=navigation_sample_at,
        )
        if update is None:
            self._wz_jump_move_x = "none"
            self.cmd_move_x = "none"
            self.cmd_move_y = "none"
            self.cmd_action = "none"
            if self.kb is not None:
                self.kb.set_command("none none none")
            return False
        if navigator.jump_active:
            if previous_command[0] in {"left", "right"}:
                self._wz_jump_move_x = previous_command[0]
        else:
            self._wz_jump_move_x = "none"
        if not self._install_wz_navigation_resources():
            return False
        self._record_wz_navigation_sample(
            update.player_navigation,
            navigation_sample_at,
        )
        self.loc_player_global = tuple(
            int(round(value)) for value in update.player_navigation
        )
        self._select_initial_wz_route(update.player_navigation)
        return True

    def load_config(self, cfg):
        '''
        load_config
        '''
        # Loading/config normalization historically mutates several nested
        # values.  Keep the caller's merged config untouched so this object can
        # always derive a fresh runtime config without accumulating scale.
        cfg = deepcopy(cfg)
        mode = cfg["bot"]["mode"]
        monster_cfg = cfg.get("monster_detect", {})
        monster_backend = str(
            monster_cfg.get("backend", "template")
        ).lower()
        marker_cfg = cfg.get("nametag", {}).get("overhead_marker", {})
        marker_backend = str(marker_cfg.get("backend", "template")).lower()
        map_name = str(cfg.get("bot", {}).get("map", "")).strip()
        wz_navigation_cfg = cfg.get("wz_navigation", {})
        if not isinstance(wz_navigation_cfg, dict):
            logger.error("[load_config] wz_navigation must be a mapping")
            return -1
        configured_wz_enable = wz_navigation_cfg.get("enable", False)
        if not isinstance(configured_wz_enable, bool):
            logger.error("[load_config] wz_navigation.enable must be boolean")
            return -1
        projectile_terrain_check = wz_navigation_cfg.get(
            "projectile_terrain_check", False
        )
        if not isinstance(projectile_terrain_check, bool):
            logger.error(
                "[load_config] wz_navigation.projectile_terrain_check must "
                "be boolean"
            )
            return -1
        if projectile_terrain_check and \
                cfg.get("bot", {}).get("attack") != "directional":
            logger.error(
                "[load_config] projectile terrain navigation requires "
                "bot.attack=directional"
            )
            return -1
        for field_name, minimum, inclusive in (
            ("projectile_height_wz", 0.0, False),
            ("projectile_clearance_wz", 0.0, True),
        ):
            value = wz_navigation_cfg.get(
                field_name,
                30.0 if field_name == "projectile_height_wz" else 8.0,
            )
            if isinstance(value, bool) or not isinstance(
                    value, (int, float)):
                invalid_range = True
            else:
                invalid_range = (
                    value < minimum if inclusive else value <= minimum
                )
            if invalid_range:
                qualifier = "non-negative" if inclusive else "positive"
                logger.error(
                    f"[load_config] wz_navigation.{field_name} must be a "
                    f"{qualifier} number"
                )
                return -1
        # A virtual WZ selection or an explicit recorded-map binding is already
        # an unambiguous request for WZ mode.  Do not let a stale UI checkbox
        # silently fall back to hand-painted route images.
        wz_navigation_enabled, bound_wz_map_id = \
            _resolve_wz_navigation_request(
                mode, map_name, wz_navigation_cfg
            )
        if (
            wz_navigation_enabled
            and bound_wz_map_id is not None
            and not configured_wz_enable
        ):
            logger.info(
                "[load_config] WZ navigation enabled by map binding: "
                f"{map_name} -> {bound_wz_map_id}"
            )
        if wz_navigation_enabled and monster_backend != "yolo":
            logger.error(
                "[load_config] WZ navigation requires YOLO monster detection "
                "because the matched map can change at runtime"
            )
            return -1
        if mode not in {"normal", "aux", "patrol", "debug"}:
            logger.error(f"[load_config] Unsupported bot mode: {mode}")
            return -1
        if monster_backend not in {"template", "yolo"}:
            logger.error(
                "[load_config] Unsupported monster detection backend: "
                f"{monster_backend}"
            )
            return -1
        if marker_backend not in {"template", "yolo"}:
            logger.error(
                "[load_config] Unsupported Hero detection backend: "
                f"{marker_backend}"
            )
            return -1
        if marker_cfg.get("enable", False) and marker_backend == "yolo":
            hero_yolo_cfg = marker_cfg.get("yolo", {})
            if not isinstance(hero_yolo_cfg, dict):
                logger.error(
                    "[load_config] nametag.overhead_marker.yolo must be a mapping"
                )
                return -1
            class_name = hero_yolo_cfg.get("class_name", "hero")
            confidence = hero_yolo_cfg.get("confidence", 0.4)
            anchor = hero_yolo_cfg.get("player_anchor", (0.5, 0.5))
            offset = hero_yolo_cfg.get("player_offset", (0, 0))
            if not isinstance(class_name, str) or not class_name.strip():
                logger.error(
                    "[load_config] nametag.overhead_marker.yolo.class_name "
                    "must be non-empty"
                )
                return -1
            if isinstance(confidence, bool) or not isinstance(
                    confidence, (int, float)) or not 0 <= confidence <= 1:
                logger.error(
                    "[load_config] nametag.overhead_marker.yolo.confidence "
                    "must be between 0 and 1"
                )
                return -1
            if not isinstance(anchor, (list, tuple)) or len(anchor) != 2 or \
                    any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not 0 <= value <= 1
                        for value in anchor
                    ):
                logger.error(
                    "[load_config] nametag.overhead_marker.yolo.player_anchor "
                    "must be two ratios between 0 and 1"
                )
                return -1
            if not isinstance(offset, (list, tuple)) or len(offset) != 2 or \
                    any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        for value in offset
                    ):
                logger.error(
                    "[load_config] nametag.overhead_marker.yolo.player_offset "
                    "must be [x, y]"
                )
                return -1
            max_stale_frames = marker_cfg.get("max_stale_frames", -1)
            if isinstance(max_stale_frames, bool) or not isinstance(
                    max_stale_frames, int) or max_stale_frames < -1:
                logger.error(
                    "[load_config] nametag.overhead_marker.max_stale_frames "
                    "must be -1 (unlimited) or a non-negative integer"
                )
                return -1
            if not isinstance(
                    marker_cfg.get("require_minimap_player", True), bool):
                logger.error(
                    "[load_config] nametag.overhead_marker."
                    "require_minimap_player must be true or false"
                )
                return -1

        try:
            validate_absolute_mouse_config(cfg)
        except ValueError as exc:
            logger.error(f"[load_config] {exc}")
            return -1

        auto_relogin_cfg = cfg.get("auto_relogin")
        if auto_relogin_cfg is not None:
            if not isinstance(auto_relogin_cfg, dict):
                logger.error("[load_config] auto_relogin must be a mapping")
                return -1
            if not isinstance(
                    auto_relogin_cfg.get("enable", False), bool):
                logger.error(
                    "[load_config] auto_relogin.enable must be true or false"
                )
                return -1
            numeric_fields = {
                "confirm_frames": (1.0, True),
                "confirm_seconds": (0.0, False),
                "cancel_confirm_misses": (1.0, True),
                "input_retry_delay": (0.0, False),
                "retry_cooldown": (0.0, False),
                "step_timeout": (1.0, False),
                "game_ready_timeout": (1.0, False),
                "max_recovery_duration": (1.0, False),
                "max_step_attempts": (1.0, True),
                "game_ready_confirm_frames": (1.0, True),
                "mouse_click_duration": (0.001, False),
                "connect_enter_retry_delay": (0.0, False),
                "connect_enter_max_attempts": (1.0, True),
                "focus_switch_hold": (0.001, False),
                "focus_switch_settle_delay": (0.0, False),
                "focus_enter_duration": (0.001, False),
                "channel_click_count": (1.0, True),
                "channel_double_click_interval": (0.0, False),
                "cursor_min_score": (0.0, False),
                "cursor_uniqueness_margin": (0.0, False),
                "cursor_mask_erode_pixels": (0.0, True),
                "cursor_min_visible_fraction": (0.001, False),
                "cursor_min_visible_pixels": (1.0, True),
                "mouse_move_gain": (0.001, False),
                "mouse_max_delta": (1.0, True),
                "mouse_target_confirm_frames": (1.0, True),
                "mouse_probe_delta": (2.0, True),
                "mouse_response_min_pixels": (1.0, False),
                "mouse_response_min_cosine": (0.001, False),
                "mouse_feedback_delay": (0.0, False),
                "mouse_feedback_frames": (1.0, True),
                "mouse_click_frame_max_age": (0.05, False),
                "mouse_pointer_timeout": (1.0, False),
                "mouse_max_moves": (1.0, True),
                "mouse_cursor_miss_limit": (1.0, True),
                "mouse_cursor_rescue_search_width": (1.0, True),
                "mouse_page_miss_limit": (1.0, True),
                "mouse_stall_limit": (1.0, True),
                "mouse_hover_template_correlation": (0.0, False),
            }
            for field_name, (minimum, integer_only) in numeric_fields.items():
                if field_name not in auto_relogin_cfg:
                    continue
                value = auto_relogin_cfg[field_name]
                if isinstance(value, bool) or not isinstance(
                        value, (int, float)) or not np.isfinite(value) or \
                        value < minimum or (
                            integer_only and float(value) != int(value)
                        ):
                    kind = "integer" if integer_only else "number"
                    logger.error(
                        f"[load_config] auto_relogin.{field_name} must be a "
                        f"finite {kind} >= {minimum:g}"
                    )
                    return -1
            remote_mouse_mode = str(auto_relogin_cfg.get(
                "remote_mouse_mode", "absolute"
            )).strip().lower()
            if remote_mouse_mode not in {"absolute", "visual_relative"}:
                logger.error(
                    "[load_config] auto_relogin.remote_mouse_mode must be "
                    "absolute or visual_relative"
                )
                return -1
            if auto_relogin_cfg.get("enable", False) and \
                    cfg.get("esp32_hid", {}).get("remote_target", False) and \
                    remote_mouse_mode == "absolute" and not \
                    has_calibrated_absolute_mouse(cfg):
                logger.error(
                    "[load_config] Remote absolute auto_relogin requires "
                    "capture_frame_is_desktop or magpie_source_rect "
                    "calibration"
                )
                return -1
            bounded_unit_fields = (
                "cursor_min_score",
                "cursor_uniqueness_margin",
                "cursor_min_visible_fraction",
                "mouse_hover_template_correlation",
                "mouse_response_min_cosine",
            )
            for field_name in bounded_unit_fields:
                if field_name not in auto_relogin_cfg:
                    continue
                value = float(auto_relogin_cfg[field_name])
                lower_ok = value > 0.0
                if not lower_ok or value > 1.0:
                    logger.error(
                        f"[load_config] auto_relogin.{field_name} must be "
                        "in the range 0..1"
                    )
                    return -1
            if int(auto_relogin_cfg.get("mouse_max_delta", 64)) > 127:
                logger.error(
                    "[load_config] auto_relogin.mouse_max_delta must be <= 127"
                )
                return -1
            if int(auto_relogin_cfg.get("mouse_probe_delta", 8)) > 127:
                logger.error(
                    "[load_config] auto_relogin.mouse_probe_delta must be <= 127"
                )
                return -1
            if int(auto_relogin_cfg.get("channel_click_count", 2)) > 2:
                logger.error(
                    "[load_config] auto_relogin.channel_click_count must be "
                    "1 or 2"
                )
                return -1

            for reference_name, default_reference in (
                    ("template_reference_size", (700, 1296)),
                    ("flow_template_reference_size", (2160, 3840))):
                reference = auto_relogin_cfg.get(
                    reference_name, default_reference
                )
                if not isinstance(reference, (list, tuple)) or \
                        len(reference) != 2 or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < 1
                            for value in reference
                        ):
                    logger.error(
                        f"[load_config] auto_relogin.{reference_name} must be "
                        "[positive height, positive width]"
                    )
                    return -1

            required_pages = {
                "disconnect", "connect", "world", "channel", "character"
            }
            recognized_pages = required_pages | {"queue"}
            flow_reference = auto_relogin_cfg.get(
                "flow_template_reference_size", (2160, 3840)
            )
            reference_h, reference_w = flow_reference

            if auto_relogin_cfg.get("enable", False):
                ocr_cfg = auto_relogin_cfg.get("ocr", {})
                if not isinstance(ocr_cfg, dict) or not isinstance(
                        ocr_cfg.get("enable", False), bool):
                    logger.error(
                        "[load_config] auto_relogin.ocr must be a mapping "
                        "with a boolean enable value"
                    )
                    return -1
                if not ocr_cfg.get("enable", False):
                    logger.error(
                        "[load_config] Enabled auto_relogin requires Chinese "
                        "OCR; template page detection is not supported"
                    )
                    return -1
                if ocr_cfg.get("enable", False):
                    idle_scan_interval = ocr_cfg.get(
                        "idle_scan_interval", 1.0
                    )
                    if isinstance(idle_scan_interval, bool) or not isinstance(
                            idle_scan_interval, (int, float)) or not np.isfinite(
                                idle_scan_interval
                            ) or float(idle_scan_interval) < 0.1:
                        logger.error(
                            "[load_config] auto_relogin.ocr."
                            "idle_scan_interval must be a finite number >= 0.1"
                        )
                        return -1
                    min_score = ocr_cfg.get("min_score", 0.85)
                    if isinstance(min_score, bool) or not isinstance(
                            min_score, (int, float)) or not np.isfinite(
                                min_score
                            ) or not 0.0 < float(min_score) <= 1.0:
                        logger.error(
                            "[load_config] auto_relogin.ocr.min_score must "
                            "be a finite number in (0, 1]"
                        )
                        return -1
                    box_threshold = ocr_cfg.get("box_threshold", 0.3)
                    if isinstance(box_threshold, bool) or not isinstance(
                            box_threshold, (int, float)) or not np.isfinite(
                                box_threshold
                            ) or not 0.0 < float(box_threshold) <= 1.0:
                        logger.error(
                            "[load_config] auto_relogin.ocr.box_threshold "
                            "must be a finite number in (0, 1]"
                        )
                        return -1
                    max_frame_age = ocr_cfg.get("max_frame_age", 1.0)
                    if isinstance(max_frame_age, bool) or not isinstance(
                            max_frame_age, (int, float)) or not np.isfinite(
                                max_frame_age
                            ) or float(max_frame_age) <= 0.0:
                        logger.error(
                            "[load_config] auto_relogin.ocr.max_frame_age "
                            "must be a positive finite number"
                        )
                        return -1
                    ocr_confirm_frames = ocr_cfg.get("confirm_frames", 2)
                    if isinstance(ocr_confirm_frames, bool) or not isinstance(
                            ocr_confirm_frames, int) or \
                            ocr_confirm_frames < 1:
                        logger.error(
                            "[load_config] auto_relogin.ocr.confirm_frames "
                            "must be a positive integer"
                        )
                        return -1
                    center_drift = ocr_cfg.get(
                        "max_center_drift", (24, 24)
                    )
                    if not isinstance(center_drift, (list, tuple)) or \
                            len(center_drift) != 2 or any(
                                isinstance(value, bool)
                                or not isinstance(value, int)
                                or value < 0
                                for value in center_drift
                            ):
                        logger.error(
                            "[load_config] auto_relogin.ocr."
                            "max_center_drift must be [non-negative x, y]"
                        )
                        return -1
                    ocr_targets = ocr_cfg.get("targets")
                    if not isinstance(ocr_targets, dict) or not ocr_targets:
                        logger.error(
                            "[load_config] auto_relogin.ocr.targets must be "
                            "a non-empty mapping"
                        )
                        return -1
                    missing_targets = required_pages.difference(ocr_targets)
                    if missing_targets:
                        logger.error(
                            "[load_config] auto_relogin.ocr.targets must "
                            "define all five recovery pages; missing "
                            f"{sorted(missing_targets)}"
                        )
                        return -1
                    for page, target_cfg in ocr_targets.items():
                        if page not in recognized_pages or not isinstance(
                                target_cfg, dict):
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets keys "
                                "must be recovery pages with mapping values"
                            )
                            return -1
                        target_texts = target_cfg.get("texts")
                        if not isinstance(target_texts, (list, tuple)) or \
                                not target_texts or any(
                                    not isinstance(text, str)
                                    or not text.strip()
                                    for text in target_texts
                                ):
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets."
                                f"{page}.texts must contain non-empty strings"
                            )
                            return -1
                        if any(
                                not is_chinese_ocr_target(text)
                                for text in target_texts):
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets."
                                f"{page}.texts must contain Chinese Han text "
                                "and no Hangul/kana"
                            )
                            return -1
                        if str(target_cfg.get(
                                "region_source", "configured"
                                )).strip().lower() != "configured":
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets."
                                f"{page}.region_source must be configured; "
                                "page templates are disabled"
                            )
                            return -1
                        target_region = target_cfg.get("search_region")
                        if not isinstance(target_region, (list, tuple)) or \
                                len(target_region) != 4 or any(
                                    isinstance(coord, bool)
                                    or not isinstance(coord, int)
                                    for coord in target_region
                                ) or not (
                                    0 <= target_region[0] < target_region[2]
                                    <= reference_w
                                    and 0 <= target_region[1]
                                    < target_region[3] <= reference_h
                                ):
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets."
                                f"{page}.search_region must be inside the "
                                "flow reference frame"
                            )
                            return -1
                        if str(target_cfg.get(
                                 "match_mode", "exact"
                                 )).strip().lower() not in {
                                     "exact", "contains", "partial"
                                 }:
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets."
                                f"{page}.match_mode must be exact, contains, "
                                "or partial"
                            )
                            return -1
                        target_action = str(target_cfg.get(
                            "action", "click"
                        )).strip().lower()
                        if target_action not in {
                                "click", "enter",
                                "focus_next_enter", "wait"}:
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets."
                                f"{page}.action must be click, enter, "
                                "focus_next_enter, or wait"
                            )
                            return -1
                        allowed_actions = {
                            "disconnect": {"enter"},
                            "connect": {
                                "click", "enter", "focus_next_enter"
                            },
                            "world": {"click"},
                            "channel": {"click"},
                            "queue": {"wait"},
                            "character": {"click"},
                        }[page]
                        if target_action not in allowed_actions:
                            logger.error(
                                "[load_config] auto_relogin.ocr.targets."
                                f"{page}.action must be one of "
                                f"{sorted(allowed_actions)}"
                            )
                            return -1
                if "channel_points" in auto_relogin_cfg:
                    logger.error(
                        "[load_config] auto_relogin.channel_points is obsolete; "
                        "the channel action clicks a live 频道N OCR target"
                    )
                    return -1
                if remote_mouse_mode == "visual_relative":
                    cursor_template = auto_relogin_cfg.get(
                        "cursor_template"
                    )
                    if not isinstance(cursor_template, str) or not \
                            cursor_template.strip():
                        logger.error(
                            "[load_config] auto_relogin.cursor_template must "
                            "be a non-empty path in visual_relative mode"
                        )
                        return -1

                    def valid_pair(value, *, allow_zero=False):
                        minimum = 0 if allow_zero else 1
                        return isinstance(value, (list, tuple)) and \
                            len(value) == 2 and all(
                                not isinstance(coord, bool)
                                and isinstance(coord, int)
                                and coord >= minimum
                                for coord in value
                            )

                    for field_name, default_value in (
                            ("cursor_hotspot", (13, 6)),
                            (
                                "disconnect_cursor_hotspot",
                                (9, 4),
                            )):
                        if not valid_pair(
                                auto_relogin_cfg.get(
                                    field_name, default_value
                                ),
                                allow_zero=True):
                            logger.error(
                                f"[load_config] auto_relogin.{field_name} "
                                "must be [non-negative x, non-negative y]"
                            )
                            return -1
                    disconnect_cursor_template = auto_relogin_cfg.get(
                        "disconnect_cursor_template"
                    )
                    if not isinstance(disconnect_cursor_template, str) or \
                            not disconnect_cursor_template.strip():
                        logger.error(
                            "[load_config] auto_relogin."
                            "disconnect_cursor_template must be a non-empty "
                            "path in visual_relative mode"
                        )
                        return -1
                    for field_name, default_value in (
                            ("cursor_local_search_radius", (450, 450)),
                            ("mouse_target_tolerance", (18, 18)),
                            ("mouse_target_drift", (50, 50)),
                            ("mouse_uncommanded_jump_tolerance", (6, 6))):
                        if not valid_pair(auto_relogin_cfg.get(
                                field_name, default_value)):
                            logger.error(
                                f"[load_config] auto_relogin.{field_name} "
                                "must be [positive x, positive y]"
                            )
                            return -1
                    cursor_region = auto_relogin_cfg.get(
                        "cursor_search_region"
                    )
                    if cursor_region is not None and (
                            not isinstance(cursor_region, (list, tuple))
                            or len(cursor_region) != 4
                            or any(
                                isinstance(coord, bool)
                                or not isinstance(coord, int)
                                for coord in cursor_region
                            )
                            or not (
                                0 <= cursor_region[0] < cursor_region[2]
                                <= reference_w
                                and 0 <= cursor_region[1] < cursor_region[3]
                                <= reference_h
                            )):
                        logger.error(
                            "[load_config] auto_relogin.cursor_search_region "
                            "must be [x0, y0, x1, y1] inside the flow frame"
                        )
                        return -1
                    rescue_deltas = auto_relogin_cfg.get(
                        "mouse_cursor_rescue_deltas",
                        (
                            (4096, 0),
                            (4096, 0),
                            (-127, 0),
                            (0, -64),
                            (0, 128),
                            (0, -192),
                            (0, 256),
                            (0, -320),
                            (0, 384),
                            (0, -448),
                            (0, 512),
                        ),
                    )
                    if not isinstance(rescue_deltas, (list, tuple)) or \
                            not rescue_deltas or any(
                                not isinstance(delta, (list, tuple))
                                or len(delta) != 2
                                or any(
                                    isinstance(coord, bool)
                                    or not isinstance(coord, int)
                                    or not -RELATIVE_MOUSE_MAX_DELTA <= coord
                                    <= RELATIVE_MOUSE_MAX_DELTA
                                    for coord in delta
                                )
                                or tuple(delta) == (0, 0)
                                for delta in rescue_deltas
                            ):
                        logger.error(
                            "[load_config] auto_relogin."
                            "mouse_cursor_rescue_deltas must contain non-zero "
                            "[dx, dy] pairs in "
                            f"-{RELATIVE_MOUSE_MAX_DELTA}.."
                            f"{RELATIVE_MOUSE_MAX_DELTA}"
                        )
                        return -1
            remote_confirm_key = auto_relogin_cfg.get(
                "remote_confirm_key", "enter"
            )
            if not isinstance(remote_confirm_key, str) or not \
                    remote_confirm_key.strip():
                logger.error(
                    "[load_config] auto_relogin.remote_confirm_key must be "
                    "a non-empty key name"
                )
                return -1
            try:
                usage_from_text(remote_confirm_key.strip())
            except ValueError as exc:
                logger.error(
                    "[load_config] Invalid auto_relogin.remote_confirm_key "
                    f"{remote_confirm_key!r}: {exc}"
                )
                return -1
            focus_switch_keys = auto_relogin_cfg.get(
                "focus_switch_keys", ("alt", "tab")
            )
            if not isinstance(focus_switch_keys, (list, tuple)) or not \
                    focus_switch_keys or len(focus_switch_keys) > 7 or any(
                        not isinstance(key, str) or not key.strip()
                        for key in focus_switch_keys
                    ):
                logger.error(
                    "[load_config] auto_relogin.focus_switch_keys must be a "
                    "non-empty list of at most seven key names"
                )
                return -1
            try:
                focus_usages = [
                    usage_from_text(key.strip()) for key in focus_switch_keys
                ]
            except ValueError as exc:
                logger.error(
                    "[load_config] Invalid auto_relogin.focus_switch_keys: "
                    f"{exc}"
                )
                return -1
            if len(set(focus_usages)) != len(focus_usages) or sum(
                    usage < 0xE0 for usage in focus_usages) > 6:
                logger.error(
                    "[load_config] auto_relogin.focus_switch_keys contains "
                    "duplicate keys or more than six ordinary keys"
                )
                return -1
        for field_name in ("min_box_width", "min_box_height"):
            value = monster_cfg.get(field_name, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or value < 0:
                logger.error(
                    f"[load_config] monster_detect.{field_name} must be a "
                    "non-negative number"
                )
                return -1

        directional_aoe_cfg = cfg.get("directional_aoe", {})
        directional_aoe_enabled = directional_aoe_cfg.get("enable", False)
        if not isinstance(directional_aoe_enabled, bool):
            logger.error("[load_config] directional_aoe.enable must be boolean")
            return -1
        if directional_aoe_enabled and cfg["bot"]["attack"] == "directional":
            min_monsters = directional_aoe_cfg.get("min_monsters")
            if isinstance(min_monsters, bool) or not isinstance(min_monsters, int) \
                    or min_monsters < 1:
                logger.error(
                    "[load_config] directional_aoe.min_monsters must be an "
                    "integer greater than or equal to 1"
                )
                return -1
            for field_name in ("range_x", "range_y"):
                value = directional_aoe_cfg.get(field_name)
                if isinstance(value, bool) or not isinstance(value, int) \
                        or value <= 0:
                    logger.error(
                        f"[load_config] directional_aoe.{field_name} must be "
                        "a positive integer"
                    )
                    return -1
            for field_name in ("cooldown",):
                value = directional_aoe_cfg.get(field_name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or value < 0:
                    logger.error(
                        f"[load_config] directional_aoe.{field_name} must be "
                        "a non-negative number"
                    )
                    return -1
            aoe_key = cfg.get("key", {}).get("aoe_skill", "")
            if not isinstance(aoe_key, str) or not aoe_key.strip():
                logger.error(
                    "[load_config] key.aoe_skill is required when "
                    "directional_aoe is enabled"
                )
                return -1

        power_knockback_cfg = cfg.get("power_knockback", {})
        power_knockback_enabled = power_knockback_cfg.get("enable", False)
        if not isinstance(power_knockback_enabled, bool):
            logger.error("[load_config] power_knockback.enable must be boolean")
            return -1
        if power_knockback_enabled and cfg["bot"]["attack"] == "directional":
            for field_name in ("trigger_distance_x", "range_y"):
                value = power_knockback_cfg.get(field_name)
                if isinstance(value, bool) or not isinstance(value, int) \
                        or value <= 0:
                    logger.error(
                        f"[load_config] power_knockback.{field_name} must be "
                        "a positive integer"
                    )
                    return -1
            for field_name in ("cooldown",):
                value = power_knockback_cfg.get(field_name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or value < 0:
                    logger.error(
                        f"[load_config] power_knockback.{field_name} must be "
                        "a non-negative number"
                    )
                    return -1
            knockback_key = cfg.get("key", {}).get("power_knockback", "")
            if not isinstance(knockback_key, str) or not knockback_key.strip():
                logger.error(
                    "[load_config] key.power_knockback is required when "
                    "power_knockback is enabled"
                )
                return -1

        # One UI controller can be paused and loaded again with another map or
        # mode. Never retain route or monster resources from that prior run.
        self.img_map = None
        self.img_route = None
        self.img_route_debug = None
        self.img_routes = []
        self._stationary_jump_targets_by_route = []
        self._rope_climb_targets_by_route = []
        self._reset_ladder_route_hold()
        self._ladder_route_exit_confirmed_at = None
        self._route_loop_goal_key = None
        self._route_loop_goal_warned_keys = set()
        self._reset_stationary_jump_proximity()
        self._reset_rope_climb(clear_locks=True)
        self._reset_portal_sweep()
        self.monsters_info = {}
        self.yolo_hero_detector = None
        self.yolo_inference_class_names = ("mob",)
        self.minimap_geometry = None
        self._native_minimap_size = None
        self._last_native_minimap_error = None
        self._last_native_minimap_log = None
        self._last_route_map_size_error = None
        self.wz_navigation = None
        self._wz_navigation_enabled = wz_navigation_enabled
        self._wz_navigation_resource_generation = -1
        self._wz_navigation_map_id = None
        self._wz_jump_move_x = "none"
        self._wz_attackable_monster_this_frame = False
        self._wz_combat_observation_valid = False
        self._reset_wz_combat_checkpoint()
        self.img_nametag = None
        self.img_nametag_gray = None
        self.img_nametag_medal = None
        self.img_nametag_medal_gray = None
        self.img_nametag_pet = None
        self.img_nametag_pet_gray = None
        self.nametag_appearance_templates = []
        self._img_nametag_source = None
        self._img_nametag_medal_source = None
        self._img_nametag_pet_source = None
        self.img_overhead_marker = None
        self.img_overhead_marker_gray = None
        self.img_overhead_marker_mask = None
        self._img_overhead_marker_source = None
        self.overhead_marker_component_bbox = None
        self._last_nametag_template_geometry = None
        self.loc_overhead_marker_player = (0, 0)
        self.has_valid_overhead_marker_location = False
        self.overhead_marker_miss_count = 0
        self.pending_overhead_marker_location = None
        self.pending_overhead_marker_count = 0
        self.t_last_overhead_marker_detected = None
        self.last_overhead_marker_match = None
        self.last_single_yolo_hero_location = None
        self.last_single_yolo_hero_frame_token = None
        self.last_yolo_hero_detection = None
        self.has_valid_appearance_location = False
        self.pending_appearance_location = None
        self.pending_appearance_count = 0
        self.last_appearance_match = None
        self._reset_auto_relogin_runtime()

        if mode == "debug":
            logger.info(
                "[load_config] Debug mode is vision-only; keyboard, mouse, "
                "health, buff, login, and channel workflows are disabled"
            )
        if cfg.get("esp32_hid", {}).get("remote_target", False):
            # Every remote game-UI click is routed through the configured
            # capture-to-desktop absolute HID transform.
            relogin_mouse_mode = str(cfg.get("auto_relogin", {}).get(
                "remote_mouse_mode", "absolute"
            )).strip().lower()
            calibrated = has_calibrated_absolute_mouse(cfg)
            logger.info(
                "[load_config] Remote HID mode: absolute game-UI mouse="
                f"{'calibrated' if calibrated else 'disabled'}, "
                f"auto_relogin mouse mode={relogin_mouse_mode}"
            )
            if not calibrated:
                logger.warning(
                    "[load_config] Remote absolute game-UI workflows are "
                    "disabled: esp32_hid.magpie_source_rect is not calibrated"
                )
                for section_name in (
                        "channel_change", "scheduled_channel_switching"):
                    section = cfg.get(section_name, {})
                    if section.get("enable", False):
                        logger.warning(
                            f"[load_config] Disabled {section_name}.enable"
                        )
                        section["enable"] = False

        # Parse color code in config
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        self.color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }

        map_dir = os.path.join("minimaps", map_name)
        if map_name != "__auto_wz__":
            self.minimap_geometry = load_minimap_geometry(map_dir)
        if self.minimap_geometry is None and map_name != "__auto_wz__":
            logger.warning(
                f"No minimap_geometry.txt for {map_name}; using legacy "
                "real-time minimap scanning until this map is re-recorded"
            )
        if (mode == "normal" and not wz_navigation_enabled) or (
            mode == "debug" and monster_backend == "template"
        ):
            # Check if the map is supported in config_data.yaml
            if map_name not in self.data["map_mobs_mapping"]:
                text = f"Invalid map name: {map_name}. "\
                        "Not supported in config/config_data.yaml."
                logger.error(text)
                return -1
                # raise RuntimeError(text)

        if mode == "normal":
            if wz_navigation_enabled:
                try:
                    self.wz_navigation = WzNavigationRuntime(
                        wz_navigation_cfg,
                        cfg["route"],
                        directional_attack_config=cfg["directional_attack"],
                        selected_map_name=map_name,
                        recorded_route_directory=map_dir,
                        log=logger.info,
                    )
                    # Fail at configuration time for a missing/empty export,
                    # instead of retrying an impossible scan in the main loop.
                    self.wz_navigation.catalog.entries(
                        self.wz_navigation.candidate_map_ids
                    )
                except (OSError, TypeError, ValueError,
                        WzMapRecognitionError) as exc:
                    logger.error(
                        f"[load_config] Unable to open WZ geometry cache: {exc}"
                    )
                    return -1

                seed_path = os.path.join(map_dir, "map.png")
                should_bootstrap = bool(
                    wz_navigation_cfg.get(
                        "bootstrap_from_selected_map", True
                    )
                )
                if should_bootstrap and os.path.isfile(seed_path):
                    try:
                        seed_map = load_image(seed_path, cv2.IMREAD_COLOR)
                        matched_map_id = self.wz_navigation.bootstrap(seed_map)
                        logger.info(
                            "[load_config] Existing map.png matched WZ map "
                            f"{matched_map_id}"
                        )
                    except (OSError, TypeError, ValueError,
                            WzMapRecognitionError) as exc:
                        logger.warning(
                            "[load_config] map.png could not bootstrap WZ "
                            f"navigation; live recognition will retry: {exc}"
                        )
            else:
                # Load hand-painted map/route images.
                self.img_map = load_image(
                    f"minimaps/{map_name}/map.png", cv2.IMREAD_COLOR
                )
                route_files = sorted(
                    glob.glob(f"minimaps/{map_name}/route*.png")
                )
                route_files = [
                    path for path in route_files
                    if not path.endswith("route_rest.png")
                ]
                for route_file in route_files:
                    img = cv2.cvtColor(load_image(route_file), cv2.COLOR_BGR2RGB)
                    img = mask_route_colors(
                        self.img_map, img, cfg["route"]["color_code"]
                    )
                    img = mask_route_colors(
                        self.img_map,
                        img,
                        cfg["route"]["color_code_up_down"],
                    )
                    self.img_routes.append(img)
                    self._stationary_jump_targets_by_route.append(
                        self._find_stationary_jump_targets(img)
                    )
                    self._rope_climb_targets_by_route.append(
                        self._find_rope_climb_targets(img, cfg["route"])
                    )

        if mode in {"normal", "debug"} and monster_backend == "template":
            # Load monsters images from monster/<monster_name>.
            for monster_name in self.data["map_mobs_mapping"][map_name]:
                imgs = []
                for file in glob.glob(f"monster/{monster_name}/{monster_name}*.png"):
                    # Add original image
                    img = load_image(file)
                    imgs.append((img, get_mask(img, (0, 255, 0))))
                    # Add flipped image
                    img_flip = cv2.flip(img, 1)
                    imgs.append((img_flip, get_mask(img_flip, (0, 255, 0))))
                if imgs:
                    self.monsters_info[monster_name] = imgs
                else:
                    logger.error(f"No images found in monster/{monster_name}/{monster_name}*")
                    return -1
                    # raise RuntimeError(f"No images found in monster/{monster_name}/{monster_name}*")
            logger.info(
                f"Loaded monsters for {map_name}: "
                f"{list(self.monsters_info.keys())}"
            )

        if mode in {"normal", "debug", "patrol"} and \
                monster_backend == "yolo":
            try:
                detector_signature = \
                    YoloMonsterDetector.signature_from_config(monster_cfg)
                if (
                    self.yolo_monster_detector is None
                    or self.yolo_monster_detector.config_signature
                    != detector_signature
                ):
                    self.yolo_monster_detector = \
                        YoloMonsterDetector.from_config(monster_cfg)
                warmup_ms = None
                if monster_cfg.get("warmup", True):
                    warmup_ms = self.yolo_monster_detector.warmup(
                        frame_size=WINDOW_WORKING_SIZE
                    )
            except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
                logger.error(f"[load_config] Unable to load YOLO model: {exc}")
                return -1
            warmup_text = (
                "" if warmup_ms is None
                else f", warmup_ms={warmup_ms:.1f}"
            )
            preprocess_size = self.yolo_monster_detector.preprocess_size
            preprocess_text = (
                "native" if preprocess_size is None
                else f"{preprocess_size[1]}x{preprocess_size[0]}"
            )
            logger.info(
                "[load_config] YOLO monster detection ready: "
                f"model={self.yolo_monster_detector.model_path}, "
                f"preprocess={preprocess_text}, "
                f"imgsz={self.yolo_monster_detector.imgsz}, "
                f"confidence={self.yolo_monster_detector.confidence}, "
                f"device={self.yolo_monster_detector.device}, "
                f"class={self.yolo_monster_detector.class_name}"
                f"{warmup_text}"
            )

        if marker_cfg.get("enable", False) and marker_backend == "yolo":
            hero_yolo_cfg = deepcopy(monster_cfg)
            hero_yolo_overrides = marker_cfg.get("yolo", {})
            hero_yolo_cfg.update(hero_yolo_overrides)
            hero_yolo_cfg["class_name"] = str(
                hero_yolo_overrides.get("class_name", "hero")
            )
            try:
                shared_model = (
                    self.yolo_monster_detector is not None
                    and YoloMonsterDetector.inference_signature_from_config(
                        hero_yolo_cfg
                    ) == YoloMonsterDetector.inference_signature_from_config(
                        monster_cfg
                    )
                )
                if shared_model:
                    self.yolo_hero_detector = self.yolo_monster_detector
                else:
                    self.yolo_hero_detector = \
                        YoloMonsterDetector.from_config(hero_yolo_cfg)
                self.yolo_hero_detector.require_class(
                    hero_yolo_cfg["class_name"]
                )
                if hero_yolo_cfg.get("warmup", True):
                    self.yolo_hero_detector.warmup(frame_size=WINDOW_WORKING_SIZE)
            except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
                logger.error(
                    "[load_config] Unable to load YOLO Hero detector: "
                    f"{exc}"
                )
                return -1
            if shared_model:
                self.yolo_inference_class_names = tuple(dict.fromkeys((
                    self.yolo_monster_detector.class_name,
                    hero_yolo_cfg["class_name"],
                )))
            logger.info(
                "[load_config] YOLO Hero detection ready: "
                f"model={self.yolo_hero_detector.model_path}, "
                f"confidence={float(hero_yolo_cfg.get('confidence', 0.4))}, "
                f"class={hero_yolo_cfg['class_name']}, "
                f"shared_with_mob={shared_model}"
            )

        # Load player's name tag
        if cfg["nametag"]["enable"]:
            self.img_nametag = load_image(f"nametag/{cfg['nametag']['name']}.png")
            self._img_nametag_source = self.img_nametag.copy()
            self.img_nametag_gray = cv2.cvtColor(
                self.img_nametag, cv2.COLOR_BGR2GRAY
            )
            medal_cfg = cfg["nametag"].get("medal", {})
            if medal_cfg.get("enable", False):
                medal_name = (
                    medal_cfg.get("name")
                    or f"{cfg['nametag']['name']}_medal"
                )
                medal_path = f"nametag/{medal_name}.png"
                if os.path.exists(medal_path):
                    self.img_nametag_medal = load_image(medal_path)
                    self._img_nametag_medal_source = \
                        self.img_nametag_medal.copy()
                    self.img_nametag_medal_gray = cv2.cvtColor(
                        self.img_nametag_medal, cv2.COLOR_BGR2GRAY
                    )
                else:
                    logger.warning(
                        f"NameTag medal template not found: {medal_path}; "
                        "falling back to ID-only matching"
                    )
            pet_cfg = cfg["nametag"].get("pet", {})
            if pet_cfg.get("enable", False):
                pet_name = (
                    pet_cfg.get("name")
                    or f"{cfg['nametag']['name']}_pet"
                )
                pet_path = f"nametag/{pet_name}.png"
                if os.path.exists(pet_path):
                    self.img_nametag_pet = load_image(pet_path)
                    self._img_nametag_pet_source = self.img_nametag_pet.copy()
                    self.img_nametag_pet_gray = cv2.cvtColor(
                        self.img_nametag_pet, cv2.COLOR_BGR2GRAY
                    )
                else:
                    logger.warning(
                        f"NameTag pet template not found: {pet_path}; "
                        "pet-assisted matching is disabled"
                    )

        # Marker/YOLO Hero detection still needs nearby climbing/standing
        # templates to decide whether a vertical route must remain held. Keep
        # pose loading independent from the disabled legacy name-tag locator.
        self._load_player_appearance_templates(cfg["nametag"])

        marker_cfg = cfg["nametag"].get("overhead_marker", {})
        if marker_cfg.get("enable", False) and marker_backend == "template":
            marker_name = (
                marker_cfg.get("name")
                or f"{cfg['nametag']['name']}_overhead_smile"
            )
            marker_path = f"nametag/{marker_name}.png"
            if not os.path.exists(marker_path):
                logger.error(
                    "Overhead Hero marker template not found: "
                    f"{marker_path}; marker-only Hero detection cannot start"
                )
                return -1
            try:
                self.img_overhead_marker = load_image(marker_path)
                self._img_overhead_marker_source = \
                    self.img_overhead_marker.copy()
                self._update_overhead_marker_template_metadata(marker_cfg)
            except (cv2.error, OSError, RuntimeError, ValueError) as exc:
                logger.error(
                    "Unable to load overhead Hero marker template "
                    f"{marker_path}: {exc}"
                )
                return -1
            if self.overhead_marker_component_bbox is None or \
                    not np.any(self.img_overhead_marker_mask):
                logger.error(
                    "Overhead Hero marker template has no usable white "
                    f"component: {marker_path}"
                )
                return -1
            logger.info(
                "Loaded overhead Hero marker template: "
                f"{marker_path}"
            )

        # Load misc image
        lang = cfg["system"]["language"]
        self.img_create_party_enable  = load_image(f"misc/party_button_create_enable_{lang}.png")
        self.img_create_party_disable = load_image(f"misc/party_button_create_disable_{lang}.png")
        self.img_login_button = load_image(f"misc/login_button_{lang}.png")
        self._img_login_button_source = self.img_login_button.copy()
        self._last_login_template_geometry = None
        self._auto_relogin_template_sources = {}
        self._auto_relogin_templates = {}
        self._auto_relogin_cursor_template_source = None
        self._auto_relogin_cursor_tracker = None
        self._auto_relogin_disconnect_cursor_template_source = None
        self._auto_relogin_disconnect_cursor_tracker = None
        self._last_auto_relogin_template_geometry = None
        if cfg.get("auto_relogin", {}).get("enable", False):
            if str(cfg["auto_relogin"].get(
                    "remote_mouse_mode", "absolute"
                    )).strip().lower() == "visual_relative":
                cursor_path = cfg["auto_relogin"].get("cursor_template", "")
                cursor_template = cv2.imread(
                    str(cursor_path), cv2.IMREAD_UNCHANGED
                )
                if cursor_template is None:
                    logger.error(
                        "[load_config] Unable to load auto-relogin cursor "
                        f"template: {cursor_path}"
                    )
                    return -1
                if cursor_template.ndim != 3 or \
                        cursor_template.shape[2] != 4:
                    logger.error(
                        "[load_config] Auto-relogin cursor template must be "
                        f"RGBA/BGRA with transparency: {cursor_path}"
                    )
                    return -1
                try:
                    CursorTracker(
                        cursor_template,
                        hotspot=tuple(cfg["auto_relogin"].get(
                            "cursor_hotspot", (13, 6)
                        )),
                        min_score=float(cfg["auto_relogin"].get(
                            "cursor_min_score", 0.90
                        )),
                        uniqueness_margin=float(cfg["auto_relogin"].get(
                            "cursor_uniqueness_margin", 0.02
                        )),
                        mask_erode_pixels=int(cfg["auto_relogin"].get(
                            "cursor_mask_erode_pixels", 1
                        )),
                    )
                except (TypeError, ValueError, IndexError) as exc:
                    logger.error(
                        "[load_config] Invalid auto-relogin cursor template "
                        f"or hotspot: {exc}"
                    )
                    return -1
                self._auto_relogin_cursor_template_source = \
                    cursor_template.copy()

                disconnect_cursor_path = cfg["auto_relogin"][
                    "disconnect_cursor_template"
                ]
                disconnect_cursor_template = cv2.imread(
                    str(disconnect_cursor_path), cv2.IMREAD_UNCHANGED
                )
                if disconnect_cursor_template is None:
                    logger.error(
                        "[load_config] Unable to load auto-relogin "
                        "disconnect cursor template: "
                        f"{disconnect_cursor_path}"
                    )
                    return -1
                if disconnect_cursor_template.ndim != 3 or \
                        disconnect_cursor_template.shape[2] != 4:
                    logger.error(
                        "[load_config] Auto-relogin disconnect cursor "
                        "template must be RGBA/BGRA with transparency: "
                        f"{disconnect_cursor_path}"
                    )
                    return -1
                try:
                    CursorTracker(
                        disconnect_cursor_template,
                        hotspot=tuple(cfg["auto_relogin"].get(
                            "disconnect_cursor_hotspot", (9, 4)
                        )),
                        min_score=float(cfg["auto_relogin"].get(
                                "cursor_min_score", 0.90
                        )),
                        uniqueness_margin=float(
                            cfg["auto_relogin"].get(
                                "cursor_uniqueness_margin", 0.02
                            )
                        ),
                        mask_erode_pixels=int(cfg["auto_relogin"].get(
                            "cursor_mask_erode_pixels", 1
                        )),
                    )
                except (TypeError, ValueError, IndexError) as exc:
                    logger.error(
                        "[load_config] Invalid auto-relogin disconnect "
                        f"cursor template or hotspot: {exc}"
                    )
                    return -1
                self._auto_relogin_disconnect_cursor_template_source = \
                    disconnect_cursor_template.copy()

        # Normalized pixel coordinate configuration
        cfg['rune_warning_cn']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_warning_cn']['top_left'], cfg['game_window']['size'])
        cfg['rune_warning_cn']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_warning_cn']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_warning_eng']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_warning_eng']['top_left'], cfg['game_window']['size'])
        cfg['rune_warning_eng']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_warning_eng']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_enable_msg_cn']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_cn']['top_left'], cfg['game_window']['size'])
        cfg['rune_enable_msg_cn']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_cn']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_enable_msg_eng']['top_left'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_eng']['top_left'], cfg['game_window']['size'])
        cfg['rune_enable_msg_eng']['bottom_right'] = normalize_pixel_coordinate(
            cfg['rune_enable_msg_eng']['bottom_right'], cfg['game_window']['size'])
        cfg['rune_solver']['arrow_box_coord'] = normalize_pixel_coordinate(
            cfg['rune_solver']['arrow_box_coord'], cfg['game_window']['size'])
        cfg['ui_coords']['login_button_top_left'] = normalize_pixel_coordinate(
            cfg['ui_coords']['login_button_top_left'], cfg['game_window']['size'])
        cfg['ui_coords']['login_button_bottom_right'] = normalize_pixel_coordinate(
            cfg['ui_coords']['login_button_bottom_right'], cfg['game_window']['size'])

        # Print mode on log
        logger.info(f"[load_config] Config AutoBot as {cfg['bot']['mode']} mode")

        # Preserve one normalized-but-unscaled reference config. Runtime
        # screen-space values are always regenerated from this copy when the
        # capture output size changes.
        self._base_cfg = deepcopy(cfg)
        self.cfg = deepcopy(cfg)
        self._last_runtime_output_size = None
        self._last_nametag_template_geometry = None
        self._last_ui_viz_emit_time = 0.0
        if wz_navigation_enabled and self.wz_navigation is not None and \
                self.wz_navigation.active:
            if not self._install_wz_navigation_resources():
                logger.error(
                    "[load_config] WZ navigation matched a map but failed to "
                    "install generated routes"
                )
                return -1

        return 0 # load successfully

    def _load_player_appearance_templates(self, nametag_cfg):
        """Load pose templates for either marker-only or name-tag detection."""
        appearance_cfg = nametag_cfg.get("appearance", {})
        appearance_mode = appearance_cfg.get("enable", False)
        appearance_enabled = appearance_mode is True or (
            appearance_mode == "auto"
            and (
                nametag_cfg.get("enable", False)
                or nametag_cfg.get("overhead_marker", {}).get(
                    "enable", False
                )
            )
        )
        if not appearance_enabled:
            return

        for template_index, template_cfg in enumerate(
                appearance_cfg.get("templates", [])):
            template_name = template_cfg.get("name", "").strip()
            if not template_name:
                suffix = template_cfg.get("suffix", "").strip()
                if suffix:
                    template_name = f"{nametag_cfg['name']}_{suffix}"
            if not template_name:
                continue
            appearance_path = f"nametag/{template_name}.png"
            if not os.path.exists(appearance_path):
                logger.warning(
                    "NameTag appearance template not found: "
                    f"{appearance_path}"
                )
                continue
            image = load_image(appearance_path)
            offset = template_cfg.get("player_offset", (0, 0))
            self.nametag_appearance_templates.append({
                "name": template_name,
                "pose": template_cfg.get("pose", "standing"),
                "image": image,
                "source_image": image.copy(),
                "config_index": template_index,
                "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                "mask": get_mask(image, (0, 255, 0)),
                "player_offset": (int(offset[0]), int(offset[1])),
            })

        if self.nametag_appearance_templates:
            logger.info(
                "Loaded player appearance templates: "
                f"{[item['name'] for item in self.nametag_appearance_templates]}"
            )
        else:
            logger.warning(
                "Player appearance detection is enabled but no templates "
                "could be loaded"
            )

    @staticmethod
    def _resize_green_screen_template(image, output_size):
        """Resize one keyed template while retaining an exact green mask."""
        output_h, output_w = map(int, output_size)
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Invalid template output size: {output_size}")
        if image.shape[:2] == (output_h, output_w):
            return image.copy()

        source_mask = get_mask(image, (0, 255, 0))
        downscaling = output_h < image.shape[0] or output_w < image.shape[1]
        interpolation = cv2.INTER_AREA if downscaling else cv2.INTER_CUBIC
        resized = cv2.resize(
            image,
            (output_w, output_h),
            interpolation=interpolation,
        )
        resized_mask = cv2.resize(
            source_mask,
            (output_w, output_h),
            interpolation=cv2.INTER_NEAREST,
        )
        resized[resized_mask == 0] = (0, 255, 0)
        return resized

    def _update_overhead_marker_template_metadata(self, marker_cfg=None):
        """Cache the keyed mask and white-component geometry for the marker."""
        image = getattr(self, "img_overhead_marker", None)
        if image is None:
            self.img_overhead_marker_gray = None
            self.img_overhead_marker_mask = None
            self.overhead_marker_component_bbox = None
            return

        if marker_cfg is None:
            marker_cfg = (
                (getattr(self, "cfg", None) or {})
                .get("nametag", {})
                .get("overhead_marker", {})
            )
        white_lower = marker_cfg.get("white_lower", (185, 185, 185))
        if not isinstance(white_lower, (list, tuple)) or len(white_lower) != 3:
            raise ValueError(
                "nametag.overhead_marker.white_lower must be [B, G, R]"
            )

        self.img_overhead_marker_gray = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )
        self.img_overhead_marker_mask = get_mask(image, (0, 255, 0))
        white_mask = cv2.inRange(
            image,
            np.asarray(white_lower, dtype=np.uint8),
            np.asarray((255, 255, 255), dtype=np.uint8),
        )
        white_mask[self.img_overhead_marker_mask == 0] = 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            white_mask, connectivity=8
        )
        if count <= 1:
            self.overhead_marker_component_bbox = None
            logger.warning(
                "Overhead Hero marker has no white connected component"
            )
            return

        component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        self.overhead_marker_component_bbox = tuple(map(int, (
            stats[component_index, cv2.CC_STAT_LEFT],
            stats[component_index, cv2.CC_STAT_TOP],
            stats[component_index, cv2.CC_STAT_WIDTH],
            stats[component_index, cv2.CC_STAT_HEIGHT],
        )))

    def _refresh_nametag_templates(self, output_size):
        """Scale source hero templates from their recorded frame geometry."""
        nametag_cfg = self.cfg.get("nametag", {})
        marker_cfg = nametag_cfg.get("overhead_marker", {})
        reference_size = nametag_cfg.get("template_reference_size")
        if not (
            nametag_cfg.get("enable", False)
            or marker_cfg.get("enable", False)
        ) or reference_size is None:
            return
        if not isinstance(reference_size, (list, tuple)) or \
                len(reference_size) != 2:
            raise ValueError(
                "nametag.template_reference_size must be [height, width]"
            )

        reference_h, reference_w = map(int, reference_size)
        output_h, output_w = map(int, output_size)
        if min(reference_h, reference_w, output_h, output_w) <= 0:
            raise ValueError(
                "nametag.template_reference_size and output size must be positive"
            )
        geometry_key = (
            reference_h, reference_w, output_h, output_w,
        )
        if geometry_key == getattr(
                self, "_last_nametag_template_geometry", None):
            return

        scale_x = output_w / reference_w
        scale_y = output_h / reference_h

        def resize_source(source_name, target_name, gray_name):
            source = getattr(self, source_name, None)
            current = getattr(self, target_name, None)
            if source is None and current is not None:
                source = current.copy()
                setattr(self, source_name, source)
            if source is None:
                setattr(self, target_name, None)
                setattr(self, gray_name, None)
                return
            target_size = (
                max(1, int(round(source.shape[0] * scale_y))),
                max(1, int(round(source.shape[1] * scale_x))),
            )
            resized = self._resize_green_screen_template(source, target_size)
            setattr(self, target_name, resized)
            setattr(
                self, gray_name, cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            )

        resize_source(
            "_img_nametag_source", "img_nametag", "img_nametag_gray"
        )
        resize_source(
            "_img_nametag_medal_source",
            "img_nametag_medal",
            "img_nametag_medal_gray",
        )
        resize_source(
            "_img_nametag_pet_source",
            "img_nametag_pet",
            "img_nametag_pet_gray",
        )
        resize_source(
            "_img_overhead_marker_source",
            "img_overhead_marker",
            "img_overhead_marker_gray",
        )
        self._update_overhead_marker_template_metadata(marker_cfg)

        runtime_templates = nametag_cfg.get("appearance", {}).get(
            "templates", []
        )
        for index, template in enumerate(getattr(
                self, "nametag_appearance_templates", [])):
            source = template.get("source_image")
            if source is None:
                source = template.get("image")
                if source is None:
                    continue
                source = source.copy()
                template["source_image"] = source
            target_size = (
                max(1, int(round(source.shape[0] * scale_y))),
                max(1, int(round(source.shape[1] * scale_x))),
            )
            image = self._resize_green_screen_template(source, target_size)
            template["image"] = image
            template["gray"] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template["mask"] = get_mask(image, (0, 255, 0))

            config_index = int(template.get("config_index", index))
            if 0 <= config_index < len(runtime_templates):
                offset = runtime_templates[config_index].get(
                    "player_offset", template.get("player_offset", (0, 0))
                )
                template["player_offset"] = (
                    int(offset[0]), int(offset[1])
                )

        self._last_nametag_template_geometry = geometry_key
        logger.info(
            "[capture] Scaled hero templates from "
            f"{(reference_h, reference_w)} to {(output_h, output_w)}"
        )

    def _refresh_login_button_template(self, output_size):
        """Scale the login template with the runtime game-frame geometry."""
        source = getattr(self, "_img_login_button_source", None)
        if source is None:
            current = getattr(self, "img_login_button", None)
            if current is None:
                return
            source = current.copy()
            self._img_login_button_source = source

        base_cfg = getattr(self, "_base_cfg", None) or self.cfg
        reference = base_cfg.get("auto_relogin", {}).get(
            "template_reference_size", (700, 1296)
        )
        try:
            reference_h, reference_w = map(int, reference[:2])
            output_h, output_w = map(int, output_size[:2])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                "auto_relogin.template_reference_size must be [height, width]"
            ) from exc
        if min(reference_h, reference_w, output_h, output_w) <= 0:
            raise ValueError(
                "Login template reference and output sizes must be positive"
            )

        geometry_key = (
            reference_h, reference_w, output_h, output_w,
        )
        if geometry_key == getattr(
                self, "_last_login_template_geometry", None):
            return

        target_width = max(
            1, int(round(source.shape[1] * output_w / reference_w))
        )
        target_height = max(
            1, int(round(source.shape[0] * output_h / reference_h))
        )
        if (target_height, target_width) == source.shape[:2]:
            self.img_login_button = source.copy()
        else:
            downscaling = target_height < source.shape[0] or \
                target_width < source.shape[1]
            self.img_login_button = cv2.resize(
                source,
                (target_width, target_height),
                interpolation=(
                    cv2.INTER_AREA if downscaling else cv2.INTER_CUBIC
                ),
            )
        self._last_login_template_geometry = geometry_key
        logger.info(
            "[capture] Scaled login template from "
            f"{source.shape[:2]} to {self.img_login_button.shape[:2]}"
        )

    def _refresh_auto_relogin_templates(self, output_size):
        """Scale the recorded page and cursor templates to the current frame."""
        sources = getattr(self, "_auto_relogin_template_sources", {})
        cursor_source = getattr(
            self, "_auto_relogin_cursor_template_source", None
        )
        disconnect_cursor_source = getattr(
            self,
            "_auto_relogin_disconnect_cursor_template_source",
            None,
        )
        if not sources and cursor_source is None and \
                disconnect_cursor_source is None:
            return

        base_cfg = getattr(self, "_base_cfg", None) or self.cfg
        auto_relogin_cfg = base_cfg.get("auto_relogin", {})
        reference = auto_relogin_cfg.get(
            "flow_template_reference_size", (2160, 3840)
        )
        try:
            reference_h, reference_w = map(int, reference[:2])
            output_h, output_w = map(int, output_size[:2])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(
                "auto_relogin.flow_template_reference_size must be "
                "[height, width]"
            ) from exc
        if min(reference_h, reference_w, output_h, output_w) <= 0:
            raise ValueError(
                "Auto-relogin template reference and output sizes must be "
                "positive"
            )

        geometry_key = (
            reference_h, reference_w, output_h, output_w,
        )
        if geometry_key == getattr(
                self, "_last_auto_relogin_template_geometry", None):
            return
        self._reset_auto_relogin_ocr_gate()

        scaled_templates = {}
        for page, source in sources.items():
            target_width = max(
                1, int(round(source.shape[1] * output_w / reference_w))
            )
            target_height = max(
                1, int(round(source.shape[0] * output_h / reference_h))
            )
            if (target_height, target_width) == source.shape[:2]:
                scaled = source.copy()
            else:
                downscaling = target_height < source.shape[0] or \
                    target_width < source.shape[1]
                scaled = cv2.resize(
                    source,
                    (target_width, target_height),
                    interpolation=(
                        cv2.INTER_AREA if downscaling else cv2.INTER_CUBIC
                    ),
                )
            scaled_templates[page] = scaled

        self._auto_relogin_templates = scaled_templates

        def build_cursor_tracker(source, hotspot):
            """Scale one RGBA cursor source and its reference hotspot."""
            target_width = max(
                1,
                int(round(source.shape[1] * output_w / reference_w)),
            )
            target_height = max(
                1,
                int(round(source.shape[0] * output_h / reference_h)),
            )
            if (target_height, target_width) == source.shape[:2]:
                scaled_cursor = source.copy()
            elif source.ndim == 3 and source.shape[2] == 4:
                downscaling = target_height < source.shape[0] or \
                    target_width < source.shape[1]
                scaled_color = cv2.resize(
                    source[:, :, :3],
                    (target_width, target_height),
                    interpolation=(
                        cv2.INTER_AREA if downscaling else cv2.INTER_CUBIC
                    ),
                )
                scaled_alpha = cv2.resize(
                    source[:, :, 3],
                    (target_width, target_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                scaled_cursor = np.dstack((scaled_color, scaled_alpha))
            else:
                scaled_cursor = cv2.resize(
                    source,
                    (target_width, target_height),
                    interpolation=(
                        cv2.INTER_AREA
                        if target_height < source.shape[0]
                        or target_width < source.shape[1]
                        else cv2.INTER_CUBIC
                    ),
                )

            hotspot_x = max(
                0,
                min(
                    target_width - 1,
                    int(round(float(hotspot[0]) * output_w / reference_w)),
                ),
            )
            hotspot_y = max(
                0,
                min(
                    target_height - 1,
                    int(round(float(hotspot[1]) * output_h / reference_h)),
                ),
            )
            return CursorTracker(
                scaled_cursor,
                hotspot=(hotspot_x, hotspot_y),
                min_score=float(auto_relogin_cfg.get(
                    "cursor_min_score", 0.90
                )),
                uniqueness_margin=float(auto_relogin_cfg.get(
                    "cursor_uniqueness_margin", 0.02
                )),
                min_visible_fraction=float(auto_relogin_cfg.get(
                    "cursor_min_visible_fraction", 0.25
                )),
                min_visible_pixels=max(
                    1,
                    int(round(
                        auto_relogin_cfg.get(
                            "cursor_min_visible_pixels", 32
                        )
                        * output_w / reference_w
                        * output_h / reference_h
                    )),
                ),
                mask_erode_pixels=max(
                    0,
                    int(round(auto_relogin_cfg.get(
                        "cursor_mask_erode_pixels", 1
                    ))),
                ),
            )

        self._auto_relogin_cursor_tracker = None
        self._auto_relogin_disconnect_cursor_tracker = None
        if cursor_source is not None:
            self._auto_relogin_cursor_tracker = build_cursor_tracker(
                cursor_source,
                auto_relogin_cfg.get("cursor_hotspot", (13, 6)),
            )
        if disconnect_cursor_source is not None:
            self._auto_relogin_disconnect_cursor_tracker = \
                build_cursor_tracker(
                    disconnect_cursor_source,
                    auto_relogin_cfg.get(
                        "disconnect_cursor_hotspot", (9, 4)
                    ),
                )

        self._last_auto_relogin_template_geometry = geometry_key
        logger.info(
            "[capture] Scaled auto-relogin page/cursor templates from "
            f"{(reference_h, reference_w)} to {(output_h, output_w)}"
        )

    def _refresh_runtime_frame_config(self, output_size):
        """Regenerate native screen-space config from one unscaled baseline."""
        output_size = tuple(map(int, output_size[:2]))
        if output_size == getattr(self, "_last_runtime_output_size", None):
            return

        base_cfg = getattr(self, "_base_cfg", None)
        if base_cfg is None:
            # Preserve compatibility with lightweight ``__new__`` test bots.
            base_cfg = deepcopy(self.cfg)
            self._base_cfg = base_cfg
        self.cfg = scale_runtime_pixel_config(base_cfg, output_size)
        self._last_runtime_output_size = output_size

        # These components are constructed before the first frame arrives.
        # Swap their shared config reference atomically after deriving native
        # screen coordinates.
        for component_name in (
                "capture", "kb", "health_monitor", "rune_solver"):
            component = getattr(self, component_name, None)
            if component is not None and hasattr(component, "cfg"):
                component.cfg = self.cfg

        self._refresh_nametag_templates(output_size)
        self._refresh_login_button_template(output_size)
        self._refresh_auto_relogin_templates(output_size)
        logger.info(
            "[capture] Runtime pixel config ready for "
            f"output_size={output_size}"
        )

    def start(self):
        '''
        Start all threads
        '''
        if self.thread_auto_bot is not None and self.thread_auto_bot.is_alive():
            raise RuntimeError("MapleStoryAutoBot is already running")

        # Allow the same UI controller to start the bot again after a pause.
        self.is_terminated = False
        if not hasattr(self, "_shutdown_lock"):
            self._shutdown_lock = threading.Lock()
        with self._shutdown_lock:
            self._components_stopped = False
        self._input_suspended_for_capture = False
        self.kb = None
        self.capture = None
        self.health_monitor = None
        self.profiler = None
        self.rune_solver = None

        try:
            # Validate and start the capture source on computer A before opening
            # the ESP32's USB serial input session.
            self.capture = create_capture_source(
                self.cfg,
                test_image_name=self.args.test_image or None,
                window_capture_cls=GameWindowCapturor,
            )

            control_enabled = (
                not self.is_disable_control
                and not self.is_debug_mode()
            )
            self.kb = KeyBoardController(
                self.cfg,
                connect_input=control_enabled,
                capture_available=(
                    control_enabled and not self.remote_keyboard_target()
                ),
                scheduled_buff_allowed=self._scheduled_buff_allowed,
            )
            self._input_suspended_for_capture = (
                control_enabled and self.remote_keyboard_target()
            )

            self.health_monitor = HealthMonitor(self.cfg, self.kb)
            if self.cfg["health_monitor"]["enable"] and control_enabled:
                self.health_monitor.start()

            self.profiler = Profiler(self.cfg)
            if self.cfg.get("rune_solver", {}).get("enable", False) is True:
                self.rune_solver = RuneSolver(self.cfg)

            # Reset all timers
            self.t_last_frame = time.time()
            self.t_watch_dog = time.time()
            self.t_last_teleport = time.time()
            self.t_last_attack = time.time()
            self.t_last_directional_aoe = time.time()
            self.t_last_power_knockback = time.time()
            self.t_last_minimap_update = time.time()
            self.t_to_change_channel = time.time()
            self._reset_stationary_jump_proximity()
            self._reset_rope_climb(clear_locks=True)
            self._reset_portal_sweep()
            self._reset_auto_relogin_runtime()

            # Set init state
            if self.args.init_state != "":
                self.fsm.set_init_state(self.args.init_state) # For debugging
            elif self.cfg["bot"]["mode"] == "aux":
                self.fsm.set_init_state("aux")
            elif self.cfg["bot"]["mode"] == "patrol":
                self.fsm.set_init_state("patrol")
            elif self.cfg["bot"]["mode"] == "debug":
                self.fsm.set_init_state("debug")
            else:
                self.fsm.set_init_state("hunting")

            self.is_first_frame = True
            self.has_valid_nametag_location = False
            self.nametag_miss_count = 0
            self.pending_nametag_location = None
            self.has_valid_overhead_marker_location = False
            self.overhead_marker_miss_count = 0
            self.pending_overhead_marker_location = None
            self.pending_overhead_marker_count = 0
            self.t_last_overhead_marker_detected = None
            self.last_overhead_marker_match = None
            self.last_single_yolo_hero_location = None
            self.last_single_yolo_hero_frame_token = None
            self.last_yolo_hero_detection = None
            self.has_valid_appearance_location = False
            self.pending_appearance_location = None
            self.pending_appearance_count = 0
            self.last_appearance_match = None
            self.screen_player_location_valid = False
            self.thread_auto_bot = threading.Thread(target=self.loop)
            self.thread_auto_bot.start()
        except BaseException:
            # A failed start must not leave the capture thread or ESP32's
            # serial connection alive and break the next retry.
            self.is_terminated = True
            self._stop_components()
            self.thread_auto_bot = None
            raise

        logger.info("[MapleStoryAutoBot] Started")

    def pause(self):
        '''
        Terminate thread except main thread
        '''
        navigator = getattr(self, "wz_navigation", None)
        suspend_navigation = getattr(
            navigator, "suspend_platform_navigation", None
        )
        if callable(suspend_navigation):
            suspend_navigation()
        self._reset_ladder_route_hold()
        self.terminate_threads()

    def enable_viz(self):
        '''
        Enable AutoBot to generate debug image
        '''
        self.is_need_show_debug_window = True
        logger.debug("[enable_viz] is_show_debug_window = True")

    def disable_viz(self):
        '''
        Disable AutoBot to generate debug image
        '''
        self.is_need_show_debug_window = False
        logger.debug("[disable_viz] is_show_debug_window = False")

    def start_record(self):
        '''
        Start recording raw capture frames and annotated diagnostic frames.
        '''
        # Make sure video/ exist
        os.makedirs("video", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        raw_path = os.path.join("video", f"{timestamp}_raw.mp4")
        annotated_path = os.path.join(
            "video", f"{timestamp}_annotated.mp4"
        )

        with self._get_video_record_lock():
            self._release_video_record_writers()
            self._video_record_path = raw_path
            self._video_record_size = None
            self._annotated_video_record_path = annotated_path
            self._annotated_video_record_size = None
            self._annotated_record_frame_pending = False

            # The raw writer can use the capture source's current dimensions
            # immediately.  The annotated writer is opened lazily after the
            # next frame has completed all diagnostic drawing.
            frame = getattr(self, "frame", None)
            if frame is not None:
                self._open_video_writer_for_frame(frame)

        logger.info(
            "[start_record] Record raw capture frames to "
            f"{raw_path}; annotated frames to {annotated_path}"
        )

    def _get_video_record_lock(self):
        """Return the recorder lock, including for lightweight test objects."""
        lock = getattr(self, "_video_record_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._video_record_lock = lock
        return lock

    def _release_video_record_writers(self):
        """Release both writers. Caller must hold the recorder lock."""
        for writer_attr in ("video_writer", "annotated_video_writer"):
            writer = getattr(self, writer_attr, None)
            if writer is not None:
                writer.release()
            setattr(self, writer_attr, None)

    def _open_record_writer_for_frame(
            self, frame, *, path_attr, writer_attr, size_attr):
        """Open or resize one pending MP4 writer for a concrete frame."""
        path = getattr(self, path_attr, None)
        if path is None or frame is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        frame_size = (frame_w, frame_h)
        writer = getattr(self, writer_attr, None)
        if writer is not None and getattr(self, size_attr, None) == frame_size:
            return writer
        if writer is not None:
            writer.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, 10, frame_size)
        setattr(self, writer_attr, writer)
        setattr(self, size_attr, frame_size)
        return writer

    def _open_video_writer_for_frame(self, frame):
        """Open the raw recorder with the actual capture-frame size."""
        with self._get_video_record_lock():
            return self._open_record_writer_for_frame(
                frame,
                path_attr="_video_record_path",
                writer_attr="video_writer",
                size_attr="_video_record_size",
            )

    def _open_annotated_video_writer_for_frame(self, frame):
        """Open the annotated recorder with the rendered-frame size."""
        with self._get_video_record_lock():
            return self._open_record_writer_for_frame(
                frame,
                path_attr="_annotated_video_record_path",
                writer_attr="annotated_video_writer",
                size_attr="_annotated_video_record_size",
            )

    def _write_raw_video_frame(self, frame):
        """Write one capture-source frame before pipeline processing."""
        with self._get_video_record_lock():
            writer = self._open_record_writer_for_frame(
                frame,
                path_attr="_video_record_path",
                writer_attr="video_writer",
                size_attr="_video_record_size",
            )
            if writer is None:
                return False
            writer.write(frame)
            return True

    def _write_annotated_video_frame(self, frame):
        """Write one frame after all available diagnostic drawing."""
        with self._get_video_record_lock():
            writer = self._open_record_writer_for_frame(
                frame,
                path_attr="_annotated_video_record_path",
                writer_attr="annotated_video_writer",
                size_attr="_annotated_video_record_size",
            )
            if writer is None:
                return False
            writer.write(frame)
            return True

    def _flush_annotated_video_frame(self):
        """Write the current loop frame once, including early-return frames."""
        if not getattr(self, "_annotated_record_frame_pending", False):
            return False
        self._annotated_record_frame_pending = False
        return self._write_annotated_video_frame(
            getattr(self, "img_frame_debug", None)
        )

    def stop_record(self):
        '''
        Stop both video recordings.
        '''
        with self._get_video_record_lock():
            was_recording = any((
                getattr(self, "_video_record_path", None),
                getattr(self, "_annotated_video_record_path", None),
                getattr(self, "video_writer", None),
                getattr(self, "annotated_video_writer", None),
            ))
            self._release_video_record_writers()
            self._video_record_path = None
            self._video_record_size = None
            self._annotated_video_record_path = None
            self._annotated_video_record_size = None
            self._annotated_record_frame_pending = False
        if was_recording:
            logger.info("[stop_record] Stop raw and annotated recording")

    def get_player_location_by_yolo(self, expected_player=None):
        """Locate the configured Hero class with guarded temporal tracking."""
        marker_cfg = (
            self.cfg.get("nametag", {}).get("overhead_marker", {})
        )
        if not marker_cfg.get("enable", False) or str(
                marker_cfg.get("backend", "template")).lower() != "yolo":
            return None
        detector = getattr(self, "yolo_hero_detector", None)
        frame = getattr(self, "img_frame", None)
        if detector is None or frame is None:
            return None

        def record_miss(status, *, reset_pending=True, match=None):
            self.overhead_marker_miss_count = getattr(
                self, "overhead_marker_miss_count", 0
            ) + 1
            if reset_pending:
                self.pending_overhead_marker_location = None
                self.pending_overhead_marker_count = 0
            max_stale_frames = int(marker_cfg.get("max_stale_frames", -1))
            retain_without_limit = max_stale_frames < 0
            use_cached_location = (
                status != "minimap-unavailable"
                and getattr(
                    self, "has_valid_overhead_marker_location", False
                )
                and (
                    retain_without_limit
                    or self.overhead_marker_miss_count <= max_stale_frames
                )
            )
            self.last_overhead_marker_match = {
                "status": f"{status},cached" if use_cached_location else status,
                **({} if match is None else match),
            }
            cache_expired = (
                not retain_without_limit
                and self.overhead_marker_miss_count > max_stale_frames
            )
            if status == "minimap-unavailable" or cache_expired:
                self.last_single_yolo_hero_location = None
                self.last_single_yolo_hero_frame_token = None
                self.last_yolo_hero_detection = None
            if use_cached_location:
                cached_detection = getattr(
                    self, "last_yolo_hero_detection", None
                )
                if cached_detection is not None:
                    self._draw_debug_rectangle(
                        cached_detection["position"],
                        cached_detection["size"],
                        (0, 165, 255),
                        f"HeroYOLO:cached {self.overhead_marker_miss_count}",
                        thickness=1,
                        text_height=0.45,
                    )
                return self.loc_overhead_marker_player
            self.has_valid_overhead_marker_location = False
            return None

        def confirm_global_candidate(candidate):
            confirm_frames = max(
                1, int(marker_cfg.get("global_confirm_frames", 2))
            )
            if confirm_frames <= 1:
                return True

            pending = getattr(
                self, "pending_overhead_marker_location", None
            )
            confirm_radius = max(
                1, int(marker_cfg.get("global_confirm_radius", 24))
            )
            if pending is not None and max(
                abs(candidate["player"][0] - pending[0]),
                abs(candidate["player"][1] - pending[1]),
            ) <= confirm_radius:
                self.pending_overhead_marker_count = getattr(
                    self, "pending_overhead_marker_count", 0
                ) + 1
            else:
                self.pending_overhead_marker_count = 1
            self.pending_overhead_marker_location = candidate["player"]
            return self.pending_overhead_marker_count >= confirm_frames

        if marker_cfg.get("require_minimap_player", True):
            minimap_source = getattr(self, "img_minimap_source", None)
            minimap_cfg = self.cfg.get("minimap", {})
            if minimap_source is None or get_player_location_on_minimap(
                minimap_source,
                minimap_player_color=minimap_cfg.get(
                    "player_color", (255, 255, 0)
                ),
                color_tolerance=minimap_cfg.get("player_color_tolerance", 0),
                min_component_area=minimap_cfg.get(
                    "player_min_component_area", 4
                ),
            ) is None:
                return record_miss("minimap-unavailable")

        camera_y_end = min(
            frame.shape[0], int(self.cfg["ui_coords"]["ui_y_start"])
        )
        if camera_y_end <= 0:
            return record_miss("invalid-camera")
        yolo_cfg = marker_cfg.get("yolo", {})
        class_name = str(yolo_cfg.get("class_name", "hero"))
        inference_classes = (
            getattr(self, "yolo_inference_class_names", (class_name,))
            if detector is getattr(self, "yolo_monster_detector", None)
            else (class_name,)
        )
        detections = detector.detect(
            frame,
            roi=(0, 0, frame.shape[1], camera_y_end),
            confidence=float(yolo_cfg.get("confidence", 0.4)),
            class_name=class_name,
            inference_class_names=inference_classes,
            inference_confidence=(
                detector.confidence
                if detector is getattr(self, "yolo_monster_detector", None)
                else None
            ),
            cache_key=getattr(self, "_current_capture_frame_token", None),
        )
        anchor = yolo_cfg.get("player_anchor", (0.5, 0.5))
        offset = yolo_cfg.get("player_offset", (0, 0))
        candidates = []
        for detection in detections:
            x, y = detection["position"]
            height, width = detection["size"]
            player = (
                int(round(x + width * float(anchor[0]) + float(offset[0]))),
                int(round(y + height * float(anchor[1]) + float(offset[1]))),
            )
            candidates.append({**detection, "player": player})

        if not candidates:
            return record_miss("not-found")
        previous_single_location = getattr(
            self, "last_single_yolo_hero_location", None
        )
        if len(candidates) == 1:
            self.last_single_yolo_hero_location = candidates[0]["player"]
            self.last_single_yolo_hero_frame_token = getattr(
                self, "_current_capture_frame_token", None
            )
        if expected_player is None and getattr(
                self, "has_valid_overhead_marker_location", False):
            expected_player = self.loc_overhead_marker_player

        # A frame containing exactly one Hero box is stronger identity history
        # than a score comparison between several boxes. Keep that location
        # separate from the accepted/cached coordinate so the immediately
        # following crowded frame can select its nearest candidate.
        selection_anchor = expected_player
        selection_scope = "local"
        if len(candidates) > 1 and previous_single_location is not None:
            selection_anchor = previous_single_location
            selection_scope = "single-history"

        match_scope = "global"
        if selection_anchor is not None:
            selection_anchor = tuple(map(int, selection_anchor))
            local_radius = max(
                1, int(marker_cfg.get("local_search_radius", 90))
            )
            local_candidates = [
                item for item in candidates
                if max(
                    abs(item["player"][0] - selection_anchor[0]),
                    abs(item["player"][1] - selection_anchor[1]),
                ) <= local_radius
            ]
            if not local_candidates:
                if len(candidates) != 1:
                    return record_miss("not-found-local")
                best_match = candidates[0]
                if not confirm_global_candidate(best_match):
                    return record_miss(
                        "reacquire-pending",
                        reset_pending=False,
                        match=best_match,
                    )
                match_scope = "reacquired"
            else:
                best_match = min(local_candidates, key=lambda item: (
                    max(
                        abs(item["player"][0] - selection_anchor[0]),
                        abs(item["player"][1] - selection_anchor[1]),
                    ),
                    -item["confidence"],
                ))
                match_scope = selection_scope
        else:
            if marker_cfg.get("require_unique_global", True) and \
                    len(candidates) != 1:
                return record_miss("ambiguous")
            best_match = max(candidates, key=lambda item: item["confidence"])
            if not confirm_global_candidate(best_match):
                self.last_overhead_marker_match = {
                    "status": "pending",
                    **best_match,
                }
                return None

        self.loc_overhead_marker_player = best_match["player"]
        self.has_valid_overhead_marker_location = True
        self.overhead_marker_miss_count = 0
        self.t_last_overhead_marker_detected = time.monotonic()
        self.pending_overhead_marker_location = None
        self.pending_overhead_marker_count = 0
        self.last_overhead_marker_match = {
            "status": match_scope,
            **best_match,
        }
        self.last_yolo_hero_detection = {
            "position": best_match["position"],
            "size": best_match["size"],
            "confidence": best_match["confidence"],
            "frame_token": getattr(
                self, "_current_capture_frame_token", None
            ),
        }
        self._draw_debug_rectangle(
            best_match["position"],
            best_match["size"],
            (255, 80, 220),
            f"HeroYOLO:{best_match['confidence']:.3f},{match_scope}",
            thickness=2,
            text_height=0.45,
        )
        self._update_ladder_state_from_smile_pose(best_match["player"])
        return best_match["player"]

    def get_player_location_by_overhead_marker(
            self, expected_player=None, allow_global=True):
        """Locate Hero from the configured white smile bubble.

        A cheap connected-component pass first filters by the bubble's white
        geometry. Only those small candidates are checked with the masked face
        template. Local matches near the previous Hero position are accepted
        immediately; cold full-frame matches use consecutive-frame protection.
        Once established, the last valid Hero position remains authoritative
        across detection misses until a new match is confirmed.
        """
        marker_cfg = (
            self.cfg.get("nametag", {}).get("overhead_marker", {})
        )
        if not marker_cfg.get("enable", False):
            return None

        template = getattr(self, "img_overhead_marker", None)
        if template is None or getattr(self, "img_frame", None) is None:
            return None
        if getattr(self, "overhead_marker_component_bbox", None) is None or \
                getattr(self, "img_overhead_marker_mask", None) is None:
            self._update_overhead_marker_template_metadata(marker_cfg)
        component_bbox = getattr(
            self, "overhead_marker_component_bbox", None
        )
        template_mask = getattr(self, "img_overhead_marker_mask", None)
        template_gray = getattr(self, "img_overhead_marker_gray", None)
        if component_bbox is None or template_mask is None or \
                template_gray is None:
            return None

        frame = self.img_frame
        camera_y_end = min(
            frame.shape[0], int(self.cfg["ui_coords"]["ui_y_start"])
        )
        if camera_y_end <= 0:
            return None
        img_camera = frame[:camera_y_end, :]
        frame_gray = getattr(self, "img_frame_gray", None)
        if frame_gray is None:
            img_camera_gray = cv2.cvtColor(img_camera, cv2.COLOR_BGR2GRAY)
        else:
            img_camera_gray = frame_gray[:camera_y_end, :]

        white_lower = marker_cfg.get("white_lower", (185, 185, 185))
        lower_white = np.asarray(white_lower, dtype=np.uint8)
        upper_white = np.asarray((255, 255, 255), dtype=np.uint8)
        expected_width = max(1, int(marker_cfg.get(
            "component_width", component_bbox[2]
        )))
        expected_height = max(1, int(marker_cfg.get(
            "component_height", component_bbox[3]
        )))
        size_tolerance = max(0.0, float(marker_cfg.get(
            "component_size_tolerance", 0.20
        )))
        width_tolerance = max(2, int(round(
            expected_width * size_tolerance
        )))
        height_tolerance = max(2, int(round(
            expected_height * size_tolerance
        )))
        min_fill_rate = float(marker_cfg.get("min_fill_rate", 0.60))
        max_fill_rate = float(marker_cfg.get("max_fill_rate", 0.90))
        match_tolerance = max(0, int(marker_cfg.get(
            "match_search_tolerance", 2
        )))
        diff_thres = float(marker_cfg.get("diff_thres", 0.02))
        player_offset = tuple(map(int, marker_cfg.get(
            "player_offset", (0, 0)
        )))
        template_h, template_w = template_gray.shape[:2]
        component_template_x, component_template_y, _, _ = component_bbox
        camera_h, camera_w = img_camera.shape[:2]

        def scan_region(bounds):
            region_x0, region_y0, region_x1, region_y1 = bounds
            region_x0 = max(0, min(camera_w, int(region_x0)))
            region_y0 = max(0, min(camera_h, int(region_y0)))
            region_x1 = max(region_x0, min(camera_w, int(region_x1)))
            region_y1 = max(region_y0, min(camera_h, int(region_y1)))
            if region_x1 <= region_x0 or region_y1 <= region_y0:
                return []

            white_mask = cv2.inRange(
                img_camera[region_y0:region_y1, region_x0:region_x1],
                lower_white,
                upper_white,
            )
            count, _, stats, _ = cv2.connectedComponentsWithStats(
                white_mask, connectivity=8
            )
            matches = []
            for component_index in range(1, count):
                x = int(stats[component_index, cv2.CC_STAT_LEFT])
                y = int(stats[component_index, cv2.CC_STAT_TOP])
                width = int(stats[component_index, cv2.CC_STAT_WIDTH])
                height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
                area = int(stats[component_index, cv2.CC_STAT_AREA])
                if abs(width - expected_width) > width_tolerance or \
                        abs(height - expected_height) > height_tolerance:
                    continue
                fill_rate = area / float(max(1, width * height))
                if not min_fill_rate <= fill_rate <= max_fill_rate:
                    continue

                component_x = region_x0 + x
                component_y = region_y0 + y
                estimated_template_x = component_x - component_template_x
                estimated_template_y = component_y - component_template_y
                search_x0 = max(
                    0, estimated_template_x - match_tolerance
                )
                search_y0 = max(
                    0, estimated_template_y - match_tolerance
                )
                search_x1 = min(
                    camera_w,
                    estimated_template_x + template_w + match_tolerance,
                )
                search_y1 = min(
                    camera_h,
                    estimated_template_y + template_h + match_tolerance,
                )
                search_image = img_camera_gray[
                    search_y0:search_y1, search_x0:search_x1
                ]
                if search_image.shape[0] < template_h or \
                        search_image.shape[1] < template_w:
                    continue

                result = cv2.matchTemplate(
                    search_image,
                    template_gray,
                    cv2.TM_SQDIFF_NORMED,
                    mask=template_mask,
                )
                result = np.nan_to_num(
                    result, nan=1.0, posinf=1.0, neginf=1.0
                )
                score, _, match_loc, _ = cv2.minMaxLoc(result)
                score = float(score)
                if score >= diff_thres:
                    continue

                template_x = search_x0 + int(match_loc[0])
                template_y = search_y0 + int(match_loc[1])
                matched_component = (
                    template_x + component_template_x,
                    template_y + component_template_y,
                )
                player = (
                    matched_component[0] + player_offset[0],
                    matched_component[1] + player_offset[1],
                )
                if not (
                    0 <= player[0] < camera_w
                    and 0 <= player[1] < camera_h
                ):
                    continue
                matches.append({
                    "score": score,
                    "component": matched_component,
                    "shape": (height, width),
                    "player": player,
                    "fill_rate": fill_rate,
                })
            return matches

        def record_miss(status, *, reset_pending=True, match=None):
            self.overhead_marker_miss_count = getattr(
                self, "overhead_marker_miss_count", 0
            ) + 1
            if reset_pending:
                self.pending_overhead_marker_location = None
                self.pending_overhead_marker_count = 0
            now = time.monotonic()
            last_detected = getattr(
                self, "t_last_overhead_marker_detected", None
            )
            cache_age_s = (
                None if last_detected is None else now - last_detected
            )
            use_cached_location = (
                getattr(
                    self, "has_valid_overhead_marker_location", False
                )
                and getattr(
                    self, "loc_overhead_marker_player", None
                ) is not None
            )
            self.last_overhead_marker_match = {
                "status": f"{status},cached" if use_cached_location else status,
                "cache_age_s": cache_age_s,
                **({} if match is None else match),
            }
            if not use_cached_location:
                return None

            cached_component = (
                self.loc_overhead_marker_player[0] - player_offset[0],
                self.loc_overhead_marker_player[1] - player_offset[1],
            )
            self._draw_debug_rectangle(
                cached_component,
                (expected_height, expected_width),
                (0, 165, 255),
                "HeroSmile:cached" if cache_age_s is None else
                f"HeroSmile:cached {cache_age_s:.1f}s",
                thickness=1,
                text_height=0.45,
            )
            return self.loc_overhead_marker_player

        if expected_player is None and getattr(
                self, "has_valid_overhead_marker_location", False):
            expected_player = self.loc_overhead_marker_player

        best_match = None
        match_scope = "global"
        if expected_player is not None:
            expected_player = tuple(map(int, expected_player))
            expected_component = (
                expected_player[0] - player_offset[0],
                expected_player[1] - player_offset[1],
            )
            local_radius = max(1, int(marker_cfg.get(
                "local_search_radius", 90
            )))
            local_matches = scan_region((
                expected_component[0] - local_radius,
                expected_component[1] - local_radius,
                expected_component[0] + expected_width + local_radius,
                expected_component[1] + expected_height + local_radius,
            ))
            if local_matches:
                if marker_cfg.get("require_unique_local", True) and \
                        len(local_matches) != 1:
                    return record_miss("ambiguous-local")
                best_match = min(local_matches, key=lambda item: (
                    max(
                        abs(item["player"][0] - expected_player[0]),
                        abs(item["player"][1] - expected_player[1]),
                    ),
                    item["score"],
                ))
                match_scope = "local"

        if best_match is None:
            if not allow_global:
                return record_miss("not-found")
            global_matches = scan_region((0, 0, camera_w, camera_h))
            if not global_matches:
                return record_miss("not-found")
            if marker_cfg.get("require_unique_global", True) and \
                    len(global_matches) != 1:
                return record_miss("ambiguous")
            best_match = min(
                global_matches, key=lambda item: item["score"]
            )

        if match_scope == "global":
            confirm_frames = max(1, int(marker_cfg.get(
                "global_confirm_frames", 2
            )))
            if confirm_frames > 1:
                pending = getattr(
                    self, "pending_overhead_marker_location", None
                )
                if pending is not None:
                    self.pending_overhead_marker_count = getattr(
                        self, "pending_overhead_marker_count", 0
                    ) + 1
                else:
                    self.pending_overhead_marker_count = 1
                self.pending_overhead_marker_location = best_match["player"]
                if self.pending_overhead_marker_count < confirm_frames:
                    self._draw_debug_rectangle(
                        best_match["component"],
                        best_match["shape"],
                        (0, 165, 255),
                        (
                            "HeroSmile:pending "
                            f"{self.pending_overhead_marker_count}/"
                            f"{confirm_frames}"
                        ),
                        thickness=1,
                        text_height=0.45,
                    )
                    return record_miss(
                        "pending", reset_pending=False, match=best_match
                    )

        self.loc_overhead_marker_player = best_match["player"]
        self.has_valid_overhead_marker_location = True
        self.overhead_marker_miss_count = 0
        self.t_last_overhead_marker_detected = time.monotonic()
        self.pending_overhead_marker_location = None
        self.pending_overhead_marker_count = 0
        self.last_overhead_marker_match = {
            "status": match_scope,
            **best_match,
        }
        self._draw_debug_rectangle(
            best_match["component"],
            best_match["shape"],
            (0, 255, 0),
            f"HeroSmile:{best_match['score']:.3f},{match_scope}",
            thickness=1,
            text_height=0.45,
        )
        self._update_ladder_state_from_smile_pose(best_match["player"])
        return best_match["player"]

    def _update_ladder_state_from_smile_pose(self, player_location):
        """Classify the tiger hood below a fresh screen-space Hero anchor.

        The historical method name is retained for compatibility. The active
        smile-template or YOLO backend provides the Hero anchor, after which
        appearance matching is restricted to that anchor. A climbing-hood
        match enters ladder state;
        every other result is treated as flat ground, so stale climbing state
        can never survive a fresh non-climbing frame.
        """
        appearance_cfg = self.cfg.get("nametag", {}).get("appearance", {})
        templates = getattr(self, "nametag_appearance_templates", [])
        if appearance_cfg.get("enable", False) is False or not templates:
            return None

        matched_player = self.get_player_location_by_appearance(
            expected_player=player_location,
            allow_global=False,
            strict_anchor=True,
        )
        match = getattr(self, "last_appearance_match", None)
        pose = str((match or {}).get("pose", "")).lower()
        new_state = matched_player is not None and pose == "climbing"

        previous_state = bool(getattr(self, "is_on_ladder", False))
        self.is_on_ladder = new_state
        if new_state:
            self._ladder_route_exit_confirmed_at = None
        elif previous_state:
            # Route geometry may be a couple of minimap pixels away from the
            # physical platform. Only a fresh climbing -> ground transition
            # may authorize handing control from Up/Down to that platform.
            self._ladder_route_exit_confirmed_at = time.monotonic()
        if not new_state:
            # A fresh frame without the climbing hood means the character is
            # on flat ground, so platform movement may take over again.
            self._reset_ladder_route_hold()
        if previous_state != new_state:
            if match:
                evidence = (
                    f"pose={pose or 'unknown'},"
                    f"score={float(match['score']):.3f}"
                )
            else:
                evidence = "climbing-template-not-matched"
            logger.info(
                "[ladder] Hero-anchored tiger pose changed state to "
                f"{'climbing' if new_state else 'ground'} "
                f"({evidence})"
            )
        return new_state

    def get_player_location_on_screen(self):
        """Run exactly one configured screen-space Hero locator."""
        expected_player = None
        if getattr(self, "screen_player_location_valid", False):
            expected_player = getattr(self, "loc_player", None)
        marker_cfg = (
            self.cfg.get("nametag", {}).get("overhead_marker", {})
        )
        if marker_cfg.get("enable", False):
            if str(marker_cfg.get("backend", "template")).lower() == "yolo":
                return self.get_player_location_by_yolo(
                    expected_player=expected_player
                ), None
            return self.get_player_location_by_overhead_marker(
                expected_player=expected_player, allow_global=True
            ), None

        if self.cfg["nametag"]["enable"]:
            return self.get_player_location_by_nametag(), None
        return self.get_player_location_by_party_red_bar()

    def get_player_location_by_appearance(
            self, expected_player=None, allow_global=True,
            strict_anchor=False):
        """Find the configured hood/head templates and infer player center.

        Standing templates may recover a fully covered nametag globally. The
        rope template contains only the tiger hood (the pet covers the lower
        head), so it is intentionally restricted to an expected local area or
        a previously established ladder state.
        """
        appearance_cfg = self.cfg.get("nametag", {}).get(
            "appearance", {}
        )
        templates = getattr(self, "nametag_appearance_templates", [])
        if appearance_cfg.get("enable", False) is False or not templates:
            return None

        camera_h = min(
            int(self.cfg["ui_coords"]["ui_y_start"]),
            self.img_frame_gray.shape[0],
        )
        img_camera = self.img_frame_gray[:camera_h, :]
        camera_w = img_camera.shape[1]
        search_radius = max(
            10, int(appearance_cfg.get("local_search_radius", 90))
        )
        diff_threshold = float(
            appearance_cfg.get("diff_thres", 0.16)
        )
        climb_threshold = float(
            appearance_cfg.get("climb_diff_thres", 0.12)
        )

        candidates = []
        for template in templates:
            pose = template["pose"]
            if pose == "climbing" and expected_player is None:
                # A hood-only full-frame match is too ambiguous. It may still
                # recover during a known ladder segment, but only near the
                # last accepted screen coordinate.
                if (
                    not getattr(self, "is_on_ladder", False)
                    or not getattr(self, "has_valid_nametag_location", False)
                ):
                    continue
                last_id = getattr(self, "loc_nametag", None)
                if last_id is None:
                    continue
                template_expected = (
                    last_id[0] + self.img_nametag.shape[1] // 2,
                    last_id[1]
                    - self.cfg["nametag"]["offset"][1],
                )
            else:
                template_expected = expected_player

            template_gray = template["gray"]
            template_mask = template["mask"]
            template_h, template_w = template_gray.shape[:2]
            offset_x, offset_y = template["player_offset"]

            x0, y0 = 0, 0
            search_image = img_camera
            is_local = template_expected is not None
            if is_local:
                expected_x, expected_y = template_expected
                expected_template_x = int(round(expected_x - offset_x))
                expected_template_y = int(round(expected_y - offset_y))
                x0 = max(0, expected_template_x - search_radius)
                y0 = max(0, expected_template_y - search_radius)
                x1 = min(
                    camera_w,
                    expected_template_x + template_w + search_radius,
                )
                y1 = min(
                    camera_h,
                    expected_template_y + template_h + search_radius,
                )
                search_image = img_camera[y0:y1, x0:x1]
            elif not allow_global:
                continue

            if (
                search_image.shape[0] < template_h
                or search_image.shape[1] < template_w
            ):
                continue

            local_loc, score, _ = find_pattern_sqdiff(
                search_image,
                template_gray,
                last_result=None,
                mask=template_mask,
                global_threshold=0.0,
            )
            score = float(score)
            threshold = (
                climb_threshold if pose == "climbing" else diff_threshold
            )
            if score >= threshold:
                continue

            appearance_loc = (
                x0 + local_loc[0],
                y0 + local_loc[1],
            )
            player = (
                appearance_loc[0] + offset_x,
                appearance_loc[1] + offset_y,
            )
            if not (0 <= player[0] < camera_w and 0 <= player[1] < camera_h):
                continue

            # A local appearance check must agree geometrically with the
            # nametag-derived point. A global standing recovery instead uses
            # a stricter score plus the existing two-frame jump confirmation.
            if is_local:
                validation_distance = int(
                    appearance_cfg.get("validation_distance", 30)
                )
                max_distance = validation_distance
                if pose == "climbing" and not strict_anchor:
                    max_distance = int(appearance_cfg.get(
                        "climb_validation_distance", validation_distance
                    ))
                distance = max(
                    abs(player[0] - template_expected[0]),
                    abs(player[1] - template_expected[1]),
                )
                if distance > max_distance:
                    continue

            candidates.append({
                "player": player,
                "loc": appearance_loc,
                "score": score,
                "name": template["name"],
                "pose": pose,
                "shape": template["image"].shape,
                "is_local": is_local,
            })

        if not candidates:
            self.has_valid_appearance_location = False
            self.last_appearance_match = None
            if expected_player is None:
                self.pending_appearance_location = None
                self.pending_appearance_count = 0
            return None

        candidates.sort(key=lambda item: item["score"])
        best = candidates[0]
        self.last_appearance_match = best

        # A head template is a stronger anchor than background color but less
        # unique than ID + medal. Require two consistent global frames before
        # it can initialize or recover control; local checks are already bound
        # to a known player coordinate and need no extra delay.
        if not best["is_local"]:
            confirm_frames = max(
                1, int(appearance_cfg.get("global_confirm_frames", 2))
            )
            confirm_radius = max(
                1, int(appearance_cfg.get("global_confirm_radius", 12))
            )
            previous_confirmed = getattr(
                self, "has_valid_appearance_location", False
            ) and max(
                abs(best["player"][0] - self.loc_appearance_player[0]),
                abs(best["player"][1] - self.loc_appearance_player[1]),
            ) <= confirm_radius
            if not previous_confirmed:
                pending = getattr(
                    self, "pending_appearance_location", None
                )
                if pending is not None and max(
                    abs(best["player"][0] - pending[0]),
                    abs(best["player"][1] - pending[1]),
                ) <= confirm_radius:
                    self.pending_appearance_count = (
                        getattr(self, "pending_appearance_count", 0) + 1
                    )
                else:
                    self.pending_appearance_location = best["player"]
                    self.pending_appearance_count = 1
                if self.pending_appearance_count < confirm_frames:
                    self._draw_debug_rectangle(
                        best["loc"],
                        best["shape"],
                        (0, 165, 255),
                        (
                            f"HeroHead:pending "
                            f"{self.pending_appearance_count}/{confirm_frames}"
                        ),
                        thickness=1,
                        text_height=0.45,
                    )
                    return None

        self.loc_appearance_player = best["player"]
        self.has_valid_appearance_location = True
        self.pending_appearance_location = None
        self.pending_appearance_count = 0
        self._draw_debug_rectangle(
            best["loc"],
            best["shape"],
            (0, 165, 255),
            f"HeroHead:{best['pose']} {best['score']:.2f}",
            thickness=1,
            text_height=0.45,
        )
        return best["player"]

    def get_player_location_by_nametag(self):
        '''
        Detect the player from their ID and, when configured, the medal below it.

        Overlapping ID fragments tolerate a pet or another player covering part of
        the text. A fragment is allowed a looser score only when the medal also
        matches at the expected center-aligned position. The medal is never used by
        itself, because nearby players can equip the same medal.

        Returns:
            loc_player (tuple): The (x, y) coordinates of the player's estimated location.
        '''
        # Get camera region in the game window
        img_camera = self.img_frame_gray[
            :self.cfg["ui_coords"]["ui_y_start"], :]

        nametag_cfg = self.cfg["nametag"]
        mode = nametag_cfg["mode"]

        def ensure_grayscale(image, color_image, resource_name):
            """Return a single-channel template after a reload/state mismatch."""
            if image is None:
                image = color_image
            if image is None:
                raise RuntimeError(f"Missing {resource_name} template")
            if image.ndim == 2:
                return image
            if image.ndim == 3 and image.shape[2] == 1:
                return image[:, :, 0]
            if image.ndim == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 and image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            raise ValueError(
                f"Unexpected {resource_name} template shape: {image.shape}"
            )

        img_nametag_gray = ensure_grayscale(
            getattr(self, "img_nametag_gray", None),
            getattr(self, "img_nametag", None),
            "nametag",
        )

        # Get nametag image and search image.
        if mode == "white_mask":
            # Apply Gaussian blur for smoother white detection
            img_camera = cv2.GaussianBlur(img_camera, (3, 3), 0)
            img_nametag = cv2.GaussianBlur(img_nametag_gray, (3, 3), 0)
            lower_white, upper_white = (150, 255)
            img_camera_search = cv2.inRange(
                img_camera, lower_white, upper_white
            )
            img_nametag  = cv2.inRange(img_nametag, lower_white, upper_white)
        elif mode == "grayscale":
            img_camera_search = img_camera
            img_nametag = img_nametag_gray
        elif mode == "histogram_eq":
            # Apply histogram equalization
            img_nametag_eq = cv2.equalizeHist(img_nametag_gray)
            img_camera_eq = cv2.equalizeHist(img_camera)

            # Apply global (fixed) threshold
            _, img_nametag = cv2.threshold(img_nametag_eq, 150, 255, cv2.THRESH_BINARY)
            _, img_camera_search = cv2.threshold(
                img_camera_eq, 150, 255, cv2.THRESH_BINARY
            )
        else:
            logger.error(f"Unsupported nametag detection mode: {mode}")
            return

        medal_cfg = nametag_cfg.get("medal", {})
        img_medal_color = getattr(self, "img_nametag_medal", None)
        img_medal_gray = getattr(self, "img_nametag_medal_gray", None)
        use_medal = bool(
            medal_cfg.get("enable", False)
            and img_medal_color is not None
            and img_medal_gray is not None
        )
        img_medal = None
        medal_mask = None
        if use_medal:
            img_medal_gray = ensure_grayscale(
                img_medal_gray, img_medal_color, "nametag medal"
            )
            if mode == "white_mask":
                img_medal = cv2.GaussianBlur(img_medal_gray, (3, 3), 0)
                img_medal = cv2.inRange(
                    img_medal, lower_white, upper_white
                )
            elif mode == "histogram_eq":
                img_medal_eq = cv2.equalizeHist(img_medal_gray)
                _, img_medal = cv2.threshold(
                    img_medal_eq, 150, 255, cv2.THRESH_BINARY
                )
            else:
                img_medal = img_medal_gray
            medal_mask = get_mask(img_medal_color, (0, 255, 0))

        pet_cfg = nametag_cfg.get("pet", {})
        img_pet_color = getattr(self, "img_nametag_pet", None)
        img_pet_gray = getattr(self, "img_nametag_pet_gray", None)
        use_pet = bool(
            use_medal
            and pet_cfg.get("enable", False)
            and img_pet_color is not None
            and img_pet_gray is not None
        )
        img_pet = None
        pet_mask = None
        if use_pet:
            img_pet_gray = ensure_grayscale(
                img_pet_gray, img_pet_color, "pet nametag"
            )
            if mode == "white_mask":
                img_pet = cv2.GaussianBlur(img_pet_gray, (3, 3), 0)
                img_pet = cv2.inRange(
                    img_pet, lower_white, upper_white
                )
            elif mode == "histogram_eq":
                img_pet_eq = cv2.equalizeHist(img_pet_gray)
                _, img_pet = cv2.threshold(
                    img_pet_eq, 150, 255, cv2.THRESH_BINARY
                )
            else:
                img_pet = img_pet_gray
            pet_mask = get_mask(img_pet_color, (0, 255, 0))

        # Pad search region to deal with fail detection when player is at map edge
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi = cv2.copyMakeBorder(
            img_camera_search,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Only use a cached search center while the prior location is valid.
        if getattr(self, "has_valid_nametag_location", False):
            last_result = (
                self.loc_nametag[0] + pad_x,
                self.loc_nametag[1] + pad_y
            )
        else:
            last_result = None

        h, w = img_nametag.shape[:2]
        # Get nametag's background mask
        mask = get_mask(self.img_nametag, (0, 255, 0))

        # ID-only mode retains the legacy equal split behavior. With a medal,
        # use the complete ID plus overlapping fragments, so an occluder cannot
        # remove the same glyphs from every candidate.
        nametag_splits = {}
        if use_medal:
            regions = [("full", 0, w)]
            fragment_width = min(
                w,
                max(
                    1,
                    int(
                        medal_cfg.get(
                            "id_fragment_width",
                            nametag_cfg.get("split_width", w),
                        )
                    ),
                ),
            )
            if fragment_width < w:
                fragment_stride = max(
                    1,
                    int(
                        medal_cfg.get(
                            "id_fragment_stride", fragment_width // 2
                        )
                    ),
                )
                starts = list(
                    range(0, w - fragment_width + 1, fragment_stride)
                )
                if starts[-1] != w - fragment_width:
                    starts.append(w - fragment_width)
                starts = sorted(set(starts))
                for index, x_s in enumerate(starts, start=1):
                    regions.append(
                        (
                            f"part {index}/{len(starts)}",
                            x_s,
                            x_s + fragment_width,
                        )
                    )
        else:
            split_width = max(1, int(nametag_cfg["split_width"]))
            num_splits = max(1, w // split_width)
            w_split = w // num_splits
            regions = []
            for i in range(num_splits):
                x_s = i * w_split
                x_e = (i + 1) * w_split if i < num_splits - 1 else w
                regions.append((f"{i+1}/{num_splits}", x_s, x_e))

        for tag_type, x_s, x_e in regions:
            nametag_splits[tag_type] = {
                "img": img_nametag[:, x_s:x_e],
                "mask": mask[:, x_s:x_e],
                "last_result": (
                    (last_result[0] + x_s, last_result[1])
                    if last_result else None
                ),
                "offset_x": x_s,
            }

        # Match ID candidates.
        matches = []
        for tag_type, split in nametag_splits.items():
            loc, score, is_cached = find_pattern_sqdiff(
                img_roi,
                split["img"],
                last_result=split["last_result"],
                mask=split["mask"],
                global_threshold=nametag_cfg["global_diff_thres"]
            )
            # A weak local result is only a candidate. Compare it with a full
            # search so a nearby rock/rope texture cannot beat the true tag.
            cache_accept_thres = self.cfg["nametag"].get(
                "cache_accept_thres", 0.12
            )
            # While a large jump is pending, always obtain a global candidate:
            # the local cache is still centered on the old pre-scroll point.
            if is_cached and (
                score > cache_accept_thres
                or getattr(self, "pending_nametag_location", None) is not None
            ):
                global_loc, global_score, _ = find_pattern_sqdiff(
                    img_roi,
                    split["img"],
                    last_result=None,
                    mask=split["mask"],
                    global_threshold=0.0,
                )
                if global_score < score:
                    loc, score, is_cached = global_loc, global_score, False
            w_match = split["img"].shape[1]
            h_match = split["img"].shape[0]
            offset_x = split["offset_x"]
            loc_nametag = (
                loc[0] - offset_x - pad_x,
                loc[1] - pad_y,
            )
            matches.append({
                "tag_type": tag_type,
                "loc_nametag": loc_nametag,
                "id_score": float(score),
                "is_cached": is_cached,
                "w_match": w_match,
                "h_match": h_match,
                "medal_loc": None,
                "medal_score": None,
                "medal_partial": False,
                "medal_match_height": None,
                "is_valid": False,
            })

        # Verify each ID candidate against the medal only inside a small area
        # below it. This geometrical gate rejects the same medal under a nearby
        # player and costs much less than a second full-frame search.
        if use_medal:
            medal_h, medal_w = img_medal.shape[:2]
            tolerance = medal_cfg.get("search_tolerance", (18, 6))
            tolerance_x = max(0, int(tolerance[0]))
            tolerance_y = max(0, int(tolerance[1]))
            center_offset_x = int(medal_cfg.get("center_offset_x", 0))
            vertical_gap = int(medal_cfg.get("vertical_gap", 0))
            camera_h, camera_w = img_camera_search.shape[:2]
            medal_threshold = float(medal_cfg.get("diff_thres", 0.18))
            assisted_id_threshold = float(
                medal_cfg.get(
                    "assisted_id_diff_thres",
                    nametag_cfg["diff_thres"],
                )
            )
            bottom_partial_threshold = float(
                medal_cfg.get("bottom_partial_diff_thres", medal_threshold)
            )
            bottom_partial_id_threshold = float(
                medal_cfg.get(
                    "bottom_partial_id_diff_thres",
                    assisted_id_threshold,
                )
            )
            bottom_min_visible_ratio = min(
                1.0,
                max(
                    0.1,
                    float(
                        medal_cfg.get("bottom_min_visible_ratio", 0.5)
                    ),
                ),
            )
            bottom_min_visible_rows = max(
                3,
                min(
                    medal_h,
                    int(np.ceil(medal_h * bottom_min_visible_ratio)),
                ),
            )

            def match_medal_near(expected_x, expected_y, tolerance):
                """Match a full medal, or its visible top at the camera bottom."""
                local_tolerance_x = max(0, int(tolerance[0]))
                local_tolerance_y = max(0, int(tolerance[1]))
                expected_x = int(expected_x)
                expected_y = int(expected_y)

                # Only relax the template vertically when its expected bottom
                # crosses the camera/UI boundary. Elsewhere a complete medal
                # remains mandatory, so normal false-positive behavior is
                # unchanged.
                is_bottom_partial = expected_y + medal_h > camera_h
                match_image = img_medal
                match_mask = medal_mask
                match_height = medal_h
                if is_bottom_partial:
                    visible_rows = camera_h - expected_y
                    if visible_rows < bottom_min_visible_rows:
                        return None, 1.0, True, 0
                    match_height = bottom_min_visible_rows
                    match_image = img_medal[:match_height, :]
                    if medal_mask is not None:
                        match_mask = medal_mask[:match_height, :]

                x0 = max(0, expected_x - local_tolerance_x)
                y0 = max(0, expected_y - local_tolerance_y)
                x1 = min(
                    camera_w,
                    expected_x + medal_w + local_tolerance_x,
                )
                y1 = (
                    camera_h
                    if is_bottom_partial
                    else min(
                        camera_h,
                        expected_y + medal_h + local_tolerance_y,
                    )
                )
                medal_roi = img_camera_search[y0:y1, x0:x1]
                if (
                    medal_roi.shape[0] < match_height
                    or medal_roi.shape[1] < medal_w
                ):
                    return None, 1.0, is_bottom_partial, 0

                local_loc, score, _ = find_pattern_sqdiff(
                    medal_roi,
                    match_image,
                    last_result=None,
                    mask=match_mask,
                    global_threshold=0.0,
                )
                return (
                    (x0 + local_loc[0], y0 + local_loc[1]),
                    float(score),
                    is_bottom_partial,
                    match_height,
                )

            for match in matches:
                id_x, id_y = match["loc_nametag"]
                expected_x = int(round(
                    id_x + w / 2 - medal_w / 2 + center_offset_x
                ))
                expected_y = id_y + h + vertical_gap
                (
                    medal_loc,
                    medal_score,
                    medal_partial,
                    medal_match_height,
                ) = match_medal_near(
                    expected_x,
                    expected_y,
                    (tolerance_x, tolerance_y),
                )
                match["medal_loc"] = medal_loc
                match["medal_score"] = float(medal_score)
                match["medal_partial"] = medal_partial
                match["medal_match_height"] = medal_match_height
                score_threshold = (
                    bottom_partial_threshold
                    if medal_partial else medal_threshold
                )
                id_score_threshold = (
                    bottom_partial_id_threshold
                    if medal_partial else assisted_id_threshold
                )
                match["is_valid"] = (
                    medal_loc is not None
                    and match["id_score"] < id_score_threshold
                    and medal_score < score_threshold
                )

            # If every ID candidate is occluded, the configured pet name can
            # recover the identity. The pet only defines a search area: the
            # medal remains the position anchor, so pet movement cannot move
            # the reported player coordinate directly.
            if use_pet and not any(match["is_valid"] for match in matches):
                pet_loc, pet_score, pet_is_cached = find_pattern_sqdiff(
                    img_camera_search,
                    img_pet,
                    last_result=None,
                    mask=pet_mask,
                    global_threshold=0.0,
                )
                if pet_score < float(pet_cfg.get("diff_thres", 0.18)):
                    pet_h, pet_w = img_pet.shape[:2]
                    medal_h, medal_w = img_medal.shape[:2]
                    relative_offset = pet_cfg.get(
                        "medal_offset", (36, 17)
                    )
                    expected_medal_x = pet_loc[0] + int(relative_offset[0])
                    expected_medal_y = pet_loc[1] + int(relative_offset[1])
                    pet_medal_tolerance = pet_cfg.get(
                        "medal_search_tolerance", (28, 10)
                    )
                    (
                        pet_medal_loc,
                        pet_medal_score,
                        pet_medal_partial,
                        pet_medal_match_height,
                    ) = match_medal_near(
                        expected_medal_x,
                        expected_medal_y,
                        pet_medal_tolerance,
                    )
                    pet_medal_score_threshold = (
                        bottom_partial_threshold
                        if pet_medal_partial else medal_threshold
                    )
                    if (
                        pet_medal_loc is not None
                        and pet_medal_score < pet_medal_score_threshold
                    ):
                            center_offset_x = int(
                                medal_cfg.get("center_offset_x", 0)
                            )
                            vertical_gap = int(
                                medal_cfg.get("vertical_gap", 0)
                            )
                            inferred_id_x = int(round(
                                pet_medal_loc[0]
                                + medal_w / 2
                                - w / 2
                                - center_offset_x
                            ))
                            inferred_id_y = (
                                pet_medal_loc[1]
                                - h
                                - vertical_gap
                            )
                            matches.append({
                                "tag_type": "pet+medal",
                                "loc_nametag": (
                                    inferred_id_x, inferred_id_y
                                ),
                                "id_score": float(pet_score),
                                "is_cached": pet_is_cached,
                                "w_match": pet_w,
                                "h_match": pet_h,
                                "medal_loc": pet_medal_loc,
                                "medal_score": float(pet_medal_score),
                                "medal_partial": pet_medal_partial,
                                "medal_match_height": pet_medal_match_height,
                                "pet_loc": pet_loc,
                                "is_valid": True,
                            })
        else:
            for match in matches:
                match["is_valid"] = (
                    match["id_score"] < nametag_cfg["diff_thres"]
                )

        # A valid pair always wins over an invalid candidate even if the latter
        # has a slightly lower ID-only score.
        def match_sort_key(match):
            combined_score = match["id_score"]
            if match["medal_score"] is not None:
                combined_score += match["medal_score"]
            return (not match["is_valid"], combined_score)

        matches.sort(key=match_sort_key)
        best_match = matches[0]
        tag_type = best_match["tag_type"]
        loc_nametag = best_match["loc_nametag"]
        score = best_match["id_score"]
        medal_loc = best_match["medal_loc"]
        medal_score = best_match["medal_score"]
        medal_partial = best_match.get("medal_partial", False)
        medal_match_height = best_match.get("medal_match_height")
        pet_loc = best_match.get("pet_loc")
        is_cached = best_match["is_cached"]

        # Only update nametag location when score is good enough. On startup,
        # never turn the default (0, 0) into a fake player coordinate when the
        # configured template is absent or does not match the current hero.
        is_valid_match = best_match["is_valid"]
        candidate_player = (
            loc_nametag[0] + w // 2,
            loc_nametag[1] - nametag_cfg["offset"][1]
        )
        camera_h, camera_w = img_camera.shape[:2]
        is_valid_match = (
            is_valid_match
            and 0 <= candidate_player[0] < camera_w
            and 0 <= candidate_player[1] < camera_h
        )

        # Appearance is an additional identity anchor. A good local head
        # match confirms the nametag pair, but a miss never vetoes a valid
        # ID/medal because animation, damage effects, or the pet may still
        # cover part of the hood.
        if is_valid_match:
            self.get_player_location_by_appearance(
                expected_player=candidate_player,
                allow_global=False,
            )

        # When every text anchor is covered, a strict standing head template
        # can recover the screen coordinate. Convert that coordinate back to
        # a synthetic ID location so the existing stale/jump safeguards stay
        # in one place. The ambiguous hood-only climbing template never takes
        # this unrestricted global path.
        if not is_valid_match:
            appearance_player = self.get_player_location_by_appearance(
                expected_player=None,
                allow_global=True,
            )
            if appearance_player is not None:
                candidate_player = appearance_player
                loc_nametag = (
                    appearance_player[0] - w // 2,
                    appearance_player[1] + nametag_cfg["offset"][1],
                )
                tag_type = "appearance"
                appearance_match = getattr(
                    self, "last_appearance_match", None
                ) or {}
                score = float(appearance_match.get("score", 0.0))
                medal_loc = None
                medal_score = None
                medal_partial = False
                medal_match_height = None
                pet_loc = None
                is_cached = False
                is_valid_match = True

        # Camera scrolling after a ladder climb can move the true tag a long
        # way in one frame, but a single low-score texture hit can do the same.
        # Require large jumps to appear in two consecutive frames before they
        # replace the current control coordinate.
        needs_jump_confirmation = False
        if is_valid_match and getattr(self, "has_valid_nametag_location", False):
            jump_distance = max(
                abs(loc_nametag[0] - self.loc_nametag[0]),
                abs(loc_nametag[1] - self.loc_nametag[1]),
            )
            jump_confirm_distance = self.cfg["nametag"].get(
                "jump_confirm_distance", 40
            )
            if jump_distance > jump_confirm_distance:
                pending = getattr(self, "pending_nametag_location", None)
                jump_confirm_radius = self.cfg["nametag"].get(
                    "jump_confirm_radius", 12
                )
                if pending is None or max(
                    abs(loc_nametag[0] - pending[0]),
                    abs(loc_nametag[1] - pending[1]),
                ) > jump_confirm_radius:
                    self.pending_nametag_location = loc_nametag
                    needs_jump_confirmation = True

        is_current_match = is_valid_match and not needs_jump_confirmation
        if is_current_match:
            self.loc_nametag = loc_nametag
            if medal_loc is not None:
                self.loc_nametag_medal = medal_loc
            self.has_valid_nametag_location = True
            self.nametag_miss_count = 0
            self.pending_nametag_location = None
        else:
            self.nametag_miss_count = getattr(self, "nametag_miss_count", 0) + 1

        max_stale_frames = self.cfg["nametag"].get("max_stale_frames", 2)
        is_stale = (
            not is_current_match
            and getattr(self, "has_valid_nametag_location", False)
            and self.nametag_miss_count <= max_stale_frames
        )
        if not is_current_match and not is_stale:
            self.has_valid_nametag_location = False
            if not needs_jump_confirmation:
                self.pending_nametag_location = None

        loc_player = (
            self.loc_nametag[0] + w // 2,
            self.loc_nametag[1] - nametag_cfg["offset"][1]
        )

        # An appearance-only recovery stores a synthetic ID location so it can
        # reuse the existing jump/stale safeguards. Do not draw that synthetic
        # box as though text was actually matched.
        if tag_type != "appearance":
            self._draw_debug_rectangle(
                self.loc_nametag,
                self.img_nametag.shape,
                (0, 255, 0),
                "",
            )
        debug_medal_loc = (
            medal_loc
            if medal_loc is not None
            else getattr(self, "loc_nametag_medal", None)
        )
        if (
            tag_type != "appearance"
            and use_medal
            and debug_medal_loc is not None
        ):
            debug_medal_shape = self.img_nametag_medal.shape
            if medal_partial and medal_match_height:
                debug_medal_shape = (
                    medal_match_height,
                    self.img_nametag_medal.shape[1],
                )
            self._draw_debug_rectangle(
                debug_medal_loc,
                debug_medal_shape,
                (255, 255, 0),
                "Medal(partial)" if medal_partial else "Medal",
                thickness=1,
                text_height=0.5,
            )
        if use_pet and pet_loc is not None:
            self._draw_debug_rectangle(
                pet_loc,
                self.img_nametag_pet.shape,
                (255, 0, 255),
                "Pet",
                thickness=1,
                text_height=0.5,
            )
        match_status = "cached" if is_cached else "global"
        if needs_jump_confirmation:
            match_status = "pending jump"
        elif is_stale:
            match_status = (
                f"stale {self.nametag_miss_count}/{max_stale_frames}"
            )
        elif not is_current_match:
            match_status = "rejected"
        medal_text = ""
        if use_medal:
            medal_text = (
                f",medal={medal_score:.2f}"
                f"{'/partial' if medal_partial else ''}"
                if medal_score is not None else ",medal=missing"
            )
        if tag_type == "appearance":
            appearance_match = getattr(
                self, "last_appearance_match", None
            ) or {}
            text = (
                f"HeroHead,{appearance_match.get('pose', 'unknown')}="
                f"{score:.2f},{match_status}"
            )
        else:
            text = (
                f"NameTag,id={score:.2f}{medal_text},"
                f"{match_status},{tag_type}"
            )
        self._draw_debug_text(
            text,
            (
                self.loc_nametag[0],
                self.loc_nametag[1] + self.img_nametag.shape[0]
                + int(round(30 * self.get_frame_visual_scale())),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # A stale box is useful in the visualization but must never drive
        # route or attack logic. Only a match accepted on this frame is current.
        if not is_current_match:
            return None

        return loc_player

    def get_player_location_by_party_red_bar(self):
        '''
        get_player_location_by_party_red_bar
        '''
        # Zero out minimap area in the img_frame
        img_frame = self.img_frame.copy()
        x, y = self.loc_minimap
        h, w = self.img_minimap_screen.shape[:2]
        img_frame[y:y+h, x:x+w] = 0

        # Get camera area
        img_camera = img_frame[:self.cfg["ui_coords"]["ui_y_start"], :]

        # Convert to HSV
        img_hsv = cv2.cvtColor(img_camera, cv2.COLOR_BGR2HSV)
        lower_red = to_opencv_hsv(self.cfg["party_red_bar"]["lower_red"])
        upper_red = to_opencv_hsv(self.cfg["party_red_bar"]["upper_red"])
        mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
        # cv2.imshow("mask_red", mask_red)

        # Find contours on mask_red
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter contour by specific geometry trait of red bar
        boxs = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            fill_rate = float(area) / (h*w)
            if 5 <= h <= 7 and 1 <= w <= 50 and 10 <= area and fill_rate >= 0.7:
                # cv2.drawContours(self.img_frame_debug, [c], -1, (0, 255, 0), 1)
                boxs.append((x, y, w, h))

        if not boxs:
            return None, None  # red bar not found

        # Sort box by area
        boxs.sort(key=lambda box: box[2] * box[3], reverse=True)

        # Consider the biggest area as party red bar
        x, y, w, h = boxs[0]

        # Offset coordinate
        loc_party_red_bar = (x, y)
        loc_player = (x + self.cfg["party_red_bar"]["offset"][0],
                      y + self.cfg["party_red_bar"]["offset"][1])

        # visualize for debug
        self._draw_debug_rectangle(
            loc_party_red_bar,
            (h, w),
            (0, 255, 0),
            "party red bar",
            thickness=1,
            text_height=0.4,
        )

        return loc_player, loc_party_red_bar

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        self.loc_minimap_global, score, _ = find_pattern_sqdiff(
                                        self.img_map,
                                        self.img_minimap)

        x_offset, y_offset = self.cfg["minimap"]["offset"]
        loc_player_global = (
            self.loc_minimap_global[0] + self.loc_player_minimap[0] + x_offset,
            self.loc_minimap_global[1] + self.loc_player_minimap[1] + y_offset
        )

        # Draw local minimap rectangle
        draw_rectangle(
            self.img_route_debug,
            self.loc_minimap_global,
            self.img_minimap.shape[:2],
            (0, 255, 255),
            "",
            thickness=1,
        )
        draw_text(
            self.img_route_debug,
            f"Minimap,score({round(score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        draw_circle(self.img_route_debug,
                    loc_player_global, radius=2,
                    color=(0, 255, 255), thickness=-1)

        return loc_player_global

    @staticmethod
    def _is_stationary_jump_route_command(command):
        """Return whether a route action is a directionless vertical jump."""
        parts = str(command).split()
        return (
            len(parts) == 3
            and parts[0] in {"none", "stop"}
            and parts[1] in {"none", "stop"}
            and parts[2] == "jump"
        )

    def _find_stationary_jump_targets(self, img_route):
        """Find explicit centers for compact stationary-jump marker blobs.

        The current recorder draws point actions as a radius-two disk. Older
        routes also contain long or fragmented magenta strokes whose geometric
        center is not necessarily the intended action point, so those retain
        the legacy pixel-overlap behavior.
        """
        if img_route is None:
            return ()

        targets = []
        for color, command in self.color_code.items():
            if not self._is_stationary_jump_route_command(command):
                continue

            mask = np.all(img_route == color, axis=2).astype(np.uint8)
            component_count, labels, stats, _ = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)
            for component_idx in range(1, component_count):
                x, y, width, height, area = map(
                    int, stats[component_idx]
                )
                # A recorder point is a compact radius-two blob. Do not infer
                # centers for historical horizontal strokes or tiny remnants.
                if width > 5 or height > 5 or width < 3 or height < 3 \
                        or area < 8:
                    continue

                ys, xs = np.where(labels == component_idx)
                center_x = float(np.mean(xs))
                center_y = float(np.mean(ys))
                distances = (
                    (xs.astype(float) - center_x) ** 2
                    + (ys.astype(float) - center_y) ** 2
                )
                center_idx = int(np.argmin(distances))
                center = (int(xs[center_idx]), int(ys[center_idx]))
                targets.append({
                    "center": center,
                    "color": tuple(color),
                    "command": command,
                    "pixels": frozenset(
                        (int(px), int(py)) for px, py in zip(xs, ys)
                    ),
                    "bbox": (x, y, width, height),
                })

        return tuple(sorted(
            targets,
            key=lambda target: (
                target["center"][1], target["center"][0]
            ),
        ))

    def _get_active_stationary_jump_targets(self):
        """Return cached targets for the active route, with a test fallback."""
        route_idx = int(getattr(self, "idx_routes", 0))
        routes = getattr(self, "img_routes", ())
        cached = getattr(self, "_stationary_jump_targets_by_route", ())
        if 0 <= route_idx < len(routes) and \
                0 <= route_idx < len(cached) and \
                self.img_route is routes[route_idx]:
            return cached[route_idx]
        return self._find_stationary_jump_targets(self.img_route)

    def _get_nearby_stationary_jump_target(self):
        """Return the nearest compact jump target inside route search range."""
        x0, y0 = self.loc_player_global
        search_range = max(0, int(self.cfg["route"]["search_range"]))
        candidates = []
        for target in self._get_active_stationary_jump_targets():
            target_x, target_y = target["center"]
            dx = target_x - x0
            dy = target_y - y0
            distance = abs(dx) + abs(dy)
            if distance > search_range:
                continue
            candidates.append((distance, target_y, target_x, target))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:3])[3]

    def _reset_ladder_route_hold(self):
        """Forget the pure Up/Down route direction held for a ladder."""
        self._ladder_route_move_y = None

    @staticmethod
    def _get_pure_ladder_route_direction(color_code_up_down):
        """Return Up/Down only for a movement-only ladder route color."""
        if not color_code_up_down:
            return None
        parts = str(color_code_up_down.get("command", "")).split()
        if len(parts) != 3 or parts[0] != "none" or parts[2] != "none":
            return None
        return parts[1] if parts[1] in {"up", "down"} else None

    def _get_ladder_route_alignment_move_x(self, color_code_up_down):
        """Return a ground alignment move before emitting ladder Up/Down.

        A legacy vertical route stroke is discoverable inside ``search_range``.
        Without an x-axis guard, that tolerance can press Up while Hero is
        beside the intended rope -- including while standing in a nearby map
        portal.  Once the climbing pose is visible the normal ladder hold owns
        input, so alignment is needed only while Hero is still on the ground.
        """
        if bool(getattr(self, "is_on_ladder", False)):
            return None
        direction = self._get_pure_ladder_route_direction(
            color_code_up_down
        )
        if direction is None or self.img_route is None:
            return None

        component = self._get_route_color_component(
            self.img_route,
            color_code_up_down["pixel"],
            color_code_up_down["color"],
        )
        if component is None:
            return None

        left, top, right, bottom = component["bbox"]
        if bottom - top + 1 <= right - left + 1:
            # Do not reinterpret a short horizontal smear as a rope/ladder.
            return None

        tolerance = max(
            0,
            int(self.cfg.get("route", {}).get(
                "ladder_alignment_tolerance", 2
            )),
        )
        player_x = int(self.loc_player_global[0])
        if player_x < left - tolerance:
            return "right"
        if player_x > right + tolerance:
            return "left"
        return None

    def _apply_ladder_route_hold(
            self, color_code_up_down=None, *, allow_restore=True):
        """Keep the selected vertical direction until ground is classified.

        Route lookup is intentionally local and can miss a vertical pixel for
        one frame near a platform junction. Cache only the dedicated pure
        Up/Down route colors; never infer a ladder hold from jump, teleport,
        portal, goal, or stop actions.
        """
        detected_direction = self._get_pure_ladder_route_direction(
            color_code_up_down
        )
        is_on_ladder = bool(getattr(self, "is_on_ladder", False))

        if not is_on_ladder:
            # Remember the approach command so the first climbing frame can
            # retain it even when the local route lookup moves to a junction.
            self._ladder_route_move_y = detected_direction
            return

        latched_direction = getattr(self, "_ladder_route_move_y", None)
        if latched_direction not in {"up", "down"}:
            latched_direction = detected_direction
            self._ladder_route_move_y = latched_direction

        if allow_restore and latched_direction in {"up", "down"}:
            self.cmd_move_y = latched_direction

    def _is_ladder_route_destination_reached(self, color_code_up_down):
        """Return whether Hero reached the directed vertical segment endpoint."""
        direction = self._get_pure_ladder_route_direction(
            color_code_up_down
        )
        if direction is None or self.img_route is None:
            return False

        component = self._get_route_color_component(
            self.img_route,
            color_code_up_down["pixel"],
            color_code_up_down["color"],
        )
        if component is None:
            return False

        left, top, right, bottom = component["bbox"]
        component_width = right - left + 1
        component_height = bottom - top + 1
        if component_height <= component_width:
            # A short horizontal smear of an Up/Down color is not a ladder.
            return False

        tolerance = max(
            0,
            int(self.cfg["route"].get("ladder_endpoint_tolerance", 2)),
        )
        player_y = int(self.loc_player_global[1])
        if direction == "up":
            return player_y <= top + tolerance
        return player_y >= bottom - tolerance

    def _has_recent_ladder_route_exit_confirmation(self):
        """Return True briefly after a visual climbing -> ground change."""
        confirmed_at = getattr(
            self, "_ladder_route_exit_confirmed_at", None
        )
        return confirmed_at is not None and \
            time.monotonic() - confirmed_at <= 0.5

    def _reset_stationary_jump_proximity(self):
        self._stationary_jump_proximity_active = False

    @staticmethod
    def _is_rope_climb_route_command(command):
        """Return whether a route color marks a rope's upper endpoint."""
        parts = str(command).split()
        return (
            len(parts) == 3
            and parts[0] in {"none", "stop"}
            and parts[1] == "up"
            and parts[2] == "climb"
        )

    def _find_rope_climb_targets(self, img_route, route_cfg=None):
        """Find guide components and bind any explicit rope-end arrows."""
        if img_route is None:
            return ()

        if route_cfg is None:
            current_cfg = getattr(self, "cfg", None) or {}
            route_cfg = current_cfg.get("route", {})
        target_color = tuple(map(
            int,
            route_cfg.get("rope_climb_target_color", (0, 191, 255)),
        ))
        arrow_bind_distance = max(
            0,
            int(route_cfg.get("rope_climb_arrow_bind_distance", 3)),
        )
        target_ys, target_xs = np.where(
            np.all(img_route == target_color, axis=2)
        )
        arrow_tips = tuple(
            (int(px), int(py))
            for px, py in zip(target_xs, target_ys)
        )

        targets = []
        for color, command in self.color_code.items():
            if not self._is_rope_climb_route_command(command):
                continue

            mask = np.all(img_route == color, axis=2).astype(np.uint8)
            component_count, labels, stats, _ = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)
            for component_idx in range(1, component_count):
                x, y, width, height, area = map(
                    int, stats[component_idx]
                )
                if area < 2:
                    continue

                ys, xs = np.where(labels == component_idx)
                points = tuple(
                    (int(px), int(py)) for px, py in zip(xs, ys)
                )

                # A two-sweep diameter gives stable endpoints for a straight
                # or gently bent hand-drawn guide without assuming it is
                # horizontal. Runtime picks the endpoint farther from Hero as
                # the rope contact point.
                seed = min(points, key=lambda point: (point[1], point[0]))

                def squared_distance(first, second):
                    return (
                        (first[0] - second[0]) ** 2
                        + (first[1] - second[1]) ** 2
                    )

                endpoint_a = max(
                    points,
                    key=lambda point: (
                        squared_distance(seed, point), point[1], point[0]
                    ),
                )
                endpoint_b = max(
                    points,
                    key=lambda point: (
                        squared_distance(endpoint_a, point),
                        point[1],
                        point[0],
                    ),
                )
                endpoints = tuple(sorted((endpoint_a, endpoint_b)))
                nearby_arrow_tips = []
                for arrow_tip in arrow_tips:
                    distance = min(
                        abs(arrow_tip[0] - point[0])
                        + abs(arrow_tip[1] - point[1])
                        for point in points
                    )
                    if distance <= arrow_bind_distance:
                        nearby_arrow_tips.append((distance, arrow_tip))

                rope_target = None
                if nearby_arrow_tips:
                    # The arrow tip is a single exact-color pixel. Reject an
                    # ambiguous tie instead of silently choosing an endpoint.
                    nearby_arrow_tips.sort(
                        key=lambda item: (item[0], item[1][1], item[1][0])
                    )
                    best_distance = nearby_arrow_tips[0][0]
                    best_tips = [
                        tip for distance, tip in nearby_arrow_tips
                        if distance == best_distance
                    ]
                    if len(best_tips) == 1:
                        rope_target = best_tips[0]
                    else:
                        logger.warning(
                            "[route] Ambiguous explicit rope arrow near "
                            f"guide bbox={(x, y, width, height)}; retain "
                            "legacy endpoint inference"
                        )
                targets.append({
                    "color": tuple(color),
                    "command": command,
                    "pixels": frozenset(points),
                    "endpoints": endpoints,
                    "bbox": (x, y, width, height),
                    "rope_target": rope_target,
                })

        return tuple(sorted(
            targets,
            key=lambda target: (
                target["bbox"][1], target["bbox"][0]
            ),
        ))

    def _get_active_rope_climb_targets(self):
        """Return cached guide components for the active route."""
        route_idx = int(getattr(self, "idx_routes", 0))
        routes = getattr(self, "img_routes", ())
        cached = getattr(self, "_rope_climb_targets_by_route", ())
        if 0 <= route_idx < len(routes) and \
                0 <= route_idx < len(cached) and \
                self.img_route is routes[route_idx]:
            return cached[route_idx]
        return self._find_rope_climb_targets(self.img_route)

    def _get_active_wz_rope_mount_plan(self, contact):
        """Return trajectory metadata paired with the active WZ route leg."""
        navigator = getattr(self, "wz_navigation", None)
        if navigator is None or not getattr(navigator, "active", False):
            return None
        route_idx = int(getattr(self, "idx_routes", 0))
        route_legs = getattr(navigator, "route_legs", ())
        if not 0 <= route_idx < len(route_legs):
            return None
        plan = getattr(route_legs[route_idx], "rope_mount", None)
        if plan is None or tuple(plan.contact) != tuple(contact):
            return None
        return plan

    def _get_active_wz_rope_mount_calibration(self):
        """Return a validated manual launch window for the active WZ map."""
        navigator = getattr(self, "wz_navigation", None)
        if navigator is None or not getattr(navigator, "active", False):
            return None
        calibration = self.cfg.get("wz_navigation", {}).get(
            "rope_mount_calibration"
        )
        if not isinstance(calibration, dict) or \
                str(calibration.get("map_id", "")) != str(
                    getattr(navigator, "map_id", "")
                ):
            return None
        try:
            launch_window = calibration["launch_window_px"]
            if not isinstance(launch_window, (list, tuple)) or \
                    len(launch_window) != 2:
                raise ValueError("launch_window_px must contain two values")
            window_min, window_max = map(int, launch_window)
            staging_offset = int(calibration["staging_offset_px"])
            approach_direction = str(
                calibration.get("approach_direction", "right")
            ).strip().lower()
            if window_min <= 0 or window_max < window_min:
                raise ValueError("launch window must be positive and ordered")
            if staging_offset <= window_max:
                raise ValueError(
                    "staging_offset_px must be beyond the launch window"
                )
            if approach_direction not in {"left", "right"}:
                raise ValueError(
                    "approach_direction must be left or right"
                )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                f"[route] Ignore invalid WZ rope calibration: {exc}"
            )
            return None
        return {
            "approach_direction": approach_direction,
            "launch_window_px": (window_min, window_max),
            "staging_offset_px": staging_offset,
        }

    def _get_active_wz_rope_destination_y(self, contact):
        """Return the upper-platform y paired with the active WZ climb."""
        navigator = getattr(self, "wz_navigation", None)
        if navigator is None or not getattr(navigator, "active", False):
            return None
        route_idx = int(getattr(self, "idx_routes", 0))
        route_legs = getattr(navigator, "route_legs", ())
        if not 0 <= route_idx < len(route_legs):
            return None
        route_leg = route_legs[route_idx]
        plan = getattr(route_leg, "rope_mount", None)
        target = getattr(route_leg, "target", None)
        if plan is None or tuple(plan.contact) != tuple(contact) or \
                target is None or len(target) < 2:
            return None
        return int(target[1])

    @staticmethod
    def _rope_climb_component_key(route_idx, target):
        return (
            int(route_idx),
            tuple(target["bbox"]),
            tuple(target["endpoints"]),
        )

    def _get_nearby_rope_climb_target(self):
        """Acquire a guide and predict its far endpoint as the rope x."""
        player = tuple(map(int, self.loc_player_global))
        route_cfg = self.cfg.get("route", {})
        detection_range = max(
            0,
            int(route_cfg.get(
                "rope_climb_detection_range",
                route_cfg.get("search_range", 10),
            )),
        )
        completed_key = getattr(self, "_rope_climb_completed_key", None)
        failed_key = getattr(self, "_rope_climb_failed_key", None)
        failed_at = getattr(self, "_rope_climb_failed_at", None)
        failure_rearm_delay = max(
            0.0,
            float(route_cfg.get("rope_climb_failure_rearm_delay", 1.5)),
        )
        failed_rearm_ready = bool(
            failed_key is not None
            and failed_at is not None
            and time.monotonic() - float(failed_at) >= failure_rearm_delay
        )
        blocked_failed_key = None if failed_rearm_ready else failed_key

        candidates = []
        for target in self._get_active_rope_climb_targets():
            key = self._rope_climb_component_key(self.idx_routes, target)
            if key in {completed_key, blocked_failed_key}:
                continue
            nearest_distance = min(
                abs(point[0] - player[0]) + abs(point[1] - player[1])
                for point in target["pixels"]
            )
            if nearest_distance > detection_range:
                continue
            endpoint_distances = [
                (
                    abs(endpoint[0] - player[0])
                    + abs(endpoint[1] - player[1]),
                    endpoint,
                )
                for endpoint in target["endpoints"]
            ]
            rope_endpoint = target.get("rope_target")
            explicit_target = rope_endpoint is not None
            if rope_endpoint is None:
                _, rope_endpoint = max(
                    endpoint_distances,
                    key=lambda item: (item[0], item[1][1], item[1][0]),
                )
            candidate = dict(target)
            candidate["key"] = key
            candidate["center"] = tuple(rope_endpoint)
            candidate["explicit_target"] = explicit_target
            candidate["rope_mount_plan"] = (
                self._get_active_wz_rope_mount_plan(rope_endpoint)
            )
            candidate["rope_destination_y"] = (
                self._get_active_wz_rope_destination_y(rope_endpoint)
            )
            candidates.append((
                nearest_distance,
                -max(distance for distance, _ in endpoint_distances),
                candidate,
            ))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:2])[-1]

    def _rope_timed_mount_status(self, state=None):
        """Return the keyboard timer state for the active rope attempt."""
        state = state or getattr(self, "_rope_climb_state", None)
        if not isinstance(state, dict):
            return None
        token = state.get("timed_mount_token")
        reader = getattr(
            getattr(self, "kb", None), "scheduled_rope_mount_status", None
        )
        if token is None or not callable(reader):
            return None
        return reader(token)

    def _cancel_rope_timed_mount(self, state=None):
        """Cancel a pending rope timer and discard its frame candidate."""
        self._rope_timed_mount_candidate = None
        state = state or getattr(self, "_rope_climb_state", None)
        if not isinstance(state, dict):
            return False
        token = state.get("timed_mount_token")
        cancel = getattr(
            getattr(self, "kb", None), "cancel_scheduled_rope_mount", None
        )
        cancelled = bool(
            token is not None and callable(cancel) and cancel(token)
        )
        status = self._rope_timed_mount_status(state)
        non_interruptible = bool(
            isinstance(status, dict)
            and status.get("state") in {"firing", "fired"}
        )
        if not non_interruptible:
            state["timed_mount_token"] = None
        return cancelled

    def _rope_timed_mount_owns_input(self, now=None):
        """Return whether a firing/recent rope timer is no longer interruptible."""
        status = self._rope_timed_mount_status()
        if not isinstance(status, dict):
            return False
        state = status.get("state")
        if state in {"pending", "firing"}:
            return True
        if state != "fired":
            return False
        now = time.monotonic() if now is None else float(now)
        finished_at = self._positive_finite_number(status.get("finished_at"))
        timeout = max(
            0.05,
            float(self.cfg.get("route", {}).get(
                "timed_jump_confirmation_timeout_ms", 450
            )) / 1000.0,
        )
        return finished_at > 0 and now - finished_at <= timeout

    def _reset_rope_climb(self, clear_locks=False):
        """Release the climb state while optionally rearming all guides."""
        self._cancel_rope_timed_mount()
        self._rope_climb_state = None
        self._rope_climb_active = False
        self._rope_climb_combat_deferred = False
        self._rope_climb_combat_deferred_at = None
        if clear_locks:
            self._rope_climb_completed_key = None
            self._rope_climb_failed_key = None
            self._rope_climb_completed_position = None
            self._rope_climb_failed_position = None
            self._rope_climb_failed_at = None

    def _set_rope_climb_combat_deferred(self, deferred):
        """Pause/resume a pre-mount climb while an attack target owns input."""
        state = getattr(self, "_rope_climb_state", None)
        if state is None:
            self._rope_climb_combat_deferred = False
            self._rope_climb_combat_deferred_at = None
            return

        now = time.monotonic()
        was_deferred = bool(getattr(
            self, "_rope_climb_combat_deferred", False
        ))
        if deferred:
            self._rope_timed_mount_candidate = None
            timed_status = self._rope_timed_mount_status(state)
            if isinstance(timed_status, dict) and \
                    timed_status.get("state") in {"firing", "fired"}:
                # Combat sampled the timer as pending, but its callback won
                # the race before cancellation. The mount now owns input and
                # must be allowed to reach the ascent state.
                return
            if isinstance(timed_status, dict) and \
                    timed_status.get("state") == "pending":
                if not self._cancel_rope_timed_mount(state):
                    timed_status = self._rope_timed_mount_status(state)
                    if isinstance(timed_status, dict) and \
                            timed_status.get("state") in {
                                "pending", "firing", "fired"
                            }:
                        return
                same_side = state.get("side")
                self._set_rope_runup_side(state, side=same_side)
                state["phase"] = "position"
                state["aligned_since"] = None
                state["mount_request_started_at"] = None
            if not was_deferred:
                self._rope_climb_combat_deferred_at = now
            self._rope_climb_combat_deferred = True
            return
        if not was_deferred:
            return

        paused_at = getattr(self, "_rope_climb_combat_deferred_at", None)
        paused_for = max(0.0, now - paused_at) \
            if paused_at is not None else 0.0
        # All rope timeouts use monotonic timestamps. Shift them together so
        # time spent killing a nearby monster cannot age or fail the climb.
        for key in (
                "started_at", "aligned_since", "mount_request_started_at",
                "last_progress_at", "last_y_change_at",
                "position_last_progress_at", "position_brake_started_at",
                "destination_reached_at"):
            value = state.get(key)
            if value is not None:
                state[key] = float(value) + paused_for
        state["approach_last_x"] = int(self.loc_player_global[0])
        self._rope_climb_combat_deferred = False
        self._rope_climb_combat_deferred_at = None

    def _clear_rope_climb_locks_if_departed(self):
        """Rearm a guide after Hero leaves its completion/failure area."""
        if getattr(self, "_rope_climb_completed_key", None) is None and \
                getattr(self, "_rope_climb_failed_key", None) is None:
            return
        player_x, player_y = map(int, self.loc_player_global)
        route_idx = int(self.idx_routes)
        route_cfg = self.cfg.get("route", {})
        detection_range = max(
            1,
            int(route_cfg.get(
                "rope_climb_detection_range",
                route_cfg.get("search_range", 10),
            )),
        )
        runup_distance = max(
            0, int(route_cfg.get("rope_climb_runup_distance", 8))
        )
        depart_distance = detection_range + runup_distance

        for key_attr, position_attr in (
            ("_rope_climb_completed_key", "_rope_climb_completed_position"),
            ("_rope_climb_failed_key", "_rope_climb_failed_position"),
        ):
            key = getattr(self, key_attr, None)
            position = getattr(self, position_attr, None)
            if key is None:
                continue
            departed = key[0] != route_idx
            if position is not None:
                departed = departed or (
                    abs(player_x - position[0]) > depart_distance
                    or abs(player_y - position[1]) > depart_distance
                )
            if departed:
                setattr(self, key_attr, None)
                setattr(self, position_attr, None)
                if key_attr == "_rope_climb_failed_key":
                    self._rope_climb_failed_at = None

    def _choose_rope_runup_side(self, target_x):
        """Choose which side to back up to before running at the rope."""
        player_x = int(self.loc_player_global[0])
        tolerance = max(
            0,
            int(self.cfg.get("route", {}).get(
                "rope_climb_align_tolerance", 1
            )),
        )
        if player_x < target_x - tolerance:
            side = "left"
        elif player_x > target_x + tolerance:
            side = "right"
        else:
            cached_facing = getattr(
                getattr(self, "kb", None), "cached_facing", None
            )
            if cached_facing == "right":
                side = "left"
            elif cached_facing == "left":
                side = "right"
            else:
                side = "left"
        return side

    def _set_rope_runup_side(self, state, side=None):
        target_x = int(state["target"][0])
        calibration = state.get("rope_mount_calibration")
        recorded_approach = state.get("rope_mount_approach_direction")
        if side is None and recorded_approach in {"left", "right"}:
            side = "left" if recorded_approach == "right" else "right"
        elif side is None and calibration is not None:
            side = (
                "left"
                if calibration.get("approach_direction") == "right"
                else "right"
            )
        side = side or self._choose_rope_runup_side(target_x)
        fallback_distance = max(
            0,
            int(self.cfg.get("route", {}).get(
                "rope_climb_runup_distance", 8
            )),
        )
        configured_staging = state.get("staging_offset_px")
        if configured_staging is None:
            configured_staging = fallback_distance
        base_staging_distance = max(0, int(configured_staging))
        def staging_for(runup_side):
            approach_direction = (
                "right" if runup_side == "left" else "left"
            )
            applies = bool(
                calibration is not None
                and calibration.get("approach_direction")
                == approach_direction
            )
            distance = base_staging_distance
            if applies:
                distance = max(
                    distance,
                    int(calibration["staging_offset_px"]),
                )
            return distance, applies

        staging_distance, calibration_applies = staging_for(side)
        configured_launch = state.get("launch_offset_px")
        if configured_launch is None:
            configured_launch = staging_distance
        launch_distance = max(0, int(configured_launch))
        staging_distance = max(staging_distance, launch_distance)
        width = self.img_route.shape[1] if self.img_route is not None else 0
        if side == "left" and target_x - staging_distance < 0:
            side = "right"
            staging_distance, calibration_applies = staging_for(side)
        elif width > 0 and side == "right" and \
                target_x + staging_distance >= width:
            side = "left"
            staging_distance, calibration_applies = staging_for(side)
        staging_distance = max(staging_distance, launch_distance)
        offset_sign = -1 if side == "left" else 1
        state["side"] = side
        state["start_x"] = target_x + offset_sign * staging_distance
        state["launch_x"] = target_x + offset_sign * launch_distance
        state["active_launch_window_px"] = (
            tuple(calibration["launch_window_px"])
            if calibration_applies else None
        )
        state["position_best_distance"] = None
        state["position_last_progress_at"] = time.monotonic()
        state["position_continuous"] = False
        state["position_brake_started_at"] = None
        state["position_last_dx"] = None
        state["position_stall_retries"] = 0
        # The continuous-approach path predicts one capture interval of
        # horizontal travel. Reset the sample whenever a retry changes sides;
        # a displacement made while repositioning is not launch velocity.
        state["approach_last_x"] = int(self.loc_player_global[0])

    @staticmethod
    def _has_calibrated_rope_runway(state, player_x):
        """Return whether the measured approach has room before its window."""
        launch_window = state.get("active_launch_window_px")
        if launch_window is None:
            return False
        target_x = int(state["target"][0])
        if state.get("side") == "left":
            remaining = target_x - int(player_x)
        else:
            remaining = int(player_x) - target_x
        return remaining > int(launch_window[1])

    def _start_rope_climb(self, route_action):
        """Create a climb state from one predicted rope guide endpoint."""
        target = tuple(map(int, route_action["target_center"]))
        key = route_action["guide_key"]
        mount_plan = route_action.get("rope_mount_plan")
        if key == getattr(self, "_rope_climb_failed_key", None):
            logger.info(
                "[route] Rearm nearby climb guide after failure cooldown"
            )
            self._rope_climb_failed_key = None
            self._rope_climb_failed_position = None
            self._rope_climb_failed_at = None
        state = getattr(self, "_rope_climb_state", None)
        if state is None or state.get("key") != key:
            now = time.monotonic()
            player_y = int(self.loc_player_global[1])
            state = {
                "key": key,
                "target": target,
                "phase": "position",
                "started_at": now,
                "aligned_since": None,
                "mount_request_started_at": None,
                "mount_kind": None,
                "timed_mount_token": None,
                "mount_origin_y": player_y,
                "best_y": player_y,
                "last_progress_at": now,
                "last_y": player_y,
                "last_y_change_at": now,
                "attempts": 0,
                "side": None,
                "start_x": None,
                "launch_x": None,
                "trajectory_mount": mount_plan is not None,
                "destination_y": route_action.get("rope_destination_y"),
                "destination_reached_at": None,
                "launch_offset_px": (
                    int(mount_plan.launch_offset_px)
                    if mount_plan is not None else None
                ),
                "staging_offset_px": (
                    int(mount_plan.staging_offset_px)
                    if mount_plan is not None else None
                ),
                "rope_mount_calibration": (
                    self._get_active_wz_rope_mount_calibration()
                    if mount_plan is not None else None
                ),
                "active_launch_window_px": None,
                "vertical_gap_px": (
                    float(mount_plan.vertical_gap_px)
                    if mount_plan is not None else None
                ),
                "jump_height_px": (
                    float(mount_plan.jump_height_px)
                    if mount_plan is not None else None
                ),
                "jump_distance_px": (
                    float(mount_plan.jump_distance_px)
                    if mount_plan is not None else None
                ),
                "launch_lead_px": (
                    int(mount_plan.launch_lead_px)
                    if mount_plan is not None else None
                ),
                "rope_mount_approach_direction": (
                    getattr(mount_plan, "approach_direction", None)
                    if mount_plan is not None else None
                ),
                "predicted_contact_height_px": (
                    float(mount_plan.predicted_contact_height_px)
                    if mount_plan is not None else None
                ),
                "contact_clearance_px": (
                    float(mount_plan.contact_clearance_px)
                    if mount_plan is not None else None
                ),
                "reachable_at_contact": (
                    bool(mount_plan.reachable_at_contact)
                    if mount_plan is not None else None
                ),
                "observed_ladder": bool(
                    getattr(self, "is_on_ladder", False)
                ),
                "position_side_switches": 0,
            }
            self._set_rope_runup_side(state)
            if state["observed_ladder"]:
                state["phase"] = "climbing"
                state["attempts"] = 1
            else:
                player_x = int(self.loc_player_global[0])
                target_x = int(target[0])
                tolerance = max(
                    0,
                    int(self.cfg.get("route", {}).get(
                        "rope_climb_align_tolerance", 1
                    )),
                )
                approach_side = None
                approach_direction = None
                if player_x < target_x - tolerance:
                    approach_side = "left"
                    approach_direction = "right"
                elif player_x > target_x + tolerance:
                    approach_side = "right"
                    approach_direction = "left"

                if approach_side is None:
                    # Live trials showed that an exact-x Up-before-Jump mount
                    # is fully repeatable and avoids an unnecessary retreat.
                    # Recheck the alignment after the normal settle interval
                    # before committing the one-shot action.
                    state["phase"] = "stationary_settle"
                    state["aligned_since"] = now

                held_direction = getattr(
                    getattr(self, "kb", None),
                    "cmd_left_right_last",
                    None,
                )
                calibration = state.get("rope_mount_calibration")
                calibrated_direction = (
                    calibration.get("approach_direction")
                    if calibration is not None else None
                )
                if approach_side is not None and \
                        held_direction == approach_direction and (
                        calibrated_direction is None
                        or approach_direction == calibrated_direction
                ):
                    self._set_rope_runup_side(
                        state, side=approach_side
                    )
                    # Boundary fallback may have moved the generated runway
                    # to the opposite side. Only use the seamless path when
                    # the existing run still points from that runway at rope.
                    calibrated_runway_ready = (
                        state.get("active_launch_window_px") is None
                        or self._has_calibrated_rope_runway(
                            state, player_x
                        )
                    )
                    if state["side"] == approach_side and \
                            calibrated_runway_ready:
                        state["phase"] = "running_approach"
            self._rope_climb_state = state
            if state["phase"] == "stationary_settle":
                logger.info(
                    "[route] Acquired climb guide at exact rope x; "
                    f"try stationary Up + Jump at {target} before using "
                    "the generated runway"
                )
            elif mount_plan is not None:
                log_message = (
                    "[route] Acquired trajectory climb guide; "
                    f"rope_bottom={target}, "
                    f"upper_platform_y={state['destination_y']}, "
                    f"gap={state['vertical_gap_px']:.2f}px, "
                    f"jump_height={state['jump_height_px']:.2f}px, "
                    f"jump_distance={state['jump_distance_px']:.2f}px, "
                    f"lead={state['launch_lead_px']}px, "
                    f"contact_height="
                    f"{state['predicted_contact_height_px']:.2f}px, "
                    f"{state['side']} staging_x={state['start_x']}, "
                    f"launch_x={state['launch_x']}"
                )
                if state.get("rope_mount_approach_direction") is not None:
                    log_message += (
                        ", recorded_approach="
                        f"{state['rope_mount_approach_direction']}"
                    )
                if state.get("active_launch_window_px") is not None:
                    window_min, window_max = \
                        state["active_launch_window_px"]
                    log_message += (
                        f", calibrated_window={window_min}..{window_max}px"
                    )
                if state["reachable_at_contact"]:
                    logger.info(log_message)
                else:
                    logger.warning(
                        log_message
                        + "; predicted rope contact lacks vertical clearance"
                    )
            elif state["phase"] == "running_approach":
                logger.info(
                    "[route] Acquired climb guide while already running "
                    f"toward {target}; keep moving without a stop and jump "
                    f"at generated runway x={state['start_x']}"
                )
            else:
                logger.info(
                    "[route] Acquired climb guide; predicted rope contact at "
                    f"{target} and generated {state['side']} runway "
                    f"x={state['start_x']}"
                )

        self._rope_climb_active = True
        return self._update_active_rope_climb()

    def _finish_rope_climb(self, evidence="coordinate landing"):
        state = getattr(self, "_rope_climb_state", None)
        if state is None:
            return
        key = state["key"]
        position = tuple(map(int, self.loc_player_global))
        logger.info(
            f"[route] Rope climb finished after {state['attempts']} "
            f"attempt(s); evidence={evidence}"
        )
        self._reset_rope_climb()
        self._rope_climb_completed_key = key
        self._rope_climb_completed_position = position

    def _fail_rope_climb(self, reason):
        state = getattr(self, "_rope_climb_state", None)
        if state is None:
            return
        key = state["key"]
        position = tuple(map(int, self.loc_player_global))
        logger.warning(
            f"[route] Rope climb {reason} near {state['target']}; "
            "pause before retrying this guide"
        )
        self._reset_rope_climb()
        self._rope_climb_failed_key = key
        self._rope_climb_failed_position = position
        self._rope_climb_failed_at = time.monotonic()

    def _prepare_rope_climb_retry(self, state):
        self._cancel_rope_timed_mount(state)
        previous_side = state.get("side")
        calibration = state.get("rope_mount_calibration")
        calibrated_direction = (
            calibration.get("approach_direction")
            if calibration is not None else None
        )
        if calibrated_direction in {"left", "right"}:
            next_side = (
                "left" if calibrated_direction == "right" else "right"
            )
        else:
            next_side = "right" if previous_side == "left" else "left"
        self._set_rope_runup_side(state, side=next_side)
        state["phase"] = "position"
        state["aligned_since"] = None
        state["mount_request_started_at"] = None
        state["mount_kind"] = None
        state["destination_reached_at"] = None
        state["position_side_switches"] = 0
        logger.info(
            "[route] Rope mount made no upward progress; retry from "
            f"the {state['side']} side at staging_x={state['start_x']}, "
            f"launch_x={state['launch_x']}"
        )

    def _request_rope_mount(self, state, now, player_y):
        """Publish one directional mount request at the launch point."""
        side = state["side"]
        direction = "right" if side == "left" else "left"
        state["phase"] = "mount_request"
        state["mount_kind"] = "directional"
        state["attempts"] += 1
        state["mount_request_started_at"] = now
        state["mount_origin_y"] = player_y
        state["best_y"] = player_y
        state["last_progress_at"] = now
        state["last_y"] = player_y
        state["last_y_change_at"] = now
        self.cmd_move_x = direction
        self.cmd_move_y = "up"
        self.cmd_action = f"rope_mount_{direction}"
        return True

    def _request_stationary_rope_mount(self, state, now, player_y):
        """Publish one exact-x Up-before-Jump rope mount request."""
        state["phase"] = "mount_request"
        state["mount_kind"] = "stationary"
        state["attempts"] += 1
        state["mount_request_started_at"] = now
        state["mount_origin_y"] = player_y
        state["best_y"] = player_y
        state["last_progress_at"] = now
        state["last_y"] = player_y
        state["last_y_change_at"] = now
        self.cmd_move_x = "none"
        self.cmd_move_y = "up"
        self.cmd_action = "rope_mount_stationary"
        return True

    def _rope_timed_mount_descriptor(self, state):
        """Describe a generated rope launch for the shared WZ predictor."""
        scheduler = getattr(
            getattr(self, "kb", None), "schedule_rope_mount", None
        )
        status_reader = getattr(
            getattr(self, "kb", None), "scheduled_rope_mount_status", None
        )
        if not isinstance(state, dict) or \
                state.get("phase") != "running_approach" or \
                not state.get("trajectory_mount", False) or \
                not callable(scheduler) or not callable(status_reader):
            return None
        side = state.get("side")
        if side not in {"left", "right"}:
            return None
        direction = "right" if side == "left" else "left"
        launch_x = float(state.get("launch_x", state.get("start_x")))
        player_x, player_y = map(float, self.loc_player_global[:2])
        token = (
            state.get("key"),
            side,
            int(state.get("attempts", 0)) + 1,
            int(round(launch_x)),
        )
        return {
            "token": token,
            "state_key": state.get("key"),
            "side": side,
            "direction": direction,
            "player_x": player_x,
            "player_y": player_y,
            "launch_x": launch_x,
        }

    def _prepare_rope_timed_mount(self, state):
        """Reserve a rope timer candidate before monster arbitration."""
        self._rope_timed_mount_candidate = None
        descriptor = self._rope_timed_mount_descriptor(state)
        if descriptor is None:
            return False
        prediction = self._predict_current_wz_timed_jump(descriptor)
        if prediction is None:
            return False
        descriptor["prediction"] = prediction
        self._rope_timed_mount_candidate = descriptor
        self.cmd_move_x = descriptor["direction"]
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        return True

    def finalize_rope_timed_mount(self):
        """Arm the rope timer after combat kept the approach command."""
        candidate = getattr(self, "_rope_timed_mount_candidate", None)
        self._rope_timed_mount_candidate = None
        if not isinstance(candidate, dict):
            return False
        state = getattr(self, "_rope_climb_state", None)
        direction = candidate["direction"]
        if not isinstance(state, dict) or \
                state.get("phase") != "running_approach" or \
                state.get("key") != candidate["state_key"] or \
                state.get("side") != candidate["side"] or (
                    self.cmd_move_x != direction
                    or self.cmd_move_y not in {"none", "stop"}
                    or self.cmd_action not in {"none", "stop"}
                ):
            return False

        prediction = self._predict_current_wz_timed_jump(candidate)
        scheduler = getattr(
            getattr(self, "kb", None), "schedule_rope_mount", None
        )
        if prediction is None or not callable(scheduler) or not scheduler(
                direction, prediction.delay_seconds, candidate["token"]):
            return False

        now = time.monotonic()
        state["phase"] = "timed_mount"
        state["mount_kind"] = "directional"
        state["timed_mount_token"] = candidate["token"]
        state["attempts"] += 1
        state["mount_request_started_at"] = None
        state["mount_origin_y"] = int(round(candidate["player_y"]))
        state["best_y"] = state["mount_origin_y"]
        state["last_progress_at"] = now
        state["last_y"] = state["mount_origin_y"]
        state["last_y_change_at"] = now
        self.cmd_move_x = direction
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        logger.info(
            "[timed-rope] Armed "
            f"{direction}: sample={prediction.sample_x:.2f}px, "
            f"predicted={prediction.predicted_x:.2f}px, "
            f"launch={prediction.launch_x:.2f}px, "
            f"speed={prediction.speed_px_per_second:.3f}px/s, "
            f"frameAge={prediction.capture_age_seconds * 1000:.0f}ms, "
            f"lead={prediction.input_lead_seconds * 1000:.0f}ms, "
            f"delay={prediction.delay_seconds * 1000:.0f}ms"
        )
        return True

    def _update_active_rope_climb(self):
        """Drive retreat, directional mount, ascent, and retry feedback."""
        state = getattr(self, "_rope_climb_state", None)
        if state is None:
            self._rope_climb_active = False
            return False
        kb = getattr(self, "kb", None)
        if kb is not None and (
                getattr(kb, "is_enable", True) is False
                or getattr(kb, "capture_available", True) is False
        ):
            # F1 pause and capture suspension release all physical keys. Drop
            # the visual transaction too, so wall-clock timeout cannot age it
            # and resume cannot emit a delayed mount from stale coordinates.
            self._reset_rope_climb()
            return False
        if int(self.idx_routes) != int(state["key"][0]):
            self._reset_rope_climb()
            return False

        self._rope_climb_active = True
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "rope_hold"

        now = time.monotonic()
        route_cfg = self.cfg.get("route", {})
        max_duration = max(
            0.1, float(route_cfg.get("rope_climb_max_duration", 15.0))
        )
        if now - state["started_at"] >= max_duration:
            self._fail_rope_climb("timed out")
            return True

        player_x, player_y = map(int, self.loc_player_global)
        phase = state["phase"]

        if phase == "timed_mount":
            direction = "right" if state["side"] == "left" else "left"
            status = self._rope_timed_mount_status(state)
            timer_state = (
                status.get("state") if isinstance(status, dict) else None
            )
            if timer_state in {"pending", "firing"}:
                self.cmd_move_x = direction
                self.cmd_move_y = "none"
                self.cmd_action = "none"
                return True
            if timer_state == "fired":
                state["phase"] = "mounting"
                state["mount_request_started_at"] = status.get(
                    "fired_at", status.get("finished_at", now)
                )
                state["last_progress_at"] = now
                state["last_y_change_at"] = now
                self.cmd_move_y = "up"
                self.cmd_action = "rope_hold"
                phase = "mounting"
            else:
                max_attempts = max(
                    1, int(route_cfg.get("rope_climb_max_attempts", 4))
                )
                if state["attempts"] >= max_attempts:
                    self._fail_rope_climb(
                        "exhausted its timed mount retries"
                    )
                    return True
                self._prepare_rope_climb_retry(state)
                return True

        if phase == "mount_request":
            stationary = state.get("mount_kind") == "stationary"
            if stationary:
                direction = "none"
                request_hold = max(
                    0.20,
                    float(route_cfg.get(
                        "rope_climb_stationary_up_lead_ms", 35
                    )) / 1000.0 + 0.05,
                )
            else:
                side = state["side"]
                direction = "right" if side == "left" else "left"
                request_hold = max(
                    0.20,
                    float(route_cfg.get("rope_climb_runup_ms", 180))
                    / 1000.0 + 0.05,
                )
            if now - state["mount_request_started_at"] < request_hold:
                self.cmd_move_x = direction
                self.cmd_move_y = "up"
                self.cmd_action = (
                    "rope_mount_stationary"
                    if stationary else f"rope_mount_{direction}"
                )
                return True
            state["phase"] = "mounting"
            phase = "mounting"

        if phase in {"mounting", "climbing"}:
            ladder_now = bool(getattr(self, "is_on_ladder", False))
            state["observed_ladder"] = (
                state["observed_ladder"] or ladder_now
            )
            if player_y < state["best_y"]:
                state["best_y"] = player_y
                state["last_progress_at"] = now
            last_y = state.get("last_y", player_y)
            if player_y != last_y:
                state["last_y"] = player_y
                state["last_y_change_at"] = now
            else:
                state.setdefault(
                    "last_y_change_at",
                    state.get("last_progress_at", now),
                )

            min_progress = max(
                1, int(route_cfg.get("rope_climb_min_progress", 3))
            )
            upward_progress = state["mount_origin_y"] - state["best_y"]
            current_elevation = state["mount_origin_y"] - player_y
            if upward_progress >= min_progress or ladder_now:
                state["phase"] = "climbing"
                phase = "climbing"

            no_progress_for = now - state["last_progress_at"]
            stationary_for = now - state["last_y_change_at"]
            finish_stall = max(
                0.2,
                float(route_cfg.get("rope_climb_finish_stall", 0.6)),
            )
            landing_tolerance = max(
                0,
                int(route_cfg.get("rope_climb_landing_tolerance", 2)),
            )
            attach_x_tolerance = max(
                0,
                int(route_cfg.get(
                    "rope_climb_attach_x_tolerance", 3
                )),
            )
            near_rope_x = (
                abs(player_x - int(state["target"][0]))
                <= attach_x_tolerance
            )
            destination_y = state.get("destination_y")
            destination_tolerance = max(
                0,
                int(route_cfg.get("ladder_endpoint_tolerance", 2)),
            )
            reached_destination = (
                destination_y is None
                or player_y <= int(destination_y) + destination_tolerance
            )
            if destination_y is not None:
                if reached_destination:
                    if state.get("destination_reached_at") is None:
                        state["destination_reached_at"] = now
                else:
                    state["destination_reached_at"] = None
            destination_reached_at = state.get("destination_reached_at")
            destination_dwell = (
                now - float(destination_reached_at)
                if destination_reached_at is not None else 0.0
            )
            visual_exit_at_destination = bool(
                destination_y is not None
                and state["observed_ladder"]
                and not ladder_now
                and reached_destination
            )
            coordinate_landing = bool(
                near_rope_x
                and reached_destination
                and (
                    destination_dwell >= finish_stall
                    if destination_y is not None
                    else stationary_for >= finish_stall
                )
            )
            if phase == "climbing" and not ladder_now and \
                    upward_progress >= min_progress and \
                    current_elevation > landing_tolerance and \
                    (visual_exit_at_destination or coordinate_landing):
                evidence = (
                    "visual ladder exit at WZ destination"
                    if visual_exit_at_destination
                    else "stable WZ destination coordinates"
                )
                self._finish_rope_climb(evidence)
                return True

            retry_interval = max(
                finish_stall,
                float(route_cfg.get("rope_climb_retry_interval", 0.9)),
            )
            # A missed jump can land a few minimap pixels above its sampled
            # origin. That residual offset is not proof of a rope attachment;
            # otherwise the mount waits for the 15 s global timeout. Preserve
            # an elevated pose only after ladder detection has confirmed the
            # attachment. Unconfirmed attempts retry after their highest point
            # has stopped improving for the normal retry interval.
            confirmed_attachment = bool(state["observed_ladder"])
            if ladder_now or no_progress_for < retry_interval or \
                    (confirmed_attachment and
                     current_elevation > landing_tolerance):
                self.cmd_move_y = "up"
                return True

            max_attempts = max(
                1, int(route_cfg.get("rope_climb_max_attempts", 4))
            )
            if state["attempts"] >= max_attempts:
                self._fail_rope_climb("exhausted its mount retries")
                return True
            self._prepare_rope_climb_retry(state)
            return True

        if phase == "stationary_settle":
            tolerance = max(
                0, int(route_cfg.get("rope_climb_align_tolerance", 1))
            )
            target_x = int(state["target"][0])
            if abs(player_x - target_x) > tolerance:
                self._set_rope_runup_side(state)
                state["phase"] = "position"
                state["aligned_since"] = None
                phase = "position"
            else:
                settle_delay = max(
                    0.0,
                    float(route_cfg.get("rope_climb_settle_delay", 0.15)),
                )
                aligned_since = state.get("aligned_since")
                if aligned_since is None:
                    state["aligned_since"] = now
                    return True
                if now - float(aligned_since) < settle_delay:
                    return True
                return self._request_stationary_rope_mount(
                    state, now, player_y
                )

        if phase == "running_approach":
            direction = "right" if state["side"] == "left" else "left"
            launch_x = int(state.get("launch_x", state["start_x"]))
            launch_window = state.get("active_launch_window_px")
            tolerance = max(
                0, int(route_cfg.get("rope_climb_align_tolerance", 1))
            )
            previous_x = int(state.get("approach_last_x", player_x))
            observed_step = (
                player_x - previous_x
                if direction == "right"
                else previous_x - player_x
            )
            observed_step = max(0, observed_step)
            state["approach_last_x"] = player_x

            # WZ routes know both the measured horizontal speed and the exact
            # launch x. Arm a host timer while the point is still ahead so a
            # 5-10 FPS vision loop cannot notice the launch only after Hero
            # has crossed it. The candidate is committed after this frame's
            # monster arbitration, matching normal timed edge jumps.
            trajectory_mount = bool(state.get("trajectory_mount", False))
            timed_descriptor = (
                self._rope_timed_mount_descriptor(state)
                if trajectory_mount else None
            )
            timed_speed = self._wz_walk_speed_px_per_second()
            if timed_descriptor is not None and timed_speed > 0:
                held_seconds = 0.0
                held_reader = getattr(
                    getattr(self, "kb", None),
                    "same_direction_move_seconds",
                    None,
                )
                if callable(held_reader):
                    try:
                        held_seconds = max(0.0, float(held_reader(
                            direction, now=now
                        )))
                    except (TypeError, ValueError, OverflowError):
                        held_seconds = 0.0
                frame_age = self._wz_capture_frame_age_seconds(now)
                projected_travel = timed_speed * min(
                    frame_age, held_seconds
                )
                projected_remaining = remaining = (
                    launch_x - player_x
                    if direction == "right"
                    else player_x - launch_x
                )
                projected_remaining -= projected_travel
                if projected_remaining < -1.0:
                    same_side = state["side"]
                    self._set_rope_runup_side(state, side=same_side)
                    state["phase"] = "position"
                    state["aligned_since"] = None
                    state["position_side_switches"] = 0
                    logger.info(
                        "[timed-rope] Captured frame is already past "
                        f"launch_x={launch_x}: x={player_x}, "
                        f"predicted_remaining={projected_remaining:.2f}px; "
                        "restart the same-side runway"
                    )
                    return True

                self.cmd_move_x = direction
                if self._prepare_rope_timed_mount(state):
                    return True

                best_distance = state.get("position_best_distance")
                if best_distance is None or remaining < best_distance:
                    state["position_best_distance"] = remaining
                    state["position_last_progress_at"] = now
                position_timeout = max(
                    0.1,
                    float(route_cfg.get(
                        "rope_climb_position_timeout", 0.9
                    )),
                )
                if now - state["position_last_progress_at"] >= \
                        position_timeout:
                    state["phase"] = "position"
                    state["aligned_since"] = None
                    logger.info(
                        "[timed-rope] Approach stopped before timer arming; "
                        "fall back to runway positioning"
                    )
                return True

            # Predict one observed minimap sample so the USB action reaches
            # the game at the calculated launch point. WZ trajectories need
            # no arbitrary extra margin: any missing 180 ms momentum is added
            # precisely by KeyBoardController before it taps Jump.
            if trajectory_mount:
                prediction_margin = 0
                if launch_window is not None:
                    prediction_limit = max(1, int(launch_window[1]))
                else:
                    prediction_limit = max(
                        1,
                        abs(int(state["target"][0]) - launch_x),
                    )
            else:
                prediction_margin = max(
                    0,
                    int(route_cfg.get(
                        "rope_climb_running_prediction_margin", 4
                    )),
                )
                runup_distance = max(
                    0,
                    int(route_cfg.get("rope_climb_runup_distance", 8)),
                )
                prediction_limit = max(1, runup_distance * 2)
            prediction_lead = (
                min(
                    prediction_limit,
                    observed_step + prediction_margin,
                )
                if observed_step > 0
                else 0
            )
            remaining = (
                launch_x - player_x
                if direction == "right"
                else player_x - launch_x
            )
            if launch_window is not None:
                window_min, window_max = map(int, launch_window)
                target_x = int(state["target"][0])
                rope_remaining = (
                    target_x - player_x
                    if direction == "right"
                    else player_x - target_x
                )
                projected_remaining = rope_remaining - prediction_lead
                effective_remaining = (
                    projected_remaining
                    if prediction_lead > 0 else rope_remaining
                )
                reached_launch = (
                    window_min <= effective_remaining <= window_max
                )
                missed_window = effective_remaining < window_min
                if missed_window:
                    same_side = state["side"]
                    self._set_rope_runup_side(state, side=same_side)
                    state["phase"] = "position"
                    state["aligned_since"] = None
                    state["position_side_switches"] = 0
                    logger.info(
                        "[route] Calibrated rope launch window was crossed "
                        f"too late: remaining={rope_remaining}px, "
                        f"predicted={projected_remaining}px, "
                        f"window={window_min}..{window_max}px; "
                        "restart the same-side runway"
                    )
                    return True
            else:
                reached_launch = remaining <= tolerance + prediction_lead
            self.cmd_move_x = direction
            if reached_launch:
                if prediction_lead > 0:
                    if launch_window is not None:
                        logger.info(
                            "[route] Continuous rope approach entered "
                            "calibrated launch window: "
                            f"x={player_x}, remaining={rope_remaining}px, "
                            f"predicted={projected_remaining}px, "
                            f"window={window_min}..{window_max}px, "
                            f"observed_step={observed_step}"
                        )
                    else:
                        logger.info(
                            "[route] Continuous rope approach predicted "
                            f"launch crossing: x={player_x}, "
                            f"launch_x={launch_x}, "
                            f"observed_step={observed_step}, "
                            f"lead={prediction_lead}"
                        )
                return self._request_rope_mount(
                    state, now, player_y
                )

            best_distance = state.get("position_best_distance")
            if best_distance is None or remaining < best_distance:
                state["position_best_distance"] = remaining
                state["position_last_progress_at"] = now
            position_timeout = max(
                0.1,
                float(route_cfg.get("rope_climb_position_timeout", 0.9)),
            )
            if now - state["position_last_progress_at"] >= position_timeout:
                state["phase"] = "position"
                state["aligned_since"] = None
                logger.info(
                    "[route] Continuous rope approach stopped making "
                    "progress; fall back to runway positioning"
                )
            return True

        tolerance = max(
            0, int(route_cfg.get("rope_climb_align_tolerance", 1))
        )
        # A recorded trajectory defines a valid launch window, not one exact
        # staging pixel. Once Hero has backed beyond that window, the HID
        # transaction can build the remaining measured momentum while running
        # at the rope. This avoids oscillating or declaring the runway blocked
        # merely because x=77 was skipped by a coarse minimap sample.
        calibrated_runway_ready = self._has_calibrated_rope_runway(
            state, player_x
        )
        dx = (
            0
            if calibrated_runway_ready
            else int(state["start_x"]) - player_x
        )
        distance = abs(dx)
        hold_distance = max(
            tolerance + 1,
            int(route_cfg.get("rope_climb_align_hold_distance", 6)),
        )
        crossing_tolerance = max(
            tolerance,
            int(route_cfg.get(
                "rope_climb_align_crossing_tolerance", 3
            )),
        )
        previous_dx = state.get("position_last_dx")
        crossed_runway = bool(
            previous_dx is not None
            and previous_dx != 0
            and dx != 0
            and (previous_dx > 0) != (dx > 0)
            and distance <= crossing_tolerance
        )
        state["position_last_dx"] = dx
        if distance > tolerance and not crossed_runway:
            state["phase"] = "position"
            state["aligned_since"] = None
            best_distance = state.get("position_best_distance")
            if best_distance is None or distance < best_distance:
                state["position_best_distance"] = distance
                state["position_last_progress_at"] = now
                state["position_stall_retries"] = 0
            position_timeout = max(
                0.1,
                float(route_cfg.get("rope_climb_position_timeout", 0.9)),
            )
            if now - state["position_last_progress_at"] >= position_timeout:
                if state.get("active_launch_window_px") is not None:
                    stall_retries = int(
                        state.get("position_stall_retries", 0)
                    ) + 1
                    state["position_stall_retries"] = stall_retries
                    if stall_retries <= 2:
                        state["position_best_distance"] = distance
                        state["position_last_progress_at"] = now
                        state["position_continuous"] = False
                        state["position_brake_started_at"] = None
                        state["position_last_dx"] = None
                        direction = "right" if dx > 0 else "left"
                        self.cmd_action = f"rope_align_{direction}"
                        logger.info(
                            "[route] Recorded rope runway made no progress; "
                            f"retry {stall_retries}/2 from the same "
                            f"{state['side']} side"
                        )
                        return True
                state["position_side_switches"] = (
                    state.get("position_side_switches", 0) + 1
                )
                if state["position_side_switches"] >= 2:
                    self._fail_rope_climb(
                        "could not reach a runway on either side"
                    )
                    return True
                previous_side = state["side"]
                next_side = "right" if previous_side == "left" else "left"
                self._set_rope_runup_side(state, side=next_side)
                logger.info(
                    "[route] Runway position is blocked; switch from "
                    f"{previous_side} to {state['side']} side"
                )
                return True
            direction = "right" if dx > 0 else "left"
            if distance > hold_distance:
                # Cover most of a long runway repositioning with a held key.
                # Repeated 30 ms taps are reserved for the final few pixels.
                state["position_continuous"] = True
                state["position_brake_started_at"] = None
                self.cmd_move_x = direction
                return True

            if state.get("position_continuous", False):
                # Release continuous movement before trusting a fine correction
                # so the next minimap sample includes any residual travel.
                state["position_continuous"] = False
                state["position_brake_started_at"] = now
                return True

            brake_started_at = state.get("position_brake_started_at")
            brake_delay = max(
                0.0,
                float(route_cfg.get(
                    "rope_climb_align_brake_delay", 0.10
                )),
            )
            if brake_started_at is not None:
                if now - brake_started_at < brake_delay:
                    return True
                state["position_brake_started_at"] = None
            self.cmd_action = (
                "rope_align_right" if dx > 0 else "rope_align_left"
            )
            return True

        state["position_continuous"] = False
        state["position_brake_started_at"] = None
        if phase != "settle" or state["aligned_since"] is None:
            state["phase"] = "settle"
            state["aligned_since"] = now
            return True

        settle_delay = max(
            0.0, float(route_cfg.get("rope_climb_settle_delay", 0.15))
        )
        if now - state["aligned_since"] < settle_delay:
            return True

        if state.get("trajectory_mount", False):
            direction = "right" if state["side"] == "left" else "left"
            state["phase"] = "running_approach"
            state["aligned_since"] = None
            state["position_best_distance"] = None
            state["position_last_progress_at"] = now
            state["approach_last_x"] = player_x
            self.cmd_move_x = direction
            logger.info(
                "[route] Rope staging complete; build measured momentum "
                + (
                    "through calibrated window="
                    f"{state['active_launch_window_px'][0]}.."
                    f"{state['active_launch_window_px'][1]}px"
                    if state.get("active_launch_window_px") is not None
                    else f"toward launch_x={state['launch_x']}"
                )
            )
            return True

        return self._request_rope_mount(state, now, player_y)

    def _ignore_completed_rope_up_pixel(self, pixel, command):
        """Keep legacy gray Up strokes from immediately re-grabbing at top."""
        completed_key = getattr(self, "_rope_climb_completed_key", None)
        completed_position = getattr(
            self, "_rope_climb_completed_position", None
        )
        if completed_key is None or completed_position is None or \
                command != "none up none":
            return False
        if int(completed_key[0]) != int(self.idx_routes):
            return False
        x, y = map(int, pixel)
        search_range = max(
            1, int(self.cfg.get("route", {}).get("search_range", 10))
        )
        return (
            abs(x - completed_position[0]) <= search_range
            and abs(y - completed_position[1]) <= search_range * 2
        )

    @staticmethod
    def _is_portal_sweep_route_command(command):
        parts = str(command).split()
        return len(parts) == 3 and parts[2] == "portal"

    @staticmethod
    def _get_route_color_component(img_route, seed, color):
        """Return the 8-connected route-color component containing seed."""
        if img_route is None:
            return None
        seed_x, seed_y = map(int, seed)
        height, width = img_route.shape[:2]
        if not (0 <= seed_x < width and 0 <= seed_y < height):
            return None
        if tuple(img_route[seed_y, seed_x]) != tuple(color):
            return None

        mask = np.all(img_route == color, axis=2).astype(np.uint8)
        component_count, labels, stats, _ = \
            cv2.connectedComponentsWithStats(mask, connectivity=8)
        component_idx = int(labels[seed_y, seed_x])
        if component_idx <= 0 or component_idx >= component_count:
            return None
        x, y, component_width, component_height, area = map(
            int, stats[component_idx]
        )
        return {
            "bbox": (
                x,
                y,
                x + component_width - 1,
                y + component_height - 1,
            ),
            "area": area,
            "color": tuple(color),
        }

    def _reset_portal_sweep(self):
        self._portal_sweep_active = False
        self._portal_sweep_key = None
        self._portal_sweep_region = None
        self._portal_sweep_direction = None
        self._portal_sweep_origin = None
        self._portal_sweep_started_at = None
        self._portal_sweep_last_observed_position = None
        self._portal_sweep_last_nudge_position = None
        self._portal_sweep_last_nudge_direction = None
        self._portal_sweep_last_nudge_time = 0.0
        self._portal_sweep_failed_key = None
        self._portal_sweep_failed_region = None

    def _clear_failed_portal_if_departed(self):
        """Allow a failed portal region to retry only after Hero leaves it."""
        failed_key = getattr(self, "_portal_sweep_failed_key", None)
        region = getattr(self, "_portal_sweep_failed_region", None)
        if failed_key is None or region is None:
            return
        player_x, player_y = map(int, self.loc_player_global)
        left, top, right, bottom = region["bbox"]
        if int(self.idx_routes) != int(failed_key[0]) or not (
                left <= player_x <= right and top <= player_y <= bottom):
            self._portal_sweep_failed_key = None
            self._portal_sweep_failed_region = None

    def _update_active_portal_sweep(self):
        """Hold Up and pulse left/right without leaving the portal region."""
        if not getattr(self, "_portal_sweep_active", False):
            return False
        region = getattr(self, "_portal_sweep_region", None)
        sweep_key = getattr(self, "_portal_sweep_key", None)
        if region is None or sweep_key is None or \
                int(self.idx_routes) != int(sweep_key[0]):
            self._reset_portal_sweep()
            return False

        player_x, player_y = map(int, self.loc_player_global)
        left, top, right, bottom = region["bbox"]
        now = time.monotonic()
        if getattr(self, "_portal_sweep_started_at", None) is None:
            self._portal_sweep_started_at = now
        max_duration = max(
            0.1,
            float(self.cfg["route"].get(
                "portal_sweep_max_duration", 6.0
            )),
        )
        if now - self._portal_sweep_started_at >= max_duration:
            logger.warning(
                "[route] Portal sweep timed out; release Up and wait until "
                "Hero leaves the marker before retrying"
            )
            failed_key = sweep_key
            failed_region = region
            self._reset_portal_sweep()
            self._portal_sweep_failed_key = failed_key
            self._portal_sweep_failed_region = failed_region
            return False

        player_position = (player_x, player_y)
        last_observed = getattr(
            self, "_portal_sweep_last_observed_position", None
        )
        exit_distance = max(
            1,
            int(self.cfg["route"].get("portal_sweep_exit_distance", 10)),
        )
        region_width = right - left + 1
        teleport_step = 0
        if last_observed is not None:
            teleport_step = (
                abs(player_x - last_observed[0])
                + abs(player_y - last_observed[1])
            )
        self._portal_sweep_last_observed_position = player_position
        if teleport_step >= max(exit_distance, region_width + 2):
            logger.info(
                "[route] Portal sweep completed after a minimap jump of "
                f"{teleport_step} pixels to {self.loc_player_global}"
            )
            self._reset_portal_sweep()
            return False

        configured_margin = max(
            0,
            int(self.cfg["route"].get("portal_sweep_edge_margin", 1)),
        )
        edge_margin = min(configured_margin, max(0, (region_width - 1) // 2))
        direction = getattr(self, "_portal_sweep_direction", None)
        if player_x <= left + edge_margin:
            direction = "right"
        elif player_x >= right - edge_margin:
            direction = "left"
        elif direction not in {"left", "right"}:
            direction = "right" if player_x <= (left + right) // 2 else "left"

        self._portal_sweep_direction = direction
        self._portal_sweep_active = True
        self.cmd_move_x = "none"
        self.cmd_move_y = "up"
        last_position = getattr(
            self, "_portal_sweep_last_nudge_position", None
        )
        last_direction = getattr(
            self, "_portal_sweep_last_nudge_direction", None
        )
        repeat_interval = max(
            0.0,
            float(self.cfg["route"].get(
                "portal_sweep_repeat_interval", 0.2
            )),
        )
        can_nudge = (
            player_position != last_position
            or direction != last_direction
            or now - getattr(
                self, "_portal_sweep_last_nudge_time", 0.0
            ) >= repeat_interval
        )
        if region_width > 1 and can_nudge:
            self.cmd_action = f"portal_sweep_{direction}"
            self._portal_sweep_last_nudge_position = player_position
            self._portal_sweep_last_nudge_direction = direction
            self._portal_sweep_last_nudge_time = now
        else:
            # Hold Up between pulses. This gives the capture pipeline time to
            # observe the prior nudge before another horizontal TAP is sent.
            self.cmd_action = "none"
        return True

    def _start_portal_sweep(self, route_action):
        region = route_action.get("portal_region")
        if region is None:
            return False
        # Portal activation has its own bounded Up/search transaction and must
        # never inherit a prior ladder direction after the minimap jump.
        self._reset_ladder_route_hold()
        sweep_key = (int(self.idx_routes), tuple(region["bbox"]))
        if sweep_key != getattr(self, "_portal_sweep_key", None):
            self._reset_portal_sweep()
            self._portal_sweep_key = sweep_key
            self._portal_sweep_region = region
            self._portal_sweep_origin = tuple(map(int, self.loc_player_global))
            self._portal_sweep_last_observed_position = \
                self._portal_sweep_origin
        self._portal_sweep_active = True
        return self._update_active_portal_sweep()

    @staticmethod
    def _route_command_requires_exact_position(command):
        """Return whether a route command represents a point action.

        Movement-only colors remain discoverable within ``search_range`` so
        small minimap localization errors do not stop route following.  A
        command with a non-``none`` action, however, must only be emitted when
        the player's center is on an action-colored route pixel.
        """
        parts = str(command).split()
        return len(parts) == 3 and parts[2] != "none"

    @staticmethod
    def _is_route_goal_command(command):
        parts = str(command).split()
        return len(parts) == 3 and parts[2] == "goal"

    def _get_route_goal_component_at_player(self, route_idx=None):
        """Return the goal component currently under Hero, if any."""
        if route_idx is None:
            route_idx = int(getattr(self, "idx_routes", 0))
        routes = getattr(self, "img_routes", ())
        if not (0 <= int(route_idx) < len(routes)):
            return None
        img_route = routes[int(route_idx)]
        if img_route is None or self.loc_player_global is None:
            return None

        player_x, player_y = map(int, self.loc_player_global)
        height, width = img_route.shape[:2]
        if not (0 <= player_x < width and 0 <= player_y < height):
            return None
        color = tuple(img_route[player_y, player_x])
        command = self.color_code.get(color)
        if not self._is_route_goal_command(command):
            return None
        return self._get_route_color_component(
            img_route, (player_x, player_y), color
        )

    def _route_loop_goal_is_suppressed(self, pixel, color, command):
        """Ignore a disconnected loop goal until Hero leaves its marker."""
        if not self._is_route_goal_command(command):
            return False
        component = self._get_route_color_component(
            self.img_route, pixel, color
        )
        if component is None:
            return False
        key = (int(self.idx_routes), tuple(component["bbox"]))
        return key == getattr(self, "_route_loop_goal_key", None)

    def _clear_route_loop_goal_if_departed(self):
        """Rearm a loop goal after Hero moves outside its painted blob."""
        key = getattr(self, "_route_loop_goal_key", None)
        if key is None or self.loc_player_global is None:
            return
        route_idx, bbox = key
        player_x, player_y = map(int, self.loc_player_global)
        left, top, right, bottom = bbox
        if int(self.idx_routes) != int(route_idx) or not (
                left <= player_x <= right and top <= player_y <= bottom):
            self._route_loop_goal_key = None

    def _route_has_local_continuation(self, route_idx):
        """Check a candidate route using the normal local lookup rules."""
        routes = getattr(self, "img_routes", ())
        if not (0 <= int(route_idx) < len(routes)):
            return False

        original_idx = int(self.idx_routes)
        original_route = self.img_route
        original_debug = self.img_route_debug
        try:
            self.idx_routes = int(route_idx)
            self.img_route = routes[int(route_idx)]
            # This is a read-only validation; do not draw candidate-route
            # lookup lines over the active route's debug image.
            self.img_route_debug = None
            color_code, color_code_up_down = self.get_nearest_color_code()
            if color_code and self._is_route_goal_command(
                    color_code.get("command")):
                color_code = None
            return bool(color_code or color_code_up_down)
        finally:
            self.idx_routes = original_idx
            self.img_route = original_route
            self.img_route_debug = original_debug

    def _current_wz_route_is_jump(self):
        """Return whether the selected generated leg is a WZ JUMP edge."""
        if not getattr(self, "_wz_navigation_enabled", False):
            return False
        navigator = getattr(self, "wz_navigation", None)
        route_legs = getattr(navigator, "route_legs", ())
        route_index = int(getattr(self, "idx_routes", -1))
        if not 0 <= route_index < len(route_legs):
            return False
        action = getattr(route_legs[route_index], "action", None)
        return getattr(action, "value", action) == "JUMP"

    @staticmethod
    def _positive_finite_number(value, fallback=0.0):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(fallback)
        return number if math.isfinite(number) and number > 0 else \
            float(fallback)

    def _wz_walk_speed_px_per_second(self):
        """Return observed, configured, or projected Hero horizontal speed."""
        navigator = getattr(self, "wz_navigation", None)
        observation = getattr(navigator, "_used_observation", None)
        walk_speed = self._positive_finite_number(
            getattr(observation, "walk_speed_px_per_sec", None)
        )
        if walk_speed <= 0:
            cfg = getattr(self, "cfg", {})
            cfg = cfg if isinstance(cfg, dict) else {}
            wz_cfg = cfg.get("wz_navigation", {})
            wz_cfg = wz_cfg if isinstance(wz_cfg, dict) else {}
            motion_cfg = wz_cfg.get("motion", {})
            motion_cfg = motion_cfg if isinstance(motion_cfg, dict) else {}
            walk_speed = self._positive_finite_number(
                motion_cfg.get("walk_speed_px_per_sec")
            )
        if walk_speed <= 0:
            profile = getattr(navigator, "motion_profile", None)
            projection = getattr(navigator, "projection", None)
            world_scale = getattr(projection, "world_scale", (0.0, 0.0))
            try:
                walk_speed = self._positive_finite_number(
                    float(profile.walk_speed_wz_per_sec)
                    * float(world_scale[0])
                )
            except (AttributeError, IndexError, TypeError, ValueError):
                walk_speed = 0.0
        return walk_speed

    def _wz_capture_frame_age_seconds(self, now=None):
        """Return the age of the exact frame that produced Hero's position."""
        now = time.monotonic() if now is None else float(now)
        token = getattr(self, "_current_capture_frame_token", None)
        if not isinstance(token, tuple) or len(token) != 2 or \
                token[0] != "capture":
            return 0.0
        try:
            captured_at = float(token[1])
        except (TypeError, ValueError, OverflowError):
            return 0.0
        if not math.isfinite(captured_at) or captured_at <= 0:
            return 0.0
        return max(0.0, now - captured_at)

    def _wz_timed_jump_route_config(self):
        cfg = getattr(self, "cfg", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        route_cfg = cfg.get("route", {})
        return route_cfg if isinstance(route_cfg, dict) else {}

    def _current_wz_timed_jump_descriptor(self):
        """Describe one horizontal generated jump at the current route leg."""
        if not getattr(self, "_wz_navigation_enabled", False):
            return None
        navigator = getattr(self, "wz_navigation", None)
        route_legs = tuple(getattr(navigator, "route_legs", ()))
        route_index = int(getattr(self, "idx_routes", -1))
        if not 0 <= route_index < len(route_legs):
            return None
        leg = route_legs[route_index]
        action = getattr(leg, "action", None)
        jump_source = getattr(leg, "jump_source", None)
        target = getattr(leg, "target", None)
        bounds = getattr(leg, "jump_trigger_bounds", None)
        if getattr(action, "value", action) != "JUMP" or not all(
                isinstance(point, (tuple, list)) and len(point) >= 2
                for point in (jump_source, target)) or not (
                    isinstance(bounds, (tuple, list)) and len(bounds) >= 4
                ):
            return None
        jump_x = float(jump_source[0])
        target_x = float(target[0])
        if target_x > jump_x:
            direction = "right"
        elif target_x < jump_x:
            direction = "left"
        else:
            return None

        player = getattr(self, "loc_player_global", None)
        if not isinstance(player, (tuple, list)) or len(player) < 2:
            return None
        player_x, player_y = map(float, player[:2])
        left, top, right, bottom = map(float, bounds[:4])
        left, right = min(left, right), max(left, right)
        top, bottom = min(top, bottom), max(top, bottom)
        if not top - 2.0 <= player_y <= bottom + 2.0:
            return None

        # The renderer already derives a physically reachable launch band
        # from the measured airborne distance.  Aim at its midpoint instead
        # of the WZ foothold edge: using jump_source directly waits until the
        # last walkable pixel, where HID/vision latency turns Jump into a fall.
        launch_x = (left + right) / 2.0

        edge_ids = tuple(str(item) for item in getattr(leg, "edge_ids", ()))
        generation = int(getattr(navigator, "resource_generation", -1))
        token = (
            generation,
            route_index,
            edge_ids,
            direction,
            int(round(launch_x)),
            int(round(target_x)),
        )
        return {
            "token": token,
            "route_index": route_index,
            "direction": direction,
            "player_x": player_x,
            "launch_x": launch_x,
        }

    def _predict_current_wz_timed_jump(self, descriptor, now=None):
        """Build a latency-compensated timer prediction for one descriptor."""
        now = time.monotonic() if now is None else float(now)
        speed = self._wz_walk_speed_px_per_second()
        if speed <= 0:
            return None
        route_cfg = self._wz_timed_jump_route_config()
        max_lookahead = max(0.05, float(route_cfg.get(
            "timed_jump_max_lookahead_ms", 1200
        )) / 1000.0)
        frame_age_limit = max(0.05, float(route_cfg.get(
            "timed_jump_frame_age_limit_ms", 500
        )) / 1000.0)
        input_latency = max(0.0, float(route_cfg.get(
            "timed_jump_input_lead_ms", 35
        )) / 1000.0)

        cfg = getattr(self, "cfg", {})
        cfg = cfg if isinstance(cfg, dict) else {}
        system_cfg = cfg.get("system", {})
        system_cfg = system_cfg if isinstance(system_cfg, dict) else {}
        fps = self._positive_finite_number(
            system_cfg.get("fps_limit_main"), 10.0
        )
        configured_interval = 1.0 / fps
        sampled_interval = self._positive_finite_number(getattr(
            self, "_wz_navigation_sample_interval_seconds", None
        ))
        cadence = max(configured_interval, sampled_interval)
        lookahead = min(max_lookahead, max(0.18, cadence * 1.35))

        held_seconds = 0.0
        held_duration = getattr(
            getattr(self, "kb", None), "same_direction_move_seconds", None
        )
        if callable(held_duration):
            try:
                held_seconds = max(0.0, float(held_duration(
                    descriptor["direction"], now=now
                )))
            except (TypeError, ValueError, OverflowError):
                held_seconds = 0.0

        navigator = getattr(self, "wz_navigation", None)
        uncertainty = self._positive_finite_number(getattr(
            getattr(navigator, "registration", None),
            "residual_p95_px",
            None,
        ))
        return predict_directional_jump(
            sample_x=descriptor["player_x"],
            launch_x=descriptor["launch_x"],
            direction=descriptor["direction"],
            speed_px_per_second=speed,
            capture_age_seconds=self._wz_capture_frame_age_seconds(now),
            direction_held_seconds=held_seconds,
            input_latency_seconds=input_latency,
            position_uncertainty_px=uncertainty,
            lookahead_seconds=lookahead,
            maximum_frame_age_seconds=frame_age_limit,
            overshoot_tolerance_px=1.0,
        )

    def _cancel_wz_timed_directional_jump(self):
        self._wz_timed_jump_candidate = None
        token = getattr(self, "_wz_timed_jump_token", None)
        cancel = getattr(
            getattr(self, "kb", None),
            "cancel_scheduled_directional_jump",
            None,
        )
        cancelled = False
        if token is not None and callable(cancel):
            cancelled = bool(cancel(token))
        status = self._wz_timed_jump_status()
        if isinstance(status, dict) and status.get("state") in {
                "firing", "airborne", "braking"}:
            # A Jump TAP that has started cannot be split from its measured
            # 500 ms horizontal hold and landing brake. Retain the token so
            # route, combat, and Buff arbitration continue to yield to it.
            return False
        self._wz_timed_jump_token = None
        self._wz_timed_jump_route_index = None
        self._wz_timed_jump_direction = None
        return cancelled

    def _wz_timed_jump_status(self):
        token = getattr(self, "_wz_timed_jump_token", None)
        status = getattr(
            getattr(self, "kb", None),
            "scheduled_directional_jump_status",
            None,
        )
        if token is None or not callable(status):
            return None
        return status(token)

    def _wz_timed_jump_owns_input(self, now=None):
        """Return whether a committed timer still owns route/combat input."""
        status = self._wz_timed_jump_status()
        if not isinstance(status, dict):
            return False
        state = status.get("state")
        if state in {"pending", "firing", "airborne", "braking"}:
            return True
        if state != "fired":
            return False
        now = time.monotonic() if now is None else float(now)
        finished_at = self._positive_finite_number(
            status.get("finished_at")
        )
        timeout = max(0.05, float(self._wz_timed_jump_route_config().get(
            "timed_jump_confirmation_timeout_ms", 450
        )) / 1000.0)
        return finished_at > 0 and now - finished_at <= timeout

    def _prepare_wz_timed_directional_jump(self):
        """Reserve an approach candidate; scheduling waits for combat arbitration."""
        self._wz_timed_jump_candidate = None
        route_cfg = self._wz_timed_jump_route_config()
        scheduler = getattr(
            getattr(self, "kb", None), "schedule_directional_jump", None
        )
        status_reader = getattr(
            getattr(self, "kb", None),
            "scheduled_directional_jump_status",
            None,
        )
        if route_cfg.get("timed_jump_enable", True) is not True or \
                not callable(scheduler) or not callable(status_reader):
            return False

        descriptor = self._current_wz_timed_jump_descriptor()
        if descriptor is None:
            return False
        token = getattr(self, "_wz_timed_jump_token", None)
        if token is not None:
            if token != descriptor["token"]:
                self._cancel_wz_timed_directional_jump()
                if getattr(self, "_wz_timed_jump_token", None) is not None:
                    self.cmd_move_x = "none"
                    self.cmd_move_y = "none"
                    self.cmd_action = "none"
                    return True
            else:
                status = self._wz_timed_jump_status()
                timer_state = (
                    status.get("state") if isinstance(status, dict) else None
                )
                if timer_state == "pending":
                    self.cmd_move_x = descriptor["direction"]
                    self.cmd_move_y = "none"
                    self.cmd_action = "none"
                    return True
                if timer_state in {"firing", "airborne", "braking"}:
                    self.cmd_move_x = "none"
                    self.cmd_move_y = "none"
                    self.cmd_action = "none"
                    return True
                if timer_state == "fired":
                    if self._wz_timed_jump_owns_input():
                        self.cmd_move_x = "none"
                        self.cmd_move_y = "none"
                        self.cmd_action = "none"
                        return True
                self._cancel_wz_timed_directional_jump()
                self._wz_route_jump_atomic_route_index = None

        prediction = self._predict_current_wz_timed_jump(descriptor)
        if prediction is None:
            return False
        descriptor["prediction"] = prediction
        self._wz_timed_jump_candidate = descriptor
        self.cmd_move_x = descriptor["direction"]
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        return True

    def finalize_wz_timed_directional_jump(self):
        """Arm the timer only after monster/stuck arbitration kept movement."""
        candidate = getattr(self, "_wz_timed_jump_candidate", None)
        self._wz_timed_jump_candidate = None
        if not isinstance(candidate, dict):
            return False
        direction = candidate["direction"]
        if (
            self.cmd_move_x != direction
            or self.cmd_move_y not in {"none", "stop"}
            or self.cmd_action not in {"none", "stop"}
        ):
            return False
        prediction = self._predict_current_wz_timed_jump(candidate)
        if prediction is None:
            return False
        scheduler = getattr(
            getattr(self, "kb", None), "schedule_directional_jump", None
        )
        if not callable(scheduler) or not scheduler(
                direction, prediction.delay_seconds, candidate["token"]):
            return False

        self._wz_timed_jump_token = candidate["token"]
        self._wz_timed_jump_route_index = candidate["route_index"]
        self._wz_timed_jump_direction = direction
        self._wz_jump_move_x = direction
        self._wz_route_jump_atomic_route_index = candidate["route_index"]
        self._wz_route_jump_atomic_pending = False
        self._wz_edge_jump_alignment_route_index = None
        logger.info(
            f"[timed-jump] Armed route{candidate['route_index'] + 1} "
            f"{direction}: sample={prediction.sample_x:.2f}px, "
            f"predicted={prediction.predicted_x:.2f}px, "
            f"launch={prediction.launch_x:.2f}px, "
            f"speed={prediction.speed_px_per_second:.3f}px/s, "
            f"frameAge={prediction.capture_age_seconds * 1000:.0f}ms, "
            f"lead={prediction.input_lead_seconds * 1000:.0f}ms, "
            f"delay={prediction.delay_seconds * 1000:.0f}ms"
        )
        return True

    def _wz_edge_jump_braking_distance(self):
        """Return the live-pixel distance needed to stop before a jump band."""
        cfg = getattr(self, "cfg", {})
        if not isinstance(cfg, dict):
            cfg = {}

        def positive_number(value, fallback=0.0):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return float(fallback)
            return number if math.isfinite(number) and number > 0 else \
                float(fallback)

        system_cfg = cfg.get("system", {})
        if not isinstance(system_cfg, dict):
            system_cfg = {}
        fps = positive_number(system_cfg.get("fps_limit_main"), 10.0)

        navigator = getattr(self, "wz_navigation", None)
        walk_speed = self._wz_walk_speed_px_per_second()

        residual = positive_number(
            getattr(
                getattr(navigator, "registration", None),
                "residual_p95_px",
                None,
            )
        )
        configured_interval = 1.0 / fps
        sampled_interval = positive_number(getattr(
            self, "_wz_navigation_sample_interval_seconds", None
        ))
        prediction_interval = max(
            configured_interval,
            min(1.2, sampled_interval) if sampled_interval > 0 else 0.0,
        )
        observed_step = positive_number(getattr(
            self, "_wz_navigation_sample_step_x", None
        ))
        if walk_speed > 0:
            observed_step = min(observed_step, walk_speed * 1.2)
        predicted_step = max(
            walk_speed * prediction_interval,
            observed_step,
        )
        # fps_limit_main is only an upper bound. Native 4K vision can produce
        # a much longer real sample interval, so brake for the actual cadence
        # and displacement. Registration error and one pixel cover rounding.
        return max(2, int(math.ceil(
            predicted_step + residual + 1.0
        )))

    def _align_compound_wz_edge_jump(self, braking_distance=None):
        """Brake and pulse into a compound jump's measured launch band."""
        if not getattr(self, "_wz_navigation_enabled", False):
            return False
        navigator = getattr(self, "wz_navigation", None)
        route_legs = tuple(getattr(navigator, "route_legs", ()))
        route_index = int(getattr(self, "idx_routes", -1))
        if not 0 <= route_index < len(route_legs):
            self._wz_edge_jump_alignment_route_index = None
            return False

        leg = route_legs[route_index]
        action = getattr(leg, "action", None)
        source = getattr(leg, "source", None)
        target = getattr(leg, "target", None)
        jump_source = getattr(leg, "jump_source", None)
        bounds = getattr(leg, "jump_trigger_bounds", None)
        points = (source, target, jump_source)
        if getattr(action, "value", action) != "JUMP" or not all(
                isinstance(point, (tuple, list)) and len(point) >= 2
                for point in points) or not (
                    isinstance(bounds, (tuple, list)) and len(bounds) >= 4
                ) or tuple(source[:2]) == tuple(jump_source[:2]):
            self._wz_edge_jump_alignment_route_index = None
            return False

        # Once this route's atomic jump has been dispatched, route pixels must
        # be allowed to recognize the landing goal instead of steering back
        # toward the takeoff edge.
        if route_index == getattr(
                self, "_wz_route_jump_atomic_route_index", None) and not \
                getattr(self, "_wz_route_jump_atomic_pending", False):
            return False

        player = getattr(self, "loc_player_global", None)
        if not isinstance(player, (tuple, list)) or len(player) < 2:
            return False
        player_x, player_y = map(float, player[:2])
        source_x = float(source[0])
        target_x = float(target[0])
        jump_x = float(jump_source[0])
        left, top, right, bottom = map(float, bounds[:4])
        left, right = min(left, right), max(left, right)
        top, bottom = min(top, bottom), max(top, bottom)
        vertical_margin = 2.0
        if not top - vertical_margin <= player_y <= bottom + vertical_margin:
            return False

        if target_x > jump_x and source_x < jump_x:
            jump_direction = "right"
            inner_distance = left - player_x
            outer_distance = player_x - right
            inward_direction = "left"
        elif target_x < jump_x and source_x > jump_x:
            jump_direction = "left"
            inner_distance = player_x - right
            outer_distance = left - player_x
            inward_direction = "right"
        else:
            return False

        if left <= player_x <= right:
            command = f"{jump_direction} none jump"
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = command.split()
            self._reserve_wz_route_jump_atomic(command)
            self._wz_edge_jump_alignment_route_index = None
            return True

        braking_distance = (
            self._wz_edge_jump_braking_distance()
            if braking_distance is None
            else max(2, int(braking_distance))
        )
        alignment_direction = None
        distance = 0.0
        if 0 < inner_distance <= braking_distance:
            # Approach from the platform interior: release the continuous key
            # and replace it with a 30 ms pulse toward the launch band.
            alignment_direction = jump_direction
            distance = inner_distance
        elif 0 < outer_distance <= braking_distance:
            # A delayed minimap sample may already be one pixel beyond the
            # band. Pulse back onto the foothold instead of renewing movement
            # toward empty space.
            alignment_direction = inward_direction
            distance = outer_distance

        if alignment_direction is None:
            if inner_distance > braking_distance:
                self._wz_edge_jump_alignment_route_index = None
                return False
            # Do not renew the outward movement after a larger overshoot. The
            # next raw sample can then classify the fall and select recovery.
            self.cmd_move_x = "none"
            self.cmd_move_y = "none"
            self.cmd_action = "none"
            return True

        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = f"jump_align_{alignment_direction}"
        if route_index != getattr(
                self, "_wz_edge_jump_alignment_route_index", None):
            logger.info(
                f"[route] Brake route{route_index + 1} {distance:.1f}px "
                f"before {jump_direction} edge jump "
                f"(guard={braking_distance}px); align "
                f"{alignment_direction} into x={int(left)}..{int(right)}"
            )
            self._wz_edge_jump_alignment_route_index = route_index
        return True

    def _clear_wz_route_jump_atomic_if_departed(self):
        """Rearm one generated jump after the executor leaves its route."""
        current_index = int(getattr(self, "idx_routes", -1))
        timed_index = getattr(self, "_wz_timed_jump_route_index", None)
        if timed_index is not None and current_index != int(timed_index):
            self._cancel_wz_timed_directional_jump()
        reserved_index = getattr(
            self, "_wz_route_jump_atomic_route_index", None
        )
        if reserved_index is None or int(getattr(
                self, "idx_routes", -1)) == int(reserved_index):
            return
        self._wz_route_jump_atomic_pending = False
        self._wz_route_jump_atomic_route_index = None

    def _reserve_wz_route_jump_atomic(self, command):
        """Reserve one HID arbitration slot for every generated JUMP leg."""
        parts = str(command).split()
        if len(parts) != 3 or parts[2] != "jump" or \
                parts[1] not in {"none", "stop"} or \
                parts[0] not in {"left", "right", "none", "stop"} or \
                not self._current_wz_route_is_jump():
            return False

        route_index = int(self.idx_routes)
        if route_index != getattr(
                self, "_wz_route_jump_atomic_route_index", None):
            self._wz_route_jump_atomic_route_index = route_index
            self._wz_route_jump_atomic_pending = True
            logger.info(
                f"[route] Reserved route{route_index + 1} WZ JUMP as an "
                f"atomic HID transaction: {' '.join(parts)}"
            )
        return True

    def _handoff_wz_jump(self):
        """Issue any exact generated JUMP during the goal-transition frame."""
        if not getattr(self, "_wz_navigation_enabled", False):
            return False
        if getattr(self, "loc_player_global", None) is None or not isinstance(
                getattr(self, "img_route", None), np.ndarray):
            return False

        color_code, _ = self.get_nearest_color_code(
            include_rope_climb=False
        )
        if not color_code or not color_code.get("exact_action", False):
            return False

        command = str(color_code.get("command", ""))
        if not self._reserve_wz_route_jump_atomic(command):
            return False

        self.cmd_move_x, self.cmd_move_y, self.cmd_action = command.split()
        logger.info(
            f"[route] Queued route{self.idx_routes + 1} WZ JUMP "
            "for same-frame HID dispatch"
        )
        return True

    def get_nearest_color_code(self, include_rope_climb=True):
        '''
        Searches for route colors around the player on the route map.

        This function:
        - Acquires a hand-drawn climb guide before reaching its rope endpoint.
        - Triggers compact stationary-jump blobs inside ``search_range``.
        - Requires the player's center pixel to overlap other point actions
          such as directional jump, teleport, goal, and stop.
        - Scans the search box for movement-only route colors.
        - Tracks the closest matching pixel using Manhattan distance (|dx| + |dy|).
        - Returns a dictionary containing the nearest matching
          pixel's position, color, action label, and distance.

        Returns:
            dict or None: Dictionary containing:
                - "pixel": (x, y) coordinate of the matched pixel
                - "color": matched RGB color tuple
                - "action": corresponding action string from config
                - "distance": Manhattan distance from player
            Returns None if no matching color is found within the region.
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg["route"]["search_range"])
        x_max = min(w, x0 + self.cfg["route"]["search_range"])
        y_min = max(0, y0 - self.cfg["route"]["search_range"])
        y_max = min(h, y0 + self.cfg["route"]["search_range"])

        nearest = None
        nearest_up_down = None
        min_dist = float('inf')
        min_dist_up_down = float('inf')

        rope_climb_target = (
            self._get_nearby_rope_climb_target()
            if include_rope_climb
            else None
        )
        if rope_climb_target is not None:
            target_center = rope_climb_target["center"]
            nearest = {
                "pixel": target_center,
                "color": rope_climb_target["color"],
                "command": rope_climb_target["command"],
                "distance": (
                    abs(target_center[0] - x0)
                    + abs(target_center[1] - y0)
                ),
                "exact_action": False,
                "rope_climb": True,
                "target_center": target_center,
                "guide_key": rope_climb_target["key"],
                "explicit_rope_target": rope_climb_target.get(
                    "explicit_target", False
                ),
                "rope_mount_plan": rope_climb_target.get(
                    "rope_mount_plan"
                ),
                "rope_destination_y": rope_climb_target.get(
                    "rope_destination_y"
                ),
            }
            min_dist = nearest["distance"]

        stationary_jump_target = (
            None
            if rope_climb_target is not None
            else self._get_nearby_stationary_jump_target()
        )
        if stationary_jump_target is not None:
            target_center = stationary_jump_target["center"]
            target_distance = (
                abs(target_center[0] - x0) + abs(target_center[1] - y0)
            )
            nearest = {
                "pixel": target_center,
                "color": stationary_jump_target["color"],
                "command": stationary_jump_target["command"],
                "distance": target_distance,
                "exact_action": True,
                "stationary_jump_proximity": True,
                "target_center": target_center,
            }
            min_dist = nearest["distance"]

        # Point actions must be under the player's center. Do this lookup
        # independently of search_range (including a configured value of 0)
        # and never pick the same action early from a neighboring pixel.
        if 0 <= x0 < w and 0 <= y0 < h:
            player_pixel = tuple(self.img_route[y0, x0])
            player_command = self.color_code.get(player_pixel)
            if player_command is not None and \
                    not self._is_rope_climb_route_command(player_command) and \
                    self._route_command_requires_exact_position(player_command):
                portal_region = None
                portal_failed = False
                if self._is_portal_sweep_route_command(player_command):
                    portal_region = self._get_route_color_component(
                        self.img_route,
                        (x0, y0),
                        player_pixel,
                    )
                    if portal_region is not None:
                        portal_key = (
                            int(self.idx_routes),
                            tuple(portal_region["bbox"]),
                        )
                        portal_failed = portal_key == getattr(
                            self, "_portal_sweep_failed_key", None
                        )
                aligned_component = any(
                    (x0, y0) in target["pixels"]
                    for target in self._get_active_stationary_jump_targets()
                )
                loop_goal_suppressed = \
                    self._route_loop_goal_is_suppressed(
                        (x0, y0), player_pixel, player_command
                    )
                if not portal_failed and not loop_goal_suppressed and not (
                    self._is_stationary_jump_route_command(player_command)
                    and aligned_component
                ):
                    nearest = {
                        "pixel": (x0, y0),
                        "color": player_pixel,
                        "command": player_command,
                        "distance": 0,
                        "exact_action": True,
                    }
                    if portal_region is not None:
                        nearest["portal_sweep"] = True
                        nearest["portal_region"] = portal_region
                    min_dist = 0

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                dist = abs(x - x0) + abs(y - y0)
                command = self.color_code.get(pixel)
                # Movement-only route colors retain the configured search
                # tolerance. Point actions were handled only at (x0, y0)
                # above and are deliberately ignored everywhere else.
                if rope_climb_target is None and \
                        stationary_jump_target is None and \
                        command is not None and \
                        not self._route_command_requires_exact_position(command) and \
                        dist < min_dist:
                    nearest = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": command,
                        "distance": dist,
                        "exact_action": False,
                    }
                    min_dist = dist
                # Get nearest color (up, dowm)
                up_down_command = self.color_code_up_down.get(pixel)
                if up_down_command is not None and \
                        not self._ignore_completed_rope_up_pixel(
                            (x, y), up_down_command
                        ) and dist < min_dist_up_down:
                    nearest_up_down = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": up_down_command,
                        "distance": dist
                    }
                    min_dist_up_down = dist

        # Debug
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (self.cfg["route"]["search_range"]*2,
             self.cfg["route"]["search_range"]*2),
            (0, 0, 255), "", text_height=0.4, thickness=1,
        )
        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if nearest is not None:
            draw_line(
                self.img_route_debug,
                self.loc_player_global, # start point
                nearest["pixel"],       # end point
                (0, 255, 0),            # green line
                1                       # thickness
            )
            # Print color code on debug image
            self._draw_debug_text(
                f"Route Action: {nearest['command']}",
                (650, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA,
                reference_position=True,
            )
            self._draw_debug_text(
                f"Route Index: {self.idx_routes}",
                (650, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA,
                reference_position=True,
            )

        if nearest_up_down is not None:
            self._draw_debug_text(
                f"Route Action: {nearest_up_down['command']}",
                (650, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA,
                reference_position=True,
            )
            draw_line(
                self.img_route_debug,
                self.loc_player_global,  # start point
                nearest_up_down["pixel"],# end point
                (0, 0, 255),             # green line
                1                        # thickness
            )

        return nearest, nearest_up_down  # if not found return none

    def get_attack_range(self, is_left=True, attack_type=None):
        '''
        get_attack_range
        '''
        attack_type = attack_type or self.cfg["bot"]["attack"]
        if attack_type == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2
            dy = self.cfg["aoe_skill"]["range_y"] // 2
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        elif attack_type in {
                "directional", "directional_aoe", "power_knockback"}:
            if attack_type == "directional":
                section = "directional_attack"
            elif attack_type == "directional_aoe":
                section = "directional_aoe"
            else:
                section = "power_knockback"
            attack_cfg = self.cfg[section]
            range_x = int(
                attack_cfg[
                    "trigger_distance_x"
                    if attack_type == "power_knockback"
                    else "range_x"
                ]
            )
            range_y = int(attack_cfg["range_y"])
            if is_left:
                x0 = self.loc_player[0] - range_x
                x1 = self.loc_player[0]
            else:
                x0 = self.loc_player[0]
                x1 = x0 + range_x
            y0 = self.loc_player[1] - range_y // 2
            y1 = y0 + range_y
        else:
            raise RuntimeError(f"Unsupported attack mode: {attack_type}")

        return (x0, y0, x1, y1)

    def _projectile_terrain_blocker(self, monster):
        """Return WZ geometry blocking a horizontal attack at ``monster``."""
        wz_cfg = self.cfg.get("wz_navigation", {})
        if not wz_cfg.get("projectile_terrain_check", False):
            return None
        navigator = getattr(self, "wz_navigation", None)
        if navigator is None or not getattr(navigator, "active", False) or \
                getattr(navigator, "jump_active", False) or \
                getattr(self, "is_on_ladder", False):
            return None
        projection = getattr(navigator, "projection", None)
        wz_map = getattr(navigator, "wz_map", None)
        registration = getattr(navigator, "registration", None)
        if projection is None or wz_map is None or registration is None:
            return None

        scale_x = float(registration.scale_x)
        scale_y = float(registration.scale_y)
        if scale_x <= 0.0 or scale_y <= 0.0:
            return None

        player_world = projection.navigation_to_world(
            self.loc_player_global
        )
        monster_x = float(detection_center(monster)[0])
        target_world_x = player_world.x + (
            monster_x - float(self.loc_player[0])
        ) / scale_x
        projectile_height = float(
            wz_cfg.get("projectile_height_wz", 30.0)
        )
        clearance = float(
            wz_cfg.get("projectile_clearance_wz", 8.0)
        )
        motion_profile = getattr(navigator, "motion_profile", None)
        origin_margin = float(
            getattr(motion_profile, "character_half_width_wz", 15.0)
        )
        blocker = wz_map.first_horizontal_projectile_blocker(
            Point(
                player_world.x,
                player_world.y - projectile_height,
            ),
            target_world_x,
            clearance=clearance,
            origin_margin=origin_margin,
        )
        if blocker is not None and \
                getattr(navigator, "map_id", None) == FOREST_FLOOR_MAP_ID and \
                blocker.kind == "wall" and blocker.geometry_id in \
                FOREST_FLOOR_P1_DECORATIVE_WALL_IDS:
            blocker = None
        if blocker is not None and wz_cfg.get(
                "projectile_terrain_debug", False) and getattr(
                    self, "img_frame_debug", None) is not None:
            block_x = int(round(
                self.loc_player[0]
                + (blocker.point.x - player_world.x) * scale_x
            ))
            shot_y = int(round(self.loc_player[1]))
            color = (0, 165, 255)
            thickness = max(1, int(round(min(scale_x, scale_y))))
            cv2.line(
                self.img_frame_debug,
                tuple(map(int, self.loc_player)),
                (block_x, shot_y),
                color,
                thickness,
            )
            cv2.circle(
                self.img_frame_debug,
                (block_x, shot_y),
                max(2, thickness * 2),
                color,
                -1,
            )
            self._draw_debug_rectangle(
                monster["position"],
                monster["size"],
                color,
                f"Terrain blocked: {blocker.kind}",
                thickness=1,
                text_height=0.45,
            )
        return blocker

    def get_power_knockback_monsters(self, is_left=True):
        """Return same-height monsters inside the close-range threshold.

        Near monsters are selected from the raw detections by center distance,
        rather than by the normal attack-overlap threshold. A sprite that
        crosses the Hero center can otherwise be mostly outside one half-box
        even though it is exactly the monster preventing an archer from
        shooting.
        """
        knockback_cfg = self.cfg["power_knockback"]
        max_distance_x = int(knockback_cfg["trigger_distance_x"])
        half_range_y = int(knockback_cfg["range_y"]) / 2.0
        player_x, player_y = self.loc_player
        keyboard_controller = getattr(self, "kb", None)
        facing = getattr(keyboard_controller, "cached_facing", None)
        if facing not in {"left", "right"}:
            facing = getattr(self, "cmd_move_x", None)
        if facing not in {"left", "right"}:
            facing = "left"

        close_monsters = []
        for monster in self.monsters:
            monster_x, monster_y = detection_center(monster)
            distance_x = abs(monster_x - player_x)
            if distance_x > max_distance_x or \
                    abs(monster_y - player_y) > half_range_y:
                continue

            if monster_x < player_x:
                direction = "left"
            elif monster_x > player_x:
                direction = "right"
            else:
                # Exact center overlap has no geometric side. Use the last
                # trustworthy facing so the escape skill remains actionable.
                direction = facing

            if (direction == "left") == bool(is_left):
                close_monsters.append(monster)

        return close_monsters

    def get_monsters_in_attack_range(
            self, is_left=True, attack_type="directional"):
        """Return monsters that can be hit in one directional attack box.

        Search margins only decide which detections are collected. This method
        applies the real overlap threshold. AoE counting assigns a box to
        exactly one side using its center, so a large detection crossing the
        player is not counted on both sides. Normal nearest-target selection
        uses the same side rule so an invalid crossing box cannot hide a valid
        target farther away.
        """
        attack_box = self.get_attack_range(
            is_left=is_left,
            attack_type=attack_type,
        )
        monsters_info = getattr(self, "monsters_info", {})
        legacy_min_mob_area = None
        if not self.is_yolo_monster_detection() and monsters_info:
            legacy_min_mob_area = min(
                img.shape[0] * img.shape[1]
                for _, imgs in monsters_info.items()
                for img, _ in imgs
            )

        player_x = self.loc_player[0]
        attackable_monsters = []
        for monster in self.monsters:
            monster_center_x = detection_center(monster)[0]
            if is_left and monster_center_x >= player_x:
                continue
            if not is_left and monster_center_x <= player_x:
                continue

            monster_box = detection_to_box(monster)
            inter_area = intersection_area(attack_box, monster_box)
            monster_area = max(0, monster_box[2] - monster_box[0]) * max(
                0, monster_box[3] - monster_box[1]
            )
            inter_area_thres = min(
                monster_area
                if legacy_min_mob_area is None
                else legacy_min_mob_area,
                self.cfg['monster_detect']['max_mob_area_trigger'],
            )
            if inter_area > 0 and inter_area >= inter_area_thres:
                attackable_monsters.append(monster)

        if self.cfg["monster_detect"].get("with_enemy_hp_bar", False):
            # The legacy fallback can report the same entity once from its
            # sprite and once from its wider inferred health-bar box. Compare
            # intersection against the smaller box instead of IoU; different
            # widths can describe the same entity while producing a low IoU.
            visual_monsters = [
                monster for monster in attackable_monsters
                if monster.get("name") != "Health Bar"
            ]
            health_bar_monsters = [
                monster for monster in attackable_monsters
                if monster.get("name") == "Health Bar"
            ]

            def box_area(monster):
                x0, y0, x1, y1 = detection_to_box(monster)
                return max(0, x1 - x0) * max(0, y1 - y0)

            unmatched_health_bars = []
            for health_bar in health_bar_monsters:
                health_box = detection_to_box(health_bar)
                health_area = box_area(health_bar)
                is_duplicate = any(
                    (
                        intersection_area(
                            health_box,
                            detection_to_box(monster),
                        )
                        / max(1, min(health_area, box_area(monster)))
                    ) >= 0.5
                    for monster in visual_monsters
                )
                if not is_duplicate:
                    unmatched_health_bars.append(health_bar)
            attackable_monsters = visual_monsters + unmatched_health_bars

        if attack_type in {"directional", "directional_aoe"}:
            attackable_monsters = [
                monster for monster in attackable_monsters
                if self._projectile_terrain_blocker(monster) is None
            ]

        return attackable_monsters

    def get_nearest_monster(self, is_left=True, attack_type="directional"):
        '''
        Finds the nearest monster within the player's attack range.

        This function:
        - Defines an attack box relative to the player position,
            depending on the facing direction (`is_left`).
        - Iterates through all detected monsters and checks which ones overlap
          with the attack box.
        - Returns the closest valid monster that meets the overlap criteria.

        Args:
            is_left (bool): If True, assume the player is facing left;
                            adjusts attack box accordingly.
        Returns:
            dict or None: The nearest monster's info dict, or None if no valid match.
        '''

        attackable_monsters = self.get_monsters_in_attack_range(
            is_left=is_left,
            attack_type=attack_type,
        )
        return min(
            attackable_monsters,
            key=lambda monster: (
                abs(detection_center(monster)[0] - self.loc_player[0])
                + abs(detection_center(monster)[1] - self.loc_player[1])
            ),
            default=None,
        )

    def get_directional_aoe_direction(
            self, monsters_left, monsters_right, min_monsters):
        """Choose a qualifying side for a single-sided AoE attack."""
        counts = {
            "left": len(monsters_left),
            "right": len(monsters_right),
        }
        qualifying = [
            direction
            for direction, count in counts.items()
            if count >= min_monsters
        ]
        if not qualifying:
            return None
        if len(qualifying) == 1:
            return qualifying[0]
        if counts["left"] != counts["right"]:
            return max(qualifying, key=counts.get)

        def nearest_distance(monsters):
            return min(
                abs(detection_center(monster)[0] - self.loc_player[0])
                + abs(detection_center(monster)[1] - self.loc_player[1])
                for monster in monsters
            )

        distances = {
            "left": nearest_distance(monsters_left),
            "right": nearest_distance(monsters_right),
        }
        if distances["left"] != distances["right"]:
            return min(qualifying, key=distances.get)

        keyboard_controller = getattr(self, "kb", None)
        facing = getattr(keyboard_controller, "cached_facing", None)
        if facing in qualifying:
            return facing
        route_direction = getattr(self, "cmd_move_x", None)
        if route_direction in qualifying:
            return route_direction
        return "left"

    def is_yolo_monster_detection(self):
        """Return whether the active config uses the YOLO mob backend."""
        return (
            str(
                self.cfg.get("monster_detect", {}).get(
                    "backend", "template"
                )
            ).lower()
            == "yolo"
        )

    def get_yolo_monsters_in_range(
        self, top_left, bottom_right, confidence=None
    ):
        """Run one full-camera YOLO pass and retain boxes touching the ROI."""
        detector = getattr(self, "yolo_monster_detector", None)
        if detector is None:
            logger.error("YOLO monster detector is not loaded")
            return []

        frame_h, frame_w = self.img_frame.shape[:2]
        x0, y0 = top_left
        x1, y1 = bottom_right
        x0 = min(max(0, int(x0)), frame_w)
        x1 = min(max(0, int(x1)), frame_w)
        y0 = min(max(0, int(y0)), frame_h)
        y1 = min(max(0, int(y1)), frame_h)
        if x1 <= x0 or y1 <= y0:
            self.draw_monster_detections([], (x0, y0), (x1, y1))
            return []

        monsters = detector.detect(
            # Keep the complete normalized frame. The model was trained with
            # the bottom UI present, and cropping at ui_y_start can cut off
            # monsters standing on the lowest platform. The ROI still limits
            # which detections are returned to the attack/debug caller.
            self.img_frame,
            roi=(x0, y0, x1, y1),
            confidence=confidence,
            class_name=detector.class_name,
            inference_class_names=getattr(
                self, "yolo_inference_class_names", (detector.class_name,)
            ),
            inference_confidence=detector.confidence,
            cache_key=getattr(self, "_current_capture_frame_token", None),
        )
        monsters = self.filter_yolo_hero_class_conflicts(monsters)
        # Pet candidates are deliberately the small boxes that the generic
        # size guard may reject next. OCR them first so the identity-specific
        # decision is both observable and limited to the intended candidates.
        monsters = self.filter_pet_yolo_detections(monsters)
        monsters = self.filter_yolo_detections_by_box_size(monsters)
        self.draw_monster_detections(monsters, (x0, y0), (x1, y1))
        return monsters

    def filter_yolo_hero_class_conflicts(self, monsters):
        """Remove a mob box duplicating this frame's accepted Hero box.

        Ultralytics NMS is class-aware by default, so one visual object can
        survive once as ``hero`` and again as ``mob``.  Only a freshly
        accepted Hero detection is authoritative here; an indefinitely cached
        Hero box must never hide a real monster after the camera has moved.
        """
        if not monsters or getattr(
                self, "yolo_hero_detector", None) is not getattr(
                    self, "yolo_monster_detector", None):
            return monsters

        frame_token = getattr(self, "_current_capture_frame_token", None)
        hero = getattr(self, "last_yolo_hero_detection", None)
        if frame_token is None or not isinstance(hero, dict) or \
                hero.get("frame_token") != frame_token:
            return monsters

        hero_box = detection_to_box(hero)
        hero_area = max(0, hero_box[2] - hero_box[0]) * max(
            0, hero_box[3] - hero_box[1]
        )
        if hero_area <= 0:
            return monsters

        kept = []
        for monster in monsters:
            monster_box = detection_to_box(monster)
            monster_area = max(0, monster_box[2] - monster_box[0]) * max(
                0, monster_box[3] - monster_box[1]
            )
            inter_area = intersection_area(hero_box, monster_box)
            union_area = hero_area + monster_area - inter_area
            iou = inter_area / max(1, union_area)
            if iou < 0.5:
                kept.append(monster)
                continue

            self._draw_debug_rectangle(
                monster["position"],
                monster["size"],
                (255, 0, 255),
                f"Hero conflict filtered: {iou:.2f}",
                thickness=1,
                text_height=0.45,
            )

        return kept

    def filter_yolo_detections_by_box_size(self, monsters):
        """Discard detections smaller than either configured box dimension."""
        if not monsters:
            return monsters

        monster_cfg = self.cfg.get("monster_detect", {})
        min_width = max(0.0, float(monster_cfg.get("min_box_width", 0)))
        min_height = max(0.0, float(monster_cfg.get("min_box_height", 0)))
        if min_width == 0 and min_height == 0:
            return monsters

        kept = []
        for monster in monsters:
            height, width = monster["size"]
            if width < min_width or height < min_height:
                continue
            kept.append(monster)
        return kept

    def filter_pet_yolo_detections(self, monsters):
        """OCR-check only small YOLO mob boxes immediately beside Hero."""
        if not monsters:
            return monsters

        pet_cfg = self.cfg.get("nametag", {}).get("pet", {})
        target = normalize_ocr_text(pet_cfg.get("yolo_ocr_text", ""))
        if not (
            pet_cfg.get("enable", False)
            and pet_cfg.get("filter_yolo_mob", True)
            and target
            and getattr(self, "screen_player_location_valid", False)
        ):
            return monsters

        try:
            max_box_width, max_box_height = (
                float(value)
                for value in pet_cfg["yolo_ocr_max_box_size"]
            )
            max_distance_x, max_distance_y = (
                float(value)
                for value in pet_cfg["yolo_ocr_max_hero_distance"]
            )
            min_score = float(pet_cfg.get("yolo_ocr_min_score", 0.8))
            box_threshold = float(
                pet_cfg.get("yolo_ocr_box_threshold", 0.2)
            )
            player_x, player_y = (float(value) for value in self.loc_player)
            if min(
                max_box_width,
                max_box_height,
                max_distance_x,
                max_distance_y,
            ) <= 0:
                raise ValueError("OCR size and distance limits must be positive")
        except (KeyError, TypeError, ValueError) as exc:
            error = f"invalid pet mob OCR configuration: {exc}"
            if error != getattr(self, "_last_pet_mob_ocr_error", None):
                logger.warning(f"[pet-mob-filter] {error}; keeping detections")
                self._last_pet_mob_ocr_error = error
            return monsters

        locator = getattr(self, "_pet_mob_ocr_locator", None)
        if locator is None:
            locator = RapidOcrTextLocator()
            self._pet_mob_ocr_locator = locator
        frame_h, frame_w = self.img_frame.shape[:2]

        kept = []
        for monster in monsters:
            height, width = monster["size"]
            center_x, center_y = detection_center(monster)
            is_nearby_small_box = (
                0 < width <= max_box_width
                and 0 < height <= max_box_height
                and abs(center_x - player_x) <= max_distance_x
                and abs(center_y - player_y) <= max_distance_y
            )
            if not is_nearby_small_box:
                kept.append(monster)
                continue

            x1, y1, x2, y2 = detection_to_box(monster)
            horizontal_padding = int(round(width * 0.4))
            roi = (
                max(0, x1 - horizontal_padding),
                max(0, y2 - int(round(height * 0.1))),
                min(frame_w, x2 + horizontal_padding),
                min(frame_h, y2 + int(round(height * 0.65))),
            )
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                kept.append(monster)
                continue

            try:
                match = locator.locate(
                    self.img_frame,
                    roi,
                    (target,),
                    min_score=min_score,
                    match_mode="exact",
                    box_threshold=box_threshold,
                )
                self._last_pet_mob_ocr_error = None
            except (RapidOcrError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if error != getattr(self, "_last_pet_mob_ocr_error", None):
                    logger.warning(
                        f"[pet-mob-filter] OCR unavailable ({error}); "
                        "keeping detections"
                    )
                    self._last_pet_mob_ocr_error = error
                kept.append(monster)
                continue
            if match is None:
                kept.append(monster)
                continue

            self._draw_debug_rectangle(
                monster["position"],
                monster["size"],
                (255, 0, 255),
                f"Pet OCR filtered: {match.score:.2f}",
                thickness=1,
                text_height=0.45,
            )
            xs = [point[0] for point in match.box]
            ys = [point[1] for point in match.box]
            self._draw_debug_rectangle(
                (min(xs), min(ys)),
                (max(ys) - min(ys), max(xs) - min(xs)),
                (255, 0, 255),
                "Pet OCR name",
                thickness=1,
                text_height=0.45,
            )

        return kept

    def get_monsters_in_range(self, top_left, bottom_right, diff_thres=None):
        '''
        get_monsters_in_range
        '''
        if self.is_yolo_monster_detection():
            # ``diff_thres`` belongs to the legacy SQDIFF matcher and must not
            # silently alter the calibrated YOLO confidence threshold.
            return self.get_yolo_monsters_in_range(top_left, bottom_right)

        # Reuse the reliable debug edge-correlation + masked-color pipeline in
        # normal mode. The caller still supplies the smaller combat ROI, so
        # normal mode does not pay for a full-camera scan every frame.
        if (
            self.cfg.get("bot", {}).get("mode") == "normal"
            and self.cfg.get("monster_detect", {}).get(
                "mode", "debug_pipeline"
            ) == "debug_pipeline"
        ):
            return self.get_debug_monsters_in_range(
                top_left,
                bottom_right,
                score_thres=diff_thres,
            )

        x0, y0 = top_left
        x1, y1 = bottom_right

        img_roi = self.img_frame[y0:y1, x0:x1]
        default_match_threshold = (
            self.cfg["monster_detect"]["diff_thres"]
            if diff_thres is None else float(diff_thres)
        )
        threshold_by_monster = self.cfg["monster_detect"].get(
            "diff_thres_by_monster", {}
        )

        # Shift player's location into ROI coordinate system
        px, py = self.loc_player
        px_in_roi = px - x0
        py_in_roi = py - y0

        # Define rectangle range around player (in ROI coordinate)
        char_x_min = max(0, px_in_roi - self.cfg["character"]["width"] // 2)
        char_x_max = min(img_roi.shape[1], px_in_roi + self.cfg["character"]["width"] // 2)
        char_y_min = max(0, py_in_roi - self.cfg["character"]["height"] // 2)
        char_y_max = min(img_roi.shape[0], py_in_roi + self.cfg["character"]["height"] // 2)

        monsters = []

        def get_match_points(result, match_threshold):
            """Return sparse, bounded SQDIFF candidates.

            A permissive threshold can match most pixels in the search ROI.
            Expanding every matching pixel into a detection makes the
            following NMS quadratic and can stall the normal-mode main loop
            before the UI receives its first visualization frame. Keep only
            well-separated local minima from each template in every mode.
            """
            result = np.nan_to_num(
                result, nan=1.0, posinf=1.0, neginf=1.0
            )
            if self.is_debug_mode():
                candidate_cfg = self.cfg.get("debug", {})
                radius = max(
                    1, int(candidate_cfg.get("local_min_radius", 9))
                )
                top_k = max(
                    1, int(candidate_cfg.get("template_top_k", 1))
                )
            else:
                candidate_cfg = self.cfg.get("monster_detect", {})
                radius = max(
                    1, int(candidate_cfg.get("local_min_radius", 9))
                )
                top_k = max(
                    1,
                    int(
                        candidate_cfg.get(
                            "max_candidates_per_template", 12
                        )
                    ),
                )

            kernel_size = radius * 2 + 1
            local_min = cv2.erode(
                result, np.ones((kernel_size, kernel_size), dtype=np.uint8)
            )
            ys, xs = np.where(
                (result <= match_threshold)
                & (result <= local_min + 1e-7)
            )
            if len(xs) == 0:
                return []

            scores = result[ys, xs]
            if len(scores) > top_k:
                # Avoid sorting and materializing Python objects for every
                # pixel when the configured threshold is very permissive.
                selected = np.argpartition(scores, top_k - 1)[:top_k]
                ys = ys[selected]
                xs = xs[selected]
                scores = scores[selected]

            order = np.argsort(scores)
            return [
                (int(xs[index]), int(ys[index]), float(scores[index]))
                for index in order
            ]

        for monster_name, monster_imgs in self.monsters_info.items():
            match_threshold = float(
                threshold_by_monster.get(
                    monster_name, default_match_threshold
                )
            )
            for img_monster, mask_monster in monster_imgs:
                h_roi, w_roi = img_roi.shape[:2]
                h_temp, w_temp = img_monster.shape[:2]
                if h_temp > h_roi or w_temp > w_roi:
                    continue
                if self.cfg["bot"]["mode"] == "patrol":
                    pass # Don't detect monster using template in patrol mode
                elif self.cfg["monster_detect"]["mode"] == "template_free":
                    # Generate mask where pixel is exactly (0,0,0)
                    black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    # cv2.imshow("Black Pixel Mask", black_mask)

                    # Zero out mask inside this region (ignore player's own character)
                    black_mask[char_y_min:char_y_max, char_x_min:char_x_max] = 0

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
                    closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
                    # cv2.imshow("Black Mask", closed_mask)

                    # draw player character bounding box
                    self._draw_debug_rectangle(
                        (char_x_min+x0, char_y_min+y0),
                        (self.cfg["character"]["height"], self.cfg["character"]["width"]),
                        (255, 0, 0), "Character Box"
                    )

                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

                    monsters = []
                    min_area = 1000
                    for i in range(1, num_labels):
                        x, y, w, h, area = stats[i]
                        if area > min_area:
                            monsters.append({
                                "name": "",
                                "position": (x0+x, y0+y),
                                "size": (h, w),
                                "score": 1.0,
                            })
                elif self.cfg["monster_detect"]["mode"] == "contour_only":
                    # Use only black lines contour to detect monsters
                    # Create masks (already grayscale)
                    mask_pattern = np.all(img_monster == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

                    # Zero out mask inside this region (ignore player's own character)
                    mask_roi[char_y_min:char_y_max, char_x_min:char_x_max] = 0

                    # Apply Gaussian blur (soften the masks)
                    blur = self.cfg["monster_detect"]["contour_blur"]
                    img_monster_blur = cv2.GaussianBlur(mask_pattern, (blur, blur), 0)
                    img_roi_blur = cv2.GaussianBlur(mask_roi, (blur, blur), 0)

                    # Check template vs ROI size before matching
                    h_roi, w_roi = img_roi_blur.shape[:2]
                    h_temp, w_temp = img_monster_blur.shape[:2]

                    if h_temp > h_roi or w_temp > w_roi:
                        return []  # template bigger than roi, skip this matching

                    # Perform template matching
                    res = cv2.matchTemplate(img_roi_blur, img_monster_blur, cv2.TM_SQDIFF_NORMED)

                    # Apply soft threshold
                    h, w = img_monster.shape[:2]
                    for x, y, score in get_match_points(
                        res, match_threshold
                    ):
                        monsters.append({
                            "name": monster_name,
                            "position": (x + x0, y + y0),
                            "size": (h, w),
                            "score": score,
                        })
                elif self.cfg["monster_detect"]["mode"] == "grayscale":
                    img_monster_gray = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
                    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(
                            img_roi_gray,
                            img_monster_gray,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    h, w = img_monster.shape[:2]
                    for x, y, score in get_match_points(
                        res, match_threshold
                    ):
                        monsters.append({
                            "name": monster_name,
                            "position": (x + x0, y + y0),
                            "size": (h, w),
                            "score": score,
                    })
                elif self.cfg["monster_detect"]["mode"] == "color":
                    res = cv2.matchTemplate(
                            img_roi,
                            img_monster,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    h, w = img_monster.shape[:2]
                    for x, y, score in get_match_points(
                        res, match_threshold
                    ):
                        monsters.append({
                            "name": monster_name,
                            "position": (x + x0, y + y0),
                            "size": (h, w),
                            "score": score,
                    })
                else:
                    logger.error(f"Unexpected camera localization mode: {self.cfg['monster_detect']['mode']}")
                    return []

        # Apply Non-Maximum Suppression to monster detection
        monsters = nms(monsters, iou_threshold=0.4)
        monsters = suppress_nearby_same_class(
            monsters,
            center_distance=self.cfg["monster_detect"].get(
                "merge_center_distance", 18
            ),
        )

        # Detect monster via health bar
        if self.cfg["monster_detect"]["with_enemy_hp_bar"]:
            # Create color mask for Monsters' HP bar
            mask = cv2.inRange(img_roi,
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]),
                               np.array(self.cfg["monster_detect"]["hp_bar_color"]))

            # Find connected components (each cluster of green pixels)
            num_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, num_labels):  # skip background (label 0)
                x, y, w, h, area = stats[i]
                if area < 3:  # small noise filter
                    continue

                # Guess a monster bounding box
                y += 10
                x = max(0, x)
                y = max(0, y)
                w = 70
                h = min(img.shape[0] for _, imgs in self.monsters_info.items() for img, _ in imgs)

                monsters.append({
                    "name": "Health Bar",
                    "position": (x0 + x, y0 + y),
                    "size": (h, w),
                    "score": 1.0,
                })

        self.draw_monster_detections(monsters, top_left, bottom_right)

        return monsters

    def get_debug_monsters_in_range(
        self, top_left, bottom_right, score_thres=None
    ):
        """Detect map monsters by shape across the full camera image.

        Full-frame masked color SQDIFF strongly favors recurring terrain in
        Fire Land. Laplacian edge correlation instead ranks the distinctive
        sprite outlines above rocks and sky while remaining template-based.
        """
        if self.is_yolo_monster_detection():
            # ``score_thres`` is an edge-correlation threshold for the legacy
            # matcher. YOLO uses the calibrated monster_detect.confidence.
            return self.get_yolo_monsters_in_range(top_left, bottom_right)

        x0, y0 = top_left
        x1, y1 = bottom_right
        img_roi = self.img_frame[y0:y1, x0:x1]
        img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_roi_edges = cv2.convertScaleAbs(
            cv2.Laplacian(img_roi_gray, cv2.CV_32F)
        )
        default_threshold = (
            float(score_thres)
            if score_thres is not None
            else float(self.cfg.get("debug", {}).get("monster_diff_thres", 0.18))
        )
        debug_cfg = self.cfg.get("debug", {})
        threshold_by_monster = debug_cfg.get(
            "monster_diff_thres_by_monster", {}
        )
        radius = max(
            1,
            int(
                debug_cfg.get(
                    "local_peak_radius",
                    debug_cfg.get("local_min_radius", 9),
                )
            ),
        )
        top_k = max(1, int(debug_cfg.get("template_top_k", 1)))
        verify_color = bool(debug_cfg.get("verify_color", True))
        verify_candidates = max(
            top_k,
            int(debug_cfg.get("color_verify_candidates", 5)),
        )
        monster_cfg = self.cfg.get("monster_detect", {})
        color_threshold_default = float(
            monster_cfg.get("diff_thres", 1.0)
        )
        color_threshold_by_monster = monster_cfg.get(
            "diff_thres_by_monster", {}
        )
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)

        monsters = []
        for monster_name, monster_imgs in self.monsters_info.items():
            threshold = float(
                threshold_by_monster.get(
                    monster_name, default_threshold
                )
            )
            color_threshold = float(
                color_threshold_by_monster.get(
                    monster_name, color_threshold_default
                )
            )
            for img_monster, mask_monster in monster_imgs:
                h, w = img_monster.shape[:2]
                if h > img_roi_edges.shape[0] or w > img_roi_edges.shape[1]:
                    continue
                template_gray = cv2.cvtColor(
                    img_monster, cv2.COLOR_BGR2GRAY
                )
                template_edges = cv2.convertScaleAbs(
                    cv2.Laplacian(template_gray, cv2.CV_32F)
                )
                result = cv2.matchTemplate(
                    img_roi_edges,
                    template_edges,
                    cv2.TM_CCOEFF_NORMED,
                )
                result = np.nan_to_num(
                    result, nan=-1.0, posinf=-1.0, neginf=-1.0
                )
                local_max = cv2.dilate(result, kernel)
                ys, xs = np.where(
                    (result >= threshold)
                    & (result >= local_max - 1e-7)
                )
                candidates = sorted(
                    (
                        (int(x), int(y), float(result[y, x]))
                        for y, x in zip(ys, xs)
                    ),
                    key=lambda item: item[2],
                    reverse=True,
                )[:verify_candidates]

                verified = []
                for x, y, score in candidates:
                    color_score = None
                    if verify_color:
                        candidate_roi = img_roi[y:y+h, x:x+w]
                        if candidate_roi.shape[:2] != (h, w):
                            continue
                        color_result = cv2.matchTemplate(
                            candidate_roi,
                            img_monster,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster,
                        )
                        color_score = float(
                            np.nan_to_num(
                                color_result[0, 0],
                                nan=1.0,
                                posinf=1.0,
                                neginf=1.0,
                            )
                        )
                        if color_score > color_threshold:
                            continue

                    verified.append({
                        "name": monster_name,
                        "position": (x + x0, y + y0),
                        "size": (h, w),
                        # Keep SQDIFF-style ordering for the shared NMS helper.
                        "score": 1.0 - score,
                        "confidence": score,
                        "color_score": color_score,
                    })
                    if len(verified) >= top_k:
                        break
                monsters.extend(verified)

        monsters = nms(monsters, iou_threshold=0.4)
        self.draw_monster_detections(monsters, top_left, bottom_right)
        return monsters

    def draw_monster_detections(self, monsters, top_left, bottom_right):
        """Draw a monster search region and already-computed detections."""
        x0, y0 = top_left
        x1, y1 = bottom_right
        self._draw_debug_rectangle(
            (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Mob Detection Box"
        )

        # Draw monsters bounding box
        for monster in monsters:
            if monster["name"] == "Health Bar":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            visible_score = monster.get("confidence", monster["score"])
            label = str(round(visible_score, 2))
            if self.is_debug_mode() or "confidence" in monster:
                color_score = monster.get("color_score")
                if color_score is None:
                    label = f'{monster["name"]}: {visible_score:.2f}'
                else:
                    label = (
                        f'{monster["name"]}: E{visible_score:.2f} '
                        f'C{color_score:.2f}'
                    )
            elif "confidence" in monster:
                label = f'{monster["name"]}: {visible_score:.2f}'

            self._draw_debug_rectangle(
                monster["position"], monster["size"],
                color, label
            )

    def get_img_frame(self):
        '''
        get_img_frame
        '''
        # Get image and timestamp as one atomic snapshot. Clear the old binding
        # first so failed capture/preprocessing cannot reuse stale freshness.
        self._current_capture_frame_token = None
        get_snapshot = getattr(self.capture, "get_frame_snapshot", None)
        if callable(get_snapshot):
            self.frame, capture_frame_time = get_snapshot()
        else:
            # Compatibility adapter for custom capture implementations.
            self.frame = self.capture.get_frame()
            capture_frame_time = getattr(
                self.capture, "last_frame_time", None
            )
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        try:
            img_frame, geometry = preprocess_capture_frame(
                self.frame,
                self.cfg,
                window_title=getattr(self.capture, "window_title", ""),
                capture_profile=capture_profile_override(self.capture),
                skip_direct_size_check=getattr(self.args, "test_image", "") != "",
            )
        except (KeyError, TypeError, ValueError) as exc:
            # A persistent bad geometry should pause remote input, but it must
            # not flood the UI log at capture FPS.
            text = str(exc)
            if text != getattr(self, "_last_capture_error", None):
                logger.error(f"[capture] {text}")
                self._last_capture_error = text
            return

        self._last_capture_error = None
        try:
            capture_frame_time = float(capture_frame_time)
        except (TypeError, ValueError):
            capture_frame_time = 0.0
        if np.isfinite(capture_frame_time) and capture_frame_time > 0:
            self._current_capture_frame_token = (
                "capture", capture_frame_time
            )
        self._refresh_runtime_frame_config(geometry["output_size"])
        geometry_key = tuple(geometry.items())
        if geometry_key != getattr(self, "_last_capture_geometry", None):
            logger.info(
                "[capture] "
                f"profile={geometry['profile']} source={geometry['source_size']} "
                f"video_roi={geometry['video_roi']} "
                f"content={geometry['content_size']} "
                f"working={geometry['working_size']}"
            )
            self._last_capture_geometry = geometry_key

        # Legacy normalized PotPlayer mode needs a second native minimap crop
        # because downscaling can collapse its 2x2 player dot. Native mode
        # already returns this exact video raster, so never process it twice.
        if geometry["profile"] == "potplayer" and geometry.get(
                "normalized", True):
            x0, y0, x1, y1 = geometry["video_roi"]
            self.img_capture_content = self.frame[y0:y1, x0:x1]
        else:
            self.img_capture_content = None
        return img_frame

    def apply_saved_minimap_geometry(self):
        """Crop the minimap from route metadata without scanning the frame."""
        geometry = getattr(self, "minimap_geometry", None)
        if geometry is None:
            return None

        try:
            x, y, w, h = scale_minimap_rect(
                geometry,
                self.img_frame.shape[:2],
            )
            self.loc_minimap = (x, y)
            self.img_minimap_screen = self.img_frame[y:y+h, x:x+w]
            self.img_minimap_source = self.img_minimap_screen

            # A normalized working frame can erase the tiny player dot. Apply
            # the same saved rectangle to the native PotPlayer content instead
            # of running a second contour scan there.
            native_frame = getattr(self, "img_capture_content", None)
            if native_frame is not None:
                nx, ny, nw, nh = scale_minimap_rect(
                    geometry,
                    native_frame.shape[:2],
                )
                self.img_minimap_source = native_frame[
                    ny:ny+nh, nx:nx+nw
                ]
        except ValueError as exc:
            text = str(exc)
            if text != getattr(self, "_last_minimap_geometry_error", None):
                logger.error(f"Unable to apply saved minimap geometry: {text}")
                self._last_minimap_geometry_error = text
            return False

        self._last_minimap_geometry_error = None
        return True

    def is_player_stuck(self):
        """
        Checks whether the player is stuck (not moving)
        based on their global position on map.

        This function:
        - Compares the player's current position with their last known position
          tracked by the watchdog.
        - If the player has moved beyond a threshold (`watch_dog_range`),
          it resets the watchdog timer.
        - If the player hasn't moved and the elapsed time exceeds (`watch_dog_timeout`),
          it flags the player as stuck and resets the watchdog.

        Returns:
            bool: True if the player is stuck, False otherwise.
        """
        if getattr(self, "_rope_climb_active", False) or \
                getattr(self, "_portal_sweep_active", False) or \
                getattr(self, "_suppress_periodic_attack", False) or \
                bool(getattr(
                    getattr(self, "wz_navigation", None),
                    "platform_intentional_idle",
                    False,
                )):
            # Route-owned feedback loops and active combat decisions may
            # intentionally spend longer than the normal stuck timeout. Do
            # not replace their positioning, Up/search, or attack commands.
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = time.time()
            return False

        dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg["watchdog"]["range"]:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg["watchdog"]["timeout"]:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {round(dt, 2)} seconds.")
            return True
        return False

    def screenshot_img_frame(self):
        '''
        Save self.img_frame
        '''
        if self.img_frame is None:
            logger.error("[screenshot_img_frame] Failed, game window is not available")
        else:
            screenshot(self.img_frame, "img_frame")

        if self.img_frame_debug is None:
            pass
        else:
            screenshot(self.img_frame_debug, "img_frame_debug")

        if self.frame is None:
            pass
        else:
            screenshot(self.frame, "frame")

    def is_near_edge(self):
        '''
        Detects whether the player is near a teleport edge region

        This function:
        - Defines a rectangular search region around the player's current global location.
        - Scans for pixels matching a specific edge teleport color code within the region.
        - If matching pixels are found, it computes the average X position of those pixels.
        - Compares that average to the player's X position to determine whether the edge is on the left or right.

        Returns:
            str: One of:
                - "edge on left"
                - "edge on right"
                - "" (empty string if no edge is detected nearby)
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        h_trigger_box = self.cfg["edge_teleport"]["trigger_box_height"]
        w_trigger_box = self.cfg["edge_teleport"]["trigger_box_width"]
        x_min = max(0, x0 - w_trigger_box//2)
        x_max = min(w, x0 + w_trigger_box//2)
        y_min = max(0, y0 - h_trigger_box//2)
        y_max = min(h, y0 + h_trigger_box//2)

        # Debug: draw search box
        # draw_rectangle(
        #     self.img_route_debug,
        #     (x_min, y_min),
        #     (y_max - y_min, x_max - x_min),
        #     (0, 0, 255), "Edge Check", thickness=1, text_height=0.4
        # )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg["edge_teleport"]["color_code"], axis=2)
        coords = np.column_stack(np.where(mask))

        # No edge pixel
        if coords.size == 0:
            return ""

        # Calculate mean position of matching pixels
        mean_x = np.mean(coords[:, 1])

        # Compare to roi center
        if mean_x < x0:
            return "edge on left"
        else:
            return "edge on right"

    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        visual_scale = self.get_frame_visual_scale()
        text_y_interval = max(1, int(round(23 * visual_scale)))
        text_x, text_y_start = self.scale_debug_reference_point((10, 460))
        dt_screenshot = time.time() - self.kb.t_last_screenshot
        h, w = self.frame.shape[:2]
        text_list = [
            f"FPS: {self.fps}",
            f"State: {self.fsm.state.name}",
            f"Resolution: {h}x{w}, Ratio: {round(w/h, 2)}",
            f"Press 'F1' to {'pause' if self.kb.is_enable else 'start'} Bot",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}",
             "Press 'F12' to quit"]
        for idx, text in enumerate(text_list):
            self._draw_debug_text(
                text,
                (text_x, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw attack boxes only when a screen-space player location is
        # available. The active Hero locator may intentionally supply its last
        # confirmed position across detection misses.
        if not getattr(self, "screen_player_location_valid", False):
            self._draw_debug_text(
                "Player location unavailable",
                (500, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                reference_position=True,
            )
        elif self.cfg["bot"]["attack"] == "aoe_skill":
            x0, y0, x1, y1 = self.get_attack_range()
            self._draw_debug_rectangle(
                (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range"
            )
        elif self.cfg["bot"]["attack"] == "directional":
            x0, y0, x1, y1 = self.get_attack_range(is_left=True)
            self._draw_debug_rectangle(
                (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Left)"
            )
            x0, y0, x1, y1 = self.get_attack_range(is_left=False)
            self._draw_debug_rectangle(
                (x0, y0),
                (y1-y0, x1-x0),
                (0, 0, 255), "Attack Range(Right)"
            )
            directional_aoe_cfg = self.cfg.get("directional_aoe", {})
            if directional_aoe_cfg.get("enable", False):
                min_monsters = directional_aoe_cfg["min_monsters"]
                for is_left, direction in ((True, "Left"), (False, "Right")):
                    x0, y0, x1, y1 = self.get_attack_range(
                        is_left=is_left,
                        attack_type="directional_aoe",
                    )
                    self._draw_debug_rectangle(
                        (x0, y0),
                        (y1-y0, x1-x0),
                        (255, 0, 255),
                        f"Directional AoE({direction}, N>={min_monsters})",
                    )
            power_knockback_cfg = self.cfg.get("power_knockback", {})
            if power_knockback_cfg.get("enable", False):
                for is_left, direction in ((True, "Left"), (False, "Right")):
                    x0, y0, x1, y1 = self.get_attack_range(
                        is_left=is_left,
                        attack_type="power_knockback",
                    )
                    self._draw_debug_rectangle(
                        (x0, y0),
                        (y1-y0, x1-x0),
                        (0, 165, 255),
                        f"Power Knockback({direction})",
                    )

        # Draw minimap rectangle on img debug
        self._draw_debug_rectangle(
            self.loc_minimap,
            self.img_minimap_screen.shape[:2],
            (0, 0, 255), "minimap",thickness=2
        )

        # Modes without a route map stop after the game-window annotations.
        if self.cfg["bot"]["mode"] in ["patrol", "aux", "debug"]:
            return

        # The WZ platform state machine has no route canvas until Hero is
        # localized and its first temporary platform path is rendered.
        if self.img_route_debug is None:
            return

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Check if valid crop region
        if x1 <= x0 or y1 <= y0:
            return

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        if mini_map_crop.size == 0:
            return
        mini_map_crop = cv2.resize(mini_map_crop,
                                (max(1, int(round(mini_map_crop.shape[1]
                                                  * 3 * visual_scale))),
                                 max(1, int(round(mini_map_crop.shape[0]
                                                  * 3 * visual_scale)))),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        paste_margin = max(1, int(round(10 * visual_scale)))
        x_paste = max(0, w_frame - w_crop - paste_margin)
        y_paste = paste_margin
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=max(1, int(round(2 * visual_scale)))
        )

        # Draw HP/MP/EXP bar on debug window
        percent_bars = [self.health_monitor.hp_percent,
                      self.health_monitor.mp_percent,
                      self.health_monitor.exp_percent]
        for i, bar_name in enumerate(["HP", "MP", "EXP"]):
            # Print bar ratio on debug window
            self._draw_debug_text(
                f"{bar_name}: {percent_bars[i]:.1f}%",
                (250, 30 + 30*i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                reference_position=True,
            )
            # Draw bar on debug window
            x_s, y_s = self.scale_debug_reference_point((410, 13 + 30*i))
            x, y, w, h = self.health_monitor.loc_size_bars[i]
            self.img_frame_debug[y_s:y_s+h, x_s:x_s+w] = \
                self.img_frame[self.cfg["ui_coords"]["ui_y_start"]:, :][y:y+h, x:x+w]

        # Print command on screen
        self._draw_debug_text(
            f"Cmd: {self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}",
            (10, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            reference_position=True,
        )

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug", self.img_frame_debug)
        # Update FPS timer
        self.t_last_frame = time.time()

    def ensure_is_in_party(self):
        '''
        ensure_is_in_party
        '''
        remote = self.remote_keyboard_target()
        if remote and not self.remote_absolute_mouse_calibrated():
            logger.error(
                "[ensure_is_in_party] Remote absolute mouse is not calibrated"
            )
            return False
        if self.is_capture_card_source() and not remote:
            logger.info(
                "[ensure_is_in_party] Disabled for DirectShow capture: no "
                "local game window is available"
            )
            return False
        owns_remote_gate = False
        party_opened = False
        workflow_succeeded = False
        close_succeeded = True
        if remote:
            keyboard_controller = getattr(self, "kb", None)
            if keyboard_controller is None or not hasattr(
                    keyboard_controller, "suspend_automation_for_game_ui"):
                logger.error(
                    "[ensure_is_in_party] Exclusive remote UI input is "
                    "unavailable"
                )
                return False
            if not getattr(keyboard_controller, "game_ui_active", False):
                owns_remote_gate = bool(
                    keyboard_controller.suspend_automation_for_game_ui()
                )
                if not owns_remote_gate:
                    return False

        def press_party_key():
            key = self.cfg["key"]["party"]
            if remote:
                return bool(self.kb.press_game_ui_key(key))
            return bool(press_key(key))

        try:
            # Open the party window only after all ordinary gameplay producers
            # have been stopped on the remote machine.
            if not press_party_key():
                return False
            party_opened = True
            time.sleep(0.5)

            self.img_frame = self.get_img_frame()
            loc_enable, score_enable, _ = find_pattern_sqdiff(
                self.img_frame, self.img_create_party_enable
            )

            lang = self.cfg["system"]["language"]
            thres = self.cfg['party_red_bar'][
                f'create_party_button_{lang}_thres'
            ]
            if score_enable < thres:
                logger.info(
                    "[ensure_is_in_party] Find party enable button("
                    f"{round(score_enable, 2)})"
                )
                h, w = self.img_create_party_enable.shape[:2]
                point = (
                    loc_enable[0] + w // 2,
                    loc_enable[1] + h // 2,
                )
                if not remote:
                    point = (
                        point[0],
                        point[1] + int(self.cfg["game_window"].get(
                            "title_bar_height", 0
                        )),
                    )
                if not self.click_game_ui(
                    point,
                    "ensure_is_in_party",
                ):
                    return False
            else:
                logger.info(
                    "[ensure_is_in_party] Cannot find create party button. "
                    "Maybe player already in party."
                )
            workflow_succeeded = True
        finally:
            if party_opened:
                close_succeeded = press_party_key()
            if not close_succeeded and hasattr(self.kb, "disable"):
                # Never resume combat while the party dialog may still own
                # keyboard focus after a failed close command.
                self.kb.disable()
            elif remote and owns_remote_gate and not workflow_succeeded and \
                    hasattr(self.kb, "disable"):
                # A standalone remote party workflow must also fail closed if
                # its open/click stage was rejected, even when the close TAP
                # succeeded. A nested channel workflow is guarded by its owner.
                self.kb.disable()
            if owns_remote_gate:
                self.kb.resume_automation_after_game_ui()
        return workflow_succeeded and close_succeeded

    def _wait_for_channel_gameplay_ready(self):
        """Wait for consecutive fresh minimap/player evidence after login."""
        channel_cfg = self.cfg.get("channel_change", {})

        def positive_number(name, default, minimum):
            try:
                value = float(channel_cfg.get(name, default))
            except (TypeError, ValueError, OverflowError):
                value = float(default)
            if not math.isfinite(value):
                value = float(default)
            return max(float(minimum), value)

        timeout = positive_number("game_ready_timeout", 60.0, 1.0)
        poll_interval = positive_number(
            "game_ready_poll_interval", 1.0, 0.05
        )
        required_frames = max(
            1,
            int(round(positive_number(
                "game_ready_confirm_frames", 2.0, 1.0
            ))),
        )
        deadline = time.monotonic() + timeout
        ready_count = 0
        last_frame_token = object()

        while not self.is_terminated:
            if time.monotonic() >= deadline:
                logger.error(
                    "[channel_change] Timed out waiting for gameplay evidence"
                )
                return False
            try:
                self.img_frame = self.get_img_frame()
                frame_token = getattr(
                    self, "_current_capture_frame_token", None
                )
                if frame_token is None:
                    # A cached frame without a capture timestamp cannot prove
                    # that gameplay survived another independent observation.
                    ready_count = 0
                    time.sleep(poll_interval)
                    continue
                if frame_token == last_frame_token:
                    time.sleep(poll_interval)
                    continue
                last_frame_token = frame_token

                if self._auto_relogin_current_gameplay_evidence() is None:
                    ready_count = 0
                else:
                    ready_count += 1
                    if ready_count >= required_frames:
                        logger.info(
                            "[channel_change] Gameplay evidence confirmed"
                        )
                        return True
            except Exception as exc:
                ready_count = 0
                logger.warning(
                    "[channel_change] Gameplay readiness check failed: "
                    f"{exc}"
                )
            time.sleep(poll_interval)

        logger.info("[channel_change] Cancelled during shutdown")
        return False

    def channel_change(self):
        '''
        channel_change
        '''
        logger.info("[channel_change] Start")

        remote = self.remote_keyboard_target()
        if remote and not self.remote_absolute_mouse_calibrated():
            logger.error(
                "[channel_change] Remote absolute mouse is not calibrated"
            )
            return False
        if self.is_capture_card_source() and not remote:
            logger.error(
                "[channel_change] Disabled for DirectShow capture: no local "
                "game window is available"
            )
            return False
        owns_remote_gate = False
        workflow_succeeded = False
        if remote:
            keyboard_controller = getattr(self, "kb", None)
            if keyboard_controller is None or not hasattr(
                    keyboard_controller, "suspend_automation_for_game_ui"):
                logger.error(
                    "[channel_change] Exclusive remote UI input is unavailable"
                )
                return False
            owns_remote_gate = bool(
                keyboard_controller.suspend_automation_for_game_ui()
            )
            if not owns_remote_gate:
                return False

        try:
            ui_coords = self.cfg["ui_coords"]
            for name, delay in (
                    ("menu", 1),
                    ("channel", 1),
                    ("random_channel", 1),
                    ("random_channel_confirm", 1)):
                point = self._configured_remote_ui_capture_point(name) \
                    if remote else ui_coords[name]
                if not self.click_game_ui(
                        point, f"channel_change_{name}"):
                    return False
                time.sleep(delay)

            try:
                ui_timeout = float(self.cfg.get(
                    "channel_change", {}
                ).get("ui_timeout", 60.0))
            except (TypeError, ValueError, OverflowError):
                ui_timeout = 60.0
            ui_deadline = time.monotonic() + max(1.0, ui_timeout)
            loc_login_button = None
            while loc_login_button is None and not self.is_terminated:
                if time.monotonic() >= ui_deadline:
                    logger.error(
                        "[channel_change] Timed out waiting for login button"
                    )
                    return False
                try:
                    self.img_frame = self.get_img_frame()
                    loc_login_button = self.get_login_button_location()
                    if loc_login_button is None:
                        logger.info("Waiting for login button to show up...")
                except Exception as exc:
                    logger.warning(
                        "Exception occurred while waiting for login button: "
                        f"{exc}"
                    )
                    if not remote and not is_mac() and self.cfg[
                            "game_window"].get("auto_resize", True):
                        resize_window(
                            self.capture.window_title,
                            width=self.cfg["game_window"].get(
                                "resize_width", 1296
                            ),
                            height=self.cfg["game_window"].get(
                                "resize_height", 759
                            ),
                        )
                    logger.info("Retrying login button detection...")
                time.sleep(3)

            if self.is_terminated:
                logger.info("[channel_change] Cancelled during shutdown")
                return False

            logger.info(f"login_button button found: {loc_login_button}")
            time.sleep(3)
            login_click_point = loc_login_button
            if not remote:
                login_click_point = (
                    int(loc_login_button[0]),
                    int(loc_login_button[1]) + int(
                        self.cfg["game_window"].get(
                            "title_bar_height", 0
                        )
                    ),
                )
            if not self.click_game_ui(
                    login_click_point, "channel_change_login"):
                return False
            time.sleep(2)
            select_character_point = \
                self._configured_remote_ui_capture_point(
                    "select_character"
                ) if remote else ui_coords["select_character"]
            if not self.click_game_ui(
                    select_character_point,
                    "channel_change_select_character"):
                return False
            time.sleep(5)

            if remote:
                # Keep the exclusive remote gate across loading and party UI
                # cleanup. Reopening it here lets background buffs race with
                # the login screen before ensure_is_in_party owns the gate.
                if not self._wait_for_channel_gameplay_ready():
                    return False
                self.kb.set_command("none none none")
                self.kb.release_all_key()
                if not self.ensure_is_in_party():
                    logger.error(
                        "[channel_change] Party UI cleanup failed; gameplay "
                        "remains paused"
                    )
                    return False

            workflow_succeeded = True
        finally:
            if owns_remote_gate:
                if not workflow_succeeded:
                    # A partially completed menu workflow must not resume
                    # movement or attacks into an unknown screen.
                    self.kb.disable()
                self.kb.resume_automation_after_game_ui()

        if not remote:
            self.kb.enable()
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            if not self.ensure_is_in_party():
                logger.error(
                    "[channel_change] Party UI cleanup failed; gameplay "
                    "remains paused"
                )
                self.kb.disable()
                return False

        self.fsm.set_init_state("hunting")
        self.t_last_attack = time.time()
        return True

    def _stop_components(self):
        """Idempotently stop all producers and the remote HID session."""
        if not hasattr(self, "_shutdown_lock"):
            # Keep lightweight test doubles created with __new__ compatible.
            self._shutdown_lock = threading.Lock()
            self._components_stopped = False

        with self._shutdown_lock:
            if self._components_stopped:
                return
            self._components_stopped = True

            # Signal every producer first, then immediately close the input
            # gate and release keys before waiting for any worker to join.
            if self.health_monitor is not None:
                try:
                    if hasattr(self.health_monitor, "request_stop"):
                        self.health_monitor.request_stop()
                    else:
                        self.health_monitor.is_terminated = True
                except Exception as exc:
                    logger.error(f"[shutdown] Failed to signal health monitor: {exc}")
            if self.kb is not None:
                try:
                    if hasattr(self.kb, "disable"):
                        self.kb.disable()
                    else:
                        self.kb.is_terminated = True
                        if hasattr(self.kb, "release_all_key"):
                            self.kb.release_all_key()
                except Exception as exc:
                    logger.error(f"[shutdown] Failed to release keyboard input: {exc}")

            if self.health_monitor is not None:
                try:
                    self.health_monitor.stop()
                except Exception as exc:
                    logger.error(f"[shutdown] Failed to stop health monitor: {exc}")
            if self.kb is not None and hasattr(self.kb, "stop"):
                try:
                    self.kb.stop()
                except Exception as exc:
                    logger.error(f"[shutdown] Failed to stop keyboard input: {exc}")
            if self.capture is not None:
                try:
                    self.capture.stop()
                except Exception as exc:
                    logger.error(f"[shutdown] Failed to stop capture: {exc}")

    def terminate_threads(self):
        '''
        terminate all threads
        '''
        self.is_terminated = True
        self._stop_components()

        # Wait for the main loop unless termination was requested by that loop.
        thread = self.thread_auto_bot
        if thread is not None and thread.is_alive() and \
            thread is not threading.current_thread():
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning("[terminate_threads] Main loop did not stop within 5 seconds")
            else:
                self.thread_auto_bot = None

        logger.info("[terminate_threads] Terminated all threads")

    def get_attack_direction(self, monster_left, monster_right):
        '''
        get_attack_direction
        '''
        # Compute distance for left
        distance_left = float('inf')
        if monster_left is not None:
            center_left = detection_center(monster_left)
            distance_left = abs(center_left[0] - self.loc_player[0]) + \
                            abs(center_left[1] - self.loc_player[1])
        # Compute distance for right
        distance_right = float('inf')
        if monster_right is not None:
            center_right = detection_center(monster_right)
            distance_right = abs(center_right[0] - self.loc_player[0]) + \
                            abs(center_right[1] - self.loc_player[1])
        # Choose attack direction and nearest monster
        attack_direction = None
        # nearest_monster = None

        # Additional validation: check if monster is actually on the correct side
        def is_monster_on_correct_side(monster, direction):
            if monster is None:
                return False
            monster_center_x = detection_center(monster)[0]
            player_x = self.loc_player[0]

            if direction == "left":
                return monster_center_x < player_x  # Monster should be left of player
            else:  # direction == "right"
                return monster_center_x > player_x  # Monster should be right of player

        # Only choose direction if there's a clear winner and monster is on correct side
        if monster_left is not None and monster_right is None and \
            is_monster_on_correct_side(monster_left, "left"):
            attack_direction = "left"
            # nearest_monster = monster_left
        elif monster_right is not None and monster_left is None and \
            is_monster_on_correct_side(monster_right, "right"):
            attack_direction = "right"
            # nearest_monster = monster_right
        elif monster_left is not None and monster_right is not None:
            # Both sides have monsters, check distance and side validation
            left_valid = is_monster_on_correct_side(monster_left, "left")
            right_valid = is_monster_on_correct_side(monster_right, "right")

            if left_valid and not right_valid:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif right_valid and not left_valid:
                attack_direction = "right"
                # nearest_monster = monster_right
            elif left_valid and right_valid and distance_left < distance_right - 50:
                attack_direction = "left"
                # nearest_monster = monster_left
            elif left_valid and right_valid and distance_right < distance_left - 50:
                attack_direction = "right"
                # nearest_monster = monster_right
            # If both valid but distances too close, don't attack to avoid confusion

        # Debug attack direction selection
        if monster_left is not None or monster_right is not None:
            left_side_ok = is_monster_on_correct_side(monster_left, "left") if monster_left else False
            right_side_ok = is_monster_on_correct_side(monster_right, "right") if monster_right else False
            debug_text = f"L:{distance_left:.0f}({left_side_ok}) R:{distance_right:.0f}({right_side_ok}) Dir:{attack_direction}"
            self._draw_debug_text(
                debug_text,
                (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
                reference_position=True,
            )
        return attack_direction

    def is_need_change_channel(self, loc_other_players):
        '''
        is_need_change_channel
        '''
        # Calculate center value
        xs = [x for (x, _) in loc_other_players]
        ys = [y for (_, y) in loc_other_players]
        if len(xs) == 0 or len(ys) == 0:
            return False
        center_x, center_y = (np.mean(xs), np.mean(ys))
        if np.isnan(center_x) or np.isnan(center_y):
            return False
        center = (int(np.mean(xs)), int(np.mean(ys)))
        #logger.info(f"[is_need_change_channel] Center of mass = {center}")

        # Change channel
        mode = self.cfg["channel_change"]["mode"]
        if mode == "true":
            logger.warning("[is_need_change_channel] Player detected, immediately change channel.")
            return True
        elif mode == "pixel":
            if self.red_dot_center_prev is None:
                self.red_dot_center_prev = center
            else:
                dx = abs(center[0] - self.red_dot_center_prev[0])
                dy = abs(center[1] - self.red_dot_center_prev[1])
                total = dx + dy
                logger.debug(f"[is_need_change_channel] Movement dx={dx}, dy={dy}, total={total}")
                thres = self.cfg["channel_change"]["other_player_move_thres"]
                if total > thres:
                    logger.warning(f"Other player movement > {thres} pixel detected. "
                                "Trigger channel change.")
                    return True
        else:
            logger.error(f"[is_need_change_channel] Unsupported mode: {mode}")

        return False

    def is_time_to_change_channel(self):
        '''
        is_time_to_change_channel
        '''
        if not self.cfg["scheduled_channel_switching"]["enable"]:
            return False
        dt = time.time() - self.t_to_change_channel
        if dt > self.cfg["scheduled_channel_switching"]["interval_seconds"]:
            self.t_to_change_channel = time.time()
            return True
        return False

    def get_login_button_location(self):
        '''
        get_login_button_location
        '''
        if getattr(self, "img_frame", None) is None or \
                getattr(self, "img_login_button", None) is None:
            return None

        # Extract the region where the login button should appear
        x0, y0 = self.cfg["ui_coords"]["login_button_top_left"]
        x1, y1 = self.cfg["ui_coords"]["login_button_bottom_right"]
        frame_h, frame_w = self.img_frame.shape[:2]
        x0 = max(0, min(int(x0), frame_w))
        x1 = max(0, min(int(x1), frame_w))
        y0 = max(0, min(int(y0), frame_h))
        y1 = max(0, min(int(y1), frame_h))
        template_h, template_w = self.img_login_button.shape[:2]
        if x1 - x0 < template_w or y1 - y0 < template_h:
            return None
        img_roi = self.img_frame[y0:y1, x0:x1]

        # Draw rectange on debug image
        self._draw_debug_rectangle(
            (x0, y0),
            (y1-y0, x1-x0),
            (0, 255, 0),
            "login_button box",
        )

        # Find the 'login' button
        loc, score, _ = find_pattern_sqdiff(
                        img_roi, self.img_login_button)
        if score < self.cfg["ui_coords"]["login_button_thres"]:
            h, w = self.img_login_button.shape[:2]
            logger.debug(
                "[get_login_button_location] Found login button with "
                f"score({score})"
            )
            return (x0 + loc[0] + w // 2,
                    y0 + loc[1] + h // 2)
        else:
            return None

    def update_cmd_by_route(self):
        # Rebuild movement from this frame. If no route matches, never renew a
        # stale movement key from the prior frame on the remote ESP32.
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        self._rope_timed_mount_candidate = None
        self._clear_wz_route_jump_atomic_if_departed()
        self._clear_rope_climb_locks_if_departed()
        self._stationary_jump_proximity_active = False
        self._clear_failed_portal_if_departed()
        self._clear_route_loop_goal_if_departed()

        # A generated rope run-up remains active after the Hero leaves the
        # endpoint marker's acquisition window. It owns input until progress,
        # retries, or the endpoint height resolves the attempt.
        if getattr(self, "_rope_climb_combat_deferred", False):
            # Monster detection from the previous frame found an attackable
            # target. Keep every pre-mount phase frozen even after the normal
            # travel combat budget expires. Mob detection runs later in this
            # frame and explicitly releases the defer as soon as the attack
            # range is clear, so no route-side timeout is needed here.
            return
        if self._update_active_rope_climb():
            return

        # Once entered, a portal region owns input until the minimap confirms
        # a sufficiently large displacement.
        if self._update_active_portal_sweep():
            return

        navigator = getattr(self, "wz_navigation", None)
        platform_fsm = bool(getattr(
            navigator, "platform_state_machine_active", False
        ))
        if navigator is not None and navigator.jump_active:
            # Route pixels describe surfaces, not the airborne arc. Preserve
            # the takeoff direction until raw WZ-space motion confirms a
            # landing, and do not switch routes or retrigger Jump in mid-air.
            move_x = getattr(self, "_wz_jump_move_x", "none")
            self.cmd_move_x = move_x if move_x in {"left", "right"} else "none"
            return
        if platform_fsm and not self.img_routes:
            # DWELLING/LOCALIZING is an explicit stationary platform state,
            # not a missing painted route that should trigger route scanning.
            return

        cleared_route_index = getattr(
            self, "_wz_combat_checkpoint_cleared_route_index", None
        )
        checkpoint_cleared = (
            cleared_route_index is not None
            and int(getattr(self, "idx_routes", -1))
            == int(cleared_route_index)
        )
        checkpoint_active = self._active_wz_combat_checkpoint() is not None
        checkpoint_reached = False
        checkpoint_detector = getattr(
            navigator, "combat_checkpoint_reached", None
        )
        edge_jump_braking_distance = self._wz_edge_jump_braking_distance()
        if callable(checkpoint_detector):
            checkpoint_reached = checkpoint_detector(
                self.idx_routes,
                self.loc_player_global,
                braking_distance=edge_jump_braking_distance,
            )
        if not checkpoint_cleared and (
                checkpoint_active or checkpoint_reached):
            # A compound WALK -> JUMP route has no goal at its internal combat
            # checkpoint. Enter the existing attack-first state before route
            # color lookup can acquire the nearby jump trigger.
            self.cmd_action = "goal"
            return

        if self._prepare_wz_timed_directional_jump():
            return

        if self._align_compound_wz_edge_jump(
                edge_jump_braking_distance):
            return

        if navigator is not None and navigator.walk_target_crossed(
                self.idx_routes,
                self.loc_player_global,
                tolerance=int(self.cfg.get("route", {}).get(
                    "search_range", 10
                )),
        ):
            # The Hero marker can advance several pixels between capture
            # frames. Treat a small direction-aware overshoot as arrival so a
            # WALK leg cannot lose its goal marker and trigger route scanning.
            self.cmd_action = "goal"
            return

        # Get color code from the active route.
        color_code, color_code_up_down = self.get_nearest_color_code()

        # A failed rope guide deliberately hides that guide until the Hero
        # leaves its local failure area.  Do not mistake that intentional
        # lookup gap for a disconnected route and switch to an overlapping
        # route (for example route2's opposite-direction ground stroke).
        # Route changes after a rope miss must still happen only through an
        # explicit goal or after the failure lock has been cleared by travel.
        failed_rope_key = getattr(self, "_rope_climb_failed_key", None)
        failed_rope_blocks_recovery = bool(
            failed_rope_key is not None
            and int(failed_rope_key[0]) == int(self.idx_routes)
        )
        failed_rope_recovery_ready = False
        if failed_rope_blocks_recovery:
            failed_at = getattr(self, "_rope_climb_failed_at", None)
            rearm_delay = max(
                0.0,
                float(self.cfg.get("route", {}).get(
                    "rope_climb_failure_rearm_delay", 1.5
                )),
            )
            failed_rope_recovery_ready = bool(
                failed_at is not None
                and time.monotonic() - float(failed_at) >= rearm_delay
            )

        if not color_code and not color_code_up_down and platform_fsm:
            if failed_rope_blocks_recovery and not failed_rope_recovery_ready:
                self._apply_ladder_route_hold()
                return
            replan = getattr(navigator, "request_platform_replan", None)
            if callable(replan):
                replan(
                    f"no local action on temporary leg "
                    f"{int(getattr(self, 'idx_routes', 0)) + 1}"
                )
            return

        # The next route may not pass through the player's current position
        # when the recorded route set is incomplete. Search each later route
        # once (with wraparound) and immediately execute the first local
        # action that can be found.
        if not color_code and not color_code_up_down and self.img_routes and \
                (
                    not failed_rope_blocks_recovery
                    or failed_rope_recovery_ready
                ):
            original_idx = self.idx_routes
            route_count = len(self.img_routes)
            for step in range(1, route_count):
                candidate_idx = (original_idx + step) % route_count
                self.idx_routes = candidate_idx
                self.img_route = self.img_routes[candidate_idx]
                if self.is_show_debug_window:
                    self.img_route_debug = cv2.cvtColor(
                        self.img_route, cv2.COLOR_RGB2BGR
                    )

                # A far-away endpoint on another route must not attract the
                # Hero away from the explicitly active route. Rope markers
                # are considered only on the initially selected route.
                color_code, color_code_up_down = \
                    self.get_nearest_color_code(include_rope_climb=False)
                if failed_rope_blocks_recovery:
                    # A nearby current-route guide would already have been
                    # reacquired above.  Once its cooldown expires and it is
                    # no longer nearby, allow escape to another route only
                    # when Hero is genuinely on that route's stroke.  Merely
                    # finding a color somewhere inside the normal 10-pixel
                    # search box caused the original premature route2 switch.
                    recovery_distance = max(
                        0,
                        int(self.cfg.get("route", {}).get(
                            "rope_climb_failed_route_recovery_distance", 2
                        )),
                    )
                    if color_code is not None and int(
                            color_code.get("distance", recovery_distance + 1)
                    ) > recovery_distance:
                        color_code = None
                    if color_code_up_down is not None and int(
                            color_code_up_down.get(
                                "distance", recovery_distance + 1
                            )
                    ) > recovery_distance:
                        color_code_up_down = None
                if color_code or color_code_up_down:
                    if failed_rope_blocks_recovery:
                        logger.info(
                            "[route] Failed rope guide is no longer nearby; "
                            f"Hero overlaps route{candidate_idx + 1}, "
                            "release the route lock"
                        )
                    else:
                        logger.info(
                            "[route] No action near player on "
                            f"route{original_idx + 1}; recovered with "
                            f"route{candidate_idx + 1}"
                        )
                    break
            else:
                # No route describes this location. Keep a deterministic
                # route selected. While the climbing pose is still current,
                # retain its latched Up/Down command across this lookup miss.
                self.idx_routes = original_idx
                self.img_route = self.img_routes[original_idx]
                self._reset_stationary_jump_proximity()
                if self.is_show_debug_window:
                    self.img_route_debug = cv2.cvtColor(
                        self.img_route, cv2.COLOR_RGB2BGR
                    )
                self._apply_ladder_route_hold()
                return

        if not color_code and not color_code_up_down and \
                failed_rope_blocks_recovery:
            self._apply_ladder_route_hold()
            return

        if color_code and color_code.get("rope_climb", False):
            self._reset_stationary_jump_proximity()
            self._reset_portal_sweep()
            self._start_rope_climb(color_code)
            return

        if color_code and color_code.get("portal_sweep", False):
            self._reset_stationary_jump_proximity()
            self._start_portal_sweep(color_code)
            return

        if color_code and color_code.get(
                "stationary_jump_proximity", False):
            # Entering the configured proximity is sufficient. Do not chase
            # the marker with alternating one-pixel left/right corrections.
            self._stationary_jump_proximity_active = True
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = \
                color_code["command"].split()
            self._reserve_wz_route_jump_atomic(color_code["command"])
            return

        self._reset_stationary_jump_proximity()

        vertical_destination_reached = bool(
            color_code
            and color_code_up_down
            and self._has_recent_ladder_route_exit_confirmation()
            and self._is_ladder_route_destination_reached(
                color_code_up_down
            )
        )
        if vertical_destination_reached:
            # The visual ground transition is the physical arrival signal;
            # route geometry only selects the adjacent platform afterwards.
            logger.info(
                "[ladder] Ground state confirmed at route endpoint; "
                "platform route resumed"
            )
            self._reset_ladder_route_hold()
            self._ladder_route_exit_confirmed_at = None
            color_code_up_down = None

        vertical_route_would_own_input = bool(
            color_code_up_down
            and not (color_code and color_code.get("exact_action", False))
            and (
                not color_code
                or color_code_up_down["distance"] <= color_code["distance"]
            )
        )
        if vertical_route_would_own_input:
            alignment_move_x = self._get_ladder_route_alignment_move_x(
                color_code_up_down
            )
            if alignment_move_x is not None:
                # Do not press Up/Down until the minimap centroid is actually
                # aligned with the intended vertical component.  In
                # particular this keeps a nearby portal from consuming an
                # early Up while Hero approaches a rope.
                self._reset_ladder_route_hold()
                self.cmd_move_x = alignment_move_x
                return

        # Use color_code and color_code_up_down to complement each other
        # To prevent character stuck at the end of ladder, we use two color color pixels
        # and let them complement with each other, to ensure smoothy ladder climbing
        if color_code and color_code.get("exact_action", False):
            # An exact point action owns the complete recorded command. In
            # particular, do not complement a stationary jump with a nearby
            # ladder up/down color after its exact trigger condition was met.
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = \
                color_code["command"].split()
            self._reserve_wz_route_jump_atomic(color_code["command"])
        elif color_code and color_code_up_down:
            if color_code["distance"] < color_code_up_down["distance"]:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
                _, cmd, _ = color_code_up_down["command"].split()
                if self.cmd_move_y == "none" and self.is_on_ladder:
                    self.cmd_move_y = cmd # only complement cmd_move_y when player is on ladder
            else:
                self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()
                cmd, _, _ = color_code["command"].split()
                if self.cmd_move_x == "none" and self.is_on_ladder:
                    self.cmd_move_x = cmd # only complement cmd_move_x when player is on ladder
        elif color_code:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code["command"].split()
        elif color_code_up_down:
            self.cmd_move_x, self.cmd_move_y, self.cmd_action = color_code_up_down["command"].split()

        # A horizontal route becomes closest near the top of a ladder. Do not
        # let that proximity release Up/Down while the visual pose still says
        # the character is climbing. Exact point actions keep their existing
        # ownership because reaching their center is itself an arrival signal.
        self._apply_ladder_route_hold(
            color_code_up_down,
            allow_restore=not bool(
                color_code and color_code.get("exact_action", False)
            ),
        )

        # teleport away from edge to avoid falling off cliff
        if self.is_near_edge() and \
            time.time() - self.t_last_teleport > self.cfg["teleport"]["cooldown"]:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # Use teleport while walking
        if self.cfg['teleport']['is_use_teleport_to_walk'] and \
            time.time() - self.t_last_teleport > self.cfg['teleport']['cooldown']:
            self.cmd_action = "teleport"
            self.t_last_teleport = time.time() # update timer

        # replace teleport to jump if user doesn't set teleport key
        if self.cfg["key"]["teleport"] == "" and self.cmd_action == "teleport":
            self.cmd_action = "jump"

    def _uses_ranged_safe_patrol(self):
        navigator = getattr(self, "wz_navigation", None)
        return getattr(navigator, "patrol_strategy", None) == \
            "ranged_safe_platforms"

    def _scheduled_buff_allowed(self):
        """Block Buffs while a WZ jump or rope launch owns timing."""
        if not getattr(self, "_wz_navigation_enabled", False):
            return True
        if self._wz_timed_jump_owns_input() or bool(getattr(
                self, "_stationary_jump_proximity_active", False)):
            return False
        rope_state = getattr(self, "_rope_climb_state", None)
        rope_pre_mount = bool(
            getattr(self, "_rope_climb_active", False)
            and isinstance(rope_state, dict)
            and rope_state.get("phase") in {
                "position", "settle", "stationary_settle",
                "running_approach", "timed_mount", "mount_request",
            }
        )
        if rope_pre_mount or isinstance(getattr(
                self, "_rope_timed_mount_candidate", None), dict) or \
                self._rope_timed_mount_owns_input():
            return False
        navigator = getattr(self, "wz_navigation", None)
        if navigator is None or not getattr(navigator, "active", False) or \
                getattr(navigator, "jump_active", False):
            return False
        return True

    def _hold_for_scheduled_buff_recovery(self):
        """Freeze route state while the keyboard finishes a Buff animation."""
        keyboard_controller = getattr(self, "kb", None)
        if keyboard_controller is None or not \
                keyboard_controller.is_buff_recovery_active():
            return False
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        keyboard_controller.set_command("none none none")
        return True

    def _reset_wz_combat_checkpoint(self):
        self._cancel_wz_timed_directional_jump()
        self._wz_combat_checkpoint_route_index = None
        self._wz_combat_checkpoint_clear_since = None
        self._wz_combat_checkpoint_cleared_route_index = None
        self._wz_combat_checkpoint_jump_alignment_direction = None
        self._wz_route_jump_atomic_pending = False
        self._wz_route_jump_atomic_route_index = None
        self._wz_edge_jump_alignment_route_index = None

    def _current_wz_combat_checkpoint(self):
        navigator = getattr(self, "wz_navigation", None)
        route_legs = getattr(navigator, "route_legs", ())
        route_index = int(getattr(self, "idx_routes", -1))
        if not 0 <= route_index < len(route_legs):
            return None
        return getattr(route_legs[route_index], "combat_checkpoint", None)

    def _clears_monsters_before_moving(self):
        navigator = getattr(self, "wz_navigation", None)
        if bool(getattr(
                navigator, "platform_state_machine_active", False)):
            return bool(getattr(
                navigator, "platform_clear_before_move", True
            ))
        plan = getattr(navigator, "plan", None)
        return bool(getattr(plan, "combat_checkpoints", ()))

    def _active_wz_combat_checkpoint(self):
        route_index = int(getattr(self, "idx_routes", -1))
        if route_index != getattr(
                self, "_wz_combat_checkpoint_route_index", None):
            return None
        return self._current_wz_combat_checkpoint()

    def _wz_checkpoint_jump_alignment(self):
        """Keep a cleared checkpoint inside the next jump's safe launch band."""
        navigator = getattr(self, "wz_navigation", None)
        route_legs = tuple(getattr(navigator, "route_legs", ()))
        route_index = int(getattr(self, "idx_routes", -1))
        if not 0 <= route_index < len(route_legs):
            return True, None

        current_leg = route_legs[route_index]
        if getattr(current_leg, "combat_checkpoint_position", None) is not \
                None and getattr(current_leg, "jump_source", None) is not None:
            # The approach and jump now share one route. The checkpoint point
            # is deliberately earlier than the measured launch band; after it
            # clears, normal route colors carry Hero into that band.
            return True, None
        next_leg = route_legs[(route_index + 1) % len(route_legs)]
        next_action = getattr(next_leg, "action", None)
        if getattr(next_action, "value", next_action) != "JUMP":
            return True, None

        jump_source = getattr(next_leg, "source", None)
        jump_target = getattr(next_leg, "target", None)
        checkpoint_target = getattr(current_leg, "target", None)
        if not all(
                isinstance(point, (tuple, list)) and len(point) >= 2
                for point in (
                    jump_source, jump_target, checkpoint_target
                )):
            return True, None
        if int(jump_source[0]) == int(jump_target[0]):
            return True, None

        player = getattr(self, "loc_player_global", None)
        if not isinstance(player, (tuple, list)) or len(player) < 2:
            # Do not clear into an edge jump without a usable minimap fix.
            return False, None

        target_x = int(checkpoint_target[0])
        player_x = int(player[0])
        delta_x = target_x - player_x
        if abs(delta_x) <= 1:
            return True, None
        return False, "right" if delta_x > 0 else "left"

    def _hold_at_wz_combat_checkpoint(self):
        checkpoint = self._current_wz_combat_checkpoint()
        if checkpoint is None:
            return False
        route_index = int(self.idx_routes)
        if route_index == getattr(
                self, "_wz_combat_checkpoint_cleared_route_index", None):
            return False
        if route_index != getattr(
                self, "_wz_combat_checkpoint_route_index", None):
            self._wz_combat_checkpoint_route_index = route_index
            self._wz_combat_checkpoint_clear_since = None
            self._wz_combat_checkpoint_cleared_route_index = None
            logger.info(
                "[combat-checkpoint] Hold movement at "
                f"{checkpoint.label}; facing={checkpoint.facing}"
            )
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        jump_aligned, alignment_direction = \
            self._wz_checkpoint_jump_alignment()
        previous_direction = getattr(
            self,
            "_wz_combat_checkpoint_jump_alignment_direction",
            None,
        )
        if alignment_direction is not None:
            self.cmd_action = f"jump_align_{alignment_direction}"
            if alignment_direction != previous_direction:
                logger.info(
                    "[combat-checkpoint] Align "
                    f"{checkpoint.label} {alignment_direction} before "
                    "horizontal jump"
                )
        self._wz_combat_checkpoint_jump_alignment_direction = (
            alignment_direction if not jump_aligned else None
        )
        self._suppress_periodic_attack = True
        return True

    def _observe_wz_combat_checkpoint(self, has_attackable_monster):
        checkpoint = self._active_wz_combat_checkpoint()
        if checkpoint is None:
            return
        self._suppress_periodic_attack = True
        if has_attackable_monster:
            self._wz_combat_checkpoint_clear_since = None
            return

        jump_aligned, _ = self._wz_checkpoint_jump_alignment()
        if not jump_aligned:
            self._wz_combat_checkpoint_clear_since = None
            return

        now = time.monotonic()
        clear_since = getattr(
            self, "_wz_combat_checkpoint_clear_since", None
        )
        if clear_since is None:
            self._wz_combat_checkpoint_clear_since = now
            return
        try:
            quiet_seconds = float(self.cfg.get("wz_navigation", {}).get(
                "combat_clear_quiet_seconds", 0.8
            ))
        except (TypeError, ValueError):
            quiet_seconds = 0.8
        if now - float(clear_since) < max(0.0, quiet_seconds):
            return

        route_index = int(self.idx_routes)
        if route_index != getattr(
                self, "_wz_combat_checkpoint_cleared_route_index", None):
            logger.info(
                f"[combat-checkpoint] Cleared {checkpoint.label}; "
                "resume patrol"
            )
        self._wz_combat_checkpoint_cleared_route_index = route_index

    def _combat_action_ready(self, now, last_action_at, action_cfg):
        """Return whether a cooldown-bound combat skill is ready."""
        interval = max(0.0, float(action_cfg.get("cooldown", 0.0)))
        return now - float(last_action_at) > interval

    def update_cmd_by_mob_detection(self):
        self._suppress_periodic_attack = False
        self._wz_attackable_monster_this_frame = False
        self._wz_combat_observation_valid = False
        timed_jump_status = self._wz_timed_jump_status()
        timed_jump_state = (
            timed_jump_status.get("state")
            if isinstance(timed_jump_status, dict) else None
        )
        armed_timed_jump_pending = timed_jump_state == "pending"
        if not armed_timed_jump_pending and self._wz_timed_jump_owns_input():
            # A timer that is already firing (or has just fired) cannot be
            # safely split into combat input. A merely pending timer remains
            # cancellable and must still pass this frame's monster check.
            self.monsters = []
            self._suppress_periodic_attack = True
            return
        timed_rope_status = self._rope_timed_mount_status()
        timed_rope_state = (
            timed_rope_status.get("state")
            if isinstance(timed_rope_status, dict) else None
        )
        armed_timed_rope_pending = timed_rope_state == "pending"
        if not armed_timed_rope_pending and \
                self._rope_timed_mount_owns_input():
            # Once the timer callback begins, Up + Jump is one indivisible HID
            # transaction. Combat resumes only after the climb state observes
            # its result.
            self.monsters = []
            self._suppress_periodic_attack = True
            return
        combat_actions = {"attack", "directional_aoe", "power_knockback"}
        clear_before_move = self._clears_monsters_before_moving()
        route_jump_pending = self.cmd_action == "jump"
        timed_jump_pending = armed_timed_jump_pending or isinstance(
            getattr(self, "_wz_timed_jump_candidate", None), dict
        )
        atomic_wz_jump_pending = bool(getattr(
            self, "_wz_route_jump_atomic_pending", False
        ))
        # A generated JUMP reservation is only a route handoff marker. Monster
        # arbitration still runs before HID dispatch, so no uncommitted jump
        # may bypass a currently attackable target.
        self._wz_route_jump_atomic_pending = False
        if atomic_wz_jump_pending and not route_jump_pending:
            # The route command changed before arbitration. Release the marker
            # so the same jump can reserve a fresh transaction if reacquired.
            self._wz_route_jump_atomic_route_index = None
        rope_state = getattr(self, "_rope_climb_state", None)
        rope_pre_mount = bool(
            getattr(self, "_rope_climb_active", False)
            and isinstance(rope_state, dict)
            and rope_state.get("phase") in {
                "position", "settle", "stationary_settle",
                "running_approach", "timed_mount", "mount_request"
            }
        )
        rope_jump_pending = bool(
            rope_pre_mount and rope_state.get("phase") in {
                "timed_mount", "mount_request"
            }
        )
        failed_rope_key = getattr(self, "_rope_climb_failed_key", None)
        rope_retry_pending = bool(
            failed_rope_key is not None
            and int(failed_rope_key[0])
            == int(getattr(self, "idx_routes", -1))
        )
        jump_transaction_pending = bool(
            route_jump_pending or timed_jump_pending or rope_jump_pending
        )
        transition_requires_combat_clear = bool(
            jump_transaction_pending or rope_pre_mount or rope_retry_pending
        )
        if getattr(self, "_rope_climb_active", False) and not rope_pre_mount:
            # Once the mount command has actually been dispatched, the rope
            # transaction owns every input through attachment/ascent. The
            # still-undispatched mount_request phase remains combat-arbitrated.
            self._suppress_periodic_attack = True
            return
        if getattr(self, "_portal_sweep_active", False):
            # Portal activation owns its visual feedback loop until minimap
            # displacement confirms completion.
            self._suppress_periodic_attack = True
            return

        if getattr(self, "is_on_ladder", False):
            # Preserve the route's vertical input exactly like the legacy
            # controller: no normal attack, directional AoE, or close-range
            # knockback may interrupt a rope/ladder command. Suppressing the
            # frame also prevents patrol attacks and stuck recovery from
            # replacing Up/Down later in the state pipeline.
            self.monsters = []
            if self.cmd_action in combat_actions:
                self.cmd_action = "none"
            self._suppress_periodic_attack = True
            return

        if getattr(self, "cmd_move_y", "none") in {"up", "down"} and not \
                jump_transaction_pending:
            # Before the climbing pose is confirmed, keep the legacy behavior
            # of not attacking during a vertical route command. Do not suppress
            # the watchdog here: an incorrect nearby Up/Down candidate must be
            # allowed to time out and recover instead of owning input forever.
            self.monsters = []
            if self.cmd_action in combat_actions:
                self.cmd_action = "none"
            return

        def hold_route_jump_for_combat():
            """Release a pending route jump while a nearby monster has priority."""
            if not (route_jump_pending or timed_jump_pending):
                return
            if armed_timed_jump_pending:
                self._cancel_wz_timed_directional_jump()
            self.cmd_move_x = "none"
            self.cmd_move_y = "none"
            self.cmd_action = "none"

        def hold_rope_approach_for_combat():
            """Freeze a light-blue pre-mount route while combat owns input."""
            if not rope_pre_mount:
                return
            self._set_rope_climb_combat_deferred(True)
            self.cmd_move_x = "none"
            self.cmd_move_y = "none"
            self.cmd_action = "none"

        def hold_navigation_for_combat():
            """Give an attackable monster priority over every route input."""
            if not clear_before_move:
                return
            hold_rope_approach_for_combat()
            self.cmd_move_x = "none"
            self.cmd_move_y = "none"
            if self.cmd_action not in combat_actions:
                self.cmd_action = "none"

        # Never build an attack decision without an accepted screen location.
        # A briefly retained Hero position remains accepted by design.
        if not getattr(self, "screen_player_location_valid", False):
            if rope_jump_pending:
                hold_rope_approach_for_combat()
            elif rope_pre_mount:
                self._set_rope_climb_combat_deferred(False)
            self.monsters = []
            hold_route_jump_for_combat()
            if not jump_transaction_pending:
                self.cmd_action = "none"
            self._suppress_periodic_attack = True
            return

        # From this point the frame has a fresh Hero anchor and normal ground
        # combat arbitration. Platform dwell/quiet timing may consume it.
        self._wz_combat_observation_valid = True

        # Get monster search box
        margin = self.cfg["monster_detect"]["search_box_margin"]
        attack_mode = self.cfg["bot"]["attack"]
        directional_aoe_cfg = self.cfg.get("directional_aoe", {})
        directional_aoe_enabled = bool(
            attack_mode == "directional"
            and directional_aoe_cfg.get("enable", False)
        )
        power_knockback_cfg = self.cfg.get("power_knockback", {})
        power_knockback_enabled = bool(
            attack_mode == "directional"
            and power_knockback_cfg.get("enable", False)
        )
        if attack_mode == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
            cooldown = self.cfg["aoe_skill"]["cooldown"]
        elif attack_mode == "directional":
            directional_cfg = self.cfg["directional_attack"]
            range_x = int(directional_cfg["range_x"])
            range_y = int(directional_cfg["range_y"])
            if directional_aoe_enabled:
                range_x = max(range_x, int(directional_aoe_cfg["range_x"]))
                range_y = max(range_y, int(directional_aoe_cfg["range_y"]))
            if power_knockback_enabled:
                range_x = max(
                    range_x,
                    int(power_knockback_cfg["trigger_distance_x"]),
                )
                range_y = max(
                    range_y,
                    int(power_knockback_cfg["range_y"]),
                )
            dx = range_x + margin
            dy = range_y + margin
        else:
            raise RuntimeError(f"Unsupported attack mode: {attack_mode}")
        x0 = max(0                      , self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0                      , self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monsters in the search box
        self.monsters = self.get_monsters_in_range((x0, y0), (x1, y1))
        navigator = getattr(self, "wz_navigation", None)
        if bool(getattr(
                navigator, "platform_state_machine_active", False)) and \
                not bool(getattr(
                    navigator, "platform_combat_priority", True
                )) and not transition_requires_combat_clear:
            # A continuous detection already consumed this transition's
            # bounded combat budget. Keep the semantic WZ path command; the
            # destination platform will re-enable clear-first combat. Never
            # apply this escape hatch while preparing/retrying a rope or an
            # uncommitted jump: those transitions must wait for a clear attack
            # range so knockback cannot leave Hero idle at their launch point.
            self.monsters = []
            return
        checkpoint = self._active_wz_combat_checkpoint()
        if checkpoint is not None:
            facing = checkpoint.facing
            player_x = self.loc_player[0]
            if facing == "left":
                self.monsters = [
                    monster for monster in self.monsters
                    if detection_center(monster)[0] < player_x
                ]
                checkpoint_sides = (True,)
            elif facing == "right":
                self.monsters = [
                    monster for monster in self.monsters
                    if detection_center(monster)[0] > player_x
                ]
                checkpoint_sides = (False,)
            else:
                checkpoint_sides = (True, False)

            if attack_mode == "directional":
                checkpoint_attackable = any(
                    self.get_monsters_in_attack_range(
                        is_left=is_left,
                        attack_type="directional",
                    )
                    for is_left in checkpoint_sides
                )
                if directional_aoe_enabled:
                    checkpoint_attackable = checkpoint_attackable or any(
                        self.get_monsters_in_attack_range(
                            is_left=is_left,
                            attack_type="directional_aoe",
                        )
                        for is_left in checkpoint_sides
                    )
                if power_knockback_enabled:
                    checkpoint_attackable = checkpoint_attackable or any(
                        self.get_power_knockback_monsters(
                            is_left=is_left
                        )
                        for is_left in checkpoint_sides
                    )
            else:
                checkpoint_attackable = bool(self.monsters)
            self._wz_attackable_monster_this_frame = bool(
                checkpoint_attackable
            )
            self._observe_wz_combat_checkpoint(checkpoint_attackable)
            if not checkpoint_attackable:
                self.monsters = []

        if len(self.monsters) == 0:
            if rope_pre_mount:
                self._set_rope_climb_combat_deferred(False)
            return

        # Update attack command
        if attack_mode == "aoe_skill":
            self._wz_attackable_monster_this_frame = True
            hold_route_jump_for_combat()
            hold_rope_approach_for_combat()
            hold_navigation_for_combat()
            self._suppress_periodic_attack = True
            if time.time() - self.t_last_attack > cooldown:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()
            else:
                hold_route_jump_for_combat()

        elif attack_mode == "directional":
            close_monsters_left = []
            close_monsters_right = []
            if power_knockback_enabled:
                close_monsters_left = self.get_power_knockback_monsters(
                    is_left=True,
                )
                close_monsters_right = self.get_power_knockback_monsters(
                    is_left=False,
                )
            left_blocked = bool(close_monsters_left)
            right_blocked = bool(close_monsters_right)
            if left_blocked or right_blocked:
                # Patrol's blind periodic attack must not overwrite the
                # close-range decision and fire a bow into a blocked side.
                self._suppress_periodic_attack = True

            if directional_aoe_enabled:
                aoe_monsters_left = (
                    []
                    if left_blocked
                    else self.get_monsters_in_attack_range(
                        is_left=True,
                        attack_type="directional_aoe",
                    )
                )
                aoe_monsters_right = (
                    []
                    if right_blocked
                    else self.get_monsters_in_attack_range(
                        is_left=False,
                        attack_type="directional_aoe",
                    )
                )
                aoe_direction = self.get_directional_aoe_direction(
                    aoe_monsters_left,
                    aoe_monsters_right,
                    directional_aoe_cfg["min_monsters"],
                )
                if aoe_direction is not None:
                    self._wz_attackable_monster_this_frame = True
                    self._suppress_periodic_attack = True
                    hold_route_jump_for_combat()
                    now = time.time()
                    if self._combat_action_ready(
                            now,
                            getattr(self, "t_last_directional_aoe", 0.0),
                            directional_aoe_cfg,
                    ):
                        hold_rope_approach_for_combat()
                        self.cmd_move_x = "none"
                        self.cmd_move_y = "none"
                        self.cmd_action = "directional_aoe"
                        self.cmd_move_x = aoe_direction
                        self.t_last_directional_aoe = now
                        self.t_last_attack = now
                    elif self.cmd_action in combat_actions:
                        self.cmd_action = "none"
                    if self.cmd_action != "directional_aoe":
                        if self._uses_ranged_safe_patrol() and \
                                not clear_before_move:
                            if rope_pre_mount:
                                self._set_rope_climb_combat_deferred(False)
                        else:
                            hold_rope_approach_for_combat()
                            self.cmd_move_x = "none"
                            self.cmd_move_y = "none"
                            hold_route_jump_for_combat()
                    # Reaching the threshold owns the combat decision. While
                    # it cools down, preserve the WZ route command so ranged
                    # patrol keeps searching instead of standing forever.
                    return

            # Get nearest monster to player
            monster_left = (
                None
                if left_blocked
                else self.get_nearest_monster(is_left=True)
            )
            monster_right = (
                None
                if right_blocked
                else self.get_nearest_monster(is_left=False)
            )
            # Determine attack direction
            attack_direction = self.get_attack_direction(monster_left, monster_right)
            if attack_direction is None and (
                    route_jump_pending or rope_pre_mount
                    or clear_before_move):
                # At a route-jump or light-blue pre-mount marker, an equally
                # close monster on each side must still win. Reuse the
                # deterministic direction tie-breaker instead of routing.
                attack_direction = self.get_directional_aoe_direction(
                    [monster_left] if monster_left is not None else [],
                    [monster_right] if monster_right is not None else [],
                    1,
                )
            if attack_direction is not None:
                self._wz_attackable_monster_this_frame = True
                self._suppress_periodic_attack = True
                now = time.time()
                hold_route_jump_for_combat()
                hold_rope_approach_for_combat()
                self.cmd_move_x = attack_direction
                self.cmd_move_y = "none"
                self.cmd_action = "attack"
                self.t_last_attack = now
                return

            if power_knockback_enabled and (left_blocked or right_blocked):
                knockback_direction = self.get_directional_aoe_direction(
                    close_monsters_left,
                    close_monsters_right,
                    1,
                )
                now = time.time()
                if knockback_direction is not None:
                    hold_route_jump_for_combat()
                if knockback_direction is not None and \
                        self._combat_action_ready(
                            now,
                            getattr(self, "t_last_power_knockback", 0.0),
                            power_knockback_cfg,
                        ):
                    self._wz_attackable_monster_this_frame = True
                    hold_rope_approach_for_combat()
                    self.cmd_move_x = "none"
                    self.cmd_move_y = "none"
                    self.cmd_action = "power_knockback"
                    self.cmd_move_x = knockback_direction
                    self.t_last_power_knockback = now
                    self.t_last_attack = now
                elif knockback_direction is not None:
                    self._wz_attackable_monster_this_frame = True
                    if self.cmd_action in combat_actions:
                        self.cmd_action = "none"
                    if self._uses_ranged_safe_patrol() and \
                            not clear_before_move:
                        if rope_pre_mount:
                            self._set_rope_climb_combat_deferred(False)
                    else:
                        hold_rope_approach_for_combat()
                        self.cmd_move_x = "none"
                        self.cmd_move_y = "none"
                        hold_route_jump_for_combat()
                elif self.cmd_action in combat_actions:
                    self.cmd_action = "none"
                if rope_pre_mount and knockback_direction is None:
                    self._set_rope_climb_combat_deferred(False)
                return

            if rope_pre_mount:
                self._set_rope_climb_combat_deferred(False)

    def update_wz_platform_combat_state(self):
        """Publish this frame's final combat decision to the platform FSM."""
        navigator = getattr(self, "wz_navigation", None)
        observer = getattr(navigator, "observe_platform_combat", None)
        if not callable(observer):
            return False
        attackable = (
            bool(getattr(
                self, "_wz_attackable_monster_this_frame", False
            ))
            if getattr(self, "_wz_combat_observation_valid", False)
            else None
        )
        return bool(observer(attackable))

    def recover_wz_platform_navigation(self, reason="stuck watchdog"):
        """Use deterministic WZ replanning instead of a random recovery jump."""
        navigator = getattr(self, "wz_navigation", None)
        if not bool(getattr(
                navigator, "platform_state_machine_active", False)):
            return False
        replan = getattr(navigator, "request_platform_replan", None)
        if not callable(replan):
            return False
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"
        return bool(replan(str(reason)))

    def update_cmd_by_random(self):
        '''
        update_cmd_by_random - pick a random action except 'up' and teleport command
        '''
        self.cmd_move_x = random.choice(["left", "right", "none"])
        self.cmd_move_y = random.choice(["down", "none"])
        self.cmd_action = random.choice(["jump", "none"])
        logger.warning("[update_cmd_by_random]"\
                    f"{self.cmd_move_x} {self.cmd_move_y} {self.cmd_action}")

    def check_reach_goal(self):
        if self.cmd_action == "goal":
            self._reset_rope_climb(clear_locks=True)
            self._reset_ladder_route_hold()
            navigator = getattr(self, "wz_navigation", None)
            if bool(getattr(
                    navigator, "platform_state_machine_active", False)):
                route_count = len(self.img_routes)
                current_idx = int(self.idx_routes)
                goal_handler = getattr(
                    navigator, "platform_route_goal_reached", None
                )
                if callable(goal_handler) and goal_handler(current_idx):
                    self.cmd_move_x = "none"
                    self.cmd_move_y = "none"
                    self.cmd_action = "none"
                    self._reset_wz_combat_checkpoint()
                    return

                next_idx = current_idx + 1
                if next_idx < route_count:
                    # Temporary WZ paths are ordered semantic legs. Advance
                    # exactly once; never scan unrelated overlapping images.
                    self.idx_routes = next_idx
                    self.img_route = self.img_routes[next_idx]
                    if getattr(self, "is_show_debug_window", False):
                        self.img_route_debug = cv2.cvtColor(
                            self.img_route, cv2.COLOR_RGB2BGR
                        )
                    self._reset_wz_combat_checkpoint()
                    if not self._handoff_wz_jump():
                        self.cmd_action = "none"
                    logger.info(
                        "[platform-fsm] Advance temporary path leg "
                        f"{current_idx + 1}->{next_idx + 1}"
                    )
                    return

                replan = getattr(navigator, "request_platform_replan", None)
                if callable(replan):
                    replan("temporary path ended before platform confirmation")
                self.cmd_move_x = "none"
                self.cmd_move_y = "none"
                self.cmd_action = "none"
                return
            if self._hold_at_wz_combat_checkpoint():
                return
            route_count = len(self.img_routes)
            current_idx = int(self.idx_routes)
            next_idx = (
                (current_idx + 1) % route_count
                if route_count > 0
                else current_idx
            )

            # A goal is also useful as the closing checkpoint of a loop.  Do
            # not blindly switch to a disconnected route: that used to make
            # the fallback lookup select this same goal every frame, leaving
            # Hero stuck near hazards.  Suppress the checkpoint temporarily
            # and let the adjacent movement stroke continue the same loop.
            has_next_continuation = (
                route_count > 1
                and self._route_has_local_continuation(next_idx)
            )
            continuation_idx = next_idx if has_next_continuation else None

            # Directed WZ patrol components can meet only after a portal/drop
            # cycle. Walk forward through the generated plan and resume at the
            # first route that is actionable where Hero is standing. Keep the
            # single-step behavior for hand-drawn routes.
            if (
                    continuation_idx is None
                    and route_count > 2
                    and getattr(self, "_wz_navigation_enabled", False)):
                for route_offset in range(2, route_count):
                    candidate_idx = (
                        current_idx + route_offset
                    ) % route_count
                    if self._route_has_local_continuation(candidate_idx):
                        continuation_idx = candidate_idx
                        logger.info(
                            f"[route] route{next_idx + 1} has no local "
                            f"action at the goal; continue with "
                            f"route{candidate_idx + 1}"
                        )
                        break

            if continuation_idx is None:
                component = self._get_route_goal_component_at_player(
                    current_idx
                )
                if component is not None:
                    goal_key = (
                        current_idx, tuple(component["bbox"])
                    )
                    self._route_loop_goal_key = goal_key
                    warned = getattr(
                        self, "_route_loop_goal_warned_keys", set()
                    )
                    if goal_key not in warned:
                        logger.warning(
                            f"[route] route{current_idx + 1} goal has no "
                            f"local continuation on route{next_idx + 1}; "
                            "keep the current route and treat this goal as "
                            "a loop checkpoint"
                        )
                        warned.add(goal_key)
                        self._route_loop_goal_warned_keys = warned
                self.cmd_action = "none"
                return

            self._route_loop_goal_key = None
            self.idx_routes = continuation_idx
            self.img_route = self.img_routes[self.idx_routes]
            if getattr(self, "is_show_debug_window", False):
                self.img_route_debug = cv2.cvtColor(
                    self.img_route, cv2.COLOR_RGB2BGR
                )
            self._reset_wz_combat_checkpoint()
            if not self._handoff_wz_jump():
                self.cmd_action = "none"
            logger.debug(f"Change to new route:{self.idx_routes}")

    def run_once(self):
        '''
        Process one game window frame
        '''
        # Start profiler for performance debugging
        self.profiler.start()
        wz_navigation_enabled = bool(
            getattr(self, "_wz_navigation_enabled", False)
        )

        # Check if need viz window
        # Annotated recording needs the same full-resolution canvas as the UI
        # visualization, but it must not permanently change the user's tab/UI
        # visualization preference.
        annotated_recording = (
            getattr(self, "_annotated_video_record_path", None) is not None
        )
        self.is_show_debug_window = (
            getattr(self, "is_need_show_debug_window", False)
            or annotated_recording
        )
        self._annotated_record_frame_pending = False
        if not self.is_show_debug_window:
            self.img_frame_debug = None
            self.img_route_debug = None

        ###########################
        ### Image Preprocessing ###
        ###########################
        # Get game window frame
        img_frame = self.get_img_frame()
        if img_frame is None:
            self.suspend_input_for_capture_loss()
            if not self.is_debug_mode() and \
                not is_mac() and not self.remote_keyboard_target() and \
                    not self.is_capture_card_source():
                activate_game_window(self.capture.window_title)
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame

        # F3 records the exact capture snapshot held in ``self.frame``. This
        # happens before OCR, cropping, resizing, or debug overlays and also
        # covers login/recovery frames that return early later in this method.
        self._write_raw_video_frame(getattr(self, "frame", None))

        # If recovery was already active, its dedicated keyboard gate prevents
        # ordinary gameplay input, so a newly restored capture can be reopened
        # immediately. On the first login frame, defer reopening until after
        # classification; otherwise a worker thread can emit a queued Buff in
        # the brief gap before session recovery suspends normal automation.
        recovery_was_active = getattr(
            self, "_auto_relogin_state", "idle"
        ) != "idle"
        if recovery_was_active:
            self.resume_input_after_capture()

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug viz
        if self.is_show_debug_window:
            self.img_frame_debug = self.img_frame.copy()
            self._annotated_record_frame_pending = annotated_recording

        # Session recovery is a global safety guard, not a gameplay FSM state.
        # It must preempt route, health, watchdog, and combat behavior even
        # when a saved minimap rectangle can still be cropped from a login page.
        if self._check_auto_relogin_screen():
            # A newly detected login page has now closed the normal-input gate;
            # reopening capture availability enables only explicit recovery
            # Enter/mouse commands.
            self.resume_input_after_capture()
            self.profiler.mark("Auto Relogin")
            return -1

        self.resume_input_after_capture()

        # Get current route image
        if self.cfg["bot"]["mode"] == "normal" and self.img_routes:
            self.idx_routes %= len(self.img_routes)
            self.img_route = self.img_routes[self.idx_routes]
            if self.is_show_debug_window:
                self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        self.profiler.mark("Image Preprocessing")

        ###################
        ### Get Minimap ###
        ###################
        # New route recordings persist a border-free minimap rectangle beside
        # map.png. Main reuses that rectangle directly; legacy maps without the
        # text file retain the old detector until they are re-recorded.
        saved_geometry_status = self.apply_saved_minimap_geometry()
        if saved_geometry_status is False:
            return -1
        minimap_updated = saved_geometry_status is True
        detected_minimap_result = None

        if saved_geometry_status is None:
            minimap_result = get_minimap_loc_size(self.img_frame)
            detected_minimap_result = minimap_result
            if minimap_result is not None:
                x, y, w, h = minimap_result
                # Shrink the detected white border by one pixel on every side.
                x += 1
                y += 1
                w -= 2
                h -= 2
                if w > 0 and h > 0:
                    self.loc_minimap = (x, y)
                    self.img_minimap_screen = self.img_frame[y:y+h, x:x+w]
                    self.img_minimap_source = self.img_minimap_screen
                    minimap_updated = True

                    # Legacy normalized captures still use native content for
                    # the tiny player dot. This fallback disappears once the
                    # route has minimap_geometry.txt.
                    native_frame = getattr(self, "img_capture_content", None)
                    if native_frame is not None:
                        native_result = get_minimap_loc_size(native_frame)
                        if native_result is not None:
                            nx, ny, nw, nh = native_result
                            nx += 1
                            ny += 1
                            nw -= 2
                            nh -= 2
                            if nw > 0 and nh > 0:
                                self.img_minimap_source = native_frame[
                                    ny:ny+nh, nx:nx+nw
                                ]

        if minimap_updated:
            native_size = self.img_minimap_source.shape[:2]
            expected_size = getattr(self, "_native_minimap_size", None)
            if expected_size is not None and native_size != expected_size and \
                    not wz_navigation_enabled:
                error_key = (expected_size, native_size)
                if error_key != getattr(
                    self, "_last_native_minimap_error", None
                ):
                    logger.error(
                        "[minimap] Native raster changed from "
                        f"{expected_size[::-1]} to {native_size[::-1]}; "
                        "discarding this frame because route coordinates "
                        "cannot be rescaled"
                    )
                    self._last_native_minimap_error = error_key
                return -1

            self._native_minimap_size = native_size
            self._last_native_minimap_error = None
            self.img_minimap = copy_minimap_native_raster(
                self.img_minimap_source
            )

            if self.minimap_geometry is not None and \
                    not wz_navigation_enabled:
                _, _, recorded_w, recorded_h = \
                    self.minimap_geometry["minimap_rect"]
                recorded_size = (recorded_h, recorded_w)
                if native_size != recorded_size:
                    error_key = (recorded_size, native_size)
                    if error_key != self._last_route_map_size_error:
                        logger.error(
                            "[minimap] Captured native raster "
                            f"{native_size[::-1]} differs from the route's "
                            f"recorded raster {recorded_size[::-1]}; use the "
                            "same capture resolution/UI scale or re-record "
                            "the map and routes"
                        )
                        self._last_route_map_size_error = error_key
                    return -1

            if self.cfg["bot"]["mode"] == "normal" and \
                    not wz_navigation_enabled and \
                    self.img_map is not None and \
                    not route_map_can_fit_minimap(
                        self.img_map, self.img_minimap
                    ):
                map_size = self.img_map.shape[:2]
                error_key = (map_size, native_size)
                if error_key != self._last_route_map_size_error:
                    logger.error(
                        "[minimap] Existing map.png is too small for the "
                        f"native minimap: map={map_size[::-1]}, "
                        f"minimap={native_size[::-1]}. Re-record map.png and "
                        "all route images without minimap resizing"
                    )
                    self._last_route_map_size_error = error_key
                return -1

            self._last_route_map_size_error = None
            if native_size != getattr(self, "_last_native_minimap_log", None):
                logger.info(
                    "[minimap] Using native capture raster "
                    f"{native_size[::-1]} without resizing"
                )
                self._last_native_minimap_log = native_size
            self.t_last_minimap_update = time.time()

        self.profiler.mark("Get Minimap Location and Size")

        # Update health monitor with current frame
        self.health_monitor.update_frame(self.img_frame[self.cfg["ui_coords"]["ui_y_start"]:, :])

        #################################
        ### Player Location Detection ###
        #################################
        # Get player location in game window
        loc_player, loc_party_red_bar = self.get_player_location_on_screen()
        if loc_party_red_bar is not None:
            self.loc_party_red_bar = loc_party_red_bar

        # Update player location
        self.screen_player_location_valid = loc_player is not None
        if loc_player is not None:
            # Update player location
            self.loc_player = loc_player

        # Draw the accepted coordinate, including a retained Hero position.
        if self.screen_player_location_valid:
            self._draw_debug_circle(
                self.loc_player,
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
            )

        # Get player location on minimap
        minimap_cfg = self.cfg["minimap"]
        loc_player_minimap_source = get_player_location_on_minimap(
            self.img_minimap_source,
            minimap_player_color=minimap_cfg["player_color"],
            color_tolerance=minimap_cfg.get("player_color_tolerance", 0),
            min_component_area=minimap_cfg.get(
                "player_min_component_area", 4
            ),
        )
        if loc_player_minimap_source:
            self.loc_player_minimap = copy_minimap_native_location(
                loc_player_minimap_source
            )

        # A fixed crop alone is not proof that the game is back. Require a
        # player dot from the current minimap raster on consecutive frames,
        # and keep every gameplay producer gated until that proof is fresh.
        relogin_state = getattr(self, "_auto_relogin_state", "idle")
        if relogin_state in {"waiting_game", "failed"} and \
                detected_minimap_result is None:
            detected_minimap_result = get_minimap_loc_size(self.img_frame)
        minimap_structure_confirmed = (
            relogin_state not in {"waiting_game", "failed"}
            or self._auto_relogin_minimap_structure_valid(
                detected_minimap_result
            )
        )
        game_ready_location = (
            loc_player_minimap_source
            if minimap_updated and minimap_structure_confirmed
            else None
        )
        if self._gate_auto_relogin_until_game_ready(game_ready_location):
            self.profiler.mark("Auto Relogin Game Ready")
            return -1

        # Get other player location on minimap
        loc_other_players = get_all_other_player_locations_on_minimap(
                                self.img_minimap,
                                self.cfg['minimap']['other_player_color'])
        # Debug
        # if self.is_first_frame:
        #     logger.info("Running minimap color analysis...")
        #     debug_minimap_colors(self.img_minimap, other_player_color)

        # Get player location on global map
        if self.cfg["bot"]["mode"] in ["patrol", "aux", "debug"]:
            self.loc_player_global = self.loc_player_minimap
        elif wz_navigation_enabled:
            if not self._update_wz_navigation_from_minimap(
                    loc_player_minimap_source):
                self.profiler.mark("WZ Navigation")
                return -1
        else:
            self.loc_player_global = self.get_player_location_on_global_map()

        self.profiler.mark("Player Location Detection")

        ######################
        ### Change Channel ###
        ######################
        if not self.is_debug_mode() and \
            self.cfg['channel_change']['enable'] and \
            self.is_need_change_channel(loc_other_players):
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            if not self.remote_keyboard_target():
                self.kb.disable()
            time.sleep(1)
            self.channel_change()
            self.red_dot_center_prev = None
            return 0

        if not self.is_debug_mode() and \
            self.is_time_to_change_channel():
            self.kb.set_command("none none none")
            self.kb.release_all_key()
            if not self.remote_keyboard_target():
                self.kb.disable()
            time.sleep(1)
            self.channel_change()
            return 0

        self.profiler.mark("Change Channel")

        #######################
        ### Attack WatchDog ###
        ####################### Check if last attack is timeout
        dt = time.time() - self.t_last_attack
        if self.cfg['bot']['mode'] == 'normal' and \
            dt > self.cfg["watchdog"]["last_attack_timeout"]:
            logger.info(f"[Attack Timeout] Last attack timeout for {round(dt, 2)} seconds")
            cfg_action = self.cfg["watchdog"]["last_attack_timeout_action"]
            if cfg_action == "change_channel":
                logger.info("[Attack Timeout] Change channel!")
                if self.remote_keyboard_target() and not \
                        self.remote_absolute_mouse_calibrated():
                    logger.warning(
                        "[Attack Timeout] Skipped channel change: remote "
                        "absolute mouse is not calibrated"
                    )
                    self.t_last_attack = time.time()
                else:
                    self.channel_change()
            elif cfg_action == "go_home":
                logger.info("[Attack Timeout] Return home!")
                if press_key(self.cfg["key"].get("return_home", "")):
                    # Terminate only after an acknowledged or uncertain TAP;
                    # uncertain TAPs are never replayed automatically.
                    self.is_terminated = True
                    self.kb.is_terminated = True
                else:
                    self.t_last_attack = time.time()
            else:
                logger.info(f"Unsupported timeout mode: {cfg_action}")

        self.profiler.mark("Attack WatchDog")

        ######################
        ### State Behavior ###
        ######################
        # Every action is a one-frame request. Individual states may replace
        # it below; this gives the keyboard controller a clean action edge and
        # prevents stale attack/jump/teleport TAP commands from repeating.
        self.cmd_action = "none"
        if not self._hold_for_scheduled_buff_recovery():
            self.fsm.do_state_stuff()

        self.is_first_frame = False

        self.profiler.mark("State per-frame behavior")

        #####################
        ### Debug Windows ###
        #####################
        # Don't show debug window to save system resource
        if not self.is_show_debug_window:
            return 0 # frame done

        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Resize img_route_debug for better visualization
        if self.cfg["bot"]["mode"] == "normal" and \
                self.img_route_debug is not None:
            self.img_route_debug = cv2.resize(
                        self.img_route_debug, (0, 0),
                        fx=self.cfg["minimap"]["debug_window_upscale"],
                        fy=self.cfg["minimap"]["debug_window_upscale"],
                        interpolation=cv2.INTER_NEAREST)

        self.profiler.mark("Debug Window Show")

        # Update FPS timer
        self.t_last_frame = time.time()

        # Print profiler result
        if self.cfg["profiler"]["enable"] and \
            self.profiler.total_frames % self.cfg["profiler"]["print_frequency"] == 0:
            logger.info('\n' + self.profiler.report())

        return 0 # frame done

    def loop(self):
        '''
        Auto Bot main loop
        Only run when call autobot from UI framework and AutoBotController
        '''
        try:
            # Computer B is already foreground in capture-card mode. Focusing
            # any preview window on A is unrelated to where BLE input lands.
            if not self.is_debug_mode() and \
                not is_mac() and not self.remote_keyboard_target() and \
                    not self.is_capture_card_source():
                activate_game_window(self.capture.window_title)
                time.sleep(0.3)
                self.ensure_is_in_party()

            while not self.is_terminated and not self.kb.is_terminated:
                t_start = time.time()

                # Process one game window frame
                self.is_frame_done = False
                ret = self.run_once()

                # Flush after run_once so both normal and safe early-return
                # paths preserve every annotated canvas produced this cycle.
                self._flush_annotated_video_frame()

                # Only proceed if the frame is valid
                if ret == 0:
                    self._emit_debug_images()

                self.is_frame_done = True

                # Cap FPS to save system resource
                frame_duration = time.time() - t_start
                target_duration = 1.0 / self.cfg["system"]["fps_limit_main"]
                if frame_duration < target_duration:
                    time.sleep(target_duration - frame_duration)
        except BaseException as exc:
            logger.error(f"[MapleStoryAutoBot] Main loop failed: {exc}")
            raise
        finally:
            # Never leave the independent keyboard/heartbeat threads alive if
            # detection or UI rendering crashes; a remote movement key could
            # otherwise be renewed indefinitely on computer B.
            try:
                self.stop_record()
            except Exception as exc:
                logger.error(f"[MapleStoryAutoBot] Recorder cleanup failed: {exc}")
            try:
                self.terminate_threads()
            except Exception as exc:
                logger.error(f"[MapleStoryAutoBot] Emergency cleanup failed: {exc}")
            if self.thread_auto_bot is threading.current_thread():
                self.thread_auto_bot = None

def main(args):
    '''
    This main function works as a fake autoBotController
    This function will only be called when the using terminal to
    run this script
    '''
    #####################
    ### Init Auto Bot ###
    #####################
    try:
        mapleStoryAutoBot = MapleStoryAutoBot(args)
    except Exception as e:
        logger.error(f"MapleStoryAutoBot Init failed: {e}")
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Init Successfully")

    ####################
    ### Apply Config ###
    ####################
    # Load defautl yaml config
    cfg = load_yaml("config/config_default.yaml")
    # Override with platform config
    if is_mac():
        cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
    # Override with user customized config
    cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
    # Dump config to log for debugging
    logger.debug(yaml.dump(cfg, sort_keys=False,
                 indent=2, default_flow_style=False))
    # autoBot load config
    if mapleStoryAutoBot.load_config(cfg) != 0:
        logger.error("MapleStoryAutoBot config failed")
        sys.exit(1)

    #####################
    ### Start AutoBot ###
    #####################
    try:
        mapleStoryAutoBot.start() # Start all threads in autoBot
    except Exception as e:
        logger.error(f"MapleStoryAutoBot start failed: {e}")
        mapleStoryAutoBot.terminate_threads() # Terminate all threads
        sys.exit(1)
    else:
        logger.info("MapleStoryAutoBot Start Successfully")

    # Start record game window for debugging
    if args.record:
        mapleStoryAutoBot.start_record()

    kb_listener = KeyBoardListener(is_autobot=True)
    kb_listener.register_func_key_handler('f1', mapleStoryAutoBot.kb.toggle_enable)
    kb_listener.register_func_key_handler('f2', mapleStoryAutoBot.screenshot_img_frame)
    kb_listener.register_func_key_handler('f12', mapleStoryAutoBot.terminate_threads)

    # While loop
    while not mapleStoryAutoBot.is_terminated:
        # Show debug image on window
        if mapleStoryAutoBot.is_frame_done:
            if mapleStoryAutoBot.img_frame_debug is not None:
                cv2.imshow(
                    "Game Window Debug",
                    mapleStoryAutoBot.img_frame_debug,
                )

            if mapleStoryAutoBot.img_route_debug is not None:
                cv2.imshow("Route Map Debug", mapleStoryAutoBot.img_route_debug)

        cv2.waitKey(1)

        time.sleep(0.01)

    #########################
    ### Terminate AutoBot ###
    #########################
    mapleStoryAutoBot.terminate_threads() # Terminate all threads

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable simulated keyboard input'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--debug',
        action="store_true",
        help="Enable debug logging"
    )

    parser.add_argument(
        '--record',
        action="store_true",
        help="Record raw capture and annotated debug videos"
    )

    parser.add_argument(
        '--disable_viz',
        action="store_true",
        help="Disable viz debug window"
    )

    parser.add_argument(
        '--test_image',
        default="",
        help="Pass in image in test/XXX.png"
    )

    parser.add_argument(
        '--init_state',
        default="",
        help="choose the init_state"
    )

    args = parser.parse_args()
    args.is_ui = False # Always set False for command line

    # Set logger level
    if args.debug:
        logger.set_level(logging.DEBUG)

    main(args)
