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
    resize_minimap_to_reference,
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
from src.input.KeyBoardController import KeyBoardController, press_key
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
from src.states.hunting import HuntingState
from src.states.finding_rune import FindingRuneState
from src.states.near_rune import NearRuneState
from src.states.solving_rune import SolvingRuneState
from src.states.auxiliary import AuxiliaryState
from src.states.patrol import PatrolState
from src.states.debug import DebugState

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
        self.monsters = [] # monster detected in current frame
        self.fps = 0 # Frame per second
        self.red_dot_center_prev = None # previous other player location in minimap
        self.video_writer = None # For video recording feature
        self._video_record_path = None
        self._video_record_size = None
        self.color_code = {} # For color code instruction
        self.color_code_up_down = {} # Color code only contain 'up' and 'down'
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
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        self.img_minimap_screen = self.img_minimap # unscaled minimap in screen coordinates
        self.img_minimap_source = self.img_minimap # native capture-card minimap
        self.img_capture_content = None
        self.minimap_geometry = None
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer
        self.t_last_attack = time.time() # Last attack timer for cooldown
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

        # A native fullscreen capture is about 22 MB at 3579x2013. Sending a
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

    def is_debug_mode(self):
        """Return whether this run is the vision-only debug mode."""
        cfg = getattr(self, "cfg", None) or {}
        return cfg.get("bot", {}).get("mode") == "debug"

    def click_game_ui(self, coord, action):
        """Click only when the captured window and game are on the same PC."""
        if self.remote_keyboard_target():
            logger.warning(
                f"[{action}] Skipped local mouse click: ESP32 is paired to the "
                "remote game computer and the current HID firmware is keyboard-only"
            )
            return False
        click_in_game_window(self.capture.window_title, coord)
        return True

    def suspend_input_for_capture_loss(self):
        """Fail closed once when the capture stream becomes stale."""
        if self.kb is None or getattr(self, "_input_suspended_for_capture", False):
            return False
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
        if mode not in {"normal", "aux", "patrol", "debug"}:
            logger.error(f"[load_config] Unsupported bot mode: {mode}")
            return -1
        if monster_backend not in {"template", "yolo"}:
            logger.error(
                "[load_config] Unsupported monster detection backend: "
                f"{monster_backend}"
            )
            return -1

        # One UI controller can be paused and loaded again with another map or
        # mode. Never retain route or monster resources from that prior run.
        self.img_map = None
        self.img_route = None
        self.img_route_debug = None
        self.img_routes = []
        self.monsters_info = {}
        self.minimap_geometry = None
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
        self.has_valid_appearance_location = False
        self.pending_appearance_location = None
        self.pending_appearance_count = 0
        self.last_appearance_match = None

        if mode == "debug":
            logger.info(
                "[load_config] Debug mode is vision-only; keyboard, mouse, "
                "health, buff, login, and channel workflows are disabled"
            )
        if cfg.get("esp32_hid", {}).get("remote_target", False):
            # This capture-card setup controls computer B with a keyboard-only
            # HID descriptor. Mouse-driven channel flows cannot be completed
            # from computer A and must not repeatedly pause the combat loop.
            logger.info(
                "[load_config] Remote keyboard-only mode: all mouse-dependent "
                "workflows are disabled"
            )
            for section_name in ("channel_change", "scheduled_channel_switching"):
                section = cfg.get(section_name, {})
                if section.get("enable", False):
                    logger.warning(
                        f"[load_config] Disabled {section_name}.enable: "
                        "remote channel changes require mouse HID"
                    )
                    section["enable"] = False

            if cfg.get("watchdog", {}).get("last_attack_timeout_action") == "change_channel":
                logger.warning(
                    "[load_config] Watchdog channel changes are unavailable in "
                    "keyboard-only capture-card mode"
                )

        # Parse color code in config
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        self.color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }

        map_name = cfg['bot']['map']
        map_dir = os.path.join("minimaps", map_name)
        self.minimap_geometry = load_minimap_geometry(map_dir)
        if self.minimap_geometry is None:
            logger.warning(
                f"No minimap_geometry.txt for {map_name}; using legacy "
                "real-time minimap scanning until this map is re-recorded"
            )
        if mode == "normal" or (
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
            # Load map.png from minimaps/
            self.img_map = load_image(f"minimaps/{map_name}/map.png",
                                      cv2.IMREAD_COLOR)
            # Load route*.png from minimaps/
            route_files = sorted(glob.glob(f"minimaps/{map_name}/route*.png"))
            route_files = [p for p in route_files if not p.endswith("route_rest.png")]
            self.img_routes = []
            for route_file in route_files:
                img = cv2.cvtColor(load_image(route_file), cv2.COLOR_BGR2RGB)
                # Remove pixel in map that is color code
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code"])
                img = mask_route_colors(self.img_map, img, cfg["route"]["color_code_up_down"])
                self.img_routes.append(img)

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
            logger.info(
                "[load_config] YOLO monster detection ready: "
                f"model={self.yolo_monster_detector.model_path}, "
                f"imgsz={self.yolo_monster_detector.imgsz}, "
                f"confidence={self.yolo_monster_detector.confidence}, "
                f"device={self.yolo_monster_detector.device}, "
                f"class={self.yolo_monster_detector.class_name}"
                f"{warmup_text}"
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
            appearance_cfg = cfg["nametag"].get("appearance", {})
            if appearance_cfg.get("enable", False):
                for template_index, template_cfg in enumerate(
                        appearance_cfg.get("templates", [])):
                    template_name = template_cfg.get("name", "").strip()
                    if not template_name:
                        suffix = template_cfg.get("suffix", "").strip()
                        if suffix:
                            template_name = (
                                f"{cfg['nametag']['name']}_{suffix}"
                            )
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
                        "player_offset": (
                            int(offset[0]), int(offset[1])
                        ),
                    })
                if self.nametag_appearance_templates:
                    logger.info(
                        "Loaded player appearance templates: "
                        f"{[item['name'] for item in self.nametag_appearance_templates]}"
                    )
                else:
                    logger.warning(
                        "Player appearance detection is enabled but no "
                        "templates could be loaded"
                    )

        marker_cfg = cfg["nametag"].get("overhead_marker", {})
        if marker_cfg.get("enable", False):
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

        return 0 # load successfully

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
            if self.args.test_image == '':
                self.capture = GameWindowCapturor(self.cfg)
            else:
                self.capture = GameWindowCapturor(self.cfg, self.args.test_image)

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
            )
            self._input_suspended_for_capture = (
                control_enabled and self.remote_keyboard_target()
            )

            self.health_monitor = HealthMonitor(self.cfg, self.kb)
            if self.cfg["health_monitor"]["enable"] and control_enabled:
                self.health_monitor.start()

            self.profiler = Profiler(self.cfg)
            self.rune_solver = RuneSolver(self.cfg)

            # Reset all timers
            self.t_last_frame = time.time()
            self.t_watch_dog = time.time()
            self.t_last_teleport = time.time()
            self.t_last_attack = time.time()
            self.t_last_minimap_update = time.time()
            self.t_to_change_channel = time.time()

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
        Start record
        '''
        # Prepare video writer if need to record
        if not self.is_show_debug_window:
            self.enable_viz()

        # Make sure video/ exist
        os.makedirs("video", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join("video", f"{timestamp}.mp4")

        self._video_record_path = path
        self._video_record_size = None
        self.video_writer = None
        frame = getattr(self, "img_frame_debug", None)
        if frame is None:
            frame = getattr(self, "img_frame", None)
        if frame is not None:
            self._open_video_writer_for_frame(frame)

        logger.info(f"[start_record] Record video to {path}")

    def _open_video_writer_for_frame(self, frame):
        """Open the pending recorder with the actual current frame size."""
        path = getattr(self, "_video_record_path", None)
        if path is None or frame is None:
            return
        frame_h, frame_w = frame.shape[:2]
        frame_size = (frame_w, frame_h)
        if getattr(self, "video_writer", None) is not None and \
                getattr(self, "_video_record_size", None) == frame_size:
            return
        if getattr(self, "video_writer", None) is not None:
            self.video_writer.release()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(path, fourcc, 10, frame_size)
        self._video_record_size = frame_size

    def stop_record(self):
        '''
        Stop Record
        '''
        if getattr(self, "video_writer", None) is not None:
            self.video_writer.release()
        self.video_writer = None
        self._video_record_path = None
        self._video_record_size = None
        logger.info("[stop_record] Stop recording")

    def get_player_location_by_overhead_marker(
            self, expected_player=None, allow_global=True):
        """Locate Hero from the configured white smile bubble.

        A cheap connected-component pass first filters by the bubble's white
        geometry. Only those small candidates are checked with the masked face
        template. Local matches near the previous Hero position are accepted
        immediately; cold full-frame matches use consecutive-frame protection.
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
            lost_timeout_s = max(0.0, float(marker_cfg.get(
                "lost_timeout_s", 2.0
            )))
            cache_age_s = (
                None if last_detected is None else now - last_detected
            )
            use_cached_location = (
                getattr(
                    self, "has_valid_overhead_marker_location", False
                )
                and cache_age_s is not None
                and cache_age_s <= lost_timeout_s
            )
            if not use_cached_location:
                self.has_valid_overhead_marker_location = False
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
                (
                    f"HeroSmile:cached {cache_age_s:.1f}/"
                    f"{lost_timeout_s:.1f}s"
                ),
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
        return best_match["player"]

    def get_player_location_on_screen(self):
        """Run exactly one configured screen-space Hero locator."""
        expected_player = None
        if getattr(self, "screen_player_location_valid", False):
            expected_player = getattr(self, "loc_player", None)
        marker_cfg = (
            self.cfg.get("nametag", {}).get("overhead_marker", {})
        )
        if marker_cfg.get("enable", False):
            return self.get_player_location_by_overhead_marker(
                expected_player=expected_player, allow_global=True
            ), None

        if self.cfg["nametag"]["enable"]:
            return self.get_player_location_by_nametag(), None
        return self.get_player_location_by_party_red_bar()

    def get_player_location_by_appearance(
            self, expected_player=None, allow_global=True):
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
                max_distance = int(appearance_cfg.get(
                    "climb_validation_distance",
                    appearance_cfg.get("validation_distance", 30),
                )) if pose == "climbing" else int(
                    appearance_cfg.get("validation_distance", 30)
                )
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

    def get_nearest_color_code(self):
        '''
        Searches for the nearest color-coded action marker
        around the player on the route map.

        This function:
        - Scans each pixel in the search box to find nearest color code
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
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                dist = abs(x - x0) + abs(y - y0)
                # Get nearest color
                if pixel in self.color_code and dist < min_dist:
                    nearest = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": self.color_code[pixel],
                        "distance": dist
                    }
                    min_dist = dist
                # Get nearest color (up, dowm)
                if pixel in self.color_code_up_down and dist < min_dist_up_down:
                    nearest_up_down = {
                        "pixel": (x, y),
                        "color": pixel,
                        "command": self.color_code_up_down[pixel],
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

    def get_attack_range(self, is_left=True):
        '''
        get_attack_range
        '''
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2
            dy = self.cfg["aoe_skill"]["range_y"] // 2
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        elif self.cfg["bot"]["attack"] == "directional":
            if is_left:
                x0 = self.loc_player[0] - self.cfg["directional_attack"]["range_x"]
                x1 = self.loc_player[0]
            else:
                x0 = self.loc_player[0]
                x1 = x0 + self.cfg["directional_attack"]["range_x"]
            y0 = self.loc_player[1] - self.cfg["directional_attack"]["range_y"] // 2
            y1 = y0 + self.cfg["directional_attack"]["range_y"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")

        return (x0, y0, x1, y1)

    def get_nearest_monster(self, is_left=True):
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

        x0, y0, x1, y1 = self.get_attack_range(is_left=is_left)

        nearest_monster = None
        min_distance = float('inf')
        attack_box = (x0, y0, x1, y1)
        legacy_min_mob_area = None
        if not self.is_yolo_monster_detection() and self.monsters_info:
            legacy_min_mob_area = min(
                img.shape[0] * img.shape[1]
                for _, imgs in self.monsters_info.items()
                for img, _ in imgs
            )
        for monster in self.monsters:
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
            if inter_area >= inter_area_thres:
                # Compute distance to player center
                monster_center = detection_center(monster)
                dx = monster_center[0] - self.loc_player[0]
                dy = monster_center[1] - self.loc_player[1]
                distance = abs(dx) + abs(dy)  # Manhattan distance

                if distance < min_distance:
                    min_distance = distance
                    nearest_monster = monster

        return nearest_monster

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
        )
        monsters = self.filter_pet_yolo_detections(monsters)
        self.draw_monster_detections(monsters, (x0, y0), (x1, y1))
        return monsters

    def filter_pet_yolo_detections(self, monsters):
        """Remove a YOLO mob only when this pet's name is directly below it.

        The YOLO checkpoint has no separate pet class, so a mushroom-shaped
        pet can receive a strong ``mob`` score.  The configured pet-name image
        is identity-specific and provides a much safer negative anchor than a
        generic exclusion zone around the player.
        """
        if not monsters:
            return monsters

        nametag_cfg = self.cfg.get("nametag", {})
        pet_cfg = nametag_cfg.get("pet", {})
        pet_color = getattr(self, "img_nametag_pet", None)
        pet_gray = getattr(self, "img_nametag_pet_gray", None)
        if not (
            nametag_cfg.get("enable", False)
            and pet_cfg.get("enable", False)
            and pet_cfg.get("filter_yolo_mob", True)
            and pet_color is not None
            and pet_gray is not None
        ):
            return monsters

        if pet_gray.ndim == 3:
            pet_gray = cv2.cvtColor(pet_gray, cv2.COLOR_BGR2GRAY)
        pet_mask = get_mask(pet_color, (0, 255, 0))
        pet_h, pet_w = pet_gray.shape[:2]
        frame_gray = getattr(self, "img_frame_gray", None)
        if frame_gray is None:
            frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)
        frame_h, frame_w = frame_gray.shape[:2]

        tolerance = pet_cfg.get("yolo_name_search_tolerance", (14, 8))
        tolerance_x = max(0, int(tolerance[0]))
        tolerance_y = max(0, int(tolerance[1]))
        vertical_gap = int(pet_cfg.get("yolo_name_vertical_gap", 3))
        max_gap = max(0, int(pet_cfg.get("yolo_name_max_gap", 12)))
        threshold = float(pet_cfg.get("yolo_name_diff_thres", 0.10))

        kept = []
        for monster in monsters:
            x1, y1, x2, y2 = detection_to_box(monster)
            expected_x = int(round((x1 + x2 - pet_w) / 2))
            expected_y = y2 + vertical_gap
            roi_x1 = max(0, expected_x - tolerance_x)
            roi_y1 = max(0, expected_y - tolerance_y)
            roi_x2 = min(frame_w, expected_x + pet_w + tolerance_x)
            roi_y2 = min(frame_h, expected_y + pet_h + tolerance_y)
            roi = frame_gray[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.shape[0] < pet_h or roi.shape[1] < pet_w:
                kept.append(monster)
                continue

            local_loc, score, _ = find_pattern_sqdiff(
                roi,
                pet_gray,
                last_result=None,
                mask=pet_mask,
                global_threshold=0.0,
            )
            pet_loc = (
                roi_x1 + local_loc[0],
                roi_y1 + local_loc[1],
            )
            name_center_x = pet_loc[0] + pet_w / 2
            box_center_x = (x1 + x2) / 2
            actual_gap = pet_loc[1] - y2
            is_pet = (
                score < threshold
                and abs(name_center_x - box_center_x) <= tolerance_x
                and -tolerance_y <= actual_gap <= max_gap
            )
            if not is_pet:
                kept.append(monster)
                continue

            self._draw_debug_rectangle(
                monster["position"],
                monster["size"],
                (255, 0, 255),
                f"Pet filtered: {score:.2f}",
                thickness=1,
                text_height=0.45,
            )
            self._draw_debug_rectangle(
                pet_loc,
                (pet_h, pet_w),
                (255, 0, 255),
                "Pet name",
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
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        try:
            img_frame, geometry = preprocess_capture_frame(
                self.frame,
                self.cfg,
                window_title=getattr(self.capture, "window_title", ""),
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

        # Draw attack boxes only when a current screen-space player location is
        # available. Debug mode still shows the full-frame monster box below.
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

        # Draw minimap rectangle on img debug
        self._draw_debug_rectangle(
            self.loc_minimap,
            self.img_minimap_screen.shape[:2],
            (0, 0, 255), "minimap",thickness=2
        )

        # Modes without a route map stop after the game-window annotations.
        if self.cfg["bot"]["mode"] in ["patrol", "aux", "debug"]:
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
        if self.remote_keyboard_target():
            logger.info(
                "[ensure_is_in_party] Disabled in remote keyboard-only mode"
            )
            return False

        # open party window
        press_key(self.cfg["key"]["party"])

        # Wait party window to show up
        time.sleep(0.5)

        # Update image frame
        self.img_frame = self.get_img_frame()

        # Find the 'create party' button
        loc_enable, score_enable, _ = find_pattern_sqdiff(
                        self.img_frame, self.img_create_party_enable)

        lang = self.cfg["system"]["language"]
        thres = self.cfg['party_red_bar'][f'create_party_button_{lang}_thres']
        if score_enable < thres:
            logger.info(f"[ensure_is_in_party] Find party enable button({round(score_enable, 2)})")
            h, w = self.img_create_party_enable.shape[:2]
            self.click_game_ui(
                (loc_enable[0] + w // 2,
                 loc_enable[1] + h // 2 + self.cfg['game_window']['title_bar_height']),
                "ensure_is_in_party",
            )
        else:
            logger.info("[ensure_is_in_party] Cannot find create party button."
                        "Maybe player already in party.")

        # close party window
        press_key(self.cfg["key"]["party"])
        return True

    def channel_change(self):
        '''
        channel_change
        '''
        logger.info("[channel_change] Start")

        if self.remote_keyboard_target():
            logger.error(
                "[channel_change] Disabled in capture-card mode: changing "
                "channels requires mouse HID support on game computer B"
            )
            return False

        window_title = self.capture.window_title
        ui_coords = self.cfg["ui_coords"]
        click_in_game_window(window_title, ui_coords["menu"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel"])
        time.sleep(1)
        click_in_game_window(window_title, ui_coords["random_channel_confirm"])
        time.sleep(1)

        loc_login_button = None
        while loc_login_button is None and not self.is_terminated:
            try:
                self.img_frame = self.get_img_frame()
                loc_login_button = self.get_login_button_location()
                if loc_login_button is None:
                    logger.info("Waiting for login button to show up...")
            except Exception as e:
                logger.warning(f"Exception occurred while waiting for login button: {e}")
                if not is_mac() and self.cfg["game_window"].get("auto_resize", True):
                    resize_window(
                        window_title,
                        width=self.cfg["game_window"].get("resize_width", 1296),
                        height=self.cfg["game_window"].get("resize_height", 759),
                    )
                logger.info("Retrying login button detection...")

            time.sleep(3)

        if self.is_terminated:
            logger.info("[channel_change] Cancelled during shutdown")
            return False

        logger.info(f"login_button button found: {loc_login_button}")

        time.sleep(3)  # wait the screen to be brighter

        # Click login button
        click_in_game_window(window_title, loc_login_button)
        time.sleep(2)

        # Click "Select Character"
        click_in_game_window(window_title, ui_coords["select_character"])
        time.sleep(5)

        self.kb.enable()
        self.kb.set_command("none none none")
        self.kb.release_all_key()

        self.ensure_is_in_party() # Make sure player is in party

        self.fsm.set_init_state("hunting")
        self.t_last_attack = time.time() # Update timer
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
        # Extract the region where the login button should appear
        x0, y0 = self.cfg["ui_coords"]["login_button_top_left"]
        x1, y1 = self.cfg["ui_coords"]["login_button_bottom_right"]
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
            logger.info(f"[get_login_button_location] Found login button with score({score})")
            return (x0 + loc[0] + w // 2,
                    y0 + loc[1] + h // 2 + self.cfg['game_window']['title_bar_height'])
        else:
            return None

    def update_cmd_by_route(self):
        # Rebuild movement from this frame. If no route matches, never renew a
        # stale movement key from the prior frame on the remote ESP32.
        self.cmd_move_x = "none"
        self.cmd_move_y = "none"
        self.cmd_action = "none"

        # Get color code from the active route.
        color_code, color_code_up_down = self.get_nearest_color_code()

        # The next route may not pass through the player's current position
        # when the recorded route set is incomplete. Search each later route
        # once (with wraparound) and immediately execute the first local
        # action that can be found.
        if not color_code and not color_code_up_down and self.img_routes:
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

                color_code, color_code_up_down = \
                    self.get_nearest_color_code()
                if color_code or color_code_up_down:
                    logger.info(
                        "[route] No action near player on "
                        f"route{original_idx + 1}; recovered with "
                        f"route{candidate_idx + 1}"
                    )
                    break
            else:
                # No route describes this location. Keep a deterministic
                # route selected and all keys released; the normal stuck
                # watchdog can then perform its recovery action safely.
                self.idx_routes = original_idx
                self.img_route = self.img_routes[original_idx]
                if self.is_show_debug_window:
                    self.img_route_debug = cv2.cvtColor(
                        self.img_route, cv2.COLOR_RGB2BGR
                    )
                return

        # Use color_code and color_code_up_down to complement each other
        # To prevent character stuck at the end of ladder, we use two color color pixels
        # and let them complement with each other, to ensure smoothy ladder climbing
        if color_code and color_code_up_down:
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

    def update_cmd_by_mob_detection(self):
        # Never build an attack decision around an expired screen location.
        if not getattr(self, "screen_player_location_valid", False):
            self.monsters = []
            self.cmd_action = "none"
            return

        # Get monster search box
        margin = self.cfg["monster_detect"]["search_box_margin"]
        if self.cfg["bot"]["attack"] == "aoe_skill":
            dx = self.cfg["aoe_skill"]["range_x"] // 2 + margin
            dy = self.cfg["aoe_skill"]["range_y"] // 2 + margin
            cooldown = self.cfg["aoe_skill"]["cooldown"]
        elif self.cfg["bot"]["attack"] == "directional":
            dx = self.cfg["directional_attack"]["range_x"] + margin
            dy = self.cfg["directional_attack"]["range_y"] + margin
            cooldown = self.cfg["directional_attack"]["cooldown"]
        else:
            raise RuntimeError(f"Unsupported attack mode: {self.cfg['bot']['attack']}")
        x0 = max(0                      , self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0                      , self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monsters in the search box
        self.monsters = self.get_monsters_in_range((x0, y0), (x1, y1))

        # Check if no mob to attack
        if len(self.monsters) == 0:
            return

        # Update attack command
        if self.cfg["bot"]["attack"] == "aoe_skill":
            if time.time() - self.t_last_attack > cooldown:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()

        elif self.cfg["bot"]["attack"] == "directional":
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)
            # Determine attack direction
            attack_direction = self.get_attack_direction(monster_left, monster_right)
            # Attack Command
            keyboard_controller = getattr(self, "kb", None)
            attack_recovering = bool(
                keyboard_controller is not None
                and getattr(
                    keyboard_controller,
                    "is_attack_recovering",
                    lambda: False,
                )()
            )
            if time.time() - self.t_last_attack > cooldown and \
                    attack_direction is not None and not attack_recovering:
                self.cmd_action = "attack"
                self.t_last_attack = time.time()
                # Set up attack direction
                self.cmd_move_x = attack_direction

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
            # Switch to next route map
            self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
            logger.debug(f"Change to new route:{self.idx_routes}")

    def run_once(self):
        '''
        Process one game window frame
        '''
        # Start profiler for performance debugging
        self.profiler.start()

        # Check if need viz window
        self.is_show_debug_window = self.is_need_show_debug_window
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
                not is_mac() and not self.remote_keyboard_target():
                activate_game_window(self.capture.window_title)
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame
            self.resume_input_after_capture()

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug viz
        if self.is_show_debug_window:
            self.img_frame_debug = self.img_frame.copy()

        # Get current route image
        if self.cfg["bot"]["mode"] == "normal":
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

        if saved_geometry_status is None:
            minimap_result = get_minimap_loc_size(self.img_frame)
            if minimap_result is None:
                if not self.is_debug_mode() and \
                    not self.remote_keyboard_target() and \
                    time.time() - self.t_last_minimap_update > 30:
                    # Unable to get minimap for 30 seconds -> assume it's login screen
                    loc_login_button = self.get_login_button_location()
                    if loc_login_button:
                        logger.info("Found login button on screen. Proceed to login.")
                        self.click_game_ui(loc_login_button, "auto_login")
                        time.sleep(3)
                        self.click_game_ui(
                            self.cfg["ui_coords"]["select_character"],
                            "auto_login",
                        )
                        time.sleep(2)
            else:
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
            original_minimap_size = self.img_minimap_source.shape[:2]
            self.img_minimap = resize_minimap_to_reference(
                self.img_minimap_source,
                self.cfg,
            )
            calibrated_size = self.img_minimap.shape[:2]
            calibration_key = (original_minimap_size, calibrated_size)
            if original_minimap_size != calibrated_size and calibration_key != getattr(
                self, "_last_minimap_calibration", None
            ):
                logger.info(
                    "[minimap] Calibrated capture from "
                    f"{original_minimap_size[::-1]} to {calibrated_size[::-1]}"
                )
                self._last_minimap_calibration = calibration_key
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
            # Check if character is on ladder
            dx = abs(loc_player[0] - self.loc_player[0])
            dy = abs(loc_player[1] - self.loc_player[1])
            if self.is_on_ladder:
                if dx > 3: # Leave ladder if there is horizontal move
                    self.is_on_ladder = False
            else:
                if dx < 3 and dy != 0:
                    self.is_on_ladder = True
            # logger.info((self.is_on_ladder, dx, dy))
            # Update player location
            self.loc_player = loc_player

        # Do not draw an expired coordinate beside "location unavailable".
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
            source_h, source_w = self.img_minimap_source.shape[:2]
            map_h, map_w = self.img_minimap.shape[:2]
            self.loc_player_minimap = (
                int(round(loc_player_minimap_source[0] * map_w / source_w)),
                int(round(loc_player_minimap_source[1] * map_h / source_h)),
            )

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
            self.kb.disable()
            time.sleep(1)
            self.channel_change()
            self.red_dot_center_prev = None
            return 0

        if not self.is_debug_mode() and \
            self.is_time_to_change_channel():
            self.kb.set_command("none none none")
            self.kb.release_all_key()
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
                if self.remote_keyboard_target():
                    logger.warning(
                        "[Attack Timeout] Skipped channel change: the remote "
                        "ESP32 firmware does not provide mouse HID"
                    )
                    # Do not retry an unavailable mouse operation every frame.
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

        # Save debug window to video
        if getattr(self, "_video_record_path", None) is not None and \
                self.video_writer is None:
            self._open_video_writer_for_frame(self.img_frame_debug)
        if self.video_writer:
            self.video_writer.write(self.img_frame_debug)

        # Resize img_route_debug for better visualization
        if self.cfg["bot"]["mode"] == "normal":
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
            # PotPlayer on A is unrelated to where the BLE keyboard types.
            if not self.is_debug_mode() and \
                not is_mac() and not self.remote_keyboard_target():
                activate_game_window(self.capture.window_title)
                time.sleep(0.3)
                self.ensure_is_in_party()

            while not self.is_terminated and not self.kb.is_terminated:
                t_start = time.time()

                # Process one game window frame
                self.is_frame_done = False
                ret = self.run_once()

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
    mapleStoryAutoBot.load_config(cfg)

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
        help="Record debug window"
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
