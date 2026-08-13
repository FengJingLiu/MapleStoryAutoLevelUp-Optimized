'''
Auto generate route map 
'''
# Standard import
import time
import argparse
import sys
import os
import shutil

# CV import
import numpy as np
import cv2

# local import
from src.utils.logger import logger
from src.utils.common import (
    find_pattern_sqdiff, draw_rectangle, screenshot,
    get_minimap_loc_size, get_player_location_on_minimap,
    resize_minimap_to_reference,
    to_opencv_hsv, load_yaml, override_cfg, is_mac, load_image,
)
from src.input.KeyBoardListener import KeyBoardListener, normalize_key_name
from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.CaptureFramePreprocessor import preprocess_capture_frame


def route_action_from_pressed_keys(key_pressing, key_cfg):
    """Translate currently pressed configured keys into a route command."""
    pressed = {
        normalized
        for key_name in tuple(key_pressing)
        if (normalized := normalize_key_name(key_name))
    }
    jump_key = normalize_key_name(key_cfg.get("jump", ""))
    teleport_key = normalize_key_name(key_cfg.get("teleport", ""))

    if jump_key and jump_key in pressed:
        if "left" in pressed:
            return "left none jump", True
        if "right" in pressed:
            return "right none jump", True
        if "down" in pressed:
            return "none down jump", True
        return "none none jump", True

    if teleport_key and teleport_key in pressed:
        if "left" in pressed:
            return "left none teleport", True
        if "right" in pressed:
            return "right none teleport", True
        if "down" in pressed:
            return "none down teleport", True
        if "up" in pressed:
            return "none up teleport", True
        return "", True

    if "up" in pressed:
        return "none up none", False
    if "down" in pressed:
        return "none down none", False
    if "left" in pressed:
        return "left none none", False
    if "right" in pressed:
        return "right none none", False
    return "", False


class RouteRecorder():
    '''
    Route recorder
    '''
    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        self.fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 550
        dt_screenshot = time.time() - self.kb.t_func_key[1]
        dt_save_route = time.time() - self.kb.t_func_key[2]
        dt_save_map = time.time() - self.kb.t_func_key[3]
        text_list = [
            f"FPS: {self.fps}",
            f"Press 'F1' to {'pause' if self.is_enable else 'start'} route record",
            f"Press 'F2' to save screenshot{' : Saved' if dt_screenshot < 0.7 else ''}",
            f"Press 'F3' to save route{' : Saved' if dt_save_route < 0.7 else ''}",
            f"Press 'F4' to save map{' : Saved' if dt_save_map < 0.7 else ''}",
        ]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap.shape[:2],
            (0, 0, 255), "minimap",thickness=1
        )

        # Compute crop region with boundary check
        crop_w, crop_h = 80, 80
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # A route canvas can grow while recording. A stale or invalid player
        # coordinate must not be passed to cv2.resize as an empty crop.
        if x1 <= x0 or y1 <= y0:
            logger.warning(
                "Skip route preview: player coordinate "
                f"{self.loc_player_global} is outside route canvas "
                f"{self.img_route_debug.shape[:2][::-1]}"
            )
            return

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        if mini_map_crop.size == 0:
            return
        mini_map_crop = cv2.resize(mini_map_crop,
                                (int(mini_map_crop.shape[1] * 3),
                                 int(mini_map_crop.shape[0] * 3)),
                                interpolation=cv2.INTER_NEAREST)
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        h_frame, w_frame = self.img_frame_debug.shape[:2]
        x_paste = w_frame - w_crop - 10  # 10px margin from right
        y_paste = 70
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=2
        )

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug",
                   self.img_frame_debug[:self.cfg["ui_coords"]["ui_y_start"], :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def get_player_location_on_global_map(self):
        '''
        get_player_location_on_global_map
        '''
        loc_player_global = (
            self.loc_minimap_global[0] + self.loc_player_minimap[0],
            self.loc_minimap_global[1] + self.loc_player_minimap[1]
        )

        # Draw local minimap rectangle
        camera_bottom_right = (
            self.loc_minimap_global[0] + self.img_minimap.shape[1],
            self.loc_minimap_global[1] + self.img_minimap.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_minimap_global,
                      camera_bottom_right, (0, 255, 255), 1)
        cv2.putText(
            self.img_route_debug,
            f"Minimap,score({round(self.minimap_match_score, 2)})",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

    def replace_color_on_map(self, lower_hsv, upper_hsv, replace_color=(0, 0, 0)):
        '''
        Replace pixels in self.img_map that fall within the given HSV range
        and are part of a connected component with area > 15.
        '''
        hsv_map = cv2.cvtColor(self.img_map, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_map, to_opencv_hsv(lower_hsv), to_opencv_hsv(upper_hsv))

        # Connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 10:
                component_mask = (labels == i)
                self.img_map[component_mask] = replace_color

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
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(f"[capture] {exc}")
            return
        if geometry["profile"] == "potplayer":
            x0, y0, x1, y1 = geometry["video_roi"]
            self.img_capture_content = self.frame[y0:y1, x0:x1]
        else:
            self.img_capture_content = None
        geometry_key = (
            geometry["profile"],
            geometry["source_size"],
            geometry["video_roi"],
            geometry["content_size"],
            geometry["working_size"],
        )
        if geometry_key != getattr(self, "_last_capture_geometry", None):
            logger.info(
                "[capture] "
                f"profile={geometry['profile']} source={geometry['source_size']} "
                f"video_roi={geometry['video_roi']} "
                f"content={geometry['content_size']} "
                f"working={geometry['working_size']}"
            )
            self._last_capture_geometry = geometry_key
        return img_frame

    def __init__(self, args):
        '''
        Init MapleStoryBot
        '''
        self.args = args # User arguments
        self.idx_routes = 0 # Index of route map
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.is_enable = True
        # Coordinate (top-left coordinate)
        self.loc_minimap = (0, 0) # minimap location on game screen
        self.loc_player = (0, 0) # player location on game screen
        self.loc_player_minimap = (0, 0) # player location on minimap
        self.loc_minimap_global = (0, 0) # minimap location on global map
        self.loc_player_global = (0, 0) # player location on global map
        self.loc_player_global_last = None # playeer location on global map last frame
        self.minimap_match_score = 0.0
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = None # minimap on game screen
        self.img_map = None # map
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_draw_blob = time.time() # Last draw blob timer

        # Load defautl yaml config
        cfg = load_yaml("config/config_default.yaml")
        # Override with platform config
        if is_mac():
            cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
        # Override with user customized config
        self.cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))
        # Reference-size calibration is keyed by the logical map name. For a
        # new recording this is --new_map; when extending an existing image,
        # infer it from the parent directory if possible.
        calibration_map = args.new_map
        if args.map:
            parent_name = os.path.basename(os.path.dirname(args.map))
            if parent_name:
                calibration_map = parent_name
        self.cfg["bot"]["map"] = calibration_map

        # Parse color_code
        self.color_code = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code"].items()
        }
        color_code_up_down = {
            tuple(map(int, k.split(','))): v
            for k, v in cfg["route"]["color_code_up_down"].items()
        }
        self.color_code.update(color_code_up_down) # Combine both dictionaries

        self.fps_limit = self.cfg["system"]["fps_limit_route_recorder"]

        # Check create new map directory
        map_dir = os.path.join("minimaps", args.new_map)
        if os.path.exists(map_dir):
            user_input = input(f"[Warning] Directory '{map_dir}' already exists. Replace it? (y/n): ").strip().lower()
            if user_input == 'y':
                shutil.rmtree(map_dir)  # Delete existing directory
                logger.info(f"Removed existing directory: {map_dir}")
            else:
                sys.exit(0)
        os.makedirs(map_dir) # Create new map directory
        logger.info(f"Created new directory: {map_dir}")

        # Load exist map
        if self.args.map != '':
            self.img_map = load_image(f"{self.args.map}")

        # Start keyboard listener thread
        self.kb = KeyBoardListener(self.cfg, is_autobot=False)

        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)
        self.wait_for_initial_capture_frame()

    def wait_for_initial_capture_frame(self, timeout=2.0, poll_interval=0.05):
        """Wait briefly for the asynchronous Windows capture callback."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while time.monotonic() < deadline:
            if self.capture.get_frame() is not None:
                return True
            time.sleep(max(0.0, float(poll_interval)))
        logger.warning(
            "No game frame arrived during capture warm-up; retrying in main loop"
        )
        return False

    def ensure_img_map_capacity(self, x, y, h, w):
        '''
        Ensure that self.img_map is large enough to contain the region defined by (x, y, h, w).
        Always add at least "map_padding" when expanding in any direction.
        '''
        map_h, map_w = self.img_map.shape[:2]
        pad = self.cfg["route_recoder"]["map_padding"]

        # Compute required expansion margins
        expand_top = pad - y if y < pad else 0
        expand_left = pad - x if x < pad else 0
        expand_bottom = y + h + pad - map_h if y + h + pad > map_h else 0
        expand_right = x + w + pad - map_w if x + w + pad > map_w else 0
        expand_top = max(0, expand_top)
        expand_left = max(0, expand_left)
        expand_bottom = max(0, expand_bottom)
        expand_right = max(0, expand_right)
        # If no expansion needed, return
        if expand_top == 0 and expand_bottom == 0 and expand_left == 0 and expand_right == 0:
            return (0, 0)

        # Create new canvas and paste old image
        new_h = map_h + expand_top + expand_bottom
        new_w = map_w + expand_left + expand_right
        new_map = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        new_map[expand_top:expand_top + map_h, expand_left:expand_left + map_w] = self.img_map
        self.img_map = new_map

        # The route image uses the same global coordinate system as img_map.
        # Keep both canvases aligned whenever newly explored areas add padding.
        if self.img_route is not None:
            route_h, route_w = self.img_route.shape[:2]
            new_route = np.zeros((new_h, new_w, 3), dtype=self.img_route.dtype)
            new_route[
                expand_top:expand_top + route_h,
                expand_left:expand_left + route_w,
            ] = self.img_route
            self.img_route = new_route

        # Also update all global coordinates that depend on the map (optional)
        self.loc_minimap_global = (
            self.loc_minimap_global[0] + expand_left,
            self.loc_minimap_global[1] + expand_top
        )
        if self.loc_player_global_last is not None:
            self.loc_player_global_last = (
                self.loc_player_global_last[0] + expand_left,
                self.loc_player_global_last[1] + expand_top,
            )
        return (expand_left, expand_top)

    def remove_color_code_pixels(self, img):
        """
        Set all pixels in self.img_map to black if they match any color in color_code (assumed RGB).
        """
        for rgb in self.color_code.keys():
            bgr = (rgb[2], rgb[1], rgb[0])  # Convert RGB → BGR
            mask = np.all(img == bgr, axis=2)
            img[mask] = (0, 0, 0)
        return img

    def update_minimap(self):
        '''
        update_minimap
        '''

    def run_once(self):
        '''
        Process with one game window frame
        '''
        # Get lastest game screen frame buffer
        img_frame = self.get_img_frame()
        if img_frame is None:
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Get minimap from game window
        if self.is_first_frame:
            minimap_result = get_minimap_loc_size(self.img_frame)
            if minimap_result is None:
                logger.warning("Minimap not found; waiting for the next frame")
                return -1
            x, y, w, h = minimap_result
            # Discard 1 pixel boundary of the minimap
            x += 1
            y += 1
            w -= 2
            h -= 2
            self.loc_minimap = (x, y)
            self.minimap_screen_size = (h, w)
            self.img_minimap_screen = self.img_frame[y:y+h, x:x+w]
        else:
            x, y = self.loc_minimap
            h, w = self.minimap_screen_size
            self.img_minimap_screen = self.img_frame[y:y+h, x:x+w]

        self.img_minimap_source = self.img_minimap_screen
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
                    self.img_minimap_source = native_frame[ny:ny+nh, nx:nx+nw]
        self.img_minimap = resize_minimap_to_reference(
            self.img_minimap_source,
            self.cfg,
        )

        # Replace black pixels (0, 0, 0) with (1, 1, 1)
        black_mask = np.all(self.img_minimap == [0, 0, 0], axis=-1)
        self.img_minimap[black_mask] = [1, 1, 1]

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
        loc_player_minimap = None
        if loc_player_minimap_source:
            source_h, source_w = self.img_minimap_source.shape[:2]
            map_h, map_w = self.img_minimap.shape[:2]
            loc_player_minimap = (
                int(round(loc_player_minimap_source[0] * map_w / source_w)),
                int(round(loc_player_minimap_source[1] * map_h / source_h)),
            )
        if loc_player_minimap is None:
            now = time.monotonic()
            if now - getattr(self, "_last_player_warning_time", 0.0) >= 1.0:
                logger.warning(
                    "Player dot not found on minimap; skipping this frame"
                )
                self._last_player_warning_time = now
            return -1
        self.loc_player_minimap = loc_player_minimap

        # Update map
        if self.is_first_frame:
            # copy minimap to map
            if self.img_map is None:
                self.img_map = self.img_minimap.copy()
                pad = self.cfg["route_recoder"]["map_padding"]
                self.img_map = cv2.copyMakeBorder(
                    self.img_map,
                    top=pad, bottom=pad, left=pad, right=pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)  # Black padding
                )
                self.loc_minimap_global = (pad, pad)
                self.minimap_match_score = 0.0
            else:
                mask = np.any(
                    self.img_minimap != [0, 0, 0], axis=2
                ).astype(np.uint8) * 255
                self.loc_minimap_global, self.minimap_match_score, _ = \
                    find_pattern_sqdiff(
                        self.img_map,
                        self.img_minimap,
                        mask=mask,
                    )
                x, y = self.loc_minimap_global
                h, w = self.img_minimap.shape[:2]
                self.ensure_img_map_capacity(x, y, h, w)

            # Replace player "yellow" dot to black on map
            self.replace_color_on_map(
                (55, 40, 80),
                (60, 100, 100)
            )
            # Replace other player "red" dot to black on map
            self.replace_color_on_map((0, 80, 80),
                                      (5, 100, 100))

            # Update route
            self.img_route = self.remove_color_code_pixels(self.img_map.copy())
            self.img_route_debug = self.img_route.copy()

        else:
            # Create mask where pixels are not black
            mask = np.any(self.img_minimap != [0, 0, 0], axis=2).astype(np.uint8)
            mask = mask * 255

            # Perform template matching to find where the current minimap fits in the global map
            self.loc_minimap_global, self.minimap_match_score, _ = \
                find_pattern_sqdiff(
                self.img_map,
                self.img_minimap,
                last_result=self.loc_minimap_global,
                mask=mask
            )
            x, y = self.loc_minimap_global
            h, w = self.img_minimap.shape[:2]
            # Ensure img_map is big enough to fit the newly explored region
            self.ensure_img_map_capacity(x, y, h, w)
            x, y = self.loc_minimap_global

            # Don't copy pixel near player
            player_yellow_dot_radius = 5
            px, py = self.loc_player_minimap
            h, w = self.img_minimap.shape[:2]
            x_min = max(0, px - player_yellow_dot_radius)
            x_max = min(w, px + player_yellow_dot_radius)
            y_min = max(0, py - player_yellow_dot_radius)
            y_max = min(h, py + player_yellow_dot_radius)
            # Apply the black color mask to mask player yellow dot
            self.img_minimap[y_min:y_max, x_min:x_max] = (0, 0, 0)

            # Update map
            if self.args.map == '':
                map_slice = self.img_map[y:y+h, x:x+w]
                if map_slice.shape[:2] != self.img_minimap.shape[:2]:
                    logger.warning(
                        "Skip map update: minimap lies outside expanded map "
                        f"at {(x, y, w, h)}"
                    )
                    return -1
                black_mask = np.all(map_slice == [0, 0, 0], axis=2)
                map_slice[black_mask] = self.img_minimap[black_mask]

            # Replace other player "red" dot to black on map
            self.replace_color_on_map((0, 78, 78),
                                      (5, 100, 100))

        cv2.imshow("Map", self.img_map)
        self.img_route_debug = self.img_route.copy()

        # Get player location on global map
        self.loc_player_global = self.get_player_location_on_global_map()

        # Determine which color code to use based on user input
        action, is_draw_blob = route_action_from_pressed_keys(
            self.kb.key_pressing,
            self.cfg["key"],
        )

        # Check if need to change route
        if self.kb.is_pressed_func_key[2]: # 'F3' is pressed
            action = "none none goal"
            is_draw_blob = True
            self.kb.is_pressed_func_key[2] = False
        elif self.kb.is_pressed_func_key[0]: # 'F1' is pressed
            self.is_enable = not self.is_enable
            logger.info(f"User press F1, is_enable = {self.is_enable}")
            self.kb.is_pressed_func_key[0] = False

        # Update route image
        if self.is_enable and action != "":
            # Get color from action
            dict_action_to_color = {v: k for k, v in self.color_code.items()}
            color_rgb = dict_action_to_color.get(action, None)
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

            # Draw a line from the last position to the current one (if available)
            px, py = self.loc_player_global
            if is_draw_blob:
                dt = time.time() - self.t_last_draw_blob
                if dt > self.cfg["route_recoder"]["blob_cooldown"]:
                    # Draw a small filled circle at current position
                    cv2.circle(self.img_route,
                            (px, py),
                            radius=2,
                            color=color_bgr,
                            thickness=-1)  # filled circle
                    self.t_last_draw_blob = time.time()
                    self.loc_player_global_last = None
            else:
                if self.loc_player_global_last is None:
                    px_last, py_last = self.loc_player_global
                else:
                    px_last, py_last = self.loc_player_global_last
                cv2.line(self.img_route,
                        (px_last, py_last),
                        (px     , py),
                        color=color_bgr,
                        thickness=1)
                self.loc_player_global_last = self.loc_player_global

        # Save route image if goal is drawn
        if action == "none none goal":
            out_path = f"minimaps/{self.args.new_map}/route{self.idx_routes+1}.png"
            cv2.imwrite(out_path, self.img_route)
            self.idx_routes += 1
            self.img_route = self.img_map.copy()
            logger.info(f"Save route image to {out_path}")

        # Save img_map to map.png
        if self.kb.is_pressed_func_key[3]: # 'F4' is pressed
            out_path = f"minimaps/{self.args.new_map}/map.png"
            cv2.imwrite(out_path, self.img_map)
            self.kb.is_pressed_func_key[3] = False
            logger.info(f"Save map image to {out_path}")

        #####################
        ### Debug Windows ###
        #####################
        # Print text on debug image
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Check if need to save screenshot
        if self.kb.is_pressed_func_key[1]: # 'F2' is pressed
            screenshot(self.img_frame)
            self.kb.is_pressed_func_key[1] = False

        # Resize img_route_debug for better visualization
        self.img_route_debug = cv2.resize(
                    self.img_route_debug, (0, 0),
                    fx=self.cfg["minimap"]["debug_window_upscale"],
                    fy=self.cfg["minimap"]["debug_window_upscale"],
                    interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Route Map Debug", self.img_route_debug)

        # Enable cached location since second frame
        self.is_first_frame = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Argument to specify map name
    parser.add_argument(
        '--new_map',
        type=str,
        default='new_map',
        help='Specify the new map name'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    parser.add_argument(
        '--map',
        type=str,
        default='',
        help='use this map instead of creating a new one'
    )

    try:
        routeRecorder = RouteRecorder(parser.parse_args())
    except Exception as e:
        logger.error(f"RouteRecorder Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            t_start = time.time()

            # Process one game window frame
            routeRecorder.run_once()

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / routeRecorder.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)

        cv2.destroyAllWindows()
