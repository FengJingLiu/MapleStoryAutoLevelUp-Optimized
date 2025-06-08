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

#
import numpy as np
import cv2

# local import
from config import Config
from logger import logger
from util import find_pattern_sqdiff, draw_rectangle, screenshot, nms, load_image, get_mask
from KeyBoardController import KeyBoardController
from GameWindowCapturor import GameWindowCapturor

class MapleStoryBot:
    '''
    MapleStoryBot
    '''
    def __init__(self, args):
        self.cfg = Config # Configuration
        self.args = args
        self.status = "hunting" # 'resting', 'finding_rune', 'near_rune', 'solving_rune'
        self.idx_routes = 0 # Index of route
        self.hp_ratio = 1.0 # HP bar ratio
        self.mp_ratio = 1.0 # MP bar ratio
        self.exp_ratio = 1.0 # EXP bar ratio
        self.monster_info = [] # monster information
        # Coordinate (top-left coordinate)
        self.loc_nametag = (0, 0) # nametag location on window
        self.loc_camera = (0, 0) # camera location on map
        self.loc_watch_dog = (0, 0) # watch dog
        self.loc_player_global = (0, 0) # player location on map
        self.loc_player = (0, 0) # player location on window
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None
        self.img_route = None # route map
        self.img_route_debug = None
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation
        self.t_last_switch_status = time.time() # Last status switches timer
        self.t_watch_dog = time.time() # Last movement timer
        self.t_last_teleport = time.time() # Last teleport timer

        # Set status to hunting for start
        self.switch_status("hunting")

        # Load map for camera localization
        self.img_map = load_image(f"maps/{args.map}/map.png",
                                  cv2.IMREAD_GRAYSCALE)
        self.img_map_resized = cv2.resize(
            self.img_map, (0, 0),
            fx=self.cfg.localize_downscale_factor,
            fy=self.cfg.localize_downscale_factor)

        # Load route*.png images
        route_files = sorted(glob.glob(f"maps/{args.map}/route*.png"))
        route_files = [p for p in route_files if not p.endswith("route_rest.png")]
        self.img_routes = [
            cv2.cvtColor(load_image(p), cv2.COLOR_BGR2RGB) for p in route_files
        ]
        # Load rest route
        self.img_route_rest = cv2.cvtColor(
            load_image(f"maps/{args.map}/route_rest.png"), cv2.COLOR_BGR2RGB)

        # Load other images
        self.img_nametag = load_image("name_tag.png")
        self.img_nametag_gray = load_image("name_tag.png", cv2.IMREAD_GRAYSCALE)
        self.img_rune_warning = load_image("rune/rune_warning.png", cv2.IMREAD_GRAYSCALE)
        self.img_rune = load_image("rune/rune.png")
        self.img_rune_gray = load_image("rune/rune.png", cv2.IMREAD_GRAYSCALE)
        self.img_arrows = {
            "left":
                [load_image("rune/arrow_left_1.png"),
                load_image("rune/arrow_left_2.png"),
                load_image("rune/arrow_left_3.png"),],
            "right":
                [load_image("rune/arrow_right_1.png"),
                load_image("rune/arrow_right_2.png"),
                load_image("rune/arrow_right_3.png"),],
            "up":
                [load_image("rune/arrow_up_1.png"),
                load_image("rune/arrow_up_2.png"),
                load_image("rune/arrow_up_3.png")],
            "down":
                [load_image("rune/arrow_down_1.png"),
                load_image("rune/arrow_down_2.png"),
                load_image("rune/arrow_down_3.png"),],
        }

        # Load monsters images
        self.monsters = {}
        for monster_name in args.monsters.split(","):
            imgs = []
            for file in glob.glob(f"monster/{monster_name}*.png"):
                # Add original image
                img = load_image(file)
                imgs.append((img, get_mask(img, (0, 255, 0))))
                # Add flipped image
                img_flip = cv2.flip(img, 1)
                imgs.append((img_flip, get_mask(img_flip, (0, 255, 0))))
            if imgs:
                self.monsters[monster_name] = imgs
            else:
                logger.error(f"No images found in monster/{monster_name}*")
                raise RuntimeError(f"No images found in monster/{monster_name}*")
        logger.info(f"Loaded monsters: {list(self.monsters.keys())}")

        # Start keyboard controller thread
        self.kb = KeyBoardController(self.cfg)
        if args.disable_control:
            self.kb.disable()

        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)

    def switch_status(self, new_status):
        '''
        Switch to new status and log the transition.

        Parameters:
        - new_status: string, the new status to switch to.
        '''
        # Ignore dummy transition
        if self.status == new_status:
            return

        t_elapsed = round(time.time() - self.t_last_switch_status)
        logger.info(f"[switch_status] From {self.status}({t_elapsed} sec) to {new_status}.")
        self.status = new_status
        self.t_last_switch_status = time.time()

    def get_nearest_monster(self, is_left = True, overlap_threshold=0.5):
        '''
        get_nearest_monster
        '''
        if is_left:
            x0 = self.loc_player[0] - self.cfg.magic_claw_range_x
        else:
            x0 = self.loc_player[0]
        y0 = self.loc_player[1] - self.cfg.magic_claw_range_y//2
        x1 = x0 + self.cfg.magic_claw_range_x
        y1 = y0 + self.cfg.magic_claw_range_y

        # Debug, magic claw hit box
        draw_rectangle(
            self.img_frame_debug, (x0, y0),
            (self.cfg.magic_claw_range_y, self.cfg.magic_claw_range_x),
            (0, 0, 255), "Attack Box"
        )

        nearest_monster = None
        min_distance = float('inf')
        for monster in self.monster_info:
            mx1, my1 = monster["position"]
            mw, mh = monster["size"]
            mx2 = mx1 + mw
            my2 = my1 + mh

            # Calculate intersection
            ix1 = max(x0, mx1)
            iy1 = max(y0, my1)
            ix2 = min(x1, mx2)
            iy2 = min(y1, my2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter_area = iw * ih

            monster_area = mw * mh
            if monster_area == 0:
                continue  # skip degenerate box

            if inter_area/monster_area >= overlap_threshold:
                # Compute distance to player center
                monster_center = (mx1 + mw // 2, my1 + mh // 2)
                dx = monster_center[0] - self.loc_player[0]
                dy = monster_center[1] - self.loc_player[1]
                distance = abs(dx) + abs(dy)  # Manhattan distance

                if distance < min_distance:
                    min_distance = distance
                    nearest_monster = monster

        return nearest_monster

    def solve_rune(self):
        '''
        Solve the rune puzzle by detecting the arrow directions and pressing corresponding keys.
        '''
        while self.is_in_rune_game():
            for arrow_idx in [0,1,2,3]:
                # Get lastest game screen frame buffer
                self.frame = self.capture.get_frame()
                # Resize game screen to 1296x759
                self.img_frame = cv2.resize(self.frame, (1296, 759),
                                            interpolation=cv2.INTER_NEAREST)

                # Crop arrow detection box
                x = self.cfg.arrow_box_start_point[0] + self.cfg.arrow_box_interval*arrow_idx
                y = self.cfg.arrow_box_start_point[1]
                size = self.cfg.arrow_box_size
                img_roi = self.img_frame[y:y+size, x:x+size]

                # Loop through all possible arrows template and choose the most possible one
                best_score = float('inf')
                best_direction = ""
                for direction, arrow_list in self.img_arrows.items():
                    for img_arrow in arrow_list:
                        _, score, _ = find_pattern_sqdiff(
                                        img_roi, img_arrow,
                                        mask=get_mask(img_arrow, (0, 255, 0)))
                        if score < best_score:
                            best_score = score
                            best_direction = direction
                logger.info(f"[solve_rune] Arrow({arrow_idx}) is {best_direction} with score({best_score})")

                # Update img_frame_debug
                self.img_frame_debug = self.img_frame.copy()
                draw_rectangle(
                    self.img_frame_debug, (x, y), (size, size),
                    (0, 0, 255), str(round(best_score, 2))
                )
                # Update debug window
                self.update_img_frame_debug()
                cv2.waitKey(1)

                # For logging
                screenshot(self.img_frame_debug, f"solve_rune")

                # Press the key for 0.5 second
                if not self.args.disable_control:
                    self.kb.press_key(best_direction, 0.5)
                time.sleep(1)


        logger.info(f"[solve_rune] Solved all arrows")

    def is_player_stuck(self):
        """
        Detect if the player is stuck (not moving).
        If stuck for more than WATCH_DOG_TIMEOUT seconds, performs a random action.
        """
        dx = abs(self.loc_player_global[0] - self.loc_watch_dog[0])
        dy = abs(self.loc_player_global[1] - self.loc_watch_dog[1])

        current_time = time.time()
        if dx + dy > self.cfg.watch_dog_range:
            # Player moved, reset watchdog timer
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            return False

        dt = current_time - self.t_watch_dog
        if dt > self.cfg.watch_dog_timeout:
            # watch dog idle for too long, player stuck
            self.loc_watch_dog = self.loc_player_global
            self.t_watch_dog = current_time
            logger.warning(f"[is_player_stuck] Player stuck for {dt} seconds.")
            return True
        return False

    def get_nearest_color_code(self):
        '''
        get_nearest_color_code
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg.color_code_search_range)
        x_max = min(w, x0 + self.cfg.color_code_search_range)
        y_min = max(0, y0 - self.cfg.color_code_search_range)
        y_max = min(h, y0 + self.cfg.color_code_search_range)

        nearest = None
        min_dist = float('inf')
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                pixel = tuple(self.img_route[y, x])  # (R, G, B)
                if pixel in self.cfg.color_code:
                    dist = abs(x - x0) + abs(y - y0)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = {
                            "pixel": (x, y),
                            "color": pixel,
                            "action": self.cfg.color_code[pixel],
                            "distance": dist
                        }

        # Debug
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (self.cfg.color_code_search_range*2, self.cfg.color_code_search_range*2),
            (0, 0, 255), "Color Search Range"
        )
        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if nearest is not None:
            cv2.line(
                self.img_route_debug,
                self.loc_player_global, # start point
                nearest["pixel"],       # end point
                (0, 255, 0),            # green line
                1                       # thickness
            )

            # Print color code on debug image
            cv2.putText(
                self.img_frame_debug, f"Route Action: {nearest["action"]}",
                (720, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )
            cv2.putText(
                self.img_frame_debug, f"Route Index: {self.idx_routes}",
                (720, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        return nearest  # if not found return none

    def get_hp_mp(self):
        '''
        get_hp_mp
        '''
        # HP crop
        hp_bar = self.img_frame[self.cfg.hp_bar_top_left[1]:self.cfg.hp_bar_bottom_right[1]+1,
                                self.cfg.hp_bar_top_left[0]:self.cfg.hp_bar_bottom_right[0]+1]
        # MP crop
        mp_bar = self.img_frame[self.cfg.mp_bar_top_left[1]:self.cfg.mp_bar_bottom_right[1]+1,
                                self.cfg.mp_bar_top_left[0]:self.cfg.mp_bar_bottom_right[0]+1]
        # EXP crop
        exp_bar = self.img_frame[self.cfg.exp_bar_top_left[1]:self.cfg.exp_bar_bottom_right[1]+1,
                                self.cfg.exp_bar_top_left[0]:self.cfg.exp_bar_bottom_right[0]+1]
        # HP Detection (detect empty part)
        empty_mask_hp = (hp_bar[:,:,0] == hp_bar[:,:,1]) & (hp_bar[:,:,0] == hp_bar[:,:,2])
        empty_pixels_hp = np.count_nonzero(empty_mask_hp)-6 # 6 pixel always be white
        total_pixels_hp = hp_bar.shape[0] * hp_bar.shape[1] - 6
        hp_ratio = 1 - (empty_pixels_hp / total_pixels_hp)

        # MP Detection (detect empty part)
        empty_mask_mp = (mp_bar[:,:,0] == mp_bar[:,:,1]) & (mp_bar[:,:,0] == mp_bar[:,:,2])
        empty_pixels_mp = np.count_nonzero(empty_mask_mp)-6 # 6 pixel always be white
        total_pixels_mp = mp_bar.shape[0] * mp_bar.shape[1] - 6
        mp_ratio = 1 - (empty_pixels_mp / total_pixels_mp)

        # EXP Detection (detect eexpty part)
        empty_mask_exp = (exp_bar[:,:,0] == exp_bar[:,:,1]) & (exp_bar[:,:,0] == exp_bar[:,:,2])
        eexpty_pixels_exp = np.count_nonzero(empty_mask_exp)-6 # 6 pixel always be white
        total_pixels_exp = exp_bar.shape[0] * exp_bar.shape[1] - 6
        exp_ratio = 1 - (eexpty_pixels_exp / total_pixels_exp)

        # Compute original bar dimensions
        hp_h, hp_w = hp_bar.shape[:2]
        mp_h, mp_w = mp_bar.shape[:2]
        exp_h, exp_w = exp_bar.shape[:2]

        # Paste HP bar directly at top-left
        self.img_frame_debug[75:75+hp_h, 180:180+hp_w] = hp_bar

        # Paste MP bar below HP bar
        self.img_frame_debug[105:105+mp_h, 180:180+mp_w] = mp_bar

        # Paste EXP bar below MP bar
        self.img_frame_debug[135:135+exp_h, 180:180+exp_w] = exp_bar

        # Overlay HP/MP/EXP text
        cv2.putText(self.img_frame_debug, f"HP: {hp_ratio*100:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(self.img_frame_debug, f"MP: {mp_ratio*100:.1f}%", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(self.img_frame_debug, f"EXP: {exp_ratio*100:.1f}%", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        return hp_ratio, mp_ratio, exp_ratio

    def is_rune_warning(self):
        '''
        is_rune_warning
        '''
        x0, y0 = self.cfg.rune_warning_top_left
        x1, y1 = self.cfg.rune_warning_bottom_right
        _, score, _ = find_pattern_sqdiff(
                        self.img_frame_gray[y0:y1, x0:x1],
                        self.img_rune_warning)
        if self.status == "hunting" and score < self.cfg.rune_warning_diff_thres:
            logger.info(f"[is_rune_warning] Detect rune warning on screen with score({score})")
            return True
        else:
            return False

    def is_rune_near_player(self):
        '''
        is_rune_near_player
        '''
        # Calculate bounding box
        h, w = self.img_frame.shape[:2]
        x0 = max(0, self.loc_player[0] - self.cfg.rune_detect_box_width // 2)
        y0 = max(0, self.loc_player[1] - self.cfg.rune_detect_box_height)
        x1 = min(w, self.loc_player[0] + self.cfg.rune_detect_box_width // 2)
        y1 = min(h, self.loc_player[1])

        # Debug
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Rune Detection Range"
        )

        # Find rune icon near player
        if  (x1 - x0) < self.img_rune.shape[1] or \
            (y1 - y0) < self.img_rune.shape[0]:
            return False # Skip check if box is out of range
        else:
            img_roi = self.img_frame[y0:y1, x0:x1]
            loc_rune, score, _ = find_pattern_sqdiff(
                            img_roi,
                            self.img_rune,
                            mask=get_mask(self.img_rune, (0, 255, 0)))
            if score < self.cfg.rune_detect_diff_thres:
                logger.info(f"[Rune Detect]Found rune near player with score({score})")
                # Draw rectangle for debug
                draw_rectangle(
                    self.img_frame_debug,
                    (x0 + loc_rune[0], y0 + loc_rune[1]),
                    self.img_rune.shape,
                    (255, 0, 255),  # purple in BGR
                    "Rune"
                )
                screenshot(self.img_frame_debug, "rune_detected")

                return True
            else:
                return False

    def is_in_rune_game(self):
        '''
        is_in_rune_game
        '''
        # Get lastest game screen frame buffer
        self.frame = self.capture.get_frame()
        # Resize game screen to 1296x759
        self.img_frame = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)

        # Crop arrow detection box
        x, y = self.cfg.arrow_box_start_point
        size = self.cfg.arrow_box_size
        img_roi = self.img_frame[y:y+size, x:x+size]

        # Check if arrow exist on screen
        best_score = float('inf')
        for direc, arrow_list in self.img_arrows.items():
            for img_arrow in arrow_list:
                _, score, _ = find_pattern_sqdiff(
                                img_roi, img_arrow,
                                mask=get_mask(img_arrow, (0, 255, 0)))
                if score < best_score:
                    best_score = score

        draw_rectangle(
            self.img_frame_debug, (x, y), (size, size),
            (0, 0, 255), str(round(best_score, 2))
        )

        if best_score < self.cfg.arrow_box_diff_thres:
            logger.info(f"Arrow screen detected with score({score})")
            return True
        return False

    def get_monsters_in_range(self, top_left, bottom_right):
        '''
        get_monsters_in_range
        '''
        x0, y0 = top_left
        x1, y1 = bottom_right

        img_roi = self.img_frame[y0:y1, x0:x1]

        monster_info = []
        for monster_name, monster_imgs in self.monsters.items():
            for img_monster, mask_monster in monster_imgs:
                if self.cfg.monster_detect_mode == "contour_only":
                    # Use only black lines contour to detect monsters
                    # Create masks (already grayscale)
                    mask_pattern = np.all(img_monster == [0, 0, 0], axis=2).astype(np.uint8) * 255
                    mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

                    # Apply Gaussian blur (soften the masks)
                    img_monster_blur = cv2.GaussianBlur(mask_pattern, (self.cfg.blur_range, self.cfg.blur_range), 0)
                    img_roi_blur = cv2.GaussianBlur(mask_roi, (self.cfg.blur_range, self.cfg.blur_range), 0)

                    # Check template vs ROI size before matching
                    h_roi, w_roi = img_roi_blur.shape[:2]
                    h_temp, w_temp = img_monster_blur.shape[:2]

                    if h_temp > h_roi or w_temp > w_roi:
                        return []  # template bigger than roi, skip this matching

                    # Perform template matching
                    res = cv2.matchTemplate(img_roi_blur, img_monster_blur, cv2.TM_SQDIFF_NORMED)

                    # Apply soft threshold
                    match_locations = np.where(res <= self.cfg.monster_diff_thres)

                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                        })
                elif self.cfg.monster_detect_mode == "grayscale":
                    img_monster_gray = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
                    img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
                    res = cv2.matchTemplate(
                            img_roi_gray,
                            img_monster_gray,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg.monster_diff_thres)
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                elif self.cfg.monster_detect_mode == "color":
                    res = cv2.matchTemplate(
                            img_roi,
                            img_monster,
                            cv2.TM_SQDIFF_NORMED,
                            mask=mask_monster)
                    match_locations = np.where(res <= self.cfg.monster_diff_thres)
                    h, w = img_monster.shape[:2]
                    for pt in zip(*match_locations[::-1]):
                        monster_info.append({
                            "name": monster_name,
                            "position": (pt[0] + x0, pt[1] + y0),
                            "size": (h, w),
                            "score": res[pt[1], pt[0]],
                    })
                else:
                    logger.error(f"Unexpected camera localization mode: {self.cfg.monster_detect_mode}")
                    return []

        # Apply Non-Maximum Suppression to monster detection
        monster_info = nms(monster_info, iou_threshold=0.4)

        # Detect monster via health bar
        if self.cfg.monster_detect_with_health_bar:
            # Create color mask for Monsters' HP bar
            mask = cv2.inRange(img_roi,
                               np.array(self.cfg.monster_health_bar_color),
                               np.array(self.cfg.monster_health_bar_color))

            # Find connected components (each cluster of green pixels)
            num_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(mask, connectivity=8)

            for i in range(1, num_labels):  # skip background (label 0)
                x, y, w, h, area = stats[i]
                if area < 3:  # small noise filter
                    continue

                # Add padding  to make viz prettier
                pad_x = 5
                pad_y = 5
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(img_roi.shape[1] - x, w + 2 * pad_x)
                h = min(img_roi.shape[0] - y, h + 2 * pad_y)

                # Append new monster to list
                monster_info.append({
                    "name": "Unknown",
                    "position": (x0 + x, y0 + y),
                    "size": (h, w),
                    "score": 1.0,
                })

        # Debug
        # Draw attack detection range
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (y1-y0, x1-x0),
            (255, 0, 0), "Monster Detection Box"
        )

        # Draw monsters bounding box
        for monster in monster_info:
            draw_rectangle(
                self.img_frame_debug, monster["position"], monster["size"],
                (0, 255, 0), str(round(monster['score'], 2))
            )

        return monster_info

    def get_player_location(self):
        '''
        get player location by detecting player's nametag
        '''
        img_roi = self.img_frame_gray[self.cfg.camera_ceiling:self.cfg.camera_floor, :]

        # Pad search region to avoid edge cut-off issue (full template size)
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi_padded = cv2.copyMakeBorder(
            img_roi,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Adjust previous location
        last_result = (
            self.loc_nametag[0] + pad_x,
            self.loc_nametag[1] - self.cfg.camera_ceiling + pad_y
        )

        # Perform template matching
        loc_nametag, score, is_cached = find_pattern_sqdiff(
            img_roi_padded,
            self.img_nametag_gray,
            last_result=last_result,
            mask=get_mask(self.img_nametag, (0, 255, 0)),
            global_threshold=0.3
        )

        # Convert back to original (unpadded) coordinates
        loc_nametag = (
            loc_nametag[0] - pad_x,
            loc_nametag[1] - pad_y + self.cfg.camera_ceiling
        )

        # Update name tag location if confidence is good
        if score < self.cfg.nametag_diff_thres:
            self.loc_nametag = loc_nametag
        loc_player = (
            self.loc_nametag[0] - self.cfg.nametag_offset[0],
            self.loc_nametag[1] - self.cfg.nametag_offset[1]
        )

        # Draw name tag detection box for debug
        draw_rectangle(
            self.img_frame_debug, self.loc_nametag, self.img_nametag.shape,
            (0, 255, 0), "")
        text = f"NameTag, {round(score, 2)}, {'cached' if is_cached else 'missed'}"
        cv2.putText(self.img_frame_debug, text,
                    (self.loc_nametag[0], self.loc_nametag[1] + self.img_nametag.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw player center
        cv2.circle(self.img_frame_debug,
                loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)

        return loc_player

    def get_player_location_global(self):
        '''
        get_player_location_global
        '''
        scale_factor = self.cfg.localize_downscale_factor
        # Downscale both template and search image
        img_roi = self.img_frame_gray[self.cfg.camera_ceiling:self.cfg.camera_floor, :]
        img_query = cv2.resize(img_roi, (0, 0), fx=scale_factor, fy=scale_factor)
        last_result = (
            int(self.loc_camera[0] * scale_factor),
            int(self.loc_camera[1] * scale_factor)
        )

        loc_camera, score, is_cached = find_pattern_sqdiff(
            self.img_map_resized,
            img_query,
            last_result=last_result,
            local_search_radius=20,
            global_threshold = 0.8)
        # if score < self.cfg.localize_diff_thres:
        self.loc_camera = (
            int(loc_camera[0] / scale_factor),
            int(loc_camera[1] / scale_factor)
        )
        loc_player_global = (
            self.loc_camera[0] + self.loc_player[0],
            self.loc_camera[1] + self.loc_player[1] - self.cfg.camera_ceiling)

        # Draw camera rectangle
        camera_bottom_right = (
            self.loc_camera[0] + self.img_frame.shape[1],
            self.loc_camera[1] + self.img_frame.shape[0]
        )
        cv2.rectangle(self.img_route_debug, self.loc_camera,
                      camera_bottom_right, (0, 255, 255), 2)
        cv2.putText(
            self.img_route_debug,
            f"Camera, score={round(score, 2)}, {'cached' if is_cached else 'missed'}",
            (self.loc_camera[0], self.loc_camera[1] + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2,
            (0, 255, 0), 2
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=3,
                   color=(0, 0, 255), thickness=-1)
        cv2.putText(self.img_route_debug, "Player",
                    (loc_player_global[0] - 30, loc_player_global[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return loc_player_global

    def is_near_edge(self):
        '''
        is_near_edge
        '''
        x0, y0 = self.loc_player_global
        h, w = self.img_route.shape[:2]
        x_min = max(0, x0 - self.cfg.edge_teleport_box_width//2)
        x_max = min(w, x0 + self.cfg.edge_teleport_box_width//2)
        y_min = max(0, y0 - self.cfg.edge_teleport_box_height//2)
        y_max = min(h, y0 + self.cfg.edge_teleport_box_height//2)

        # Debug: draw search box
        draw_rectangle(
            self.img_route_debug,
            (x_min, y_min),
            (y_max - y_min, x_max - x_min),
            (0, 0, 255), "Edge Check"
        )

        # Find mask of matching pixels
        roi = self.img_route[y_min:y_max, x_min:x_max]
        mask = np.all(roi == self.cfg.edge_teleport_color_code, axis=2)
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

    def get_random_action(self):
        '''
        get_random_action
        '''
        action = random.choice(list(self.cfg.color_code.values()))
        logger.warning(f"Perform random action: {action}")
        return action

    def update_info_on_img_frame_debug(self):
        '''
        update_info_on_img_frame_debug
        '''
        # Print text at bottom left corner
        fps = round(1.0 / (time.time() - self.t_last_frame))
        text_y_interval = 23
        text_y_start = 550
        dt_screenshot = time.time() - self.kb.t_last_screenshot
        text_list = [
            f"FPS: {fps}",
            f"Status: {self.status}",
            f"Press 'F1' to {"pause" if self.kb.is_enable else "start"} Bot",
            f"Press 'F2' to save screenshot{" : Saved" if dt_screenshot < 0.7 else ""}"]
        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (10, text_y_start + text_y_interval*idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
                2, cv2.LINE_AA
            )

        # mini-map on debug image
        # Compute crop region with boundary check
        crop_w, crop_h = 400, 400
        x0 = max(0, self.loc_player_global[0] - crop_w // 2)
        y0 = max(0, self.loc_player_global[1] - crop_h // 2)
        x1 = min(self.img_route_debug.shape[1], x0 + crop_w)
        y1 = min(self.img_route_debug.shape[0], y0 + crop_h)

        # Crop region
        mini_map_crop = self.img_route_debug[y0:y1, x0:x1]
        mini_map_crop = cv2.resize(mini_map_crop,
                                (mini_map_crop.shape[1] // 2, mini_map_crop.shape[0] // 2),
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
                   self.img_frame_debug[self.cfg.camera_ceiling:self.cfg.camera_floor, :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def run_once(self):
        '''
        Process with one game window frame
        '''
        # Get lastest game screen frame buffer
        self.frame = self.capture.get_frame()

        # Get current route image
        self.img_route = self.img_routes[self.idx_routes]

        # Resize game screen to 1296x759
        self.img_frame = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()
        self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        # Detect HP/MP/EXP bar on UI
        self.hp_ratio, self.mp_ratio, self.exp_ratio = self.get_hp_mp()

        # Check whether "PLease remove runes" warning appears on screen
        if self.is_rune_warning():
            self.switch_status("finding_rune")

        # Get player location in game window
        self.loc_player = self.get_player_location()

        # Get player location on map
        self.loc_player_global = self.get_player_location_global()

        # Check whether a rune icon is near player
        if self.is_rune_near_player():
            self.switch_status("near_rune")

        # Check whether we entered the rune mini-game
        if self.status == "near_rune":
            # stop character
            self.kb.set_command("stop")
            time.sleep(0.1) # Wait for character to stop
            self.kb.disable() # Disable kb thread during rune solving

            # Attempt to trigger rune
            if not self.args.disable_control:
                self.kb.press_key("up", 0.02)
            time.sleep(1) # Wait rune game to pop up

            # If entered the game, start solving rune
            if self.is_in_rune_game():
                self.solve_rune() # Blocking until runes solved
                self.switch_status("hunting")

            # Restore kb thread
            self.kb.enable()

        # Get all monster near player
        if self.cfg.is_use_aoe:
            # Search monster near player
            x0 = max(0, self.loc_player[0] - self.cfg.aoe_skill_range_x//2)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + self.cfg.aoe_skill_range_x//2)
            y0 = max(0, self.loc_player[1] - self.cfg.aoe_skill_range_y//2)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + self.cfg.aoe_skill_range_y//2)
        else:
            # Search monster nearby magic claw range
            dx = self.cfg.magic_claw_range_x + self.cfg.monster_search_margin
            dy = self.cfg.magic_claw_range_y + self.cfg.monster_search_margin
            x0 = max(0, self.loc_player[0] - dx)
            x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
            y0 = max(0, self.loc_player[1] - dy)
            y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        # Get monster in skill range
        self.monster_info = self.get_monsters_in_range((x0, y0), (x1, y1))

        if self.cfg.is_use_aoe:
            if len(self.monster_info) == 0:
                attack_direction = None
            else:
                attack_direction = "I don't care"
        else:
            # Get nearest monster to player
            monster_left  = self.get_nearest_monster(is_left = True)
            monster_right = self.get_nearest_monster(is_left = False)

            # Compute distance for left
            distance_left = float('inf')
            if monster_left is not None:
                mx, my = monster_left["position"]
                mw, mh = monster_left["size"]
                center_left = (mx + mw // 2, my + mh // 2)
                distance_left = abs(center_left[0] - self.loc_player[0]) + \
                                abs(center_left[1] - self.loc_player[1])

            # Compute distance for right
            distance_right = float('inf')
            if monster_right is not None:
                mx, my = monster_right["position"]
                mw, mh = monster_right["size"]
                center_right = (mx + mw // 2, my + mh // 2)
                distance_right = abs(center_right[0] - self.loc_player[0]) + \
                                abs(center_right[1] - self.loc_player[1])

            # Choose attack direction
            attack_direction = None
            if distance_left < distance_right:
                attack_direction = "left"
            elif distance_right < distance_left:
                attack_direction = "right"

        # get color code from img_route
        command = ""
        color_code = self.get_nearest_color_code()
        if color_code:
            if color_code["action"] == "goal":
                # Switch to next route map
                self.idx_routes = (self.idx_routes+1)%len(self.img_routes)
                logger.debug(f"Change to new route:{self.idx_routes}")
            command = color_code["action"]

        # teleport away from edge to avoid fall off
        if self.is_near_edge() and \
            time.time() - self.t_last_teleport > self.cfg.teleport_cooldown:
            command = command.replace("walk", "teleport")
            self.t_last_teleport = time.time() # update timer

        # Special logic for each status, overwrite color code action
        if self.status == "hunting":
            if self.is_player_stuck():
                # Perform a random action when player stuck
                command = self.get_random_action()
            elif command in ["up", "down"]:
                pass # Don't attack or heal while character is on rope
            elif self.hp_ratio <= self.cfg.heal_ratio:
                command = "heal"
            elif self.mp_ratio <= self.cfg.add_mp_ratio:
                command = "add mp"
            elif attack_direction == "I don't care":
                command = "attack"
            elif attack_direction == "left":
                command = "attack left"
            elif attack_direction == "right":
                command = "attack right"
            # WIP: teleport while walking is unstable 
            # elif command[:4] == "walk":
            #     if self.cfg.is_use_teleport_to_walk and \
            #         time.time() - self.t_last_teleport > self.cfg.teleport_cooldown:
            #         command = command.replace("walk", "teleport")
            #         self.t_last_teleport = time.time() # update timer

        elif self.status == "finding_rune":
            if self.is_player_stuck():
                command = self.get_random_action()
            # Check if finding rune timeout
            if time.time() - self.t_last_switch_status > self.cfg.rune_finding_timeout:
                self.switch_status("resting")

        elif self.status == "near_rune":
            # Stay in near_rune status for only a few seconds
            if time.time() - self.t_last_switch_status > self.cfg.near_rune_duration:
                self.switch_status("hunting")

        elif self.status == "resting":
            self.img_routes = [self.img_route_rest] # Set up resting route
            self.idx_routes = 0

        else:
            logger.error(f"Unknown status: {self.status}")

        # send command to keyboard controller
        self.kb.set_command(command)

        #############
        ### Debug ###
        #############
        # Print text on debug image 
        self.update_info_on_img_frame_debug()

        # Show debug image on window
        self.update_img_frame_debug()

        # Check if need to save screenshot 
        if self.kb.is_need_screen_shot:
            screenshot(mapleStoryBot.img_frame)
            self.kb.is_need_screen_shot = False

        # Resize img_route_debug for better visualization
        h, w = self.img_route_debug.shape[:2]
        self.img_route_debug = cv2.resize(self.img_route_debug, (w // 2, h // 2),
                                interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Route Map Debug", self.img_route_debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable fake keyboard input'
    )

    # TODO: WIP
    parser.add_argument(
        '--patrol',
        action='store_true',
        help='Enable patrol mode'
    )

    # Argument to specify map name
    parser.add_argument(
        '--map',
        type=str,
        default='ant_cave_2',
        help='Specify the map name'
    )

    parser.add_argument(
        "--monsters",
        type=str,
        default="all",
        help="Specify which monsters to load, comma-separated"
             "(e.g., --monsters green_mushroom,zombie_mushroom)"
    )

    try:
        mapleStoryBot = MapleStoryBot(parser.parse_args())
    except Exception as e:
        logger.error(f"MapleStoryBot Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            mapleStoryBot.run_once()

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF  # Get the pressed key (8-bit)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
