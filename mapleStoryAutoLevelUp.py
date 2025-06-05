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

import numpy as np
import cv2

# local import
from config import Config
from logger import logger
from util import find_pattern_sqdiff, draw_rectangle, screenshot, nms
from KeyBoardController import KeyBoardController
from GameWindowCapturor import GameWindowCapturor

class MapleStoryBot:
    '''
    MapleStoryBot
    '''
    def __init__(self, args):
        self.cfg = Config # Configuration
        self.status = "hunting" # 'resting', 'finding_rune', 'near_rune', 'solving_rune'
        self.idx_routes = 0 # Index of route
        self.hp_ratio = 1.0 # HP bar ratio
        self.mp_ratio = 1.0 # MP bar ratio
        self.monster_info = []
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
        self.t_last_frame = time.time() # Last frame timestamp, for fps calculation
        self.t_last_switch_status = time.time() # timestamp when status switches last time
        self.t_watch_dog = time.time()

        # Set status to hunting
        self.switch_status("hunting")

        # Load images
        self.img_nametag = cv2.imread("name_tag.png", cv2.IMREAD_GRAYSCALE)
        self.img_map     = cv2.imread(f"maps/{args.map}/map.png", cv2.IMREAD_GRAYSCALE)
        self.img_routes = [
            cv2.cvtColor(cv2.imread(f"maps/{args.map}/route1.png"), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.imread(f"maps/{args.map}/route2.png"), cv2.COLOR_BGR2RGB)
        ]
        self.img_route_rest = cv2.cvtColor(
                                cv2.imread(f"maps/{args.map}/route_rest.png"), cv2.COLOR_BGR2RGB)
        self.img_rune_warning = cv2.imread("rune/rune_warning.png", cv2.IMREAD_GRAYSCALE)
        self.img_rune = cv2.imread("rune/rune.png")
        self.img_arrows = {
            "left":
                [cv2.imread("rune/arrow_left_1.png"),
                cv2.imread("rune/arrow_left_2.png"),
                cv2.imread("rune/arrow_left_3.png"),],
            "right":
                [cv2.imread("rune/arrow_right_1.png"),
                cv2.imread("rune/arrow_right_2.png"),
                cv2.imread("rune/arrow_right_3.png"),],
            "up":
                [cv2.imread("rune/arrow_up_1.png"),
                cv2.imread("rune/arrow_up_2.png"),
                cv2.imread("rune/arrow_up_3.png")],
            "down":
                [cv2.imread("rune/arrow_down_1.png"),
                cv2.imread("rune/arrow_down_2.png"),
                cv2.imread("rune/arrow_down_3.png"),],
        }

        # Convert nametag image to make mask work
        self.img_nametag[self.img_nametag == 255] = 254
        # self.img_rune[self.img_rune == 255] = 254

        # Load monsters images
        self.monsters = {}
        for monster_name in args.monsters.split(","):
            imgs = []
            for file in glob.glob(f"monster/{monster_name}*.png"):
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                imgs.append(img) # Add original image
                imgs.append(cv2.flip(img, 1))  # Add flipped image

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


    def get_nearest_monster(self, is_left, overlap_threshold=0.5):
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
        while True:
            for arrow_idx in [0,1,2,3]:
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
                        _, score, _ = find_pattern_sqdiff(img_roi, img_arrow)
                        if score < best_score:
                            best_score = score
                            best_direction = direction
                logger.info(f"[solve_rune] Arrow({arrow_idx}) is {best_direction} with score({best_score})")

                # Press the key for 0.5 second
                self.kb.press_key(best_direction, 0.5)
                time.sleep(1)

            # Check if solved
            # Get lastest game screen frame buffer
            self.frame = self.capture.get_frame()
            # Resize game screen to 1296x759
            self.img_frame = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)

            if not self.is_in_rune_game():
                logger.info(f"[solve_rune] Solved all arrows")
                return

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
            (0, 0, 255), "Color Code Search Range"
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

        # Compute original bar dimensions
        hp_h, hp_w = hp_bar.shape[:2]
        mp_h, mp_w = mp_bar.shape[:2]

        # Paste HP bar directly at top-left
        self.img_frame_debug[45:45+hp_h, 180:180+hp_w] = hp_bar

        # Paste MP bar below HP bar
        self.img_frame_debug[75:75+mp_h, 180:180+mp_w] = mp_bar

        # Overlay HP/MP text
        cv2.putText(self.img_frame_debug, f"HP: {hp_ratio*100:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(self.img_frame_debug, f"MP: {mp_ratio*100:.1f}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    
        return hp_ratio, mp_ratio

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
            False

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
                            self.img_rune)

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
                screenshot(self.img_frame_debug)

                return True
            else:
                return False

    def is_in_rune_game(self):
        '''
        is_in_rune_game
        '''
        # Crop arrow detection box
        x, y = self.cfg.arrow_box_start_point
        size = self.cfg.arrow_box_size
        img_roi = self.img_frame[y:y+size, x:x+size]
        # draw_rectangle(
        #     img_debug, (x, y), (size, size),
        #     (255, 0, 0), "arrow detection box"
        # )
        # Check if arrow exist on screen
        for direc, arrow_list in self.img_arrows.items():
            for img_arrow in arrow_list:
                _, score, _ = find_pattern_sqdiff(img_roi, img_arrow)
                if score < self.cfg.arrow_box_dif_thres:
                    logger.info(f"Arrow screen detected with score({score})")
                    return True
        return False

    def get_monsters_in_range(self):
        '''
        get_monsters_in_range
        '''
        # Search monster nearby magic claw range
        dx = self.cfg.magic_claw_range_x + self.cfg.monster_search_margin
        dy = self.cfg.magic_claw_range_y + self.cfg.monster_search_margin
        x0 = max(0, self.loc_player[0] - dx)
        x1 = min(self.img_frame.shape[1], self.loc_player[0] + dx)
        y0 = max(0, self.loc_player[1] - dy)
        y1 = min(self.img_frame.shape[0], self.loc_player[1] + dy)

        img_roi = self.img_frame[y0:y1, x0:x1]

        monster_info = []
        for monster_name, monster_imgs in self.monsters.items():
            for img_monster in monster_imgs:
                #  Use black lines to detect monsters
                # Create masks (already grayscale)
                mask_pattern = np.all(img_monster == [0, 0, 0], axis=2).astype(np.uint8) * 255
                mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255

                # Apply Gaussian blur (soften the masks)
                img_monster_blur = cv2.GaussianBlur(mask_pattern, (self.cfg.blur_range, self.cfg.blur_range), 0)
                img_roi_blur = cv2.GaussianBlur(mask_roi, (self.cfg.blur_range, self.cfg.blur_range), 0)

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

                # img_monster_gray = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
                # _, mask_pattern = cv2.threshold(img_monster_gray, 254, 255, cv2.THRESH_BINARY_INV)
                # res = cv2.matchTemplate(
                #         img_roi,
                #         img_monster,
                #         cv2.TM_SQDIFF_NORMED,
                #         mask=mask_pattern)
                # match_locations = np.where(res <= self.cfg.monster_diff_thres)
                # h, w = img_monster.shape[:2]
                # for pt in zip(*match_locations[::-1]):
                #     monster_info.append({
                #         "name": monster_name,
                #         "position": (pt[0] + x0, pt[1] + y0),
                #         "size": (h, w),
                #         "score": res[pt[1], pt[0]],
                #     })

            # # TODO:
            # # Normalize and convert to BGR
            # res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
            # res_norm = np.uint8(res_norm)
            # res_bgr = cv2.cvtColor(res_norm, cv2.COLOR_GRAY2BGR)
            # # Get image sizes
            # h_debug, w_debug = self.img_frame.shape[:2]
            # h_res, w_res = res_bgr.shape[:2]
            # # Calculate padding needed
            # pad_left = (w_debug - w_res) // 2
            # pad_right = w_debug - w_res - pad_left
            # # Apply horizontal padding to match img_debug width
            # res_monster = cv2.copyMakeBorder(
            #     res_bgr,
            #     top=0, bottom=0,
            #     left=pad_left, right=pad_right,
            #     borderType=cv2.BORDER_CONSTANT,
            #     value=(0, 0, 0)  # black padding
            # )

        # Apply Non-Maximum Suppression to monster detection
        monster_info = nms(monster_info, iou_threshold=0.4)

        # Debug
        # Draw attack detection range
        draw_rectangle(
            self.img_frame_debug, (x0, y0), (dy*2, dx*2),
            (255, 0, 0), "Detection Range"
        )

        # Draw monsters bounding box
        for monster in monster_info:
            draw_rectangle(
                self.img_frame_debug, monster["position"], monster["size"],
                (0, 255, 0), f"{round(monster["score"], 2)}"
            )

        return monster_info

    def get_player_location(self):
        '''
        get player location by detecting player's nametag
        '''
        # Find player location by searching player's name tag
        img_roi = self.img_frame_gray[self.cfg.camera_ceiling:self.cfg.camera_floor, :]
        last_result = (
            self.loc_nametag[0],
            self.loc_nametag[1] - self.cfg.camera_ceiling)
        loc_nametag, score, is_cached = find_pattern_sqdiff(
                                            img_roi,
                                            self.img_nametag,
                                            last_result=last_result,
                                            global_threshold = 0.5)
        if score < self.cfg.nametag_diff_thres:
            self.loc_nametag = (
                loc_nametag[0],
                loc_nametag[1] + self.cfg.camera_ceiling)
        loc_player = (
            self.loc_nametag[0] - self.cfg.nametag_offset[0],
            self.loc_nametag[1] - self.cfg.nametag_offset[1])

        # Draw name tag detection box for debug
        draw_rectangle(
            self.img_frame_debug, self.loc_nametag, self.img_nametag.shape,
            (0, 255, 0),
            f"NameTag ({round(score, 2)}, {'cached' if is_cached else 'missed'})"
        )
        # Draw player center
        cv2.circle(self.img_frame_debug,
                loc_player, radius=3,
                color=(0, 0, 255), thickness=-1)
        # Draw camera ceiling line
        cv2.putText(
            self.img_frame_debug, "Camera Ceiling",
            (10, self.cfg.camera_ceiling - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
        cv2.line(
            self.img_frame_debug,
            (0, self.cfg.camera_ceiling),
            (self.img_frame_debug.shape[1], self.cfg.camera_ceiling),
            (255, 0, 0), 1
        )

        # Draw camera floor line
        cv2.putText(
            self.img_frame_debug, "Camera Floor",
            (10, self.cfg.camera_floor - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1
        )
        cv2.line(
            self.img_frame_debug,
            (0, self.cfg.camera_floor),
            (self.img_frame_debug.shape[1], self.cfg.camera_floor),
            (255, 0, 0), 1
        )

        return loc_player

    def get_player_location_global(self):
        '''
        get_player_location_global
        '''
        scale_factor = self.cfg.localize_downscale_factor
        # Downscale both template and search image
        img_map = cv2.resize(self.img_map, (0, 0), fx=scale_factor, fy=scale_factor)
        img_roi = self.img_frame_gray[self.cfg.camera_ceiling:self.cfg.camera_floor, :]
        img_query = cv2.resize(img_roi, (0, 0), fx=scale_factor, fy=scale_factor)
        last_result = (
            int(self.loc_camera[0] * scale_factor),
            int(self.loc_camera[1] * scale_factor)
        )

        # TODO: make this resizable to improve performance
        # I want to performa matching on 1/4 or 1/2 ratio , i wish this ratio is configuration
        loc_camera, score, is_cached = find_pattern_sqdiff(
            img_map,
            img_query,
            last_result=last_result,
            local_search_radius=20,
            global_threshold = 0.8)
        if score < self.cfg.localize_diff_thres:
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
                    (loc_player_global[0], loc_player_global[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return loc_player_global

    def run_once(self):
        '''
        Deal with one frame
        '''
        # Get lastest game screen frame buffer
        self.frame = self.capture.get_frame()

        # Get current route image
        self.img_route = self.img_routes[self.idx_routes]

        # Resize game screen to 1296x759
        self.img_frame = cv2.resize(self.frame, (1296, 759), interpolation=cv2.INTER_NEAREST)

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # image for debug use
        self.img_frame_debug = self.img_frame.copy()
        self.img_route_debug = cv2.cvtColor(self.img_route, cv2.COLOR_RGB2BGR)

        # Detect HP and MP ratio by detecting bars
        self.hp_ratio, self.mp_ratio = self.get_hp_mp()

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
            self.kb.press_key("up", 0.02) # Attemp to trigger rune

        # Check whether we entered the rune mini-game
        if self.status in ["finding_rune", "near_rune"] and self.is_in_rune_game():
            self.kb.set_command("stop")
            time.sleep(0.5) # Wait for character to stop
            self.kb.disable() # Disable kb thread during rune solving
            self.switch_status("solving_rune")
            screenshot(self.img_frame) # for debugging

        # Get all monster near player
        self.monster_info = self.get_monsters_in_range()

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
            command = color_code["action"]

        # Special logic for each status, overwrite color code
        if self.status == "hunting":
            if self.hp_ratio <= self.cfg.heal_ratio:
                command = "heal"
                self.t_watch_dog = time.time()
            elif self.mp_ratio <= self.cfg.add_mp_ratio:
                command = "add mp"
                self.t_watch_dog = time.time()
            elif attack_direction == "left":
                command = "attack left"
                self.t_watch_dog = time.time()
            elif attack_direction == "right":
                command = "attack right"
                self.t_watch_dog = time.time()
            elif self.is_player_stuck():
                # Perform a random action
                command = random.choice(list(self.cfg.color_code.values()))
        elif self.status == "finding_rune":
            # Check if finding rune timeout
            if time.time() - self.t_last_switch_status > self.cfg.rune_finding_timeout:
                self.switch_status("resting")
        elif self.status == "near_rune":
            self.kb.press_key("up", 0.02) # Attempt to trigger rune
            # Stay in near_rune status for only a few seconds
            if time.time() - self.t_last_switch_status > self.cfg.near_rune_duration:
                self.switch_status("hunting")
        elif self.status == "solving_rune":
            self.solve_rune() # Blocking function
            self.switch_status("hunting")
            self.kb.enable()
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
        # Draw FPS text on top left corner
        fps = round(1.0 / (time.time() - self.t_last_frame))
        self.t_last_frame = time.time()
        cv2.putText(
            self.img_frame_debug, f"FPS: {fps}, Press 's' to save screenshot",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 
            2, cv2.LINE_AA
        )

        # Vertically concatenate
        # self.img_frame_debug = cv2.vconcat([self.img_frame_debug, res_monster])

        # Resize img_route_debug for better visualization
        h, w = self.img_route_debug.shape[:2]
        self.img_route_debug = cv2.resize(self.img_route_debug, (w // 2, h // 2),
                                interpolation=cv2.INTER_NEAREST)

        # Show debug image
        cv2.imshow("Game Window Debug", self.img_frame_debug)
        cv2.imshow("Route Map Debug", self.img_route_debug)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--disable_control',
        action='store_true',
        help='Disable fake keyboard input'
    )

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
            if key == ord('s'):
                screenshot(mapleStoryBot.img_frame)
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()
