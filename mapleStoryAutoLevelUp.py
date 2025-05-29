import time
import sys
import cv2
import numpy as np
import pydirectinput
from collections import Counter
import threading
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
import keyboard
import pygetwindow as gw
import random

# local import
from config import Config

# Do not modify the following parameters
# color code for patrol route
COLOR_CODE = {
    # R   G   B
    (255, 0, 0): "walk left", # red
    (0, 0, 255): "walk right", # blue
    (255,127,0): "jump left", # orange
    (0,255,255): "jump right", # sky blue
    (255,0,255): "jump", # purple
    (127,127,127): "up", # gray
    (255,255,0): "goal", # yellow
}
GAME_WINDOW_TITLE = 'MapleStory Worlds-Artale (繁體中文版)'

# Global Variables
IS_RUN = True
FAKE_KEYBOARD_COMMAND = ""
WIN_LIST = []
FRAME_BUFFER = None
CFG = Config
LOCK_FRAME_BUFFER = threading.Lock()

# Create the capture session
capture = WindowsCapture(window_name=GAME_WINDOW_TITLE)

@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    '''
    on_frame_arrived
    '''
    global FRAME_BUFFER
    with LOCK_FRAME_BUFFER:
        FRAME_BUFFER = frame.frame_buffer
    time.sleep(0.033) # Cap FPS to 30

# Capture closed callback
@capture.event
def on_closed():
    '''
    on_closed
    '''
    print("Capture session closed.")
    cv2.destroyAllWindows()

def is_game_window_active():
    active_window = gw.getActiveWindow()
    return active_window is not None and GAME_WINDOW_TITLE in active_window.title

def find_nearest_color_code(player_loc, img_route):

    x0, y0 = player_loc
    h, w = img_route.shape[:2]  # Get image height and width

    x_min = max(0, x0 - CFG.COLOR_CODE_SEARCH_RANGE)
    x_max = min(w, x0 + CFG.COLOR_CODE_SEARCH_RANGE)
    y_min = max(0, y0 - CFG.COLOR_CODE_SEARCH_RANGE)
    y_max = min(h, y0 + CFG.COLOR_CODE_SEARCH_RANGE)

    nearest = None
    min_dist = float('inf')
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            pixel = tuple(img_route[y, x])  # (R, G, B)
            if pixel in COLOR_CODE:
                dist = abs(x - x0) + abs(y - y0)
                if dist < min_dist:
                    min_dist = dist
                    nearest = {
                        "pixel": (x, y),
                        "color": pixel,
                        "action": COLOR_CODE[pixel],
                        "distance": dist
                    }

    return nearest  # if not found return none

def nms(monsters, iou_threshold=0.3):
    '''
    Apply Non-Maximum Suppression (NMS) to remove overlapping detections.

    Parameters:
    - monsters: List of dictionaries, each representing a detected monster with:
        - "position": (x, y) top-left corner
        - "size": (width, height)
        - "score": similarity/confidence score from template matching
    - iou_threshold: Float, intersection-over-union threshold to suppress overlapping boxes

    Returns:
    - List of filtered monster dictionaries after applying NMS
    '''
    boxes = []
    for m in monsters:
        x, y = m["position"]
        w, h = m["size"]
        # [x1, y1, x2, y2, score, original_data]
        boxes.append([x, y, x + w, y + h, m["score"], m])

    # Sort by score descending
    boxes.sort(key=lambda x: x[4], reverse=True)

    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best[5])  # original monster_info

        boxes = [b for b in boxes if get_iou(best, b) < iou_threshold]

    return keep

def get_iou(box1, box2):
    '''
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Each box is expected to be a tuple or list with at least 4 values:
    (x1, y1, x2, y2), where:
        - (x1, y1) is the top-left corner
        - (x2, y2) is the bottom-right corner

    Returns:
        A float representing the IoU value (0.0 ~ 1.0).
        If there is no overlap, returns 0.0.
    '''
    x1, y1, x2, y2 = box1[:4]
    x1_p, y1_p, x2_p, y2_p = box2[:4]

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)
    union = area1 + area2 - inter_area

    return inter_area / union

def is_monster_in_range(
        monster_info,
        attack_box_top_left,
        overlap_threshold=0.5):
    '''
    Check if any monster's box overlaps with the player's attack box
    by at least a certain threshold (default is 50%).

    Parameters:
    - monster_info: list of dicts, each containing monster's "position" and "size"
    - attack_box_top_left: tuple (x, y), top-left corner of the attack range box
    - overlap_threshold: float, minimum overlap ratio (intersection / monster area)
                         to consider a attackable monster

    Returns:
    - True if any monster has sufficient overlap with attack box, False otherwise
    '''
    ax1, ay1 = attack_box_top_left
    ax2, ay2 = (attack_box_top_left[0] + CFG.MAGIC_CLAW_RANGE_X,
                attack_box_top_left[1] + CFG.MAGIC_CLAW_RANGE_Y)

    for monster in monster_info:
        mx1, my1 = monster["position"]
        mw, mh = monster["size"]
        mx2 = mx1 + mw
        my2 = my1 + mh

        # Calculate intersection
        ix1 = max(ax1, mx1)
        iy1 = max(ay1, my1)
        ix2 = min(ax2, mx2)
        iy2 = min(ay2, my2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter_area = iw * ih

        monster_area = mw * mh
        if monster_area == 0:
            continue  # skip degenerate box

        overlap_ratio = inter_area / monster_area

        if overlap_ratio >= overlap_threshold:
            return True  # found monster within range

    return False

def find_pattern(
        img,
        img_pattern,
        last_result=None,
        local_search_radius=50,
        global_threshold=0.6):
    '''
    Locate the most possible pattern in a larger image
    It first attempts a fast local search around the previous result;
    if it fails, it falls back to a global search.

    Parameters:
    - img: The target image (numpy array) as search space.
    - img_pattern: The smaller pattern/template image to search for.
    - last_result: Tuple (x, y) of the previous pattern location.
                   If provided, enables local search.
    - local_search_radius: window size (in pixels) to perform local search.
    - global_threshold: Match threshold to accept local match;
                        otherwise fallback to full image search.

    Returns:
    - top_left: (x, y) top-left corner coordinate of the best matching.
    '''
    h, w = img_pattern.shape[:2]

    if last_result is not None:
        lx, ly = last_result
        x1 = max(0, lx - local_search_radius)
        y1 = max(0, ly - local_search_radius)
        x2 = min(img.shape[1], lx + local_search_radius + w)
        y2 = min(img.shape[0], ly + local_search_radius + h)

        # Local search
        img_local = img[y1:y2, x1:x2]
        res_local = cv2.matchTemplate(img_local, img_pattern, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_local)

        if max_val >= global_threshold:
            top_left = (x1 + max_loc[0], y1 + max_loc[1])
            return top_left, max_val

    # Global fallback
    res = cv2.matchTemplate(img, img_pattern, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, top_left = cv2.minMaxLoc(res)

    return top_left, max_val

loc_last = (0, 0)
t_last_move = time.time()
def is_stuck(loc):
    """
    Detect if the player is stuck (not moving).
    If stuck for more than WATCH_DOG_TIMEOUT seconds, performs a random action.
    """
    global loc_last, t_last_move

    current_time = time.time()

    dx = abs(loc[0] - loc_last[0])
    dy = abs(loc[1] - loc_last[1])

    if dx + dy > CFG.WATCH_DOG_RANGE:
        # Player moved, reset
        loc_last = loc
        t_last_move = current_time
        return False

    if current_time - t_last_move > CFG.WATCH_DOG_TIMEOUT:
        # Stuck too long
        loc_last = loc
        t_last_move = current_time
        return True

    return False

def press_key(key, duration):
    '''
    Simulates a key press for a specified duration using pydirectinput.
    '''
    pydirectinput.keyDown(key)
    time.sleep(duration)
    pydirectinput.keyUp(key)

def fake_keyboard_input_worker():
    '''
    Continuously listens to the global FAKE_KEYBOARD_COMMAND
    and simulates key inputs using pydirectinput.
    '''
    # store last 'up' command time
    t_last_up = 0.0

    while True:
        # Check if game window is active
        if not is_game_window_active():
            time.sleep(0.001)
            continue

        # check if is needed to release 'Up' key
        if time.time() - t_last_up > CFG.UP_DRAG_DURATION:
            keyboard.release("up")

        if FAKE_KEYBOARD_COMMAND == "walk left":
            keyboard.release("right")
            keyboard.press("left")

        elif FAKE_KEYBOARD_COMMAND == "walk right":
            keyboard.release("left")
            keyboard.press("right")

        elif FAKE_KEYBOARD_COMMAND == "jump left":
            keyboard.release("right")
            keyboard.press("left")
            press_key(CFG.JUMP_KEY, 0.02)
            keyboard.release("left")

        elif FAKE_KEYBOARD_COMMAND == "jump right":
            keyboard.release("left")
            keyboard.press("right")
            press_key(CFG.JUMP_KEY, 0.02)
            keyboard.release("right")

        elif FAKE_KEYBOARD_COMMAND == "jump":
            keyboard.release("left")
            keyboard.release("right")
            press_key(CFG.JUMP_KEY, 0.02)

        elif FAKE_KEYBOARD_COMMAND == "up":
            keyboard.press("up")
            t_last_up = time.time()

        elif FAKE_KEYBOARD_COMMAND == "attack left":
            keyboard.release("right")
            keyboard.press("left")
            press_key(CFG.ATTACK_KEY, 0.02)
            keyboard.release("left")

        elif FAKE_KEYBOARD_COMMAND == "attack right":
            keyboard.release("left")
            keyboard.press("right")
            press_key(CFG.ATTACK_KEY, 0.02)
            keyboard.release("right")

        else:
            # Release all keys, stop the character
            keyboard.release("left")
            keyboard.release("right")

        time.sleep(0.001)

def draw_rectangle(img, top_left, size, color, text):
    '''
    Draws a rectangle with an text label.

    Parameters:
    - img: The image on which to draw (numpy array).
    - top_left: Tuple (x, y), the top-left corner of the rectangle.
    - size: Tuple (height, width) of the rectangle.
    - color: Tuple (B, G, R), color of the rectangle and text.
    - text: String to display above the rectangle.
    '''
    bottom_right = (top_left[0] + size[1],
                    top_left[1] + size[0])
    cv2.rectangle(img, top_left, bottom_right, color, 2)
    cv2.putText(img, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    '''
    Main function
    '''
    global SCREENS_CAP, FAKE_KEYBOARD_COMMAND

    # Load images
    img_nametag = cv2.imread("nameTag.png", cv2.IMREAD_GRAYSCALE)
    img_map     = cv2.imread("maps/north_forest_training_ground_2/map.png", cv2.IMREAD_GRAYSCALE)
    img_routes = [
        cv2.cvtColor(cv2.imread("maps/north_forest_training_ground_2/route1.png"), cv2.COLOR_BGR2RGB),
        cv2.cvtColor(cv2.imread("maps/north_forest_training_ground_2/route2.png"), cv2.COLOR_BGR2RGB)
    ]
    monster_dict = {
        'Green Mushroom':
            cv2.imread("monster/green_mushroom.png", cv2.IMREAD_GRAYSCALE),
        'Spike Mushroom':
            cv2.imread("monster/spike_mushroom.png", cv2.IMREAD_GRAYSCALE),
    }

    # Start fake keyboard input thread
    threading.Thread(target=fake_keyboard_input_worker, daemon=True).start()

    # Start screen capture thread
    threading.Thread(target=capture.start, daemon=True).start()

    route_idx = 0
    t_last_frame = time.time()
    last_nametag_top_left = (0, 0)
    last_camera = None

    while IS_RUN:
        if FRAME_BUFFER is None:
            print(f"Cannot find window:'{GAME_WINDOW_TITLE}'.")
            print("Please start up maple story!")
            continue

        # Get lastest frame buffer
        with LOCK_FRAME_BUFFER:
            img_window = FRAME_BUFFER.copy() # (R, B, G)

        # Resize game screen to 1296x759
        img_window = cv2.resize(img_window, (1296, 759), interpolation=cv2.INTER_NEAREST)
        # print(f"img_window.shape = {img_window.shape}") # (759, 1296, 3)

        img_window_gray = cv2.cvtColor(img_window, cv2.COLOR_RGB2GRAY)

        ########################
        ### Player Detection ###
        ########################
        NAMETAG_SIM_THRES = 0.5
        # Find player location by searching player's name tag
        name_tag_top_left, score = find_pattern(
                                    img_window_gray,
                                    img_nametag,
                                    last_result=last_nametag_top_left
                                )
        if score > NAMETAG_SIM_THRES:
            # Update last detection
            last_nametag_top_left = name_tag_top_left
        else:
            # Cannot find name tag, than use last detection result
            name_tag_top_left = last_nametag_top_left
        loc_player = (name_tag_top_left[0] - CFG.NAMETAG_OFFSET_X,
                      name_tag_top_left[1] - CFG.NAMETAG_OFFSET_Y)

        ###########################
        ### Player Localization ###
        ###########################
        # Localizae camera position and localize player on map
        camera_top_left, score = find_pattern(
                                    img_map,
                                    img_window_gray[CFG.CAMERA_CEILING:CFG.CAMERA_FLOOR, :],
                                    last_result=last_camera
        )
        last_camera = camera_top_left
        # TODO: make this resizable to improve performance
        map_loc_player = (camera_top_left[0] + loc_player[0],
                          camera_top_left[1] + loc_player[1] - CFG.CAMERA_CEILING,)

        #########################
        ### Monster Detection ###
        #########################
        # Search monster nearby magic claw range
        x_min = max(0, loc_player[0] - CFG.MAGIC_CLAW_RANGE_X - CFG.MONSTER_SEARCH_MARGIN)
        y_min = max(0, loc_player[1] - CFG.MAGIC_CLAW_RANGE_Y - CFG.MONSTER_SEARCH_MARGIN)
        x_max = min(img_window.shape[1], loc_player[0] + CFG.MAGIC_CLAW_RANGE_X + CFG.MONSTER_SEARCH_MARGIN)
        y_max = min(img_window.shape[0], loc_player[1] + CFG.MAGIC_CLAW_RANGE_Y + CFG.MONSTER_SEARCH_MARGIN)

        # Crop the region of interest (ROI)
        img_roi = img_window_gray[y_min:y_max, x_min:x_max]

        monster_info = []
        for monster_name, monster_img in monster_dict.items():
            for flipped in [False, True]:
                img = cv2.flip(monster_img, 1) if flipped else monster_img
                res = cv2.matchTemplate(img_roi, img, cv2.TM_CCOEFF_NORMED)
                match_locations = np.where(res >= CFG.MONSTER_SIM_THRES)
                h, w = img.shape[:2]

                for pt in zip(*match_locations[::-1]):
                    monster_info.append({
                        "name": monster_name,
                        "position": (pt[0] + x_min, pt[1] + y_min),
                        "size": (w, h),
                        "flipped": flipped,
                        "score": res[pt[1], pt[0]],
                    })

        # Apply Non-Maximum Suppression to monster detection
        monster_info = nms(monster_info, iou_threshold=0.4)
        # monster_count = Counter([m["name"] for m in monster_info])
        # print(monster_count)  # e.g., {'GreenMushroom': 3, 'Slime': 2}

        ###############################
        ### Magic Claw Attack Range ###
        ###############################
        # Get magic claw left box
        attack_left_top_left = (loc_player[0] - CFG.MAGIC_CLAW_RANGE_X,
                                loc_player[1] - int(CFG.MAGIC_CLAW_RANGE_Y/2))
        # Get magic claw right box
        attack_right_top_left = (loc_player[0] ,
                                 loc_player[1] - int(CFG.MAGIC_CLAW_RANGE_Y/2))

        #############################
        ### Fake Keyboard Command ###
        #############################
        if is_monster_in_range(monster_info, attack_left_top_left):
            FAKE_KEYBOARD_COMMAND = "attack left"
        elif is_monster_in_range(monster_info, attack_right_top_left):
            FAKE_KEYBOARD_COMMAND = "attack right"
        elif is_stuck(map_loc_player):
            FAKE_KEYBOARD_COMMAND = random.choice(list(COLOR_CODE.values()))
        else:
            color_code = find_nearest_color_code(
                            map_loc_player,
                            img_routes[route_idx]
            )
            if color_code:
                FAKE_KEYBOARD_COMMAND = color_code["action"]
                # Check if reach goal
                if color_code["action"] == "goal":
                    route_idx = (route_idx+1)%len(img_routes)
            else:
                FAKE_KEYBOARD_COMMAND = ""

        ####################
        ### Window Debug ###
        ####################
        if not CFG.ENABLE_DEBUG_WINDOWS:
            continue

        # Copy window image for debugging use
        # img_debug = cv2.cvtColor(img_window.copy(), cv2.COLOR_BGR2RGB)
        img_debug = img_window.copy()

        # Draw name tag detection box
        draw_rectangle(
            img_debug, name_tag_top_left, img_nametag.shape,
            (0, 255, 0), "Name Tag"
        )

        # Draw player center
        cv2.circle(img_debug, loc_player, radius=3, color=(0, 0, 255), thickness=-1)

        # Draw monsters bounding box
        for monster in monster_info:
            draw_rectangle(
                img_debug, monster["position"], monster["size"],
                (0, 255, 0), monster["name"]
            )

        # Draw magic claw right bounding box
        draw_rectangle(
            img_debug, attack_right_top_left,
            (CFG.MAGIC_CLAW_RANGE_Y, CFG.MAGIC_CLAW_RANGE_X),
            (0, 0, 255), "Attack Right Range"
        )

        # Draw magic claw left bounding box
        draw_rectangle(
            img_debug, attack_left_top_left,
            (CFG.MAGIC_CLAW_RANGE_Y, CFG.MAGIC_CLAW_RANGE_X),
            (0, 0, 255), "Attack Left Range"
        )

        # Draw FPS
        fps = round(1.0 / (time.time() - t_last_frame))
        t_last_frame = time.time()
        cv2.putText(
            img_debug, f"FPS: {fps}, Press 's' to save screenshot",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 
            2, cv2.LINE_AA
        )

        # Show debug image
        cv2.imshow("Screen Debug", img_debug)

        #################
        ### Map Debug ###
        #################
        img_map_debug = cv2.cvtColor(img_routes[route_idx].copy(), cv2.COLOR_BGR2RGB)

        # Draw camera
        cv2.circle(img_map_debug, camera_top_left, radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(img_map_debug, "Camera", (camera_top_left[0], camera_top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw player center
        cv2.circle(img_map_debug, map_loc_player, radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(img_map_debug, "Player", (map_loc_player[0], map_loc_player[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw search range
        draw_rectangle(
            img_map_debug,
            (map_loc_player[0] - CFG.COLOR_CODE_SEARCH_RANGE,
             map_loc_player[1] - CFG.COLOR_CODE_SEARCH_RANGE),
            (CFG.COLOR_CODE_SEARCH_RANGE*2, CFG.COLOR_CODE_SEARCH_RANGE*2),
            (0, 0, 255), "Color Code Search Range"
        )

        # Draw a straigt line from map_loc_player to color_code["pixel"]
        if color_code is not None:
            cv2.line(
                img_map_debug,
                map_loc_player,      # start point
                color_code["pixel"], # end point
                (0, 255, 0),         # green line
                1                    # thickness
            )

        # Show debug image, donw size
        h, w = img_map_debug.shape[:2]
        img_map_debug = cv2.resize(img_map_debug, (w // 2, h // 2),
                                   interpolation=cv2.INTER_NEAREST)
        # img_map_debug = cv2.resize(img_map_debug)
        cv2.imshow("Map Debug", img_map_debug)

        # Exit if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF  # Get the pressed key (8-bit)
        if key == ord('s'):
            filename = f"screenshot/screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, img_window)
            print(f"Screenshot saved: {filename}")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
