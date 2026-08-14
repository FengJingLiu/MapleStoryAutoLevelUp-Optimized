'''
Utility functions
'''
# Standard Import
import cv2
import datetime
import os
import platform
import smtplib
from email.message import EmailMessage
import imaplib
import mimetypes
import email
from collections import defaultdict
import time

# Libarary Import
import numpy as np
import yaml
import pyautogui
import pygetwindow as gw
from ruamel.yaml import YAML

# macOS Import
if platform.system() == 'Darwin':
    import Quartz
else:
    import win32gui
    import win32con

# Local import
from src.utils.logger import logger
from src.utils.global_var import WINDOW_WORKING_SIZE
from src.utils.detection import get_iou, nms

OS_NAME = platform.system()

def is_mac():
    return OS_NAME == 'Darwin'

def is_windows():
    return OS_NAME == 'Windows'

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        logger.info(f"Load yaml: {path}")
        data = yaml.safe_load(f) or {}
        return convert_lists_to_tuples(data)

def load_yaml_with_comments(path):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)

    field_comments = defaultdict(dict)
    section_comments = {}

    for title, sub in data.items():
        # Extract section comment (before key)
        if sub.ca.comment and sub.ca.comment[1]:
            section_comment_lines = [line.value.strip('#').strip() for line in sub.ca.comment[1]]
            section_comments[title] = "\n".join(section_comment_lines)

        # Extract field-level comments
        if hasattr(sub, 'ca'):
            for key in sub:
                comment = sub.ca.items.get(key)
                if comment and comment[2]:
                    field_comments[title][key] = comment[2].value.strip('#').strip()

    return data, dict(field_comments), section_comments

def save_yaml(data, path):
    data = convert_tuples_to_lists(data)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)
    logger.info(f"Save yaml: {path}")

def get_cfg_diff(base, current):
    """
    Recursively compute the diff between base and current configs.
    Return only the values from current that are different.
    """
    diff = {}
    for key in current:
        if key not in base:
            diff[key] = current[key]
        elif isinstance(current[key], dict) and isinstance(base.get(key), dict):
            sub_diff = get_cfg_diff(base[key], current[key])  # recursive call
            if sub_diff:
                diff[key] = sub_diff
        else:
            norm_current = normalize(current[key])
            norm_base = normalize(base.get(key))
            if norm_current != norm_base:
                diff[key] = current[key]
    return diff

def normalize(value):
    """
    Normalize value for comparison:
    - Convert tuples to lists
    - Recursively normalize lists and dicts
    """
    if isinstance(value, tuple):
        return [normalize(v) for v in value]
    elif isinstance(value, list):
        return [normalize(v) for v in value]
    elif isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    else:
        return value

def convert_tuples_to_lists(obj):
    if isinstance(obj, dict):
        return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        return [convert_tuples_to_lists(i) for i in obj]
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(i) for i in obj]
    else:
        return obj

def override_cfg(base, override):
    '''
    override_cfg (in-place)
    Modifies `base` directly by overriding keys from `override`.
    '''
    for k, v in override.items():
        if (
            k in base and isinstance(base[k], dict)
            and isinstance(v, dict)
        ):
            override_cfg(base[k], v)  # recursive override
        else:
            base[k] = v  # direct override or new key
    return base

def convert_lists_to_tuples(obj):
    if isinstance(obj, list):
        return tuple(convert_lists_to_tuples(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: convert_lists_to_tuples(v) for k, v in obj.items()}
    else:
        return obj

def load_image(path, mode=cv2.IMREAD_COLOR):
    '''
    Load image from disk and verify existence.
    '''
    if not os.path.exists(path):
        logger.error(f"Image not found: {path}")
        raise FileNotFoundError(f"Image not found: {path}")

    # Load image
    img = cv2.imread(path, mode)
    if img is None:
        logger.error(f"Failed to load image file: {path}")
        raise ValueError(f"Failed to load image: {path}")

    logger.info(f"Loaded image: {path}")

    return img

def screenshot(img, suffix="screenshot"):
    '''
    Save the given image as a screenshot file.

    Parameters:
    - img: numpy array (image to save).

    Behavior:
    - Saves the image to the "screenshot/" directory with the current timestamp as filename.
    '''

    if img is None:
        return

    # ensure directory exists
    os.makedirs("screenshot", exist_ok=True)

    # Generate timestamp string
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"screenshot/{timestamp}_{suffix}.png"
    cv2.imwrite(filename, img)
    logger.info(f"[screenshot] save to {filename}")

def draw_rectangle(img, top_left, size, color, text,
                   thickness=2, text_height=0.7, *, visual_scale=1.0):
    '''
    Draws a rectangle with an text label.

    Parameters:
    - img: The image on which to draw (numpy array).
    - top_left: Tuple (x, y), the top-left corner of the rectangle.
    - size: Tuple (height, width) of the rectangle.
    - color: Tuple (B, G, R), color of the rectangle and text.
    - text: String to display above the rectangle.
    '''
    if img is None:
        return

    try:
        visual_scale = max(0.1, float(visual_scale))
    except (TypeError, ValueError):
        visual_scale = 1.0
    line_thickness = (
        -1
        if thickness < 0
        else max(1, int(round(thickness * visual_scale)))
    )
    label_scale = text_height * visual_scale
    label_offset = max(1, int(round(10 * visual_scale)))

    bottom_right = (top_left[0] + size[1],
                    top_left[1] + size[0])
    cv2.rectangle(img, top_left, bottom_right, color, line_thickness)
    text_thickness = max(1, int(round(abs(thickness) * visual_scale)))
    cv2.putText(img, text, (top_left[0], top_left[1] - label_offset),
                cv2.FONT_HERSHEY_SIMPLEX, label_scale, color,
                text_thickness)

def draw_circle(img, *args, **kwargs):
    """Draw a circle when a debug image is available."""
    if img is not None:
        cv2.circle(img, *args, **kwargs)

def draw_line(img, *args, **kwargs):
    """Draw a line when a debug image is available."""
    if img is not None:
        cv2.line(img, *args, **kwargs)

def draw_text(img, *args, **kwargs):
    """Draw text when a debug image is available."""
    if img is not None:
        cv2.putText(img, *args, **kwargs)

def pad_to_size(img, size, pad_value=0):
    '''
    pad_to_size
    '''
    h_img, w_img = img.shape[:2]
    h_target, w_target = size

    pad_h = max(0, h_target - h_img)
    pad_w = max(0, w_target - w_img)

    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(
            img,
            top   = pad_h // 2,
            bottom= pad_h - pad_h // 2,
            left  = pad_w // 2,
            right = pad_w - pad_w // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_value
        )

    return img

def find_pattern_sqdiff(
        img, img_pattern,
        last_result=None,
        mask=None,
        local_search_radius=50,
        global_threshold=0.4
    ):
    '''
    Perform masked template matching using SQDIFF_NORMED method.

    The function searches for the best matching location of img_pattern inside img.
    It automatically converts the pattern to grayscale and generates a mask to ignore
    pure white (or near-white) pixels in the template, treating them as transparent background.

    Parameters:
    - img: Target search image (numpy array), can be grayscale or BGR.
    - img_pattern: Template image to search for (numpy array, BGR).

    Returns:
    - min_loc: The top-left coordinate (x, y) of the best match position.
    - min_val: The matching score (lower = better for SQDIFF_NORMED).
    - bool: local search success or not
    '''
    # Padding if img is smaller than pattern
    img = pad_to_size(img, img_pattern.shape[:2])

    # search last result location first to speedup
    h, w = img_pattern.shape[:2]
    if last_result is not None and global_threshold > 0.0:
        lx, ly = last_result
        x0 = max(0, lx - local_search_radius)
        y0 = max(0, ly - local_search_radius)
        x1 = min(img.shape[1], lx + local_search_radius + w)
        y1 = min(img.shape[0], ly + local_search_radius + h)

        img_roi = img[y0:y1, x0:x1]
        if img_roi.shape[0] >= h and img_roi.shape[1] >= w:
            res = cv2.matchTemplate(
                    img_roi,
                    img_pattern,
                    cv2.TM_SQDIFF_NORMED,
                    mask=mask
            )
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if min_val < global_threshold:
                return (x0 + min_loc[0], y0 + min_loc[1]), min_val, True

    # Global fallback
    res = cv2.matchTemplate(
            img,
            img_pattern,
            cv2.TM_SQDIFF_NORMED,
            mask=mask
    )

    # Replace -inf/+inf/nan to 1.0 to avoid numerical error
    res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    return min_loc, min_val, False

def get_mask(img, ignore_pixel_color):
    '''
    get_mask
    '''
    mask = np.all(img == ignore_pixel_color, axis=2).astype(np.uint8) * 255
    mask = cv2.bitwise_not(mask)
    return mask

def to_opencv_hsv(color_hsv):
    """
    Convert HSV from standard scale:
    - Hue: 0–360
    - Saturation: 0–100
    - Value: 0–100
    to OpenCV HSV format:
    - Hue: 0–179
    - Saturation/Value: 0–255

    Args:
        color_hsv (tuple/list/np.ndarray): HSV in standard scale (H, S, V)

    Returns:
        np.ndarray: HSV in OpenCV scale
    """
    h, s, v = color_hsv
    h_opencv = round(h / 360 * 179)
    s_opencv = round(s / 100 * 255)
    v_opencv = round(v / 100 * 255)
    return np.array([h_opencv, s_opencv, v_opencv], dtype=np.uint8)

def to_standard_hsv(color_hsv):
    """
    Convert HSV from OpenCV scale to standard HSV scale.
    """
    h, s, v = color_hsv
    h_std = h / 179 * 360
    s_std = s / 255 * 100
    v_std = v / 255 * 100
    return (h_std, s_std, v_std)

def get_minimap_loc_size(img_frame):
    '''
    Detects the location and size of the minimap within the game frame.

    The function works by:
    - Thresholding near-white pixels so capture-card scaling and video-player
      interpolation do not erase the minimap border.
    - Looking for a rectangular border in the top-left part of the frame.
    - Requiring strong coverage on all four sides to reject text and icons.

    Returns:
        (x, y, w, h): Top-left coordinate and width/height of the minimap.
                    Returns None if not found.
    '''
    if img_frame is None or img_frame.ndim != 3:
        return None

    frame_h, frame_w = img_frame.shape[:2]
    # The game minimap is always near the top-left. Limiting the thresholding
    # pass matters for native 4K capture-card frames and avoids scanning the
    # whole game scene every frame.
    search_h = max(1, int(np.ceil(frame_h * 0.4)))
    search_w = max(1, int(np.ceil(frame_w * 0.4)))
    search_frame = img_frame[:search_h, :search_w]
    near_white_min = np.array([210, 210, 210], dtype=np.uint8)
    white_max = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(search_frame, near_white_min, white_max)
    contours, hierarchy = cv2.findContours(
        mask_white,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    def contour_depth(index):
        """Return nesting depth so an expanded panel's map wins its frame."""
        if hierarchy is None:
            return 0

        depth = 0
        parent = int(hierarchy[0][index][3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        return depth

    candidates = []
    for contour_index, contour in enumerate(contours):
        x0, y0, rw, rh = cv2.boundingRect(contour)

        # The compact minimap used by the bot is in the top-left quadrant.
        # HDMI capture can reduce a wide, shallow minimap below 50px after the
        # PotPlayer frame is normalized to 1296x700. Four-edge validation below
        # still rejects ordinary text and open UI lines.
        if rw < 80 or rh < 40:
            continue
        if x0 > frame_w * 0.4 or y0 > frame_h * 0.4:
            continue
        if rw > frame_w * 0.5 or rh > frame_h * 0.5:
            continue

        candidate = mask_white[y0:y0+rh, x0:x0+rw]
        max_border_depth = min(16, rw // 4, rh // 4)
        row_coverages = [
            np.mean(candidate[row, :] > 0)
            for row in range(max_border_depth)
        ]
        reverse_row_coverages = [
            np.mean(candidate[rh - 1 - row, :] > 0)
            for row in range(max_border_depth)
        ]
        column_coverages = [
            np.mean(candidate[:, col] > 0)
            for col in range(max_border_depth)
        ]
        reverse_column_coverages = [
            np.mean(candidate[:, rw - 1 - col] > 0)
            for col in range(max_border_depth)
        ]
        edge_coverages = tuple(
            max(coverages[:3])
            for coverages in (
                row_coverages,
                reverse_row_coverages,
                column_coverages,
                reverse_column_coverages,
            )
        )

        # A small amount of border damage is expected after HDMI scaling, but
        # ordinary white text will not form four mostly complete edges.
        if min(edge_coverages) < 0.82:
            continue

        def border_depth(coverages):
            # A contour can begin one antialiased pixel before the continuous
            # border. Start at the first strong row/column near the edge and
            # strip the complete run from there.
            start = next(
                (idx for idx, coverage in enumerate(coverages[:3])
                 if coverage >= 0.82),
                None,
            )
            if start is None:
                return 0
            depth = start
            for coverage in coverages[start:]:
                if coverage < 0.82:
                    break
                depth += 1
            return depth

        top_depth = border_depth(row_coverages)
        bottom_depth = border_depth(reverse_row_coverages)
        left_depth = border_depth(column_coverages)
        right_depth = border_depth(reverse_column_coverages)
        inner_w = rw - left_depth - right_depth
        inner_h = rh - top_depth - bottom_depth
        if inner_w <= 0 or inner_h <= 0:
            continue

        inner = candidate[
            top_depth:top_depth + inner_h,
            left_depth:left_depth + inner_w,
        ]
        # A complete white block has four perfect edges too, but it is not a
        # minimap. Require a meaningful non-white interior.
        if np.mean(inner == 0) < 0.15:
            continue

        # An expanded minimap has a complete outer panel border around another
        # complete border that encloses the actual map raster.  RETR_EXTERNAL
        # only exposed the panel and made player-dot detection inspect its
        # title/icons. Prefer a valid nested rectangle; edge quality and area
        # retain the previous ordering between candidates at the same depth.
        score = (
            contour_depth(contour_index),
            min(edge_coverages),
            np.mean(edge_coverages),
            rw * rh,
        )
        candidates.append((score, (
            x0 + left_depth,
            y0 + top_depth,
            inner_w,
            inner_h,
        )))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1]

    # logger.warning("Minimap not found in the game frame.")
    return None  # minimap not found

def copy_minimap_native_raster(img_minimap):
    """Return an independent minimap image without changing its dimensions.

    Route maps and action coordinates now use the captured minimap's native
    raster.  Keeping a copy prevents alignment cleanup from modifying the
    source crop (and therefore the captured game frame) in-place.
    """
    if img_minimap is None or img_minimap.size == 0:
        return img_minimap
    return img_minimap.copy()


def copy_minimap_native_location(location):
    """Keep a detected minimap point in the native raster coordinate system."""
    if location is None:
        return None
    return tuple(map(int, location[:2]))


def route_map_can_fit_minimap(img_map, img_minimap):
    """Return whether a route-map canvas can contain the native minimap.

    ``find_pattern_sqdiff`` pads an undersized target image for general-purpose
    matching.  That behavior would hide the most common legacy-route mismatch:
    an old, downscaled map that is narrower or shorter than the native minimap.
    """
    if img_map is None or img_minimap is None:
        return False
    if img_map.size == 0 or img_minimap.size == 0:
        return False
    map_h, map_w = img_map.shape[:2]
    minimap_h, minimap_w = img_minimap.shape[:2]
    return map_h >= minimap_h and map_w >= minimap_w

def get_player_location_on_minimap(
        img_minimap,
        minimap_player_color=(136, 255, 255),
        color_tolerance=0,
        min_component_area=4,
    ):
    """
    Detects the player's position on the minimap.

    The function works by:
    - Creating a binary mask around the configured player color.
    - Keeping connected clusters above the configured minimum area.
    - Selecting the cluster with the smallest average color error.
    - Falling back to a compact, bright yellow HSV cluster when capture-card
      interpolation changes the marker's saturation over a rope or platform.

    Returns:
        (x, y): The player's location in minimap coordinates as a tuple.
                Returns None if not enough matching pixels are found.
    """
    target = np.asarray(minimap_player_color, dtype=np.int16)
    tolerance = max(0, int(color_tolerance))
    lower = np.clip(target - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(target + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(img_minimap, lower, upper)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    min_component_area = max(1, int(min_component_area))
    components = [
        label
        for label in range(1, num_labels)
        if stats[label, cv2.CC_STAT_AREA] >= min_component_area
    ]
    if components:
        # Tolerance matching can include terrain pixels, so compare valid
        # clusters instead of averaging unrelated pixels across the map.
        target_i16 = target.astype(np.int16)

        def component_score(label):
            pixels = img_minimap[labels == label].astype(np.int16)
            mean_color_error = np.mean(
                np.max(np.abs(pixels - target_i16), axis=1)
            )
            # Prefer the cluster closest to the configured player color; use
            # area only as the tie-breaker.
            return (mean_color_error, -stats[label, cv2.CC_STAT_AREA])

        best = min(components, key=component_score)
        center = centroids[best]
    elif tolerance > 0:
        # The player marker is alpha-blended with the minimap.  On a green rope
        # its B channel can move outside the per-channel tolerance even though
        # its hue remains yellow.  Use HSV only as a strict-detection fallback.
        hsv = cv2.cvtColor(img_minimap, cv2.COLOR_BGR2HSV)
        target_bgr = np.asarray(minimap_player_color, dtype=np.uint8).reshape(
            1, 1, 3
        )
        target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0, 0]
        hue = hsv[:, :, 0].astype(np.int16)
        target_hue = int(target_hsv[0])
        hue_delta = np.abs(hue - target_hue)
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        hue_tolerance = max(4, min(12, int(round(tolerance / 7))))
        saturation_min = max(40, int(target_hsv[1]) - tolerance)
        value_min = max(150, int(target_hsv[2]) - int(round(tolerance * 1.5)))
        hsv_mask = (
            (hue_delta <= hue_tolerance)
            & (hsv[:, :, 1] >= saturation_min)
            & (hsv[:, :, 2] >= value_min)
        ).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = \
            cv2.connectedComponentsWithStats(hsv_mask, connectivity=8)

        # A real marker stays compact.  Reject large yellow map artwork before
        # scoring candidates so the fallback cannot lock onto terrain.
        min_side = min(img_minimap.shape[:2])
        max_marker_span = max(6, int(round(min_side * 0.06)))
        max_marker_area = max(16, max_marker_span * max_marker_span)
        components = [
            label
            for label in range(1, num_labels)
            if min_component_area <= stats[label, cv2.CC_STAT_AREA]
            <= max_marker_area
            and stats[label, cv2.CC_STAT_WIDTH] <= max_marker_span
            and stats[label, cv2.CC_STAT_HEIGHT] <= max_marker_span
        ]
        if not components:
            return None

        def hsv_component_score(label):
            component_mask = labels == label
            mean_hue_error = np.mean(hue_delta[component_mask])
            mean_value = np.mean(hsv[:, :, 2][component_mask])
            # Hue is stable across alpha blending; brightness and area resolve
            # ties between otherwise similar compact components.
            return (
                mean_hue_error,
                -mean_value,
                -stats[label, cv2.CC_STAT_AREA],
            )

        best = min(components, key=hsv_component_score)
        center = centroids[best]
    else:
        return None

    loc_player_minimap = (
        int(round(center[0])),
        int(round(center[1])),
    )

    return loc_player_minimap

def get_all_other_player_locations_on_minimap(img_minimap, red_bgr=(0, 0, 255)):
    '''
    Detect red dot (0,0,255) and calculate the center to define as other player position.
    '''
    red_bgr = tuple(map(int, red_bgr))
    # 智能選擇容錯範圍：從較小開始，如果檢測不到就增加
    tolerances = [10, 20, 30, 40]  # 嘗試不同的容錯範圍
    
    for tolerance in tolerances:
        lower_bgr = tuple(max(0, c - tolerance) for c in red_bgr)
        upper_bgr = tuple(min(255, c + tolerance) for c in red_bgr)

        # 使用範圍檢測
        mask = cv2.inRange(img_minimap, lower_bgr, upper_bgr)
        coords = cv2.findNonZero(mask)

        if coords is not None and len(coords) >= 3:
            logger.debug(f"Found {len(coords)} red pixels with tolerance {tolerance}")
            logger.debug(f"Color range: {lower_bgr} to {upper_bgr}")
            return [tuple(pt[0]) for pt in coords]  # List of (x, y)

    # 如果所有容錯範圍都檢測不到，記錄調試信息
    logger.debug(f"Red dot detection failed with all tolerances: {tolerances}")
    return []

def debug_minimap_colors(img_minimap, target_color=(0, 0, 255)):
    """
    調試函數：分析小地圖中的顏色分布，幫助找到正確的紅色點顏色值
    """
    # 保存原始小地圖
    cv2.imwrite("debug_minimap_original.png", img_minimap)
    
    # 分析顏色分布
    h, w = img_minimap.shape[:2]
    colors_found = {}
    
    # 掃描整個小地圖，統計顏色
    for y in range(0, h, 2):  # 每2個像素取一個樣本以提高效率
        for x in range(0, w, 2):
            color = tuple(img_minimap[y, x])
            if color not in colors_found:
                colors_found[color] = 0
            colors_found[color] += 1
    
    # 找出最常見的顏色（排除黑色和白色）
    sorted_colors = sorted(colors_found.items(), key=lambda x: x[1], reverse=True)
    
    logger.info("=== Minimap Color Analysis ===")
    logger.info(f"Target color (BGR): {target_color}")
    logger.info("Top 10 most common colors:")
    
    for i, (color, count) in enumerate(sorted_colors[:10]):
        if color != (0, 0, 0) and color != (255, 255, 255):  # 排除純黑和純白
            logger.info(f"  {i+1}. BGR{color}: {count} pixels")
            
            # 檢查是否接近目標顏色
            diff = sum(abs(c1 - c2) for c1, c2 in zip(color, target_color))
            if diff < 50:  # 如果顏色差異小於50
                logger.info(f"    *** Close to target color! Difference: {diff} ***")
    
    # 創建不同容錯範圍的檢測結果
    for tolerance in [10, 20, 30, 40, 50]:
        lower_bgr = tuple(max(0, c - tolerance) for c in target_color)
        upper_bgr = tuple(min(255, c + tolerance) for c in target_color)
        mask = cv2.inRange(img_minimap, lower_bgr, upper_bgr)
        coords = cv2.findNonZero(mask)
        count = len(coords) if coords is not None else 0
        logger.info(f"Tolerance {tolerance}: Found {count} pixels")
        cv2.imwrite(f"debug_red_detection_tolerance_{tolerance}.png", mask)
    
    return sorted_colors

def get_bar_percent(img):
    '''
    Get HP/MP/EXP bar ratio with given bar image

    Return: float [0.0 - 1.0]
    '''
    # Sample a horizontal line at the vertical center of the bar
    h, w = img.shape[:2]
    line_pixels = img[h // 2, :]

    # Get left white boundary of bar
    lb = 0
    while lb < w and np.all(line_pixels[lb] >= 255):
        lb += 1

    # Get right white boundary of bar
    rb = w - 1
    while rb > lb and np.all(line_pixels[rb] >= 255):
        rb -= 1

    # Sanity check
    if rb <= lb:
        return 0.0

    # Get unfill pixel count in bar
    unfill_pixel_cnt = 0
    tolerance = 10
    for i in range(lb, rb + 1):
        r, g, b = line_pixels[i]
        if  abs(int(r) - int(g)) <= tolerance and \
            abs(int(r) - int(b)) <= tolerance and \
            int(r) > 0:
            unfill_pixel_cnt += 1

    # Compute fill ratio
    total_width = rb - lb + 1
    fill_width = total_width - unfill_pixel_cnt
    fill_ratio = fill_width / total_width if total_width > 0 else 0.0
    return fill_ratio*100

def nms_matches(matches, iou_thresh=0.0):
    '''
    Apply non-maximum suppression to remove overlapping matches.

    Args:
        matches: List of tuples (idx, loc, score, shape)
        iou_thresh: IoU threshold to trigger suppression (default 0.0 = any overlap)

    Returns:
        List of filtered matches (same format as input)
    '''
    filtered = matches.copy()
    i = 0
    while i < len(filtered):
        j = i + 1
        while j < len(filtered):
            _, loc_i, score_i, shape_i = filtered[i]
            _, loc_j, score_j, shape_j = filtered[j]

            box_i = (loc_i[0], loc_i[1],
                     loc_i[0] + shape_i[1], loc_i[1] + shape_i[0])
            box_j = (loc_j[0], loc_j[1],
                     loc_j[0] + shape_j[1], loc_j[1] + shape_j[0])

            if get_iou(box_i, box_j) > iou_thresh:
                if score_i > score_j:
                    filtered.pop(i)
                    i -= 1
                    break
                else:
                    filtered.pop(j)
                    j -= 1
            j += 1
        i += 1

    return filtered

def get_window_region_mac(window_title):
    '''
    Get window region on macOS using Quartz
    '''
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID
    )
    # Get all exist windows
    all_titles = []
    for window in window_list:
        title = window.get(Quartz.kCGWindowName, '')
        owner = window.get(Quartz.kCGWindowOwnerName, '')
        if title:
            all_titles.append(f"{title} (Owner: {owner})")
    logger.debug(f"all_titles: {all_titles}")
    for window in window_list:
        if window.get(Quartz.kCGWindowName, '') == window_title:
            bounds = window.get(Quartz.kCGWindowBounds, {})
            return {
                "left": int(bounds.get('X', 0)),
                "top": int(bounds.get('Y', 0)),
                "width": int(bounds.get('Width', 0)),
                "height": int(bounds.get('Height', 0))
            }
    return None


def click_in_game_window(window_title, coord):
    '''
    Mouse click on a game window coordinate
    '''
    # game_window = gw.getWindowsWithTitle(window_title)[0]
    # win_left, win_top = game_window.left, game_window.top

    # If mac then coord / 2 and y position + 3
    if is_mac():
        coord = (coord[0] // 2, coord[1] // 2 + 10)

    if is_mac():
        # macOS implementation using Quartz
        region = get_window_region_mac(window_title)
        if region is None:
            text = f"Cannot find window: {window_title}"
            logger.error(text)
            raise RuntimeError(text)
        win_left, win_top = region["left"], region["top"]
    else:
        # Windows implementation using pygetwindow
        game_window = gw.getWindowsWithTitle(window_title)[0]
        win_left, win_top = game_window.left, game_window.top

    loc_click = (win_left + coord[0], win_top + coord[1])
    pyautogui.click(loc_click)
    logger.info(f"[click_in_game_window] click at {loc_click}")

def send_email(email_addr, password,
               to, subject, body, attachment_path):
    '''
    send_email
    '''
    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = email_addr
    msg['To'] = to

    # Attach PNG image
    with open(attachment_path, 'rb') as f:
        file_data = f.read()
        maintype, subtype = mimetypes.guess_type(attachment_path)[0].split('/')
        filename = f.name.split("/")[-1]
        msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=filename)

    # Send Email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(email_addr, password)
        smtp.send_message(msg)
        logger.info(f"[send_email] {subject} to {to}")

def check_inbox(email_addr, password, token):
    '''
    Check inbox for replies containing the expected token in the subject
    '''
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(email_addr, password)
    imap.select("inbox")

    # IMAP search: only look for subjects that contain token
    status, messages = imap.search(None, f'(SUBJECT "{token}")')
    if status != "OK":
        logger.error("Search failed")
        imap.logout()
        return None

    for num in messages[0].split():
        status, data = imap.fetch(num, '(RFC822)')
        msg = email.message_from_bytes(data[0][1])
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
                imap.logout()
                return body.strip()

    imap.logout()
    return None

def mask_route_colors(img_map, img_route, color_code):
    """
    Masks all pixels in img_route where img_map contains any route color.
    Pixels at those positions in img_route are set to black (0,0,0).
    """
    # Parse color_code keys to list of RGB tuples
    target_colors = [tuple(map(int, color_str.split(','))) for color_str in color_code.keys()]

    # Ensure dimensions match
    if img_map.shape[:2] != img_route.shape[:2]:
        logger.warning("[mask_route_colors] Resizing img_map from "
                       f"{img_map.shape} to {img_route.shape}")
        img_map = cv2.resize(img_map, (img_route.shape[1], img_route.shape[0]))

    # Build mask for each color
    mask = np.zeros(img_map.shape[:2], dtype=bool)
    for color in target_colors:
        matches = np.all(img_map == color, axis=-1)
        mask |= matches

    # Apply mask to img_route (set those pixels to black)
    img_route[mask] = (0, 0, 0)

    return img_route

def activate_game_window(window_title):
    '''
    activate_game_window
    This function only support Windows OS
    '''
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        raise Exception(f"Cannot find window with title: {window_title}")

    try:
        # Try to restore the window first
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

        # Try to set foreground
        win32gui.SetForegroundWindow(hwnd)

        logger.info(f"[activate_game_window] Set game window to foreground")
    except:
        # If SetForegroundWindow fails, try alternative methods
        win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetActiveWindow(hwnd)

def get_game_window_title_by_token(token, exact_match=False):
    '''
    Find a visible window whose title matches `token`.

    Works on Windows OS. The bot is not tied to "MapleStory Worlds" - any
    program window can be targeted by passing the right `token`.

    Args:
        token: the (sub)string to look for in window titles.
        exact_match: when True, only a window whose full title equals `token`
                     (case-insensitive) is returned. When False, a substring
                     match is used; an exact match is still preferred when one
                     exists among the candidates.

    Returns:
        The matched window title, or None if nothing matched.
    '''
    if not token:
        return None

    def callback(hwnd, matches):
        # Only consider visible windows with a non-empty title
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        if exact_match:
            if token.lower() == title.lower():
                matches.append(title)
        elif token.lower() in title.lower():
            matches.append(title)

    matches = []
    win32gui.EnumWindows(callback, matches)

    if not matches:
        return None

    # Prefer an exact (case-insensitive) match when available
    for title in matches:
        if title.lower() == token.lower():
            return title
    return matches[0]

def list_visible_window_titles():
    '''
    Return a de-duplicated, sorted list of the titles of all visible top-level
    windows. Used by the UI to let the user pick the target program.

    Cross-platform: uses win32 on Windows and Quartz on macOS.
    '''
    titles = []
    if is_mac():
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly |
            Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        for window in window_list:
            title = window.get(Quartz.kCGWindowName, '')
            if title:
                titles.append(title)
    else:
        def callback(hwnd, collected):
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd)
            if title:
                collected.append(title)
        win32gui.EnumWindows(callback, titles)

    # De-duplicate while keeping things tidy for the UI
    return sorted(set(titles), key=lambda s: s.lower())

def is_img_16_to_9(img, cfg):
    """
    Check if image aspect ratio is approximately 16:9.
    """
    tolerance = cfg["game_window"]["ratio_tolerance"]
    h, w = img.shape[:2]
    return abs(w/h - 16/9) <= tolerance

def normalize_pixel_coordinate(coord, window_size):
    '''
    Normalize pixel coordinate from current window size to standard (693x1282).
    '''
    h_win, w_win = window_size
    h_std, w_std = (693, 1282)

    # Standard size, no need to normalize
    if h_win == h_std and w_win == w_std:
        return coord

    scale_y = h_std / h_win
    scale_x = w_std / w_win

    x, y = coord
    norm_y = round(y * scale_y)
    norm_x = round(x * scale_x)

    logger.info("[normalize_pixel_coordinate] "\
                f"Normalized coord{coord} to coord{(norm_x, norm_y)}")

    return (norm_x, norm_y)

def resize_window(window_title, width=1296, height=759):
    # 取得視窗句柄
    hwnd = win32gui.FindWindow(None, window_title)
    if hwnd == 0:
        logger.warning(f"找不到視窗: {window_title}")
        return None

    # MoveWindow cannot resize a maximized/minimized window reliably. Restore
    # it first so auto_resize means what the configuration says it means.
    window_placement = win32gui.GetWindowPlacement(hwnd)
    show_command = window_placement[1]
    non_restored_states = {
        win32con.SW_SHOWMINIMIZED,
        win32con.SW_SHOWMAXIMIZED,
        win32con.SW_MINIMIZE,
    }
    if show_command in non_restored_states:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.05)

    # 取得目前視窗位置
    rect = win32gui.GetWindowRect(hwnd)
    x, y = rect[0], rect[1]  # 保持視窗左上角位置不變

    # 調整視窗大小
    win32gui.MoveWindow(hwnd, x, y, width, height, True)
    actual_rect = win32gui.GetWindowRect(hwnd)
    actual_size = (
        actual_rect[2] - actual_rect[0],
        actual_rect[3] - actual_rect[1],
    )
    requested_size = (int(width), int(height))
    if actual_size != requested_size:
        logger.warning(
            f"視窗「{window_title}」要求調整為 {requested_size[0]}x"
            f"{requested_size[1]}，實際為 {actual_size[0]}x{actual_size[1]}"
        )
    else:
        logger.info(
            f"已將「{window_title}」調整為 {actual_size[0]}x{actual_size[1]}"
        )
    return actual_size
