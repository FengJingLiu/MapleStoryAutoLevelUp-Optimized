'''
Utility functions
'''

# Standard import
import cv2
import datetime
import os
import platform

#
import numpy as np

# Local import
from logger import logger

OS_NAME = platform.system()

def is_mac():
    return OS_NAME == 'Darwin'

def is_windows():
    return OS_NAME == 'Windows'

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

def screenshot(img, prefix="screenshot"):
    '''
    Save the given image as a screenshot file.

    Parameters:
    - img: numpy array (image to save).

    Behavior:
    - Saves the image to the "screenshot/" directory with the current timestamp as filename.
    '''

    os.makedirs("screenshot", exist_ok=True)  # ensure directory exists
    
    # Generate timestamp string
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"screenshot/{prefix}_{timestamp}.png"
    cv2.imwrite(filename, img)
    logger.info(f"Screenshot saved: {filename}")

def draw_rectangle(img, top_left, size, color, text,
                   thickness=2, text_height=0.7):
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
    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    cv2.putText(img, text, (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, text_height, color, thickness)

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

def get_minimap_loc_size(img_frame):
    '''
    Detects the location and size of the minimap within the game frame.

    The function works by:
    - Thresholding the image get pure white(255,255,255) pixels.
    - Using connected components to find white-bordered regions.
    - Filtering candidates based on expected minimap size and margin rules:
        - Top, bottom, left, right margins must be 1px white lines.

    Returns:
        (x, y, w, h): Top-left coordinate and width/height of the minimap.
                    Returns None if not found.
    '''
    white = np.array([255, 255, 255])

    # Mask for pure white
    mask_white = cv2.inRange(img_frame, white, white)

    # Connected components with stats
    num_labels, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(mask_white, connectivity=8)

    # Loop over components (skip label 0, which is background)
    for i in range(1, num_labels):
        x0, y0, rw, rh, area = stats[i]

        # Filter out small blobs
        if rw < 100 or rh < 100:
            continue

        x1 = x0 + rw - 1
        y1 = y0 + rh - 1

        # Check 1px white top and bottom margins
        if not (np.all(img_frame[y0, x0:x0+rw] == white) and \
                np.all(img_frame[y1, x0:x0+rw] == white)):
            continue

        # Check 1px white left and right margins
        if not (np.all(img_frame[y0:y0:rh, x0] == white) and \
                np.all(img_frame[y0:y0:rh, x1] == white)):
            continue

        # Create a mask of non-white pixels
        mask_minimap = np.any(img_frame[y0:y0+rh, x0:x0+rw] != white, axis=2).astype(np.uint8)

        # Find bounding box of mask_minimap
        coords = cv2.findNonZero(mask_minimap)
        if coords is None:
            continue  # skip empty block
        x_minimap, y_minimap, w_minimap, h_minimap = cv2.boundingRect(coords)

        # Offset by original x0, y0 to get coords in original image
        x_minimap += x0
        y_minimap += y0

        return x_minimap, y_minimap, w_minimap, h_minimap

    logger.warning("Minimap not found in the game frame.")
    return None  # minimap not found

def get_player_location_on_minimap(img_minimap, minimap_player_color=(136, 255, 255)):
    """
    Detects the player's position on the minimap.

    The function works by:
    - Creating a binary mask of all pixels in the minimap that match the configured
    player color exactly.
    - Verifying that at least 4 matching pixels are found (to avoid false positives).
    - Computing the average of these pixel coordinates to determine the center of
    the player icon on the minimap.

    Returns:
        (x, y): The player's location in minimap coordinates as a tuple.
                Returns None if not enough matching pixels are found.
    """
    mask = cv2.inRange(img_minimap,
                        minimap_player_color,
                        minimap_player_color)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 4:
        logger.warning(f"Fail to locate player location on minimap.")
        return None

    # Calculate the average location of the matching pixels
    avg = coords.mean(axis=0)[0]  # shape (1,2), so we take [0]
    loc_player_minimap = (int(round(avg[0])), int(round(avg[1])))

    return loc_player_minimap

def get_bar_ratio(img):
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
    return fill_ratio

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
