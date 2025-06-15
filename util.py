'''
Utility functions
'''

# Standard import
import cv2
import datetime
import os

#
import numpy as np

# Local import
from logger import logger

def load_image(path, mode=cv2.IMREAD_COLOR):
    '''
    Load image from disk and verify existence.
    '''
    if not os.path.exists(path):
        logger.error(f"Image not found: {path}")
        raise FileNotFoundError(f"Image not found: {path}")

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
    # search last result location first to speedup
    h, w = img_pattern.shape[:2]
    if last_result is not None:
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
    # Find pixels matching the player color
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
