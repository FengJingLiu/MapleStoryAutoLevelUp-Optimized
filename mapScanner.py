import argparse
import sys
import time

import cv2
import numpy as np
# Local import
from logger import logger
from GameWindowCapturor import GameWindowCapturor
from config.config import Config
from util import draw_rectangle, load_image, find_pattern_sqdiff, get_mask


class MapScanner:
    '''
    MapScanner
    '''
    def __init__(self):
        self.cfg = Config

        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_debug = None
        self.img_camera = None
        self.img_camera_gray = None
        # NameTag
        self.img_nametag = load_image("name_tag.png")
        self.img_nametag_gray = load_image("name_tag.png", cv2.IMREAD_GRAYSCALE)
        # Cooridnate of top-left
        self.loc_nametag = (0, 0) # nametag location on window

        # Feature extraction
        self.feature_extractor = cv2.ORB_create(500)
        
        # Feature matching
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # map_canvas = np.zeros((output_map_size[1], output_map_size[0], 3), dtype=np.uint8)


        # Start game window capturing thread
        logger.info("Waiting for game window to activate, please click on game window")
        self.capture = GameWindowCapturor(self.cfg)

    def update_nametag_loc(self):
        '''
        get player location by detecting player's nametag
        '''
        # Pad search region to avoid edge cut-off issue (full template size)
        (pad_y, pad_x) = self.img_nametag.shape[:2]
        img_roi_padded = cv2.copyMakeBorder(
            self.img_camera_gray,
            pad_y, pad_y, pad_x, pad_x,
            borderType=cv2.BORDER_REPLICATE  # replicate border for safe matching
        )

        # Adjust previous location
        last_result = (
            self.loc_nametag[0] + pad_x,
            self.loc_nametag[1] + pad_y
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
            loc_nametag[1] - pad_y
        )

        # Update name tag location if confidence is good
        if score < self.cfg.nametag_diff_thres:
            self.loc_nametag = loc_nametag

        # Draw name tag detection box for debug
        draw_rectangle(
            self.img_debug, self.loc_nametag, self.img_nametag.shape,
            (0, 255, 0), "")
        text = f"NameTag, {round(score, 2)}, {'cached' if is_cached else 'missed'}"
        cv2.putText(self.img_debug, text,
                    (self.loc_nametag[0], self.loc_nametag[1] + self.img_nametag.shape[0] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    def is_inside_bbox(self, pt, bbox):
        """Check if point pt = (x, y) is inside bounding box"""
        x, y = pt
        x0, y0 = bbox["position"]
        h, w = bbox["size"]
        return x0 <= x <= x0 + w and y0 <= y <= y0 + h


    def runOnce(self):
        '''
        run once
        '''
        # Get lastest game screen frame buffer
        self.frame = self.capture.get_frame()
        # Resize game screen to 1296x759
        self.img_frame = cv2.resize(self.frame, (1296, 759),
                                    interpolation=cv2.INTER_NEAREST)
        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Camera
        self.img_camera = self.img_frame[self.cfg.camera_ceiling:self.cfg.camera_floor, :]
        self.img_camera_gray = cv2.cvtColor(self.img_camera, cv2.COLOR_BGR2GRAY)

        # Debug image
        self.img_debug = self.img_camera.copy()

        self.update_nametag_loc()

        # Generate mask where pixel is exactly (0,0,0)
        black_mask = np.all(self.img_camera == [0, 0, 0], axis=2).astype(np.uint8) * 255
        cv2.imshow("Black Pixel Mask", black_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Black Mask", closed_mask)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

        dynamic_objs = [{
            "position": self.loc_nametag,
            "size": self.img_nametag.shape[:2],
            "score": 1.0,
        }]
        min_area = 1000
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > min_area:
                dynamic_objs.append({
                    "position": (x, y),
                    "size": (h, w),
                    "score": 1.0,
                })

        # Draw monsters bounding box
        for obj in dynamic_objs:
            draw_rectangle(
                self.img_debug, obj["position"], obj["size"],
                (0, 255, 0), str(round(obj['score'], 2))
            )

        # Feature extraction
        kp_all, desc_all = self.feature_extractor.detectAndCompute(self.img_camera_gray, None)

        # Filter out keypoints inside dynamic object bounding boxes
        kp_filtered = []
        desc_filtered = []
        for i, kp in enumerate(kp_all):
            pt = kp.pt
            if any(self.is_inside_bbox(pt, obj) for obj in dynamic_objs):
                continue  # Skip keypoints in dynamic areas
            kp_filtered.append(kp)
            desc_filtered.append(desc_all[i])

        # Convert descriptors to numpy array (OpenCV expects this)
        if desc_filtered:
            desc_filtered = np.array(desc_filtered)
        else:
            desc_filtered = None  # In case no keypoints remain

        # Draw filtered keypoints
        self.img_debug = cv2.drawKeypoints(
            self.img_debug, kp_filtered, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        cv2.imshow("Map Scanner Debug", self.img_debug)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    try:
        # mapScanner = MapScanner(parser.parse_args())
        mapScanner = MapScanner()
    except Exception as e:
        logger.error(f"MapleStoryBot Init failed: {e}")
        sys.exit(1)
    else:
        while True:
            t_start = time.time()

            mapScanner.runOnce()

            # Exit if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Cap FPS to save system resource
            frame_duration = time.time() - t_start
            target_duration = 1.0 / mapScanner.cfg.fps_limit
            if frame_duration < target_duration:
                time.sleep(target_duration - frame_duration)
