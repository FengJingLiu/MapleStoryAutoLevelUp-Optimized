'''
Execute this script:
python AutoDiceRoller.py --attribute 4,4,13,4 --cfg XXX
'''
# Standard import
import time
import argparse
import sys

# Library import
import numpy as np
import cv2

# Local import
from src.utils.logger import logger
from src.utils.common import (
    find_pattern_sqdiff, screenshot, load_image,
    is_mac, override_cfg, load_yaml, click_in_game_window,
)
from src.input.CaptureFramePreprocessor import preprocess_capture_frame
from src.input.CaptureSource import (
    DIRECTSHOW_SOURCE,
    capture_profile_override,
    create_capture_source,
    resolve_capture_source,
)
from src.input.KeyBoardListener import KeyBoardListener
if is_mac():
    from src.input.GameWindowCapturorForMac import GameWindowCapturor
else:
    from src.input.GameWindowCapturor import GameWindowCapturor


DICE_REFERENCE_SIZE = (700, 1296)  # height, width
DICE_ROLL_POINT = (981, 445)
DICE_FIRST_BOX = (890, 371)
DICE_BOX_SIZE = (22, 37)  # height, width
DICE_BOX_Y_INTERVAL = 25

class AutoDiceRoller:
    '''
    AutoDiceRoller
    '''
    def __init__(self, args):
        '''
        Init AutoDiceRoller
        '''
        # self.cfg = Config # Configuration
        self.args = args # User arguments
        self.fps = 0 # Frame per second
        self.is_first_frame = True # first frame flag
        self.is_enable = True
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_gray = None # game window frame graysale
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = np.zeros((10, 10, 3), dtype=np.uint8) # minimap on game screen
        # Timers
        self.t_last_frame = time.time() # Last frame timer, for fps calculation

        # Load defautl yaml config
        cfg = load_yaml("config/config_default.yaml")
        # Override with platform config
        if is_mac():
            cfg = override_cfg(cfg, load_yaml("config/config_macOS.yaml"))
        # Override with user customized config
        self.cfg = override_cfg(cfg, load_yaml(f"config/config_{args.cfg}.yaml"))

        # Set up fps limit
        self.fps_limit = self.cfg["system"]["fps_limit_auto_dice_roller"]

        # Load number image
        # Templates are loaded at their authored scale.  DirectShow users must
        # replace these with native 4K crops; legacy PNGs are never enlarged.
        self.img_numbers = [
            load_image(f"numbers/{i}.png", cv2.IMREAD_GRAYSCALE)
            for i in range(4, 14)
        ]
        self._runtime_geometry_size = None
        self.loc_dice = DICE_ROLL_POINT
        self.loc_first_box = DICE_FIRST_BOX
        self.box_size = DICE_BOX_SIZE
        self.box_y_interval = DICE_BOX_Y_INTERVAL
        self.debug_ui_y_start = self.cfg["ui_coords"]["ui_y_start"]

        # Start keyboard listener thread
        self.kb = KeyBoardListener(self.cfg, is_autobot=False)

        # Start the configured window or DirectShow capture source.
        logger.info("Starting configured capture source")
        self.capture_source = resolve_capture_source(self.cfg)
        self.capture = create_capture_source(
            self.cfg,
            window_capture_cls=GameWindowCapturor,
        )

    def refresh_runtime_geometry(self, output_size):
        """Scale dice ROI geometry while preserving authored template pixels."""
        output_h, output_w = map(int, output_size[:2])
        if output_h <= 0 or output_w <= 0:
            raise ValueError(
                f"Invalid dice output size: {(output_h, output_w)}"
            )

        reference = self.cfg.get("game_window", {}).get(
            "coordinate_reference_size", DICE_REFERENCE_SIZE
        )
        if not isinstance(reference, (list, tuple)) or len(reference) != 2:
            raise ValueError(
                "game_window.coordinate_reference_size must contain "
                "[height, width]"
            )
        reference_h, reference_w = map(int, reference)
        if reference_h <= 0 or reference_w <= 0:
            raise ValueError(
                "game_window.coordinate_reference_size values must be positive"
            )

        geometry_key = (
            output_h,
            output_w,
            reference_h,
            reference_w,
        )
        if geometry_key == self._runtime_geometry_size:
            return

        scale_x = output_w / reference_w
        scale_y = output_h / reference_h

        def scale_point(point):
            return (
                int(round(point[0] * scale_x)),
                int(round(point[1] * scale_y)),
            )

        self.loc_dice = scale_point(DICE_ROLL_POINT)
        self.loc_first_box = scale_point(DICE_FIRST_BOX)
        self.box_size = (
            max(1, int(round(DICE_BOX_SIZE[0] * scale_y))),
            max(1, int(round(DICE_BOX_SIZE[1] * scale_x))),
        )
        self.box_y_interval = max(
            1, int(round(DICE_BOX_Y_INTERVAL * scale_y))
        )
        self.debug_ui_y_start = max(
            1,
            int(round(
                self.cfg["ui_coords"]["ui_y_start"] * scale_y
            )),
        )

        self._runtime_geometry_size = geometry_key
        logger.info(
            "[AutoDiceRoller] Scaled dice ROI geometry from "
            f"{(reference_h, reference_w)} to {(output_h, output_w)}"
        )

    def click_dice(self):
        """Click only when capture coordinates belong to a local window."""
        if self.capture_source == DIRECTSHOW_SOURCE:
            remote = self.cfg.get("esp32_hid", {}).get(
                "remote_target", False
            )
            target = "remote HDMI target" if remote else "local target window"
            raise RuntimeError(
                "AutoDiceRoller DirectShow recognition is available, but "
                f"clicking the {target} is not supported; no click was sent"
            )
        window_title = getattr(
            self.capture,
            "window_title",
            self.cfg["game_window"]["title"],
        )
        click_in_game_window(window_title, self.loc_dice)

    def stop(self):
        """Release listener and capture resources, including DirectShow."""
        if getattr(self, "kb", None) is not None:
            self.kb.stop()
            self.kb = None
        if getattr(self, "capture", None) is not None:
            self.capture.stop()
            self.capture = None

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        cv2.imshow("Game Window Debug",
            self.img_frame_debug[:self.debug_ui_y_start, :])
        # Update FPS timer
        self.t_last_frame = time.time()

    def run_once(self):
        '''
        Process one game window frame
        '''
        # Get window game raw frame
        self.frame = self.capture.get_frame()
        if self.frame is None:
            logger.warning("Failed to capture game frame.")
            return

        try:
            self.img_frame, _ = preprocess_capture_frame(
                self.frame,
                self.cfg,
                window_title=getattr(self.capture, "window_title", ""),
                capture_profile=capture_profile_override(self.capture),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(f"[capture] {exc}")
            return

        self.refresh_runtime_geometry(self.img_frame.shape[:2])

        # Grayscale game window
        self.img_frame_gray = cv2.cvtColor(self.img_frame, cv2.COLOR_BGR2GRAY)

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Enable cached location since second frame
        self.is_first_frame = False

        # Check if user want to disable dice rolling
        if self.kb.is_pressed_func_key[0]: # 'F1' is pressed
            self.is_enable = not self.is_enable
            logger.info(f"User press F1, is_enable = {self.is_enable}")
            self.kb.is_pressed_func_key[0] = False

        # Check if need to save screenshot
        if self.kb.is_pressed_func_key[1]: # 'F2' is pressed
            screenshot(self.img_frame)
            self.kb.is_pressed_func_key[1] = False

        target_active = (
            self.capture_source == DIRECTSHOW_SOURCE
            or self.kb.is_game_window_active()
        )
        if self.is_enable and target_active:

            # Parse the attribute number
            attibutes_info = []
            for i, attibute in enumerate(["STR", "DEX", "INT", "LUK"]):
                # Calculate the box position
                p0 = (
                    self.loc_first_box[0],
                    self.loc_first_box[1] + i * self.box_y_interval,
                )
                p1 = (
                    p0[0] + self.box_size[1],
                    p0[1] + self.box_size[0],
                )

                # Crop the box region from the image
                img_roi = self.img_frame_gray[p0[1]:p1[1], p0[0]:p1[0]]

                # Match with each number template (from 4 to 11)
                best_score = float('inf')
                best_digit = None
                for idx, img_number in enumerate(self.img_numbers, start=4):
                    _, score, _ = find_pattern_sqdiff(img_roi, img_number)
                    if score < best_score:
                        best_score = score
                        best_digit = idx
                logger.info(f"[{attibute}]: {best_digit} (score: {round(best_score, 2)})")
                attibutes_info.append((best_digit, best_score))

                # Draw box and put text on debug image
                cv2.rectangle(self.img_frame_debug, p0, p1, (0, 0, 255), 1)
                cv2.putText(
                    self.img_frame_debug,
                    f"{best_digit}",
                    (p0[0], p0[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # for val, score in attibutes_info:
            #     if score > 0.11:
            #         logger.warning(f"Stop! Unable to recognize number: {(val, score)})")
            #         self.is_enable = False

            # Check if is equal to target
            is_jackpot = True
            for i, (val, score) in enumerate(attibutes_info):
                target = self.args.attribute[i]
                if target is not None and target != val:
                    is_jackpot = False

            # Stop rolling dice if reach target
            if is_jackpot:
                self.is_enable = False
                logger.info("Hit Jackpot! Stop!")

            # Click to roll the dice or not
            if self.is_enable:
                self.click_dice()
                logger.info("Roll the dice")

        # Show debug image on window
        self.update_img_frame_debug()

def parse_and_validate_attributes(attr_str):
    raw_values = attr_str.split(',')
    if len(raw_values) != 4:
        raise argparse.ArgumentTypeError("You must provide exactly 4 attributes: STR,DEX,INT,LUK")

    parsed = []
    total_known = 0
    unknown_count = 0

    for v in raw_values:
        if v.strip() == '?':
            parsed.append(None)
            unknown_count += 1
        else:
            try:
                val = int(v)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid attribute value: {v}")
            if not (4 <= val <= 13):
                raise argparse.ArgumentTypeError("Each attribute must be between 4 and 13.")
            parsed.append(val)
            total_known += val

    if unknown_count > 0 and (total_known > 25 or total_known + 4 * unknown_count > 25):
        raise argparse.ArgumentTypeError("Impossible to satisfy sum of 25 with current values.")

    return parsed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--attribute',
        type=parse_and_validate_attributes,
        default=[4, 4, 13, 4],
        help='Assign the attributes in order: STR,DEX,INT,LUK.'
             'Each must be between 4-13 and total must sum to 25.'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='custom',
        help='Choose customized config yaml file in config/'
    )

    autoDiceRoller = None
    try:
        autoDiceRoller = AutoDiceRoller(parser.parse_args())
    except Exception as e:
        logger.error(f"AutoDiceRoller Init failed: {e}")
        sys.exit(1)
    else:
        try:
            while True:
                t_start = time.time()

                # Process one game window frame
                autoDiceRoller.run_once()

                # Exit if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # Cap FPS to save system resource
                frame_duration = time.time() - t_start
                target_duration = 1.0 / autoDiceRoller.fps_limit
                if frame_duration < target_duration:
                    time.sleep(target_duration - frame_duration)
        finally:
            autoDiceRoller.stop()
            cv2.destroyAllWindows()
