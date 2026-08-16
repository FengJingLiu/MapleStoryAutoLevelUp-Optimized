'''
Auto generate route map 
'''
# Standard import
import time
import argparse
import sys
import os
from copy import deepcopy
from pathlib import Path

# CV import
import numpy as np
import cv2

# local import
from src.utils.logger import logger
from src.utils.common import (
    find_pattern_sqdiff, draw_rectangle, screenshot,
    get_minimap_loc_size, get_player_location_on_minimap,
    copy_minimap_native_raster, copy_minimap_native_location,
    route_map_can_fit_minimap,
    load_yaml, override_cfg, is_mac, load_image,
)
from src.input.KeyBoardListener import KeyBoardListener, normalize_key_name
from src.input.Esp32KeyForwarder import Esp32KeyForwarder
from src.input.GameWindowCapturor import GameWindowCapturor
from src.input.CaptureFramePreprocessor import preprocess_capture_frame
from src.input.CaptureSource import (
    capture_profile_override,
    create_capture_source,
)
from src.utils.frame_geometry import (
    LEGACY_FRAME_SIZE,
    scale_runtime_pixel_config,
)
from src.utils.minimap_geometry import (
    build_minimap_geometry,
    load_minimap_geometry,
    minimap_geometry_path,
    serialize_minimap_geometry,
)


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


def route_forward_keys_from_config(cfg):
    """Return local gameplay keys that should be mirrored to computer B."""
    keys = {"left", "right", "up", "down"}
    keys.update(cfg.get("key", {}).values())
    keys.update(cfg.get("buff_skill", {}).get("keys", []))
    keys.update(cfg.get("route_recoder", {}).get("forward_keys", []))
    normalized = {normalize_key_name(key) for key in keys if key}

    # Route-recorder controls stay local on computer A even if a user happens
    # to include one of them in the optional forwarding list.
    return normalized.difference({"f1", "f2", "f3", "f4"})


def prepare_minimap_for_alignment(
        img_minimap,
        player_location,
        player_radius=None,
        max_colored_marker_area=None,
    ):
    """Return a minimap copy and mask with moving UI markers excluded.

    Player/party dots move independently of the minimap background.  Including
    them in template matching can shift a repeating platform map by one level.
    The capture-card path also reduces these dots to only a few pixels, so the
    old HSV cleanup (which required a component larger than 10 pixels) could
    not remove them.
    """
    alignment_image = img_minimap.copy()
    match_mask = np.any(
        alignment_image != [0, 0, 0], axis=2
    ).astype(np.uint8) * 255
    excluded = np.zeros(alignment_image.shape[:2], dtype=np.uint8)
    min_side = min(alignment_image.shape[:2])
    if player_radius is None:
        # About five pixels on the former 149px-tall Wisdom Forest raster and
        # ten pixels on its 293px native capture.
        player_radius = max(5, int(round(min_side * 0.034)))
    if max_colored_marker_area is None:
        # Component area grows quadratically when the UI raster grows.
        marker_span = max(10, int(round(min_side * 0.067)))
        max_colored_marker_area = marker_span * marker_span

    if player_location is not None:
        px, py = map(int, player_location)
        cv2.circle(
            excluded,
            (px, py),
            radius=max(1, int(player_radius)),
            color=255,
            thickness=-1,
        )

    # Other players/party members are rendered as compact red or blue dots.
    # Select only small, saturated components so real minimap artwork is kept.
    hsv = cv2.cvtColor(alignment_image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored_dot_mask = (
        (
            (hue <= 8)
            | (hue >= 170)
            | ((hue >= 85) & (hue <= 130))
        )
        & (saturation >= 100)
        & (value >= 70)
    ).astype(np.uint8) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        colored_dot_mask,
        connectivity=8,
    )
    max_area = max(1, int(max_colored_marker_area))
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area <= max_area:
            excluded[labels == label] = 255

    # Cover interpolation halos around the tiny HDMI-scaled marker pixels.
    excluded = cv2.dilate(
        excluded,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    match_mask[excluded != 0] = 0
    alignment_image[excluded != 0] = (0, 0, 0)
    return alignment_image, match_mask


def select_stable_minimap_match(
        img_map,
        img_minimap,
        last_result,
        mask=None,
        local_search_radius=35,
        local_accept_threshold=0.22,
        global_accept_threshold=0.22,
        teleport_score_margin=0.03,
    ):
    """Prefer continuity while allowing a clearly better distant match.

    A portal can legitimately move the minimap by an arbitrary distance in a
    single frame, so distance must never be a hard rejection criterion.  The
    previous location is only a tie-breaker when repeated map artwork gives
    similarly good local and global candidates.
    """
    if last_result is None:
        location, score, _ = find_pattern_sqdiff(
            img_map,
            img_minimap,
            mask=mask,
            global_threshold=0.0,
        )
        accepted = (
            np.isfinite(score)
            and score <= global_accept_threshold
        )
        return location, score, accepted

    global_location, global_score, _ = find_pattern_sqdiff(
        img_map,
        img_minimap,
        mask=mask,
        global_threshold=0.0,
    )
    local_location, local_score, local_found = find_pattern_sqdiff(
        img_map,
        img_minimap,
        last_result=last_result,
        mask=mask,
        local_search_radius=local_search_radius,
        # Any finite local SQDIFF result is returned so it can be compared with
        # the independently calculated global result below.
        global_threshold=float("inf"),
    )
    local_ok = (
        local_found
        and np.isfinite(local_score)
        and local_score <= local_accept_threshold
    )
    global_ok = (
        np.isfinite(global_score)
        and global_score <= global_accept_threshold
    )

    if not global_ok:
        if local_ok:
            return local_location, local_score, True
        return last_result, min(local_score, global_score), False
    if not local_ok or global_location == local_location:
        return global_location, global_score, True

    # A materially better distant result represents a portal/camera jump and
    # is accepted immediately, regardless of displacement.  Similar scores
    # indicate repeated artwork, where continuity is the safer tie-breaker.
    if global_score + teleport_score_margin < local_score:
        return global_location, global_score, True
    return local_location, local_score, True


def fill_empty_canvas_pixels(canvas, source, location):
    """Copy visible source pixels into empty canvas pixels at ``location``.

    Route pixels already drawn on the canvas are deliberately preserved.  This
    lets the route background follow newly explored map areas without erasing
    recorded movement/action colors.
    """
    x, y = map(int, location)
    h, w = source.shape[:2]
    canvas_slice = canvas[y:y+h, x:x+w]
    if canvas_slice.shape[:2] != source.shape[:2]:
        return False

    source_visible = np.any(source != [0, 0, 0], axis=2)
    canvas_empty = np.all(canvas_slice == [0, 0, 0], axis=2)
    fill_mask = source_visible & canvas_empty
    canvas_slice[fill_mask] = source[fill_mask]
    return True


def fit_debug_preview(img, preferred_scale, max_size=None):
    """Scale a debug image up, but keep the full image inside ``max_size``."""
    h, w = img.shape[:2]
    scale = max(float(preferred_scale), 0.01)
    if max_size is not None:
        max_w, max_h = max_size
        if max_w > 0 and max_h > 0:
            scale = min(scale, max_w / w, max_h / h)
    scale = max(scale, 0.01)

    dst_w = max(1, int(round(w * scale)))
    dst_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (dst_w, dst_h), interpolation=interpolation), scale


def get_debug_monitor_work_size():
    """Return the work-area size of the monitor containing the debug windows."""
    if os.name != "nt":
        return None
    try:
        import win32api
        import win32con
        import win32gui
    except ImportError:
        return None

    # These windows are created earlier in the same frame.  Prefer the route
    # window after it exists so moving it to another monitor is also handled.
    for title in ("Route Map Debug", "Game Window Debug", "Map"):
        hwnd = win32gui.FindWindow(None, title)
        if not hwnd:
            continue
        monitor = win32api.MonitorFromWindow(
            hwnd,
            win32con.MONITOR_DEFAULTTONEAREST,
        )
        left, top, right, bottom = win32api.GetMonitorInfo(monitor)["Work"]
        return right - left, bottom - top
    try:
        monitor = win32api.MonitorFromPoint(
            (0, 0),
            win32con.MONITOR_DEFAULTTONEAREST,
        )
        left, top, right, bottom = win32api.GetMonitorInfo(monitor)["Work"]
        return right - left, bottom - top
    except Exception:
        return None


def get_debug_preview_max_size(cfg, monitor_size=None):
    """Return the configured debug-preview limit for one monitor.

    This helper deliberately affects only images passed to ``cv2.imshow``.
    The route and minimap canvases retain their native coordinate systems.
    ``None`` is valid in headless/non-Windows environments and means no
    monitor-derived limit is available.
    """
    if monitor_size is None:
        monitor_size = get_debug_monitor_work_size()
    if monitor_size is None:
        return None

    minimap_cfg = cfg.get("minimap", {}) if isinstance(cfg, dict) else {}
    screen_ratio = minimap_cfg.get("debug_window_max_screen_ratio", 0.85)
    screen_ratio = min(max(float(screen_ratio), 0.1), 1.0)
    return tuple(max(1, int(size * screen_ratio)) for size in monitor_size)


def prepare_route_output_directory(map_dir, confirm=input):
    """Prepare one recording directory without deleting its saved map.

    A repeated recording normally means replacing route*.png while reusing the
    stitched map.png.  Other files in the map directory are also preserved.
    """
    if not os.path.isdir(map_dir):
        os.makedirs(map_dir, exist_ok=True)
        logger.info(f"Created new directory: {map_dir}")
        return True

    answer = confirm(
        f"[Warning] Directory '{map_dir}' already exists. "
        "Clear route*.png only and keep map.png? (y/n): "
    ).strip().lower()
    if answer != "y":
        return False

    removed = 0
    for name in os.listdir(map_dir):
        lower_name = name.lower()
        if not (lower_name.startswith("route") and lower_name.endswith(".png")):
            continue
        path = os.path.join(map_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed += 1

    map_path = os.path.join(map_dir, "map.png")
    logger.info(
        f"Cleared {removed} route image(s) from {map_dir}; "
        f"map.png {'preserved' if os.path.isfile(map_path) else 'not present'}"
    )
    return True


def stage_png_write(path, image):
    """Encode and verify a PNG beside its destination before replacement."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.recording.tmp{target.suffix}")
    encoded_ok, encoded = cv2.imencode(target.suffix, image)
    if not encoded_ok:
        raise OSError(f"Unable to encode image: {target}")
    temporary.write_bytes(encoded.tobytes())

    decoded = cv2.imdecode(
        np.frombuffer(temporary.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if (
        decoded is None
        or decoded.shape != image.shape
        or decoded.dtype != image.dtype
        or not np.array_equal(decoded, image)
    ):
        try:
            temporary.unlink()
        except OSError:
            pass
        raise OSError(f"PNG verification failed: {target}")
    return temporary


def stage_text_write(path, content):
    """Write and verify UTF-8 text beside its final destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.recording.tmp{target.suffix}")
    temporary.write_text(content, encoding="utf-8")
    if temporary.read_text(encoding="utf-8") != content:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise OSError(f"Text verification failed: {target}")
    return temporary


def commit_staged_files(staged_files):
    """Replace a staged file set, restoring all old targets on any failure."""
    token = f"{os.getpid()}.{time.time_ns()}"
    backups = []
    committed = set()
    try:
        for target, temporary in staged_files:
            target = Path(target)
            backup = None
            if target.exists():
                backup = target.with_name(
                    f".{target.name}.recording.backup.{token}"
                )
                os.replace(target, backup)
            backups.append((target, backup))
            os.replace(temporary, target)
            committed.add(target)
    except OSError:
        rollback_errors = []
        for target, backup in reversed(backups):
            try:
                if backup is not None and backup.exists():
                    if target.exists():
                        target.unlink()
                    os.replace(backup, target)
                elif target in committed and target.exists():
                    target.unlink()
            except OSError as rollback_error:
                rollback_errors.append(str(rollback_error))
        if rollback_errors:
            logger.error(
                "Recording bundle rollback was incomplete: "
                + "; ".join(rollback_errors)
            )
        raise
    else:
        for _, backup in backups:
            if backup is not None:
                try:
                    backup.unlink()
                except OSError as exc:
                    logger.warning(
                        f"Unable to remove recording backup {backup}: {exc}"
                    )


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
        visual_scale = self.get_frame_visual_scale()
        preferred_text_y_start = int(round(550 * visual_scale))
        text_y_interval = max(1, int(round(23 * visual_scale)))
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

        # The debug window only displays the game area above ``ui_y_start``.
        # Keep the complete text block inside that crop instead of drawing the
        # final shortcut lines into the hidden game-UI area below it.
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 * visual_scale
        font_thickness = max(1, int(round(2 * visual_scale)))
        text_size, text_baseline = cv2.getTextSize(
            "Ag", font_face, font_scale, font_thickness
        )
        ui_y_start = getattr(self, "cfg", {}).get(
            "ui_coords", {}
        ).get("ui_y_start", self.img_frame_debug.shape[0])
        visible_height = min(self.img_frame_debug.shape[0], int(ui_y_start))
        edge_margin = max(1, int(round(4 * visual_scale)))
        first_baseline_min = text_size[1] + edge_margin
        last_baseline_max = max(
            first_baseline_min,
            visible_height - text_baseline - edge_margin,
        )
        text_y_start = min(
            preferred_text_y_start,
            last_baseline_max - text_y_interval * (len(text_list) - 1),
        )
        text_y_start = max(first_baseline_min, text_y_start)

        for idx, text in enumerate(text_list):
            cv2.putText(
                self.img_frame_debug, text,
                (max(1, int(round(10 * visual_scale))),
                 text_y_start + text_y_interval*idx),
                font_face, font_scale, (0, 0, 255),
                font_thickness, cv2.LINE_AA
            )

        # Draw minimap rectangle on img debug
        draw_rectangle(
            self.img_frame_debug,
            self.loc_minimap,
            self.img_minimap_screen.shape[:2],
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
        frame_h, frame_w = self.img_frame_debug.shape[:2]
        mini_map_crop, _ = fit_debug_preview(
            mini_map_crop,
            preferred_scale=3 * visual_scale,
            max_size=(
                max(1, int(round(frame_w * 0.35))),
                max(1, int(round(visible_height * 0.45))),
            ),
        )
        # Paste into top-right corner of self.img_frame_debug
        h_crop, w_crop = mini_map_crop.shape[:2]
        paste_margin = max(1, int(round(10 * visual_scale)))
        x_paste = max(0, frame_w - w_crop - paste_margin)
        preferred_y = max(0, int(round(70 * visual_scale)))
        y_paste = min(preferred_y, max(0, visible_height - h_crop - paste_margin))
        self.img_frame_debug[y_paste:y_paste + h_crop, x_paste:x_paste + w_crop] = mini_map_crop

        # Draw border around minimap
        cv2.rectangle(
            self.img_frame_debug,
            (x_paste, y_paste),
            (x_paste + w_crop, y_paste + h_crop),
            color=(255, 255, 255),   # White border
            thickness=max(1, int(round(2 * visual_scale)))
        )

    def get_frame_visual_scale(self):
        """Return a visualization scale relative to the legacy game frame."""
        frame_h, frame_w = self.img_frame_debug.shape[:2]
        base_cfg = getattr(self, "_cfg_reference", None)
        if not isinstance(base_cfg, dict):
            base_cfg = getattr(self, "cfg", {})
        reference = base_cfg.get("game_window", {}).get(
            "coordinate_reference_size",
            LEGACY_FRAME_SIZE,
        )
        try:
            ref_h, ref_w = map(float, reference[:2])
            if ref_h <= 0 or ref_w <= 0:
                raise ValueError
        except (TypeError, ValueError, IndexError):
            ref_h, ref_w = LEGACY_FRAME_SIZE
        return max(0.1, min(frame_h / ref_h, frame_w / ref_w))

    def update_img_frame_debug(self):
        '''
        update_img_frame_debug
        '''
        ui_y_start = self.cfg.get("ui_coords", {}).get(
            "ui_y_start",
            self.img_frame_debug.shape[0],
        )
        visible_height = min(
            self.img_frame_debug.shape[0],
            max(1, int(ui_y_start)),
        )
        debug_crop = self.img_frame_debug[:visible_height, :]
        debug_preview, _ = fit_debug_preview(
            debug_crop,
            preferred_scale=1.0,
            max_size=get_debug_preview_max_size(self.cfg),
        )
        cv2.imshow("Game Window Debug", debug_preview)
        # Update FPS timer
        self.t_last_frame = time.time()

    def update_runtime_config(self, output_size):
        """Derive fresh frame-space settings from the unscaled base config."""
        output_size = tuple(map(int, output_size[:2]))
        if output_size == getattr(self, "_runtime_output_size", None):
            return

        base_cfg = getattr(self, "_cfg_reference", None)
        if not isinstance(base_cfg, dict):
            base_cfg = deepcopy(self.cfg)
            self._cfg_reference = base_cfg
        self.cfg = scale_runtime_pixel_config(base_cfg, output_size)
        self._runtime_output_size = output_size

        reference = base_cfg.get("game_window", {}).get(
            "coordinate_reference_size",
            LEGACY_FRAME_SIZE,
        )
        logger.info(
            "[capture] runtime pixel config "
            f"reference={tuple(reference[:2])} output={output_size}"
        )

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
            f"Minimap,score({round(self.minimap_match_score, 2)})"
            f"{',held' if getattr(self, 'minimap_match_held', False) else ''}",
            (self.loc_minimap_global[0], self.loc_minimap_global[1]+15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (0, 255, 255), 1
        )

        # Draw player center
        cv2.circle(self.img_route_debug,
                   loc_player_global, radius=2,
                   color=(0, 255, 255), thickness=-1)

        return loc_player_global

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
            base_cfg = getattr(self, "_cfg_reference", self.cfg)
            img_frame, geometry = preprocess_capture_frame(
                self.frame,
                base_cfg,
                window_title=getattr(self.capture, "window_title", ""),
                capture_profile=capture_profile_override(self.capture),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.error(f"[capture] {exc}")
            return
        if geometry["profile"] == "potplayer":
            x0, y0, x1, y1 = geometry["video_roi"]
            self.img_capture_content = self.frame[y0:y1, x0:x1]
        else:
            self.img_capture_content = None
        self.update_runtime_config(geometry["output_size"])
        geometry_key = (
            geometry["profile"],
            geometry["source_size"],
            geometry["video_roi"],
            geometry["content_size"],
            geometry["working_size"],
            geometry["normalized"],
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

    def _apply_locked_minimap_rect(self, minimap_rect):
        """Crop one already validated, border-free minimap rectangle."""
        x, y, w, h = map(int, minimap_rect)
        frame_h, frame_w = self.img_frame.shape[:2]
        if x < 0 or y < 0 or w <= 0 or h <= 0 \
                or x + w > frame_w or y + h > frame_h:
            logger.error(
                "Locked minimap rectangle is outside the current frame: "
                f"rect={(x, y, w, h)}, frame={(frame_w, frame_h)}"
            )
            return False

        self.loc_minimap = (x, y)
        self.minimap_screen_size = (h, w)
        self.img_minimap_screen = self.img_frame[y:y+h, x:x+w]
        self.img_minimap_source = self.img_minimap_screen
        return True

    def update_minimap_from_current_frame(self):
        """Calibrate once, then keep one fixed minimap crop for the session."""
        if getattr(self, "_minimap_geometry_error", False):
            return False

        frame_size = tuple(map(int, self.img_frame.shape[:2]))
        locked_rect = getattr(self, "_locked_minimap_rect", None)
        locked_frame_size = getattr(self, "_minimap_lock_frame_size", None)
        if locked_rect is not None:
            if frame_size != locked_frame_size:
                logger.error(
                    "Game-frame size changed after minimap calibration: "
                    f"{locked_frame_size[::-1]} -> {frame_size[::-1]}. "
                    "Recording is paused; restart the recorder so one route "
                    "cannot mix coordinate systems."
                )
                self._minimap_geometry_error = True
                self.is_enable = False
                self.break_route_segment()
                return False
            return self._apply_locked_minimap_rect(locked_rect)

        saved_geometry = getattr(self, "_saved_minimap_geometry", None)
        if saved_geometry is not None:
            saved_frame_size = tuple(saved_geometry["frame_size"])
            if frame_size != saved_frame_size:
                logger.error(
                    "Current game-frame size does not match the preserved "
                    "map geometry: "
                    f"saved={saved_frame_size[::-1]}, "
                    f"current={frame_size[::-1]}. Use a fresh map directory."
                )
                self._minimap_geometry_error = True
                self.is_enable = False
                self.break_route_segment()
                return False
            self._locked_minimap_rect = tuple(saved_geometry["minimap_rect"])
            self._minimap_lock_frame_size = saved_frame_size
            logger.info(
                "Using preserved fixed minimap crop: "
                f"{self._locked_minimap_rect}"
            )
            return self._apply_locked_minimap_rect(
                self._locked_minimap_rect
            )

        calibration_frame_size = getattr(
            self, "_minimap_calibration_frame_size", None
        )
        if calibration_frame_size != frame_size:
            self._minimap_calibration_frame_size = frame_size
            self._minimap_rect_samples = []

        minimap_result = get_minimap_loc_size(self.img_frame)
        if minimap_result is None:
            logger.warning("Minimap not found; waiting for the next frame")
            return False

        x, y, w, h = map(int, minimap_result)
        # Discard one remaining pixel on each edge so the white border cannot
        # leak into route-map matching.
        interior_rect = (x + 1, y + 1, w - 2, h - 2)
        if interior_rect[2] <= 0 or interior_rect[3] <= 0:
            logger.warning("Detected minimap has no usable interior")
            return False

        samples = getattr(self, "_minimap_rect_samples", None)
        if samples is None:
            samples = []
            self._minimap_rect_samples = samples
        samples.append(interior_rect)
        route_cfg = getattr(self, "cfg", {}).get("route_recoder", {})
        required_samples = max(
            1, int(route_cfg.get("minimap_lock_frames", 5))
        )
        if len(samples) < required_samples:
            logger.info(
                "Calibrating fixed minimap crop: "
                f"{len(samples)}/{required_samples}"
            )
            return False

        recent_samples = np.asarray(samples[-required_samples:], dtype=np.int32)
        locked_rect = tuple(
            int(round(value)) for value in np.median(recent_samples, axis=0)
        )
        self._locked_minimap_rect = locked_rect
        self._minimap_lock_frame_size = frame_size
        self._minimap_rect_samples = []
        logger.info(
            "Locked minimap crop for this recording session: "
            f"rect={locked_rect}, frame={frame_size[::-1]}"
        )
        return self._apply_locked_minimap_rect(locked_rect)

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
        self._last_route_action = None
        self.minimap_match_score = 0.0
        self.minimap_match_held = False
        self.input_forwarder = None
        self.kb = None
        self.capture = None
        self.map_dir = None
        # Images
        self.frame = None # raw image
        self.img_frame = None # game window frame
        self.img_frame_debug = None # game window frame for visualization
        self.img_route = None # route map
        self.img_route_debug = None # route map for visualization
        self.img_minimap = None # minimap on game screen
        self.img_map = None # map
        self._native_minimap_size = None
        self._last_native_minimap_error = None
        self._last_route_map_size_error = None
        self._locked_minimap_rect = None
        self._minimap_lock_frame_size = None
        self._minimap_calibration_frame_size = None
        self._minimap_rect_samples = []
        self._minimap_geometry_error = False
        self._saved_minimap_geometry = None
        self.completed_routes = []
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
        # Always derive frame-space settings from this untouched snapshot.
        # Re-deriving from ``self.cfg`` after a fullscreen/legacy transition
        # would compound x/y scaling. Route and minimap coordinates are not
        # among the settings scaled by ``scale_runtime_pixel_config``.
        self._cfg_reference = deepcopy(self.cfg)
        self._runtime_output_size = None

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

        # Re-recording a route must preserve the previously stitched map.
        self.map_dir = os.path.join("minimaps", args.new_map)
        if not prepare_route_output_directory(self.map_dir):
            sys.exit(0)

        # An explicit --map takes precedence. Otherwise automatically continue
        # from this map directory's preserved map.png when one exists.
        if self.args.map != '':
            self.img_map = load_image(f"{self.args.map}")
        else:
            existing_map_path = os.path.join(self.map_dir, "map.png")
            if os.path.isfile(existing_map_path):
                self.img_map = load_image(existing_map_path)
                logger.info(f"Loaded preserved map: {existing_map_path}")

        if self.img_map is not None:
            geometry_dir = self.map_dir
            if self.args.map != '':
                explicit_map_dir = os.path.dirname(self.args.map)
                if explicit_map_dir:
                    geometry_dir = explicit_map_dir
            self._saved_minimap_geometry = load_minimap_geometry(geometry_dir)

        try:
            route_cfg = self.cfg.get("route_recoder", {})
            forward_input = route_cfg.get(
                "forward_input_to_esp32",
                self.cfg.get("esp32_hid", {}).get("remote_target", False),
            )
            if forward_input:
                self.input_forwarder = Esp32KeyForwarder(
                    self.cfg,
                    route_forward_keys_from_config(self.cfg),
                )

            # Start keyboard listener thread. Function keys remain local, while
            # gameplay key edges are queued to the ESP32 forwarder above.
            event_handler = (
                self.input_forwarder.handle_key_event
                if self.input_forwarder is not None
                else None
            )
            self.kb = KeyBoardListener(
                self.cfg,
                is_autobot=False,
                key_event_handler=event_handler,
            )

            # Start the configured window or DirectShow capture source.
            logger.info("Starting configured capture source")
            self.capture = create_capture_source(
                self.cfg,
                window_capture_cls=GameWindowCapturor,
            )
            self.wait_for_initial_capture_frame()
        except BaseException:
            self.stop()
            raise

    def stop(self):
        """Stop capture/listener threads and guarantee remote key release."""
        if self.kb is not None:
            self.kb.stop()
            self.kb = None
        if self.input_forwarder is not None:
            self.input_forwarder.close()
            self.input_forwarder = None
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

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

    def break_route_segment(self):
        """Prevent the next valid sample from bridging an invalid time gap."""
        self.loc_player_global_last = None
        self._last_route_action = None

    def record_route_sample(self, action, is_draw_blob):
        """Draw one action sample while keeping unrelated segments separate."""
        if not getattr(self, "is_enable", False) or not action \
                or self.img_route is None:
            self.break_route_segment()
            return False

        dict_action_to_color = {v: k for k, v in self.color_code.items()}
        color_rgb = dict_action_to_color.get(action)
        if color_rgb is None:
            logger.error(f"Unknown route action; skipping sample: {action}")
            self.break_route_segment()
            return False
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        px, py = map(int, self.loc_player_global)

        if is_draw_blob:
            # A discrete action must never leave a pending point that can be
            # connected across its cooldown or to a later movement action.
            self.break_route_segment()
            force_draw = action == "none none goal"
            dt = time.time() - self.t_last_draw_blob
            if force_draw or dt > self.cfg["route_recoder"]["blob_cooldown"]:
                cv2.circle(
                    self.img_route,
                    (px, py),
                    radius=2,
                    color=color_bgr,
                    thickness=-1,
                )
                self.t_last_draw_blob = time.time()
                return True
            return False

        if action != getattr(self, "_last_route_action", None):
            self.break_route_segment()

        previous = self.loc_player_global_last
        if previous is None:
            previous = (px, py)
        cv2.line(
            self.img_route,
            tuple(map(int, previous)),
            (px, py),
            color=color_bgr,
            thickness=1,
        )
        self.loc_player_global_last = (px, py)
        self._last_route_action = action
        return True

    def _recording_geometry(self):
        """Return and pre-validate the geometry saved beside a route bundle."""
        minimap_rect = getattr(self, "_locked_minimap_rect", None)
        frame_size = getattr(self, "_minimap_lock_frame_size", None)
        if minimap_rect is None:
            x, y = self.loc_minimap
            h, w = self.minimap_screen_size
            minimap_rect = (x, y, w, h)
        if frame_size is None:
            frame_size = self.img_frame.shape[:2]
        return build_minimap_geometry(frame_size, minimap_rect)

    def save_recording_bundle(self, include_current_route=False):
        """Save map, geometry, and every completed route at one canvas size."""
        routes = list(getattr(self, "completed_routes", []))
        if include_current_route:
            if self.img_route is None:
                logger.error("Unable to save route: route canvas is missing")
                return False
            routes.append(self.img_route.copy())

        map_shape = self.img_map.shape
        for index, route in enumerate(routes, start=1):
            if route.shape != map_shape:
                logger.error(
                    "Refusing inconsistent route bundle: "
                    f"map={map_shape}, route{index}={route.shape}"
                )
                return False

        try:
            geometry = self._recording_geometry()
        except (AttributeError, TypeError, ValueError) as exc:
            logger.error(f"Unable to save minimap geometry: {exc}")
            return False

        image_targets = [(Path(self.map_dir) / "map.png", self.img_map)]
        image_targets.extend(
            (Path(self.map_dir) / f"route{index}.png", route)
            for index, route in enumerate(routes, start=1)
        )
        staged = []
        try:
            for target, image in image_targets:
                staged.append((target, stage_png_write(target, image)))
            geometry_target = minimap_geometry_path(self.map_dir)
            staged.append(
                (
                    geometry_target,
                    stage_text_write(
                        geometry_target,
                        serialize_minimap_geometry(geometry),
                    ),
                )
            )
            commit_staged_files(staged)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            logger.error(f"Unable to save recording bundle: {exc}")
            return False
        finally:
            for _, temporary in staged:
                try:
                    temporary.unlink()
                except OSError:
                    pass

        logger.info(
            f"Saved recording bundle: {self.map_dir} "
            f"({len(routes)} route image(s), canvas={map_shape[1]}x{map_shape[0]})"
        )
        return True

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

        if self.img_route is not None \
                and self.img_route.shape[:2] != (map_h, map_w):
            route_h, route_w = self.img_route.shape[:2]
            raise RuntimeError(
                "Route canvas lost alignment before expansion: "
                f"map={(map_w, map_h)}, route={(route_w, route_h)}"
            )
        completed_routes = getattr(self, "completed_routes", [])
        for index, route in enumerate(completed_routes, start=1):
            route_h, route_w = route.shape[:2]
            if (route_h, route_w) != (map_h, map_w):
                raise RuntimeError(
                    "Completed route lost alignment before expansion: "
                    f"map={(map_w, map_h)}, "
                    f"route{index}={(route_w, route_h)}"
                )

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

        expanded_routes = []
        for index, route in enumerate(completed_routes, start=1):
            route_h, route_w = route.shape[:2]
            new_route = np.zeros((new_h, new_w, 3), dtype=route.dtype)
            new_route[
                expand_top:expand_top + route_h,
                expand_left:expand_left + route_w,
            ] = route
            expanded_routes.append(new_route)
        self.completed_routes = expanded_routes

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
            self.break_route_segment()
            return -1 # Wait for game window to be ready
        else:
            self.img_frame = img_frame

        # Image for debug use
        self.img_frame_debug = self.img_frame.copy()

        # Calibrate one fixed minimap rectangle for this session. Per-frame
        # rectangle jitter would otherwise move the entire route coordinate
        # system even when the in-game camera is stationary.
        if not self.update_minimap_from_current_frame():
            self.break_route_segment()
            return -1
        native_size = self.img_minimap_source.shape[:2]
        expected_size = getattr(self, "_native_minimap_size", None)
        if expected_size is not None and native_size != expected_size:
            error_key = (expected_size, native_size)
            if error_key != getattr(
                self, "_last_native_minimap_error", None
            ):
                logger.error(
                    "Native minimap raster changed from "
                    f"{expected_size[::-1]} to {native_size[::-1]}; "
                    "discarding this frame to preserve route coordinates"
                )
                self._last_native_minimap_error = error_key
            self.break_route_segment()
            return -1
        self._native_minimap_size = native_size
        self._last_native_minimap_error = None
        self.img_minimap = copy_minimap_native_raster(
            self.img_minimap_source
        )

        if self.img_map is not None and not route_map_can_fit_minimap(
            self.img_map, self.img_minimap
        ):
            map_size = self.img_map.shape[:2]
            error_key = (map_size, native_size)
            if error_key != getattr(
                self, "_last_route_map_size_error", None
            ):
                logger.error(
                    "Existing map.png is too small for the native minimap: "
                    f"map={map_size[::-1]}, minimap={native_size[::-1]}. "
                    "Back up the old map directory and create a fresh native "
                    "map instead of reusing this canvas"
                )
                self._last_route_map_size_error = error_key
            self.break_route_segment()
            return -1
        self._last_route_map_size_error = None

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
            loc_player_minimap = copy_minimap_native_location(
                loc_player_minimap_source
            )
        if loc_player_minimap is None:
            now = time.monotonic()
            if now - getattr(self, "_last_player_warning_time", 0.0) >= 1.0:
                logger.warning(
                    "Player dot not found on minimap; skipping this frame"
                )
                self._last_player_warning_time = now
            self.break_route_segment()
            return -1
        self.loc_player_minimap = loc_player_minimap

        # Build the image used for map stitching before template matching.
        # Dynamic player/party markers must never influence the map offset or
        # be copied into the persistent map image.
        img_minimap_alignment, minimap_match_mask = \
            prepare_minimap_for_alignment(
                self.img_minimap,
                self.loc_player_minimap,
            )

        # Update map
        if self.is_first_frame:
            # copy minimap to map
            if self.img_map is None:
                self.img_map = img_minimap_alignment.copy()
                pad = self.cfg["route_recoder"]["map_padding"]
                self.img_map = cv2.copyMakeBorder(
                    self.img_map,
                    top=pad, bottom=pad, left=pad, right=pad,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(0, 0, 0)  # Black padding
                )
                self.loc_minimap_global = (pad, pad)
                self.minimap_match_score = 0.0
                self.minimap_match_held = False
            else:
                (
                    self.loc_minimap_global,
                    self.minimap_match_score,
                    match_accepted,
                ) = select_stable_minimap_match(
                        self.img_map,
                        img_minimap_alignment,
                        last_result=None,
                        mask=minimap_match_mask,
                        global_accept_threshold=self.cfg["route_recoder"].get(
                            "global_match_threshold", 0.22
                        ),
                    )
                self.minimap_match_held = not match_accepted
                if not match_accepted:
                    error_key = round(float(self.minimap_match_score), 4)
                    if error_key != getattr(
                        self, "_last_initial_match_error", None
                    ):
                        logger.error(
                            "Existing map.png does not match the locked "
                            "minimap crop closely enough: "
                            f"score={self.minimap_match_score:.4f}. "
                            "Use the map's saved geometry or create a fresh "
                            "map directory."
                        )
                        self._last_initial_match_error = error_key
                    self.break_route_segment()
                    return -1
                self._last_initial_match_error = None
                x, y = self.loc_minimap_global
                h, w = self.img_minimap.shape[:2]
                self.ensure_img_map_capacity(x, y, h, w)

            # Update route
            self.img_route = self.remove_color_code_pixels(self.img_map.copy())
            self.img_route_debug = self.img_route.copy()

        else:
            # Perform template matching to find where the current minimap fits in the global map
            self.loc_minimap_global, self.minimap_match_score, match_accepted = \
                select_stable_minimap_match(
                self.img_map,
                img_minimap_alignment,
                last_result=self.loc_minimap_global,
                mask=minimap_match_mask,
                local_search_radius=self.cfg["route_recoder"].get(
                    "local_search_radius", 35
                ),
                local_accept_threshold=self.cfg["route_recoder"].get(
                    "local_match_threshold", 0.22
                ),
                global_accept_threshold=self.cfg["route_recoder"].get(
                    "global_match_threshold", 0.22
                ),
                teleport_score_margin=self.cfg["route_recoder"].get(
                    "teleport_score_margin", 0.03
                ),
            )
            self.minimap_match_held = not match_accepted
            if not match_accepted:
                self.break_route_segment()
                return -1
            x, y = self.loc_minimap_global
            h, w = self.img_minimap.shape[:2]
            # Ensure img_map is big enough to fit the newly explored region
            self.ensure_img_map_capacity(x, y, h, w)
            x, y = self.loc_minimap_global

            # Update map
            # A rejected jump keeps the last reliable location for preview, but
            # must not write the unmatched frame into the persistent canvas.
            if self.args.map == '' and match_accepted:
                if not fill_empty_canvas_pixels(
                    self.img_map,
                    img_minimap_alignment,
                    (x, y),
                ):
                    logger.warning(
                        "Skip map update: minimap lies outside expanded map "
                        f"at {(x, y, w, h)}"
                    )
                    self.break_route_segment()
                    return -1

                # img_route is a separate canvas.  Expanding it kept the
                # coordinates aligned, but previously left all newly explored
                # floors black.  Fill only empty background pixels so existing
                # route/action colors remain untouched.
                route_background = self.remove_color_code_pixels(
                    img_minimap_alignment.copy()
                )
                if not fill_empty_canvas_pixels(
                    self.img_route,
                    route_background,
                    (x, y),
                ):
                    logger.warning(
                        "Skip route background update: minimap lies outside "
                        f"route canvas at {(x, y, w, h)}"
                    )
                    self.break_route_segment()
                    return -1

        cv2.imshow("Map", self.img_map)
        self.img_route_debug = self.img_route.copy()

        # Use the current-frame hero-marker centroid without route smoothing.
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

        # Any key release, action change, discrete action, or invalid frame
        # breaks continuity so unrelated samples cannot be joined by a line.
        self.record_route_sample(action, is_draw_blob)

        # Save route image if goal is drawn
        if action == "none none goal":
            if self.save_recording_bundle(include_current_route=True):
                self.completed_routes.append(self.img_route.copy())
                self.idx_routes = len(self.completed_routes)
                self.img_route = self.remove_color_code_pixels(
                    self.img_map.copy()
                )
                self.break_route_segment()

        # Save img_map to map.png
        if self.kb.is_pressed_func_key[3]: # 'F4' is pressed
            self.save_recording_bundle(include_current_route=False)
            self.kb.is_pressed_func_key[3] = False

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

        # Enlarge the route preview while keeping the entire growing canvas on
        # the monitor.  This changes visualization only; route coordinates and
        # the saved map retain their native resolution.
        monitor_size = get_debug_monitor_work_size()
        max_preview_size = None
        if monitor_size is not None:
            screen_ratio = self.cfg["minimap"].get(
                "debug_window_max_screen_ratio", 0.85
            )
            screen_ratio = min(max(float(screen_ratio), 0.1), 1.0)
            max_preview_size = tuple(
                max(1, int(size * screen_ratio)) for size in monitor_size
            )
        img_route_preview, _ = fit_debug_preview(
            self.img_route_debug,
            self.cfg["minimap"]["debug_window_upscale"],
            max_size=max_preview_size,
        )
        cv2.imshow("Route Map Debug", img_route_preview)

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
        try:
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
        except KeyboardInterrupt:
            logger.info("RouteRecorder interrupted by user")
        finally:
            routeRecorder.stop()
            cv2.destroyAllWindows()
