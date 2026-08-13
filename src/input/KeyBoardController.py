'''
KeyBoardController
Simulate user keyboard input to control character in the game 
'''
# Standard Import
import threading
import time

# Library import
from pynput import keyboard

# Local import
from src.input.Esp32HidClient import (
    Esp32HidClient,
    Esp32HidTapUncertainError,
    usage_from_text,
)
from src.utils.logger import logger
from src.utils.common import is_mac

if is_mac():
    import Quartz
else:
    import pygetwindow as gw

_input_client = None
_input_client_lock = threading.RLock()
_input_error_lock = threading.Lock()
_last_input_error_time = 0.0
_input_allowed = threading.Event()


def _log_input_error(message):
    """Limit repeated transport error messages while the ESP32 reconnects."""
    global _last_input_error_time
    with _input_error_lock:
        now = time.monotonic()
        if now - _last_input_error_time >= 2.0:
            logger.error(message)
            _last_input_error_time = now


def _get_input_client():
    with _input_client_lock:
        return _input_client


def configure_esp32_input(cfg):
    """Replace the process-wide keyboard output with one ESP32 connection."""
    global _input_client
    _input_allowed.clear()
    with _input_client_lock:
        previous, _input_client = _input_client, None
    if previous is not None:
        previous.close()

    client = Esp32HidClient.from_config(cfg)
    with _input_client_lock:
        _input_client = client
    return client


def close_esp32_input(client=None):
    """Close the active client if it is the one owned by this controller."""
    global _input_client
    _input_allowed.clear()
    with _input_client_lock:
        if client is not None and _input_client is not client:
            return
        closing, _input_client = _input_client, None
    if closing is not None:
        closing.close()


def _call_input(method, *args):
    client = _get_input_client()
    if client is None:
        _log_input_error("[ESP32 HID] Keyboard client is not connected")
        return False
    try:
        return getattr(client, method)(*args)
    except Esp32HidTapUncertainError as exc:
        # The TAP may have executed and only its response was lost. Treat it as
        # consumed so cooldown/action code never replays an uncertain input.
        _log_input_error(
            f"[ESP32 HID] {method} result is uncertain; not retrying: {exc}"
        )
        return True
    except (OSError, RuntimeError, ValueError, ConnectionError) as exc:
        _log_input_error(f"[ESP32 HID] {method} failed: {exc}")
        return False

def key_down(key):
    '''
    Press key down
    '''
    if not key or not _input_allowed.is_set():
        return False
    return _call_input("key_down", key)

def key_up(key):
    '''
    Release key
    '''
    if not key or not _input_allowed.is_set():
        return False
    return _call_input("key_up", key)

def press_key(key, duration=0.05):
    '''
    Simulates a key press for a specified duration
    '''
    if isinstance(key, str) and key.strip() and _input_allowed.is_set():
        duration_ms = max(1, int(round(float(duration) * 1000)))
        return _call_input("tap", key.strip(), duration_ms)
    return False


def release_all_keys():
    """Release every key in one atomic firmware command."""
    if _get_input_client() is None:
        return False
    return _call_input("release_all")


def set_key_state(keys):
    """Atomically replace the persistent key state on the ESP32."""
    if not _input_allowed.is_set():
        return False
    return _call_input("set_state", keys)


def validate_config_keys(cfg):
    """Fail at startup instead of discovering an invalid key during combat."""
    configured = list(cfg.get("key", {}).items())
    configured.extend(
        (f"buff_skill.keys[{index}]", key)
        for index, key in enumerate(cfg.get("buff_skill", {}).get("keys", []))
    )
    for name, key in configured:
        if not key:
            continue
        try:
            usage_from_text(key)
        except ValueError as exc:
            raise ValueError(f"Invalid keyboard mapping {name}={key!r}: {exc}") from exc

class KeyBoardController():
    '''
    KeyBoardController
    '''
    def __init__(self, cfg, connect_input=True, capture_available=True):
        self.cfg = cfg
        self.cmd_action = "none"
        self.cmd_up_down = "none"
        self.cmd_left_right = "none"
        self.cmd_up_down_last = ""
        self.cmd_left_right_last = ""
        self._last_source_action = "none"
        self.window_title = cfg["game_window"]["title"]
        self.fps = 0 # Frame per seconds
        # Timer
        self.t_last_up = 0.0
        self.t_last_down = 0.0
        self.t_last_toggle = 0.0
        self.t_last_screenshot = 0.0
        self.t_last_jump_down = 0.0
        self.t_last_run = time.time()
        self.t_last_skill = 0.0 # Last time character perform action(attack, cast spell, ...)
        self.t_last_buff_cast = [0] * len(self.cfg["buff_skill"]["keys"]) # Last time cast buff skill
        # Flags
        self.is_enable = bool(connect_input)
        self.capture_available = bool(capture_available)
        self.is_need_force_heal = False
        self.is_terminated = False
        # Parameters
        self.debounce_interval = self.cfg["system"]["key_debounce_interval"]
        self.fps_limit = self.cfg["system"]["fps_limit_keyboard_controller"]

        # use 'ctrl', 'alt' for mac, because it's hard to get around
        # macOS's security settings
        if is_mac():
            self.toggle_key = keyboard.Key.ctrl
            self.screenshot_key = keyboard.Key.alt
            self.terminate_key = keyboard.Key.esc
        else:
            self.toggle_key = keyboard.Key.f1
            self.screenshot_key = keyboard.Key.f2
            self.terminate_key = keyboard.Key.f12

        # set up attack key
        self.attack_key = ""
        if cfg["bot"]["attack"] == "aoe_skill":
            self.attack_key = cfg["key"]["aoe_skill"]
        elif cfg["bot"]["attack"] == "directional":
            self.attack_key = cfg["key"]["directional_attack"]
        else:
            raise ValueError(f"Unexpected attack type: {cfg['bot']['attack']}")

        validate_config_keys(cfg)
        if connect_input:
            self.input_client = configure_esp32_input(cfg)
        else:
            close_esp32_input()
            self.input_client = None

        # Start keyboard control thread
        self.thread = threading.Thread(
            target=self.run,
            name="keyboard-controller",
            daemon=True,
        )
        self.thread.start()

        if self.input_client is None:
            logger.info("[KeyBoardController] Keyboard output disabled")
        else:
            logger.info(
                f"[KeyBoardController] ESP32 HID ready over USB serial at "
                f"{self.input_client.endpoint}"
            )

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        self.is_enable = not self.is_enable
        if self.is_enable and getattr(
            self, "capture_available", True
        ) and self.is_game_window_active():
            _input_allowed.set()
        else:
            _input_allowed.clear()
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

        # Make sure all key are released
        self.release_all_key()

    def disable(self):
        '''
        disable keyboard controlller
        '''
        self.is_enable = False
        _input_allowed.clear()
        self.release_all_key()

    def enable(self):
        '''
        enable keyboard controlller
        '''
        self.release_all_key()
        self.is_enable = True
        if getattr(self, "capture_available", True) and self.is_game_window_active():
            _input_allowed.set()

    def set_capture_available(self, available):
        """Gate remote input independently from the user's pause setting."""
        available = bool(available)
        if available == self.capture_available:
            return False
        self.capture_available = available
        if not available:
            _input_allowed.clear()
            self.release_all_key()
        elif self.is_enable and self.is_game_window_active():
            _input_allowed.set()
        return True

    def set_command(self, new_command):
        '''
        Set keyboard command
        '''
        cmd_left_right, cmd_up_down, cmd_action = new_command.split()
        self.cmd_left_right = cmd_left_right
        self.cmd_up_down = cmd_up_down
        # Actions are edge-triggered. Movement persists as state, but a route
        # pixel that remains visible must not generate 15-30 TAPs per second.
        if cmd_action != self._last_source_action:
            self.cmd_action = cmd_action
        elif cmd_action == "none":
            self.cmd_action = "none"
        self._last_source_action = cmd_action

    def is_game_window_active(self):
        '''
        Check if the game window is currently the active (foreground) window.

        Returns:
        - True
        - False
        '''
        # In capture-card mode this process sees PotPlayer on computer A while
        # the BLE keyboard controls computer B. A's foreground window is not a
        # meaningful safety signal for B, so input follows only the bot's
        # enable/pause state.
        if self.cfg.get("esp32_hid", {}).get("remote_target", False):
            return True
        if is_mac():
            active_window = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID
            )
            for window in active_window:
                window_name = window.get(Quartz.kCGWindowName, '')
                if window_name and self.window_title in window_name:
                    return True
            return False
        else:
            try:
                active_window = gw.getActiveWindow()
                if not active_window:
                    return False
                return self.window_title in active_window.title
            except Exception as e:
                return False

    def release_all_key(self):
        '''
        Release all key
        '''
        release_all_keys()

    def update_movement_state(self):
        """Send both movement axes as one deduplicated HID report."""
        keys = []
        if self.cmd_left_right in {"left", "right"}:
            keys.append(self.cmd_left_right)
        elif self.cmd_left_right not in {"stop", "none"}:
            logger.error(
                "[KeyBoardController] Unsupported left-right command: "
                f"{self.cmd_left_right}"
            )

        if self.cmd_up_down in {"up", "down"}:
            keys.append(self.cmd_up_down)
        elif self.cmd_up_down not in {"stop", "none"}:
            logger.error(
                "[KeyBoardController] Unsupported up-down command: "
                f"{self.cmd_up_down}"
            )

        set_key_state(keys)
        self.cmd_left_right_last = self.cmd_left_right
        self.cmd_up_down_last = self.cmd_up_down

    def stop(self):
        """Stop the controller, release the device, and free its serial session."""
        self.is_terminated = True
        _input_allowed.clear()
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning(
                    "[KeyBoardController] Controller thread did not stop within 5 seconds"
                )
        self.release_all_key()
        if self.input_client is not None:
            close_esp32_input(self.input_client)

    def limit_fps(self):
        '''
        Limit FPS
        '''
        # If the loop finished early, sleep to maintain target FPS
        target_duration = 1.0 / self.fps_limit  # seconds per frame
        frame_duration = time.time() - self.t_last_run
        if frame_duration < target_duration:
            time.sleep(target_duration - frame_duration)

        # Update FPS
        self.fps = round(1.0 / (time.time() - self.t_last_run))
        self.t_last_run = time.time()
        # logger.info(f"FPS = {self.fps}")

    def run(self):
        '''
        run
        '''
        was_input_active = False
        try:
            while not self.is_terminated:
                # A physical BLE keyboard follows foreground focus. Release held
                # movement on the active -> inactive edge so another app cannot
                # receive a stuck direction key.
                input_active = (
                    self.is_enable
                    and getattr(self, "capture_available", True)
                    and self.is_game_window_active()
                )
                if not input_active:
                    _input_allowed.clear()
                    if was_input_active:
                        self.release_all_key()
                    was_input_active = False
                    self.limit_fps()
                    continue
                was_input_active = True
                _input_allowed.set()

                # Buff skill
                for i, buff_skill_key in enumerate(self.cfg["buff_skill"]["keys"]):
                    if not buff_skill_key:
                        continue
                    cooldown = self.cfg["buff_skill"]["cooldown"][i]
                    if time.time() - self.t_last_buff_cast[i] >= cooldown and \
                        time.time() - self.t_last_skill > self.cfg["buff_skill"]["action_cooldown"]:
                        if press_key(buff_skill_key):
                            logger.info(f"[Buff] Press buff skill key: '{buff_skill_key}' (cooldown: {cooldown}s)")
                            # Only consume cooldown after ESP32 acknowledged it.
                            self.t_last_buff_cast[i] = time.time()
                            self.t_last_skill = time.time()
                        break

                # Force Heal
                if self.is_need_force_heal:
                    self.cmd_action = "add_hp"

                # Movement is one atomic STATE report. The client suppresses an
                # unchanged state, so the 30 FPS controller loop creates no
                # repeated serial traffic while a direction remains held.
                self.update_movement_state()

                ######################
                ### Action Command ###
                ######################
                if self.cmd_action == "jump":
                    if press_key(self.cfg["key"]["jump"]):
                        self.cmd_action = "none"
                elif self.cmd_action == "teleport":
                    if press_key(self.cfg["key"]["teleport"]):
                        self.cmd_action = "none"
                elif self.cmd_action == "attack":
                    if press_key(self.attack_key):
                        self.t_last_skill = time.time()
                        self.cmd_action = "none"
                elif self.cmd_action == "add_hp":
                    add_hp_key = self.cfg["key"].get("add_hp", "")
                    has_add_hp_key = (
                        isinstance(add_hp_key, str) and bool(add_hp_key.strip())
                    )
                    if not has_add_hp_key or press_key(add_hp_key):
                        self.cmd_action = "none"  # Reset only after acknowledgement
                elif self.cmd_action == "add_mp":
                    add_mp_key = self.cfg["key"].get("add_mp", "")
                    has_add_mp_key = (
                        isinstance(add_mp_key, str) and bool(add_mp_key.strip())
                    )
                    if not has_add_mp_key or press_key(add_mp_key):
                        self.cmd_action = "none"  # Reset only after acknowledgement
                elif self.cmd_action == "goal":
                    pass
                elif self.cmd_action == "none":
                    pass
                else:
                    logger.error("[KeyBoardController] Unsupported action command: "
                                 f"{self.cmd_action}")

                self.limit_fps()
        finally:
            _input_allowed.clear()
            self.release_all_key() # Prevent key keep press down after termination
            if self.input_client is not None:
                close_esp32_input(self.input_client)

        logger.info("[KeyBoardController] terminated")
