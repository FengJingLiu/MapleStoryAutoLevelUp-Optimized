'''
KeyBoardController
Simulate user keyboard input to control character in the game 
'''
# Standard Import
import math
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
_input_transaction_lock = threading.RLock()


ABSOLUTE_MOUSE_MAX_COORDINATE = 32767


def _capture_frame_is_absolute_desktop(cfg):
    """Return whether capture pixels already use the remote desktop plane."""
    section = cfg.get("esp32_hid", {}) if isinstance(cfg, dict) else {}
    value = section.get("capture_frame_is_desktop", False) \
        if isinstance(section, dict) else False
    if not isinstance(value, bool):
        raise ValueError(
            "esp32_hid.capture_frame_is_desktop must be true or false"
        )
    return value


def _absolute_mouse_rect(cfg, name):
    """Return one configured remote-desktop rectangle, or ``None``.

    Rectangles use ``[left, top, width, height]`` in physical pixels on the
    computer receiving BLE HID input.  Width and height describe pixel counts,
    so both must be at least two for endpoint-preserving normalization.
    """
    section = cfg.get("esp32_hid", {}) if isinstance(cfg, dict) else {}
    value = section.get(name) if isinstance(section, dict) else None
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
            isinstance(component, bool) or not isinstance(component, int)
            for component in value):
        raise ValueError(
            f"esp32_hid.{name} must be [left, top, width, height] integers"
        )
    left, top, width, height = map(int, value)
    if width <= 1 or height <= 1:
        raise ValueError(
            f"esp32_hid.{name} width and height must be greater than one"
        )
    return left, top, width, height


def validate_absolute_mouse_config(cfg):
    """Validate the optional capture-to-Magpie absolute-pointer geometry."""
    capture_frame_is_desktop = _capture_frame_is_absolute_desktop(cfg)
    desktop = _absolute_mouse_rect(cfg, "absolute_desktop_rect")
    source = _absolute_mouse_rect(cfg, "magpie_source_rect")
    if capture_frame_is_desktop and desktop is None:
        raise ValueError(
            "esp32_hid.absolute_desktop_rect is required when "
            "capture_frame_is_desktop is true"
        )
    if source is None:
        return desktop, None
    if desktop is None:
        raise ValueError(
            "esp32_hid.absolute_desktop_rect is required when "
            "esp32_hid.magpie_source_rect is configured"
        )

    desktop_left, desktop_top, desktop_width, desktop_height = desktop
    source_left, source_top, source_width, source_height = source
    if not (
        desktop_left <= source_left
        and desktop_top <= source_top
        and source_left + source_width <= desktop_left + desktop_width
        and source_top + source_height <= desktop_top + desktop_height
    ):
        raise ValueError(
            "esp32_hid.magpie_source_rect must lie inside "
            "esp32_hid.absolute_desktop_rect"
        )
    return desktop, source


def has_calibrated_absolute_mouse(cfg):
    """Return whether a Magpie source client was explicitly calibrated."""
    desktop, source = validate_absolute_mouse_config(cfg)
    if _capture_frame_is_absolute_desktop(cfg):
        return desktop is not None
    return source is not None


def capture_point_to_absolute_hid(cfg, x, y, frame_width, frame_height):
    """Map one capture-frame point to normalized absolute HID coordinates.

    Magpie displays a scaled copy of the original game client.  Its input
    transform therefore expects the physical desktop coordinate inside that
    source client, not the corresponding point in the 4K scaled output.  The
    source conversion intentionally rounds to one source pixel before HID
    normalization, matching Magpie's endpoint-preserving integer mapping.

    With no configured rectangles, retain the historical full-frame mapping
    for old/local integrations.  Production capture-card configuration uses
    explicit calibrated rectangles and never takes that fallback.
    """
    try:
        x = float(x)
        y = float(y)
        frame_width = int(frame_width)
        frame_height = int(frame_height)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid capture point or frame geometry") from exc
    if not all(math.isfinite(value) for value in (x, y)) or \
            frame_width <= 1 or frame_height <= 1 or \
            not 0 <= x < frame_width or not 0 <= y < frame_height:
        raise ValueError("capture point must lie inside a non-empty frame")

    desktop, source = validate_absolute_mouse_config(cfg)
    if desktop is None:
        return (
            int(round(x * ABSOLUTE_MOUSE_MAX_COORDINATE /
                      (frame_width - 1))),
            int(round(y * ABSOLUTE_MOUSE_MAX_COORDINATE /
                      (frame_height - 1))),
        )

    desktop_left, desktop_top, desktop_width, desktop_height = desktop
    if _capture_frame_is_absolute_desktop(cfg) or source is None:
        source = desktop
    source_left, source_top, source_width, source_height = source

    source_x = source_left + int(round(
        x * (source_width - 1) / (frame_width - 1)
    ))
    source_y = source_top + int(round(
        y * (source_height - 1) / (frame_height - 1)
    ))
    absolute_x = int(round(
        (source_x - desktop_left) * ABSOLUTE_MOUSE_MAX_COORDINATE /
        (desktop_width - 1)
    ))
    absolute_y = int(round(
        (source_y - desktop_top) * ABSOLUTE_MOUSE_MAX_COORDINATE /
        (desktop_height - 1)
    ))
    if not (
        0 <= absolute_x <= ABSOLUTE_MOUSE_MAX_COORDINATE
        and 0 <= absolute_y <= ABSOLUTE_MOUSE_MAX_COORDINATE
    ):
        raise ValueError("mapped absolute HID point lies outside the desktop")
    return absolute_x, absolute_y


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
    # Signal an in-flight turn transaction to abort at its next permission
    # check, then wait for it to leave the shared HID transaction.
    _input_allowed.clear()
    with _input_transaction_lock:
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
    with _input_transaction_lock:
        with _input_client_lock:
            # A stale controller may finish after a replacement is already
            # active. It must not disable the new session or clear its gate.
            if client is not None and _input_client is not client:
                return
            closing, _input_client = _input_client, None
        _input_allowed.clear()
        if closing is not None:
            closing.close()


def _invoke_input(method, *args):
    """Call the ESP32 and distinguish a valid no-op from a send failure."""
    client = _get_input_client()
    if client is None:
        _log_input_error("[ESP32 HID] Keyboard client is not connected")
        return False, False
    try:
        return True, getattr(client, method)(*args)
    except Esp32HidTapUncertainError as exc:
        # The TAP may have executed and only its response was lost. Treat it as
        # consumed so cooldown/action code never replays an uncertain input.
        _log_input_error(
            f"[ESP32 HID] {method} result is uncertain; not retrying: {exc}"
        )
        return True, True
    except (OSError, RuntimeError, ValueError, ConnectionError) as exc:
        _log_input_error(f"[ESP32 HID] {method} failed: {exc}")
        return False, False


def _invoke_input_confirmed(method, *args):
    """Call a one-shot that must have a confirmed ACK before a UI sequence.

    Session recovery can classify the next captured page after an uncertain
    command. Fixed multi-step menu workflows cannot safely do that: advancing
    would make the next absolute point land on an unknown page. They consume
    the uncertain command without replay, but report failure so the caller
    pauses instead of continuing.
    """
    client = _get_input_client()
    if client is None:
        _log_input_error("[ESP32 HID] Keyboard client is not connected")
        return False, False
    try:
        return True, getattr(client, method)(*args)
    except Esp32HidTapUncertainError as exc:
        _log_input_error(
            f"[ESP32 HID] {method} ACK is uncertain; stopping UI sequence: "
            f"{exc}"
        )
        return False, False
    except (OSError, RuntimeError, ValueError, ConnectionError) as exc:
        _log_input_error(f"[ESP32 HID] {method} failed: {exc}")
        return False, False


def _call_input(method, *args):
    success, result = _invoke_input(method, *args)
    return result if success else False


def _regular_input_allowed():
    return _input_allowed.is_set()


def _input_state_generation():
    """Return the ESP32 held-state continuity epoch when supported."""
    client = _get_input_client()
    generation = getattr(client, "state_continuity_token", None)
    # unittest.mock and legacy clients synthesize arbitrary attributes. Only
    # a concrete integer is a usable continuity token.
    return generation if isinstance(generation, int) else None

def key_down(key):
    '''
    Press key down
    '''
    with _input_transaction_lock:
        if not key or not _regular_input_allowed():
            return False
        return _call_input("key_down", key)

def key_up(key):
    '''
    Release key
    '''
    with _input_transaction_lock:
        if not key or not _regular_input_allowed():
            return False
        return _call_input("key_up", key)

def press_key(key, duration=0.05):
    '''
    Simulates a key press for a specified duration
    '''
    with _input_transaction_lock:
        if (
            isinstance(key, str)
            and key.strip()
            and _regular_input_allowed()
        ):
            duration_ms = max(1, int(round(float(duration) * 1000)))
            return _call_input("tap", key.strip(), duration_ms)
        return False


def release_all_keys():
    """Release every key in one atomic firmware command."""
    # RELEASE_ALL bypasses the recovery deadline, but is serialized with a
    # turn/attack transaction. Clearing _input_allowed first (pause/capture
    # loss) makes a transaction abort after its short turn delay.
    with _input_transaction_lock:
        if _get_input_client() is None:
            return False
        return _call_input("release_all")


def set_key_state(keys):
    """Atomically replace the persistent key state on the ESP32."""
    with _input_transaction_lock:
        if not _regular_input_allowed():
            return False
        success, _ = _invoke_input("set_state", keys)
        # An unchanged state is a successful no-op, not a transport failure.
        return success


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
    def __init__(
            self, cfg, connect_input=True, capture_available=True,
            scheduled_buff_allowed=None):
        self.cfg = cfg
        validate_absolute_mouse_config(cfg)
        self.cmd_action = "none"
        self.cmd_up_down = "none"
        self.cmd_left_right = "none"
        self.cmd_up_down_last = ""
        self.cmd_left_right_last = ""
        self._last_source_action = "none"
        self.command_lock = threading.RLock()
        self.cached_facing = None
        # Monotonic timestamp for the currently held horizontal direction.
        # Live minimap motion consumes this only as observable state; launch
        # commands themselves are now frame-driven and immediate.
        self.direction_held_since = None
        self.direction_held_generation = None
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
        self.buff_recovery_until = 0.0
        if scheduled_buff_allowed is not None and not callable(
                scheduled_buff_allowed):
            raise TypeError("scheduled_buff_allowed must be callable")
        self.scheduled_buff_allowed = scheduled_buff_allowed
        # Flags
        self.is_enable = bool(connect_input)
        self.capture_available = bool(capture_available)
        self.minimap_available = True
        self.session_recovery_active = False
        self.game_ui_active = False
        self.is_need_force_heal = False
        self.is_terminated = False
        # Parameters
        self.debounce_interval = self.cfg["system"]["key_debounce_interval"]
        self.fps_limit = self.cfg["system"]["fps_limit_keyboard_controller"]
        directional_cfg = self.cfg.get("directional_attack", {})
        self.character_turn_delay = max(
            0.0, float(directional_cfg.get("character_turn_delay", 0.08))
        )
        directional_aoe_cfg = self.cfg.get("directional_aoe", {})
        self.directional_aoe_key = self.cfg.get("key", {}).get(
            "aoe_skill", ""
        )
        power_knockback_cfg = self.cfg.get("power_knockback", {})
        self.power_knockback_key = self.cfg.get("key", {}).get(
            "power_knockback", ""
        )
        self.jump_up_settle_delay = max(
            0.0,
            float(cfg.get("route", {}).get("jump_up_settle_delay", 0.15)),
        )
        self.jump_alignment_nudge_ms = max(
            1,
            int(cfg.get("route", {}).get("jump_alignment_nudge_ms", 30)),
        )
        self.rope_climb_runup_ms = max(
            0,
            int(cfg.get("route", {}).get("rope_climb_runup_ms", 180)),
        )
        self.rope_climb_align_nudge_ms = max(
            1,
            int(cfg.get("route", {}).get("rope_climb_align_nudge_ms", 30)),
        )
        self.portal_sweep_nudge_ms = max(
            1,
            int(cfg.get("route", {}).get("portal_sweep_nudge_ms", 30)),
        )

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

        if cfg["bot"]["attack"] == "directional" and \
                directional_aoe_cfg.get("enable", False) and \
                (
                    not isinstance(self.directional_aoe_key, str)
                    or not self.directional_aoe_key.strip()
                ):
            raise ValueError(
                "key.aoe_skill is required when directional_aoe is enabled"
            )
        if cfg["bot"]["attack"] == "directional" and \
                power_knockback_cfg.get("enable", False) and \
                (
                    not isinstance(self.power_knockback_key, str)
                    or not self.power_knockback_key.strip()
                ):
            raise ValueError(
                "key.power_knockback is required when power_knockback is enabled"
            )

        validate_config_keys(cfg)
        if cfg["bot"]["attack"] == "directional" and \
                directional_aoe_cfg.get("enable", False) and \
                usage_from_text(self.directional_aoe_key) == \
                usage_from_text(self.attack_key):
            raise ValueError(
                "key.aoe_skill must differ from key.directional_attack when "
                "directional_aoe is enabled"
            )
        if cfg["bot"]["attack"] == "directional" and \
                power_knockback_cfg.get("enable", False) and \
                usage_from_text(self.power_knockback_key) == \
                usage_from_text(self.attack_key):
            raise ValueError(
                "key.power_knockback must differ from "
                "key.directional_attack when power_knockback is enabled"
            )
        if cfg["bot"]["attack"] == "directional" and \
                power_knockback_cfg.get("enable", False) and \
                directional_aoe_cfg.get("enable", False) and \
                usage_from_text(self.power_knockback_key) == \
                usage_from_text(self.directional_aoe_key):
            raise ValueError(
                "key.power_knockback must differ from key.aoe_skill when "
                "power_knockback and directional_aoe are enabled"
            )
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
        with _input_transaction_lock:
            if self._automation_input_active():
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
        with _input_transaction_lock:
            if self._automation_input_active():
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
        else:
            with _input_transaction_lock:
                if self._automation_input_active():
                    _input_allowed.set()
        return True

    def set_minimap_available(self, available):
        """Gate normal automation while the gameplay minimap is absent."""
        available = bool(available)
        if available == self.minimap_available:
            return False
        self.minimap_available = available
        if not available:
            _input_allowed.clear()
            self.release_all_key()
        else:
            with _input_transaction_lock:
                if self._automation_input_active():
                    _input_allowed.set()
        return True

    def _automation_input_active(self):
        """Return whether the normal movement/action worker may emit HID."""
        return bool(
            getattr(self, "is_enable", False)
            and getattr(self, "capture_available", True)
            and getattr(self, "minimap_available", True)
            and not getattr(self, "session_recovery_active", False)
            and not getattr(self, "game_ui_active", False)
            and not getattr(self, "is_terminated", False)
            and self.is_game_window_active()
        )

    def suspend_automation_for_game_ui(self):
        """Pause ordinary producers while allowing an explicit game-UI flow."""
        if getattr(self, "session_recovery_active", False) or \
                not getattr(self, "is_enable", False) or \
                not getattr(self, "capture_available", True) or \
                getattr(self, "is_terminated", False) or \
                not self.is_game_window_active():
            return False
        self.game_ui_active = True
        _input_allowed.clear()
        with _input_transaction_lock:
            self.game_ui_active = True
            _input_allowed.clear()
            with self._ensure_command_lock():
                self.cmd_left_right = "none"
                self.cmd_up_down = "none"
                self.cmd_action = "none"
                self._last_source_action = "none"
                self.is_need_force_heal = False
            success, _ = _invoke_input("release_all")
            self._invalidate_facing_cache()
            if not success:
                # The ordinary gate was already closed before RELEASE_ALL.
                # Keep automation visibly paused, but do not leave a hidden
                # game_ui_active latch that no caller believes it owns.
                self.is_enable = False
                self.game_ui_active = False
                _input_allowed.clear()
            return success

    def press_game_ui_key(self, key, duration=0.05):
        """Send one key while the exclusive game-UI input gate is active."""
        key = key.strip() if isinstance(key, str) else ""
        if not key:
            return False
        with _input_transaction_lock:
            if not getattr(self, "game_ui_active", False) or \
                    not getattr(self, "is_enable", False) or \
                    not getattr(self, "capture_available", True) or \
                    getattr(self, "is_terminated", False) or \
                    not self.is_game_window_active():
                return False
            duration_ms = max(1, int(round(float(duration) * 1000)))
            success, result = _invoke_input_confirmed(
                "tap", key, duration_ms
            )
            return result if success else False

    def click_game_ui_point(
        self,
        x,
        y,
        frame_width,
        frame_height,
        button="left",
        duration=0.05,
    ):
        """Click a capture-frame point during an exclusive game-UI flow."""
        try:
            if self.cfg.get("esp32_hid", {}).get(
                    "remote_target", False) and not \
                    has_calibrated_absolute_mouse(self.cfg):
                return False
            absolute_x, absolute_y = capture_point_to_absolute_hid(
                self.cfg, x, y, frame_width, frame_height
            )
            duration_ms = max(1, int(round(float(duration) * 1000)))
        except (TypeError, ValueError, OverflowError):
            return False
        with _input_transaction_lock:
            if not getattr(self, "game_ui_active", False) or \
                    not getattr(self, "is_enable", False) or \
                    not getattr(self, "capture_available", True) or \
                    getattr(self, "is_terminated", False) or \
                    not self.is_game_window_active():
                return False
            success, result = _invoke_input_confirmed(
                "mouse_click_at",
                absolute_x,
                absolute_y,
                button,
                duration_ms,
            )
            return result if success else False

    def resume_automation_after_game_ui(self):
        """Close the exclusive UI gate and restore ordinary input if safe."""
        with _input_transaction_lock:
            was_active = getattr(self, "game_ui_active", False)
            self.game_ui_active = False
            if self._automation_input_active():
                _input_allowed.set()
            else:
                _input_allowed.clear()
            return was_active

    def suspend_automation_for_session_recovery(self):
        """Stop every automatic producer while retaining an explicit TAP path.

        The global regular-input gate must stay closed throughout recovery so
        the worker cannot replay a stale movement snapshot, forced heal, or
        scheduled buff after RELEASE_ALL. Only the explicit recovery methods
        ``press_session_recovery_key``, ``move_session_recovery_mouse``, and
        the two explicit recovery click methods may emit HID while this flag
        is active.
        """
        # Signal in-flight multi-step actions to abort at their next regular
        # permission check, then serialize the final state clear and release.
        self.session_recovery_active = True
        _input_allowed.clear()
        with _input_transaction_lock:
            self.session_recovery_active = True
            _input_allowed.clear()
            with self._ensure_command_lock():
                self.cmd_left_right = "none"
                self.cmd_up_down = "none"
                self.cmd_action = "none"
                self._last_source_action = "none"
                self.is_need_force_heal = False
            success, _ = _invoke_input("release_all")
            self._invalidate_facing_cache()
            return success

    def press_session_recovery_key(self, key, duration=0.05):
        """Send one explicit recovery TAP while regular automation is gated."""
        key = key.strip() if isinstance(key, str) else ""
        if not key:
            return False
        with _input_transaction_lock:
            if not getattr(self, "session_recovery_active", False) or \
                    not getattr(self, "is_enable", False) or \
                    not getattr(self, "capture_available", True) or \
                    getattr(self, "is_terminated", False) or \
                    not self.is_game_window_active():
                return False
            duration_ms = max(1, int(round(float(duration) * 1000)))
            success, result = _invoke_input("tap", key, duration_ms)
            return result if success else False

    def focus_next_window_and_press_session_recovery_key(
        self,
        key,
        *,
        focus_keys=("alt", "tab"),
        focus_hold=0.10,
        settle_delay=0.50,
        duration=0.10,
    ):
        """Switch the remote foreground window, then send one recovery key.

        The focus chord is installed as one held HID state and released before
        the final key is sent. A failed focus transition never advances to the
        final key, and the cleanup path prevents a stuck modifier.
        """
        key = key.strip() if isinstance(key, str) else ""
        if not key or not isinstance(focus_keys, (list, tuple)) or not \
                focus_keys:
            return False
        try:
            focus_keys = tuple(
                item.strip() if isinstance(item, str) else ""
                for item in focus_keys
            )
            usages = tuple(usage_from_text(item) for item in focus_keys)
            usage_from_text(key)
            focus_hold = float(focus_hold)
            settle_delay = float(settle_delay)
            duration = float(duration)
        except (TypeError, ValueError, OverflowError):
            return False
        if any(not item for item in focus_keys) or \
                len(set(usages)) != len(usages) or \
                sum(usage < 0xE0 for usage in usages) > 6 or \
                not all(math.isfinite(value) for value in (
                    focus_hold, settle_delay, duration
                )) or not 0.001 <= focus_hold <= 1.0 or \
                not 0.0 <= settle_delay <= 5.0 or \
                not 0.001 <= duration <= 1.0:
            return False

        with _input_transaction_lock:
            if not getattr(self, "session_recovery_active", False) or \
                    not getattr(self, "is_enable", False) or \
                    not getattr(self, "capture_available", True) or \
                    getattr(self, "is_terminated", False) or \
                    not self.is_game_window_active():
                return False

            logger.info(
                "[KeyBoardController] Session recovery focus switch: "
                f"keys={list(focus_keys)}, then key={key!r}"
            )
            focus_state_may_be_held = False
            try:
                focus_state_may_be_held = True
                focus_sent, _ = _invoke_input_confirmed(
                    "set_state", focus_keys
                )
                if not focus_sent:
                    return False
                time.sleep(focus_hold)

                released, _ = _invoke_input_confirmed("release_all")
                if not released:
                    return False
                focus_state_may_be_held = False

                if settle_delay:
                    time.sleep(settle_delay)
                duration_ms = max(1, int(round(duration * 1000)))
                success, result = _invoke_input("tap", key, duration_ms)
                return result if success else False
            finally:
                if focus_state_may_be_held:
                    _invoke_input("release_all")

    def move_session_recovery_mouse(self, dx, dy):
        """Move the remote pointer once while regular automation is gated.

        Relative movement is deliberately issued as a one-shot command.  If
        its serial acknowledgement is lost, ``_invoke_input`` treats the
        command as consumed; the visual recovery loop observes the next frame
        instead of replaying a movement that may already have happened.
        """
        if isinstance(dx, bool) or not isinstance(dx, int) or \
                isinstance(dy, bool) or not isinstance(dy, int):
            return False
        with _input_transaction_lock:
            if not getattr(self, "session_recovery_active", False) or \
                    not getattr(self, "is_enable", False) or \
                    not getattr(self, "capture_available", True) or \
                    getattr(self, "is_terminated", False) or \
                    not self.is_game_window_active():
                return False
            success, result = _invoke_input("mouse_move", dx, dy, 0)
            return result if success else False

    def click_session_recovery_mouse(
        self,
        button="left",
        duration=0.05,
    ):
        """Click once at the current remote pointer position."""
        try:
            duration_ms = max(1, int(round(float(duration) * 1000)))
        except (TypeError, ValueError, OverflowError):
            return False
        with _input_transaction_lock:
            if not getattr(self, "session_recovery_active", False) or \
                    not getattr(self, "is_enable", False) or \
                    not getattr(self, "capture_available", True) or \
                    getattr(self, "is_terminated", False) or \
                    not self.is_game_window_active():
                return False
            success, result = _invoke_input(
                "mouse_click", button, duration_ms
            )
            return result if success else False

    def click_session_recovery_point(
        self,
        x,
        y,
        frame_width,
        frame_height,
        button="left",
        duration=0.05,
    ):
        """Click one capture-frame point through the gated absolute HID path."""
        try:
            if self.cfg.get("esp32_hid", {}).get(
                    "remote_target", False) and not \
                    has_calibrated_absolute_mouse(self.cfg):
                return False
            absolute_x, absolute_y = capture_point_to_absolute_hid(
                self.cfg, x, y, frame_width, frame_height
            )
            duration_ms = max(1, int(round(float(duration) * 1000)))
        except (TypeError, ValueError, OverflowError):
            return False
        with _input_transaction_lock:
            if not getattr(self, "session_recovery_active", False) or \
                    not getattr(self, "is_enable", False) or \
                    not getattr(self, "capture_available", True) or \
                    getattr(self, "is_terminated", False) or \
                    not self.is_game_window_active():
                return False
            logger.info(
                "[KeyBoardController] Session recovery absolute click: "
                f"capture=({int(round(float(x)))}, "
                f"{int(round(float(y)))})/{frame_width}x{frame_height}, "
                f"hid=({absolute_x}, {absolute_y}), "
                "capture_frame_is_desktop="
                f"{self.cfg.get('esp32_hid', {}).get('capture_frame_is_desktop')}, "
                "source_rect="
                f"{self.cfg.get('esp32_hid', {}).get('magpie_source_rect')}"
            )
            success, result = _invoke_input(
                "mouse_click_at",
                absolute_x,
                absolute_y,
                button,
                duration_ms,
            )
            return result if success else False

    def resume_automation_after_session_recovery(self):
        """Reopen regular HID only if the user's other safety gates allow it."""
        with _input_transaction_lock:
            was_active = getattr(self, "session_recovery_active", False)
            self.session_recovery_active = False
            if self._automation_input_active():
                _input_allowed.set()
            else:
                _input_allowed.clear()
            return was_active

    def set_command(self, new_command):
        '''
        Set keyboard command
        '''
        cmd_left_right, cmd_up_down, cmd_action = new_command.split()
        lock = self._ensure_command_lock()
        with lock:
            # Direction, vertical movement, and action form one immutable
            # snapshot for the worker. Never pair a new attack with the prior
            # frame's direction while an ESP32 request is in flight.
            self.cmd_left_right = cmd_left_right
            self.cmd_up_down = cmd_up_down
            # Route actions are edge-triggered so a visible route pixel cannot
            # generate 30 TAPs/s. Combat actions are already rate-limited by
            # the engine cooldown/recovery gates, so allow a newly consumed
            # attack to be queued again even if its action name is unchanged.
            retriggerable_action = (
                cmd_action in {
                    "attack",
                    "directional_aoe",
                    "power_knockback",
                    "jump_align_left",
                    "jump_align_right",
                    "rope_align_left",
                    "rope_align_right",
                    "portal_sweep_left",
                    "portal_sweep_right",
                }
                and self.cmd_action == "none"
            )
            if cmd_action != self._last_source_action or retriggerable_action:
                self.cmd_action = cmd_action
            elif cmd_action == "none":
                self.cmd_action = "none"
            self._last_source_action = cmd_action

    def _ensure_command_lock(self):
        """Keep lightweight __new__-based tests backward compatible."""
        lock = getattr(self, "command_lock", None)
        if lock is None:
            lock = threading.RLock()
            self.command_lock = lock
        return lock

    def _command_snapshot(self):
        with self._ensure_command_lock():
            return (
                self.cmd_left_right,
                self.cmd_up_down,
                self.cmd_action,
            )

    def _consume_action(self, action):
        with self._ensure_command_lock():
            if self.cmd_action == action:
                self.cmd_action = "none"

    def _buff_action_cooldown(self):
        try:
            value = float(self.cfg.get("buff_skill", {}).get(
                "action_cooldown", 1.0
            ))
        except (TypeError, ValueError):
            value = 1.0
        return max(0.0, value)

    def _buff_recovery_active(self, now=None):
        """Return whether a completed Buff still owns the action window."""
        now = time.monotonic() if now is None else float(now)
        return now < float(getattr(self, "buff_recovery_until", 0.0))

    def is_buff_recovery_active(self):
        """Expose the Buff animation gate to the vision/navigation loop."""
        return self._buff_recovery_active()

    def _scheduled_buff_is_safe(self, command=None):
        """Let ready Buffs preempt ordinary input, but never a Jump."""
        if command is None:
            command = self._command_snapshot()
        _, _, cmd_action = command
        jump_transaction = (
            cmd_action == "jump"
            or str(cmd_action).startswith("jump_")
            # Rope mounting includes the Jump TAP used to catch the rope.
            or str(cmd_action).startswith("rope_mount_")
        )
        if jump_transaction:
            return False
        allowed = getattr(self, "scheduled_buff_allowed", None)
        return allowed is None or bool(allowed())

    def _try_cast_ready_buff(self):
        """Stop movement, cast one ready Buff, then reserve its recovery."""
        if getattr(self, "is_need_force_heal", False) or \
                not self._scheduled_buff_is_safe():
            return False

        with _input_transaction_lock:
            if not _regular_input_allowed() or \
                    not self._scheduled_buff_is_safe():
                return False

            now = time.time()
            action_cooldown = self._buff_action_cooldown()
            for i, buff_skill_key in enumerate(
                    self.cfg.get("buff_skill", {}).get("keys", ())):
                if not buff_skill_key:
                    continue
                cooldown = self.cfg["buff_skill"]["cooldown"][i]
                if now - self.t_last_buff_cast[i] < cooldown:
                    continue

                # A Buff is an exclusive action, not a TAP inserted between
                # a stale movement snapshot and a newly queued route jump.
                success, _ = _invoke_input("set_state", [])
                if not success:
                    self._invalidate_facing_cache()
                    return False
                with self._ensure_command_lock():
                    self.cmd_left_right_last = "none"
                    self.cmd_up_down_last = "none"
                    self.direction_held_since = None
                    self.direction_held_generation = None

                if not _regular_input_allowed():
                    self._invalidate_facing_cache()
                    return False
                success, _ = _invoke_input(
                    "tap", str(buff_skill_key).strip(), 50
                )
                if not success:
                    return False

                cast_at = time.time()
                self.t_last_buff_cast[i] = cast_at
                self.t_last_skill = cast_at
                self.buff_recovery_until = (
                    time.monotonic() + action_cooldown
                )
                logger.info(
                    f"[Buff] Press buff skill key: '{buff_skill_key}' "
                    f"(cooldown: {cooldown}s); hold navigation for "
                    f"{action_cooldown:.2f}s"
                )
                return True
        return False

    def is_game_window_active(self):
        '''
        Check if the game window is currently the active (foreground) window.

        Returns:
        - True
        - False
        '''
        # In capture-card mode this process sees DirectShow on computer A while
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
        self._invalidate_facing_cache()

    def _invalidate_facing_cache(self):
        """Forget direction after a safety event or uncertain transport state."""
        with self._ensure_command_lock():
            # The game keeps its facing after an ordinary direction release,
            # but a safety release/pause permits external input, so the cached
            # facing can no longer be trusted.
            self.cached_facing = None
            self.cmd_left_right_last = ""
            self.cmd_up_down_last = ""
            self.direction_held_since = None
            self.direction_held_generation = None

    def _record_horizontal_motion(
            self, direction, *, now=None, generation=None):
        """Record uninterrupted same-direction HID movement."""
        now = time.monotonic() if now is None else float(now)
        if generation is None:
            generation = _input_state_generation()
        with self._ensure_command_lock():
            previous_direction = getattr(self, "cmd_left_right_last", "")
            held_since = getattr(self, "direction_held_since", None)
            held_generation = getattr(
                self, "direction_held_generation", None
            )
            if direction in {"left", "right"}:
                if previous_direction != direction or held_since is None \
                        or held_since > now \
                        or held_generation != generation:
                    held_since = now
                self.direction_held_since = held_since
                self.direction_held_generation = generation
                self.cached_facing = direction
            else:
                held_since = None
                self.direction_held_since = None
                self.direction_held_generation = None
            self.cmd_left_right_last = direction
            return held_since

    def same_direction_move_seconds(self, direction=None, *, now=None):
        """Return Hero's uninterrupted current horizontal-input duration."""
        now = time.monotonic() if now is None else float(now)
        generation = _input_state_generation()
        with self._ensure_command_lock():
            current_direction = getattr(self, "cmd_left_right_last", "")
            held_since = getattr(self, "direction_held_since", None)
            held_generation = getattr(
                self, "direction_held_generation", None
            )
        if direction is not None and direction != current_direction:
            return 0.0
        if current_direction not in {"left", "right"} or \
                held_since is None or held_since > now or \
                held_generation != generation:
            return 0.0
        return now - float(held_since)

    def update_movement_state(self, cmd_left_right=None, cmd_up_down=None):
        """Send both movement axes as one deduplicated HID report."""
        if cmd_left_right is None or cmd_up_down is None:
            # Read the two movement axes atomically without requiring action
            # state; callers that only drive movement may not initialize it.
            with self._ensure_command_lock():
                if cmd_left_right is None:
                    cmd_left_right = self.cmd_left_right
                if cmd_up_down is None:
                    cmd_up_down = self.cmd_up_down

        keys = []
        if cmd_left_right in {"left", "right"}:
            keys.append(cmd_left_right)
        elif cmd_left_right not in {"stop", "none"}:
            logger.error(
                "[KeyBoardController] Unsupported left-right command: "
                f"{cmd_left_right}"
            )

        if cmd_up_down in {"up", "down"}:
            keys.append(cmd_up_down)
        elif cmd_up_down not in {"stop", "none"}:
            logger.error(
                "[KeyBoardController] Unsupported up-down command: "
                f"{cmd_up_down}"
            )

        if not set_key_state(keys):
            self._invalidate_facing_cache()
            return False
        now = time.monotonic()
        generation = _input_state_generation()
        self._record_horizontal_motion(
            cmd_left_right, now=now, generation=generation
        )
        with self._ensure_command_lock():
            self.cmd_up_down_last = cmd_up_down
        return True

    def perform_directional_attack(self, direction, attack_key=None):
        """Turn, release movement, and attack as one indivisible HID action."""
        if direction not in {"left", "right"}:
            return False
        if attack_key is None:
            attack_key = self.attack_key
        if not isinstance(attack_key, str) or not attack_key.strip():
            return False
        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            with self._ensure_command_lock():
                needs_turn = self.cached_facing != direction

            if needs_turn:
                success, _ = _invoke_input("set_state", [direction])
                if not success:
                    self._invalidate_facing_cache()
                    return False
                self._record_horizontal_motion(direction)
                with self._ensure_command_lock():
                    self.cmd_up_down_last = "none"

                if self.character_turn_delay > 0:
                    time.sleep(self.character_turn_delay)
                # Pause/capture loss can occur while the direction report is
                # waiting for its serial acknowledgement or turn delay.
                if not _input_allowed.is_set():
                    self._invalidate_facing_cache()
                    return False

            # Stop movement before firing. Releasing a direction does not
            # change the cached in-game facing.
            success, _ = _invoke_input("set_state", [])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "none"
                self.direction_held_since = None
                self.direction_held_generation = None

            # The state request above can block on serial I/O. Recheck after
            # it returns so an F1 pause or capture-loss event cannot be
            # followed by a late attack TAP, including the same-facing path.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            duration_ms = 50
            success, _ = _invoke_input(
                "tap", attack_key.strip(), duration_ms
            )
            if not success:
                self._invalidate_facing_cache()
                return False

            self.t_last_skill = time.time()
            return True

    @staticmethod
    def is_stationary_jump_command(cmd_left_right, cmd_up_down, cmd_action):
        """Return whether a route command requests a straight-up jump."""
        return (
            cmd_action == "jump"
            and cmd_left_right in {"none", "stop"}
            and cmd_up_down in {"none", "stop"}
        )

    @staticmethod
    def is_directional_jump_command(cmd_left_right, cmd_up_down, cmd_action):
        """Return whether a route command requests a running jump."""
        return (
            cmd_action == "jump"
            and cmd_left_right in {"left", "right"}
            and cmd_up_down in {"none", "stop"}
        )

    def perform_directional_jump(self, direction):
        """Hold the requested direction and jump immediately."""
        if direction not in {"left", "right"}:
            return False

        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            # Keep horizontal input held through and after the immediate TAP.
            success, _ = _invoke_input("set_state", [direction])
            if not success:
                self._invalidate_facing_cache()
                return False

            now = time.monotonic()
            self._record_horizontal_motion(
                direction,
                now=now,
                generation=_input_state_generation(),
            )
            same_direction_seconds = self.same_direction_move_seconds(
                direction, now=now
            )
            with self._ensure_command_lock():
                self.cmd_up_down_last = "none"
            logger.debug(
                "[directional-jump] Immediate TAP after "
                f"{same_direction_seconds:.3f}s moving {direction}"
            )

            # STATE can block on the ESP32 acknowledgement. Never send Jump
            # after input was suspended while that request was in flight.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "tap", self.cfg["key"]["jump"], 50
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            return True

    def perform_stationary_jump(self):
        """Release movement, wait for inertia to settle, then jump atomically."""
        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            # An empty state releases horizontal and vertical movement. A
            # false client result is an acknowledged deduplicated no-op, so
            # _invoke_input's success flag is what matters here.
            success, _ = _invoke_input("set_state", [])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "none"
                self.direction_held_since = None
                self.direction_held_generation = None

            if self.jump_up_settle_delay > 0:
                time.sleep(self.jump_up_settle_delay)

            # F1 pause or capture loss may occur while waiting to stop. Never
            # send a delayed jump after input has been suspended.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "tap", self.cfg["key"]["jump"], 50
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            return True

    def perform_jump_alignment_nudge(self, direction):
        """Release held movement and apply one short centering pulse."""
        if direction not in {"left", "right"}:
            return False

        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            success, _ = _invoke_input("set_state", [])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "none"
                self.direction_held_since = None
                self.direction_held_generation = None

            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "tap", direction, self.jump_alignment_nudge_ms
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cached_facing = direction
            return True

    def perform_rope_alignment_nudge(self, direction):
        """Release held movement and apply one atomic rope-centering pulse."""
        if direction not in {"left", "right"}:
            return False

        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            success, _ = _invoke_input("set_state", [])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "none"
                self.direction_held_since = None
                self.direction_held_generation = None

            # Pause/capture loss may arrive while STATE is in flight. Keep the
            # centering TAP in this transaction, but never send it afterwards.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "tap", direction, self.rope_climb_align_nudge_ms
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cached_facing = direction
            return True

    def perform_rope_mount(self, direction):
        """Keep/build a sideways run, add Up, and jump onto the rope."""
        if direction not in {"left", "right"}:
            return False

        with _input_transaction_lock:
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input("set_state", [direction])
            if not success:
                self._invalidate_facing_cache()
                return False
            now = time.monotonic()
            self._record_horizontal_motion(
                direction,
                now=now,
                generation=_input_state_generation(),
            )
            with self._ensure_command_lock():
                self.cmd_up_down_last = "none"

            # The 60 Hz vision frame has already confirmed the speed-adjusted
            # spatial point. Never add a host-side run-up wait here.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "set_state", [direction, "up"]
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = direction
                self.cmd_up_down_last = "up"

            # STATE may block on the ESP32 acknowledgement. Recheck so a pause
            # during that request cannot be followed by a late jump.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "tap", self.cfg["key"]["jump"], 50
            )
            if not success:
                self._invalidate_facing_cache()
                return False

            # The lateral key is only for mounting momentum. Release it as
            # soon as the jump has been issued so the character does not pass
            # through the rope; Up remains held for the climb.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False
            success, _ = _invoke_input("set_state", ["up"])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "up"
                self.direction_held_since = None
                self.direction_held_generation = None
            return True

    def perform_stationary_rope_mount(self):
        """Mount a rope from an already verified exact horizontal alignment."""
        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            success, _ = _invoke_input("set_state", [])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "none"
                self.direction_held_since = None
                self.direction_held_generation = None

            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False
            success, _ = _invoke_input("set_state", ["up"])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_up_down_last = "up"

            # STATE acknowledgement provides ordering; no fixed Up lead is
            # added after the exact minimap alignment was observed.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "tap", self.cfg["key"]["jump"], 50
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            return True

    def perform_aligned_jump(self):
        """Jump immediately after the vision loop verified a stable center."""
        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            success, _ = _invoke_input("set_state", [])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "none"
                self.direction_held_since = None
                self.direction_held_generation = None

            # There is deliberately no fixed sleep here. The vision loop has
            # already observed the exact route center continuously for the
            # configured settle interval; sleeping now would reintroduce the
            # stale-coordinate jump this action is designed to avoid.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False
            success, _ = _invoke_input(
                "tap", self.cfg["key"]["jump"], 50
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            return True

    def perform_portal_sweep_step(self, direction):
        """Hold Up and apply one short horizontal portal-search pulse."""
        if direction not in {"left", "right"}:
            return False

        with _input_transaction_lock:
            if not _regular_input_allowed():
                return False

            # Keep Up persistent while the horizontal direction is only a
            # short TAP. Both reports stay in one transaction so no movement,
            # attack, or safety release can be interleaved between them.
            success, _ = _invoke_input("set_state", ["up"])
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cmd_left_right_last = "none"
                self.cmd_up_down_last = "up"
                self.direction_held_since = None
                self.direction_held_generation = None

            # Pause/capture loss can occur while waiting for the STATE ACK.
            # Never send a late horizontal pulse after input was suspended.
            if not _regular_input_allowed():
                self._invalidate_facing_cache()
                return False

            success, _ = _invoke_input(
                "tap", direction, self.portal_sweep_nudge_ms
            )
            if not success:
                self._invalidate_facing_cache()
                return False
            with self._ensure_command_lock():
                self.cached_facing = direction
            return True

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
                with _input_transaction_lock:
                    input_active = self._automation_input_active()
                    if input_active:
                        _input_allowed.set()
                    else:
                        _input_allowed.clear()
                if not input_active:
                    if was_input_active:
                        self.release_all_key()
                    was_input_active = False
                    self.limit_fps()
                    continue
                was_input_active = True

                cmd_left_right, cmd_up_down, cmd_action = \
                    self._command_snapshot()

                # Forced healing takes precedence over the current command.
                if self.is_need_force_heal:
                    cmd_action = "add_hp"
                    with self._ensure_command_lock():
                        self.cmd_action = cmd_action

                # Buff animations can reject Jump/Up even after their HID TAP
                # has completed. Keep the latest route command queued, release
                # movement, and execute it only after the configured recovery.
                if not self.is_need_force_heal and \
                        self._buff_recovery_active():
                    self.update_movement_state("none", "none")
                    self.limit_fps()
                    continue

                # A due Buff outranks walking, attacking, and an established
                # rope climb. _scheduled_buff_is_safe vetoes current or timed
                # Jump transactions, so their launch timing remains atomic.
                if self._try_cast_ready_buff():
                    self.limit_fps()
                    continue

                # With no due Buff, preserve persistent Up while climbing.
                if cmd_action == "rope_hold":
                    self.update_movement_state(
                        cmd_left_right, cmd_up_down
                    )
                    self.limit_fps()
                    continue

                # If no Buff or Jump owns the frame, keep direction and attack
                # paired until the HID TAP has been sent.
                if cmd_action in {
                        "attack", "directional_aoe", "power_knockback"} and \
                        self.cfg["bot"]["attack"] == "directional":
                    attack_key = None
                    if cmd_action == "directional_aoe":
                        attack_key = getattr(
                            self,
                            "directional_aoe_key",
                            self.cfg.get("key", {}).get("aoe_skill", ""),
                        )
                    elif cmd_action == "power_knockback":
                        attack_key = getattr(
                            self,
                            "power_knockback_key",
                            self.cfg.get("key", {}).get(
                                "power_knockback", ""
                            ),
                        )
                    if self.perform_directional_attack(
                            cmd_left_right,
                            attack_key=attack_key,
                    ):
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                if cmd_action in {"jump_align_left", "jump_align_right"}:
                    direction = cmd_action.rsplit("_", 1)[-1]
                    if self.perform_jump_alignment_nudge(direction):
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                if cmd_action in {"rope_align_left", "rope_align_right"}:
                    direction = cmd_action.rsplit("_", 1)[-1]
                    if self.perform_rope_alignment_nudge(direction):
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                if cmd_action in {"rope_mount_left", "rope_mount_right"}:
                    direction = cmd_action.rsplit("_", 1)[-1]
                    if self.perform_rope_mount(direction):
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                if cmd_action == "rope_mount_stationary":
                    if self.perform_stationary_rope_mount():
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                if cmd_action in {"portal_sweep_left", "portal_sweep_right"}:
                    direction = cmd_action.rsplit("_", 1)[-1]
                    if self.perform_portal_sweep_step(direction):
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                if cmd_action == "jump_aligned":
                    if self.perform_aligned_jump():
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                # Orange/cyan route points request an immediate directional
                # jump. Continuous same-direction movement is tracked, but
                # never converted into a stale-coordinate run-up delay.
                if self.is_directional_jump_command(
                    cmd_left_right, cmd_up_down, cmd_action
                ):
                    if self.perform_directional_jump(cmd_left_right):
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                # Magenta route points request a vertical jump with no
                # directional input. Let horizontal inertia decay before the
                # jump TAP; down-jump commands keep their original behavior.
                if self.is_stationary_jump_command(
                    cmd_left_right, cmd_up_down, cmd_action
                ):
                    if self.perform_stationary_jump():
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                # Movement is one atomic STATE report. The client suppresses an
                # unchanged state, so the 30 FPS controller loop creates no
                # repeated serial traffic while a direction remains held.
                self.update_movement_state(cmd_left_right, cmd_up_down)

                ######################
                ### Action Command ###
                ######################
                if cmd_action == "jump":
                    if press_key(self.cfg["key"]["jump"]):
                        self._consume_action(cmd_action)
                elif cmd_action == "teleport":
                    if press_key(self.cfg["key"]["teleport"]):
                        self._consume_action(cmd_action)
                elif cmd_action == "attack":
                    if press_key(self.attack_key):
                        with self._ensure_command_lock():
                            self.direction_held_since = None
                            self.direction_held_generation = None
                        self.t_last_skill = time.time()
                        self._consume_action(cmd_action)
                elif cmd_action == "add_hp":
                    add_hp_key = self.cfg["key"].get("add_hp", "")
                    has_add_hp_key = (
                        isinstance(add_hp_key, str) and bool(add_hp_key.strip())
                    )
                    if not has_add_hp_key or press_key(add_hp_key):
                        self._consume_action(cmd_action)
                elif cmd_action == "add_mp":
                    add_mp_key = self.cfg["key"].get("add_mp", "")
                    has_add_mp_key = (
                        isinstance(add_mp_key, str) and bool(add_mp_key.strip())
                    )
                    if not has_add_mp_key or press_key(add_mp_key):
                        self._consume_action(cmd_action)
                elif cmd_action == "goal":
                    pass
                elif cmd_action == "none":
                    pass
                else:
                    logger.error("[KeyBoardController] Unsupported action command: "
                                 f"{cmd_action}")

                self.limit_fps()
        finally:
            _input_allowed.clear()
            self.release_all_key() # Prevent key keep press down after termination
            if self.input_client is not None:
                close_esp32_input(self.input_client)

        logger.info("[KeyBoardController] terminated")
