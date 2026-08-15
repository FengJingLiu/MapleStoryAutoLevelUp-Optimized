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
_input_transaction_lock = threading.RLock()
_input_recovery_lock = threading.Lock()
_input_recovery_until = 0.0


def input_recovery_remaining():
    """Return the remaining global HID lockout after an attack."""
    with _input_recovery_lock:
        return max(0.0, _input_recovery_until - time.monotonic())


def _set_input_recovery(duration, started_at=None):
    """Block every non-safety HID command until the attack recovery ends."""
    global _input_recovery_until
    duration = max(0.0, float(duration))
    started_at = time.monotonic() if started_at is None else float(started_at)
    with _input_recovery_lock:
        _input_recovery_until = max(
            _input_recovery_until,
            started_at + duration,
        )


def clear_input_recovery():
    """Clear the attack lockout when replacing or closing the controller."""
    global _input_recovery_until
    with _input_recovery_lock:
        _input_recovery_until = 0.0


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
        clear_input_recovery()
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
        clear_input_recovery()
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


def _call_input(method, *args):
    success, result = _invoke_input(method, *args)
    return result if success else False


def _regular_input_allowed():
    return _input_allowed.is_set() and input_recovery_remaining() <= 0.0


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
    def __init__(self, cfg, connect_input=True, capture_available=True):
        self.cfg = cfg
        self.cmd_action = "none"
        self.cmd_up_down = "none"
        self.cmd_left_right = "none"
        self.cmd_up_down_last = ""
        self.cmd_left_right_last = ""
        self._last_source_action = "none"
        self.command_lock = threading.RLock()
        self.cached_facing = None
        # Monotonic timestamp for the currently held horizontal direction.
        # Rope mounting uses this to preserve an existing same-direction run
        # instead of stopping and rebuilding momentum.
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
        # Flags
        self.is_enable = bool(connect_input)
        self.capture_available = bool(capture_available)
        self.is_need_force_heal = False
        self.is_terminated = False
        # Parameters
        self.debounce_interval = self.cfg["system"]["key_debounce_interval"]
        self.fps_limit = self.cfg["system"]["fps_limit_keyboard_controller"]
        directional_cfg = self.cfg.get("directional_attack", {})
        self.character_turn_delay = max(
            0.0, float(directional_cfg.get("character_turn_delay", 0.08))
        )
        self.attack_recovery_delay = max(
            0.0, float(directional_cfg.get("attack_recovery_delay", 0.90))
        )
        directional_aoe_cfg = self.cfg.get("directional_aoe", {})
        self.directional_aoe_key = self.cfg.get("key", {}).get(
            "aoe_skill", ""
        )
        self.directional_aoe_recovery_delay = max(
            0.0,
            float(directional_aoe_cfg.get("attack_recovery_delay", 0.90)),
        )
        power_knockback_cfg = self.cfg.get("power_knockback", {})
        self.power_knockback_key = self.cfg.get("key", {}).get(
            "power_knockback", ""
        )
        self.power_knockback_recovery_delay = max(
            0.0,
            float(power_knockback_cfg.get("attack_recovery_delay", 0.90)),
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

    def is_attack_recovering(self):
        """Return whether the global post-attack input gate is active."""
        return input_recovery_remaining() > 0.0

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
        with self._ensure_command_lock():
            previous_direction = getattr(self, "cmd_left_right_last", "")
            held_since = getattr(self, "direction_held_since", None)
            held_generation = getattr(
                self, "direction_held_generation", None
            )
            if cmd_left_right in {"left", "right"}:
                if previous_direction != cmd_left_right or held_since is None \
                        or held_since > now \
                        or held_generation != generation:
                    self.direction_held_since = now
                self.direction_held_generation = generation
                self.cached_facing = cmd_left_right
            else:
                self.direction_held_since = None
                self.direction_held_generation = None
            self.cmd_left_right_last = cmd_left_right
            self.cmd_up_down_last = cmd_up_down
        return True

    def perform_directional_attack(
            self, direction, attack_key=None, recovery_delay=None):
        """Turn, release movement, and attack as one indivisible HID action."""
        if direction not in {"left", "right"}:
            return False
        if attack_key is None:
            attack_key = self.attack_key
        if not isinstance(attack_key, str) or not attack_key.strip():
            return False
        if recovery_delay is None:
            recovery_delay = self.attack_recovery_delay
        recovery_delay = max(0.0, float(recovery_delay))

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
                with self._ensure_command_lock():
                    self.cached_facing = direction
                    self.cmd_left_right_last = direction
                    self.cmd_up_down_last = "none"

                if self.character_turn_delay > 0:
                    time.sleep(self.character_turn_delay)
                # Pause/capture loss can occur while the direction report is
                # waiting for its serial acknowledgement or turn delay.
                if not _input_allowed.is_set():
                    self._invalidate_facing_cache()
                    return False

            # Stop movement before firing and throughout the recovery. Releasing
            # a direction does not change the cached in-game facing.
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

            # Begin recovery after the TAP call returns. Besides guaranteeing
            # the full configured animation margin after the key release, this
            # is conservative when an ACK is delayed or the TAP result is
            # uncertain: no command can leak through immediately after timeout.
            _set_input_recovery(
                recovery_delay,
            )
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
            generation = _input_state_generation()
            with self._ensure_command_lock():
                previous_direction = getattr(
                    self, "cmd_left_right_last", ""
                )
                held_since = getattr(self, "direction_held_since", None)
                held_generation = getattr(
                    self, "direction_held_generation", None
                )
                if previous_direction != direction or held_since is None \
                        or held_since > now \
                        or held_generation != generation:
                    held_since = now
                self.direction_held_since = held_since
                self.direction_held_generation = generation
                self.cached_facing = direction
                self.cmd_left_right_last = direction
                self.cmd_up_down_last = "none"

            required_runup = max(
                0.0, float(self.rope_climb_runup_ms) / 1000.0
            )
            remaining_runup = max(
                0.0, required_runup - (now - held_since)
            )
            if remaining_runup > 0.0:
                time.sleep(remaining_runup)

            # The run-up wait is deliberately inside the HID transaction, but
            # pause/capture loss can still clear the permission gate.
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

                # No HID reports may be sent during the attack animation. Drop
                # no commands here: the main loop keeps replacing the command
                # snapshot, and the newest one is applied when recovery ends.
                if input_recovery_remaining() > 0.0:
                    self.limit_fps()
                    continue

                cmd_left_right, cmd_up_down, cmd_action = \
                    self._command_snapshot()

                # Forced healing takes precedence as soon as the attack
                # recovery ends. It is still blocked by the recovery gate
                # above, so no potion HID report can interrupt an animation.
                if self.is_need_force_heal:
                    cmd_action = "add_hp"
                    with self._ensure_command_lock():
                        self.cmd_action = cmd_action

                # Rope climbing owns the frame even though it has no TAP. This
                # prevents a buff from interrupting persistent direction/Up
                # while the engine waits for the character to finish climbing.
                if cmd_action == "rope_hold":
                    self.update_movement_state(
                        cmd_left_right, cmd_up_down
                    )
                    self.limit_fps()
                    continue

                # Direction and attack must remain paired. This branch runs
                # before buffs, movement, or healing and owns the HID output
                # transaction until TAP has been sent.
                if cmd_action in {
                        "attack", "directional_aoe", "power_knockback"} and \
                        self.cfg["bot"]["attack"] == "directional":
                    attack_key = None
                    recovery_delay = None
                    if cmd_action == "directional_aoe":
                        attack_key = getattr(
                            self,
                            "directional_aoe_key",
                            self.cfg.get("key", {}).get("aoe_skill", ""),
                        )
                        recovery_delay = getattr(
                            self,
                            "directional_aoe_recovery_delay",
                            self.attack_recovery_delay,
                        )
                    elif cmd_action == "power_knockback":
                        attack_key = getattr(
                            self,
                            "power_knockback_key",
                            self.cfg.get("key", {}).get(
                                "power_knockback", ""
                            ),
                        )
                        recovery_delay = getattr(
                            self,
                            "power_knockback_recovery_delay",
                            self.attack_recovery_delay,
                        )
                    if self.perform_directional_attack(
                            cmd_left_right,
                            attack_key=attack_key,
                            recovery_delay=recovery_delay,
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

                # Magenta route points request a vertical jump with no
                # directional input. Let horizontal inertia decay before the
                # jump TAP; directional and down-jump commands keep their
                # original immediate behavior.
                if self.is_stationary_jump_command(
                    cmd_left_right, cmd_up_down, cmd_action
                ):
                    if self.perform_stationary_jump():
                        self._consume_action(cmd_action)
                    self.limit_fps()
                    continue

                # Buff skill. Do not delay a pending forced heal.
                if not self.is_need_force_heal:
                    for i, buff_skill_key in enumerate(
                            self.cfg["buff_skill"]["keys"]):
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
