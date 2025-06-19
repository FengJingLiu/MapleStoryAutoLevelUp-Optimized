'''
KeyBoardController
'''
# Standard Import
import threading
import time

import pyautogui
from pynput import keyboard

# Local import
from logger import logger
from util import is_mac

if is_mac():
    import Quartz
else:
    import pygetwindow as gw

pyautogui.PAUSE = 0  # remove delay

class KeyBoardController():
    '''
    KeyBoardController
    '''
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.command = ""
        self.window_title = cfg.game_window_title
        self.fps = 0 # Frame per seconds
        # Timer
        self.t_last_up = 0.0
        self.t_last_down = 0.0
        self.t_last_toggle = 0.0
        self.t_last_screenshot = 0.0
        self.t_last_run = time.time()
        self.t_last_action = 0.0 # Last time character perform action(attack, cast spell, ...)
        self.t_last_buff_cast = [0] * len(self.cfg.buff_skill_keys) # Last time cast buff skill
        # Flags
        self.is_enable = True
        self.is_need_screen_shot = False
        self.is_need_toggle = False
        # Parameters
        self.debounce_interval = 1 # second
        self.fps_limit = 30


        # use 'ctrl', 'alt' for mac, because it's hard to get around
        # macOS's security settings
        if is_mac():
            self.toggle_key = keyboard.Key.ctrl
            self.screenshot_key = keyboard.Key.alt
        else:
            self.toggle_key = keyboard.Key.f1
            self.screenshot_key = keyboard.Key.f2

        # set up attack key
        self.attack_key = ""
        if args.attack == "aoe_skill":
            self.attack_key = cfg.aoe_skill_key
        elif args.attack == "magic_claw":
            self.attack_key = cfg.magic_claw_key
        else:
            logger.error(f"Unexpected attack argument: {args.attack}")

        # Start keyboard control thread
        threading.Thread(target=self.run, daemon=True).start()

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        '''
        Handle key press events.
        '''
        try:
            # Handle regular character keys
            key.char
        except AttributeError:
            # Handle special keys
            if key == self.toggle_key:
                if time.time() - self.t_last_toggle > self.debounce_interval:
                    self.toggle_enable()
                    self.t_last_toggle = time.time()
            elif key == self.screenshot_key:
                if time.time() - self.t_last_screenshot > self.debounce_interval:
                    self.is_need_screen_shot = True
                    self.t_last_screenshot = time.time()

    def toggle_enable(self):
        '''
        toggle_enable
        '''
        self.is_enable = not self.is_enable
        logger.info(f"Player pressed F1, is_enable:{self.is_enable}")

        # Make sure all key are released
        self.release_all_key()

    def press_key(self, key, duration=0.05):
        '''
        Simulates a key press for a specified duration
        '''
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)

    def disable(self):
        '''
        disable keyboard controlller
        '''
        self.is_enable = False

    def enable(self):
        '''
        enable keyboard controlller
        '''
        self.is_enable = True

    def set_command(self, new_command):
        '''
        Set keyboard command
        '''
        self.command = new_command
        # logger.info(f"Set command to {new_command}")

    def is_game_window_active(self):
        '''
        Check if the game window is currently the active (foreground) window.

        Returns:
        - True
        - False
        '''
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
            active_window = gw.getActiveWindow()
            return active_window is not None and self.window_title in active_window.title

    def release_all_key(self):
        '''
        Release all key
        '''
        pyautogui.keyUp("left")
        pyautogui.keyUp("right")
        pyautogui.keyUp("up")
        pyautogui.keyUp("down")
        # Also release attack keys to stop any ongoing attacks
        pyautogui.keyUp(self.attack_key)


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
        while True:
            # Check if game window is active
            if not self.is_enable or not self.is_game_window_active():
                self.limit_fps()
                continue

            # Buff skill
            # if not self.is_in_buffer_skill_active_duration():
            for i, buff_skill_key in enumerate(self.cfg.buff_skill_keys):
                cooldown = self.cfg.buff_skill_cooldown[i]
                if time.time() - self.t_last_buff_cast[i] >= cooldown and \
                    time.time() - self.t_last_action > self.cfg.buff_skill_action_cooldown:
                    self.press_key(buff_skill_key)
                    logger.info(f"[Buff] Press buff skill key: '{buff_skill_key}' (cooldown: {cooldown}s)")
                    # Reset timers
                    self.t_last_buff_cast[i] = time.time()
                    self.t_last_action = time.time()
                    break

            # check if is needed to release 'Up' key
            if time.time() - self.t_last_up > self.cfg.up_drag_duration:
                pyautogui.keyUp("up")

            # check if is needed to release 'Down' key
            if time.time() - self.t_last_down > self.cfg.down_drag_duration:
                pyautogui.keyUp("down")

            if self.command == "walk left":
                pyautogui.keyUp("right")
                pyautogui.keyDown("left")

            elif self.command == "walk right":
                pyautogui.keyUp("left")
                pyautogui.keyDown("right")

            elif self.command == "jump left":
                pyautogui.keyUp("right")
                pyautogui.keyDown("left")
                self.press_key(self.cfg.jump_key)
                pyautogui.keyUp("left")

            elif self.command == "jump right":
                pyautogui.keyUp("left")
                pyautogui.keyDown("right")
                self.press_key(self.cfg.jump_key)
                pyautogui.keyUp("right")

            elif self.command == "jump down":
                pyautogui.keyUp("right")
                pyautogui.keyUp("left")
                pyautogui.keyDown("down")
                self.press_key(self.cfg.jump_key)
                pyautogui.keyUp("down")

            elif self.command == "jump":
                pyautogui.keyUp("left")
                pyautogui.keyUp("right")
                self.press_key(self.cfg.jump_key)

            elif self.command == "up":
                pyautogui.keyUp("down")
                pyautogui.keyDown("up")
                self.t_last_up = time.time()

            elif self.command == "down":
                pyautogui.keyUp("up")
                pyautogui.keyDown("down")
                self.t_last_down = time.time()

            if self.command == "teleport left":
                pyautogui.keyUp("right")
                pyautogui.keyDown("left")
                self.press_key(self.cfg.teleport_key)

            elif self.command == "teleport right":
                pyautogui.keyUp("left")
                pyautogui.keyDown("right")
                self.press_key(self.cfg.teleport_key)

            elif self.command == "teleport up":
                pyautogui.keyDown("up")
                self.press_key(self.cfg.teleport_key)
                pyautogui.keyUp("up")

            elif self.command == "teleport down":
                pyautogui.keyDown("down")
                self.press_key(self.cfg.teleport_key)
                pyautogui.keyUp("down")

            elif self.command == "attack":
                self.press_key(self.attack_key)
                self.t_last_action = time.time()

            elif self.command == "attack left":
                pyautogui.keyUp("right")
                pyautogui.keyDown("left")
                time.sleep(self.cfg.character_turn_delay)  # Small delay for character to turn
                self.press_key(self.attack_key)
                pyautogui.keyUp("left")
                self.t_last_action = time.time()

            elif self.command == "attack right":
                pyautogui.keyUp("left")
                pyautogui.keyDown("right")
                time.sleep(self.cfg.character_turn_delay)  # Small delay for character to turn
                self.press_key(self.attack_key)
                pyautogui.keyUp("right")
                self.t_last_action = time.time()

            elif self.command == "stop":
                self.release_all_key()
                self.command = ""  # Clear command after stopping

            elif self.command == "heal":
                self.press_key(self.cfg.heal_key)
                self.command = ""

            elif self.command == "add mp":
                self.press_key(self.cfg.add_mp_key)
                self.command = ""

            else:
                pass

            self.limit_fps()
