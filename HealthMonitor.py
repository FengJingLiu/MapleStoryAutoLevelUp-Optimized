import threading
import time
from logger import logger

class HealthMonitor:
    '''
    Independent health monitoring thread that can heal while other actions are running
    '''
    def __init__(self, cfg, args, kb_controller):
        self.cfg = cfg
        self.args = args
        self.kb = kb_controller
        self.running = False
        self.enabled = True
        self.thread = None
        
        # Health monitoring state
        self.hp_ratio = 1.0
        self.mp_ratio = 1.0
        self.last_heal_time = 0
        self.last_mp_time = 0
        
        # Frame data (will be updated by main thread)
        self.img_frame = None
        self.frame_lock = threading.Lock()
        
    def start(self):
        '''
        Start health monitoring thread
        '''
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            logger.info("Health monitor started")
    
    def stop(self):
        '''
        Stop health monitoring thread
        '''
        self.running = False
        if self.thread:
            self.thread.join()
            logger.info("Health monitor stopped")
    
    def enable(self):
        '''
        Enable health monitoring
        '''
        self.enabled = True
        
    def disable(self):
        '''
        Disable health monitoring
        '''
        self.enabled = False
        
    def update_frame(self, img_frame):
        '''
        Update frame data from main thread
        '''
        with self.frame_lock:
            self.img_frame = img_frame
    
    def get_hp_mp_ratio(self):
        '''
        Extract HP and MP ratios from current frame
        '''
        if self.img_frame is None:
            return 1.0, 1.0
            
        with self.frame_lock:
            img_frame = self.img_frame.copy()
        
        # HP crop
        hp_bar = img_frame[self.cfg.hp_bar_top_left[1]:self.cfg.hp_bar_bottom_right[1]+1,
                          self.cfg.hp_bar_top_left[0]:self.cfg.hp_bar_bottom_right[0]+1]
        # MP crop
        mp_bar = img_frame[self.cfg.mp_bar_top_left[1]:self.cfg.mp_bar_bottom_right[1]+1,
                          self.cfg.mp_bar_top_left[0]:self.cfg.mp_bar_bottom_right[0]+1]
        
        # HP Detection (detect empty part)
        empty_mask_hp = (hp_bar[:,:,0] == hp_bar[:,:,1]) & (hp_bar[:,:,0] == hp_bar[:,:,2])
        empty_pixels_hp = max(0, len(empty_mask_hp[empty_mask_hp]) - 6)  # 6 pixel always be white
        total_pixels_hp = hp_bar.shape[0] * hp_bar.shape[1] - 6
        hp_ratio = 1 - (empty_pixels_hp / max(1, total_pixels_hp))
        
        # MP Detection (detect empty part)
        empty_mask_mp = (mp_bar[:,:,0] == mp_bar[:,:,1]) & (mp_bar[:,:,0] == mp_bar[:,:,2])
        empty_pixels_mp = max(0, len(empty_mask_mp[empty_mask_mp]) - 6)  # 6 pixel always be white
        total_pixels_mp = mp_bar.shape[0] * mp_bar.shape[1] - 6
        mp_ratio = 1 - (empty_pixels_mp / max(1, total_pixels_mp))
        
        return max(0, min(1, hp_ratio)), max(0, min(1, mp_ratio))
    
    def _monitor_loop(self):
        '''
        Main monitoring loop running in separate thread
        '''
        while self.running:
            try:
                if not self.enabled or self.args.disable_control:
                    time.sleep(0.1)
                    continue
                
                # Get current HP/MP ratios
                hp_ratio, mp_ratio = self.get_hp_mp_ratio()
                self.hp_ratio = hp_ratio
                self.mp_ratio = mp_ratio
                
                current_time = time.time()
                
                # Check if need to heal (with cooldown)
                if (hp_ratio <= self.cfg.heal_ratio and 
                    current_time - self.last_heal_time > self.cfg.heal_cooldown):
                    # self._heal()
                    self.last_heal_time = current_time
                    logger.info(f"Auto heal triggered, HP: {hp_ratio*100:.1f}%")
                
                # Check if need MP (with cooldown)
                if (mp_ratio <= self.cfg.add_mp_ratio and 
                    current_time - self.last_mp_time > self.cfg.mp_cooldown):
                    # self._add_mp()
                    self.last_mp_time = current_time
                    logger.info(f"Auto MP triggered, MP: {mp_ratio*100:.1f}%")
                
                # Sleep to avoid excessive CPU usage
                time.sleep(0.05)  # Check every 50ms
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(0.1)
    
    def _heal(self):
        '''
        Execute heal action
        '''
        try:
            self.kb.press_key(self.cfg.heal_key, 0.05)
        except Exception as e:
            logger.error(f"Heal action failed: {e}")
    
    def _add_mp(self):
        '''
        Execute MP recovery action
        '''
        try:
            self.kb.press_key(self.cfg.add_mp_key, 0.05)
        except Exception as e:
            logger.error(f"MP action failed: {e}")