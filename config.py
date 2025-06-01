class Config:
    '''
    ====== Configuration Parameters ======
    Tune these values based on your gameplay and detection needs.
    --------------------------------------
    '''
    # ────────────────
    # Keyboard Mapping
    # ────────────────
    ATTACK_KEY = "c" # Assume is magic claw
    JUMP_KEY = "space"

    # ────────────────
    # Player Localization
    # ────────────────
    # Offset from the nametag's top-left corner to the player's center
    NAMETAG_OFFSET_X = -38 # pixel
    NAMETAG_OFFSET_Y = 40  # pixel
    NAMETAG_SIM_THRES = 0.5

    # ────────────────
    # Camera Localization
    # ────────────────
    # Only use this vertical range of the screen to localize camera on map
    CAMERA_CEILING = 60  # pixel (top)
    CAMERA_FLOOR = 620   # pixel (bottom)

    # ────────────────
    # Attack Settings
    # ────────────────
    # Magic Claw skill attack range relative to player position
    MAGIC_CLAW_RANGE_X = 350 # pixels (horizontal range)
    MAGIC_CLAW_RANGE_Y = 70  # pixels (vertical range)

    # ────────────────
    # Monster Detection
    # ────────────────
    # MONSTER_SIM_THRES = 0.7   # Template match similarity threshold, this was for TM_CCOEFF_NORMED
    MONSTER_DIF_THRES = 0.3   # Template match similarity threshold
    MONSTER_SEARCH_MARGIN = 50  # Extra margin around attack box for monster search

    # ────────────────
    # Route Detection (Color Code)
    # ────────────────
    COLOR_CODE_SEARCH_RANGE = 80 # Radius to find nearest color route from player center

    # ────────────────
    # Movement Behavior
    # ────────────────
    UP_DRAG_DURATION = 0.1 # Hold duration for 'up' key to prevent rope-sticking (in seconds)
    WATCH_DOG_TIMEOUT = 3 # seconds, if player doesn't move for 3 second, random perform an action
    WATCH_DOG_RANGE = 10 # pixel, if player location is smaller than WATCH_DOG_RANGE, consider it doesn't move

    # ────────────────
    # Debug Options
    # ────────────────
    ENABLE_DEBUG_WINDOWS = True   # Show debug window (False = better performance)

    # ────────────────
    # Please Remove Runes Warning
    # ────────────────
    PLEASE_REMOVE_RUNES_TOP_LEFT     = (513, 196)
    PLEASE_REMOVE_RUNES_BOTTOM_RIGHT = (768, 236)
    PLEASE_REMOVE_RUNES_SIM_THRES = 0.8

    # ────────────────
    # Rune Test
    # ────────────────
    RUNE_DETECT_BOX_WIDTH = 140
    RUNE_DETECT_BOX_HEIGHT = 140
    RUNE_DETECT_SIM_THRES = 0.7
    ARROW_BOX_SIZE = 80 # pixel, 75x75 box
    ARROW_BOX_INTERVAL = 170 # pixel width
    ARROW_BOX_START_POINT = (355, 355)
    ARROW_BOX_DIF_THRES = 0.2

    RUNE_FINDING_TIMEOUT = 600 # second
    NEAR_RUNE_DURATION = 5 # second
