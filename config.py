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
    MONSTER_SIM_THRES = 0.7   # Template match similarity threshold
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
