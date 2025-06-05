class Config:
    '''
    ====== Configuration Parameters ======
    Tune these values based on your gameplay and detection needs.
    --------------------------------------
    '''
    # ────────────────
    # Keyboard Mapping
    # ────────────────
    attack_key = "w" # assume it is magic claw
    jump_key = "space"
    heal_key = "q"
    add_mp_key = "2"
    teleport_key = "e"

    # ────────────────
    # Player Localization
    # ────────────────
    # offset from the nametag's top-left corner to the player's center
    nametag_offset = (-38, 40) # pixel
    nametag_diff_thres = 1.0

    # ────────────────
    # Camera Localization
    # ────────────────
    # only use this vertical range of the screen to localize camera on map
    camera_ceiling = 60  # pixel (top)
    camera_floor = 665   # pixel (bottom)
    localize_diff_thres = 0.5
    localize_downscale_factor = 0.25 # ratio = 1/4

    # ────────────────
    # Attack Settings
    # ────────────────
    # magic claw skill attack range relative to player position
    magic_claw_range_x = 350 # pixels (horizontal range)
    magic_claw_range_y = 70  # pixels (vertical range)

    # ────────────────
    # Monster Detection
    # ────────────────
    monster_diff_thres = 0.8   # template match similarity threshold
    monster_search_margin = 50  # extra margin around attack box for monster search
    blur_range = 5

    # ────────────────
    # Route Detection (color code)
    # ────────────────
    color_code_search_range = 80 # radius to find nearest color route from player center

    # ────────────────
    # Movement Behavior
    # ────────────────
    up_drag_duration = 0.1 # hold duration for 'up' key to prevent rope-sticking (in seconds)
    watch_dog_timeout = 3 # seconds, if player doesn't move for 3 second, random perform an action
    watch_dog_range = 10 # pixel, if player location is smaller than watch_dog_range, consider it doesn't move

    # ────────────────
    # Debug Options
    # ────────────────
    enable_debug_windows = True   # show debug window (false = better performance)

    # ────────────────
    # Runes Warning
    # ────────────────
    rune_warning_top_left     = (513, 196)
    rune_warning_bottom_right = (768, 236)
    rune_warning_diff_thres = 0.2

    # ────────────────
    # Rune Detection
    # ────────────────
    rune_detect_box_width = 120
    rune_detect_box_height = 150
    rune_detect_diff_thres = 0.05
    rune_finding_timeout = 600 # second
    near_rune_duration = 5 # second

    # ────────────────
    # Rune mini-game
    # ────────────────
    arrow_box_size = 80 # pixel, 75x75 box
    arrow_box_interval = 170 # pixel width
    arrow_box_start_point = (355, 355)
    arrow_box_dif_thres = 0.2

    # ────────────────
    # HP Bar and HP bar
    # ────────────────
    hp_bar_top_left = (348, 732)
    hp_bar_bottom_right = (509, 749)
    mp_bar_top_left = (517, 732)
    mp_bar_bottom_right = (678, 749)
    heal_ratio = 0.5 # heal when hp is below 50%
    add_mp_ratio = 0.5 # drink potion when mp is below 50%

    # ────────────────
    # Patrol
    # ────────────────
    monster_patrol_dif_thres = 0.8

    # ────────────────
    # Don't modify the following parameter unless you know what you are doing
    # ────────────────
    game_window_title = 'MapleStory Worlds-Artale (繁體中文版)'
    # color code for patrol route
    color_code = {
        # R   G   B
        (255, 0, 0): "walk left", # red
        (0, 0, 255): "walk right", # blue
        (255,127,0): "jump left", # orange
        (0,255,255): "jump right", # sky blue
        (255,0,255): "jump", # purple
        (127,127,127): "up", # gray
        (0,255,0): "stop", # green
        (255,255,0): "goal", # yellow
        (255,0,127): "teleport up", # pink
        (127,0,255): "teleport down", # light_purple
    }

    window_size = (752, 1282)
