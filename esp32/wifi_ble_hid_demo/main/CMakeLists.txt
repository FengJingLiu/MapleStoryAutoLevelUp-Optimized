idf_component_register(
    SRCS
        "main.c"
        "ble_keyboard.c"
        "wifi_control.c"
    INCLUDE_DIRS "."
    REQUIRES
        bt
        esp_event
        esp_hid
        esp_netif
        esp_timer
        esp_wifi
        lwip
        nvs_flash
)
