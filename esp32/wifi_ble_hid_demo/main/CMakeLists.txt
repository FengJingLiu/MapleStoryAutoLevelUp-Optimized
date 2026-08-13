idf_component_register(
    SRCS
        "main.c"
        "ble_keyboard.c"
        "serial_control.c"
    INCLUDE_DIRS "."
    REQUIRES
        bt
        esp_driver_usb_serial_jtag
        esp_hid
        esp_timer
        nvs_flash
)
