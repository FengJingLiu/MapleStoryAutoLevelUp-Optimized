#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t ble_keyboard_init(void);

bool ble_keyboard_connected(void);
bool ble_keyboard_ready(void);

esp_err_t ble_keyboard_key_down(uint8_t usage);
esp_err_t ble_keyboard_key_up(uint8_t usage);
esp_err_t ble_keyboard_tap(uint8_t usage, uint32_t hold_ms);
esp_err_t ble_keyboard_set_state(const uint8_t *usages, size_t count);
esp_err_t ble_keyboard_release_all(void);

esp_err_t ble_mouse_move(int32_t dx, int32_t dy, int32_t wheel);
esp_err_t ble_mouse_button_down(uint8_t button);
esp_err_t ble_mouse_button_up(uint8_t button);
esp_err_t ble_mouse_click(uint8_t button, uint32_t hold_ms);
esp_err_t ble_mouse_click_at(uint16_t x, uint16_t y, uint8_t button,
                             uint32_t hold_ms);
esp_err_t ble_mouse_retry_pending_release(void);

bool ble_keyboard_usage_valid(uint8_t usage);

#ifdef __cplusplus
}
#endif
