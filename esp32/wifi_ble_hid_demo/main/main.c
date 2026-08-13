#include "ble_keyboard.h"
#include "serial_control.h"

#include "esp_err.h"
#include "esp_log.h"
#include "nvs_flash.h"

static const char *TAG = "maple_hid_demo";

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ESP_LOGI(TAG, "starting ESP32-S3 USB serial + BLE HID bridge");
    ESP_ERROR_CHECK(ble_keyboard_init());
    ESP_ERROR_CHECK(serial_control_start());
}
