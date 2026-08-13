#include "serial_control.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "driver/usb_serial_jtag.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_timer.h"

#include "ble_keyboard.h"

#define SERIAL_LINE_CAPACITY 160
#define SERIAL_CHUNK_CAPACITY 96
#define DEFAULT_TAP_MS 60

typedef enum {
    COMMAND_REJECTED = 0,
    COMMAND_ACCEPTED = 1,
} command_result_t;

static const char *TAG = "serial_control";

static bool send_all(const char *text)
{
    size_t remaining = strlen(text);
    const char *cursor = text;

    while (remaining > 0) {
        int sent = usb_serial_jtag_write_bytes(
            cursor, remaining, pdMS_TO_TICKS(100));
        if (sent <= 0) {
            return false;
        }
        cursor += sent;
        remaining -= (size_t)sent;
    }
    return true;
}

static bool send_error_for_hid_result(esp_err_t err)
{
    const char *message = "ERR HID_SEND_FAILED\n";
    if (err == ESP_ERR_INVALID_ARG) {
        message = "ERR INVALID_KEY_OR_ARGUMENT\n";
    } else if (err == ESP_ERR_INVALID_STATE) {
        message = "ERR BLE_NOT_READY\n";
    } else if (err == ESP_ERR_NO_MEM) {
        message = "ERR TOO_MANY_KEYS\n";
    }
    return send_all(message);
}

static bool parse_usage(const char *token, uint8_t *usage)
{
    if (token == NULL || *token == '\0') {
        return false;
    }

    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(token, &end, 16);
    if (errno != 0 || end == token || *end != '\0' || value > UINT8_MAX) {
        return false;
    }
    *usage = (uint8_t)value;
    return ble_keyboard_usage_valid(*usage);
}

static bool parse_decimal(const char *token, uint32_t *value)
{
    if (token == NULL || *token == '\0') {
        return false;
    }

    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(token, &end, 10);
    if (errno != 0 || end == token || *end != '\0' || parsed > UINT32_MAX) {
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static bool no_extra_token(char **save)
{
    return strtok_r(NULL, " \t", save) == NULL;
}

static command_result_t handle_command(char *line)
{
    char *save = NULL;
    char *command = strtok_r(line, " \t", &save);
    if (command == NULL) {
        (void)send_all("ERR EMPTY_COMMAND\n");
        return COMMAND_REJECTED;
    }

    for (char *p = command; *p != '\0'; ++p) {
        *p = (char)toupper((unsigned char)*p);
    }

    if (strcmp(command, "PING") == 0) {
        if (!no_extra_token(&save)) {
            (void)send_all("ERR BAD_ARGUMENTS\n");
            return COMMAND_REJECTED;
        }
        (void)send_all("PONG\n");
        return COMMAND_ACCEPTED;
    }

    if (strcmp(command, "STATUS") == 0) {
        if (!no_extra_token(&save)) {
            (void)send_all("ERR BAD_ARGUMENTS\n");
            return COMMAND_REJECTED;
        }
        char response[96];
        snprintf(response, sizeof(response),
                 "OK SERIAL=1 BLE_CONNECTED=%d BLE_READY=%d\n",
                 ble_keyboard_connected(), ble_keyboard_ready());
        (void)send_all(response);
        return COMMAND_ACCEPTED;
    }

    if (strcmp(command, "HELP") == 0) {
        if (!no_extra_token(&save)) {
            (void)send_all("ERR BAD_ARGUMENTS\n");
            return COMMAND_REJECTED;
        }
        (void)send_all(
            "OK COMMANDS=PING,STATUS,DOWN,UP,TAP,STATE,RELEASE_ALL,HELP\n");
        return COMMAND_ACCEPTED;
    }

    if (strcmp(command, "RELEASE") == 0 ||
        strcmp(command, "RELEASE_ALL") == 0) {
        if (!no_extra_token(&save)) {
            (void)send_all("ERR BAD_ARGUMENTS\n");
            return COMMAND_REJECTED;
        }
        esp_err_t err = ble_keyboard_release_all();
        if (err != ESP_OK) {
            (void)send_error_for_hid_result(err);
            return COMMAND_REJECTED;
        }
        (void)send_all("OK RELEASE_ALL\n");
        return COMMAND_ACCEPTED;
    }

    if (strcmp(command, "DOWN") == 0 || strcmp(command, "UP") == 0) {
        char *usage_token = strtok_r(NULL, " \t", &save);
        uint8_t usage;
        if (!parse_usage(usage_token, &usage) || !no_extra_token(&save)) {
            (void)send_all("ERR BAD_USAGE\n");
            return COMMAND_REJECTED;
        }

        esp_err_t err = strcmp(command, "DOWN") == 0
                            ? ble_keyboard_key_down(usage)
                            : ble_keyboard_key_up(usage);
        if (err != ESP_OK) {
            (void)send_error_for_hid_result(err);
            return COMMAND_REJECTED;
        }

        char response[32];
        snprintf(response, sizeof(response), "OK %s 0x%02X\n", command, usage);
        (void)send_all(response);
        return COMMAND_ACCEPTED;
    }

    if (strcmp(command, "TAP") == 0) {
        char *usage_token = strtok_r(NULL, " \t", &save);
        char *duration_token = strtok_r(NULL, " \t", &save);
        uint8_t usage;
        uint32_t duration_ms = DEFAULT_TAP_MS;
        bool valid = parse_usage(usage_token, &usage);
        if (duration_token != NULL) {
            valid = valid && parse_decimal(duration_token, &duration_ms);
        }
        if (!valid || !no_extra_token(&save) || duration_ms == 0 ||
            duration_ms > CONFIG_HID_MAX_TAP_MS) {
            (void)send_all("ERR BAD_TAP_ARGUMENTS\n");
            return COMMAND_REJECTED;
        }

        esp_err_t err = ble_keyboard_tap(usage, duration_ms);
        if (err != ESP_OK) {
            (void)send_error_for_hid_result(err);
            return COMMAND_REJECTED;
        }

        char response[48];
        snprintf(response, sizeof(response), "OK TAP 0x%02X %" PRIu32 "ms\n",
                 usage, duration_ms);
        (void)send_all(response);
        return COMMAND_ACCEPTED;
    }

    if (strcmp(command, "STATE") == 0) {
        uint8_t usages[14];
        size_t count = 0;
        char *token;
        while ((token = strtok_r(NULL, " \t", &save)) != NULL) {
            if (count >= sizeof(usages) || !parse_usage(token, &usages[count])) {
                (void)send_all("ERR BAD_STATE\n");
                return COMMAND_REJECTED;
            }
            ++count;
        }

        esp_err_t err = count == 0
                            ? ble_keyboard_release_all()
                            : ble_keyboard_set_state(usages, count);
        if (err != ESP_OK) {
            (void)send_error_for_hid_result(err);
            return COMMAND_REJECTED;
        }
        (void)send_all("OK STATE\n");
        return COMMAND_ACCEPTED;
    }

    (void)send_all("ERR UNKNOWN_COMMAND\n");
    return COMMAND_REJECTED;
}

static void serial_command_task(void *arg)
{
    (void)arg;

    char chunk[SERIAL_CHUNK_CAPACITY];
    char line[SERIAL_LINE_CAPACITY];
    size_t used = 0;
    bool discard_line = false;
    bool lease_active = false;
    int64_t deadline = 0;

    (void)ble_keyboard_release_all();
    ESP_LOGI(TAG, "USB Serial/JTAG command service ready");

    while (true) {
        if (lease_active && esp_timer_get_time() >= deadline) {
            (void)ble_keyboard_release_all();
            lease_active = false;
            ESP_LOGW(TAG, "serial command lease expired; all keys released");
        }

        int received = usb_serial_jtag_read_bytes(
            chunk, sizeof(chunk), pdMS_TO_TICKS(10));
        if (received <= 0) {
            continue;
        }

        for (int i = 0; i < received; ++i) {
            char ch = chunk[i];
            if (ch != '\n') {
                if (!discard_line) {
                    if (used + 1 < sizeof(line)) {
                        line[used++] = ch;
                    } else {
                        discard_line = true;
                    }
                }
                continue;
            }

            if (discard_line) {
                (void)send_all("ERR LINE_TOO_LONG\n");
                discard_line = false;
                used = 0;
                continue;
            }

            if (used > 0 && line[used - 1] == '\r') {
                --used;
            }
            line[used] = '\0';
            used = 0;

            if (handle_command(line) == COMMAND_ACCEPTED) {
                deadline = esp_timer_get_time() +
                           (int64_t)CONFIG_HID_LEASE_TIMEOUT_MS * 1000;
                lease_active = true;
            }
        }
    }
}

esp_err_t serial_control_start(void)
{
    usb_serial_jtag_driver_config_t config = {
        .tx_buffer_size = 1024,
        .rx_buffer_size = 512,
    };
    esp_err_t err = usb_serial_jtag_driver_install(&config);
    if (err != ESP_OK) {
        return err;
    }

    if (xTaskCreate(serial_command_task, "hid_serial", 6144, NULL, 6, NULL) !=
        pdPASS) {
        (void)usb_serial_jtag_driver_uninstall();
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
