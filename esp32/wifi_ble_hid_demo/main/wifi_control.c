#include "wifi_control.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"
#include "freertos/task.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/tcp.h"

#include "ble_keyboard.h"

#define WIFI_READY_BIT BIT0
#define TCP_LINE_CAPACITY 160
#define TCP_CHUNK_CAPACITY 96
#define DEFAULT_TAP_MS 60

typedef enum {
    COMMAND_SOCKET_ERROR = -1,
    COMMAND_REJECTED = 0,
    COMMAND_ACCEPTED = 1,
} command_result_t;

static const char *TAG = "wifi_control";
static EventGroupHandle_t s_wifi_events;

static bool wifi_ready(void)
{
    return (xEventGroupGetBits(s_wifi_events) & WIFI_READY_BIT) != 0;
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    (void)arg;

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "connecting to Wi-Fi SSID '%s'", CONFIG_HID_WIFI_SSID);
        (void)esp_wifi_connect();
        return;
    }

    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(s_wifi_events, WIFI_READY_BIT);
        (void)ble_keyboard_release_all();
        ESP_LOGW(TAG, "Wi-Fi disconnected; retrying");
        (void)esp_wifi_connect();
        return;
    }

    if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        const ip_event_got_ip_t *got_ip = event_data;
        ESP_LOGI(TAG, "got ip: " IPSTR ", TCP port: %d",
                 IP2STR(&got_ip->ip_info.ip), CONFIG_HID_TCP_PORT);
        xEventGroupSetBits(s_wifi_events, WIFI_READY_BIT);
    }
}

static bool send_all(int socket_fd, const char *text)
{
    size_t remaining = strlen(text);
    const char *cursor = text;

    while (remaining > 0) {
        int sent = send(socket_fd, cursor, remaining, 0);
        if (sent > 0) {
            cursor += sent;
            remaining -= (size_t)sent;
            continue;
        }
        if (sent < 0 && errno == EINTR) {
            continue;
        }
        return false;
    }
    return true;
}

static bool send_error_for_hid_result(int socket_fd, esp_err_t err)
{
    const char *message = "ERR HID_SEND_FAILED\n";
    if (err == ESP_ERR_INVALID_ARG) {
        message = "ERR INVALID_KEY_OR_ARGUMENT\n";
    } else if (err == ESP_ERR_INVALID_STATE) {
        message = "ERR BLE_NOT_READY\n";
    } else if (err == ESP_ERR_NO_MEM) {
        message = "ERR TOO_MANY_KEYS\n";
    }
    return send_all(socket_fd, message);
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

static command_result_t handle_command(int socket_fd, char *line)
{
    char *save = NULL;
    char *command = strtok_r(line, " \t", &save);
    if (command == NULL) {
        return send_all(socket_fd, "ERR EMPTY_COMMAND\n")
                   ? COMMAND_REJECTED
                   : COMMAND_SOCKET_ERROR;
    }

    for (char *p = command; *p != '\0'; ++p) {
        *p = (char)toupper((unsigned char)*p);
    }

    if (strcmp(command, "PING") == 0) {
        if (!no_extra_token(&save)) {
            return send_all(socket_fd, "ERR BAD_ARGUMENTS\n")
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }
        return send_all(socket_fd, "PONG\n")
                   ? COMMAND_ACCEPTED
                   : COMMAND_SOCKET_ERROR;
    }

    if (strcmp(command, "STATUS") == 0) {
        if (!no_extra_token(&save)) {
            return send_all(socket_fd, "ERR BAD_ARGUMENTS\n")
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }
        char response[96];
        snprintf(response, sizeof(response),
                 "OK WIFI=%d BLE_CONNECTED=%d BLE_READY=%d\n",
                 wifi_ready(), ble_keyboard_connected(), ble_keyboard_ready());
        return send_all(socket_fd, response)
                   ? COMMAND_ACCEPTED
                   : COMMAND_SOCKET_ERROR;
    }

    if (strcmp(command, "HELP") == 0) {
        if (!no_extra_token(&save)) {
            return send_all(socket_fd, "ERR BAD_ARGUMENTS\n")
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }
        return send_all(socket_fd,
                        "OK COMMANDS=PING,STATUS,DOWN,UP,TAP,STATE,RELEASE_ALL,HELP\n")
                   ? COMMAND_ACCEPTED
                   : COMMAND_SOCKET_ERROR;
    }

    if (strcmp(command, "RELEASE") == 0 ||
        strcmp(command, "RELEASE_ALL") == 0) {
        if (!no_extra_token(&save)) {
            return send_all(socket_fd, "ERR BAD_ARGUMENTS\n")
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }
        esp_err_t err = ble_keyboard_release_all();
        if (err != ESP_OK) {
            return send_error_for_hid_result(socket_fd, err)
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }
        return send_all(socket_fd, "OK RELEASE_ALL\n")
                   ? COMMAND_ACCEPTED
                   : COMMAND_SOCKET_ERROR;
    }

    if (strcmp(command, "DOWN") == 0 || strcmp(command, "UP") == 0) {
        char *usage_token = strtok_r(NULL, " \t", &save);
        uint8_t usage;
        if (!parse_usage(usage_token, &usage) || !no_extra_token(&save)) {
            return send_all(socket_fd, "ERR BAD_USAGE\n")
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }

        esp_err_t err = strcmp(command, "DOWN") == 0
                            ? ble_keyboard_key_down(usage)
                            : ble_keyboard_key_up(usage);
        if (err != ESP_OK) {
            return send_error_for_hid_result(socket_fd, err)
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }

        char response[32];
        snprintf(response, sizeof(response), "OK %s 0x%02X\n", command, usage);
        return send_all(socket_fd, response)
                   ? COMMAND_ACCEPTED
                   : COMMAND_SOCKET_ERROR;
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
            return send_all(socket_fd, "ERR BAD_TAP_ARGUMENTS\n")
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }

        esp_err_t err = ble_keyboard_tap(usage, duration_ms);
        if (err != ESP_OK) {
            return send_error_for_hid_result(socket_fd, err)
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }

        char response[48];
        snprintf(response, sizeof(response), "OK TAP 0x%02X %" PRIu32 "ms\n",
                 usage, duration_ms);
        return send_all(socket_fd, response)
                   ? COMMAND_ACCEPTED
                   : COMMAND_SOCKET_ERROR;
    }

    if (strcmp(command, "STATE") == 0) {
        uint8_t usages[14];
        size_t count = 0;
        char *token;
        while ((token = strtok_r(NULL, " \t", &save)) != NULL) {
            if (count >= sizeof(usages) || !parse_usage(token, &usages[count])) {
                return send_all(socket_fd, "ERR BAD_STATE\n")
                           ? COMMAND_REJECTED
                           : COMMAND_SOCKET_ERROR;
            }
            ++count;
        }

        esp_err_t err = count == 0
                            ? ble_keyboard_release_all()
                            : ble_keyboard_set_state(usages, count);
        if (err != ESP_OK) {
            return send_error_for_hid_result(socket_fd, err)
                       ? COMMAND_REJECTED
                       : COMMAND_SOCKET_ERROR;
        }
        return send_all(socket_fd, "OK STATE\n")
                   ? COMMAND_ACCEPTED
                   : COMMAND_SOCKET_ERROR;
    }

    return send_all(socket_fd, "ERR UNKNOWN_COMMAND\n")
               ? COMMAND_REJECTED
               : COMMAND_SOCKET_ERROR;
}

static void configure_client_socket(int socket_fd)
{
    int enabled = 1;
    struct timeval timeout = {
        .tv_sec = 0,
        .tv_usec = 100000,
    };
    (void)setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY,
                     &enabled, sizeof(enabled));
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_KEEPALIVE,
                     &enabled, sizeof(enabled));
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_RCVTIMEO,
                     &timeout, sizeof(timeout));
    (void)setsockopt(socket_fd, SOL_SOCKET, SO_SNDTIMEO,
                     &timeout, sizeof(timeout));
}

static void serve_client(int socket_fd)
{
    configure_client_socket(socket_fd);
    (void)ble_keyboard_release_all();

    char chunk[TCP_CHUNK_CAPACITY];
    char line[TCP_LINE_CAPACITY];
    size_t used = 0;
    bool discard_line = false;
    bool lease_expired = false;
    int64_t deadline = esp_timer_get_time() +
                       (int64_t)CONFIG_HID_LEASE_TIMEOUT_MS * 1000;

    while (wifi_ready()) {
        if (esp_timer_get_time() >= deadline) {
            lease_expired = true;
            break;
        }

        int received = recv(socket_fd, chunk, sizeof(chunk), 0);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
                continue;
            }
            break;
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
                if (!send_all(socket_fd, "ERR LINE_TOO_LONG\n")) {
                    goto session_finished;
                }
                discard_line = false;
                used = 0;
                continue;
            }

            if (used > 0 && line[used - 1] == '\r') {
                --used;
            }
            line[used] = '\0';
            used = 0;

            command_result_t result = handle_command(socket_fd, line);
            if (result == COMMAND_SOCKET_ERROR) {
                goto session_finished;
            }
            if (result == COMMAND_ACCEPTED) {
                deadline = esp_timer_get_time() +
                           (int64_t)CONFIG_HID_LEASE_TIMEOUT_MS * 1000;
            }
        }
    }

session_finished:
    (void)ble_keyboard_release_all();
    if (lease_expired) {
        (void)send_all(socket_fd, "ERR LEASE_EXPIRED\n");
    }
    (void)shutdown(socket_fd, SHUT_RDWR);
    close(socket_fd);
    ESP_LOGI(TAG, "TCP client disconnected; all keys released");
}

static int create_listen_socket(void)
{
    int listen_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (listen_fd < 0) {
        ESP_LOGE(TAG, "socket() failed, errno=%d", errno);
        return -1;
    }

    int reuse = 1;
    (void)setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR,
                     &reuse, sizeof(reuse));

    struct sockaddr_in address = {
        .sin_family = AF_INET,
        .sin_port = htons(CONFIG_HID_TCP_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (bind(listen_fd, (struct sockaddr *)&address, sizeof(address)) != 0) {
        ESP_LOGE(TAG, "bind() failed, errno=%d", errno);
        close(listen_fd);
        return -1;
    }
    if (listen(listen_fd, 1) != 0) {
        ESP_LOGE(TAG, "listen() failed, errno=%d", errno);
        close(listen_fd);
        return -1;
    }

    ESP_LOGI(TAG, "TCP server listening on port %d", CONFIG_HID_TCP_PORT);
    return listen_fd;
}

static void tcp_server_task(void *arg)
{
    (void)arg;

    while (true) {
        xEventGroupWaitBits(s_wifi_events, WIFI_READY_BIT,
                            pdFALSE, pdTRUE, portMAX_DELAY);
        int listen_fd = create_listen_socket();
        if (listen_fd < 0) {
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        while (wifi_ready()) {
            fd_set read_set;
            FD_ZERO(&read_set);
            FD_SET(listen_fd, &read_set);
            struct timeval timeout = {
                .tv_sec = 0,
                .tv_usec = 200000,
            };
            int selected = select(listen_fd + 1, &read_set, NULL, NULL, &timeout);
            if (selected == 0) {
                continue;
            }
            if (selected < 0) {
                if (errno == EINTR) {
                    continue;
                }
                ESP_LOGW(TAG, "select() failed, errno=%d", errno);
                break;
            }

            struct sockaddr_in client_address;
            socklen_t client_length = sizeof(client_address);
            int client_fd = accept(
                listen_fd, (struct sockaddr *)&client_address, &client_length);
            if (client_fd < 0) {
                ESP_LOGW(TAG, "accept() failed, errno=%d", errno);
                continue;
            }

            char client_ip[INET_ADDRSTRLEN] = {0};
            inet_ntoa_r(client_address.sin_addr, client_ip, sizeof(client_ip));
            ESP_LOGI(TAG, "TCP client connected from %s", client_ip);
            serve_client(client_fd);
        }

        close(listen_fd);
        (void)ble_keyboard_release_all();
    }
}

esp_err_t wifi_control_start(void)
{
    if (CONFIG_HID_WIFI_SSID[0] == '\0') {
        ESP_LOGW(TAG,
                 "Wi-Fi SSID is empty. Configure 'Wi-Fi BLE HID Demo' with idf.py menuconfig. BLE will still advertise.");
        return ESP_OK;
    }

    s_wifi_events = xEventGroupCreate();
    if (s_wifi_events == NULL) {
        return ESP_ERR_NO_MEM;
    }

    esp_err_t err = esp_netif_init();
    if (err != ESP_OK) {
        return err;
    }
    err = esp_event_loop_create_default();
    if (err != ESP_OK) {
        return err;
    }
    if (esp_netif_create_default_wifi_sta() == NULL) {
        return ESP_FAIL;
    }

    wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
    err = esp_wifi_init(&init_config);
    if (err != ESP_OK) {
        return err;
    }

    err = esp_event_handler_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL);
    if (err != ESP_OK) {
        return err;
    }
    err = esp_event_handler_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL);
    if (err != ESP_OK) {
        return err;
    }

    wifi_config_t wifi_config = {0};
    strlcpy((char *)wifi_config.sta.ssid, CONFIG_HID_WIFI_SSID,
            sizeof(wifi_config.sta.ssid));
    strlcpy((char *)wifi_config.sta.password, CONFIG_HID_WIFI_PASSWORD,
            sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = CONFIG_HID_WIFI_PASSWORD[0] == '\0'
                                             ? WIFI_AUTH_OPEN
                                             : WIFI_AUTH_WPA2_PSK;
    wifi_config.sta.pmf_cfg.capable = true;
    wifi_config.sta.pmf_cfg.required = false;

    err = esp_wifi_set_mode(WIFI_MODE_STA);
    if (err != ESP_OK) {
        return err;
    }
    err = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    if (err != ESP_OK) {
        return err;
    }
    err = esp_wifi_start();
    if (err != ESP_OK) {
        return err;
    }
    err = esp_wifi_set_ps(WIFI_PS_NONE);
    if (err != ESP_OK) {
        return err;
    }

    if (xTaskCreate(tcp_server_task, "hid_tcp_server", 6144, NULL, 5, NULL) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}
