#include "ble_keyboard.h"

#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

#include "esp_bt.h"
#include "esp_hid_common.h"
#include "esp_hidd.h"
#include "esp_log.h"

#include "host/ble_gap.h"
#include "host/ble_hs.h"
#include "host/ble_hs_adv.h"
#include "host/ble_hs_id.h"
#include "host/ble_sm.h"
#include "host/ble_store.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "services/gap/ble_svc_gap.h"

#define KEYBOARD_REPORT_ID 1
#define KEYBOARD_KEY_SLOTS 6
#define HID_SERVICE_UUID 0x1812

static const char *TAG = "ble_keyboard";

/*
 * Standard boot-compatible keyboard report:
 *   byte 0: E0-E7 modifiers
 *   byte 1: reserved
 *   byte 2-7: up to six simultaneous keys
 * The report ID is handled by the ESP-IDF HID service and is not part of the
 * eight bytes passed to esp_hidd_dev_input_set().
 */
static const uint8_t s_keyboard_report_map[] = {
    0x05, 0x01,        /* Usage Page (Generic Desktop) */
    0x09, 0x06,        /* Usage (Keyboard) */
    0xA1, 0x01,        /* Collection (Application) */
    0x85, KEYBOARD_REPORT_ID,
    0x05, 0x07,        /* Usage Page (Keyboard/Keypad) */
    0x19, 0xE0,        /* Usage Minimum (Left Control) */
    0x29, 0xE7,        /* Usage Maximum (Right GUI) */
    0x15, 0x00,
    0x25, 0x01,
    0x75, 0x01,
    0x95, 0x08,
    0x81, 0x02,        /* Input (Data, Variable, Absolute) */
    0x95, 0x01,
    0x75, 0x08,
    0x81, 0x03,        /* Input (Constant) */
    0x95, 0x05,
    0x75, 0x01,
    0x05, 0x08,        /* Usage Page (LEDs) */
    0x19, 0x01,
    0x29, 0x05,
    0x91, 0x02,        /* Output (Data, Variable, Absolute) */
    0x95, 0x01,
    0x75, 0x03,
    0x91, 0x03,        /* Output (Constant) */
    0x95, KEYBOARD_KEY_SLOTS,
    0x75, 0x08,
    0x15, 0x00,
    0x25, 0x73,
    0x05, 0x07,
    0x19, 0x00,
    0x29, 0x73,
    0x81, 0x00,        /* Input (Data, Array, Absolute) */
    0xC0
};

static esp_hid_raw_report_map_t s_report_maps[] = {
    {
        .data = s_keyboard_report_map,
        .len = sizeof(s_keyboard_report_map),
    },
};

static const esp_hid_device_config_t s_hid_config = {
    .vendor_id = 0x303A,
    .product_id = 0x4001,
    .version = 0x0100,
    .device_name = CONFIG_HID_DEVICE_NAME,
    .manufacturer_name = "Espressif",
    .serial_number = "MAPLE-S3-DEMO",
    .report_maps = s_report_maps,
    .report_maps_len = 1,
};

static esp_hidd_dev_t *s_hid_dev;
static SemaphoreHandle_t s_report_mutex;
static TaskHandle_t s_advertise_task;
static volatile bool s_gap_connected;
static volatile bool s_link_encrypted;
static volatile bool s_hid_connected;
static uint8_t s_modifier;
static uint8_t s_keys[KEYBOARD_KEY_SLOTS];

void ble_store_config_init(void);

static bool usage_is_modifier(uint8_t usage)
{
    return usage >= 0xE0 && usage <= 0xE7;
}

bool ble_keyboard_usage_valid(uint8_t usage)
{
    return (usage >= 0x04 && usage <= 0x73) || usage_is_modifier(usage);
}

static bool ready_unlocked(void)
{
    return s_gap_connected && s_link_encrypted && s_hid_connected &&
           s_hid_dev != NULL;
}

bool ble_keyboard_connected(void)
{
    return s_gap_connected && s_hid_connected;
}

bool ble_keyboard_ready(void)
{
    return ready_unlocked();
}

static esp_err_t send_report_locked(void)
{
    if (!ready_unlocked()) {
        return ESP_ERR_INVALID_STATE;
    }

    uint8_t report[8] = {
        s_modifier, 0,
        s_keys[0], s_keys[1], s_keys[2],
        s_keys[3], s_keys[4], s_keys[5],
    };
    esp_err_t err = esp_hidd_dev_input_set(
        s_hid_dev, 0, KEYBOARD_REPORT_ID, report, sizeof(report));
    if (err == ESP_OK) {
        return ESP_OK;
    }

    /* A failed report must never leave our local state believing a key is down. */
    ESP_LOGW(TAG, "HID report failed (%s); forcing an all-keys-up report",
             esp_err_to_name(err));
    s_modifier = 0;
    memset(s_keys, 0, sizeof(s_keys));
    memset(report, 0, sizeof(report));
    (void)esp_hidd_dev_input_set(
        s_hid_dev, 0, KEYBOARD_REPORT_ID, report, sizeof(report));
    return err;
}

static bool key_present_locked(uint8_t usage)
{
    if (usage_is_modifier(usage)) {
        return (s_modifier & (1U << (usage - 0xE0))) != 0;
    }
    for (size_t i = 0; i < sizeof(s_keys); ++i) {
        if (s_keys[i] == usage) {
            return true;
        }
    }
    return false;
}

static esp_err_t add_key_locked(uint8_t usage)
{
    if (usage_is_modifier(usage)) {
        s_modifier |= (uint8_t)(1U << (usage - 0xE0));
        return ESP_OK;
    }
    if (key_present_locked(usage)) {
        return ESP_OK;
    }
    for (size_t i = 0; i < sizeof(s_keys); ++i) {
        if (s_keys[i] == 0) {
            s_keys[i] = usage;
            return ESP_OK;
        }
    }
    return ESP_ERR_NO_MEM;
}

static void remove_key_locked(uint8_t usage)
{
    if (usage_is_modifier(usage)) {
        s_modifier &= (uint8_t)~(1U << (usage - 0xE0));
        return;
    }
    for (size_t i = 0; i < sizeof(s_keys); ++i) {
        if (s_keys[i] == usage) {
            s_keys[i] = 0;
        }
    }
}

esp_err_t ble_keyboard_key_down(uint8_t usage)
{
    if (!ble_keyboard_usage_valid(usage)) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(s_report_mutex, portMAX_DELAY);
    if (!ready_unlocked()) {
        xSemaphoreGive(s_report_mutex);
        return ESP_ERR_INVALID_STATE;
    }
    esp_err_t err = add_key_locked(usage);
    if (err == ESP_OK) {
        err = send_report_locked();
    }
    xSemaphoreGive(s_report_mutex);
    return err;
}

esp_err_t ble_keyboard_key_up(uint8_t usage)
{
    if (!ble_keyboard_usage_valid(usage)) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(s_report_mutex, portMAX_DELAY);
    if (!ready_unlocked()) {
        xSemaphoreGive(s_report_mutex);
        return ESP_ERR_INVALID_STATE;
    }
    remove_key_locked(usage);
    esp_err_t err = send_report_locked();
    xSemaphoreGive(s_report_mutex);
    return err;
}

esp_err_t ble_keyboard_tap(uint8_t usage, uint32_t hold_ms)
{
    if (!ble_keyboard_usage_valid(usage) || hold_ms == 0 ||
        hold_ms > CONFIG_HID_MAX_TAP_MS) {
        return ESP_ERR_INVALID_ARG;
    }

    xSemaphoreTake(s_report_mutex, portMAX_DELAY);
    if (!ready_unlocked()) {
        xSemaphoreGive(s_report_mutex);
        return ESP_ERR_INVALID_STATE;
    }

    bool was_down = key_present_locked(usage);
    esp_err_t err = ESP_OK;
    if (!was_down) {
        err = add_key_locked(usage);
        if (err == ESP_OK) {
            err = send_report_locked();
        }
    }
    xSemaphoreGive(s_report_mutex);

    if (err != ESP_OK || was_down) {
        return err;
    }

    vTaskDelay(pdMS_TO_TICKS(hold_ms));
    return ble_keyboard_key_up(usage);
}

esp_err_t ble_keyboard_set_state(const uint8_t *usages, size_t count)
{
    if (count > 0 && usages == NULL) {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t modifier = 0;
    uint8_t keys[KEYBOARD_KEY_SLOTS] = {0};
    size_t key_count = 0;

    for (size_t i = 0; i < count; ++i) {
        uint8_t usage = usages[i];
        if (!ble_keyboard_usage_valid(usage)) {
            return ESP_ERR_INVALID_ARG;
        }
        if (usage_is_modifier(usage)) {
            modifier |= (uint8_t)(1U << (usage - 0xE0));
            continue;
        }

        bool duplicate = false;
        for (size_t j = 0; j < key_count; ++j) {
            if (keys[j] == usage) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        if (key_count >= KEYBOARD_KEY_SLOTS) {
            return ESP_ERR_NO_MEM;
        }
        keys[key_count++] = usage;
    }

    xSemaphoreTake(s_report_mutex, portMAX_DELAY);
    if (!ready_unlocked()) {
        xSemaphoreGive(s_report_mutex);
        return ESP_ERR_INVALID_STATE;
    }
    s_modifier = modifier;
    memcpy(s_keys, keys, sizeof(s_keys));
    esp_err_t err = send_report_locked();
    xSemaphoreGive(s_report_mutex);
    return err;
}

esp_err_t ble_keyboard_release_all(void)
{
    if (s_report_mutex == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    xSemaphoreTake(s_report_mutex, portMAX_DELAY);
    s_modifier = 0;
    memset(s_keys, 0, sizeof(s_keys));
    esp_err_t err = ESP_OK;
    if (ready_unlocked()) {
        err = send_report_locked();
    }
    xSemaphoreGive(s_report_mutex);
    return err;
}

static void clear_disconnected_state(void)
{
    s_gap_connected = false;
    s_link_encrypted = false;
    s_hid_connected = false;
    if (s_report_mutex != NULL) {
        xSemaphoreTake(s_report_mutex, portMAX_DELAY);
        s_modifier = 0;
        memset(s_keys, 0, sizeof(s_keys));
        xSemaphoreGive(s_report_mutex);
    }
}

static void request_advertising(void)
{
    if (s_advertise_task != NULL) {
        xTaskNotifyGive(s_advertise_task);
    }
}

static int gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status != 0) {
            ESP_LOGW(TAG, "BLE connection failed, status=%d", event->connect.status);
            clear_disconnected_state();
            request_advertising();
            return 0;
        }

        s_gap_connected = true;
        s_link_encrypted = false;
        ESP_LOGI(TAG, "BLE connected; starting security");
        {
            int rc = ble_gap_security_initiate(event->connect.conn_handle);
            if (rc != 0 && rc != BLE_HS_EALREADY) {
                ESP_LOGW(TAG, "security initiation returned rc=%d", rc);
            }
        }
        return 0;

    case BLE_GAP_EVENT_DISCONNECT:
        ESP_LOGI(TAG, "BLE disconnected, reason=%d", event->disconnect.reason);
        clear_disconnected_state();
        request_advertising();
        return 0;

    case BLE_GAP_EVENT_ENC_CHANGE: {
        struct ble_gap_conn_desc desc;
        int rc = ble_gap_conn_find(event->enc_change.conn_handle, &desc);
        s_link_encrypted = event->enc_change.status == 0 && rc == 0 &&
                           desc.sec_state.encrypted;
        ESP_LOGI(TAG, "BLE encryption %s (status=%d)",
                 s_link_encrypted ? "ready" : "not ready",
                 event->enc_change.status);
        return 0;
    }

    case BLE_GAP_EVENT_ADV_COMPLETE:
        ESP_LOGW(TAG, "BLE advertising completed, reason=%d; restarting",
                 event->adv_complete.reason);
        request_advertising();
        return 0;

    case BLE_GAP_EVENT_REPEAT_PAIRING: {
        struct ble_gap_conn_desc desc;
        if (ble_gap_conn_find(event->repeat_pairing.conn_handle, &desc) == 0) {
            (void)ble_store_util_delete_peer(&desc.peer_id_addr);
        }
        return BLE_GAP_REPEAT_PAIRING_RETRY;
    }

    case BLE_GAP_EVENT_SUBSCRIBE:
        ESP_LOGI(TAG, "HID notification subscription changed: notify=%d",
                 event->subscribe.cur_notify);
        return 0;

    default:
        return 0;
    }
}

static esp_err_t advertise_once(void)
{
    if (s_gap_connected || ble_gap_adv_active()) {
        return ESP_OK;
    }

    static const ble_uuid16_t hid_uuid = BLE_UUID16_INIT(HID_SERVICE_UUID);
    struct ble_hs_adv_fields adv_fields = {0};
    adv_fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    adv_fields.appearance = ESP_HID_APPEARANCE_KEYBOARD;
    adv_fields.appearance_is_present = 1;
    adv_fields.uuids16 = &hid_uuid;
    adv_fields.num_uuids16 = 1;
    adv_fields.uuids16_is_complete = 1;

    int rc = ble_gap_adv_set_fields(&adv_fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "setting BLE advertising fields failed, rc=%d", rc);
        return ESP_FAIL;
    }

    struct ble_hs_adv_fields response_fields = {0};
    response_fields.name = (const uint8_t *)CONFIG_HID_DEVICE_NAME;
    response_fields.name_len = strlen(CONFIG_HID_DEVICE_NAME);
    response_fields.name_is_complete = 1;
    rc = ble_gap_adv_rsp_set_fields(&response_fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "setting BLE scan response failed, rc=%d", rc);
        return ESP_FAIL;
    }

    uint8_t own_addr_type;
    rc = ble_hs_id_infer_auto(0, &own_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "selecting BLE address failed, rc=%d", rc);
        return ESP_FAIL;
    }

    struct ble_gap_adv_params params = {0};
    params.conn_mode = BLE_GAP_CONN_MODE_UND;
    params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    params.itvl_min = BLE_GAP_ADV_ITVL_MS(30);
    params.itvl_max = BLE_GAP_ADV_ITVL_MS(50);

    rc = ble_gap_adv_start(
        own_addr_type, NULL, BLE_HS_FOREVER, &params, gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "starting BLE advertising failed, rc=%d", rc);
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "advertising as '%s'", CONFIG_HID_DEVICE_NAME);
    return ESP_OK;
}

static void advertise_task(void *arg)
{
    (void)arg;
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        /* GAP advertising must be restarted outside the GAP callback context. */
        vTaskDelay(pdMS_TO_TICKS(20));
        (void)advertise_once();
    }
}

static void hid_event(void *handler_args, esp_event_base_t base,
                      int32_t id, void *event_data)
{
    (void)handler_args;
    (void)base;
    esp_hidd_event_data_t *data = event_data;

    switch ((esp_hidd_event_t)id) {
    case ESP_HIDD_START_EVENT:
        ESP_LOGI(TAG, "BLE HID service started");
        request_advertising();
        break;
    case ESP_HIDD_CONNECT_EVENT:
        s_gap_connected = true;
        s_hid_connected = true;
        ESP_LOGI(TAG, "BLE HID host connected");
        break;
    case ESP_HIDD_OUTPUT_EVENT:
        if (data != NULL && data->output.length > 0) {
            ESP_LOGI(TAG, "keyboard LED output=0x%02x", data->output.data[0]);
        }
        break;
    case ESP_HIDD_DISCONNECT_EVENT:
        clear_disconnected_state();
        request_advertising();
        break;
    default:
        break;
    }
}

static void nimble_host_task(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "NimBLE host task started");
    nimble_port_run();
    nimble_port_freertos_deinit();
}

esp_err_t ble_keyboard_init(void)
{
    s_report_mutex = xSemaphoreCreateMutex();
    if (s_report_mutex == NULL) {
        return ESP_ERR_NO_MEM;
    }

    if (xTaskCreate(advertise_task, "ble_advertise", 3072, NULL, 5,
                    &s_advertise_task) != pdPASS) {
        return ESP_ERR_NO_MEM;
    }

    esp_err_t err = esp_bt_controller_mem_release(ESP_BT_MODE_CLASSIC_BT);
    if (err != ESP_OK) {
        return err;
    }

    esp_bt_controller_config_t controller_config = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    err = esp_bt_controller_init(&controller_config);
    if (err != ESP_OK) {
        return err;
    }
    err = esp_bt_controller_enable(ESP_BT_MODE_BLE);
    if (err != ESP_OK) {
        return err;
    }
    err = esp_nimble_init();
    if (err != ESP_OK) {
        return err;
    }

    ble_hs_cfg.sm_io_cap = BLE_HS_IO_NO_INPUT_OUTPUT;
    ble_hs_cfg.sm_bonding = 1;
    ble_hs_cfg.sm_mitm = 0;
    ble_hs_cfg.sm_sc = 1;
    ble_hs_cfg.sm_our_key_dist = BLE_SM_PAIR_KEY_DIST_ID | BLE_SM_PAIR_KEY_DIST_ENC;
    ble_hs_cfg.sm_their_key_dist = BLE_SM_PAIR_KEY_DIST_ID | BLE_SM_PAIR_KEY_DIST_ENC;

    err = esp_hidd_dev_init(
        &s_hid_config, ESP_HID_TRANSPORT_BLE, hid_event, &s_hid_dev);
    if (err != ESP_OK) {
        return err;
    }

    int rc = ble_svc_gap_device_name_set(CONFIG_HID_DEVICE_NAME);
    if (rc != 0) {
        ESP_LOGE(TAG, "setting GAP device name failed, rc=%d", rc);
        return ESP_FAIL;
    }

    ble_store_config_init();
    ble_hs_cfg.store_status_cb = ble_store_util_status_rr;
    nimble_port_freertos_init(nimble_host_task);
    return ESP_OK;
}
