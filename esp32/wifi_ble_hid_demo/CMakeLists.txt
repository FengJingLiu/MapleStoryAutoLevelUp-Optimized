cmake_minimum_required(VERSION 3.22)

set(SDKCONFIG_DEFAULTS "sdkconfig.defaults.example")
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(maple_serial_ble_hid_demo)
