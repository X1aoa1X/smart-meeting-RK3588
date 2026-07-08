cmake_minimum_required(VERSION 3.10)
project(yolo_camera_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── 查找 OpenCV ──────────────────────────────────────────────
find_package(OpenCV REQUIRED
    COMPONENTS core imgproc videoio highgui)

# ── 查找 RKNN ───────────────────────────────────────────────
# 自动探测: 优先系统路径, 其次 vendored rknn_model_zoo
if(NOT DEFINED RKNN_INCLUDE_DIR)
    find_path(RKNN_INCLUDE_DIR rknn_api.h
        PATHS /usr/include/rknn
              "${CMAKE_CURRENT_SOURCE_DIR}/../rknn_model_zoo-2.1.0/3rdparty/rknpu2/include")
endif()
if(NOT DEFINED RKNN_LIB)
    find_library(RKNN_LIB rknnrt
        PATHS /usr/lib /usr/lib/aarch64-linux-gnu)
endif()
set(RKNN_INCLUDE_DIR "${RKNN_INCLUDE_DIR}" CACHE PATH "RKNN header path")
set(RKNN_LIB "${RKNN_LIB}" CACHE FILEPATH "RKNN runtime library")

# ── yolo_camera (原始单线程版本) ────────────────────────────
add_executable(yolo_camera
    yolo_camera.cc)

target_include_directories(yolo_camera PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR})

target_link_libraries(yolo_camera
    ${OpenCV_LIBS}
    ${RKNN_LIB}
    pthread
    dl)

# ── yolo_camera_fast (多线程 CPU 版本) ──────────────────────
add_executable(yolo_camera_fast
    yolo_camera_fast.cc)

target_include_directories(yolo_camera_fast PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR})

target_link_libraries(yolo_camera_fast
    ${OpenCV_LIBS}
    ${RKNN_LIB}
    pthread
    dl)

# ── yolo_camera_hw (多线程 + 硬件 MPP/RGA 解码) ────────────
# 条件编译: 需要 librga + librockchip_mpp
find_library(RGA_LIB rga PATHS /usr/lib /usr/lib/aarch64-linux-gnu)
find_library(MPP_LIB rockchip_mpp PATHS /usr/lib /usr/lib/aarch64-linux-gnu)
find_path(RGA_INCLUDE_DIR im2d.h PATHS /usr/include/rga)
find_path(MPP_INCLUDE_DIR rk_mpi.h PATHS /usr/include/rockchip)

if(RGA_LIB AND MPP_LIB AND RGA_INCLUDE_DIR AND MPP_INCLUDE_DIR)
    message(STATUS "Building yolo_camera_hw with MPP+RGA hardware decode")
    add_executable(yolo_camera_hw
        yolo_camera_fast.cc)
    target_compile_definitions(yolo_camera_hw PRIVATE ENABLE_HW_DECODE)
    target_include_directories(yolo_camera_hw PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${RKNN_INCLUDE_DIR}
        ${RGA_INCLUDE_DIR}
        ${MPP_INCLUDE_DIR})
    target_link_libraries(yolo_camera_hw
        ${OpenCV_LIBS}
        ${RKNN_LIB}
        ${RGA_LIB}
        ${MPP_LIB}
        pthread
        dl)
else()
    message(WARNING "librga or librockchip_mpp not found — skipping yolo_camera_hw")
endif()

# ── 安装 ────────────────────────────────────────────────────
install(TARGETS yolo_camera yolo_camera_fast DESTINATION .)
install(FILES "${CMAKE_SOURCE_DIR}/../best.rknn"
        DESTINATION . OPTIONAL)
