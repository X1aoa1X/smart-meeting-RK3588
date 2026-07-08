# yolo_core/CMakeLists.txt — 构建 pybind11 扩展模块 yolo_core.so
#
# 产物: build/yolo_core.cpython-3XX-aarch64-linux-gnu.so
#       (导入: `import yolo_core`)
#
# 依赖: librknnrt.so, OpenCV 4.x, pybind11, Python3 开发头文件, numpy 头文件
cmake_minimum_required(VERSION 3.14)
project(yolo_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── OpenCV ──────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# ── Python3 + pybind11 ──────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# ── numpy 头文件 (pybind11 numpy 互转需要) ──────────────────
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NUMPY_RESULT)
if(NOT NUMPY_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to locate numpy include dir")
endif()

# ── RKNN (自动探测: 系统 → vendored rknn_model_zoo) ─────────
if(NOT DEFINED RKNN_INCLUDE_DIR)
    find_path(RKNN_INCLUDE_DIR rknn_api.h
        PATHS /usr/include/rknn
              "${CMAKE_CURRENT_SOURCE_DIR}/../rknn_model_zoo-2.1.0/3rdparty/rknpu2/include")
endif()
if(NOT DEFINED RKNN_LIB)
    find_library(RKNN_LIB rknnrt
        PATHS /usr/lib /usr/lib/aarch64-linux-gnu)
endif()
if(NOT RKNN_INCLUDE_DIR OR NOT RKNN_LIB)
    message(FATAL_ERROR "RKNN header/library not found. Set RKNN_INCLUDE_DIR and RKNN_LIB.")
endif()
message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN lib:     ${RKNN_LIB}")
message(STATUS "NumPy include: ${NUMPY_INCLUDE_DIR}")

# ── pybind11 模块 ───────────────────────────────────────────
pybind11_add_module(yolo_core
    bindings/yolo_module.cc
    src/rknn_context.cc
    src/post_process.cc
    src/rknn_pool.cc
    src/yolo_engine.cc)

target_include_directories(yolo_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_INCLUDE_DIR}
    ${NUMPY_INCLUDE_DIR})

target_link_libraries(yolo_core PRIVATE
    ${OpenCV_LIBS}
    ${RKNN_LIB}
    pthread
    dl)

# 优化: Release 下 -O3
target_compile_options(yolo_core PRIVATE
    $<$<CONFIG:Release>:-O3>)

# ── 安装: 拷贝 .so 到 demo/ (方便 fusion_tracker.py 导入) ───
install(TARGETS yolo_core
        LIBRARY DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/../../demo")
