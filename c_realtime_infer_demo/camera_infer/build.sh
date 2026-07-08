#!/bin/bash
# build.sh — 编译 yolo_camera (RK3588 板端一键构建)
#
# 用法:
#   ./build.sh                        # 使用默认编译器和路径
#   ./build.sh Release                # Release 模式
#   CXX=aarch64-linux-gnu-g++ ./build.sh  # 交叉编译
#
# 依赖:
#   - librknnrt.so (Rockchip NPU SDK)
#   - OpenCV 4.x  (libopencv_core, libopencv_imgproc, libopencv_videoio, libopencv_highgui)

set -e

BUILD_TYPE="${1:-Release}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# C/C++ 编译器 (可环境变量覆盖)
CXX="${CXX:-g++}"

# 路径检测 (自动探测 vendored 和系统路径)
if [ -z "${RKNN_HEADER}" ]; then
    for candidate in \
        /usr/include/rknn \
        "${SCRIPT_DIR}/../rknn_model_zoo-2.1.0/3rdparty/rknpu2/include"; do
        if [ -f "${candidate}/rknn_api.h" ]; then
            RKNN_HEADER="${candidate}"
            break
        fi
    done
fi
RKNN_HEADER="${RKNN_HEADER:-/usr/include/rknn}"

if [ -z "${RKNN_LIB}" ]; then
    for candidate in \
        /usr/lib/librknnrt.so \
        /usr/lib/aarch64-linux-gnu/librknnrt.so; do
        if [ -f "${candidate}" ]; then
            RKNN_LIB="${candidate}"
            break
        fi
    done
fi
RKNN_LIB="${RKNN_LIB:-/usr/lib/librknnrt.so}"

# OpenCV 路径 (自动探测: 系统 → vendored rknn_model_zoo)
_VENDORED_OCV="${SCRIPT_DIR}/../rknn_model_zoo-2.1.0/3rdparty/opencv/opencv-linux-aarch64"
if [ -z "${OPENCV_CFLAGS}" ]; then
    if pkg-config --cflags opencv4 &>/dev/null; then
        OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
    elif [ -d "${_VENDORED_OCV}/include" ]; then
        OPENCV_CFLAGS="-I${_VENDORED_OCV}/include"
    else
        OPENCV_CFLAGS="-I/usr/include/opencv4"
    fi
fi
if [ -z "${OPENCV_LIBS}" ]; then
    if pkg-config --libs opencv4 &>/dev/null; then
        OPENCV_LIBS="$(pkg-config --libs opencv4)"
    elif [ -d "${_VENDORED_OCV}/lib" ]; then
        OPENCV_LIBS="-L${_VENDORED_OCV}/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui"
    else
        OPENCV_LIBS="-lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui"
    fi
fi

echo "========================================"
echo "  build yolo_camera"
echo "========================================"
echo "CXX:        ${CXX}"
echo "BUILD_TYPE: ${BUILD_TYPE}"
echo "RKNN_H:     ${RKNN_HEADER}"
echo "RKNN_LIB:   ${RKNN_LIB}"
echo "OPENCV_CF:  ${OPENCV_CFLAGS}"
echo "OPENCV_LD:  ${OPENCV_LIBS}"
echo "========================================"

mkdir -p "${BUILD_DIR}"

"${CXX}" -std=c++17 -O2 -Wall \
    "${SCRIPT_DIR}/yolo_camera.cc" \
    -o "${BUILD_DIR}/yolo_camera" \
    -I"${RKNN_HEADER}" \
    ${OPENCV_CFLAGS} \
    "${RKNN_LIB}" \
    ${OPENCV_LIBS} \
    -lpthread -ldl

echo ""
echo "✅ Build success: ${BUILD_DIR}/yolo_camera"
echo ""
echo "Run:"
echo "  cd ${SCRIPT_DIR}/.. && ./camera_infer/build/yolo_camera ./best.rknn [/dev/video21]"
