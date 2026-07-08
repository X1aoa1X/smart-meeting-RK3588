#!/bin/bash
# build.sh — 编译 yolo_core pybind11 扩展 (RK3588 板端一键构建)
#
# 用法:
#   ./build.sh                 # Release 构建并安装到 demo/
#   ./build.sh Debug           # Debug 构建
#   ./build.sh --no-install    # 不拷贝 .so 到 demo/
#
# 产物: build/yolo_core.cpython-3XX-aarch64-linux-gnu.so
# 依赖: librknnrt.so, OpenCV 4.x, pybind11 (pip install pybind11),
#       cmake (pip install cmake), Python3 开发头文件
set -e

BUILD_TYPE="${1:-Release}"
NO_INSTALL=0
if [ "$1" = "--no-install" ]; then NO_INSTALL=1; BUILD_TYPE="Release"; fi
if [ "$2" = "--no-install" ]; then NO_INSTALL=1; fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# 确保 cmake / pybind11 在 PATH 中 (pip --user 安装的情况)
export PATH="$HOME/.local/bin:$PATH"

echo "========================================"
echo "  build yolo_core (pybind11 extension)"
echo "========================================"
echo "BUILD_TYPE: ${BUILD_TYPE}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "========================================"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 定位 pybind11 的 cmake 目录 (pip --user 安装时不在默认搜索路径)
PYBIND11_CMAKE_DIR=$("${Python3_EXECUTABLE:-python3}" -m pybind11 --cmakedir 2>/dev/null || true)
if [ -z "${PYBIND11_CMAKE_DIR}" ]; then
    PYBIND11_CMAKE_DIR="$HOME/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11"
fi
echo "pybind11 cmake dir: ${PYBIND11_CMAKE_DIR}"

cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
      -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" \
      "${SCRIPT_DIR}"

cmake --build . -j4

echo ""
echo "✅ Build success: ${BUILD_DIR}/yolo_core*.so"
echo ""

# 自动拷贝到 demo/ 目录, 方便 import yolo_core
if [ "${NO_INSTALL}" -eq 0 ]; then
    DEMO_DIR="${SCRIPT_DIR}/../../demo"
    for so in "${BUILD_DIR}"/yolo_core*.so; do
        if [ -f "${so}" ]; then
            cp -v "${so}" "${DEMO_DIR}/"
            echo "📦 Installed: ${DEMO_DIR}/$(basename ${so})"
        fi
    done
fi

echo ""
echo "Usage (from demo/):"
echo "  python3 -c 'import yolo_core; print(yolo_core.__version__)'"
echo "  python3 demos/fusion_tracker.py"
