#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# 一键启动: fusion_tracker + Streamlit 控制台
#
# 用法:
#   sudo ./scripts/start_all.sh
#
# 启动后:
#   - fusion_tracker Qt 界面显示在 :0
#   - 控制 API:  http://127.0.0.1:8800
#   - Streamlit:  http://<设备IP>:8501
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/data"
mkdir -p "$LOG_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║  智会追声 — 全部服务启动                      ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 0. 防止重复启动 ───────────────────────────────────────────────────
if [ -f "$LOG_DIR/fusion_tracker.pid" ]; then
    OLD_PID=$(cat "$LOG_DIR/fusion_tracker.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "  ⚠️  fusion_tracker 已在运行 (PID: $OLD_PID)"
        echo "  如需重启，请先运行: sudo ./scripts/stop_all.sh"
        exit 1
    fi
fi
if [ -f "$LOG_DIR/streamlit.pid" ]; then
    OLD_PID=$(cat "$LOG_DIR/streamlit.pid")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "  ⚠️  Streamlit 已在运行 (PID: $OLD_PID)"
        echo "  如需重启，请先运行: ./scripts/stop_all.sh"
        exit 1
    fi
fi

echo "  启动 fusion_tracker (root, 后台运行)..."
echo "  启动 Streamlit (elf, 后台运行)..."
echo ""

# ── 1. 启动 fusion_tracker (后台) ──────────────────────────────────────
sudo -E env \
    DISPLAY=:0 \
    XAUTHORITY=/run/user/1000/gdm/Xauthority \
    PYTHONPATH="/home/elf/.local/lib/python3.10/site-packages" \
    python3 "$PROJECT_DIR/demos/fusion_tracker.py" \
    > "$LOG_DIR/fusion_tracker.log" 2>&1 &

FUSION_PID=$!
echo "  fusion_tracker PID: $FUSION_PID"

# 等待 API 就绪
echo "  等待控制 API 就绪..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8800/api/status > /dev/null 2>&1; then
        echo "  ✅ 控制 API 已就绪 (${i}s)"
        break
    fi
    sleep 1
done

# ── 2. 启动 Streamlit (后台) ───────────────────────────────────────────
export PYTHONPATH="/home/elf/.local/lib/python3.10/site-packages"
/home/elf/.local/bin/streamlit run "$PROJECT_DIR/app_streamlit/Home.py" \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > "$LOG_DIR/streamlit.log" 2>&1 &

STREAMLIT_PID=$!
echo "  Streamlit PID: $STREAMLIT_PID"
echo ""

# ── 保存 PID 文件 ──────────────────────────────────────────────────────
echo "$FUSION_PID" > "$LOG_DIR/fusion_tracker.pid"
echo "$STREAMLIT_PID" > "$LOG_DIR/streamlit.pid"

# ── 获取本机 IP ────────────────────────────────────────────────────────
DEVICE_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")

echo "╔══════════════════════════════════════════════╗"
echo "║  服务已启动                                     ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  Qt 调试界面:   DISPLAY=:0                     ║"
echo "║  控制 API:      http://127.0.0.1:8800          ║"
echo "║  Streamlit:     http://${DEVICE_IP}:8501       ║"
echo "║  测试命令:      curl http://127.0.0.1:8800/api/status"
echo "╠══════════════════════════════════════════════╣"
echo "║  停止服务:      ./scripts/stop_all.sh          ║"
echo "╚══════════════════════════════════════════════╝"
