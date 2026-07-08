#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# 停止所有服务
# ──────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/data"

echo "停止所有服务..."

# ── 1. 停止 fusion_tracker ────────────────────────────────────────────
if [ -f "$LOG_DIR/fusion_tracker.pid" ]; then
    PID=$(cat "$LOG_DIR/fusion_tracker.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        echo "  fusion_tracker (PID $PID) 已停止"
    fi
    rm -f "$LOG_DIR/fusion_tracker.pid"
fi

# 回退: 按进程名杀
pkill -f "fusion_tracker.py" 2>/dev/null || true

# ── 2. 停止 Streamlit ────────────────────────────────────────────────
if [ -f "$LOG_DIR/streamlit.pid" ]; then
    PID=$(cat "$LOG_DIR/streamlit.pid")
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null || true
        echo "  Streamlit (PID $PID) 已停止"
    fi
    rm -f "$LOG_DIR/streamlit.pid"
fi

pkill -f "streamlit run" 2>/dev/null || true

echo "所有服务已停止"
