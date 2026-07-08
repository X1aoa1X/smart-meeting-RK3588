#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# 启动 Streamlit 主持人控制台
#
# 用法:
#   ./scripts/start_streamlit.sh
#
# 环境要求:
#   - fusion_tracker 已启动 (否则控制台显示 Runtime 离线)
#   - streamlit 已安装 (pip install streamlit)
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 环境变量 ────────────────────────────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-/home/elf/.local/lib/python3.10/site-packages}"
export RUNTIME_HOST="${RUNTIME_HOST:-127.0.0.1}"
export RUNTIME_PORT="${RUNTIME_PORT:-8800}"

# ── 切换到项目目录 ──────────────────────────────────────────────────────
cd "$PROJECT_DIR"

# ── 日志 ────────────────────────────────────────────────────────────────
LOG_DIR="$PROJECT_DIR/data"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/streamlit_$(date +%Y%m%d).log"

echo "══════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "Streamlit 启动: $(date)" | tee -a "$LOG_FILE"
echo "  RUNTIME_API=http://${RUNTIME_HOST}:${RUNTIME_PORT}" | tee -a "$LOG_FILE"
echo "══════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ── 启动 (监听所有网络接口，方便笔记本访问) ──────────────────────────────
exec /home/elf/.local/bin/streamlit run app_streamlit/Home.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    2>&1 | tee -a "$LOG_FILE"
