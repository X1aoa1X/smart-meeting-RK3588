#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# 启动 fusion_tracker 运行时（需要 root 权限控制 PWM 舵机）
#
# 用法:
#   sudo ./scripts/start_fusion_tracker.sh
#   sudo ./scripts/start_fusion_tracker.sh --debug-ui   # 带 Qt 调试界面
#
# 环境要求:
#   - root 权限 (PWM sysfs)
#   - DISPLAY=:0 (Qt GUI)
#   - XAUTHORITY 指向 GDM 的 X 授权文件
#   - PYTHONPATH 包含用户安装的包
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── 环境变量 ────────────────────────────────────────────────────────────
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"
export PYTHONPATH="${PYTHONPATH:-/home/elf/.local/lib/python3.10/site-packages}"

# ── 切换到项目目录 ──────────────────────────────────────────────────────
cd "$PROJECT_DIR"

# ── 日志 ────────────────────────────────────────────────────────────────
LOG_DIR="$PROJECT_DIR/data"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/fusion_tracker_$(date +%Y%m%d).log"

echo "══════════════════════════════════════════════════" | tee -a "$LOG_FILE"
echo "fusion_tracker 启动: $(date)" | tee -a "$LOG_FILE"
echo "  DISPLAY=$DISPLAY" | tee -a "$LOG_FILE"
echo "  XAUTHORITY=$XAUTHORITY" | tee -a "$LOG_FILE"
echo "  PYTHONPATH=$PYTHONPATH" | tee -a "$LOG_FILE"
echo "══════════════════════════════════════════════════" | tee -a "$LOG_FILE"

# ── 启动 ────────────────────────────────────────────────────────────────
exec python3 demos/fusion_tracker.py "$@" 2>&1 | tee -a "$LOG_FILE"
