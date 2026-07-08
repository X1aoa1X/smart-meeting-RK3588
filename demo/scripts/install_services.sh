#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# 安装 systemd 服务 + 桌面自启动（开机自动运行）
#
# 用法:
#   sudo ./scripts/install_services.sh
#
# 安装内容:
#   1. systemd 服务: smart-meeting-runtime.service (fusion_tracker)
#   2. systemd 服务: smart-meeting-streamlit.service (Streamlit)
#   3. 桌面自启动: ~/.config/autostart/fusion_tracker.desktop (备选)
#
# 安装后:
#   sudo systemctl start smart-meeting-runtime
#   sudo systemctl start smart-meeting-streamlit
#   sudo systemctl enable smart-meeting-runtime    # 开机自启
#   sudo systemctl enable smart-meeting-streamlit  # 开机自启
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════╗"
echo "║  安装 systemd 服务                            ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. 安装 systemd 服务 ───────────────────────────────────────────────
echo ""
echo "[1/3] 安装 systemd 服务文件..."

cp "$SCRIPT_DIR/smart-meeting-runtime.service" /etc/systemd/system/
cp "$SCRIPT_DIR/smart-meeting-streamlit.service" /etc/systemd/system/

echo "  ✅ smart-meeting-runtime.service"
echo "  ✅ smart-meeting-streamlit.service"

# ── 2. 创建数据目录 + 预建 DB (确保 elf 可写) ───────────────────────────
echo ""
echo "[2/3] 创建数据目录..."
mkdir -p "$PROJECT_DIR/data"
chown -R elf:elf "$PROJECT_DIR/data" 2>/dev/null || true
chmod 775 "$PROJECT_DIR/data"
# 预建空 DB 文件，确保 owner 为 elf（避免 root 建库后 Streamlit 写 -wal 失败）
DB_FILE="$PROJECT_DIR/data/meeting_tracker.db"
if [ ! -f "$DB_FILE" ]; then
    su -s /bin/bash elf -c "touch '$DB_FILE'" 2>/dev/null || true
    echo "  ✅ DB 文件已预建: $DB_FILE"
fi

# ── 3. 安装桌面自启动（备选方案）────────────────────────────────────────
echo ""
echo "[3/3] 安装桌面自启动（备选）..."
mkdir -p /home/elf/.config/autostart
cp "$SCRIPT_DIR/fusion_tracker.desktop" /home/elf/.config/autostart/
chown elf:elf /home/elf/.config/autostart/fusion_tracker.desktop 2>/dev/null || true
echo "  ✅ ~/.config/autostart/fusion_tracker.desktop"

# ── 重新加载 systemd ────────────────────────────────────────────────────
systemctl daemon-reload

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  安装完成                                       ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  手动启动:                                      ║"
echo "║    sudo systemctl start smart-meeting-runtime   ║"
echo "║    sudo systemctl start smart-meeting-streamlit ║"
echo "║                                                ║"
echo "║  开机自启:                                      ║"
echo "║    sudo systemctl enable smart-meeting-runtime   ║"
echo "║    sudo systemctl enable smart-meeting-streamlit ║"
echo "║                                                ║"
echo "║  查看状态:                                      ║"
echo "║    sudo systemctl status smart-meeting-runtime   ║"
echo "║    sudo systemctl status smart-meeting-streamlit ║"
echo "║                                                ║"
echo "║  查看日志:                                      ║"
echo "║    journalctl -u smart-meeting-runtime -f       ║"
echo "║    tail -f data/fusion_tracker.log              ║"
echo "╚══════════════════════════════════════════════╝"
