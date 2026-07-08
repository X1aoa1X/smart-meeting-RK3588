"""会议控制台 — 主持人实时导播与状态监控面板。

核心定位: 会议导播台 + 状态监控台 + 事件标记台。
通过 HTTP API 与 fusion_tracker 运行时通信，不直接操作硬件。

用法:
  streamlit run app_streamlit/Home.py
  # 然后导航到 "🎛️ 会议控制台"
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

import streamlit as st

# ---------------------------------------------------------------------------
# Shared presentation layer — concise office UI styling
# ---------------------------------------------------------------------------
_APP_STREAMLIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_STREAMLIT_DIR not in sys.path:
    sys.path.insert(0, _APP_STREAMLIT_DIR)
from ui_style import apply_office_theme, render_page_header, render_sidebar_nav, render_stepper


# ── 配置 ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="会议控制台",
    page_icon="🎛️",
    layout="wide",
)
apply_office_theme()

# ── 确保能找到项目根目录的模块 ────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Runtime API 地址 ──────────────────────────────────────────────────────
RUNTIME_HOST = os.environ.get("RUNTIME_HOST", "127.0.0.1")
RUNTIME_PORT = int(os.environ.get("RUNTIME_PORT", "8800"))
RUNTIME_API = f"http://{RUNTIME_HOST}:{RUNTIME_PORT}"

# ── 初始化数据库 ──────────────────────────────────────────────────────────
try:
    from storage import init as storage_init
    storage_init()
except Exception as e:
    st.warning(f"数据库初始化失败: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# API 通信辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _api_get(path: str, timeout: float = 3.0) -> dict | None:
    """发送 GET 请求到 Runtime API。"""
    try:
        url = f"{RUNTIME_API}{path}"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return None
    except Exception:
        return None


def _api_post(path: str, body: dict | None = None, timeout: float = 5.0) -> dict | None:
    """发送 POST 请求到 Runtime API。"""
    try:
        url = f"{RUNTIME_API}{path}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return None
    except Exception:
        return None


def _api_ok(resp: dict | None) -> bool:
    """检查 API 响应是否成功。"""
    return resp is not None and resp.get("ok", False)


# ═══════════════════════════════════════════════════════════════════════════
# UI 渲染辅助
# ═══════════════════════════════════════════════════════════════════════════

STATE_ICONS = {
    "IDLE": "👂",
    "AWAIT": "⏳",
    "TRACKING": "🎯",
}

STATE_COLORS = {
    "IDLE": "green",
    "AWAIT": "orange",
    "TRACKING": "blue",
}

MEETING_STATE_LABELS = {
    "planned": "📋 未开始",
    "in_progress": "🔴 进行中",
    "completed": "✅ 已结束",
    "cancelled": "❌ 已取消",
}

SOURCE_LABELS = {
    "april_tag": "🏷️ AprilTag",
    "manual": "👤 手动指定",
    "unknown": "❓ 未知",
}

NOTE_TYPE_ICONS = {
    "评委问题": "❓",
    "重点结论": "💡",
    "待办事项": "📋",
    "系统异常": "⚠️",
    "主持人备注": "📝",
}


def _format_duration(seconds: float) -> str:
    """将秒数格式化为 MM:SS。"""
    if seconds < 0:
        return "--:--"
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def _status_badge(label: str, color: str = "gray") -> str:
    """生成带颜色的状态徽章 HTML。"""
    return f'<span style="color:{color};font-weight:bold;">{label}</span>'


# ═══════════════════════════════════════════════════════════════════════════
# 主页面
# ═══════════════════════════════════════════════════════════════════════════

render_page_header("会议控制台", "主持人实时导播控制面板：统一管理会议状态、发言人识别、画面控制、事件时间轴与备注。", "Live Control")

# ── 会话状态初始化 ──────────────────────────────────────────────────────────
if "control_auto_refresh" not in st.session_state:
    st.session_state.control_auto_refresh = True
if "control_refresh_interval" not in st.session_state:
    st.session_state.control_refresh_interval = 2
if "control_last_command" not in st.session_state:
    st.session_state.control_last_command = None
if "control_command_status" not in st.session_state:
    st.session_state.control_command_status = None

# LLM 助手状态
if "llm_chat_history" not in st.session_state:
    st.session_state.llm_chat_history = []
if "llm_available" not in st.session_state:
    st.session_state.llm_available = None

# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — 导播控制 + 系统健康
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="office-section-label">设置</div>', unsafe_allow_html=True)

    auto_refresh = st.checkbox(
        "自动刷新", value=st.session_state.control_auto_refresh,
        key="chk_auto_refresh")
    st.session_state.control_auto_refresh = auto_refresh

    refresh_interval = st.slider(
        "刷新间隔 (秒)", 1, 10,
        value=st.session_state.control_refresh_interval,
        key="sld_refresh_interval")
    st.session_state.control_refresh_interval = refresh_interval

    st.divider()

    # ── 导播控制 ──────────────────────────────────────────────────────────
    st.markdown('<div class="office-section-label">导播控制</div>', unsafe_allow_html=True)

    # 先获取状态以判断按钮是否可用
    status = _api_get("/api/status")

    runtime_online = status is not None and status.get("ok")
    meeting_state = status.get("data", {}).get("meeting_state", "") if runtime_online else None
    tracking = status.get("data", {}).get("tracking", {}) if runtime_online else {}
    tracking_active = tracking.get("tracking_active", False) if runtime_online else False
    tracking_paused = tracking.get("tracking_paused", False) if runtime_online else False
    speaker_locked = tracking.get("speaker_locked", False) if runtime_online else False

    in_meeting = meeting_state == "in_progress"

    col_ctrl_1, col_ctrl_2 = st.columns(2)

    with col_ctrl_1:
        if st.button("▶ 开始追踪", disabled=not runtime_online or not in_meeting or tracking_active,
                     width="stretch", key="btn_start_tracking"):
            resp = _api_post("/api/control/start_tracking")
            st.session_state.control_last_command = "开始追踪"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

        if st.button("⏸ 暂停追踪", disabled=not runtime_online or not tracking_active or tracking_paused,
                     width="stretch", key="btn_pause_tracking"):
            resp = _api_post("/api/meeting/pause")
            st.session_state.control_last_command = "暂停追踪"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

        if st.button("▶ 恢复追踪", disabled=not runtime_online or not tracking_paused,
                     width="stretch", key="btn_resume_tracking"):
            resp = _api_post("/api/meeting/resume")
            st.session_state.control_last_command = "恢复追踪"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

        if st.button("⬅ 云台回中", disabled=not runtime_online,
                     width="stretch", key="btn_recenter"):
            resp = _api_post("/api/control/recenter")
            st.session_state.control_last_command = "云台回中"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

        # TTS 测试按钮 — 无需会议状态，可随时测试
        st.divider()
        st.caption("🔊 语音播报测试")
        if st.button("🔊 测试 TTS 播报", disabled=not runtime_online,
                     width="stretch", key="btn_tts_test"):
            resp = _api_post("/api/tts/test")
            st.session_state.control_last_command = "TTS 测试"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

    with col_ctrl_2:
        if st.button("🔒 锁定发言人", disabled=not runtime_online or not tracking_active or speaker_locked,
                     width="stretch", key="btn_lock"):
            resp = _api_post("/api/control/lock_speaker")
            st.session_state.control_last_command = "锁定发言人"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

        if st.button("🔓 解除锁定", disabled=not runtime_online or not speaker_locked,
                     width="stretch", key="btn_unlock"):
            resp = _api_post("/api/control/unlock_speaker")
            st.session_state.control_last_command = "解除锁定"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

        if st.button("👤 手动指定…", disabled=not runtime_online,
                     width="stretch", key="btn_manual"):
            # 点击后展开选择框
            st.session_state.show_manual_select = not st.session_state.get("show_manual_select", False)

    # ── 手动指定发言人 ────────────────────────────────────────────────────
    if st.session_state.get("show_manual_select", False):
        participants = status.get("data", {}).get("participants", []) if runtime_online else []
        tag_options = [p.get("tag_id", "") for p in participants if p.get("tag_id")]
        if tag_options:
            selected_tag = st.selectbox("选择发言人", tag_options, key="sel_manual_speaker")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                if st.button("✅ 确认指定", width="stretch", key="btn_confirm_manual"):
                    resp = _api_post("/api/control/manual_speaker", {"tag_id": selected_tag})
                    st.session_state.control_last_command = f"手动指定: {selected_tag}"
                    st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"
                    st.session_state.show_manual_select = False
            with col_m2:
                if st.button("❌ 取消", width="stretch", key="btn_cancel_manual"):
                    st.session_state.show_manual_select = False
        else:
            st.caption("无可用参与人")
            if st.button("关闭", key="btn_close_manual"):
                st.session_state.show_manual_select = False

    st.divider()

    # ── 画面控制 ──────────────────────────────────────────────────────────
    st.markdown('<div class="office-section-label">画面控制</div>', unsafe_allow_html=True)

    overlay = status.get("data", {}).get("overlay", {}) if runtime_online else {}
    overlay_enabled = overlay.get("enabled", True)
    show_debug = overlay.get("show_debug", False)

    if st.button(
        f"{'👁 隐藏名片' if overlay_enabled else '👁 显示名片'}",
        disabled=not runtime_online,
        width="stretch",
        key="btn_toggle_overlay"
    ):
        resp = _api_post("/api/control/set_overlay", {"enabled": not overlay_enabled})
        st.session_state.control_last_command = "切换名片显示"
        st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

    if st.button(
        f"{'🐛 隐藏调试框' if show_debug else '🐛 显示调试框'}",
        disabled=not runtime_online,
        width="stretch",
        key="btn_toggle_debug"
    ):
        resp = _api_post("/api/control/set_overlay", {"show_debug": not show_debug})
        st.session_state.control_last_command = "切换调试框"
        st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

    # ── RTSP 推流控制 ──────────────────────────────────────────────────────
    rtsp_status = tracking.get("rtsp_status", "stopped") if runtime_online else "stopped"
    streaming_active = rtsp_status == "OK"

    if st.button(
        f"{'📡 停止推流' if streaming_active else '📡 开始推流'}",
        disabled=not runtime_online,
        width="stretch",
        key="btn_toggle_stream"
    ):
        endpoint = "/api/control/stop_stream" if streaming_active else "/api/control/start_stream"
        resp = _api_post(endpoint)
        st.session_state.control_last_command = "停止推流" if streaming_active else "开始推流"
        st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"

    st.divider()

    # ── VAD 音频设备选择 ────────────────────────────────────────────────────
    st.caption("🔊 VAD 音频设备")

    # 获取设备列表
    devices_resp = _api_get("/api/audio/devices")
    available_devices = []
    device_names = []
    current_vad_device = tracking.get("vad_device", "hw:1,0") if runtime_online else "hw:1,0"
    selected_index = 0

    if devices_resp and devices_resp.get("ok"):
        devices_list = devices_resp.get("data", {}).get("devices", [])
        available_devices = [
            f"{d['card_name']} ({d['name']})"
            for d in devices_list
        ]
        device_names = [d["name"] for d in devices_list]
        try:
            selected_index = device_names.index(current_vad_device)
        except (ValueError, IndexError):
            selected_index = 0

    if available_devices:
        selected_label = st.selectbox(
            "选择录音设备",
            available_devices,
            index=min(selected_index, len(available_devices) - 1),
            disabled=not runtime_online,
            key="sel_vad_device",
            label_visibility="collapsed",
        )

        # 初始化 session state for last applied device
        if "last_applied_vad_device" not in st.session_state:
            st.session_state.last_applied_vad_device = current_vad_device

        selected_idx = available_devices.index(selected_label)
        selected_device_name = device_names[selected_idx]

        if (selected_device_name != st.session_state.last_applied_vad_device
                and selected_device_name != current_vad_device):
            if st.button("🔄 应用设备", disabled=not runtime_online,
                         key="btn_apply_vad_device"):
                resp = _api_post("/api/control/set_vad_device",
                               {"device": selected_device_name})
                if _api_ok(resp):
                    st.session_state.last_applied_vad_device = selected_device_name
                    st.session_state.control_last_command = f"切换音频设备: {selected_device_name}"
                    st.session_state.control_command_status = "✅ 已发送"
                    st.rerun()
                else:
                    st.session_state.control_command_status = "❌ 失败"
        elif selected_device_name == current_vad_device:
            st.caption(f"当前: {selected_label}")
    else:
        st.caption(f"当前设备: {current_vad_device}")
        st.caption("(未能枚举设备或设备繁忙)")

    st.divider()

    # ── 系统健康 ──────────────────────────────────────────────────────────
    st.markdown('<div class="office-section-label">系统状态</div>', unsafe_allow_html=True)

    if runtime_online:
        st.success(f"Runtime: ✅ 在线 ({RUNTIME_HOST}:{RUNTIME_PORT})")
    else:
        st.error("Runtime: ❌ 离线")

    # 检查 DB
    try:
        from storage.db import session_scope
        from sqlalchemy import text
        with session_scope() as session:
            session.execute(text("SELECT 1"))
        st.success("DB: ✅ 正常")
    except Exception:
        st.error("DB: ❌ 异常")

    if runtime_online and tracking:
        rtsp_status = tracking.get("rtsp_status", "stopped")
        if rtsp_status == "OK":
            st.success("RTSP: ✅ 正常")
        elif rtsp_status == "stopped":
            st.info("RTSP: ⏸ 未推流")
        else:
            st.error(f"RTSP: ❌ {rtsp_status}")
    else:
        st.info("RTSP: --")

    # 上一次命令状态
    if st.session_state.control_last_command:
        st.caption(f"上次操作: {st.session_state.control_last_command} "
                  f"{st.session_state.control_command_status or ''}")

    st.divider()

    # ── 会议控制 ──────────────────────────────────────────────────────────
    st.markdown('<div class="office-section-label">会议控制</div>', unsafe_allow_html=True)

    # 从 DB 获取可选的会议列表 (planned + in_progress)
    # 同时包含 in_progress 以处理 tracker 重启后 DB 残留的进行中会议
    available_meetings = []
    try:
        from storage.db import session_scope
        from storage.repo import MeetingRepo
        with session_scope() as session:
            mr = MeetingRepo(session)
            planned = mr.list_all(status="planned")
            in_progress = mr.list_all(status="in_progress")
            available_meetings = planned + in_progress
    except Exception as e:
        st.caption(f"⚠️ 加载会议列表失败: {e}")

    if not in_meeting:
        # 可以开始新会议 (或恢复残留的 in_progress 会议)
        meeting_options = {}
        for m in available_meetings:
            label = f"[{m.id}] {m.name}"
            if m.status == "in_progress":
                label += " ⚠️进行中"
            meeting_options[label] = (m.id, m.status)

        if meeting_options:
            selected_meeting_label = st.selectbox(
                "选择要开始的会议",
                list(meeting_options.keys()),
                key="sel_start_meeting")
            selected_meeting_id, selected_meeting_status = meeting_options[selected_meeting_label]

            is_stale = selected_meeting_status == "in_progress"
            if is_stale:
                st.warning(
                    "⚠️ 此会议状态为「进行中」，可能是 tracker 重启后残留。"
                    "点击下方按钮将先结束旧记录再重新开始。"
                )

            if st.button("▶ 开始会议" if not is_stale else "🔄 结束并重新开始",
                         disabled=not runtime_online,
                         width="stretch", key="btn_start_meeting"):
                if is_stale:
                    # 先在 DB 中结束残留的 in_progress 会议
                    try:
                        from storage.db import session_scope
                        from storage.repo import MeetingRepo
                        with session_scope() as session:
                            stale = MeetingRepo(session).get_by_id(selected_meeting_id)
                            if stale and stale.status == "in_progress":
                                MeetingRepo(session).end_meeting(stale)
                    except Exception:
                        pass
                resp = _api_post("/api/meeting/start", {"meeting_id": selected_meeting_id})
                st.session_state.control_last_command = f"开始会议 (id={selected_meeting_id})"
                st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"
        else:
            st.caption("没有可开始的会议 (planned / in_progress)")
            st.caption("请先在「会前准备」中创建会议")
    else:
        if st.button("⏹ 结束会议", disabled=not runtime_online,
                     width="stretch", key="btn_end_meeting", type="primary"):
            resp = _api_post("/api/meeting/end")
            st.session_state.control_last_command = "结束会议"
            st.session_state.control_command_status = "✅ 已发送" if _api_ok(resp) else "❌ 失败"


    st.divider()
    render_sidebar_nav("control")

# ═══════════════════════════════════════════════════════════════════════════
# Main Area — 状态显示 (自动刷新)
# ═══════════════════════════════════════════════════════════════════════════

# 如果 runtime 不在线，显示大警告
if not runtime_online:
    st.error("## ⚠️ Runtime 离线")
    st.markdown(f"""
    **fusion_tracker 运行时未响应** (`{RUNTIME_API}/api/status`)

    请确认:
    1. `fusion_tracker` 是否正在运行
    2. 控制 API 是否已启动 (端口 {RUNTIME_PORT})
    3. 网络连接是否正常
    """)
    st.stop()

# ── 解析当前状态 ──────────────────────────────────────────────────────────
data = status.get("data", {}) if runtime_online else {}

runtime_state = data.get("runtime_state", "IDLE")
meeting_name = data.get("meeting_name", "--")
meeting_status_label = MEETING_STATE_LABELS.get(meeting_state, meeting_state or "--")
speaker = data.get("current_speaker") or {}
tracking_info = data.get("tracking", {})
overlay_info = data.get("overlay", {})
participants = data.get("participants", [])

# ── LLM 可用性检查（首次加载时检查一次）──────────────────────────────────────
if st.session_state.llm_available is None and runtime_online:
    llm_status = _api_get("/api/llm/status")
    st.session_state.llm_available = (
        llm_status is not None
        and llm_status.get("ok")
        and llm_status.get("data", {}).get("available", False)
    )

# ═══════════════════════════════════════════════════════════════════════════
# 状态指标卡片 (4 列)
# ═══════════════════════════════════════════════════════════════════════════

st.subheader("📊 实时状态")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**👤 当前发言人**")
    if speaker.get("name"):
        speaker_name = speaker.get("name", "--")
        speaker_role = speaker.get("role", "")
        speaker_tag = speaker.get("tag_id", "")
        speaker_source = speaker.get("source", "unknown")
        speaker_dur = speaker.get("speaking_duration", 0)

        source_label = SOURCE_LABELS.get(speaker_source, speaker_source)
        st.markdown(f"### {speaker_name}")
        if speaker_role:
            st.caption(f"**{speaker_role}**")
        st.caption(f"{speaker_tag} · {source_label}")
        st.metric("发言时长", _format_duration(speaker_dur))
    else:
        st.markdown("### --")
        st.caption("无发言人")

with col2:
    st.markdown("**🎯 系统状态**")
    state_icon = STATE_ICONS.get(runtime_state, "")
    state_color = STATE_COLORS.get(runtime_state, "gray")
    st.markdown(f"### {state_icon} {runtime_state}")
    st.caption(meeting_status_label)
    if tracking_info.get("speaker_locked"):
        st.caption("🔒 发言人已锁定")
    if tracking_info.get("tracking_paused"):
        st.caption("⏸ 追踪已暂停")
    if not tracking_info.get("tracking_active"):
        st.caption("⏹ 追踪未启动")

with col3:
    st.markdown("**🔊 音频 / 视觉**")
    doa = tracking_info.get("doa_angle", 0)
    pan = tracking_info.get("pan_angle", 0)
    tilt = tracking_info.get("tilt_angle", 0)
    fps = tracking_info.get("yolo_fps", 0)

    st.metric("DOA 角度", f"{doa:.1f}°")
    st.metric("云台 H / V", f"{pan:.1f}° / {tilt:.1f}°")
    st.metric("YOLO FPS", f"{fps:.1f}")

    vad_enabled = tracking_info.get("vad_enabled", False)
    vad_speech = tracking_info.get("vad_is_speech", False)
    if vad_enabled:
        st.caption(f"VAD: {'🔊 语音' if vad_speech else '🔇 静音'}")
    else:
        st.caption("VAD: 已禁用 (硬件模式)")

with col4:
    st.markdown("**📡 RTSP**")
    rtsp_status = tracking_info.get("rtsp_status", "stopped")
    rtsp_url = tracking_info.get("rtsp_url", "")

    if rtsp_status == "OK":
        st.success("推流中")
        if rtsp_url:
            st.code(rtsp_url, language=None)
        st.caption(f"名片叠加: {'👁 显示' if overlay_info.get('enabled', True) else '🚫 隐藏'}")
        st.caption(f"调试框: {'🐛 显示' if overlay_info.get('show_debug', False) else '🚫 隐藏'}")
    elif rtsp_status == "stopped":
        st.info("未推流")
    else:
        st.error(f"异常: {rtsp_status}")

    tag_fps = tracking_info.get("tag_detect_fps", 0)
    if tag_fps > 0:
        st.metric("Tag FPS", f"{tag_fps:.1f}")

# ═══════════════════════════════════════════════════════════════════════════
# 参会人员列表
# ═══════════════════════════════════════════════════════════════════════════

st.subheader(f"📋 参会人员 ({len(participants)}人)")

if participants:
    # 构建表格数据
    table_data = []
    for p in participants:
        tag_id = p.get("tag_id", "--")
        name = p.get("name", "--")
        role = p.get("role", "")
        detected = p.get("detected", False)
        is_current = p.get("is_current_speaker", False)
        duration = p.get("accumulated_duration", 0)
        count = p.get("speaking_count", 0)

        # 状态判断
        if is_current:
            status_display = "🔴 发言中"
        elif detected:
            status_display = "✅ 已到场"
        else:
            status_display = "⬜ 未检测"

        table_data.append({
            "Tag": tag_id,
            "姓名": name,
            "角色": role,
            "状态": status_display,
            "累计发言": _format_duration(duration),
            "发言次数": count,
        })

    import pandas as pd
    df = pd.DataFrame(table_data)

    # 为当前发言人高亮行
    def _highlight_current(row):
        if row["状态"] == "🔴 发言中":
            return ["background-color: #fff3cd; font-weight: bold;"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_current, axis=1)
    st.dataframe(styled, width="stretch", hide_index=True)
else:
    st.info("没有参会人员数据（请先在「会前准备」中导入人员名单）")

# ═══════════════════════════════════════════════════════════════════════════
# 事件时间轴 + 主持人备注
# ═══════════════════════════════════════════════════════════════════════════

col_timeline, col_notes = st.columns([3, 2])

with col_timeline:
    st.subheader("📜 事件时间轴")

    meeting_id = data.get("meeting_id")
    if meeting_id:
        events_resp = _api_get(f"/api/events?meeting_id={meeting_id}&minutes=120")
        if events_resp and events_resp.get("ok"):
            events_data = events_resp.get("data", {})
            raw_events = events_data.get("events", [])
            raw_notes = events_data.get("notes", [])

            # 合并事件和备注，按时间排序
            timeline_items = []

            for evt in raw_events:
                etype = evt.get("event_type", "")
                ts = evt.get("timestamp", "")
                payload = evt.get("payload") or {}

                # 转换事件类型为友好显示
                icon_map = {
                    "meeting_started": ("📌", "会议开始"),
                    "meeting_ended": ("🏁", "会议结束"),
                    "meeting_paused": ("⏸", "追踪暂停"),
                    "meeting_resumed": ("▶", "追踪恢复"),
                    "speaker_started": ("🔴", f"{payload.get('name', '?')} 开始发言 ({payload.get('source', '')})"),
                    "speaker_ended": ("🔵", f"{payload.get('name', '?')} 发言结束"),
                    "speaker_switched": ("🔄", f"{payload.get('prev_name', '?')} → {payload.get('name', '?')}"),
                    "speaker_lost": ("⚠️", f"标签丢失: {payload.get('name', '?')}"),
                    "speaker_reidentified": ("✅", f"标签恢复: {payload.get('name', '?')}"),
                    "state_changed": ("🔀", f"状态: {payload.get('from_state', '?')} → {payload.get('to_state', '?')}"),
                    "host_locked_speaker": ("🔒", "主持人锁定发言人"),
                    "host_unlocked_speaker": ("🔓", "主持人解除锁定"),
                    "speaker_override": ("👤", f"手动指定发言人: {payload.get('tag_id', '?')}"),
                    "host_note_added": ("📝", f"主持人备注: [{payload.get('note_type', '')}] {payload.get('content', '')[:50]}"),
                    "servo_moved": ("🔧", f"舵机移动: {payload.get('angle', payload.get('h_angle', '?'))}°"),
                }

                icon, label = icon_map.get(etype, ("📎", etype))
                timeline_items.append({
                    "ts_sort": ts,
                    "display": f"{icon} {label}",
                    "type": "event",
                })

            for note in raw_notes:
                note_type = note.get("note_type", "")
                note_icon = NOTE_TYPE_ICONS.get(note_type, "📝")
                related = note.get("related_speaker", "")
                content = note.get("content", "")
                ts = note.get("timestamp", "")

                timeline_items.append({
                    "ts_sort": ts,
                    "display": f"{note_icon} **[{note_type}]** {content}",
                    "type": "note",
                })

            # 按时间排序 (最新在前)
            timeline_items.sort(key=lambda x: x.get("ts_sort", ""), reverse=True)

            if timeline_items:
                # 格式化时间显示
                for item in timeline_items[:50]:  # 只显示最近 50 条
                    ts = item.get("ts_sort", "")
                    try:
                        ts_dt = datetime.fromisoformat(ts)
                        ts_display = ts_dt.strftime("%H:%M:%S")
                    except Exception:
                        ts_display = ts[:19] if len(ts) > 19 else ts

                    if item["type"] == "note":
                        st.markdown(f"`{ts_display}` {item['display']}")
                    else:
                        st.caption(f"`{ts_display}` {item['display']}")
            else:
                st.caption("暂无事件记录")
        else:
            st.caption("无法加载事件数据")
    else:
        st.info("没有活跃会议 — 请先开始会议")

with col_notes:
    st.subheader("📝 主持人备注")

    # ── 添加新备注表单 ────────────────────────────────────────────────────
    with st.expander("➕ 添加备注", expanded=False):
        if meeting_id:
            note_type = st.selectbox(
                "备注类型",
                ["评委问题", "重点结论", "待办事项", "系统异常", "主持人备注"],
                key="sel_note_type")

            # 相关发言人选择
            speaker_options = ["(无)"] + [
                f"{p.get('tag_id', '')} {p.get('name', '')}"
                for p in participants if p.get("tag_id")
            ]
            selected_speaker_display = st.selectbox(
                "相关发言人 (可选)",
                speaker_options,
                key="sel_note_speaker")
            related_speaker = None
            if selected_speaker_display != "(无)":
                # 提取 tag_id
                related_speaker = selected_speaker_display.split(" ")[0] if selected_speaker_display else None

            note_content = st.text_area("备注内容", key="txt_note_content",
                                       placeholder="输入备注内容…")

            if st.button("💾 保存备注", disabled=not note_content.strip(),
                         width="stretch", key="btn_save_note"):
                resp = _api_post("/api/host_note", {
                    "meeting_id": meeting_id,
                    "note_type": note_type,
                    "content": note_content.strip(),
                    "related_speaker": related_speaker,
                })
                if _api_ok(resp):
                    st.success("✅ 备注已保存")
                    st.session_state.control_last_command = f"添加备注: {note_type}"
                else:
                    st.error("❌ 保存失败")
        else:
            st.caption("请先开始会议")

    # ── 已有备注列表 ──────────────────────────────────────────────────────
    if meeting_id:
        notes_resp = _api_get(f"/api/events?meeting_id={meeting_id}&minutes=120")
        if notes_resp and notes_resp.get("ok"):
            existing_notes = notes_resp.get("data", {}).get("notes", [])
            if existing_notes:
                st.caption(f"已有 {len(existing_notes)} 条备注:")
                for note in reversed(existing_notes):
                    note_type = note.get("note_type", "")
                    note_icon = NOTE_TYPE_ICONS.get(note_type, "📝")
                    related = note.get("related_speaker", "")
                    content = note.get("content", "")
                    ts = note.get("timestamp", "")
                    try:
                        ts_dt = datetime.fromisoformat(ts)
                        ts_display = ts_dt.strftime("%H:%M:%S")
                    except Exception:
                        ts_display = ts[:19]

                    with st.container(border=True):
                        st.markdown(f"{note_icon} **[{note_type}]** `{ts_display}`")
                        if related:
                            st.caption(f"相关: {related}")
                        st.text(content)
            else:
                st.caption("暂无备注")
    else:
        st.caption("--")

# ═══════════════════════════════════════════════════════════════════════════
# LLM 助手 (AI Assistant)
# ═══════════════════════════════════════════════════════════════════════════

st.divider()
st.subheader("🤖 LLM 助手")

if not st.session_state.llm_available:
    # 不可用：显示配置说明
    st.info(
        "💡 **LLM 助手未启用**\n\n"
        "请在启动 `fusion_tracker` 前设置环境变量:\n\n"
        "```bash\n"
        "export DEEPSEEK_API_KEY=\"your-api-key-here\"\n"
        "export DEEPSEEK_BASE_URL=\"https://api.deepseek.com\"  # 可选，这是默认值\n"
        "```\n\n"
        "设置后重启 runtime 即可使用 AI 助手功能。\n\n"
        "支持的功能:\n"
        "- 💬 会中问答（谁在发言？系统状态如何？）\n"
        "- 📝 会后摘要生成\n"
        "- 📋 待办事项提取\n"
        "- 🔧 系统故障诊断"
    )

    # 手动刷新按钮
    if st.button("🔄 重新检查", key="btn_recheck_llm", width="small"):
        st.session_state.llm_available = None
        st.rerun()

else:
    # 可用：显示对话界面
    meeting_id = data.get("meeting_id")

    if not meeting_id:
        st.info("请先开始一场会议以使用 LLM 助手")
    else:
        col_chat, col_quick = st.columns([3, 1])

        with col_chat:
            st.caption("💬 AI 对话")

            # 显示对话历史
            for msg in st.session_state.llm_chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])

            # 输入框
            question = st.chat_input("输入问题，如：谁发言最多？", key="llm_chat_input")
            if question:
                # 添加用户消息
                st.session_state.llm_chat_history.append({"role": "user", "content": question})

                with st.spinner("🤔 AI 思考中…（最长 90 秒）"):
                    resp = _api_post("/api/llm/chat", {
                        "meeting_id": meeting_id,
                        "question": question,
                    }, timeout=90.0)

                if resp and resp.get("ok"):
                    answer = resp.get("data", {}).get("answer", "")
                    st.session_state.llm_chat_history.append({"role": "assistant", "content": answer})
                else:
                    error_msg = resp.get("error", "未知错误") if resp else "API 无响应（请确认 runtime 正在运行）"
                    st.session_state.llm_chat_history.append(
                        {"role": "assistant", "content": f"❌ **调用失败**: {error_msg}"}
                    )

                st.rerun()

        with col_quick:
            st.caption("⚡ 快捷提问")

            quick_questions = [
                ("👥 谁发言最多？", "谁发言最多？各发言人累计时长排名如何？请列出统计数据。"),
                ("📊 系统状态如何？", "根据当前系统状态，追踪系统运行是否正常？DOA角度、YOLO帧率、RTSP推流等指标有什么需要注意的？"),
                ("📜 最近发生了什么？", "最近10分钟内发生了什么重要事件？包括发言人切换、系统状态变更等。"),
                ("📝 查看备注", "总结一下当前会议中主持人记录的所有备注和标记。"),
            ]

            for label, q in quick_questions:
                if st.button(label, disabled=not meeting_id, use_container_width=True,
                             key=f"quick_{label[:4]}"):
                    st.session_state.llm_chat_history.append({"role": "user", "content": q})

                    with st.spinner("🤔 AI 思考中…"):
                        resp = _api_post("/api/llm/chat", {
                            "meeting_id": meeting_id,
                            "question": q,
                        }, timeout=90.0)

                    if resp and resp.get("ok"):
                        answer = resp.get("data", {}).get("answer", "")
                        st.session_state.llm_chat_history.append({"role": "assistant", "content": answer})
                    else:
                        error_msg = resp.get("error", "未知错误") if resp else "API 无响应"
                        st.session_state.llm_chat_history.append(
                            {"role": "assistant", "content": f"❌ **调用失败**: {error_msg}"}
                        )

                    st.rerun()

            st.divider()

            if st.button("🗑 清空对话", use_container_width=True):
                st.session_state.llm_chat_history = []
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# 自动刷新逻辑
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.control_auto_refresh:
    time.sleep(st.session_state.control_refresh_interval)
    st.rerun()
