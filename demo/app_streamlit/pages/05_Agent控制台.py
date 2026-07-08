"""Agent 控制台 — LLM Agent TTS 播报状态与手动控制。

核心定位: 查看 Agent 状态 + 手动触发播报 + 配置开关。
通过 HTTP API 与 fusion_tracker 运行时通信，不直接操作硬件。

设计原则（来自 add_LLM.md）:
  - Streamlit 只显示 Agent 状态和发送显式用户命令
  - 页面自动刷新绝不触发 LLM 调用
  - 所有"自动判断"逻辑在 fusion_tracker 后端执行，Streamlit 只是控制面板

用法:
  streamlit run app_streamlit/Home.py
  # 然后导航到 "🤖 Agent 控制台"
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
    page_title="Agent 控制台",
    page_icon="🤖",
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


# ═══════════════════════════════════════════════════════════════════════════
# API 通信辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def _api_get(path: str, timeout: float = 5.0) -> dict | None:
    try:
        url = f"{RUNTIME_API}{path}"
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _api_post(path: str, body: dict | None = None, timeout: float = 5.0) -> dict | None:
    try:
        url = f"{RUNTIME_API}{path}"
        data = json.dumps(body or {}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _post_agent_command(cmd_type: str, meeting_id: int | None = None,
                        **params) -> bool:
    """发送 Agent 命令到 Runtime API。"""
    body = {**params}
    if meeting_id is not None:
        body["meeting_id"] = meeting_id
    result = _api_post(f"/api/agent/{cmd_type}", body=body)
    return result is not None and result.get("ok", False)


# ═══════════════════════════════════════════════════════════════════════════
# 页面状态
# ═══════════════════════════════════════════════════════════════════════════

if "agent_auto_refresh" not in st.session_state:
    st.session_state.agent_auto_refresh = True
if "agent_refresh_interval" not in st.session_state:
    st.session_state.agent_refresh_interval = 3


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar — 控制
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="office-section-label">Agent 控制</div>', unsafe_allow_html=True)

    st.checkbox("自动刷新", key="agent_auto_refresh")
    st.slider("刷新间隔 (秒)", 1, 10, key="agent_refresh_interval")

    st.divider()

    # ── 手动触发按钮 ──
    st.markdown('<div class="office-section-label">手动触发</div>', unsafe_allow_html=True)

    # 获取当前 meeting_id
    status = _api_get("/api/status")
    meeting_id = None
    if status and status.get("ok"):
        meeting_id = status.get("data", {}).get("meeting_id")

    if meeting_id:
        st.caption(f"当前会议 ID: {meeting_id}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 总结最近 3 分钟", use_container_width=True):
                ok = _post_agent_command("summarize", meeting_id=meeting_id, minutes=3)
                if ok:
                    st.success("已触发阶段总结")
                else:
                    st.error("触发失败")

        with col2:
            if st.button("📅 提醒进入下一议题", use_container_width=True):
                ok = _post_agent_command("agenda", meeting_id=meeting_id)
                if ok:
                    st.success("已触发议题提醒")
                else:
                    st.error("触发失败")

        if st.button("📊 播报系统状态", use_container_width=True):
            ok = _post_agent_command("status", meeting_id=meeting_id)
            if ok:
                st.success("已触发状态播报")
            else:
                st.error("触发失败")

    else:
        st.info("等待会议开始…")

    st.divider()

    # ── 自定义文本 ──
    st.markdown('<div class="office-section-label">自定义播报</div>', unsafe_allow_html=True)
    custom_text = st.text_input("播报文本", placeholder="输入要播报的文字…", key="agent_custom_text")
    if st.button("🔊 播报", use_container_width=True, disabled=not custom_text):
        if custom_text and meeting_id:
            ok = _post_agent_command("custom_tts", meeting_id=meeting_id, text=custom_text)
            if ok:
                st.success("已触发播报")
            else:
                st.error("播报失败")

    st.divider()

    # ── 导航 ──
    render_sidebar_nav("agent")


# ═══════════════════════════════════════════════════════════════════════════
# 主区域
# ═══════════════════════════════════════════════════════════════════════════

render_page_header("Agent 控制台", "查看 LLM Agent 播报状态、决策审计与手动触发入口；页面自动刷新不会触发 LLM 调用。", "AI Assistant")

# ── 数据加载 ──────────────────────────────────────────────────────────────

# Agent 状态（从 /api/status 获取）
status = _api_get("/api/status")
agent_info = {}
if status and status.get("ok"):
    agent_info = status.get("data", {}).get("agent", {})

# Agent 决策记录（从独立 API 获取）
decisions_raw = _api_get(f"/api/agent/decisions?limit=50"
                         + (f"&meeting_id={meeting_id}" if meeting_id else ""))
tts_events_raw = _api_get(f"/api/agent/tts_events?limit=50"
                          + (f"&meeting_id={meeting_id}" if meeting_id else ""))

# ── 状态指标卡 ────────────────────────────────────────────────────────────

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    enabled = agent_info.get("enabled", True)
    st.metric("Agent 状态", "✅ 启用" if enabled else "⏸ 禁用")

with col2:
    muted = agent_info.get("muted", False)
    st.metric("静音", "🔇 已静音" if muted else "🔊 正常")

with col3:
    llm_calls = agent_info.get("llm_calls_this_meeting", 0)
    max_calls = 20
    st.metric("LLM 调用", f"{llm_calls}/{max_calls}")

with col4:
    suppressed = agent_info.get("suppressed_count", 0)
    st.metric("抑制决策", suppressed)

with col5:
    last_tts = agent_info.get("last_tts_at", 0)
    if last_tts > 0:
        ago = int(time.time() - last_tts)
        st.metric("上次播报", f"{ago}s 前")
    else:
        st.metric("上次播报", "—")

# ── 最近决策 ──────────────────────────────────────────────────────────────

st.subheader("📊 最近 Agent 决策")

decisions = []
if decisions_raw and decisions_raw.get("ok"):
    decisions = decisions_raw.get("data", {}).get("decisions", [])

if decisions:
    import pandas as pd
    rows = []
    for d in decisions[:20]:
        rows.append({
            "时间": d.get("created_at", "")[:19] if d.get("created_at") else "",
            "触发类型": d.get("trigger_type", ""),
            "优先级": d.get("priority", 0),
            "LLM": "🤖" if d.get("llm_used") else "📋",
            "决策": "✅ 播报" if d.get("decision") == "spoken" else "🚫 抑制",
            "播报文本": (d.get("final_text") or "")[:50],
            "抑制原因": d.get("suppressed_reason") or "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("暂无 Agent 决策记录")

# ── 最近 TTS 播报 ─────────────────────────────────────────────────────────

st.subheader("🔊 最近 TTS 播报")

tts_events = []
if tts_events_raw and tts_events_raw.get("ok"):
    tts_events = tts_events_raw.get("data", {}).get("tts_events", [])

if tts_events:
    import pandas as pd
    rows = []
    for e in tts_events[:20]:
        rows.append({
            "时间": e.get("created_at", "")[:19] if e.get("created_at") else "",
            "文本": e.get("text", "")[:60],
            "来源": e.get("source", ""),
            "优先级": e.get("priority", 0),
            "状态": e.get("status", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("暂无 TTS 播报记录")

# ── 配置信息（GUI 编辑） ──────────────────────────────────────────────────

config_path = os.path.join(_PROJECT_ROOT, "configs", "agent_policy.json")

# 触发器中文名映射
_TRIGGER_LABELS = {
    "meeting_started": "会议开始",
    "meeting_ended": "会议结束",
    "speaker_confirmed": "发言人确认",
    "speaker_overtime": "发言人超时",
    "silence_timeout": "静默超时",
    "agenda_timeout": "议题超时",
    "identity_lost": "身份丢失",
    "system_error": "系统错误",
    "tracking_started": "追踪开始",
    "tracking_paused": "追踪暂停",
    "tracking_lost": "目标丢失",
    "host_locked_speaker": "主持人锁定",
    "host_unlocked_speaker": "主持人解锁",
    "manual_summary": "手动总结",
    "manual_agenda": "手动议题提醒",
    "manual_status": "手动状态播报",
    "speaker_switched": "发言人切换",
}
_TEMPLATE_OPTIONS = [
    "(无)", "meeting_started", "meeting_ended", "speaker_confirmed",
    "speaker_overtime", "silence_timeout", "agenda_timeout", "identity_lost",
    "tracking_started", "tracking_paused", "tracking_lost",
    "host_locked_speaker", "host_unlocked_speaker", "manual_system_status",
]
_TONE_OPTIONS = ["polite", "neutral", "concise", "warm"]

with st.expander("⚙️ Agent 策略配置", expanded=False):
    # 读取当前策略文件
    policy = None
    try:
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                policy = json.load(f)
        else:
            st.warning(f"配置文件不存在: `{config_path}`")
    except Exception as e:
        st.error(f"读取配置失败: {e}")

    if policy is not None:
        st.caption(f"📁 `{config_path}`")
        if st.session_state.get("agent_auto_refresh"):
            st.caption("💡 编辑时建议在侧栏关闭「自动刷新」以免输入被打断")

        # ── 顶层开关 ──
        st.markdown("**顶层开关**")
        tc1, tc2 = st.columns(2)
        with tc1:
            policy["enabled"] = st.checkbox(
                "Agent 总开关", value=bool(policy.get("enabled", True)),
                key="policy_enabled")
        with tc2:
            policy["demo_mode"] = st.checkbox(
                "演示模式", value=bool(policy.get("demo_mode", False)),
                help="演示模式下触发条件更宽松，方便展示效果",
                key="policy_demo_mode")

        st.divider()

        # ── 全局速率限制 ──
        st.markdown("**全局速率限制**")
        g = policy.setdefault("global", {})
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            g["min_tts_interval_sec"] = st.number_input(
                "TTS 最小间隔 (秒)", 0, 600,
                value=int(g.get("min_tts_interval_sec", 45)),
                key="p_g_min_tts")
            g["max_pending_tts"] = st.number_input(
                "最大挂起 TTS 数", 0, 20,
                value=int(g.get("max_pending_tts", 3)),
                key="p_g_max_pending")
        with gc2:
            g["min_llm_interval_sec"] = st.number_input(
                "LLM 最小间隔 (秒)", 0, 3600,
                value=int(g.get("min_llm_interval_sec", 90)),
                key="p_g_min_llm")
            g["max_tts_per_5min"] = st.number_input(
                "5 分钟最大 TTS 次数", 0, 50,
                value=int(g.get("max_tts_per_5min", 5)),
                key="p_g_max_tts_5min")
        with gc3:
            g["max_llm_calls_per_meeting"] = st.number_input(
                "每场会议最大 LLM 调用", 0, 200,
                value=int(g.get("max_llm_calls_per_meeting", 20)),
                key="p_g_max_llm")

        st.divider()

        # ── 语音设置 ──
        st.markdown("**语音设置**")
        sp = policy.setdefault("speech", {})
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            sp["max_chars"] = st.number_input(
                "播报最大字符数", 10, 500,
                value=int(sp.get("max_chars", 60)),
                key="p_s_max_chars")
        with sc2:
            current_tone = sp.get("tone", "polite")
            tone_idx = (_TONE_OPTIONS.index(current_tone)
                        if current_tone in _TONE_OPTIONS else 0)
            sp["tone"] = st.selectbox(
                "语气", _TONE_OPTIONS, index=tone_idx,
                key="p_s_tone")
        with sc3:
            sp["fallback_on_llm_error"] = st.checkbox(
                "LLM 失败时回退模板",
                value=bool(sp.get("fallback_on_llm_error", True)),
                key="p_s_fallback")

        st.divider()

        # ── 触发器规则 ──
        st.markdown("**触发器规则**")
        triggers = policy.setdefault("triggers", {})

        for trig_key, t in triggers.items():
            label = _TRIGGER_LABELS.get(trig_key, trig_key)
            with st.container(border=True):
                trc1, trc2, trc3, trc4, trc5 = st.columns([2, 1, 1, 1, 2])
                with trc1:
                    t["enabled"] = st.checkbox(
                        f"{label}", value=bool(t.get("enabled", True)),
                        key=f"p_t_en_{trig_key}")
                with trc2:
                    t["requires_llm"] = st.checkbox(
                        "需要 LLM", value=bool(t.get("requires_llm", False)),
                        key=f"p_t_llm_{trig_key}")
                with trc3:
                    t["fixed_rule_enabled"] = st.checkbox(
                        "固定规则播报",
                        value=bool(t.get("fixed_rule_enabled", True)),
                        help="关闭后，该触发器不会产生模板播报（fixed_rule TTS）；"
                             "若已开启 LLM 则仅 LLM 路径生效，LLM 失败也不回退模板",
                        key=f"p_t_fr_{trig_key}")
                with trc4:
                    t["cooldown_sec"] = st.number_input(
                        "冷却 (秒)", 0, 3600,
                        value=int(t.get("cooldown_sec", 0)),
                        key=f"p_t_cd_{trig_key}")
                with trc5:
                    cur_tid = t.get("template_id")
                    if cur_tid is None:
                        tid_idx = 0
                    elif cur_tid in _TEMPLATE_OPTIONS:
                        tid_idx = _TEMPLATE_OPTIONS.index(cur_tid)
                    else:
                        tid_idx = len(_TEMPLATE_OPTIONS)
                        _TEMPLATE_OPTIONS.append(cur_tid)
                    sel_tid = st.selectbox(
                        "模板 ID", _TEMPLATE_OPTIONS, index=tid_idx,
                        key=f"p_t_tid_{trig_key}")
                    t["template_id"] = None if sel_tid == "(无)" else sel_tid

                # 触发器特有字段
                if "threshold_sec" in t:
                    if isinstance(t["threshold_sec"], dict):
                        # silence_timeout: 按会议阶段配置
                        st.caption("静默阈值（按会议阶段，秒）")
                        phases = ["opening", "report", "discussion", "qa", "free_talk"]
                        existing = t["threshold_sec"]
                        ph_cols = st.columns(5)
                        for i, phase in enumerate(phases):
                            with ph_cols[i]:
                                existing[phase] = st.number_input(
                                    phase, 0, 600,
                                    value=int(existing.get(phase, 30)),
                                    key=f"p_t_th_{trig_key}_{phase}")
                    else:
                        t["threshold_sec"] = st.number_input(
                            "阈值 (秒)", 0, 7200,
                            value=int(t["threshold_sec"]),
                            key=f"p_t_th_{trig_key}")

                if "stable_sec" in t:
                    t["stable_sec"] = st.number_input(
                        "稳定时间 (秒)", 0.0, 60.0,
                        value=float(t["stable_sec"]), step=0.5,
                        key=f"p_t_st_{trig_key}")

        st.divider()

        # ── 保存 / 重置 ──
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("💾 保存到文件", use_container_width=True, type="primary"):
                try:
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(policy, f, ensure_ascii=False, indent=2)
                    st.success("配置已写入 agent_policy.json")
                    st.session_state["agent_policy_saved"] = True
                except Exception as e:
                    st.error(f"保存失败: {e}")
        with bc2:
            if st.button("↺ 重新读取文件", use_container_width=True):
                for k in list(st.session_state.keys()):
                    if k.startswith("p_t_") or k.startswith("p_g_") \
                            or k.startswith("p_s_") or k == "policy_enabled" \
                            or k == "policy_demo_mode":
                        del st.session_state[k]
                st.rerun()

        if st.session_state.get("agent_policy_saved"):
            st.info("⚠️ 配置已保存到磁盘，需重启 fusion_tracker 后端才能让新策略生效。")

        # ── 原始 JSON 预览 ──
        with st.expander("查看当前 JSON", expanded=False):
            st.json(policy)

# ── 自动刷新 ──────────────────────────────────────────────────────────────

if st.session_state.agent_auto_refresh:
    time.sleep(st.session_state.agent_refresh_interval)
    st.rerun()
