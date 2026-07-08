"""会议记录与总结 — 会后时间轴导出、发言统计、报告生成。

回顾已结束的会议：查看发言时间轴、发言人统计、系统事件和主持人备注，
支持导出 CSV / Excel / JSON 三种格式。

用法:
  streamlit run app_streamlit/Home.py
  # 然后导航到 "📊 会议记录与总结"
"""

import io
import json
import os
import sys
import urllib.request
import urllib.error
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.environ.get(
    "MEETING_TRACKER_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Shared presentation layer — concise office UI styling
# ---------------------------------------------------------------------------
_APP_STREAMLIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_STREAMLIT_DIR not in sys.path:
    sys.path.insert(0, _APP_STREAMLIT_DIR)
from ui_style import apply_office_theme, render_page_header, render_sidebar_nav, render_stepper


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="会议记录与总结",
    page_icon="📊",
    layout="wide",
)
apply_office_theme()

# ---------------------------------------------------------------------------
# Database initialization (cached once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def _init_storage():
    """Initialize the storage module once. Must be called before any DB access."""
    from storage import init
    init()
    return True


def _get_db_path() -> str:
    from storage.db import db_path
    return db_path()


_storage_ready = _init_storage()

# ── Runtime API 地址（用于 LLM 调用）─────────────────────────────────────────
RUNTIME_HOST = os.environ.get("RUNTIME_HOST", "127.0.0.1")
RUNTIME_PORT = int(os.environ.get("RUNTIME_PORT", "8800"))
RUNTIME_API = f"http://{RUNTIME_HOST}:{RUNTIME_PORT}"


def _api_post(path: str, body: dict | None = None, timeout: float = 60.0) -> dict | None:
    """发送 POST 请求到 Runtime API（LLM 调用需要较长超时）。"""
    try:
        url = f"{RUNTIME_API}{path}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        return None
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVENT_ICON_MAP = {
    "meeting_started":         ("📌", "会议开始"),
    "meeting_ended":           ("🏁", "会议结束"),
    "meeting_paused":          ("⏸",  "追踪暂停"),
    "meeting_resumed":         ("▶",   "追踪恢复"),
    "speaker_started":         ("🎤", "发言人开始"),
    "speaker_ended":           ("🔇", "发言人结束"),
    "speaker_switched":        ("🔄", "发言人切换"),
    "speaker_lost":            ("⚠️",  "标签丢失"),
    "speaker_reidentified":    ("✅", "标签恢复"),
    "state_changed":           ("🔀", "状态变更"),
    "host_locked_speaker":     ("🔒", "锁定发言人"),
    "host_unlocked_speaker":   ("🔓", "解除锁定"),
    "speaker_override":        ("👤", "手动指定"),
    "host_note_added":         ("📝", "主持人备注"),
    "servo_moved":             ("🔧", "舵机移动"),
    "tracking_started":        ("🎯", "开始追踪"),
    "tracking_stopped":        ("⏹",  "停止追踪"),
    "tag_detected":            ("🏷",  "检测到标签"),
    "tag_lost":                ("💨", "标签丢失"),
    "tracking_lost":           ("❓", "追踪丢失"),
    "tracking_recovered":      ("🔄", "追踪恢复"),
    "system_warning":          ("⚠️",  "系统警告"),
    "rtsp_error":              ("📡", "RTSP 异常"),
}

# ── LLM 生成结果缓存 ────────────────────────────────────────────────────────
if "llm_summary" not in st.session_state:
    st.session_state.llm_summary = None       # str | None
if "llm_action_items" not in st.session_state:
    st.session_state.llm_action_items = None  # str | None
if "llm_judge_questions" not in st.session_state:
    st.session_state.llm_judge_questions = None  # str | None
if "llm_system_report" not in st.session_state:
    st.session_state.llm_system_report = None  # str | None
if "llm_generating" not in st.session_state:
    st.session_state.llm_generating = False


NOTE_TYPE_ICONS = {
    "评委问题":   "❓",
    "重点结论":   "💡",
    "待办事项":   "📋",
    "系统异常":   "⚠️",
    "主持人备注": "📝",
}


# ---------------------------------------------------------------------------
# Helpers — formatting
# ---------------------------------------------------------------------------
def _fmt_duration(seconds: float | None) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    if seconds is None or seconds < 0:
        return "--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_datetime(dt: datetime | None) -> str:
    """Format a datetime as HH:MM:SS or '--'."""
    if dt is None:
        return "--"
    return dt.strftime("%H:%M:%S")


def _fmt_datetime_full(dt: datetime | None) -> str:
    """Format a datetime as YYYY-MM-DD HH:MM:SS or '--'."""
    if dt is None:
        return "--"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Helpers — data loading
# ---------------------------------------------------------------------------
def _load_meeting_data(meeting_id: int) -> dict | None:
    """Load all data for a meeting from the database.

    Returns a dict with keys: meeting, segments, events, notes, participants, summary.
    Returns None if the meeting doesn't exist.
    """
    from storage.db import session_scope
    from storage.repo import (
        MeetingRepo, SpeakerSegmentRepo, EventRepo, HostNoteRepo, ParticipantRepo,
    )
    from core.meeting_service import MeetingService

    # Load meeting + related data in a single session
    with session_scope() as s:
        meeting = MeetingRepo(s).get_by_id(meeting_id)
        if meeting is None:
            return None

        segments = SpeakerSegmentRepo(s).get_segments_for_meeting(meeting_id)
        events = EventRepo(s).get_for_meeting(meeting_id)
        notes = HostNoteRepo(s).get_notes_for_meeting(meeting_id)
        participants = {p.tag_id: p for p in ParticipantRepo(s).list_all()}

    # Summary uses its own session_scope internally
    svc = MeetingService()
    summary = svc.get_meeting_summary(meeting_id)
    if "error" in summary:
        summary = None

    return {
        "meeting": meeting,
        "segments": segments,
        "events": events,
        "notes": notes,
        "participants": participants,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Helpers — timeline building
# ---------------------------------------------------------------------------
def _build_segment_items(segments: list, participants: dict) -> list[dict]:
    """Convert speaker segments into timeline items (start + end pairs)."""
    items = []
    for seg in segments:
        p = participants.get(seg.speaker_tag_id) if seg.speaker_tag_id else None
        speaker_name = seg.speaker_name or (p.name if p else "未知发言人")
        role = seg.role or (p.role if p else "")
        source = seg.source or "unknown"

        # Start item
        items.append({
            "sort_key": seg.start_time,
            "time_str": _fmt_datetime(seg.start_time),
            "type": "发言开始",
            "speaker": speaker_name,
            "tag_id": seg.speaker_tag_id or "",
            "role": role,
            "source": source,
            "duration_seconds": seg.duration_seconds or 0,
            "confidence": seg.confidence or 0,
            "content": f"🎤 **{speaker_name}** ({role})" if role else f"🎤 **{speaker_name}**",
        })

        # End item
        if seg.end_time:
            dur_str = _fmt_duration(seg.duration_seconds)
            items.append({
                "sort_key": seg.end_time,
                "time_str": _fmt_datetime(seg.end_time),
                "type": "发言结束",
                "speaker": speaker_name,
                "tag_id": seg.speaker_tag_id or "",
                "role": role,
                "source": source,
                "duration_seconds": seg.duration_seconds or 0,
                "confidence": seg.confidence or 0,
                "content": f"🔇 **{speaker_name}** 结束 (持续 {dur_str})",
            })

    return items


def _build_event_items(events: list) -> list[dict]:
    """Convert system events into timeline items."""
    items = []
    for evt in events:
        # Parse payload
        payload = {}
        if evt.payload_json:
            try:
                payload = json.loads(evt.payload_json)
            except (json.JSONDecodeError, TypeError):
                payload = {"_raw": str(evt.payload_json)}

        icon, label = EVENT_ICON_MAP.get(evt.event_type, ("📎", evt.event_type))

        # Build a human-readable detail
        detail = _event_detail(evt.event_type, payload)

        # Extract speaker info from payload
        speaker = (
            payload.get("name")
            or payload.get("speaker_name")
            or payload.get("prev_name")
            or ""
        )

        items.append({
            "sort_key": evt.timestamp,
            "time_str": _fmt_datetime(evt.timestamp),
            "type": "系统事件",
            "speaker": speaker,
            "tag_id": payload.get("tag_id", "") or payload.get("prev_tag_id", ""),
            "role": payload.get("role", ""),
            "source": payload.get("source", ""),
            "duration_seconds": payload.get("duration_seconds", 0) or 0,
            "confidence": 0,
            "content": f"{icon} {label}{f': {detail}' if detail else ''}",
        })

    return items


def _event_detail(event_type: str, payload: dict) -> str:
    """Extract a short Chinese description from an event payload."""
    if event_type == "meeting_started":
        return f"{payload.get('meeting_name', '')} ({payload.get('location', '')})" if payload.get('location') else payload.get('meeting_name', '')
    elif event_type == "meeting_ended":
        dur = payload.get("duration_seconds")
        return f"时长 {_fmt_duration(dur)}" if dur else ""
    elif event_type == "meeting_paused":
        return payload.get("reason", "")
    elif event_type == "meeting_resumed":
        return payload.get("reason", "")
    elif event_type == "speaker_started":
        return f"{payload.get('name', '?')} ({payload.get('source', '')})"
    elif event_type == "speaker_ended":
        return payload.get("name", "?")
    elif event_type == "speaker_switched":
        prev = payload.get("prev_name", "?")
        curr = payload.get("name", "?")
        return f"{prev} → {curr}"
    elif event_type == "speaker_lost":
        return payload.get("name", payload.get("tag_id", "?"))
    elif event_type == "speaker_reidentified":
        return payload.get("name", payload.get("tag_id", "?"))
    elif event_type == "state_changed":
        frm = payload.get("from_state", "?")
        to = payload.get("to_state", "?")
        return f"{frm} → {to}"
    elif event_type == "host_locked_speaker":
        return payload.get("name", payload.get("tag_id", "?"))
    elif event_type == "host_unlocked_speaker":
        return ""
    elif event_type == "speaker_override":
        return payload.get("name", payload.get("tag_id", "?"))
    elif event_type == "host_note_added":
        return payload.get("note_type", "") + (f": {payload.get('content', '')[:40]}" if payload.get('content') else "")
    elif event_type == "servo_moved":
        pan = payload.get("pan", payload.get("angle", "?"))
        return f"角度={pan}°"
    elif event_type == "tag_detected":
        return payload.get("tag_id", "?")
    elif event_type == "phone_detected":
        return payload.get("status", "")
    # Generic: show the most useful payload field
    for key in ("reason", "message", "detail", "description"):
        if key in payload and payload[key]:
            return str(payload[key])[:80]
    return ""


def _build_note_items(notes: list) -> list[dict]:
    """Convert host notes into timeline items."""
    items = []
    for note in notes:
        icon = NOTE_TYPE_ICONS.get(note.note_type, "📝")
        related = f" ({note.related_speaker})" if note.related_speaker else ""

        items.append({
            "sort_key": note.timestamp,
            "time_str": _fmt_datetime(note.timestamp),
            "type": "主持人备注",
            "speaker": note.related_speaker or "",
            "tag_id": "",
            "role": "",
            "source": "",
            "duration_seconds": 0,
            "confidence": 0,
            "content": f"{icon} **[{note.note_type}]**{related} {note.content}",
        })

    return items


def _build_timeline(data: dict) -> list[dict]:
    """Build a merged chronological timeline from all data sources."""
    segments = data.get("segments", [])
    events = data.get("events", [])
    notes = data.get("notes", [])
    participants = data.get("participants", {})

    all_items = (
        _build_segment_items(segments, participants)
        + _build_event_items(events)
        + _build_note_items(notes)
    )
    all_items.sort(key=lambda x: x["sort_key"])
    return all_items


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def _generate_timeline_csv(items: list[dict]) -> bytes:
    """Generate a UTF-8 BOM CSV from timeline items."""
    rows = []
    for item in items:
        rows.append({
            "时间": item["time_str"],
            "类型": item["type"],
            "发言人": item["speaker"] or "",
            "Tag ID": item["tag_id"] or "",
            "角色": item["role"] or "",
            "识别来源": item["source"] or "",
            "持续秒数": item["duration_seconds"],
            "置信度": f"{item['confidence']:.2f}" if item.get("confidence") else "",
            "内容": item["content"],
        })
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def _generate_timeline_excel(
    items: list[dict],
    speaker_df: pd.DataFrame,
    meeting_meta: dict,
    meeting_id: int,
) -> bytes | None:
    """Generate a multi-sheet Excel report. Returns None if openpyxl is unavailable."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return None

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Merged Timeline
        timeline_rows = []
        for item in items:
            timeline_rows.append({
                "时间": item["time_str"],
                "类型": item["type"],
                "发言人": item["speaker"] or "",
                "Tag ID": item["tag_id"] or "",
                "角色": item["role"] or "",
                "识别来源": item["source"] or "",
                "持续秒数": item["duration_seconds"],
                "置信度": f"{item['confidence']:.2f}" if item.get("confidence") else "",
                "内容": item["content"],
            })
        pd.DataFrame(timeline_rows).to_excel(writer, sheet_name="合并时间轴", index=False)

        # Sheet 2: Speaker Statistics
        if not speaker_df.empty:
            speaker_df.to_excel(writer, sheet_name="发言统计", index=False)

        # Sheet 3: Meeting Info
        pd.DataFrame([meeting_meta]).to_excel(writer, sheet_name="会议信息", index=False)

    return buf.getvalue()


def _generate_full_json(
    meeting, segments: list, events: list, notes: list, summary: dict | None
) -> str:
    """Generate a complete JSON dump of all meeting data."""
    payload = {
        "exported_at": datetime.now().isoformat(),
        "meeting": {
            "id": meeting.id,
            "name": meeting.name,
            "location": meeting.location or "",
            "status": meeting.status,
            "start_time": meeting.start_time.isoformat() if meeting.start_time else None,
            "end_time": meeting.end_time.isoformat() if meeting.end_time else None,
            "description": meeting.description or "",
        },
        "summary": summary,
        "segments": [
            {
                "speaker_name": seg.speaker_name,
                "speaker_tag_id": seg.speaker_tag_id,
                "role": seg.role,
                "source": seg.source,
                "start_time": seg.start_time.isoformat() if seg.start_time else None,
                "end_time": seg.end_time.isoformat() if seg.end_time else None,
                "duration_seconds": seg.duration_seconds,
                "confidence": seg.confidence,
            }
            for seg in segments
        ],
        "events": [
            {
                "event_type": evt.event_type,
                "timestamp": evt.timestamp.isoformat() if evt.timestamp else None,
                "payload": json.loads(evt.payload_json) if evt.payload_json else None,
            }
            for evt in events
        ],
        "notes": [
            {
                "note_type": note.note_type,
                "content": note.content,
                "related_speaker": note.related_speaker,
                "timestamp": note.timestamp.isoformat() if note.timestamp else None,
            }
            for note in notes
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📊 会议记录")
    st.caption(f"数据库: `{_get_db_path()}`")

    st.divider()

    # ── Meeting selector ──
    st.subheader("选择会议")
    show_all = st.checkbox("显示所有状态", value=False, help="默认只显示已完成的会议")

    try:
        from storage.db import session_scope
        from storage.repo import MeetingRepo

        with session_scope() as s:
            mr = MeetingRepo(s)
            if show_all:
                meetings = mr.list_all()
            else:
                meetings = mr.list_all(status="completed")
    except Exception as e:
        st.error(f"加载会议列表失败: {e}")
        meetings = []

    if not meetings:
        st.info("暂无会议记录")
    else:
        meeting_options = {f"#{m.id}  {m.name}  [{m.status}]": m.id for m in meetings}
        selected_label = st.selectbox(
            "会议",
            options=list(meeting_options.keys()),
            key="timeline_meeting_selector",
        )
        selected_meeting_id = meeting_options[selected_label]

    st.divider()

    render_sidebar_nav("records")


# ═══════════════════════════════════════════════════════════════════════════════
# Main area
# ═══════════════════════════════════════════════════════════════════════════════
render_page_header("会议记录与总结", "回顾已结束会议，导出时间轴、查看发言统计，并生成会后摘要与系统报告。", "Post-meeting")

if not meetings:
    st.info("📭 暂无已完成的会议记录。请先在「会议控制台」中开始并结束一场会议。")
    st.stop()

# selected_meeting_id is set in sidebar above; safe to use here
try:
    _meeting_id = selected_meeting_id  # noqa: F821 — set in sidebar
except NameError:
    st.stop()

# ── Load data ────────────────────────────────────────────────────────────────
data = _load_meeting_data(_meeting_id)
if data is None:
    st.error(f"加载会议 #{_meeting_id} 失败")
    st.stop()

meeting = data["meeting"]
segments = data["segments"]
events = data["events"]
notes = data["notes"]
summary = data["summary"]

# ═══════════════════════════════════════════════════════════════════════════════
# Section A: Meeting Overview Cards
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader(f"📋 {meeting.name}")

# Metadata row
meta_cols = st.columns(3)
with meta_cols[0]:
    st.caption(f"📍 地点: {meeting.location or '—'}")
with meta_cols[1]:
    st.caption(f"📅 开始: {_fmt_datetime_full(meeting.start_time)}")
with meta_cols[2]:
    st.caption(f"🏁 结束: {_fmt_datetime_full(meeting.end_time)}")

st.divider()

# Metric cards
if summary:
    total_dur = summary.get("total_duration_seconds")
    total_speakers = summary.get("total_speakers", 0)
    total_switches = summary.get("total_switches", 0)
    total_segments = summary.get("total_segments", 0)
else:
    total_dur = None
    total_speakers = 0
    total_switches = 0
    total_segments = 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("⏱ 会议总时长", _fmt_duration(total_dur) if total_dur else "--")
with col2:
    st.metric("👥 发言人总数", total_speakers)
with col3:
    st.metric("🔄 切换次数", total_switches)
with col4:
    st.metric("💬 发言片段", total_segments)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section B: Speaker Statistics Table
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("👥 发言人统计")

if summary and summary.get("participant_stats"):
    total_secs = total_dur or 0
    rows = []
    for ps in summary["participant_stats"]:
        dur = ps.get("total_duration", 0) or 0
        pct = (dur / total_secs * 100) if total_secs > 0 else 0
        rows.append({
            "Tag ID": ps.get("tag_id") or "—",
            "姓名": ps.get("name", "未知"),
            "角色": ps.get("role") or "—",
            "累计发言": _fmt_duration(dur),
            "发言次数": ps.get("segment_count", 0),
            "时长占比": f"{pct:.1f}%",
        })
    df_speakers = pd.DataFrame(rows)
    st.dataframe(df_speakers, width="stretch", hide_index=True)
else:
    st.info("暂无发言统计数据")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section C: Merged Timeline
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📜 会议时间轴")

timeline_items = _build_timeline(data)

if timeline_items:
    # Build display dataframe
    tl_rows = []
    for item in timeline_items:
        tl_rows.append({
            "时间": item["time_str"],
            "类型": item["type"],
            "发言人": item["speaker"] or "—",
            "内容": item["content"],
        })
    df_timeline = pd.DataFrame(tl_rows)
    st.dataframe(df_timeline, width="stretch", hide_index=True)
    st.caption(f"共 {len(timeline_items)} 条记录")
else:
    st.info("暂无时间轴数据")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section D: Export Buttons
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📥 导出数据")

if not timeline_items:
    st.info("没有可导出的数据")
else:
    export_col1, export_col2, export_col3 = st.columns(3)

    # ── CSV Export ──
    csv_data = _generate_timeline_csv(timeline_items)
    with export_col1:
        st.download_button(
            "📥 下载时间轴 (CSV)",
            data=csv_data,
            file_name=f"timeline_meeting_{meeting.id}.csv",
            mime="text/csv",
            use_container_width=True,
            help="UTF-8 编码，可直接用 Excel 打开",
        )

    # ── Excel Export ──
    meeting_meta = {
        "会议名称": meeting.name,
        "地点": meeting.location or "",
        "状态": meeting.status,
        "开始时间": _fmt_datetime_full(meeting.start_time),
        "结束时间": _fmt_datetime_full(meeting.end_time),
        "总时长": _fmt_duration(total_dur) if total_dur else "--",
        "发言人总数": total_speakers,
        "切换次数": total_switches,
        "发言片段数": total_segments,
    }
    excel_data = _generate_timeline_excel(
        timeline_items,
        df_speakers if summary and summary.get("participant_stats") else pd.DataFrame(),
        meeting_meta,
        meeting.id,
    )
    with export_col2:
        if excel_data is not None:
            st.download_button(
                "📥 下载完整报告 (Excel)",
                data=excel_data,
                file_name=f"meeting_report_{meeting.id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                help="多工作表: 合并时间轴、发言统计、会议信息",
            )
        else:
            st.warning("⚠️ Excel 导出需要 `openpyxl` 库\n\n安装: `pip install openpyxl`")

    # ── JSON Export ──
    json_data = _generate_full_json(meeting, segments, events, notes, summary)
    with export_col3:
        st.download_button(
            "📥 下载完整数据 (JSON)",
            data=json_data,
            file_name=f"meeting_full_{meeting.id}.json",
            mime="application/json",
            use_container_width=True,
            help="结构化 JSON，包含所有会议数据",
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# Section E: LLM 摘要生成
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("🤖 LLM 摘要生成")
st.caption("使用 DeepSeek V4 Flash AI 自动分析会议数据，生成摘要和报告")

# 检查 runtime 是否在线
try:
    from urllib.request import Request, urlopen
    req = Request(f"{RUNTIME_API}/api/llm/status")
    req.add_header("Accept", "application/json")
    with urlopen(req, timeout=3.0) as resp:
        llm_status = json.loads(resp.read().decode("utf-8"))
    runtime_online = True
    llm_available = llm_status.get("data", {}).get("available", False)
except Exception:
    runtime_online = False
    llm_available = False

if not runtime_online:
    st.info(
        "💡 **LLM 功能需要 runtime 在线**\n\n"
        "请确保 `fusion_tracker` 正在运行（端口 8800），"
        "并在启动前设置环境变量:\n\n"
        "```bash\n"
        "export DEEPSEEK_API_KEY=\"your-api-key-here\"\n"
        "export DEEPSEEK_BASE_URL=\"https://api.deepseek.com\"  # 可选\n"
        "```"
    )
elif not llm_available:
    st.info(
        "💡 **LLM 助手未启用**\n\n"
        "请设置环境变量后重启 runtime:\n\n"
        "```bash\n"
        "export DEEPSEEK_API_KEY=\"your-api-key-here\"\n"
        "export DEEPSEEK_BASE_URL=\"https://api.deepseek.com\"  # 可选\n"
        "```"
    )
else:
    # ── 生成按钮 (4 列) ──
    gen_col1, gen_col2, gen_col3, gen_col4 = st.columns(4)

    with gen_col1:
        if st.button("📝 生成会议摘要", use_container_width=True,
                     disabled=st.session_state.llm_generating,
                     key="btn_gen_summary"):
            st.session_state.llm_generating = True
            with st.spinner("🤔 AI 正在生成摘要…（最长 120 秒）"):
                resp = _api_post("/api/llm/summary",
                                 {"meeting_id": _meeting_id}, timeout=120.0)
                if resp and resp.get("ok"):
                    st.session_state.llm_summary = resp.get("data", {}).get("summary", "")
                    st.success("✅ 摘要生成完成")
                else:
                    error = resp.get("error", "未知错误") if resp else "API 无响应"
                    st.error(f"❌ 摘要生成失败: {error}")
            st.session_state.llm_generating = False
            st.rerun()

    with gen_col2:
        if st.button("📋 提取待办事项", use_container_width=True,
                     disabled=st.session_state.llm_generating,
                     key="btn_gen_actions"):
            st.session_state.llm_generating = True
            with st.spinner("🤔 AI 正在提取待办事项…（最长 90 秒）"):
                resp = _api_post("/api/llm/action_items",
                                 {"meeting_id": _meeting_id}, timeout=90.0)
                if resp and resp.get("ok"):
                    st.session_state.llm_action_items = resp.get("data", {}).get("action_items", "")
                    st.success("✅ 待办事项提取完成")
                else:
                    error = resp.get("error", "未知错误") if resp else "API 无响应"
                    st.error(f"❌ 待办提取失败: {error}")
            st.session_state.llm_generating = False
            st.rerun()

    with gen_col3:
        if st.button("❓ 整理评委问题", use_container_width=True,
                     disabled=st.session_state.llm_generating,
                     key="btn_gen_judge"):
            st.session_state.llm_generating = True
            with st.spinner("🤔 AI 正在整理评委问题…（最长 90 秒）"):
                question = ("请根据会议数据，整理出所有评委提出的问题。"
                            "如果主持人备注中有「评委问题」类型的记录，请重点列出。"
                            "按时间顺序排列，每条标注提问者和问题内容。")
                resp = _api_post("/api/llm/chat", {
                    "meeting_id": _meeting_id,
                    "question": question,
                }, timeout=90.0)
                if resp and resp.get("ok"):
                    st.session_state.llm_judge_questions = resp.get("data", {}).get("answer", "")
                    st.success("✅ 评委问题整理完成")
                else:
                    error = resp.get("error", "未知错误") if resp else "API 无响应"
                    st.error(f"❌ 整理失败: {error}")
            st.session_state.llm_generating = False
            st.rerun()

    with gen_col4:
        if st.button("📊 生成系统报告", use_container_width=True,
                     disabled=st.session_state.llm_generating,
                     key="btn_gen_report"):
            st.session_state.llm_generating = True
            with st.spinner("🤔 AI 正在生成系统报告…（最长 90 秒）"):
                question = ("请根据当前会议的事件日志和数据，生成一份系统运行报告。"
                            "包括：发言人识别准确率评估、系统响应时间、"
                            "追踪切换频率、异常事件统计等。")
                resp = _api_post("/api/llm/chat", {
                    "meeting_id": _meeting_id,
                    "question": question,
                }, timeout=90.0)
                if resp and resp.get("ok"):
                    st.session_state.llm_system_report = resp.get("data", {}).get("answer", "")
                    st.success("✅ 系统报告生成完成")
                else:
                    error = resp.get("error", "未知错误") if resp else "API 无响应"
                    st.error(f"❌ 报告生成失败: {error}")
            st.session_state.llm_generating = False
            st.rerun()

    # ── 显示生成结果 ──
    st.divider()

    if st.session_state.llm_summary:
        with st.container(border=True):
            st.markdown("### 📝 会议摘要")
            st.markdown(st.session_state.llm_summary)
            st.download_button(
                "📥 下载摘要 (Markdown)",
                data=st.session_state.llm_summary,
                file_name=f"summary_meeting_{_meeting_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    if st.session_state.llm_action_items:
        with st.container(border=True):
            st.markdown("### 📋 待办事项")
            st.markdown(st.session_state.llm_action_items)
            st.download_button(
                "📥 下载待办事项 (Markdown)",
                data=st.session_state.llm_action_items,
                file_name=f"action_items_{_meeting_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    if st.session_state.llm_judge_questions:
        with st.container(border=True):
            st.markdown("### ❓ 评委问题整理")
            st.markdown(st.session_state.llm_judge_questions)
            st.download_button(
                "📥 下载评委问题 (Markdown)",
                data=st.session_state.llm_judge_questions,
                file_name=f"judge_questions_{_meeting_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    if st.session_state.llm_system_report:
        with st.container(border=True):
            st.markdown("### 📊 系统运行报告")
            st.markdown(st.session_state.llm_system_report)
            st.download_button(
                "📥 下载系统报告 (Markdown)",
                data=st.session_state.llm_system_report,
                file_name=f"system_report_{_meeting_id}.md",
                mime="text/markdown",
                use_container_width=True,
            )

st.caption("智会追声 · 会后复盘与归档")
