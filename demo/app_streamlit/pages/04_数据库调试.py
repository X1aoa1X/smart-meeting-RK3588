"""数据库调试 — 6张业务表的增删查改 + 原始SQL执行。

仅供开发调试使用，提供对所有 ORM 表的完整 CRUD 操作。
"""

import streamlit as st
import pandas as pd
import os
import sys
import json
from datetime import datetime
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="数据库调试", page_icon="🗄️", layout="wide")
apply_office_theme()

_storage_ready = _init_storage()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "dbg_confirm_delete": None,  # (table_name, record_id, display_label) or None
    "dbg_sql_query": "",
    "dbg_sql_result": None,
    "dbg_sql_error": None,
    "dbg_sql_confirm_pending": False,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def _get_row_counts() -> dict:
    """Query all 6 business tables and return {table_name: count}."""
    from storage.db import session_scope
    from storage.models import (
        Participant, Meeting, SpeakerSegment, Event, HostNote, SystemConfig,
    )

    with session_scope() as s:
        return {
            "participants": s.query(Participant).count(),
            "meetings": s.query(Meeting).count(),
            "speaker_segments": s.query(SpeakerSegment).count(),
            "events": s.query(Event).count(),
            "host_notes": s.query(HostNote).count(),
            "system_config": s.query(SystemConfig).count(),
        }


def _records_to_df(records: list, columns: list[dict]) -> pd.DataFrame:
    """Convert ORM record list to DataFrame using column config.

    Args:
        records: list of ORM model instances
        columns: [{"key": "attr_name", "label": "显示名"}, ...]
    """
    if not records:
        return pd.DataFrame()

    data = []
    for r in records:
        row = {}
        for col in columns:
            val = getattr(r, col["key"], None)
            # Format special types
            if isinstance(val, datetime):
                val = val.strftime("%Y-%m-%d %H:%M:%S")
            elif col["key"] == "payload_json" and val:
                try:
                    parsed = json.loads(val)
                    val = json.dumps(parsed, ensure_ascii=False, indent=2)
                except (json.JSONDecodeError, TypeError):
                    pass
            row[col["label"]] = val if val is not None else ""
        data.append(row)
    return pd.DataFrame(data)


def _list_records(session, model_class, repo=None, search: str = None,
                  search_fields: list[str] = None, filters: dict = None,
                  order_by=None, limit: int = None) -> list:
    """List records using repo.list_all() if available, otherwise direct query.

    Args:
        session: SQLAlchemy session
        model_class: ORM model class
        repo: repository instance (optional)
        search: search string for text filtering
        search_fields: model attribute names to search on
        filters: {field_name: value} for exact matching
        order_by: model attribute to order by
        limit: max records to return
    """
    # Try repo method first
    if repo is not None and hasattr(repo, 'list_all') and not filters:
        try:
            return repo.list_all(search=search) if search else repo.list_all()
        except TypeError:
            pass  # list_all doesn't accept search, fall through

    # Direct query
    q = session.query(model_class)

    # Apply exact filters
    if filters:
        for key, val in filters.items():
            if val is not None and val != "" and val != []:
                q = q.filter(getattr(model_class, key) == val)

    # Apply text search
    if search and search_fields:
        from sqlalchemy import or_
        patterns = []
        for field in search_fields:
            patterns.append(getattr(model_class, field).ilike(f"%{search}%"))
        q = q.filter(or_(*patterns))

    # Apply ordering
    if order_by is not None:
        q = q.order_by(order_by)
    else:
        # Default: by id desc
        if hasattr(model_class, "id"):
            q = q.order_by(model_class.id.desc())

    if limit is not None:
        q = q.limit(limit)

    return q.all()


def _get_record_by_id(session, model_class, record_id, repo=None):
    """Get a single record by primary key id."""
    if repo is not None and hasattr(repo, 'get_by_id'):
        return repo.get_by_id(record_id)
    return session.get(model_class, record_id)


def _create_record(session, model_class, repo=None, **kwargs):
    """Create a record using repo.create() if available, otherwise direct add."""
    if repo is not None and hasattr(repo, 'create'):
        return repo.create(**kwargs)
    instance = model_class(**kwargs)
    session.add(instance)
    session.flush()
    return instance


def _update_record(session, record, repo=None, **kwargs):
    """Update a record using repo.update() if available, otherwise set attrs."""
    if repo is not None and hasattr(repo, 'update'):
        return repo.update(record, **kwargs)
    for key, value in kwargs.items():
        if hasattr(record, key):
            setattr(record, key, value)
    if hasattr(record, 'updated_at'):
        record.updated_at = datetime.utcnow()
    session.flush()
    return record


def _delete_record(session, record, repo=None):
    """Delete a record using repo.delete() if available, otherwise session.delete."""
    if repo is not None and hasattr(repo, 'delete'):
        repo.delete(record)
    else:
        session.delete(record)
        session.flush()


def _parse_datetime(val: str) -> datetime | None:
    """Parse a datetime string, returning None for empty/whitespace-only input."""
    if not val or not val.strip():
        return None
    val = val.strip()
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
        try:
            return datetime.strptime(val, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析日期时间: '{val}' (支持格式: YYYY-MM-DD HH:MM:SS)")


def _fmt_datetime(dt) -> str:
    """Format a datetime or None to string for form display."""
    if dt is None:
        return ""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return str(dt)


# ═══════════════════════════════════════════════════════════════════════════════
# Table configuration
# ═══════════════════════════════════════════════════════════════════════════════

def _get_table_configs() -> list[dict]:
    """Return configuration for all 6 business tables."""
    from storage.models import (
        Participant, Meeting, SpeakerSegment, Event, HostNote, SystemConfig,
        MeetingStatus,
    )
    from storage.repo import (
        ParticipantRepo, MeetingRepo, SpeakerSegmentRepo,
        EventRepo, HostNoteRepo, ConfigRepo,
    )

    return [
        # ── 1. Participants ──────────────────────────────────────────────
        {
            "name": "participants",
            "model": Participant,
            "repo": ParticipantRepo,
            "tab_label": "👥 参会人员",
            "search_placeholder": "按姓名或标签ID搜索...",
            "search_fields": ["name", "tag_id"],
            "filters": None,
            "cascade_warning": None,
            "limit": None,
            "columns": [
                {"key": "id", "label": "ID"},
                {"key": "tag_id", "label": "标签ID"},
                {"key": "name", "label": "姓名"},
                {"key": "organization", "label": "单位"},
                {"key": "role", "label": "角色"},
                {"key": "title", "label": "职务"},
                {"key": "phone", "label": "电话"},
                {"key": "email", "label": "邮箱"},
                {"key": "avatar_path", "label": "头像路径"},
                {"key": "created_at", "label": "创建时间"},
                {"key": "updated_at", "label": "更新时间"},
            ],
            "form_fields": [
                {"key": "tag_id", "label": "标签ID", "type": "text", "required": True},
                {"key": "name", "label": "姓名", "type": "text", "required": True},
                {"key": "organization", "label": "单位", "type": "text", "required": False},
                {"key": "role", "label": "角色", "type": "text", "required": False,
                 "default": "参会人员"},
                {"key": "title", "label": "职务", "type": "text", "required": False},
                {"key": "phone", "label": "电话", "type": "text", "required": False},
                {"key": "email", "label": "邮箱", "type": "text", "required": False},
                {"key": "avatar_path", "label": "头像路径", "type": "text", "required": False},
            ],
            "id_label": lambda r: f"[{r.tag_id}] {r.name}",
        },
        # ── 2. Meetings ──────────────────────────────────────────────────
        {
            "name": "meetings",
            "model": Meeting,
            "repo": MeetingRepo,
            "tab_label": "📋 会议",
            "search_placeholder": "按会议名称搜索...",
            "search_fields": ["name"],
            "filters": [
                {"key": "status", "label": "状态筛选",
                 "options": [("全部", ""), ("planned", "planned"),
                            ("in_progress", "in_progress"),
                            ("completed", "completed"), ("cancelled", "cancelled")]},
            ],
            "cascade_warning": (
                "⚠️ 删除会议将**级联删除**其所有发言片段、事件记录和主持人备注。此操作不可撤销！"
            ),
            "limit": None,
            "columns": [
                {"key": "id", "label": "ID"},
                {"key": "name", "label": "名称"},
                {"key": "location", "label": "地点"},
                {"key": "status", "label": "状态"},
                {"key": "start_time", "label": "开始时间"},
                {"key": "end_time", "label": "结束时间"},
                {"key": "description", "label": "描述"},
                {"key": "created_at", "label": "创建时间"},
                {"key": "updated_at", "label": "更新时间"},
            ],
            "form_fields": [
                {"key": "name", "label": "会议名称", "type": "text", "required": True},
                {"key": "location", "label": "地点", "type": "text", "required": False},
                {"key": "description", "label": "描述", "type": "textarea", "required": False},
                {"key": "status", "label": "状态", "type": "select",
                 "options": ["planned", "in_progress", "completed", "cancelled"],
                 "required": False, "default": "planned"},
                {"key": "start_time", "label": "开始时间 (YYYY-MM-DD HH:MM:SS)",
                 "type": "datetime", "required": False},
                {"key": "end_time", "label": "结束时间 (YYYY-MM-DD HH:MM:SS, 留空=N/A)",
                 "type": "datetime", "required": False},
            ],
            "id_label": lambda r: f"[{r.id}] {r.name} ({r.status})",
        },
        # ── 3. Speaker Segments ──────────────────────────────────────────
        {
            "name": "speaker_segments",
            "model": SpeakerSegment,
            "repo": None,  # No full CRUD repo — use direct session
            "tab_label": "🎤 发言片段",
            "search_placeholder": "",
            "search_fields": ["speaker_name"],
            "filters": None,
            "cascade_warning": None,
            "limit": 500,
            "columns": [
                {"key": "id", "label": "ID"},
                {"key": "meeting_id", "label": "会议ID"},
                {"key": "speaker_tag_id", "label": "发言人标签"},
                {"key": "speaker_name", "label": "发言人"},
                {"key": "role", "label": "角色"},
                {"key": "source", "label": "来源"},
                {"key": "start_time", "label": "开始时间"},
                {"key": "end_time", "label": "结束时间"},
                {"key": "duration_seconds", "label": "时长(秒)"},
                {"key": "confidence", "label": "置信度"},
            ],
            "form_fields": [
                {"key": "meeting_id", "label": "会议ID", "type": "number", "required": True},
                {"key": "speaker_tag_id", "label": "发言人标签ID", "type": "text", "required": False},
                {"key": "speaker_name", "label": "发言人姓名", "type": "text", "required": False},
                {"key": "role", "label": "角色", "type": "text", "required": False},
                {"key": "source", "label": "来源", "type": "select",
                 "options": ["AprilTag", "manual", "unknown"], "required": False,
                 "default": "unknown"},
                {"key": "start_time", "label": "开始时间 (YYYY-MM-DD HH:MM:SS)",
                 "type": "datetime", "required": False},
                {"key": "end_time", "label": "结束时间 (留空=进行中)",
                 "type": "datetime", "required": False},
                {"key": "duration_seconds", "label": "时长(秒)", "type": "number",
                 "required": False, "default": 0.0},
                {"key": "confidence", "label": "置信度 (0.0-1.0)", "type": "number",
                 "required": False, "default": 0.0},
            ],
            "id_label": lambda r: f"[{r.id}] meeting={r.meeting_id} speaker={r.speaker_name or '?'}",
        },
        # ── 4. Events ────────────────────────────────────────────────────
        {
            "name": "events",
            "model": Event,
            "repo": None,  # No full CRUD repo — use direct session
            "tab_label": "📡 系统事件",
            "search_placeholder": "",
            "search_fields": ["event_type"],
            "filters": None,
            "cascade_warning": None,
            "limit": 500,
            "columns": [
                {"key": "id", "label": "ID"},
                {"key": "meeting_id", "label": "会议ID"},
                {"key": "event_type", "label": "事件类型"},
                {"key": "timestamp", "label": "时间戳"},
                {"key": "payload_json", "label": "载荷(JSON)"},
            ],
            "form_fields": [
                {"key": "meeting_id", "label": "会议ID (可选)", "type": "number",
                 "required": False},
                {"key": "event_type", "label": "事件类型", "type": "text", "required": True},
                {"key": "timestamp", "label": "时间戳 (YYYY-MM-DD HH:MM:SS)",
                 "type": "datetime", "required": False},
                {"key": "payload_json", "label": "载荷 (JSON 字符串)", "type": "json",
                 "required": False},
            ],
            "id_label": lambda r: f"[{r.id}] {r.event_type} @ {_fmt_datetime(r.timestamp)}",
        },
        # ── 5. Host Notes ────────────────────────────────────────────────
        {
            "name": "host_notes",
            "model": HostNote,
            "repo": None,  # HostNoteRepo lacks list_all/update/delete
            "tab_label": "📝 主持人备注",
            "search_placeholder": "",
            "search_fields": ["content"],
            "filters": None,
            "cascade_warning": None,
            "limit": 500,
            "columns": [
                {"key": "id", "label": "ID"},
                {"key": "meeting_id", "label": "会议ID"},
                {"key": "note_type", "label": "备注类型"},
                {"key": "content", "label": "内容"},
                {"key": "related_speaker", "label": "相关发言人"},
                {"key": "timestamp", "label": "时间戳"},
            ],
            "form_fields": [
                {"key": "meeting_id", "label": "会议ID", "type": "number", "required": True},
                {"key": "note_type", "label": "备注类型", "type": "select",
                 "options": ["评委问题", "重点结论", "待办事项", "系统异常", "主持人备注"],
                 "required": True, "default": "主持人备注"},
                {"key": "content", "label": "内容", "type": "textarea", "required": True},
                {"key": "related_speaker", "label": "相关发言人", "type": "text",
                 "required": False},
                {"key": "timestamp", "label": "时间戳 (YYYY-MM-DD HH:MM:SS)",
                 "type": "datetime", "required": False},
            ],
            "id_label": lambda r: f"[{r.id}] {r.note_type}: {r.content[:50]}...",
        },
        # ── 6. System Config ─────────────────────────────────────────────
        {
            "name": "system_config",
            "model": SystemConfig,
            "repo": None,  # ConfigRepo has different API — direct session
            "tab_label": "⚙️ 系统配置",
            "search_placeholder": "",
            "search_fields": ["config_section", "config_key"],
            "filters": None,
            "cascade_warning": None,
            "limit": None,
            "columns": [
                {"key": "id", "label": "ID"},
                {"key": "config_section", "label": "配置分区"},
                {"key": "config_key", "label": "配置键"},
                {"key": "config_value", "label": "配置值"},
                {"key": "description", "label": "描述"},
                {"key": "updated_at", "label": "更新时间"},
            ],
            "form_fields": [
                {"key": "config_section", "label": "配置分区", "type": "text",
                 "required": True},
                {"key": "config_key", "label": "配置键", "type": "text", "required": True},
                {"key": "config_value", "label": "配置值", "type": "textarea",
                 "required": True},
                {"key": "description", "label": "描述", "type": "text", "required": False},
            ],
            "id_label": lambda r: f"[{r.id}] [{r.config_section}] {r.config_key}",
        },
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Table summary (metric cards)
# ═══════════════════════════════════════════════════════════════════════════════

def _render_table_summary():
    """Render row-count metric cards below title."""
    counts = _get_row_counts()
    labels = {
        "participants": "👥 参会人员",
        "meetings": "📋 会议",
        "speaker_segments": "🎤 发言片段",
        "events": "📡 事件",
        "host_notes": "📝 备注",
        "system_config": "⚙️ 配置",
    }
    cols = st.columns(6)
    for i, (key, label) in enumerate(labels.items()):
        with cols[i]:
            st.metric(label, counts.get(key, 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Generic table CRUD tab
# ═══════════════════════════════════════════════════════════════════════════════

def _render_table_tab(cfg: dict):
    """Render a full CRUD tab for a single table.

    Args:
        cfg: table configuration dict with keys:
            name, model, repo, columns, form_fields, id_label, search_fields,
            search_placeholder, filters, cascade_warning, limit
    """
    from storage.db import session_scope
    from sqlalchemy.exc import IntegrityError

    model_class = cfg["model"]
    repo_class = cfg["repo"]
    columns = cfg["columns"]
    form_fields = cfg["form_fields"]
    table_name = cfg["name"]

    # ── Build filters row ────────────────────────────────────────────────
    filter_values = {}
    if cfg.get("filters"):
        filter_cols = st.columns(len(cfg["filters"]) + 1)
        for i, fdef in enumerate(cfg["filters"]):
            with filter_cols[i]:
                options = fdef["options"]
                labels = [o[0] for o in options]
                values = [o[1] for o in options]
                selected_label = st.selectbox(
                    fdef["label"], labels, key=f"filter_{table_name}_{fdef['key']}"
                )
                selected_idx = labels.index(selected_label)
                filter_values[fdef["key"]] = values[selected_idx]
    else:
        filter_cols = [st.columns(1)[0]]

    # Search input
    search_fields = cfg.get("search_fields", [])
    if search_fields:
        with filter_cols[-1]:
            search = st.text_input(
                "🔍 搜索", placeholder=cfg.get("search_placeholder", "搜索..."),
                key=f"search_{table_name}"
            )
    else:
        search = ""

    # ── Fetch records ────────────────────────────────────────────────────
    with session_scope() as s:
        repo = repo_class(s) if repo_class is not None else None
        records = _list_records(
            s, model_class, repo=repo,
            search=(search if search else None),
            search_fields=search_fields,
            filters={k: v for k, v in filter_values.items() if v},
            limit=cfg.get("limit"),
        )

    # ── Read: dataframe display ──────────────────────────────────────────
    if records:
        df = _records_to_df(records, columns)
        st.caption(f"共 {len(records)} 条记录")
        st.dataframe(df, use_container_width=True, hide_index=True,
                     height=min(400, 35 * len(records) + 38))
    else:
        st.info("暂无记录")

    # ── Create ───────────────────────────────────────────────────────────
    with st.expander("➕ 新增记录", expanded=False):
        with st.form(f"create_form_{table_name}", clear_on_submit=True):
            create_values = {}
            cols = st.columns(2)
            for i, field in enumerate(form_fields):
                with cols[i % 2]:
                    create_values[field["key"]] = _render_form_field(
                        field, f"create_{table_name}_{field['key']}",
                        default=field.get("default")
                    )
            submitted = st.form_submit_button("✅ 创建", type="primary",
                                               use_container_width=True)
            if submitted:
                # Validate required fields
                missing = []
                for field in form_fields:
                    if field.get("required") and not _is_filled(create_values[field["key"]]):
                        missing.append(field["label"])
                if missing:
                    st.error(f"请填写必填字段: {', '.join(missing)}")
                else:
                    try:
                        with session_scope() as s:
                            repo = repo_class(s) if repo_class is not None else None
                            kwargs = _prepare_kwargs(create_values, form_fields, for_create=True)
                            record = _create_record(s, model_class, repo=repo, **kwargs)
                        st.success(f"✅ 创建成功 (ID={record.id})")
                        st.rerun()
                    except IntegrityError as e:
                        st.error(f"唯一约束冲突: {e}")
                    except ValueError as e:
                        st.error(f"参数错误: {e}")
                    except Exception as e:
                        st.error(f"创建失败: {e}")

    # ── Update ───────────────────────────────────────────────────────────
    with st.expander("✏️ 编辑记录", expanded=False):
        if not records:
            st.caption("无记录可编辑")
        else:
            # Select record
            record_options = {cfg["id_label"](r): r.id for r in records}
            selected_label = st.selectbox(
                "选择要编辑的记录", list(record_options.keys()),
                key=f"edit_select_{table_name}"
            )
            if selected_label:
                edit_id = record_options[selected_label]
                with session_scope() as s:
                    repo = repo_class(s) if repo_class is not None else None
                    edit_record = _get_record_by_id(s, model_class, edit_id, repo=repo)

                if edit_record:
                    with st.form(f"edit_form_{table_name}", clear_on_submit=True):
                        edit_values = {}
                        cols = st.columns(2)
                        for i, field in enumerate(form_fields):
                            with cols[i % 2]:
                                current_val = getattr(edit_record, field["key"], None)
                                edit_values[field["key"]] = _render_form_field(
                                    field, f"edit_{table_name}_{field['key']}",
                                    default=current_val
                                )
                        submitted = st.form_submit_button("💾 更新", type="primary",
                                                           use_container_width=True)
                        if submitted:
                            missing = []
                            for field in form_fields:
                                if field.get("required") and not _is_filled(edit_values[field["key"]]):
                                    missing.append(field["label"])
                            if missing:
                                st.error(f"请填写必填字段: {', '.join(missing)}")
                            else:
                                try:
                                    with session_scope() as s:
                                        repo = repo_class(s) if repo_class is not None else None
                                        record = _get_record_by_id(s, model_class, edit_id, repo=repo)
                                        kwargs = _prepare_kwargs(edit_values, form_fields, for_create=False)
                                        _update_record(s, record, repo=repo, **kwargs)
                                    st.success(f"✅ 更新成功 (ID={edit_id})")
                                    st.rerun()
                                except IntegrityError as e:
                                    st.error(f"唯一约束冲突: {e}")
                                except ValueError as e:
                                    st.error(f"参数错误: {e}")
                                except Exception as e:
                                    st.error(f"更新失败: {e}")

    # ── Delete ───────────────────────────────────────────────────────────
    with st.expander("🗑️ 删除记录", expanded=False):
        if not records:
            st.caption("无记录可删除")
        else:
            # Cascade warning
            if cfg.get("cascade_warning"):
                st.warning(cfg["cascade_warning"])

            record_options = {cfg["id_label"](r): r.id for r in records}
            selected_label = st.selectbox(
                "选择要删除的记录", list(record_options.keys()),
                key=f"delete_select_{table_name}"
            )
            if selected_label:
                delete_id = record_options[selected_label]
                if st.button("🗑️ 删除此记录", key=f"delete_btn_{table_name}",
                             type="secondary"):
                    st.session_state.dbg_confirm_delete = (
                        table_name, delete_id, selected_label
                    )
                    st.rerun()

    # ── Delete confirmation banner (shown outside expanders) ─────────────
    confirm = st.session_state.dbg_confirm_delete
    if confirm is not None and confirm[0] == table_name:
        _, confirm_id, confirm_label = confirm
        st.warning(f"⚠️ 确认删除 `{confirm_label}`？此操作**不可撤销**。")

        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            if st.button("✅ 确认删除", key=f"confirm_delete_{table_name}",
                         type="primary"):
                try:
                    with session_scope() as s:
                        repo = repo_class(s) if repo_class is not None else None
                        record = _get_record_by_id(s, model_class, confirm_id, repo=repo)
                        if record:
                            _delete_record(s, record, repo=repo)
                        else:
                            st.error("记录不存在，可能已被删除")
                    st.session_state.dbg_confirm_delete = None
                    st.success(f"✅ 已删除")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除失败: {e}")
        with col2:
            if st.button("❌ 取消", key=f"cancel_delete_{table_name}"):
                st.session_state.dbg_confirm_delete = None
                st.rerun()


# ── Form field rendering helpers ─────────────────────────────────────────────

def _render_form_field(field: dict, key: str, default=None):
    """Render a single form field based on its type config.

    Args:
        field: {"key", "label", "type", "required", "options", "default"}
        key: unique streamlit key
        default: current value (for edit form) or None (for create form)

    Returns:
        The field's current value.
    """
    # Resolve default value: edit form's default takes priority over field default
    field_default = field.get("default")
    if default is not None and default != "":
        display_default = default
    elif field_default is not None:
        display_default = field_default
    else:
        display_default = None

    ftype = field["type"]
    label = field["label"]

    if ftype == "text":
        val = display_default if isinstance(display_default, str) else ""
        return st.text_input(label, value=val, key=key)

    elif ftype == "textarea":
        val = display_default if isinstance(display_default, str) else ""
        return st.text_area(label, value=val, key=key, height=100)

    elif ftype == "number":
        # Determine numeric type from the actual value to keep value/step consistent
        field_def = field.get("default")
        if display_default is not None and display_default != "":
            if isinstance(display_default, float) or isinstance(field_def, float):
                val = float(display_default)
                step = 0.01
            else:
                val = int(display_default)
                step = 1
        elif isinstance(field_def, float):
            val = float(field_def)
            step = 0.01
        elif isinstance(field_def, int):
            val = field_def
            step = 1
        else:
            val = 0
            step = 1
        return st.number_input(label, value=val, step=step, key=key)

    elif ftype == "select":
        options = field.get("options", [])
        # Find index of default value
        if isinstance(display_default, str) and display_default in options:
            idx = options.index(display_default)
        else:
            idx = 0
        return st.selectbox(label, options, index=idx, key=key)

    elif ftype == "datetime":
        val = _fmt_datetime(display_default) if display_default else ""
        return st.text_input(label, value=val, key=key,
                             placeholder="YYYY-MM-DD HH:MM:SS")

    elif ftype == "json":
        val = display_default
        if isinstance(val, str):
            pass  # keep as-is
        elif isinstance(val, (dict, list)):
            val = json.dumps(val, ensure_ascii=False, indent=2)
        else:
            val = ""
        return st.text_area(label, value=val, key=key, height=120)

    else:
        val = str(display_default) if display_default is not None else ""
        return st.text_input(label, value=val, key=key)


def _is_filled(val) -> bool:
    """Check if a form value is non-empty."""
    if val is None:
        return False
    if isinstance(val, str) and not val.strip():
        return False
    return True


def _prepare_kwargs(values: dict, form_fields: list, for_create: bool = True) -> dict:
    """Convert form values to kwargs for create/update.

    Handles type conversions: datetime strings → datetime objects,
    JSON strings → validated strings, empty strings for nullable fields → None.
    """
    kwargs = {}
    for field in form_fields:
        key = field["key"]
        val = values[key]
        ftype = field["type"]

        if ftype == "datetime":
            if isinstance(val, str):
                try:
                    kwargs[key] = _parse_datetime(val)
                except ValueError:
                    raise ValueError(f"字段 '{field['label']}' 日期格式无效: {val}")
            elif isinstance(val, datetime):
                kwargs[key] = val
            elif val is None:
                kwargs[key] = None
            else:
                kwargs[key] = val
        elif ftype == "json":
            if isinstance(val, str) and val.strip():
                # Validate JSON
                try:
                    json.loads(val)
                    kwargs[key] = val  # store as string in TEXT column
                except json.JSONDecodeError as e:
                    raise ValueError(f"字段 '{field['label']}' JSON 格式无效: {e}")
            elif val:
                kwargs[key] = val
            # Empty → don't include in kwargs (keep existing or NULL)
        elif ftype == "number":
            if isinstance(val, (int, float)):
                kwargs[key] = val
            elif isinstance(val, str) and val.strip():
                try:
                    kwargs[key] = float(val) if '.' in val else int(val)
                except ValueError:
                    kwargs[key] = val
            elif val == "" or val is None:
                kwargs[key] = None
            else:
                kwargs[key] = val
        else:
            # text, textarea, select
            if isinstance(val, str) and not val.strip():
                # For required fields, keep empty string; for optional, convert to None
                # Actually, keep as empty string for most text fields
                kwargs[key] = val.strip() if val else ""
            else:
                kwargs[key] = val

    return kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# Raw SQL tab
# ═══════════════════════════════════════════════════════════════════════════════

DANGEROUS_KEYWORDS = ["DROP", "DELETE", "UPDATE", "ALTER", "TRUNCATE", "INSERT", "CREATE"]


def _is_dangerous_sql(sql: str) -> bool:
    """Check if SQL contains dangerous keywords (DML/DDL beyond SELECT)."""
    import re
    # Strip comments and normalize whitespace
    cleaned = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    upper = cleaned.strip().upper()

    for kw in DANGEROUS_KEYWORDS:
        # Match as a standalone keyword (preceded by start/nothing/whitespace/newline)
        if re.search(r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$)', upper):
            return True
    return False


def _execute_raw_sql(sql: str) -> dict:
    """Execute raw SQL via SQLAlchemy engine connection.

    Returns:
        {"type": "select"|"dml", "columns": [...], "rows": [...], "rowcount": int}
    """
    from storage.db import get_engine
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        # Use explicit transaction for DML
        with conn.begin():
            result = conn.execute(text(sql))
            if result.returns_rows:
                rows = [dict(row._mapping) for row in result]
                columns = list(rows[0].keys()) if rows else []
                return {"type": "select", "columns": columns, "rows": rows}
            else:
                return {"type": "dml", "rowcount": result.rowcount}


def _render_raw_sql_tab():
    """Render the raw SQL execution tab."""
    st.warning(
        "⚠️ **危险操作区域** — 请谨慎使用 DROP / DELETE / UPDATE / ALTER / TRUNCATE 语句。"
    )

    # SQL input
    sql = st.text_area(
        "SQL 语句",
        value=st.session_state.dbg_sql_query,
        height=200,
        placeholder="SELECT * FROM participants;\nSELECT id, name, status FROM meetings;",
        key="raw_sql_input",
    )

    execute_col1, execute_col2 = st.columns([1, 4])
    with execute_col1:
        execute_clicked = st.button("▶ 执行", type="primary", key="execute_sql_btn",
                                    use_container_width=True)

    if execute_clicked:
        if not sql.strip():
            st.error("请输入 SQL 语句")
        else:
            # Store query for display
            st.session_state.dbg_sql_query = sql

            # Safety check
            if _is_dangerous_sql(sql):
                st.session_state.dbg_sql_confirm_pending = True
                st.session_state.dbg_sql_result = None
                st.session_state.dbg_sql_error = None
            else:
                # Safe SELECT — execute directly
                st.session_state.dbg_sql_confirm_pending = False
                try:
                    result = _execute_raw_sql(sql)
                    st.session_state.dbg_sql_result = result
                    st.session_state.dbg_sql_error = None
                except Exception as e:
                    st.session_state.dbg_sql_result = None
                    st.session_state.dbg_sql_error = str(e)

    # Dangerous SQL confirmation
    if st.session_state.dbg_sql_confirm_pending:
        st.warning(
            "⚠️ 检测到危险的 SQL 语句 (DROP / DELETE / UPDATE / ALTER / TRUNCATE / INSERT / CREATE)。"
            " 请确认执行。"
        )
        c1, c2, c3 = st.columns([1, 1, 4])
        with c1:
            if st.button("⚠️ 确认执行", key="confirm_dangerous_sql", type="primary"):
                st.session_state.dbg_sql_confirm_pending = False
                try:
                    result = _execute_raw_sql(sql)
                    st.session_state.dbg_sql_result = result
                    st.session_state.dbg_sql_error = None
                except Exception as e:
                    st.session_state.dbg_sql_result = None
                    st.session_state.dbg_sql_error = str(e)
                st.rerun()
        with c2:
            if st.button("❌ 取消", key="cancel_dangerous_sql"):
                st.session_state.dbg_sql_confirm_pending = False
                st.rerun()

    # Display results
    st.divider()
    st.subheader("📊 查询结果")

    if st.session_state.dbg_sql_error:
        st.error(f"执行失败: {st.session_state.dbg_sql_error}")

    if st.session_state.dbg_sql_result:
        result = st.session_state.dbg_sql_result
        if result["type"] == "select":
            if result["rows"]:
                st.caption(f"返回 {len(result['rows'])} 行")
                df = pd.DataFrame(result["rows"], columns=result["columns"])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("查询返回空结果")
        else:
            st.success(f"执行成功 — 影响行数: {result['rowcount']}")


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="office-section-label">数据库调试</div>', unsafe_allow_html=True)
    st.caption(f"路径: `{_get_db_path()}`")

    st.divider()

    # Row counts
    st.markdown('<div class="office-section-label">表行数</div>', unsafe_allow_html=True)
    counts = _get_row_counts()
    for key, label in [
        ("participants", "👥 参会人员"),
        ("meetings", "📋 会议"),
        ("speaker_segments", "🎤 发言片段"),
        ("events", "📡 事件"),
        ("host_notes", "📝 备注"),
        ("system_config", "⚙️ 配置"),
    ]:
        st.metric(label, counts.get(key, 0))

    st.divider()

    # Navigation
    render_sidebar_nav("debug")

    st.divider()
    st.caption("仅供开发调试使用，请勿在生产环境操作。")


# ═══════════════════════════════════════════════════════════════════════════════
# Main page
# ═══════════════════════════════════════════════════════════════════════════════

render_page_header("数据库调试", f"所有业务表的增删查改 + 原始 SQL 执行。仅供开发调试。数据库路径：<code>{_get_db_path()}</code>", "Developer Tools")

# Row count summary
_render_table_summary()

st.divider()

# Tabs
table_configs = _get_table_configs()
tab_labels = [cfg["tab_label"] for cfg in table_configs] + ["🔧 原始 SQL"]
tabs = st.tabs(tab_labels)

# Render each table tab
for i, cfg in enumerate(table_configs):
    with tabs[i]:
        _render_table_tab(cfg)

# Render raw SQL tab
with tabs[-1]:
    _render_raw_sql_tab()
