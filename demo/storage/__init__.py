"""Storage 模块 — 数据库、ORM 模型、仓库层和事件持久化。

初始化入口:
    from storage import init
    init()   # 创建数据库 + 运行迁移

模块:
    storage.models       — SQLAlchemy ORM 模型 (6 张表)
    storage.db            — 数据库连接、会话管理
    storage.repo          — Repository/DAO 层
    storage.event_bridge  — EventBus → DB 异步持久化
    storage.migration     — 最小化迁移框架

注意: 本模块依赖 SQLAlchemy。如果未安装，import 本模块会报错。
安装: pip install sqlalchemy
"""

import logging

logger = logging.getLogger(__name__)


def init():
    """初始化数据库：创建所有表 + 运行待处理的迁移。

    安全多次调用 — 已存在的表不会被重建。
    在 demo 启动时调用（如 demos/fusion_tracker.py 的 main() 中）。

    Raises:
        ImportError: 如果 SQLAlchemy 未安装
    """
    from storage.db import init_db
    from storage.migration import run_migrations

    init_db()
    applied = run_migrations()
    if applied > 0:
        logger.info(f"Applied {applied} migration(s)")
    return applied


# 导出常用符号
from storage.models import (
    Participant,
    Meeting,
    MeetingStatus,
    SpeakerSegment,
    SegmentSource,
    Event,
    HostNote,
    NoteType,
    SystemConfig,
    SchemaVersion,
    AgentDecision,
    TTSEvent,
)

from storage.db import (
    get_session,
    session_scope,
    init_db,
    reset_db,
    db_path,
)

__all__ = [
    # Models
    "Participant", "Meeting", "MeetingStatus",
    "SpeakerSegment", "SegmentSource",
    "Event", "HostNote", "NoteType",
    "SystemConfig", "SchemaVersion",
    "AgentDecision", "TTSEvent",
    # DB
    "get_session", "session_scope", "init_db", "reset_db", "db_path",
    # Entry point
    "init",
]
