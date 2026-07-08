"""数据库连接、会话管理和初始化。

设计要点（嵌入式 ARM 板）：
  - SQLite WAL 模式 — 读不阻塞写，适合多线程并发的 EventBridge
  - StaticPool + check_same_thread=False — 单连接共享，线程安全靠 WAL
  - busy_timeout=5000ms — 防止 "database is locked" 错误
  - foreign_keys=ON — SQLite 默认关闭外键约束，必须显式开启
  - scoped_session — 每线程独立 session，通过 threading.current_thread 区分
  - 惰性初始化 — Engine 在首次 DB 访问时才创建，import 零副作用

用法:
    from storage.db import init_db, get_session, session_scope

    init_db()    # 可选，提前创建表
    with session_scope() as session:
        ...
"""

import os
import threading
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool

# ── 数据库路径 ────────────────────────────────────────────────────────────────
# 默认: <项目根>/data/meeting_tracker.db
# 可通过环境变量 MEETING_TRACKER_DB 覆盖 (用于测试)

def _default_db_path() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data", "meeting_tracker.db")

DB_PATH = os.environ.get("MEETING_TRACKER_DB", _default_db_path())

# ── 全局状态 (惰性初始化, 线程安全) ───────────────────────────────────────────

_Engine = None
_SessionFactory: scoped_session | None = None
_lock = threading.RLock()  # RLock — get_session_factory() 内部调用 get_engine()，都需获取锁


def get_engine():
    """惰性创建并返回 SQLAlchemy Engine。线程安全的单例。

    Returns:
        sqlalchemy.engine.Engine
    """
    global _Engine
    if _Engine is not None:
        return _Engine
    with _lock:
        if _Engine is not None:
            return _Engine
        # 确保 data/ 目录存在
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

        _Engine = create_engine(
            f"sqlite:///{DB_PATH}",
            connect_args={"check_same_thread": False},
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            echo=False,
        )

        # 启用 WAL、外键、忙等超时
        @event.listens_for(_Engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA wal_autocheckpoint=1000")
            cursor.close()

        return _Engine


def get_session_factory() -> scoped_session:
    """惰性创建并返回线程安全的 scoped_session 工厂。

    Returns:
        sqlalchemy.orm.scoped_session
    """
    global _SessionFactory
    if _SessionFactory is not None:
        return _SessionFactory
    with _lock:
        if _SessionFactory is not None:
            return _SessionFactory
        _SessionFactory = scoped_session(
            sessionmaker(bind=get_engine()),
            scopefunc=threading.current_thread,
        )
        return _SessionFactory


def get_session():
    """返回当前线程的 scoped session。

    调用者不应手动关闭这个 session — scoped_session 管理其生命周期。
    推荐使用 session_scope() 上下文管理器来管理事务边界。
    """
    return get_session_factory()()


@contextmanager
def session_scope():
    """上下文管理器：提供 session，成功自动提交，异常自动回滚。

    退出时调用 scoped_session.remove() 将会话归还连接池。
    这在 Streamlit 等多线程/重跑环境下至关重要 — 否则连接池会耗尽。

    Usage:
        with session_scope() as session:
            session.add(some_object)
            # 多步操作自动共享同一事务
    """
    factory = get_session_factory()
    session = factory()
    session.expire_on_commit = False  # commit 后不使对象过期，避免 DetachedInstanceError
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.expunge_all()  # 解除所有对象与会话的绑定（保留已加载属性）
        factory.remove()       # 归还连接到池，防止 QueuePool 耗尽


def init_db():
    """创建所有表（如果不存在）。安全多次调用。"""
    from storage.models import Base
    Base.metadata.create_all(get_engine())


def reset_db():
    """删除并重建所有表。**仅用于测试**。"""
    from storage.models import Base
    Base.metadata.drop_all(get_engine())
    Base.metadata.create_all(get_engine())


def db_path() -> str:
    """返回当前数据库文件路径。"""
    return DB_PATH
