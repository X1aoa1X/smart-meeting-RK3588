"""最小化 Schema 迁移框架。

不使用 Alembic (避免额外依赖)，而是基于简单的版本号追踪:

  - _schema_version 表记录当前已应用的迁移版本
  - 迁移文件放在 storage/migrations/ 下，命名格式: NNN_description.py
  - 每个迁移文件导出 upgrade(engine) 函数
  - run_migrations() 自动发现并应用未执行的迁移

用法:
    from storage.migration import run_migrations
    run_migrations()  # 在 storage.init() 中自动调用
"""

import os
import re
import importlib
import importlib.util
import logging

logger = logging.getLogger(__name__)

_MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")


def _get_current_version(engine) -> int:
    """查询 _schema_version 表中最大的版本号。"""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(version) FROM _schema_version")
            ).fetchone()
            return result[0] if result[0] is not None else 0
    except Exception:
        # _schema_version 表不存在 → 从未运行过迁移
        return 0


def _set_version(engine, version: int):
    """记录迁移 N 已执行。"""
    from sqlalchemy import text
    with engine.begin() as conn:
        conn.execute(
            text("INSERT OR REPLACE INTO _schema_version (version, applied_at) "
                 "VALUES (:v, datetime('now'))"),
            {"v": version},
        )


def _discover_migrations() -> list[tuple[int, str]]:
    """扫描 storage/migrations/ 目录，返回按版本号排序的 (version, filename) 列表。"""
    if not os.path.isdir(_MIGRATIONS_DIR):
        return []
    migrations = []
    for fname in sorted(os.listdir(_MIGRATIONS_DIR)):
        m = re.match(r"^(\d{3})_.*\.py$", fname)
        if m and fname != "__init__.py":
            migrations.append((int(m.group(1)), fname))
    return sorted(migrations)


def run_migrations(engine=None) -> int:
    """执行所有未应用的迁移。

    在 storage.init() 中自动调用。安全多次调用 — 已执行的迁移会跳过。

    Args:
        engine: SQLAlchemy Engine。如果为 None，自动从 db.get_engine() 获取。

    Returns:
        int: 本次新应用的迁移数量。

    Raises:
        RuntimeError: 如果某个迁移执行失败。
    """
    if engine is None:
        from storage.db import get_engine
        engine = get_engine()

    # 确保 _schema_version 表存在
    from storage.models import SchemaVersion, Base
    # 只创建 _schema_version 表 (如果不存在)
    SchemaVersion.__table__.create(engine, checkfirst=True)

    current = _get_current_version(engine)
    available = _discover_migrations()

    applied = 0
    for version, fname in available:
        if version <= current:
            logger.debug(f"Migration {version} already applied, skipping")
            continue

        mod_name = f"storage.migrations.{fname[:-3]}"
        try:
            # 使用文件路径加载 (因为 001_initial 不是合法的 Python 模块名)
            module_path = os.path.join(_MIGRATIONS_DIR, fname)
            spec = importlib.util.spec_from_file_location(mod_name, module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            logger.info(f"Applying migration {version:03d}: {fname}")
            mod.upgrade(engine)
            _set_version(engine, version)
            applied += 1
            logger.info(f"Migration {version:03d} applied successfully")
        except Exception as e:
            logger.error(f"Migration {version:03d} FAILED: {e}")
            raise RuntimeError(f"Migration {version:03d} ({fname}) failed: {e}") from e

    if applied == 0 and current == 0:
        logger.info("No migrations found; DB is at initial state")
    elif applied == 0:
        logger.info(f"DB is up-to-date at version {current}")

    return applied
