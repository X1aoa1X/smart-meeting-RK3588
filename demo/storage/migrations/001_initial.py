"""Migration 001: 创建所有初始表 + 从旧 JSON 文件导入配置。

创建 6 张业务表:
  - participants, meetings, speaker_segments, events, host_notes, system_config

从旧文件导入配置:
  - fusion_params.json → system_config (section="fusion" + "vad")
  - xvf_calibration.json → system_config (section="calibration")

导入后重命名旧文件为 *.imported，防止重复导入。
"""

import os
import json
import logging

logger = logging.getLogger(__name__)


def upgrade(engine):
    """执行初始迁移: 创建所有表 + 导入旧 JSON 配置。"""
    from storage.models import Base

    # 创建所有表 (checkfirst=True 确保安全多次调用)
    Base.metadata.create_all(engine)

    # 导入旧的 JSON 配置文件
    _import_json_configs(engine)


def _import_json_configs(engine):
    """从旧 JSON 文件导入配置到 system_config 表。"""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # ── 1. 导入 fusion_params.json ─────────────────────────────────────────
    _import_fusion_params(project_root, engine)

    # ── 2. 导入 xvf_calibration.json ───────────────────────────────────────
    _import_calibration(project_root, engine)


def _import_fusion_params(project_root: str, engine):
    """导入 fusion_params.json 到 system_config (fusion 和 vad section)。"""
    old_path = os.path.join(project_root, "fusion_params.json")
    if not os.path.exists(old_path):
        return

    try:
        with open(old_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        from storage.db import session_scope
        from storage.models import SystemConfig

        # 分离 fusion 和 vad 参数
        VAD_KEYS = {"vad_enabled", "vad_speech_duration", "vad_threshold",
                     "vad_pregain", "vad_device", "vad_capture_rate"}

        with session_scope() as session:
            for key, value in params.items():
                section = "vad" if key in VAD_KEYS else "fusion"
                # 查找是否已存在 (支持重新运行迁移)
                existing = session.query(SystemConfig).filter_by(
                    config_section=section, config_key=key
                ).first()
                if existing is not None:
                    continue  # 已导入过，跳过

                session.add(SystemConfig(
                    config_section=section,
                    config_key=key,
                    config_value=json.dumps(value) if not isinstance(value, str) else value,
                    description=f"Imported from fusion_params.json",
                ))
            session.commit()

        # 重命名旧文件，防止重复导入
        imported_path = old_path + ".imported"
        os.rename(old_path, imported_path)
        logger.info(f"Imported {len(params)} keys from fusion_params.json → system_config, "
                     f"renamed to {os.path.basename(imported_path)}")

    except Exception as e:
        logger.warning(f"Could not import fusion_params.json: {e} (file left untouched)")


def _import_calibration(project_root: str, engine):
    """导入 xvf_calibration.json 到 system_config (calibration section)。"""
    old_path = os.path.join(project_root, "xvf_calibration.json")
    if not os.path.exists(old_path):
        return

    try:
        with open(old_path, "r", encoding="utf-8") as f:
            calib = json.load(f)

        from storage.db import session_scope
        from storage.models import SystemConfig

        with session_scope() as session:
            for key, value in calib.items():
                existing = session.query(SystemConfig).filter_by(
                    config_section="calibration", config_key=key
                ).first()
                if existing is not None:
                    continue

                if isinstance(value, (list, dict)):
                    stored = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, bool):
                    stored = json.dumps(value)
                elif isinstance(value, (int, float)):
                    stored = json.dumps(value)
                else:
                    stored = str(value) if value is not None else ""

                session.add(SystemConfig(
                    config_section="calibration",
                    config_key=key,
                    config_value=stored,
                    description="Imported from xvf_calibration.json",
                ))
            session.commit()

        imported_path = old_path + ".imported"
        os.rename(old_path, imported_path)
        logger.info(f"Imported {len(calib)} keys from xvf_calibration.json → system_config, "
                     f"renamed to {os.path.basename(imported_path)}")

    except Exception as e:
        logger.warning(f"Could not import xvf_calibration.json: {e} (file left untouched)")
