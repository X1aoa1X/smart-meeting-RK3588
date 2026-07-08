"""Migration 002 — 添加 AgentDecision 和 TTSEvent 表。

Agent TTS 系统的审计日志表：
  - agent_decisions: 每次规则触发评估记录（播报或抑制）
  - tts_events: 每次 TTS 播报记录
"""


def upgrade(engine):
    """创建 agent_decisions 和 tts_events 表。"""
    from storage.models import Base

    tables = [
        Base.metadata.tables["agent_decisions"],
        Base.metadata.tables["tts_events"],
    ]
    Base.metadata.create_all(engine, tables=tables)
