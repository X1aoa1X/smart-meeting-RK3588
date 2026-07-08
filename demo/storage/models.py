"""SQLAlchemy ORM 模型 — 6 张表 + _schema_version 迁移追踪表。

纯 Python，零 Qt/PyQt5 依赖，可在 headless 脚本中导入。
"""

import enum
import json
from datetime import datetime

from sqlalchemy import (Column, Integer, String, Float, Text, DateTime,
                        ForeignKey, Index, UniqueConstraint)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ═══════════════════════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════════════════════

class MeetingStatus(enum.Enum):
    PLANNED      = "planned"
    IN_PROGRESS  = "in_progress"
    COMPLETED    = "completed"
    CANCELLED    = "cancelled"

    @classmethod
    def valid_transition(cls, from_status: str, to_status: str) -> bool:
        """验证会议状态迁移是否合法。"""
        if to_status == cls.CANCELLED.value:
            return True  # 任何状态都可以取消
        if from_status == cls.PLANNED.value and to_status == cls.IN_PROGRESS.value:
            return True
        if from_status == cls.IN_PROGRESS.value and to_status == cls.COMPLETED.value:
            return True
        return False


class SegmentSource:
    APRILTAG = "AprilTag"
    MANUAL   = "manual"
    UNKNOWN  = "unknown"


class NoteType(enum.Enum):
    JUDGE_QUESTION = "评委问题"
    KEY_CONCLUSION = "重点结论"
    TODO_ITEM      = "待办事项"
    SYSTEM_ISSUE   = "系统异常"
    HOST_NOTE      = "主持人备注"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Participant — 人员表
# ═══════════════════════════════════════════════════════════════════════════════

class Participant(Base):
    __tablename__ = "participants"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    tag_id       = Column(String(32), unique=True, nullable=False, index=True)
    name         = Column(String(128), nullable=False)
    organization = Column(String(256), default="")
    role         = Column(String(64), default="参会人员")
    title        = Column(String(128), default="")
    phone        = Column(String(32), default="")
    email        = Column(String(128), default="")
    avatar_path  = Column(String(512), default="")
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    segments = relationship("SpeakerSegment", back_populates="participant",
                            foreign_keys="SpeakerSegment.speaker_tag_id",
                            primaryjoin="Participant.tag_id == SpeakerSegment.speaker_tag_id")

    def __repr__(self):
        return f"<Participant tag_id={self.tag_id} name={self.name}>"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Meeting — 会议表
# ═══════════════════════════════════════════════════════════════════════════════

class Meeting(Base):
    __tablename__ = "meetings"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    name        = Column(String(256), nullable=False)
    location    = Column(String(256), default="")
    start_time  = Column(DateTime, nullable=True)        # 实际开始时间（点击"开始会议"时设置）
    end_time    = Column(DateTime, nullable=True)        # 实际结束时间（点击"结束会议"时设置）
    status      = Column(String(32), default=MeetingStatus.PLANNED.value)
    description = Column(Text, default="")
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    segments = relationship("SpeakerSegment", back_populates="meeting",
                            cascade="all, delete-orphan")
    events   = relationship("Event", back_populates="meeting",
                            cascade="all, delete-orphan")
    notes    = relationship("HostNote", back_populates="meeting",
                            cascade="all, delete-orphan")
    agent_decisions = relationship("AgentDecision", back_populates="meeting",
                                   cascade="all, delete-orphan")
    tts_events = relationship("TTSEvent", back_populates="meeting",
                              cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Meeting id={self.id} name={self.name} status={self.status}>"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SpeakerSegment — 发言片段表
# ═══════════════════════════════════════════════════════════════════════════════

class SpeakerSegment(Base):
    __tablename__ = "speaker_segments"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id       = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    speaker_tag_id   = Column(String(32), nullable=True, index=True)
    speaker_name     = Column(String(128), nullable=True)
    role             = Column(String(64), nullable=True)
    source           = Column(String(32), default=SegmentSource.UNKNOWN)
    start_time       = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time         = Column(DateTime, nullable=True)    # NULL = 当前正在发言
    duration_seconds = Column(Float, default=0.0)         # 结束时计算
    confidence       = Column(Float, default=0.0)

    # Relationships
    meeting     = relationship("Meeting", back_populates="segments")
    participant = relationship("Participant", back_populates="segments",
                               foreign_keys=[speaker_tag_id],
                               primaryjoin="SpeakerSegment.speaker_tag_id == Participant.tag_id")

    def __repr__(self):
        active = " (active)" if self.end_time is None else ""
        return (f"<SpeakerSegment id={self.id} speaker={self.speaker_name}"
                f" duration={self.duration_seconds:.1f}s{active}>")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Event — 事件表
# ═══════════════════════════════════════════════════════════════════════════════

class Event(Base):
    __tablename__ = "events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id   = Column(Integer, ForeignKey("meetings.id", ondelete="SET NULL"),
                         nullable=True, index=True)
    event_type   = Column(String(64), nullable=False, index=True)
    timestamp    = Column(DateTime, nullable=False, default=datetime.utcnow)
    payload_json = Column(Text, nullable=True)  # 灵活的 JSON 附加字段

    # Relationships
    meeting = relationship("Meeting", back_populates="events")

    __table_args__ = (
        Index("idx_events_meeting_type", "meeting_id", "event_type"),
        Index("idx_events_timestamp", "timestamp"),
    )

    def __repr__(self):
        return f"<Event type={self.event_type} ts={self.timestamp}>"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. HostNote — 主持人备注表
# ═══════════════════════════════════════════════════════════════════════════════

class HostNote(Base):
    __tablename__ = "host_notes"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id      = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"),
                            nullable=False, index=True)
    timestamp       = Column(DateTime, nullable=False, default=datetime.utcnow)
    related_speaker = Column(String(128), nullable=True)  # tag_id 或姓名
    note_type       = Column(String(32), nullable=False)
    content         = Column(Text, nullable=False)

    # Relationships
    meeting = relationship("Meeting", back_populates="notes")

    def __repr__(self):
        return f"<HostNote id={self.id} type={self.note_type}>"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. AgentDecision — Agent 决策审计表
# ═══════════════════════════════════════════════════════════════════════════════

class AgentDecision(Base):
    """Agent 决策审计日志 — 每次触发评估都记录（播报或抑制）。

    可解释性核心：评委可以看到"系统为什么说话 / 为什么沉默"。
    """
    __tablename__ = "agent_decisions"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id          = Column(Integer, ForeignKey("meetings.id", ondelete="SET NULL"),
                                nullable=True, index=True)
    trigger_type        = Column(String(64), nullable=False, index=True)
    trigger_key         = Column(String(256), nullable=False)          # 去重键
    priority            = Column(Integer, default=0)
    input_summary       = Column(Text, nullable=True)   # JSON context 或摘要
    rule_reason         = Column(Text, nullable=True)   # 规则触发原因
    llm_used            = Column(Integer, default=0)    # 0 或 1
    llm_prompt_tokens   = Column(Integer, default=0)
    llm_completion_tokens = Column(Integer, default=0)
    decision            = Column(String(32), nullable=False,
                                default="suppressed")   # spoken|suppressed|error|expired
    final_text          = Column(Text, nullable=True)   # 最终播报文本（或被抑制的文本）
    suppressed_reason   = Column(String(128), nullable=True)
    created_at          = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    meeting = relationship("Meeting", back_populates="agent_decisions")

    def __repr__(self):
        return (f"<AgentDecision trigger={self.trigger_type}"
                f" decision={self.decision} llm={self.llm_used}>")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TTSEvent — TTS 播报事件表
# ═══════════════════════════════════════════════════════════════════════════════

class TTSEvent(Base):
    """每次 TTS 播报的审计日志 — 说了什么、为什么说、什么时候说的。"""
    __tablename__ = "tts_events"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id    = Column(Integer, ForeignKey("meetings.id", ondelete="SET NULL"),
                          nullable=True, index=True)
    text          = Column(Text, nullable=False)
    source        = Column(String(32), nullable=False)   # fixed_rule|llm_agent|host_manual|system
    priority      = Column(Integer, default=0)
    status        = Column(String(16), nullable=False,
                          default="queued")   # queued|speaking|spoken|skipped|failed
    cooldown_key  = Column(String(256), nullable=True)
    reason        = Column(Text, nullable=True)
    created_at    = Column(DateTime, nullable=False, default=datetime.utcnow)
    spoken_at     = Column(DateTime, nullable=True)

    # Relationships
    meeting = relationship("Meeting", back_populates="tts_events")

    def __repr__(self):
        return (f"<TTSEvent text={self.text[:30]}..."
                f" src={self.source} status={self.status}>")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. SystemConfig — 系统配置表 (替代 fusion_params.json 等)
# ═══════════════════════════════════════════════════════════════════════════════

class SystemConfig(Base):
    """键值对配置存储。

    按 (config_section, config_key) 唯一标识一条配置。
    替代多个散落的 JSON 文件：
      - section="fusion" → 原 fusion_params.json 中的融合参数
      - section="vad"    → 原 fusion_params.json 中的 VAD 参数
      - section="calibration" → 原 xvf_calibration.json
    """

    __tablename__ = "system_config"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    config_section  = Column(String(64), nullable=False)
    config_key      = Column(String(64), nullable=False)
    config_value    = Column(Text, nullable=False)    # JSON 序列化 (非字符串类型时)
    description     = Column(String(256), default="")
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("config_section", "config_key", name="uq_config_section_key"),
    )

    def __repr__(self):
        return f"<SystemConfig [{self.config_section}] {self.config_key}={self.config_value[:40]}>"


# ═══════════════════════════════════════════════════════════════════════════════
# _schema_version — 迁移版本追踪 (仅用于 migration.py, 非业务模型)
# ═══════════════════════════════════════════════════════════════════════════════

class SchemaVersion(Base):
    """迁移版本追踪表。记录已应用的迁移编号。"""
    __tablename__ = "_schema_version"

    version    = Column(Integer, primary_key=True)
    applied_at = Column(String(32), nullable=False)

    def __repr__(self):
        return f"<SchemaVersion {self.version}>"
