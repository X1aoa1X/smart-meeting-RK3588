"""Repository / DAO 层 — 为每个 ORM 模型提供 CRUD 和查询方法。

设计模式:
  - 每个 Repository 通过构造函数接收 session (依赖注入)
  - 调用者通过 session_scope() 管理事务边界
  - Repository 方法不提交事务 — 由 session_scope() 统一提交

用法:
    from storage.db import session_scope
    from storage.repo import MeetingRepo, ParticipantRepo

    with session_scope() as session:
        mr = MeetingRepo(session)
        pr = ParticipantRepo(session)
        meeting = mr.create("项目路演")
        pr.bulk_import([{"tag_id": "A001", "name": "王强"}, ...])
"""

import json
from datetime import datetime
from typing import Optional

from sqlalchemy.exc import IntegrityError

from storage.models import (
    Participant,
    Meeting,
    MeetingStatus,
    SpeakerSegment,
    SegmentSource,
    Event,
    HostNote,
    SystemConfig,
    AgentDecision,
    TTSEvent,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Base
# ═══════════════════════════════════════════════════════════════════════════════

class BaseRepo:
    """所有 Repository 的基类。"""

    def __init__(self, session):
        self._session = session


# ═══════════════════════════════════════════════════════════════════════════════
# ParticipantRepo
# ═══════════════════════════════════════════════════════════════════════════════

class ParticipantRepo(BaseRepo):
    """人员仓库。"""

    def create(self, tag_id: str, name: str, **kwargs) -> Participant:
        """创建参会人员。

        Args:
            tag_id: AprilTag ID (如 "A001")，必须唯一
            name: 姓名
            **kwargs: organization, role, title, phone, email, avatar_path

        Returns:
            Participant

        Raises:
            ValueError: tag_id 已存在
        """
        existing = self.get_by_tag_id(tag_id)
        if existing is not None:
            raise ValueError(f"tag_id '{tag_id}' already exists (assigned to '{existing.name}')")

        p = Participant(tag_id=tag_id, name=name, **kwargs)
        self._session.add(p)
        self._session.flush()  # 立即获取 id，但事务由调用者管理
        return p

    def get_by_tag_id(self, tag_id: str) -> Participant | None:
        """按 tag_id 查询。"""
        return self._session.query(Participant).filter_by(tag_id=tag_id).first()

    def get_by_id(self, id: int) -> Participant | None:
        """按主键查询。"""
        return self._session.get(Participant, id)

    def list_all(self, search: str = None) -> list[Participant]:
        """列出所有人员，可按姓名或 tag_id 搜索。"""
        q = self._session.query(Participant)
        if search:
            pattern = f"%{search}%"
            q = q.filter(
                (Participant.name.ilike(pattern)) |
                (Participant.tag_id.ilike(pattern))
            )
        return q.order_by(Participant.tag_id).all()

    def update(self, participant: Participant, **kwargs) -> Participant:
        """更新人员字段。

        Args:
            participant: 要更新的 Participant 实例
            **kwargs: 要更新的字段名=新值

        Returns:
            更新后的 Participant

        Raises:
            ValueError: 如果更新 tag_id 导致冲突
        """
        # 如果更新 tag_id，检查唯一性
        if "tag_id" in kwargs and kwargs["tag_id"] != participant.tag_id:
            existing = self.get_by_tag_id(kwargs["tag_id"])
            if existing is not None and existing.id != participant.id:
                raise ValueError(
                    f"tag_id '{kwargs['tag_id']}' already exists (assigned to '{existing.name}')"
                )

        for key, value in kwargs.items():
            if hasattr(participant, key):
                setattr(participant, key, value)
        participant.updated_at = datetime.utcnow()
        self._session.flush()
        return participant

    def delete(self, participant: Participant):
        """删除人员。"""
        self._session.delete(participant)
        self._session.flush()

    def bulk_import(self, rows: list[dict]) -> dict:
        """批量导入人员名单。

        Args:
            rows: [{"tag_id": "A001", "name": "王强", "organization": "XX大学", ...}, ...]

        Returns:
            {"created": int, "updated": int, "errors": list[str]}
        """
        created, updated, errors = 0, 0, []

        for i, row in enumerate(rows, start=1):
            tag_id = row.get("tag_id", "").strip()
            name = row.get("name", "").strip()

            if not tag_id:
                errors.append(f"Row {i}: tag_id is empty")
                continue
            if not name:
                errors.append(f"Row {i}: name is empty")
                continue

            existing = self.get_by_tag_id(tag_id)
            if existing is not None:
                # 更新已有记录
                for key in ("name", "organization", "role", "title", "phone", "email"):
                    if key in row and row[key]:
                        setattr(existing, key, row[key])
                existing.updated_at = datetime.utcnow()
                updated += 1
            else:
                p = Participant(
                    tag_id=tag_id,
                    name=name,
                    organization=row.get("organization", ""),
                    role=row.get("role", "参会人员"),
                    title=row.get("title", ""),
                    phone=row.get("phone", ""),
                    email=row.get("email", ""),
                )
                self._session.add(p)
                created += 1

        self._session.flush()
        return {"created": created, "updated": updated, "errors": errors}

    def tag_ids(self) -> list[str]:
        """返回所有已分配的 tag_id 列表。"""
        return [r[0] for r in self._session.query(Participant.tag_id).order_by(Participant.tag_id).all()]


# ═══════════════════════════════════════════════════════════════════════════════
# MeetingRepo
# ═══════════════════════════════════════════════════════════════════════════════

class MeetingRepo(BaseRepo):
    """会议仓库。"""

    def create(self, name: str, **kwargs) -> Meeting:
        """创建新会议 (默认状态: planned)。"""
        m = Meeting(name=name, **kwargs)
        self._session.add(m)
        self._session.flush()
        return m

    def get_active(self) -> Meeting | None:
        """获取当前正在进行的会议 (status='in_progress')。

        设计假设: 同一时间最多一个会议在进行中。
        """
        return self._session.query(Meeting).filter_by(
            status=MeetingStatus.IN_PROGRESS.value
        ).first()

    def get_by_id(self, id: int) -> Meeting | None:
        """按主键查询。"""
        return self._session.get(Meeting, id)

    def list_all(self, status: str = None) -> list[Meeting]:
        """列出所有会议，可按状态筛选。"""
        q = self._session.query(Meeting)
        if status:
            q = q.filter_by(status=status)
        return q.order_by(Meeting.created_at.desc()).all()

    def start_meeting(self, meeting: Meeting) -> Meeting:
        """开始会议: planned → in_progress。"""
        if not MeetingStatus.valid_transition(meeting.status, MeetingStatus.IN_PROGRESS.value):
            raise ValueError(
                f"Cannot start meeting: invalid transition {meeting.status} → in_progress"
            )
        meeting.status = MeetingStatus.IN_PROGRESS.value
        meeting.start_time = datetime.utcnow()
        meeting.updated_at = datetime.utcnow()
        self._session.flush()
        return meeting

    def end_meeting(self, meeting: Meeting) -> Meeting:
        """结束会议: in_progress → completed。"""
        if not MeetingStatus.valid_transition(meeting.status, MeetingStatus.COMPLETED.value):
            raise ValueError(
                f"Cannot end meeting: invalid transition {meeting.status} → completed"
            )
        # 结束所有活跃的发言片段
        active_segments = self._session.query(SpeakerSegment).filter_by(
            meeting_id=meeting.id, end_time=None
        ).all()
        now = datetime.utcnow()
        for seg in active_segments:
            seg.end_time = now
            seg.duration_seconds = (now - seg.start_time).total_seconds()

        meeting.status = MeetingStatus.COMPLETED.value
        meeting.end_time = now
        meeting.updated_at = now
        self._session.flush()
        return meeting

    def cancel_meeting(self, meeting: Meeting) -> Meeting:
        """取消会议: any → cancelled。"""
        meeting.status = MeetingStatus.CANCELLED.value
        meeting.end_time = datetime.utcnow()
        meeting.updated_at = datetime.utcnow()
        self._session.flush()
        return meeting


# ═══════════════════════════════════════════════════════════════════════════════
# SpeakerSegmentRepo
# ═══════════════════════════════════════════════════════════════════════════════

class SpeakerSegmentRepo(BaseRepo):
    """发言片段仓库。

    核心不变式: 同一会议中最多一个活跃片段 (end_time IS NULL)。
    start_segment() 自动结束当前活跃片段再创建新的。
    """

    def start_segment(
        self,
        meeting_id: int,
        speaker_tag_id: str | None = None,
        speaker_name: str | None = None,
        source: str = SegmentSource.UNKNOWN,
        start_time: datetime | None = None,
        confidence: float = 0.0,
        role: str | None = None,
    ) -> SpeakerSegment:
        """开始一个新的发言片段。

        自动结束当前活跃片段 (如果存在)。

        Args:
            meeting_id: 会议 ID
            speaker_tag_id: 发言人 tag_id (可选，未知时为 None)
            speaker_name: 发言人姓名 (可选)
            source: 识别来源 (AprilTag/manual/unknown)
            start_time: 开始时间 (默认 now)
            confidence: 置信度 0.0-1.0
            role: 发言人角色

        Returns:
            新创建的 SpeakerSegment (end_time=NULL)
        """
        # 结束当前活跃片段
        active = self.get_active_segment(meeting_id)
        if active is not None:
            self.end_segment(active)

        seg = SpeakerSegment(
            meeting_id=meeting_id,
            speaker_tag_id=speaker_tag_id,
            speaker_name=speaker_name,
            role=role,
            source=source,
            start_time=start_time or datetime.utcnow(),
            end_time=None,  # 正在发言
            confidence=confidence,
        )
        self._session.add(seg)
        self._session.flush()
        return seg

    def end_segment(self, segment: SpeakerSegment, end_time: datetime | None = None) -> SpeakerSegment:
        """结束发言片段，自动计算 duration_seconds。"""
        now = end_time or datetime.utcnow()
        segment.end_time = now
        segment.duration_seconds = (now - segment.start_time).total_seconds()
        self._session.flush()
        return segment

    def get_active_segment(self, meeting_id: int) -> SpeakerSegment | None:
        """获取会议当前活跃的发言片段 (end_time IS NULL)。"""
        return self._session.query(SpeakerSegment).filter_by(
            meeting_id=meeting_id, end_time=None
        ).first()

    def get_segments_for_meeting(self, meeting_id: int) -> list[SpeakerSegment]:
        """获取会议的所有发言片段 (按时间排序)。"""
        return self._session.query(SpeakerSegment).filter_by(
            meeting_id=meeting_id
        ).order_by(SpeakerSegment.start_time).all()

    def get_total_duration(self, meeting_id: int, speaker_tag_id: str | None = None) -> float:
        """计算指定会议/发言人的总发言时长 (秒)。

        Args:
            meeting_id: 会议 ID
            speaker_tag_id: 可选，按发言人过滤
        """
        q = self._session.query(SpeakerSegment).filter_by(meeting_id=meeting_id)
        if speaker_tag_id:
            q = q.filter_by(speaker_tag_id=speaker_tag_id)
        segments = q.all()
        return sum(s.duration_seconds or 0.0 for s in segments)


# ═══════════════════════════════════════════════════════════════════════════════
# EventRepo
# ═══════════════════════════════════════════════════════════════════════════════

class EventRepo(BaseRepo):
    """事件仓库。

    支持单条插入和批量插入 (用于 EventBridge 的批量写入)。
    """

    def log_event(
        self,
        event_type: str,
        meeting_id: int | None = None,
        timestamp: datetime | None = None,
        payload: dict | None = None,
    ) -> Event:
        """记录单条事件。

        Args:
            event_type: 事件类型 (如 "state_changed", "speaker_started")
            meeting_id: 所属会议 ID (可选)
            timestamp: 事件时间 (默认 now)
            payload: 附加 JSON 载荷

        Returns:
            Event
        """
        payload_json = json.dumps(payload, ensure_ascii=False) if payload else None
        evt = Event(
            meeting_id=meeting_id,
            event_type=event_type,
            timestamp=timestamp or datetime.utcnow(),
            payload_json=payload_json,
        )
        self._session.add(evt)
        self._session.flush()
        return evt

    def log_batch(self, events_data: list[dict]) -> list[Event]:
        """批量插入事件 (高性能，用于 EventBridge 的批量刷新)。

        Args:
            events_data: [{"event_type": str, "meeting_id": int|None,
                           "timestamp": datetime|float, "payload": dict|None}, ...]

        Returns:
            创建的 Event 对象列表
        """
        results = []
        for data in events_data:
            payload = data.get("payload") or data.get("payload_json")
            payload_json = json.dumps(payload, ensure_ascii=False) if payload else None

            ts = data.get("timestamp", datetime.utcnow())
            if isinstance(ts, (int, float)):
                ts = datetime.utcfromtimestamp(ts)

            evt = Event(
                meeting_id=data.get("meeting_id"),
                event_type=data.get("event_type", "unknown"),
                timestamp=ts,
                payload_json=payload_json,
            )
            self._session.add(evt)
            results.append(evt)
        self._session.flush()
        return results

    def get_recent(
        self,
        minutes: float = 10.0,
        meeting_id: int | None = None,
        event_types: list[str] | None = None,
    ) -> list[Event]:
        """获取最近的 N 分钟事件。

        Args:
            minutes: 时间窗口 (分钟)
            meeting_id: 按会议过滤 (可选)
            event_types: 按事件类型过滤 (可选)
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        q = self._session.query(Event).filter(Event.timestamp >= cutoff)
        if meeting_id is not None:
            q = q.filter_by(meeting_id=meeting_id)
        if event_types:
            q = q.filter(Event.event_type.in_(event_types))
        return q.order_by(Event.timestamp.asc()).all()

    def get_for_meeting(
        self,
        meeting_id: int,
        event_types: list[str] | None = None,
    ) -> list[Event]:
        """获取会议的所有事件 (按时间排序)。

        Args:
            meeting_id: 会议 ID
            event_types: 按事件类型过滤 (可选)
        """
        q = self._session.query(Event).filter_by(meeting_id=meeting_id)
        if event_types:
            q = q.filter(Event.event_type.in_(event_types))
        return q.order_by(Event.timestamp.asc()).all()

    def count(self, meeting_id: int | None = None, event_type: str | None = None) -> int:
        """统计事件数量。"""
        q = self._session.query(Event)
        if meeting_id is not None:
            q = q.filter_by(meeting_id=meeting_id)
        if event_type:
            q = q.filter_by(event_type=event_type)
        return q.count()


# ═══════════════════════════════════════════════════════════════════════════════
# HostNoteRepo
# ═══════════════════════════════════════════════════════════════════════════════

class HostNoteRepo(BaseRepo):
    """主持人备注仓库。"""

    def create_note(
        self,
        meeting_id: int,
        note_type: str,
        content: str,
        related_speaker: str | None = None,
        timestamp: datetime | None = None,
    ) -> HostNote:
        """创建一条主持人备注。

        Args:
            meeting_id: 会议 ID
            note_type: 备注类型 (NoteType 枚举值)
            content: 备注内容
            related_speaker: 相关发言人 (tag_id 或姓名，可选)
            timestamp: 备注时间 (默认 now)
        """
        note = HostNote(
            meeting_id=meeting_id,
            note_type=note_type,
            content=content,
            related_speaker=related_speaker,
            timestamp=timestamp or datetime.utcnow(),
        )
        self._session.add(note)
        self._session.flush()
        return note

    def get_notes_for_meeting(
        self,
        meeting_id: int,
        note_type: str | None = None,
    ) -> list[HostNote]:
        """获取会议的所有备注。"""
        q = self._session.query(HostNote).filter_by(meeting_id=meeting_id)
        if note_type:
            q = q.filter_by(note_type=note_type)
        return q.order_by(HostNote.timestamp.asc()).all()


# ═══════════════════════════════════════════════════════════════════════════════
# ConfigRepo
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigRepo(BaseRepo):
    """系统配置仓库。

    配置值存储为字符串。非字符串类型通过 JSON 序列化/反序列化。
    """

    def get(self, section: str, key: str, default=None):
        """获取单个配置值 (自动 JSON 反序列化)。

        Args:
            section: 配置分区 (如 "fusion", "vad", "calibration")
            key: 配置键
            default: 不存在时的默认值

        Returns:
            反序列化后的 Python 对象
        """
        row = self._session.query(SystemConfig).filter_by(
            config_section=section, config_key=key
        ).first()
        if row is None:
            return default
        return self._deserialize(row.config_value)

    def set(self, section: str, key: str, value):
        """设置单个配置值 (自动 JSON 序列化)。

        Args:
            section: 配置分区
            key: 配置键
            value: 任意可 JSON 序列化的 Python 值
        """
        row = self._session.query(SystemConfig).filter_by(
            config_section=section, config_key=key
        ).first()
        serialized = self._serialize(value)

        if row is not None:
            row.config_value = serialized
            row.updated_at = datetime.utcnow()
        else:
            row = SystemConfig(
                config_section=section,
                config_key=key,
                config_value=serialized,
            )
            self._session.add(row)
        self._session.flush()

    def get_section(self, section: str) -> dict:
        """获取整个分区的所有配置 (返回反序列化的 dict)。

        Args:
            section: 配置分区

        Returns:
            {key: deserialized_value, ...}
        """
        rows = self._session.query(SystemConfig).filter_by(
            config_section=section
        ).all()
        return {row.config_key: self._deserialize(row.config_value) for row in rows}

    def set_section(self, section: str, values: dict):
        """批量设置整个分区的配置。

        Args:
            section: 配置分区
            values: {key: value, ...}
        """
        for key, value in values.items():
            self.set(section, key, value)

    def delete_section(self, section: str):
        """删除整个分区的所有配置。"""
        self._session.query(SystemConfig).filter_by(
            config_section=section
        ).delete()
        self._session.flush()

    # ── 类型转换辅助 ────────────────────────────────────────────────────────

    @staticmethod
    def _serialize(value) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return json.dumps(value)
        if isinstance(value, (int, float)):
            return json.dumps(value)
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _deserialize(raw: str):
        """尝试 JSON 反序列化；失败则返回原始字符串。"""
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw


# ═══════════════════════════════════════════════════════════════════════════════
# AgentDecisionRepo
# ═══════════════════════════════════════════════════════════════════════════════

class AgentDecisionRepo(BaseRepo):
    """Agent 决策审计仓库。"""

    def log_decision(
        self,
        meeting_id: int | None,
        trigger_type: str,
        trigger_key: str,
        priority: int = 0,
        input_summary: str | None = None,
        rule_reason: str | None = None,
        llm_used: int = 0,
        llm_prompt_tokens: int = 0,
        llm_completion_tokens: int = 0,
        decision: str = "suppressed",
        final_text: str | None = None,
        suppressed_reason: str | None = None,
    ) -> AgentDecision:
        """记录一次 Agent 决策。

        Args:
            meeting_id: 会议 ID
            trigger_type: 触发类型 (speaker_overtime, silence_timeout 等)
            trigger_key: 去重键
            priority: 优先级 0-100
            input_summary: 输入摘要（发往 LLM 的 JSON 或规则摘要）
            rule_reason: 规则触发原因
            llm_used: 是否调用了 LLM (0/1)
            llm_prompt_tokens: LLM prompt token 数
            llm_completion_tokens: LLM completion token 数
            decision: spoken | suppressed | error | expired
            final_text: 最终播报文本
            suppressed_reason: 抑制原因

        Returns:
            AgentDecision
        """
        d = AgentDecision(
            meeting_id=meeting_id,
            trigger_type=trigger_type,
            trigger_key=trigger_key,
            priority=priority,
            input_summary=input_summary,
            rule_reason=rule_reason,
            llm_used=llm_used,
            llm_prompt_tokens=llm_prompt_tokens,
            llm_completion_tokens=llm_completion_tokens,
            decision=decision,
            final_text=final_text,
            suppressed_reason=suppressed_reason,
        )
        self._session.add(d)
        self._session.flush()
        return d

    def get_recent(self, meeting_id: int | None = None,
                   limit: int = 50) -> list[AgentDecision]:
        """获取最近的决策记录。

        Args:
            meeting_id: 按会议过滤 (可选，None=全部)
            limit: 最大返回条数
        """
        q = self._session.query(AgentDecision)
        if meeting_id is not None:
            q = q.filter_by(meeting_id=meeting_id)
        return q.order_by(AgentDecision.created_at.desc()).limit(limit).all()

    def count_by_trigger(self, meeting_id: int) -> dict:
        """统计每种触发类型的决策次数。

        Returns:
            {trigger_type: count, ...}
        """
        from sqlalchemy import func
        rows = (
            self._session.query(
                AgentDecision.trigger_type,
                func.count(AgentDecision.id),
            )
            .filter_by(meeting_id=meeting_id)
            .group_by(AgentDecision.trigger_type)
            .all()
        )
        return {trigger_type: count for trigger_type, count in rows}

    def count_llm_calls(self, meeting_id: int) -> int:
        """统计指定会议的 LLM 调用次数。"""
        return (
            self._session.query(AgentDecision)
            .filter_by(meeting_id=meeting_id, llm_used=1)
            .count()
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TTSEventRepo
# ═══════════════════════════════════════════════════════════════════════════════

class TTSEventRepo(BaseRepo):
    """TTS 播报事件仓库。"""

    def log(
        self,
        meeting_id: int | None,
        text: str,
        source: str,
        priority: int = 0,
        status: str = "queued",
        cooldown_key: str | None = None,
        reason: str | None = None,
    ) -> TTSEvent:
        """记录一次 TTS 播报事件。

        Args:
            meeting_id: 会议 ID
            text: 播报文本
            source: 来源 (fixed_rule|llm_agent|host_manual|system)
            priority: 优先级 0-100
            status: queued|speaking|spoken|skipped|failed
            cooldown_key: 去重键
            reason: 播报原因

        Returns:
            TTSEvent
        """
        evt = TTSEvent(
            meeting_id=meeting_id,
            text=text,
            source=source,
            priority=priority,
            status=status,
            cooldown_key=cooldown_key,
            reason=reason,
        )
        self._session.add(evt)
        self._session.flush()
        return evt

    def update_status(self, tts_event_id: int, status: str,
                      spoken_at: datetime | None = None):
        """更新 TTS 事件状态。

        Args:
            tts_event_id: TTSEvent ID
            status: 新状态 (spoken|skipped|failed)
            spoken_at: 播报完成时间
        """
        evt = self._session.get(TTSEvent, tts_event_id)
        if evt is not None:
            evt.status = status
            if spoken_at:
                evt.spoken_at = spoken_at
            elif status == "spoken":
                evt.spoken_at = datetime.utcnow()
            self._session.flush()

    def get_recent(self, meeting_id: int | None = None,
                   limit: int = 50) -> list[TTSEvent]:
        """获取最近的 TTS 播报记录。"""
        q = self._session.query(TTSEvent)
        if meeting_id is not None:
            q = q.filter_by(meeting_id=meeting_id)
        return q.order_by(TTSEvent.created_at.desc()).limit(limit).all()

    def count_recent(self, meeting_id: int | None = None,
                     minutes: float = 5.0) -> int:
        """统计最近 N 分钟内的 TTS 播报次数。"""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        q = self._session.query(TTSEvent).filter(
            TTSEvent.created_at >= cutoff,
            TTSEvent.status.in_(["spoken", "speaking"]),
        )
        if meeting_id is not None:
            q = q.filter_by(meeting_id=meeting_id)
        return q.count()
