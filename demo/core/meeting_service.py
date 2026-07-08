"""Meeting Service — 会议生命周期业务逻辑层。

封装 MeetingRepo、SpeakerSegmentRepo 的操作，提供:
  - 单活跃会议约束（同时最多一个会议在进行中）
  - 会议启动/结束/暂停/恢复 生命周期管理
  - EventBus 事件发布（meeting_started, meeting_ended, meeting_paused, meeting_resumed）
  - 会议摘要统计（发言人累计时长、切换次数等）

纯 Python，零 Qt/PyQt5 依赖，在 headless 和 GUI 环境均可使用。

用法:
  from core.meeting_service import MeetingService
  from storage import init as storage_init

  storage_init()
  svc = MeetingService()
  meeting = svc.start_meeting_by_id(1)  # 开始 meeting_id=1 的会议
  svc.end_active_meeting()               # 结束当前会议
"""

import logging
from datetime import datetime
from typing import Optional

from storage.models import Meeting, MeetingStatus

logger = logging.getLogger(__name__)


class MeetingService:
    """会议生命周期管理服务。

    封装了 MeetingRepo 和 SpeakerSegmentRepo 的业务操作，
    通过 EventBus 发布生命周期事件。
    """

    def __init__(self):
        self._active_meeting_id: int | None = None

    # ═════════════════════════════════════════════════════════════════════
    # 活跃会议查询
    # ═════════════════════════════════════════════════════════════════════

    def get_active_meeting(self) -> Meeting | None:
        """获取当前正在进行中的会议。

        Returns:
            Meeting 或 None（如果没有活跃会议）
        """
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo

            with session_scope() as session:
                repo = MeetingRepo(session)
                meeting = repo.get_active()
                if meeting is not None:
                    # 刷新内部缓存
                    self._active_meeting_id = meeting.id
                else:
                    self._active_meeting_id = None
                return meeting
        except Exception as e:
            logger.error(f"查询活跃会议失败: {e}")
            return None

    def get_active_meeting_id(self) -> int | None:
        """获取当前活跃会议的 ID。

        Returns:
            meeting_id (int) 或 None
        """
        if self._active_meeting_id is not None:
            return self._active_meeting_id
        meeting = self.get_active_meeting()
        return meeting.id if meeting else None

    def active_meeting_exists(self) -> bool:
        """检查是否有会议正在进行中。"""
        return self.get_active_meeting() is not None

    # ═════════════════════════════════════════════════════════════════════
    # 会议生命周期
    # ═════════════════════════════════════════════════════════════════════

    def start_meeting_by_id(self, meeting_id: int) -> Meeting:
        """通过 ID 开始一个已创建的会议 (planned → in_progress)。

        约束: 如果已有活跃会议，会先自动结束它。

        Args:
            meeting_id: 会议 ID

        Returns:
            已开始的 Meeting 实例

        Raises:
            ValueError: 会议不存在或状态转换非法
        """
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo

            with session_scope() as session:
                repo = MeetingRepo(session)

                # 检查是否已有活跃会议
                active = repo.get_active()
                if active is not None and active.id != meeting_id:
                    logger.warning(f"检测到已有活跃会议 (id={active.id})，自动结束")
                    repo.end_meeting(active)
                    self._publish_event("meeting_ended", {
                        "meeting_id": active.id,
                        "meeting_name": active.name,
                        "reason": "auto_ended_by_new_meeting",
                    })

                # 开始目标会议
                meeting = repo.get_by_id(meeting_id)
                if meeting is None:
                    raise ValueError(f"会议不存在: id={meeting_id}")

                meeting = repo.start_meeting(meeting)
                self._active_meeting_id = meeting.id

                # 发布事件
                self._publish_event("meeting_started", {
                    "meeting_id": meeting.id,
                    "meeting_name": meeting.name,
                    "location": meeting.location or "",
                })

                logger.info(f"会议已开始: [{meeting.id}] {meeting.name}")
                return meeting
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"开始会议失败: {e}")
            raise

    def end_active_meeting(self) -> Meeting | None:
        """结束当前活跃会议 (in_progress → completed)。

        自动结束所有活跃发言片段并计算时长。

        Returns:
            已结束的 Meeting 实例，或 None（没有活跃会议）
        """
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo

            with session_scope() as session:
                repo = MeetingRepo(session)
                meeting = repo.get_active()
                if meeting is None:
                    logger.warning("没有活跃会议可结束")
                    return None

                meeting = repo.end_meeting(meeting)
                meeting_id = meeting.id
                meeting_name = meeting.name
                self._active_meeting_id = None

                # 发布事件
                duration = None
                if meeting.start_time and meeting.end_time:
                    duration = (meeting.end_time - meeting.start_time).total_seconds()

                self._publish_event("meeting_ended", {
                    "meeting_id": meeting_id,
                    "meeting_name": meeting_name,
                    "duration_seconds": duration,
                })

                logger.info(f"会议已结束: [{meeting_id}] {meeting_name} "
                           f"(时长: {duration:.0f}s)" if duration else "")
                return meeting
        except Exception as e:
            logger.error(f"结束会议失败: {e}")
            raise

    def pause_tracking(self):
        """暂停追踪 — 发布 meeting_paused 事件。"""
        mid = self.get_active_meeting_id()
        self._publish_event("meeting_paused", {
            "meeting_id": mid,
            "reason": "host_paused",
        })
        logger.info("追踪已暂停")

    def resume_tracking(self):
        """恢复追踪 — 发布 meeting_resumed 事件。"""
        mid = self.get_active_meeting_id()
        self._publish_event("meeting_resumed", {
            "meeting_id": mid,
            "reason": "host_resumed",
        })
        logger.info("追踪已恢复")

    # ═════════════════════════════════════════════════════════════════════
    # 会议摘要统计
    # ═════════════════════════════════════════════════════════════════════

    def get_meeting_summary(self, meeting_id: int) -> dict:
        """获取会议摘要统计。

        包括: 参会人发言时长、发言人切换次数、会议总时长等。

        Args:
            meeting_id: 会议 ID

        Returns:
            {
                "meeting_id": int,
                "meeting_name": str,
                "status": str,
                "total_duration_seconds": float | None,
                "total_speakers": int,
                "total_switches": int,
                "participant_stats": [
                    {
                        "tag_id": "A001",
                        "name": "王强",
                        "role": "项目负责人",
                        "total_duration": 423.0,    # 秒
                        "segment_count": 2,          # 发言次数
                    },
                    ...
                ],
            }
        """
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo, SpeakerSegmentRepo, ParticipantRepo

            with session_scope() as session:
                mr = MeetingRepo(session)
                meeting = mr.get_by_id(meeting_id)
                if meeting is None:
                    return {"error": f"会议不存在: id={meeting_id}"}

                sr = SpeakerSegmentRepo(session)
                segments = sr.get_segments_for_meeting(meeting_id)

                pr = ParticipantRepo(session)
                all_participants = {p.tag_id: p for p in pr.list_all()}

                # 计算总时长
                total_duration = None
                if meeting.start_time and meeting.end_time:
                    total_duration = (meeting.end_time - meeting.start_time).total_seconds()

                # 按发言人聚合
                speaker_stats: dict[str, dict] = {}
                for seg in segments:
                    tag = seg.speaker_tag_id or "__unknown__"
                    if tag not in speaker_stats:
                        p = all_participants.get(tag)
                        speaker_stats[tag] = {
                            "tag_id": tag if tag != "__unknown__" else None,
                            "name": seg.speaker_name or (p.name if p else "未知"),
                            "role": seg.role or (p.role if p else ""),
                            "total_duration": 0.0,
                            "segment_count": 0,
                        }
                    speaker_stats[tag]["total_duration"] += seg.duration_seconds or 0.0
                    speaker_stats[tag]["segment_count"] += 1

                # 切换次数 = 片段数 - 1
                total_switches = max(0, len(segments) - 1)

                return {
                    "meeting_id": meeting_id,
                    "meeting_name": meeting.name,
                    "status": meeting.status,
                    "total_duration_seconds": total_duration,
                    "total_speakers": len(speaker_stats),
                    "total_segments": len(segments),
                    "total_switches": total_switches,
                    "participant_stats": list(speaker_stats.values()),
                }
        except Exception as e:
            logger.error(f"获取会议摘要失败: {e}")
            return {"error": str(e)}

    # ═════════════════════════════════════════════════════════════════════
    # 内部方法
    # ═════════════════════════════════════════════════════════════════════

    @staticmethod
    def _publish_event(event_type: str, payload: dict):
        """发布事件到 EventBus（由 EventBridge 异步写入 DB）。"""
        try:
            from core.event_bus import EventBus
            EventBus().publish(event_type, **payload)
        except Exception as e:
            logger.warning(f"EventBus 发布失败 ({event_type}): {e}")
