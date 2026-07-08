# -*- coding: utf-8 -*-
"""Agent LLM 上下文构建器 — 为 LLM 生成最小 JSON 上下文。

原则：不上传原始视频帧、音频、DOA 数据。只上传决策摘要。
每次 LLM 调用输入控制在 800-1200 字符。

纯 Python，零 Qt 依赖。
"""

import json
import logging
from datetime import datetime

from core.agent_types import TriggerType

logger = logging.getLogger(__name__)


class AgentContextBuilder:
    """构建发往 LLM 的最小 JSON 上下文。

    按 trigger_type 组装 5 个 context block：
      1. meeting_brief     — 标题、阶段、已过分钟数、当前议题
      2. current_speaker   — 姓名、角色、状态、连续发言秒数
      3. speaker_stats     — 每人总秒数 + 发言次数（top 5）
      4. recent_summary    — 本地聚合的 bullet points
      5. speech_policy     — 语言、max_chars、tone、must_not 列表
    """

    # ── 每种触发类型需要的 context block ──
    BLOCK_SPECS = {
        TriggerType.SPEAKER_OVERTIME: [
            "meeting_brief", "current_speaker", "speaker_stats", "speech_policy"
        ],
        TriggerType.AGENDA_TIMEOUT: [
            "meeting_brief", "current_speaker", "speaker_stats", "speech_policy"
        ],
        TriggerType.MANUAL_SUMMARY: [
            "meeting_brief", "speaker_stats", "recent_summary", "speech_policy"
        ],
        TriggerType.MANUAL_AGENDA: [
            "meeting_brief", "speaker_stats", "speech_policy"
        ],
    }

    def __init__(self, max_input_chars: int = 1200):
        self._max_input_chars = max_input_chars

    def build(self, meeting_id: int, trigger_type: str,
              agent_state,        # AgentState
              policy: dict | None = None,
              ) -> dict:
        """构建 LLM 上下文字典。

        Args:
            meeting_id: 会议 ID
            trigger_type: 触发类型
            agent_state: 当前 AgentState
            policy: Agent 策略配置（用于 speech_policy block）

        Returns:
            上下文字典（可直接 JSON 序列化）
        """
        blocks_needed = self.BLOCK_SPECS.get(trigger_type, ["meeting_brief", "speech_policy"])

        ctx = {}

        if "meeting_brief" in blocks_needed:
            ctx["meeting"] = self._build_meeting_brief(meeting_id, agent_state)

        if "current_speaker" in blocks_needed:
            ctx["current_speaker"] = self._build_current_speaker(agent_state)

        if "speaker_stats" in blocks_needed:
            ctx["speaker_stats"] = self._build_speaker_stats(meeting_id)

        if "recent_summary" in blocks_needed:
            ctx["recent_summary"] = self._build_recent_summary(meeting_id)

        if "speech_policy" in blocks_needed:
            ctx["policy"] = self._build_speech_policy(policy)

        return ctx

    # ── Block Builders ────────────────────────────────────────────────────

    def _build_meeting_brief(self, meeting_id: int, agent_state) -> dict:
        """Block 1: 会议基本信息。"""
        brief = {
            "id": meeting_id,
            "phase": agent_state.meeting_phase if agent_state else "discussion",
        }
        if agent_state and agent_state.current_agenda:
            brief["current_agenda"] = agent_state.current_agenda
        if agent_state and agent_state.next_agenda:
            brief["next_agenda"] = agent_state.next_agenda
        if agent_state and agent_state.agenda_owner:
            brief["agenda_owner"] = agent_state.agenda_owner
        return brief

    def _build_current_speaker(self, agent_state) -> dict | None:
        """Block 2: 当前发言人。"""
        if not agent_state or not agent_state.current_speaker_tag_id:
            return None
        duration = agent_state.get_speaker_duration()
        return {
            "name": agent_state.current_speaker_name or agent_state.current_speaker_tag_id,
            "tag_id": agent_state.current_speaker_tag_id,
            "state": agent_state.current_speaker_state or "unknown",
            "continuous_speaking_sec": round(duration, 1),
        }

    def _build_speaker_stats(self, meeting_id: int) -> list[dict]:
        """Block 3: 发言统计（每位发言人的总时长和发言次数，top 5）。"""
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo, SpeakerSegmentRepo, ParticipantRepo

            with session_scope() as session:
                meeting = MeetingRepo(session).get_by_id(meeting_id)
                if meeting is None:
                    return []

                seg_repo = SpeakerSegmentRepo(session)
                segments = seg_repo.get_segments_for_meeting(meeting_id)
                participants = ParticipantRepo(session).list_all()

            # 聚合
            stats: dict[str, dict] = {}
            for seg in segments:
                tid = seg.speaker_tag_id or "__unknown__"
                if tid not in stats:
                    p = next((x for x in participants if x.tag_id == tid), None)
                    stats[tid] = {
                        "name": seg.speaker_name or (p.name if p else "未知"),
                        "tag_id": tid,
                        "total_sec": 0.0,
                        "turns": 0,
                    }
                stats[tid]["total_sec"] += seg.duration_seconds or 0.0
                stats[tid]["turns"] += 1

            # 按总时长排序，取 top 5
            result = sorted(stats.values(), key=lambda x: x["total_sec"], reverse=True)[:5]
            for r in result:
                r["total_sec"] = round(r["total_sec"], 1)
            return result
        except Exception as e:
            logger.warning(f"Failed to build speaker stats: {e}")
            return []

    def _build_recent_summary(self, meeting_id: int) -> list[str]:
        """Block 4: 最近事件摘要（本地聚合的 bullet points，最多 5 条）。"""
        try:
            from storage.db import session_scope
            from storage.repo import EventRepo, HostNoteRepo

            with session_scope() as session:
                events = EventRepo(session).get_recent(minutes=5, meeting_id=meeting_id)
                notes = HostNoteRepo(session).get_notes_for_meeting(meeting_id)

            bullets = []

            # 发言人切换摘要
            speaker_starts = [e for e in events if e.event_type == "speaker_started"]
            if speaker_starts:
                names = set()
                for e in speaker_starts:
                    if e.payload_json:
                        try:
                            p = json.loads(e.payload_json)
                            names.add(p.get("name", "?"))
                        except Exception:
                            pass
                if names:
                    bullets.append(f"最近 5 分钟共有 {len(names)} 位发言人：{'、'.join(list(names)[:4])}")

            # 状态变更
            state_changes = [e for e in events if e.event_type == "state_changed"]
            if state_changes:
                bullets.append(f"追踪状态发生 {len(state_changes)} 次变更")

            # 丢失事件
            lost = [e for e in events if e.event_type in ("speaker_lost", "tracking_lost")]
            if lost:
                bullets.append(f"发生 {len(lost)} 次目标丢失")

            # 主持人备注
            if notes:
                recent_notes = notes[-3:]
                for n in recent_notes:
                    short = n.content[:40] + "..." if len(n.content) > 40 else n.content
                    bullets.append(f"主持人备注[{n.note_type}]: {short}")

            return bullets[:5]
        except Exception as e:
            logger.warning(f"Failed to build recent summary: {e}")
            return []

    def _build_speech_policy(self, policy: dict | None) -> dict:
        """Block 5: 语音策略。"""
        speech_cfg = (policy or {}).get("speech", {})
        return {
            "language": "zh-CN",
            "max_chars": speech_cfg.get("max_chars", 60),
            "tone": speech_cfg.get("tone", "polite"),
            "audience": "meeting_room",
            "must_not": [
                "编造未提供的信息",
                "直接命令参会人",
                "提及隐私或内部实现细节",
                "提及LLM、模型、Token等术语",
                "输出多句话",
            ],
        }


# ── System Prompt ──────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """你是一个会议现场语音助手，只能根据输入 JSON 生成一条简短中文播报。
要求：
1. 只输出最终播报文本，不要解释、不要加引号。
2. 不超过 policy.max_chars 个汉字。
3. 语气礼貌、克制，不要责备参会人。
4. 不要编造 JSON 中没有的信息。
5. 不要输出控制硬件、操作系统或网络的指令。
6. 不要提及"LLM""模型""Token""系统内部实现"。
7. 如果信息不足，输出一条通用、保守的提醒。"""
