# -*- coding: utf-8 -*-
"""Agent 数据类型 — CandidateSpeech 和 TriggerType 等核心数据结构。

纯 Python，零 Qt 依赖，可在 AgentState / RuleEngine / AgentWorker / Streamlit 间共享。
"""

from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# 触发类型
# ═══════════════════════════════════════════════════════════════════════════════

class TriggerType(str, Enum):
    """Agent 播报触发类型。"""
    # A 类：确定性系统播报（不用 LLM）
    MEETING_STARTED   = "meeting_started"
    MEETING_ENDED     = "meeting_ended"
    TRACKING_STARTED  = "tracking_started"
    TRACKING_PAUSED   = "tracking_paused"
    HOST_LOCKED       = "host_locked_speaker"
    HOST_UNLOCKED     = "host_unlocked_speaker"
    MANUAL_SPEAKER    = "manual_speaker"
    TRACKING_LOST     = "tracking_lost"

    # B 类：规则触发 + 模板优先（LLM 可选）
    SPEAKER_CONFIRMED = "speaker_confirmed"
    SPEAKER_OVERTIME  = "speaker_overtime"
    SILENCE_TIMEOUT   = "silence_timeout"
    IDENTITY_LOST     = "identity_lost"
    SPEAKER_SWITCHED  = "speaker_switched"

    # C 类：LLM 低频事件
    AGENDA_TIMEOUT    = "agenda_timeout"

    # D 类：主持人手动触发
    MANUAL_SUMMARY    = "manual_summary"
    MANUAL_AGENDA     = "manual_agenda"
    MANUAL_STATUS     = "manual_status"

    # 系统异常
    SYSTEM_ERROR      = "system_error"


# ═══════════════════════════════════════════════════════════════════════════════
# 候选播报
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateSpeech:
    """规则引擎输出的候选播报 — 尚未通过 SpeechGate 和冷却检查。

    由 RuleEngine.evaluate() 产生，包含所有播报决策所需信息。
    """
    trigger_type: str                                   # TriggerType 值
    trigger_key: str                                    # 去重键
    meeting_id: int                                     # 会议 ID
    priority: int                                       # 优先级 0-100
    requires_llm: bool = False                          # 是否需要 LLM 改写
    template_id: str | None = None                      # 模板 ID（如 "speaker_confirmed"）
    template_text: str = ""                             # 模板文本（已填充，LLM 不可用时直接使用）
    reason: str = ""                                    # 触发原因（英文，用于 DB）
    context_spec: dict | None = None                    # LLM 上下文需求规格
    cooldown_sec: int = 120                             # 同类型冷却时长（秒）
    expires_in_sec: int = 20                            # 候选过期时间（秒）

    def __repr__(self) -> str:
        return (f"<CandidateSpeech trigger={self.trigger_type}"
                f" pri={self.priority} llm={self.requires_llm}>")


# ═══════════════════════════════════════════════════════════════════════════════
# 决策结果
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentDecision:
    """一次完整的 Agent 决策结果 — 用于 DB 审计和 Streamlit 展示。"""
    candidate: CandidateSpeech
    spoken: bool = False
    suppressed_reason: str | None = None
    final_text: str | None = None
    llm_used: bool = False
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
