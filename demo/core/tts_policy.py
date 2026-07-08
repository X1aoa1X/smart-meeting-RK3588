# -*- coding: utf-8 -*-
"""TTS 策略 — 优先级、冷却、限流常量和请求数据结构。

纯 Python，零 Qt/PyQt5 依赖 — 可被 Agent、TTSRouter、Streamlit 面板共享。
"""

from dataclasses import dataclass, field
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════════════
# 优先级常量
# ═══════════════════════════════════════════════════════════════════════════════

class TTSPriority:
    """TTS 播报优先级（0-100，数值越大越紧急）。"""
    SYSTEM_CRITICAL: int = 100   # 严重系统异常
    HOST_MANUAL: int     = 80    # 主持人手动触发
    AGENDA_TIMEOUT: int  = 60    # 议题超时
    SPEAKER_OVERTIME: int = 50   # 发言超时
    SILENCE_REMINDER: int = 40   # 静默提醒
    IDENTITY_STATUS: int = 30    # 身份识别状态
    GENERAL_STATUS: int  = 20    # 普通状态播报


# ═══════════════════════════════════════════════════════════════════════════════
# 全局限流常量
# ═══════════════════════════════════════════════════════════════════════════════

# 两次普通播报最小间隔 (秒)
GLOBAL_MIN_INTERVAL_SEC: float = 45.0

# 两次 LLM 调用最小间隔 (秒)
LLM_MIN_INTERVAL_SEC: float = 90.0

# 每场会议最大 LLM 调用次数
MAX_LLM_CALLS_PER_MEETING: int = 20

# 每 5 分钟最多 TTS 条数
MAX_TTS_PER_5MIN: int = 5

# 待播报队列最大长度
MAX_PENDING_TTS: int = 3

# 候选播报默认过期时间 (秒)
DEFAULT_EXPIRES_IN_SEC: float = 20.0

# 高优先级事件 (绕过正常冷却)
PRIORITY_BYPASS_THRESHOLD: int = TTSPriority.SYSTEM_CRITICAL


# ═══════════════════════════════════════════════════════════════════════════════
# TTS 请求数据结构
# ═══════════════════════════════════════════════════════════════════════════════

TTSSource = Literal["fixed_rule", "llm_agent", "host_manual", "system"]
TTSStatus = Literal["queued", "speaking", "spoken", "skipped", "failed"]


@dataclass
class TTSRequest:
    """一次 TTS 播报请求。

    由 RuleEngine / AgentWorker / Streamlit 手动触发产生，
    经过 SpeechGate 和 TTSRouter 的冷却/队列/过期检查后最终播报。
    """
    text: str
    source: str               # "fixed_rule" | "llm_agent" | "host_manual" | "system"
    priority: int             # 0-100, see TTSPriority
    meeting_id: int | None
    cooldown_key: str         # 去重键（如 "speaker_confirmed:meeting_12:tag_A003"）
    reason: str               # 人类可读的触发原因

    # 过期时间戳 (time.time() 值)，超时后自动丢弃
    expires_at: float = 0.0

    # 是否允许打断当前播报（仅高优先级异常可用）
    interruptible: bool = False

    # 可选: 关联的 AgentDecision ID（用于 audit trail）
    decision_id: int | None = None

    def is_expired(self, now: float | None = None) -> bool:
        """检查是否已过期。"""
        import time
        if self.expires_at <= 0:
            return False
        return (now or time.time()) > self.expires_at

    def __repr__(self) -> str:
        return (f"<TTSRequest pri={self.priority} src={self.source}"
                f" key={self.cooldown_key[:40]} text={self.text[:30]}>")
