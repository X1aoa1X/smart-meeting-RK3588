# -*- coding: utf-8 -*-
"""Agent 规则引擎 — 纯规则触发器，不调用 LLM。

评估 EventBus 事件 + AgentState 状态，产生 CandidateSpeech 或 None。
所有触发条件都是确定性规则，决策可解释、可审计。

配置从 agent_policy.json 加载，可在 Streamlit 中修改。
"""

import json
import logging
import os
import time

from core.agent_types import CandidateSpeech, TriggerType
from core.agent_state import AgentState
from core.tts_policy import TTSPriority
from core.tts_templates import format_template

logger = logging.getLogger(__name__)

# ── 禁入 Agent 的高频事件 ──────────────────────────────────────────────
HIGH_FREQ_EVENTS = {
    "frame_ready", "person_box_ready", "deviation_data",
    "servo_moved", "vad_tick", "doa_update",
}

# ── 默认策略（agent_policy.json 缺失时使用） ─────────────────────────
DEFAULT_POLICY: dict = {}


def _load_default_policy() -> dict:
    """加载默认策略（内联定义，无需文件读取）。"""
    return {
        "enabled": True,
        "demo_mode": False,
        "global": {
            "min_tts_interval_sec": 45,
            "min_llm_interval_sec": 90,
            "max_llm_calls_per_meeting": 20,
            "max_tts_per_5min": 5,
            "max_pending_tts": 3,
        },
        "triggers": {
            "meeting_started": {"enabled": True, "requires_llm": False, "cooldown_sec": 0, "template_id": "meeting_started"},
            "meeting_ended": {"enabled": True, "requires_llm": False, "cooldown_sec": 0, "template_id": "meeting_ended"},
            "speaker_confirmed": {"enabled": True, "requires_llm": False, "stable_sec": 1.5, "cooldown_sec": 120, "template_id": "speaker_confirmed"},
            "speaker_overtime": {"enabled": True, "requires_llm": True, "threshold_sec": 180, "cooldown_sec": 300, "template_id": "speaker_overtime"},
            "silence_timeout": {"enabled": True, "requires_llm": False, "threshold_sec": {"discussion": 40}, "cooldown_sec": 180, "template_id": "silence_timeout"},
            "agenda_timeout": {"enabled": False, "requires_llm": True, "cooldown_sec": 300, "template_id": "agenda_timeout"},
            "identity_lost": {"enabled": True, "requires_llm": False, "stable_sec": 8, "cooldown_sec": 120, "template_id": "identity_lost"},
            "system_error": {"enabled": True, "requires_llm": False, "cooldown_sec": 60, "template_id": None},
            "tracking_started": {"enabled": True, "requires_llm": False, "cooldown_sec": 30, "template_id": "tracking_started"},
            "tracking_paused": {"enabled": True, "requires_llm": False, "cooldown_sec": 30, "template_id": "tracking_paused"},
            "tracking_lost": {"enabled": True, "requires_llm": False, "cooldown_sec": 30, "template_id": "tracking_lost"},
            "host_locked_speaker": {"enabled": True, "requires_llm": False, "cooldown_sec": 10, "template_id": "host_locked_speaker"},
            "host_unlocked_speaker": {"enabled": True, "requires_llm": False, "cooldown_sec": 10, "template_id": "host_unlocked_speaker"},
            "manual_summary": {"enabled": True, "requires_llm": True, "cooldown_sec": 15, "template_id": None},
            "manual_agenda": {"enabled": True, "requires_llm": True, "cooldown_sec": 15, "template_id": "agenda_timeout"},
            "manual_status": {"enabled": True, "requires_llm": False, "cooldown_sec": 10, "template_id": "manual_system_status"},
            "speaker_switched": {"enabled": False, "requires_llm": False, "cooldown_sec": 60, "template_id": "speaker_confirmed"},
        },
        "speech": {"max_chars": 60, "tone": "polite", "fallback_on_llm_error": True},
    }


def load_agent_policy(config_path: str = None) -> dict:
    """加载 Agent 策略配置。

    Args:
        config_path: 配置文件路径（默认 configs/agent_policy.json）

    Returns:
        策略字典
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "agent_policy.json",
        )

    try:
        with open(config_path, "r") as f:
            policy = json.load(f)
        logger.info(f"Loaded agent policy from {config_path}")
        return policy
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load agent policy: {e} — using defaults")
        return _load_default_policy()


class RuleEngine:
    """规则引擎 — 纯 Python，不调 LLM，不依赖 Qt。

    职责:
      1. 过滤高频无关事件
      2. 评估事件 + AgentState → CandidateSpeech | None
      3. 填充模板文本

    用法:
        engine = RuleEngine(policy)
        candidate = engine.evaluate(event, agent_state)
        if candidate:
            agent_worker.handle_candidate(candidate)
    """

    def __init__(self, policy: dict | None = None):
        self._policy = policy or _load_default_policy()

    @property
    def enabled(self) -> bool:
        return self._policy.get("enabled", True)

    @property
    def demo_mode(self) -> bool:
        return self._policy.get("demo_mode", False)

    def get_trigger_config(self, trigger_type: str) -> dict:
        """获取某触发类型的配置。"""
        return self._policy.get("triggers", {}).get(trigger_type, {})

    def is_trigger_enabled(self, trigger_type: str) -> bool:
        """检查某触发类型是否启用。"""
        if not self.enabled:
            return False
        cfg = self.get_trigger_config(trigger_type)
        return cfg.get("enabled", False)

    def evaluate(self, event: dict, state: AgentState) -> CandidateSpeech | None:
        """评估一个 EventBus 事件，返回候选播报或 None。

        Args:
            event: EventBus 事件字典（包含 event_type, meeting_id, ...）
            state: 当前 AgentState

        Returns:
            CandidateSpeech 或 None（不需要播报）
        """
        event_type = event.get("event_type", "")

        # 过滤高频事件
        if event_type in HIGH_FREQ_EVENTS:
            return None

        now = time.time()

        # ── 生命周期事件 ──
        if event_type == "meeting_started":
            return self._eval_meeting_started(event, state, now)
        elif event_type == "meeting_ended":
            return self._eval_meeting_ended(event, state, now)

        # ── 发言人事件 ──
        elif event_type == "speaker_started":
            return self._eval_speaker_confirmed(event, state, now)
        elif event_type == "speaker_switched":
            return self._eval_speaker_switched(event, state, now)
        elif event_type == "speaker_lost":
            return self._eval_identity_lost(event, state, now)

        # ── 导播控制事件 ──
        elif event_type == "host_locked_speaker":
            return self._eval_host_locked(event, state, now)
        elif event_type == "host_unlocked_speaker":
            return self._eval_host_unlocked(event, state, now)
        elif event_type == "tracking_lost":
            return self._eval_tracking_lost(event, state, now)

        # ── 异常事件 ──
        elif event_type == "vad_error":
            return self._eval_system_error("audio", event, state, now)

        return None

    # ══════════════════════════════════════════════════════════════════════
    # 触发评估器
    # ══════════════════════════════════════════════════════════════════════

    def _eval_meeting_started(self, event: dict, state: AgentState,
                              now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("meeting_started"):
            return None
        cfg = self.get_trigger_config("meeting_started")
        meeting_id = event.get("meeting_id", 0)
        text = format_template(cfg.get("template_id", "meeting_started"))
        if not text:
            return None
        return CandidateSpeech(
            trigger_type=TriggerType.MEETING_STARTED,
            trigger_key=f"meeting_started:{meeting_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.GENERAL_STATUS,
            requires_llm=False,
            template_id=cfg.get("template_id", "meeting_started"),
            template_text=text,
            reason="会议开始",
            cooldown_sec=cfg.get("cooldown_sec", 0),
            expires_in_sec=10,
        )

    def _eval_meeting_ended(self, event: dict, state: AgentState,
                            now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("meeting_ended"):
            return None
        cfg = self.get_trigger_config("meeting_ended")
        meeting_id = event.get("meeting_id", 0)
        text = format_template(cfg.get("template_id", "meeting_ended"))
        if not text:
            return None
        return CandidateSpeech(
            trigger_type=TriggerType.MEETING_ENDED,
            trigger_key=f"meeting_ended:{meeting_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.GENERAL_STATUS,
            requires_llm=False,
            template_id=cfg.get("template_id", "meeting_ended"),
            template_text=text,
            reason="会议结束",
            cooldown_sec=cfg.get("cooldown_sec", 0),
            expires_in_sec=10,
        )

    def _eval_speaker_confirmed(self, event: dict, state: AgentState,
                                now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("speaker_confirmed"):
            return None
        cfg = self.get_trigger_config("speaker_confirmed")

        tag_id = event.get("tag_id", "")
        name = event.get("name", tag_id)
        meeting_id = event.get("meeting_id", state.meeting_id or 0)

        # 检查冷却
        if state.speaker_already_announced_recently(tag_id, cfg.get("cooldown_sec", 120), now):
            return None

        # 只在 demo_mode 下播报（减少打扰）
        if not self.demo_mode:
            return None

        text = format_template(cfg.get("template_id", "speaker_confirmed"), name=name)
        if not text:
            return None

        return CandidateSpeech(
            trigger_type=TriggerType.SPEAKER_CONFIRMED,
            trigger_key=f"speaker_confirmed:meeting_{meeting_id}:{tag_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.IDENTITY_STATUS,
            requires_llm=False,
            template_id=cfg.get("template_id", "speaker_confirmed"),
            template_text=text,
            reason=f"speaker_confirmed: {name} (tag={tag_id})",
            cooldown_sec=cfg.get("cooldown_sec", 120),
            expires_in_sec=15,
        )

    def _eval_speaker_switched(self, event: dict, state: AgentState,
                               now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("speaker_switched"):
            return None
        cfg = self.get_trigger_config("speaker_switched")

        name = event.get("name", "")
        tag_id = event.get("new_tag_id", event.get("tag_id", ""))
        meeting_id = event.get("meeting_id", state.meeting_id or 0)

        if state.speaker_already_announced_recently(tag_id, cfg.get("cooldown_sec", 60), now):
            return None
        if not self.demo_mode:
            return None

        text = format_template(cfg.get("template_id", "speaker_confirmed"), name=name)
        if not text:
            return None

        return CandidateSpeech(
            trigger_type=TriggerType.SPEAKER_SWITCHED,
            trigger_key=f"speaker_switched:meeting_{meeting_id}:{tag_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.IDENTITY_STATUS,
            requires_llm=False,
            template_id=cfg.get("template_id", "speaker_confirmed"),
            template_text=text,
            reason=f"speaker_switched to: {name}",
            cooldown_sec=cfg.get("cooldown_sec", 60),
            expires_in_sec=15,
        )

    def _eval_identity_lost(self, event: dict, state: AgentState,
                            now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("identity_lost"):
            return None
        cfg = self.get_trigger_config("identity_lost")

        if not state.meeting_active:
            return None
        if not state.tracking_active:
            return None

        tag_id = event.get("tag_id", "")
        name = event.get("name", tag_id)
        meeting_id = event.get("meeting_id", state.meeting_id or 0)

        text = format_template(cfg.get("template_id", "identity_lost"), name=name)
        if not text:
            return None

        return CandidateSpeech(
            trigger_type=TriggerType.IDENTITY_LOST,
            trigger_key=f"identity_lost:meeting_{meeting_id}:{tag_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.IDENTITY_STATUS,
            requires_llm=False,
            template_id=cfg.get("template_id", "identity_lost"),
            template_text=text,
            reason=f"identity_lost for: {name}",
            cooldown_sec=cfg.get("cooldown_sec", 120),
            expires_in_sec=15,
        )

    def _eval_host_locked(self, event: dict, state: AgentState,
                          now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("host_locked_speaker"):
            return None
        cfg = self.get_trigger_config("host_locked_speaker")
        meeting_id = event.get("meeting_id", state.meeting_id or 0)
        template_id = cfg.get("template_id", "host_locked_speaker")
        text = format_template(template_id)
        if not text:
            return None
        return CandidateSpeech(
            trigger_type=TriggerType.HOST_LOCKED,
            trigger_key=f"host_locked:{meeting_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.GENERAL_STATUS,
            requires_llm=False,
            template_id=template_id,
            template_text=text,
            reason="主持人锁定发言人",
            cooldown_sec=cfg.get("cooldown_sec", 10),
            expires_in_sec=10,
        )

    def _eval_host_unlocked(self, event: dict, state: AgentState,
                            now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("host_unlocked_speaker"):
            return None
        cfg = self.get_trigger_config("host_unlocked_speaker")
        meeting_id = event.get("meeting_id", state.meeting_id or 0)
        template_id = cfg.get("template_id", "host_unlocked_speaker")
        text = format_template(template_id)
        if not text:
            return None
        return CandidateSpeech(
            trigger_type=TriggerType.HOST_UNLOCKED,
            trigger_key=f"host_unlocked:{meeting_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.GENERAL_STATUS,
            requires_llm=False,
            template_id=template_id,
            template_text=text,
            reason="主持人解锁发言人",
            cooldown_sec=cfg.get("cooldown_sec", 10),
            expires_in_sec=10,
        )

    def _eval_tracking_lost(self, event: dict, state: AgentState,
                            now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("tracking_lost"):
            return None
        cfg = self.get_trigger_config("tracking_lost")
        meeting_id = event.get("meeting_id", state.meeting_id or 0)
        template_id = cfg.get("template_id", "tracking_lost")
        text = format_template(template_id)
        if not text:
            return None
        return CandidateSpeech(
            trigger_type=TriggerType.TRACKING_LOST,
            trigger_key=f"tracking_lost:{meeting_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.IDENTITY_STATUS,
            requires_llm=False,
            template_id=template_id,
            template_text=text,
            reason="视觉目标丢失",
            cooldown_sec=cfg.get("cooldown_sec", 30),
            expires_in_sec=10,
        )

    def _eval_system_error(self, error_kind: str, event: dict, state: AgentState,
                           now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("system_error"):
            return None
        cfg = self.get_trigger_config("system_error")

        meeting_id = event.get("meeting_id", state.meeting_id or 0)
        template_id = f"system_error_{error_kind}"
        text = format_template(template_id)
        if not text:
            text = format_template("system_error_general")

        return CandidateSpeech(
            trigger_type=TriggerType.SYSTEM_ERROR,
            trigger_key=f"system_error:{error_kind}:{meeting_id}",
            meeting_id=meeting_id,
            priority=TTSPriority.SYSTEM_CRITICAL,
            requires_llm=False,
            template_id=template_id,
            template_text=text,
            reason=f"system_error: {error_kind}",
            cooldown_sec=cfg.get("cooldown_sec", 60),
            expires_in_sec=10,
        )

    # ══════════════════════════════════════════════════════════════════════
    # 时间驱动触发（由 AgentWorker tick 主动检查，不靠 EventBus 事件）
    # ══════════════════════════════════════════════════════════════════════

    def evaluate_tick(self, state: AgentState, now: float | None = None) -> CandidateSpeech | None:
        """每 tick (100ms) 主动检查的触发条件。

        这些条件不依赖于某个具体的 EventBus 事件，而是基于状态累积：
          - 发言超时（speaker_overtime）
          - 静默超时（silence_timeout）
          - 议程超时（agenda_timeout）
        """
        _now = now or time.time()
        if not state.meeting_active:
            return None

        # 发言超时
        cand = self._eval_speaker_overtime(state, _now)
        if cand:
            return cand

        # 静默超时
        cand = self._eval_silence_timeout(state, _now)
        if cand:
            return cand

        return None

    def _eval_speaker_overtime(self, state: AgentState, now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("speaker_overtime"):
            return None
        cfg = self.get_trigger_config("speaker_overtime")

        threshold = cfg.get("threshold_sec", 180)
        duration = state.get_speaker_duration(now)
        if duration < threshold:
            return None
        if not state.current_speaker_tag_id:
            return None

        name = state.current_speaker_name or state.current_speaker_tag_id
        tag_id = state.current_speaker_tag_id
        meeting_id = state.meeting_id or 0

        cooldown_key = f"speaker_overtime:meeting_{meeting_id}:{tag_id}"
        if state.is_in_cooldown(cooldown_key, now):
            return None

        text = format_template(cfg.get("template_id", "speaker_overtime"), name=name)
        if not text:
            return None

        return CandidateSpeech(
            trigger_type=TriggerType.SPEAKER_OVERTIME,
            trigger_key=cooldown_key,
            meeting_id=meeting_id,
            priority=TTSPriority.SPEAKER_OVERTIME,
            requires_llm=cfg.get("requires_llm", True),
            template_id=cfg.get("template_id", "speaker_overtime"),
            template_text=text,
            reason=f"speaker_duration={int(duration)}s >= threshold={threshold}s",
            context_spec={
                "include": ["meeting_brief", "current_speaker", "speaker_stats", "speech_policy"],
                "window_sec": 300,
            },
            cooldown_sec=cfg.get("cooldown_sec", 300),
            expires_in_sec=20,
        )

    def _eval_silence_timeout(self, state: AgentState, now: float) -> CandidateSpeech | None:
        if not self.is_trigger_enabled("silence_timeout"):
            return None
        cfg = self.get_trigger_config("silence_timeout")

        # 按会议阶段获取阈值
        thresholds = cfg.get("threshold_sec", {"discussion": 40})
        if isinstance(thresholds, dict):
            threshold = thresholds.get(state.meeting_phase, 40)
        else:
            threshold = thresholds

        silence_dur = state.get_silence_duration(now)
        if silence_dur < threshold:
            return None

        meeting_id = state.meeting_id or 0
        cooldown_key = f"silence_timeout:meeting_{meeting_id}"
        if state.is_in_cooldown(cooldown_key, now):
            return None

        text = format_template(cfg.get("template_id", "silence_timeout"))
        if not text:
            return None

        return CandidateSpeech(
            trigger_type=TriggerType.SILENCE_TIMEOUT,
            trigger_key=cooldown_key,
            meeting_id=meeting_id,
            priority=TTSPriority.SILENCE_REMINDER,
            requires_llm=cfg.get("requires_llm", False),
            template_id=cfg.get("template_id", "silence_timeout"),
            template_text=text,
            reason=f"silence_duration={int(silence_dur)}s >= threshold={threshold}s (phase={state.meeting_phase})",
            context_spec={
                "include": ["meeting_brief", "recent_summary", "speech_policy"],
            },
            cooldown_sec=cfg.get("cooldown_sec", 180),
            expires_in_sec=20,
        )
