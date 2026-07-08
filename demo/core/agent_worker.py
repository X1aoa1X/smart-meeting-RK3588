# -*- coding: utf-8 -*-
"""Agent Worker — 后台 Agent 工作线程（主 Qt 线程上运行）。

职责:
  1. 订阅 EventBus("*") 接收所有事件
  2. 调用 RuleEngine.evaluate() 评估事件 → CandidateSpeech
  3. 每 tick 调用 RuleEngine.evaluate_tick() 检查时间驱动触发
  4. 通过 SpeechGate 对候选播报进行限流/去重
  5. 通过 TTSRouter.say() 发出 TTS 请求
  6. 记录 AgentDecision 到 DB

Phase 2: 全部使用模板，不调用 LLM。
Phase 3: 增加 LLM 集成。
"""

import json
import logging
import time
from typing import Callable

from PyQt5.QtCore import QObject

from core.event_bus import EventBus
from core.agent_types import CandidateSpeech, TriggerType
from core.agent_state import AgentState
from core.agent_rules import RuleEngine
from core.tts_policy import (
    TTSRequest, TTSPriority,
    GLOBAL_MIN_INTERVAL_SEC,
    LLM_MIN_INTERVAL_SEC,
    MAX_LLM_CALLS_PER_MEETING,
    MAX_TTS_PER_5MIN,
    MAX_PENDING_TTS,
    PRIORITY_BYPASS_THRESHOLD,
)

logger = logging.getLogger(__name__)


class SpeechGate:
    """播报门控 — 在 RuleEngine 和 TTSRouter 之间进行限流/去重。

    纯 Python，零 Qt 依赖。
    """

    def __init__(self):
        self._last_tts_at: float = 0.0

    def evaluate(
        self,
        candidate: CandidateSpeech,
        state: AgentState,
        duplex_can_speak: bool = True,
        pending_count: int = 0,
        now: float | None = None,
    ) -> tuple[bool, str | None]:
        """评估候选播报是否可以通过门控。

        Returns:
            (allowed, suppressed_reason)
        """
        _now = now or time.time()

        # 1. 过期检查
        if candidate.expires_in_sec > 0:
            expires_at = _now + candidate.expires_in_sec  # 实际上是从创建时间算
        # 简化：检查 cooldown_key 的冷却
        if state.is_in_cooldown(candidate.trigger_key, _now):
            return False, "cooldown:trigger_key"

        # 2. 全局 TTS 冷却
        if candidate.priority < PRIORITY_BYPASS_THRESHOLD:
            if self._last_tts_at > 0 and (_now - self._last_tts_at) < GLOBAL_MIN_INTERVAL_SEC:
                return False, "cooldown:global_tts_interval"

        # 3. 5 分钟窗口限制
        if candidate.priority < PRIORITY_BYPASS_THRESHOLD:
            if state.get_tts_count_last_5min(_now) >= MAX_TTS_PER_5MIN:
                return False, "limit:max_tts_per_5min"

        # 4. Duplex 状态
        if not duplex_can_speak:
            return False, "duplex:speaking_or_cooldown"

        # 5. 队列已满
        if pending_count >= MAX_PENDING_TTS:
            return False, "queue_full"

        return True, None

    def record_tts(self, now: float | None = None):
        self._last_tts_at = now or time.time()


class AgentWorker(QObject):
    """Agent 工作器 — 订阅 EventBus，评估规则，触发播报。

    在主 Qt 线程上运行（因为 EventBus 不是线程安全的）。
    所有事件处理都是纯 Python 同步操作，不阻塞。

    Phase 2: 全部使用模板播报。
    Phase 3: 增加 LLM 调用（通过 AgentLLMWorker）。

    用法:
        worker = AgentWorker(rule_engine, tts_router, duplex, agent_state,
                             on_decision_log=cb, parent=self)
        worker.start()   # 开始监听 EventBus
        worker.tick()    # 每 100ms 调用一次，检查时间驱动触发
    """

    def __init__(
        self,
        rule_engine: RuleEngine,
        tts_router,               # TTSRouter
        duplex,                   # DuplexController
        agent_state: AgentState,
        on_decision_log: Callable | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._rule_engine = rule_engine
        self._tts_router = tts_router
        self._duplex = duplex
        self._state = agent_state
        self._gate = SpeechGate()
        self._started = False
        self._on_decision_log = on_decision_log
        # Workers list — LLM workers stored by Phase 3
        self._workers: list = []

        # 日志回调
        self.on_log = lambda msg: logger.info(msg)

    # ── 生命周期 ──────────────────────────────────────────────────────────

    def start(self):
        """订阅 EventBus 开始监听。"""
        if self._started:
            return
        EventBus().subscribe("*", self._on_event)
        self._started = True
        self._log("Agent Worker 已启动 — 已订阅 EventBus(*)")

    def stop(self):
        """取消订阅。"""
        if not self._started:
            return
        try:
            EventBus().unsubscribe("*", self._on_event)
        except Exception:
            pass
        self._started = False
        self._log("Agent Worker 已停止")

    # ── EventBus 回调 ─────────────────────────────────────────────────────

    def _on_event(self, event: dict):
        """EventBus 事件回调（主线程）。"""
        if not self._rule_engine.enabled:
            return

        event_type = event.get("event_type", "")
        if not event_type:
            return

        # 更新 AgentState
        self._state.update_from_event(event)

        # 规则评估
        candidate = self._rule_engine.evaluate(event, self._state)
        if candidate is not None:
            self._handle_candidate(candidate)

    def tick(self):
        """每 100ms 由 fusion_tracker._tracking_tick() 调用。

        检查时间驱动的触发条件（发言超时、静默超时）。
        """
        if not self._rule_engine.enabled:
            return

        now = time.time()
        candidate = self._rule_engine.evaluate_tick(self._state, now)
        if candidate is not None:
            self._handle_candidate(candidate, now)

    # ── 候选播报处理 ──────────────────────────────────────────────────────

    def _handle_candidate(self, candidate: CandidateSpeech,
                          now: float | None = None):
        """处理候选播报：门控检查 → (LLM 或模板) → TTSRouter 播报。"""
        _now = now or time.time()

        # SpeechGate 检查
        duplex_ok = self._duplex.can_speak()
        pending_count = self._tts_router.get_pending_count()
        allowed, reason = self._gate.evaluate(
            candidate, self._state, duplex_ok, pending_count, _now)

        if not allowed:
            self._log_decision(candidate, spoken=False,
                              suppressed_reason=reason,
                              final_text=candidate.template_text)
            return

        # 检查该触发器的固定规则播报（模板 TTS）是否被禁用
        trigger_cfg = self._rule_engine.get_trigger_config(candidate.trigger_type)
        allow_fixed_rule = trigger_cfg.get("fixed_rule_enabled", True)

        # Phase 3: 如果需要 LLM 且可用 → 异步调用 LLM
        if candidate.requires_llm and self._state.can_call_llm(
            max_per_meeting=MAX_LLM_CALLS_PER_MEETING,
            min_interval_sec=LLM_MIN_INTERVAL_SEC,
            now=_now,
        ):
            self._maybe_call_llm(candidate)
        elif not allow_fixed_rule:
            # 固定规则播报被禁用 — 抑制模板 TTS
            self._log_decision(candidate, spoken=False,
                              suppressed_reason="fixed_rule_disabled",
                              final_text=candidate.template_text)
        else:
            self._speak_template(candidate)

    def _speak_template(self, candidate: CandidateSpeech):
        """使用模板文本播报。"""
        text = candidate.template_text
        if not text:
            self._log_decision(candidate, spoken=False,
                              suppressed_reason="empty_template")
            return

        now = time.time()

        # ★ 先设冷却再发请求 — 防止 TTSRouter 拒绝后下一 tick 立即重触发（flooding）
        self._state.set_cooldown(candidate.trigger_key, candidate.cooldown_sec, now)
        self._state.record_tts(now)
        self._gate.record_tts(now)
        if candidate.trigger_type == TriggerType.SPEAKER_CONFIRMED:
            self._state.mark_speaker_announced(
                self._state.current_speaker_tag_id or "", now)

        # 创建 TTS 请求
        req = TTSRequest(
            text=text,
            source="fixed_rule",
            priority=candidate.priority,
            meeting_id=candidate.meeting_id,
            cooldown_key=candidate.trigger_key,
            reason=candidate.reason,
            expires_at=now + candidate.expires_in_sec,
        )

        ok = self._tts_router.say(req)
        if ok:
            self._log_decision(candidate, spoken=True, final_text=text)
        else:
            self._log_decision(candidate, spoken=False,
                              suppressed_reason="router_rejected",
                              final_text=text)

    def _maybe_call_llm(self, candidate: CandidateSpeech):
        """尝试通过 LLM 生成播报文本（后台 QThread）。

        失败/超时 → fallback 到模板文本。
        """
        from core.agent_context import AgentContextBuilder
        from core.agent_llm_worker import AgentLLMWorker

        # 构建上下文
        builder = AgentContextBuilder()
        context = builder.build(
            meeting_id=candidate.meeting_id,
            trigger_type=candidate.trigger_type,
            agent_state=self._state,
            policy=self._rule_engine._policy,
        )

        # 创建后台 worker
        worker = AgentLLMWorker(
            context=context,
            template_text=candidate.template_text,
            policy=self._rule_engine._policy,
        )
        self._workers.append(worker)

        def on_done(text: str, stats: dict):
            self._speak_llm_text(candidate, text, stats)
            self._workers.remove(worker)
            worker.deleteLater()

        def on_error(msg: str):
            trigger_cfg = self._rule_engine.get_trigger_config(
                candidate.trigger_type)
            if not trigger_cfg.get("fixed_rule_enabled", True):
                self._log(f"⚠️ LLM 失败且固定规则播报已禁用: {msg}")
                self._log_decision(candidate, spoken=False,
                                  suppressed_reason="llm_failed_fixed_rule_disabled",
                                  final_text=candidate.template_text)
            else:
                self._log(f"⚠️ LLM 失败，fallback 模板: {msg}")
                self._speak_template(candidate)
            self._workers.remove(worker)
            worker.deleteLater()

        worker.llm_done.connect(on_done)
        worker.llm_error.connect(on_error)
        worker.start()

    def _speak_llm_text(self, candidate: CandidateSpeech, text: str, stats: dict):
        """使用 LLM 生成的文本播报。"""
        llm_used = stats.get("llm_used", False)
        prompt_tokens = stats.get("prompt_tokens", 0)
        completion_tokens = stats.get("completion_tokens", 0)

        # 记录 LLM 调用
        if llm_used:
            self._state.record_llm_call()

        now = time.time()

        # ★ 先设冷却再发请求 — 防止 flooding
        self._state.set_cooldown(candidate.trigger_key, candidate.cooldown_sec, now)
        self._state.record_tts(now)
        self._gate.record_tts(now)

        req = TTSRequest(
            text=text,
            source="llm_agent",
            priority=candidate.priority,
            meeting_id=candidate.meeting_id,
            cooldown_key=candidate.trigger_key,
            reason=candidate.reason,
            expires_at=now + candidate.expires_in_sec,
        )

        ok = self._tts_router.say(req)
        self._log_decision(
            candidate, spoken=ok, final_text=text,
            llm_used=llm_used,
            llm_prompt_tokens=prompt_tokens,
            llm_completion_tokens=completion_tokens,
        )

    # ── 手动触发 ──────────────────────────────────────────────────────────

    def request_summary(self, meeting_id: int, minutes: int = 3):
        """手动触发阶段总结（Streamlit 按钮）。"""
        self._log(f"📋 手动触发: 总结最近 {minutes} 分钟")
        # Phase 2: 简单模板
        text = f"正在生成最近{minutes}分钟的会议总结。"
        self._manual_tts(TriggerType.MANUAL_SUMMARY, meeting_id,
                        text, "阶段总结", requires_llm=True)

    def request_agenda_reminder(self, meeting_id: int):
        """手动触发议题提醒。"""
        cfg = self._rule_engine.get_trigger_config("manual_agenda")
        template_id = cfg.get("template_id", "agenda_timeout")
        from core.tts_templates import format_template
        text = format_template(
            template_id,
            agenda_name=self._state.current_agenda or "当前议题",
            next_agenda=self._state.next_agenda or "下一议题",
        )
        self._manual_tts(TriggerType.MANUAL_AGENDA, meeting_id,
                        text, "议题提醒", requires_llm=cfg.get("requires_llm", True))

    def request_status_broadcast(self, meeting_id: int):
        """手动触发系统状态播报。"""
        from core.tts_templates import format_template
        text = format_template("manual_system_status")
        self._manual_tts(TriggerType.MANUAL_STATUS, meeting_id,
                        text, "系统状态播报", requires_llm=False)

    def request_custom_tts(self, text: str, meeting_id: int):
        """手动输入文字触发 TTS。"""
        self._log(f"📝 手动 TTS: \"{text[:40]}...\"")
        req = TTSRequest(
            text=text,
            source="host_manual",
            priority=TTSPriority.HOST_MANUAL,
            meeting_id=meeting_id,
            cooldown_key=f"manual_custom:{time.time()}",
            reason="手动输入播报",
            expires_at=time.time() + 30,
        )
        self._tts_router.say(req)

    def _manual_tts(self, trigger_type: str, meeting_id: int,
                    text: str, reason: str, requires_llm: bool = False):
        """通用手动触发 TTS 方法。"""
        if not text:
            return
        cfg = self._rule_engine.get_trigger_config(trigger_type)
        cooldown_sec = cfg.get("cooldown_sec", 15)

        candidate = CandidateSpeech(
            trigger_type=trigger_type,
            trigger_key=f"{trigger_type}:{meeting_id}:{time.time()}",
            meeting_id=meeting_id,
            priority=TTSPriority.HOST_MANUAL,
            requires_llm=requires_llm,
            template_text=text,
            reason=reason,
            cooldown_sec=cooldown_sec,
            expires_in_sec=30,
        )
        self._handle_candidate(candidate, time.time())

    # ── 决策日志 ──────────────────────────────────────────────────────────

    def _log_decision(self, candidate: CandidateSpeech, spoken: bool,
                      suppressed_reason: str | None = None,
                      final_text: str | None = None,
                      llm_used: bool = False,
                      llm_prompt_tokens: int = 0,
                      llm_completion_tokens: int = 0):
        """记录 Agent 决策（DB + 日志）。"""
        llm_tag = "🤖" if llm_used else ""
        self._log(
            f"{'✅ 播报' if spoken else '🚫 抑制'}{llm_tag}"
            f" [{candidate.trigger_type}]: \"{(final_text or candidate.template_text or '?')[:40]}\""
            + (f" — {suppressed_reason}" if suppressed_reason else "")
        )

        if self._on_decision_log is not None:
            try:
                self._on_decision_log(
                    candidate=candidate,
                    spoken=spoken,
                    suppressed_reason=suppressed_reason,
                    final_text=final_text or candidate.template_text,
                    llm_used=llm_used,
                    llm_prompt_tokens=llm_prompt_tokens,
                    llm_completion_tokens=llm_completion_tokens,
                )
            except Exception:
                pass

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)
