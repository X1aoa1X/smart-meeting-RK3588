# -*- coding: utf-8 -*-
"""Agent 会话状态缓存 — 追踪冷却、已播报发言人和 LLM 调用计数。

纯 Python，零 Qt 依赖。在主线程使用（由 AgentWorker 驱动）。
Meeting 边界自动重置。
"""

import time
import logging

from core.agent_types import TriggerType

logger = logging.getLogger(__name__)


class AgentState:
    """会话状态缓存 — 单次会议的有效期内状态。

    每个 meeting 重置一次。存储:
      - 冷却计时器（按 trigger_key）
      - 已播报发言人记录（按 tag_id）
      - 静默时长追踪
      - 当前发言人时长追踪
      - LLM 调用计数
      - TTS 播报计数（最近 5 分钟滑动窗口）
    """

    def __init__(self):
        self.reset()

    # ── 冷却管理 ──────────────────────────────────────────────────────────

    def is_in_cooldown(self, cooldown_key: str, now: float | None = None) -> bool:
        """检查是否在冷却期内。"""
        if not cooldown_key:
            return False
        _now = now or time.time()
        next_at = self._cooldown_timers.get(cooldown_key)
        return next_at is not None and _now < next_at

    def set_cooldown(self, cooldown_key: str, duration_sec: float,
                     now: float | None = None):
        """设置冷却期。"""
        _now = now or time.time()
        self._cooldown_timers[cooldown_key] = _now + duration_sec

    def clear_cooldown(self, cooldown_key: str):
        """清除指定冷却。"""
        self._cooldown_timers.pop(cooldown_key, None)

    # ── 发言人追踪 ────────────────────────────────────────────────────────

    def speaker_already_announced_recently(self, tag_id: str,
                                           window_sec: float = 120.0,
                                           now: float | None = None) -> bool:
        """检查该发言人最近是否已播报过。"""
        if not tag_id:
            return False
        _now = now or time.time()
        last_at = self._announced_speakers.get(tag_id)
        return last_at is not None and (_now - last_at) < window_sec

    def mark_speaker_announced(self, tag_id: str, now: float | None = None):
        """标记发言人已被播报。"""
        if tag_id:
            self._announced_speakers[tag_id] = now or time.time()

    # ── 静默追踪 ──────────────────────────────────────────────────────────

    def update_silence(self, is_speech: bool, now: float | None = None):
        """更新静默状态。

        由 fusion_tracker._tracking_tick() 每 100ms 调用一次。

        Args:
            is_speech: VAD 是否检测到语音
            now: 当前时间戳
        """
        _now = now or time.time()
        if is_speech:
            # 有声音 → 重置静默计时
            self._silence_started_at = None
            self._last_speech_at = _now
        else:
            # 无声音 → 记录静默开始时间
            if self._silence_started_at is None:
                self._silence_started_at = _now

    def get_silence_duration(self, now: float | None = None) -> float:
        """获取当前持续静默时长（秒）。"""
        _now = now or time.time()
        if self._silence_started_at is None:
            return 0.0
        return _now - self._silence_started_at

    # ── 发言人时长追踪 ────────────────────────────────────────────────────

    def update_current_speaker(self, tag_id: str | None, name: str | None,
                               state: str | None, now: float | None = None):
        """更新当前发言人信息。"""
        _now = now or time.time()

        if tag_id != self._current_speaker_tag_id:
            # 发言人切换
            self._current_speaker_tag_id = tag_id
            self._current_speaker_name = name
            self._current_speaker_state = state
            self._current_speaker_started_at = _now if tag_id else None
        else:
            # 同一发言人 → 更新状态
            self._current_speaker_state = state

    def get_speaker_duration(self, now: float | None = None) -> float:
        """获取当前发言人连续发言时长（秒）。"""
        _now = now or time.time()
        if self._current_speaker_started_at is None:
            return 0.0
        return _now - self._current_speaker_started_at

    # ── EventBus 事件 → 状态同步 ─────────────────────────────────────────

    def update_from_event(self, event: dict):
        """从 EventBus 事件更新内部状态（在 AgentWorker._on_event 中调用）。

        只处理有状态影响的事件类型，其余静默忽略。
        """
        event_type = event.get("event_type", "")
        _now = time.time()

        if event_type == "speaker_started":
            tag_id = event.get("tag_id", "")
            name = event.get("name", tag_id)
            self.update_current_speaker(
                tag_id=tag_id, name=name, state="confirmed", now=_now)

        elif event_type == "speaker_switched":
            tag_id = event.get("new_tag_id", event.get("tag_id", ""))
            name = event.get("name", tag_id)
            self.update_current_speaker(
                tag_id=tag_id, name=name, state="confirmed", now=_now)

        elif event_type == "speaker_lost":
            name = event.get("name", "")
            self.update_current_speaker(
                tag_id=self._current_speaker_tag_id,
                name=name or self._current_speaker_name,
                state="lost", now=_now)

        elif event_type == "speaker_ended":
            self.update_current_speaker(
                tag_id=None, name=None, state=None, now=_now)

        elif event_type == "meeting_started":
            self.reset()

        elif event_type == "meeting_ended":
            self._meeting_active = False
            self._tracking_active = False

    # ── LLM 调用管理 ──────────────────────────────────────────────────────

    def can_call_llm(self, max_per_meeting: int = 20,
                     min_interval_sec: float = 90.0,
                     now: float | None = None) -> bool:
        """检查是否允许再次调用 LLM。"""
        _now = now or time.time()

        if self._llm_calls_this_meeting >= max_per_meeting:
            return False
        if self._last_llm_call_at > 0 and (_now - self._last_llm_call_at) < min_interval_sec:
            return False
        return True

    def record_llm_call(self, now: float | None = None):
        """记录一次 LLM 调用。"""
        _now = now or time.time()
        self._llm_calls_this_meeting += 1
        self._last_llm_call_at = _now

    # ── TTS 追踪 ──────────────────────────────────────────────────────────

    def record_tts(self, now: float | None = None):
        """记录一次 TTS 播报（维护最近 5 分钟滑动窗口）。"""
        _now = now or time.time()
        self._tts_timestamps.append(_now)
        # 清理 5 分钟前的记录
        cutoff = _now - 300.0
        while self._tts_timestamps and self._tts_timestamps[0] < cutoff:
            self._tts_timestamps.pop(0)

    def get_tts_count_last_5min(self, now: float | None = None) -> int:
        """获取最近 5 分钟 TTS 播报次数。"""
        _now = now or time.time()
        cutoff = _now - 300.0
        count = 0
        for ts in self._tts_timestamps:
            if ts >= cutoff:
                count += 1
        return count

    # ── 会议生命周期 ──────────────────────────────────────────────────────

    def update_meeting_state(self, meeting_active: bool, meeting_id: int | None,
                             meeting_phase: str = "discussion",
                             tracking_active: bool = False):
        """更新会议状态。"""
        self._meeting_active = meeting_active
        self._meeting_id = meeting_id
        self._meeting_phase = meeting_phase
        self._tracking_active = tracking_active

    def update_agenda(self, agenda_name: str | None = None,
                      agenda_owner: str | None = None,
                      next_agenda: str | None = None):
        """更新当前议程信息。"""
        self._current_agenda = agenda_name
        self._agenda_owner = agenda_owner
        self._next_agenda = next_agenda

    def reset(self):
        """重置所有状态（新会议开始时调用）。"""
        self._cooldown_timers: dict[str, float] = {}
        self._announced_speakers: dict[str, float] = {}
        self._silence_started_at: float | None = None
        self._last_speech_at: float = 0.0
        self._current_speaker_tag_id: str | None = None
        self._current_speaker_name: str | None = None
        self._current_speaker_state: str | None = None
        self._current_speaker_started_at: float | None = None
        self._llm_calls_this_meeting: int = 0
        self._last_llm_call_at: float = 0.0
        self._tts_timestamps: list[float] = []
        self._meeting_active: bool = False
        self._meeting_id: int | None = None
        self._meeting_phase: str = "discussion"
        self._tracking_active: bool = False
        self._current_agenda: str | None = None
        self._agenda_owner: str | None = None
        self._next_agenda: str | None = None

    # ── 属性访问 ──────────────────────────────────────────────────────────

    @property
    def meeting_active(self) -> bool:
        return self._meeting_active

    @property
    def meeting_id(self) -> int | None:
        return self._meeting_id

    @property
    def meeting_phase(self) -> str:
        return self._meeting_phase

    @property
    def tracking_active(self) -> bool:
        return self._tracking_active

    @property
    def current_speaker_tag_id(self) -> str | None:
        return self._current_speaker_tag_id

    @property
    def current_speaker_name(self) -> str | None:
        return self._current_speaker_name

    @property
    def current_speaker_state(self) -> str | None:
        return self._current_speaker_state

    @property
    def llm_calls_this_meeting(self) -> int:
        return self._llm_calls_this_meeting

    @property
    def current_agenda(self) -> str | None:
        return self._current_agenda

    @property
    def agenda_owner(self) -> str | None:
        return self._agenda_owner

    @property
    def next_agenda(self) -> str | None:
        return self._next_agenda

    @property
    def last_speech_at(self) -> float:
        return self._last_speech_at
