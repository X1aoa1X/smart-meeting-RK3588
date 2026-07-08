# -*- coding: utf-8 -*-
"""TTS 语音播报器 — 订阅 EventBus 事件，自动播报关键会议状态。

事件 → 播报映射:
  meeting_started       → "会议已开始"
  meeting_ended         → "会议已结束"
  host_locked_speaker   → "已锁定当前发言人"
  host_unlocked_speaker → "已解锁发言人"
  tracking_lost         → "目标丢失，正在重新搜索"

播报流程:
  1. EventBus 事件 → _on_event() (主线程)
  2. 映射事件到播报文本
  3. 检查 DuplexController 状态
  4. 查缓存 → 命中直接播放；未命中 → 异步合成 → 缓存 → 播放
  5. 播放结束后进入 COOLDOWN

调试: 所有事件（包括未映射的）都会记录到日志。
"""

import logging

from PyQt5.QtCore import QObject, QTimer

from core.event_bus import EventBus
from core.audio_player import AudioPlayer
from core.tts_cache import TTSCache
from core.tts_engine import TTSEngine
from core.duplex_controller import DuplexController, DuplexState, STATE_NAMES

logger = logging.getLogger(__name__)

# ── 所有已知的 EventBus 事件（用于调试日志） ──────────────────────────────
ALL_KNOWN_EVENTS = {
    "meeting_started", "meeting_ended", "meeting_paused", "meeting_resumed",
    "state_changed", "tracking_lost",
    "speaker_started", "speaker_switched", "speaker_lost",
    "speaker_reidentified", "speaker_ended",
    "speaker_override", "speaker_override_cleared",
    "host_locked_speaker", "host_unlocked_speaker",
    "host_note_added",
}


class Announcer(QObject):
    """播报器 — 订阅 EventBus，在会议关键节点播放语音通知。"""

    # 事件 → 播报文本映射
    ANNOUNCEMENT_MAP = {
        "meeting_started":       "会议已开始",
        "meeting_ended":         "会议已结束",
        "host_locked_speaker":   "已锁定当前发言人",
        "host_unlocked_speaker": "已解锁发言人",
        "tracking_lost":         "目标丢失，正在重新搜索",
    }

    # 高优先级事件: 即使 RECORDING 状态也播报
    PRIORITY_EVENTS = {"meeting_started", "meeting_ended"}

    # 播报完成轮询间隔 (ms)
    POLL_INTERVAL_MS = 50

    # 前置静默时长 (ms) — 在音频输出前让 VAD 管道清空:
    #   1. FusionEngine._speech_count 衰减归零 (每 100ms tick 减 1)
    #   2. 在途的 Silero/ReSpeaker VAD 事件过期
    #   3. ReSpeaker DOA 读数刷新
    # 默认 300ms = 3 个 tick 周期，足以让 speech_count 从 3 降到 0
    PRE_SPEAK_DELAY_MS = 300

    def __init__(self, cache: TTSCache, engine: TTSEngine,
                 player: AudioPlayer, duplex: DuplexController,
                 parent=None):
        super().__init__(parent)
        self._cache = cache
        self._engine = engine
        self._player = player
        self._duplex = duplex
        self._started = False

        # 待播报 (当前正在 PRE_SPEAKING/SPEAKING/COOLDOWN 时入队)
        self._pending_text: str = ""

        # 当前正在播报的文本 (避免重复播报)
        self._current_text: str = ""

        # 正在异步合成中的文本
        self._synthesizing_text: str = ""

        # 播报完成轮询
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(self.POLL_INTERVAL_MS)
        self._poll_timer.timeout.connect(self._poll_playback)

        # 播报日志回调 (外部设置)
        self.on_log = lambda msg: logger.info(msg)

        # 播报完成回调 (外部设置) — TTSRouter 用于写 TTSEvent 到 DB
        # Callable[[str, str, str], None] — (text, source, reason)
        self.on_spoken: object = None

        # 当前播报的 source (由 TTSRouter 在调用 _announce 前设置)
        self._current_source: str = "system"
        self._current_reason: str = ""

    # ── 生命周期 ──────────────────────────────────────────────────────

    def start(self):
        """订阅 EventBus 并开始监听。幂等 — 重复调用安全。"""
        if self._started:
            return self
        EventBus().subscribe("*", self._on_event)
        self._started = True
        self._log("播报器已启动 — 已订阅 EventBus(*)")
        # 打印当前订阅数
        bus = EventBus()
        sub_count = sum(len(v) for v in bus._subscribers.values())
        self._log(f"  EventBus 当前订阅者总数: {sub_count}")
        return self

    def stop(self):
        """取消订阅并停止所有正在进行的播报。"""
        if not self._started:
            return
        self._poll_timer.stop()
        self._player.stop()
        self._pending_text = ""
        self._current_text = ""
        try:
            EventBus().unsubscribe("*", self._on_event)
            self._log("播报器已停止 — 已取消订阅")
        except Exception:
            pass
        self._started = False

    # ── EventBus 回调 ──────────────────────────────────────────────────

    def _on_event(self, event: dict):
        """EventBus 事件回调 (主线程) — 仅日志，不触发播报。

        所有播报决策由 AgentWorker 统一管理，通过 TTSRouter → Announcer.announce()
        路径执行，确保每一条播报都有完整的策略审核和数据库审计。
        """
        event_type = event.get("event_type", "")
        if not event_type:
            return

        if event_type in ANNOUNCEMENT_MAP:
            self._log(f"📢 事件: {event_type} → 由 AgentWorker 处理")
        elif event_type in ALL_KNOWN_EVENTS:
            self._log(f"  ℹ️ 已知事件(未触发播报): {event_type}")
        # else: 静默忽略未知事件

    def _do_announce(self, event_type: str, text: str):
        """执行播报决策: 检查 duplex 状态，决定立即播报/排队/丢弃。"""
        is_priority = event_type in self.PRIORITY_EVENTS
        state = self._duplex.current_state

        if state in (DuplexState.PRE_SPEAKING, DuplexState.SPEAKING):
            self._pending_text = text
            self._log(f"  ⏳ 正在播报中，排队: \"{text}\"")
            return
        elif state == DuplexState.COOLDOWN:
            self._pending_text = text
            self._log(f"  ⏳ 冷却中，排队: \"{text}\"")
            return
        elif state == DuplexState.RECORDING and not is_priority:
            self._log(f"  🚫 录音中，丢弃非优先播报: \"{text}\"")
            return

        self._announce(text)

    # ── 播报逻辑 ──────────────────────────────────────────────────────

    def _announce(self, text: str, source: str = "system", reason: str = ""):
        """开始播报指定文本。查缓存优先，未命中则异步合成。

        关键: 调用 duplex.start_speaking(pre_delay_ms=...) 立即抑制 VAD，
        但音频输出延迟到前置静默期结束后，确保:
          1. FusionEngine._speech_count 衰减归零
          2. 在途 VAD 事件过期
          3. ReSpeaker DOA 读数刷新

        Args:
            text: 播报文本
            source: 来源标识 (fixed_rule|llm_agent|host_manual|system)
            reason: 触发原因 (用于 DB 审计)
        """
        if text == self._current_text:
            self._log(f"  ⏭ 重复抑制: \"{text}\"")
            return

        # ── 重复合成抑制: 必须在状态变更前检查 ──
        if text == self._synthesizing_text:
            self._log(f"  ⏳ 已在合成中: \"{text}\"")
            return

        self._current_text = text
        self._current_source = source
        self._current_reason = reason
        # ★ 立即抑制 VAD → PRE_SPEAKING → 定时器 → SPEAKING
        self._duplex.start_speaking(pre_delay_ms=self.PRE_SPEAK_DELAY_MS)
        self._log(f"  ▶ 开始播报: \"{text}\"  duplex → PRE_SPEAKING "
                  f"(前置静默 {self.PRE_SPEAK_DELAY_MS}ms)")

        # 查缓存
        pcm = self._cache.get(text)
        if pcm is not None:
            # 缓存命中: 延迟到前置静默期结束后播放
            self._log(f"  ✅ 缓存命中 ({len(pcm)} bytes) → "
                      f"{self.PRE_SPEAK_DELAY_MS}ms 后播放")
            QTimer.singleShot(
                self.PRE_SPEAK_DELAY_MS,
                lambda: self._start_playback(text, pcm)
            )
            return

        # 缓存未命中 → 异步合成 (自然延迟 > 前置静默期)
        if not self._engine.is_available():
            self._log(f"  ❌ TTS 引擎不可用 → 无法合成 \"{text}\"")
            self._on_announce_done()
            return

        self._synthesizing_text = text
        self._log(f"  🔄 缓存未命中 → 异步合成 \"{text}\"")

        self._engine.synthesize_async(
            text,
            on_done=lambda pcm_data: self._on_synthesis_ready(text, pcm_data),
            on_error=lambda msg: self._on_synthesis_error(text, msg),
        )

    def _start_playback(self, text: str, pcm: bytes):
        """实际开始音频播放 (在前置静默期结束后调用)。"""
        if text != self._current_text:
            self._log(f"  ⏭ 播放取消 (文本已过时): \"{text}\"")
            return

        self._log(f"  🔊 播放: \"{text}\" ({len(pcm)} bytes)")
        ok = self._player.play_pcm(pcm)
        if ok:
            self._start_polling()
        else:
            self._log(f"  ❌ 播放失败 (AudioPlayer 不可用)")
            self._on_announce_done()

    def test(self, text: str = "这是测试语音"):
        """手动触发测试播报 — 无需 EventBus 事件。

        可直接从 API 命令或 UI 按钮调用，用于验证 TTS 链路。
        """
        self._log(f"🧪 手动测试播报: \"{text}\"")
        self._do_announce("__test__", text)

    def announce(self, text: str, source: str = "system", reason: str = ""):
        """外部播报入口（供 TTSRouter 调用）。

        与 test() 的区别：不走 EventBus 映射，直接调用 _announce() 并传递 source/reason。

        Args:
            text: 播报文本
            source: 来源 (fixed_rule|llm_agent|host_manual|system)
            reason: 触发原因
        """
        self._log(f"📢 外部播报: \"{text}\" (source={source})")
        self._do_announce_with_source(text, source, reason)

    def _do_announce_with_source(self, text: str, source: str, reason: str):
        """同 _do_announce 但带 source/reason 透传。"""
        state = self._duplex.current_state

        if state in (DuplexState.PRE_SPEAKING, DuplexState.SPEAKING):
            self._pending_text = text
            self._log(f"  ⏳ 正在播报中，排队: \"{text}\"")
            return
        elif state == DuplexState.COOLDOWN:
            self._pending_text = text
            self._log(f"  ⏳ 冷却中，排队: \"{text}\"")
            return
        elif state == DuplexState.RECORDING:
            self._log(f"  🚫 录音中，丢弃播报: \"{text}\"")
            return

        self._announce(text, source=source, reason=reason)

    def _on_synthesis_ready(self, text: str, pcm_data: bytes):
        """合成完成 → 缓存 → 播放。"""
        self._log(f"  ✅ 合成完成 \"{text}\" ({len(pcm_data)} bytes)")

        # 写入缓存（即使 text 已过期，缓存仍有效）
        self._cache.put(text, pcm_data)

        # 仅在当前 synthesis 仍是此 text 时清除标记
        if text == self._synthesizing_text:
            self._synthesizing_text = ""

        # 如果当前文本没被覆盖，播放 (此时前置静默期通常已过)
        if text == self._current_text:
            self._start_playback(text, pcm_data)
        else:
            self._log(f"  ⏭ 合成结果已过时: \"{text}\" (当前: \"{self._current_text}\")")

    def _on_synthesis_error(self, text: str, msg: str):
        """合成失败 → 清理状态。"""
        self._log(f"  ❌ 合成失败 \"{text}\": {msg}")

        # 仅在当前 synthesis 仍是此 text 时清除标记
        if text == self._synthesizing_text:
            self._synthesizing_text = ""

        if text == self._current_text:
            self._on_announce_done()

    # ── 播放完成检测 ──────────────────────────────────────────────────

    def _start_polling(self):
        """启动播放完成轮询。"""
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_playback(self):
        """轮询 pygame mixer 是否仍在播放。"""
        if not self._player.is_busy():
            self._poll_timer.stop()
            self._on_announce_done()

    def _on_announce_done(self):
        """一次播报完成 → 进入 COOLDOWN → 处理队列。"""
        current = self._current_text
        current_source = self._current_source
        current_reason = self._current_reason
        self._current_text = ""
        self._current_source = "system"
        self._current_reason = ""

        # 通知外部监听者（TTSRouter 用于写 TTSEvent）
        if self.on_spoken is not None and current:
            try:
                self.on_spoken(current, current_source, current_reason)  # type: ignore
            except Exception:
                pass

        state_before = self._duplex.current_state
        self._duplex.finish_speaking()
        state_after = self._duplex.current_state

        if state_after == DuplexState.COOLDOWN:
            self._log(f"  ✓ 播报完成: \"{current}\"  duplex → COOLDOWN")
        elif state_before == state_after:
            # finish_speaking 未触发状态变更 (可能已是 COOLDOWN/LISTENING)
            self._log(f"  ℹ️ 播报完成: \"{current}\"  duplex 保持 "
                      f"{STATE_NAMES.get(state_after, '?')}")
        else:
            self._log(f"  ⚠️ 播报完成但状态异常: \"{current}\"  duplex → "
                      f"{STATE_NAMES.get(state_after, '?')} "
                      f"(之前: {STATE_NAMES.get(state_before, '?')})")

        # 处理等待队列
        if self._pending_text:
            pending = self._pending_text
            self._pending_text = ""
            delay = self._duplex.DEFAULT_COOLDOWN_MS + 50
            self._log(f"  ⏰ 排队播报将在 {delay}ms 后触发: \"{pending}\"")
            QTimer.singleShot(
                delay,
                lambda: self._announce(pending) if not self._duplex.is_vad_suppressed() else None
            )

    # ── 工具 ──────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)
