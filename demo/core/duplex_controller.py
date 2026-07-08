# -*- coding: utf-8 -*-
"""半双工音频控制状态机。

控制 TTS 播报与麦克风拾音之间的互斥关系，防止系统"自己说的话被自己听见"。

状态说明:
  LISTENING    — VAD 启用，正常监听
  PRE_SPEAKING — 播报前置静默期 (默认 300ms)，VAD 已抑制但音频尚未输出
  SPEAKING     — TTS 正在播报，VAD 抑制
  COOLDOWN     — 播报结束后的冷却期 (300-800ms)，VAD 仍抑制
  RECORDING    — 用户正在录音 (未来 ASR)，TTS 被阻塞 (除高优先级告警)

用法:
  duplex = DuplexController()
  duplex.state_changed.connect(on_state_change)

  # TTS 播报前 (带前置静默)
  duplex.start_speaking(pre_delay_ms=300)
  # 播报结束后
  duplex.finish_speaking(cooldown_ms=500)

  # VAD 线程检查
  if duplex.is_vad_suppressed():
      skip_vad_processing()
"""

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


class DuplexState:
    LISTENING = 0
    PRE_SPEAKING = 4   # 前置静默: VAD 已抑制，等待音频输出
    SPEAKING = 1
    COOLDOWN = 2
    RECORDING = 3


STATE_NAMES = {
    DuplexState.LISTENING:    "LISTENING",
    DuplexState.PRE_SPEAKING: "PRE_SPEAKING",
    DuplexState.SPEAKING:     "SPEAKING",
    DuplexState.COOLDOWN:     "COOLDOWN",
    DuplexState.RECORDING:    "RECORDING",
}


class DuplexController(QObject):
    """半双工音频状态机 — 纯逻辑控制，不涉及硬件操作。

    所有状态变更通过 pyqtSignal 通知观察者（Announcer、VAD）。
    状态转换在主线程通过 QTimer.singleShot 驱动。
    """

    state_changed = pyqtSignal(int, int)  # (old_state, new_state)

    DEFAULT_COOLDOWN_MS = 500       # 播报后默认冷却时间
    DEFAULT_PRE_DELAY_MS = 300      # 播报前默认前置静默时间

    def __init__(self, cooldown_ms: int = DEFAULT_COOLDOWN_MS,
                 pre_delay_ms: int = DEFAULT_PRE_DELAY_MS, parent=None):
        # ── 防御: 如果第一个位置参数是 QObject (常见误用 DuplexController(widget)) ──
        if isinstance(cooldown_ms, QObject):
            import logging
            logging.getLogger(__name__).warning(
                "DuplexController(cooldown_ms) 收到了 QObject — "
                "可能是误用 DuplexController(widget) 而非 DuplexController(parent=widget)，"
                "已自动纠正。")
            parent = cooldown_ms
            cooldown_ms = DEFAULT_COOLDOWN_MS
        elif not isinstance(cooldown_ms, int):
            cooldown_ms = int(cooldown_ms)
        super().__init__(parent)
        self._state = DuplexState.LISTENING
        self._cooldown_ms = cooldown_ms
        self._pre_delay_ms = pre_delay_ms
        self._cooldown_timer: QTimer | None = None
        self._pre_speak_timer: QTimer | None = None

    # ── 公开属性 ──────────────────────────────────────────────────────

    @property
    def current_state(self) -> int:
        return self._state

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self._state, "?")

    def is_vad_suppressed(self) -> bool:
        """VAD 是否应被抑制 — PRE_SPEAKING / SPEAKING / COOLDOWN 状态时为 True。"""
        return self._state in (DuplexState.PRE_SPEAKING, DuplexState.SPEAKING,
                               DuplexState.COOLDOWN)

    def can_speak(self) -> bool:
        """当前是否可以开始 TTS 播报。RECORDING 时仅允许高优先级。"""
        return self._state in (DuplexState.LISTENING, DuplexState.PRE_SPEAKING,
                               DuplexState.SPEAKING)

    # ── 状态转换 ──────────────────────────────────────────────────────

    def start_speaking(self, pre_delay_ms: int | None = None):
        """进入 SPEAKING 状态，可选前置静默期。

        Args:
            pre_delay_ms: 前置静默时长 (ms)。None 表示使用默认值 (300ms)。
                          设为 0 则跳过 PRE_SPEAKING 直接进入 SPEAKING。

        前置静默期内 VAD 已被抑制但音频尚未输出，让:
          1. FusionEngine._speech_count 衰减归零
          2. 在途的 Silero/ReSpeaker VAD 事件过期
          3. ReSpeaker DOA 读数刷新为当前真实值

        重复调用是幂等的: 若已在 PRE_SPEAKING，不会重置计时器；
        若已在 SPEAKING，完全无操作。
        """
        if self._state == DuplexState.SPEAKING:
            return  # 幂等: 已在播报
        self._cancel_cooldown()

        delay = pre_delay_ms if pre_delay_ms is not None else self._pre_delay_ms
        if delay <= 0:
            # 无前置静默 → 直接 SPEAKING
            self._cancel_pre_speak()
            self._transition_to(DuplexState.SPEAKING)
            return

        if self._state == DuplexState.PRE_SPEAKING:
            return  # 幂等: 已在前置静默中

        # → PRE_SPEAKING → 定时器 → SPEAKING
        self._transition_to(DuplexState.PRE_SPEAKING)
        self._pre_speak_timer = QTimer(self)
        self._pre_speak_timer.setSingleShot(True)
        self._pre_speak_timer.timeout.connect(self._on_pre_speak_expired)
        self._pre_speak_timer.start(delay)

    def finish_speaking(self, cooldown_ms: int | None = None):
        """播报结束 → COOLDOWN。启动冷却定时器。
        也接受从 PRE_SPEAKING 状态调用 (播报被取消时)。
        """
        if self._state not in (DuplexState.PRE_SPEAKING, DuplexState.SPEAKING):
            return
        self._cancel_pre_speak()
        ms = cooldown_ms if cooldown_ms is not None else self._cooldown_ms
        # 确保 ms 是 int 类型（QTimer.start 严格要求 int）
        try:
            ms = int(ms)
        except (TypeError, ValueError):
            import logging
            logging.getLogger(__name__).error(
                f"finish_speaking: cooldown_ms 类型错误 ({type(ms).__name__})，使用默认值 {self.DEFAULT_COOLDOWN_MS}ms")
            ms = self.DEFAULT_COOLDOWN_MS
        self._transition_to(DuplexState.COOLDOWN)
        self._cooldown_timer = QTimer(self)
        self._cooldown_timer.setSingleShot(True)
        self._cooldown_timer.timeout.connect(self._on_cooldown_expired)
        self._cooldown_timer.start(ms)

    def enter_recording(self):
        """进入 RECORDING 状态 (未来 ASR 使用)。"""
        self._cancel_cooldown()
        self._cancel_pre_speak()
        self._transition_to(DuplexState.RECORDING)

    def exit_recording(self):
        """退出 RECORDING → LISTENING。"""
        if self._state != DuplexState.RECORDING:
            return
        self._transition_to(DuplexState.LISTENING)

    def reset(self):
        """强制回到 LISTENING，取消任何定时器。"""
        self._cancel_cooldown()
        self._cancel_pre_speak()
        if self._state != DuplexState.LISTENING:
            self._transition_to(DuplexState.LISTENING)

    # ── 内部 ──────────────────────────────────────────────────────────

    def _transition_to(self, new_state: int):
        if new_state == self._state:
            return
        old = self._state
        self._state = new_state
        self.state_changed.emit(old, new_state)

    def _on_cooldown_expired(self):
        """冷却定时器到期 → LISTENING。"""
        self._cooldown_timer = None
        if self._state == DuplexState.COOLDOWN:
            self._transition_to(DuplexState.LISTENING)

    def _on_pre_speak_expired(self):
        """前置静默定时器到期 → SPEAKING。"""
        self._pre_speak_timer = None
        if self._state == DuplexState.PRE_SPEAKING:
            self._transition_to(DuplexState.SPEAKING)

    def _cancel_cooldown(self):
        if self._cooldown_timer is not None:
            self._cooldown_timer.stop()
            self._cooldown_timer = None

    def _cancel_pre_speak(self):
        if self._pre_speak_timer is not None:
            self._pre_speak_timer.stop()
            self._pre_speak_timer = None
