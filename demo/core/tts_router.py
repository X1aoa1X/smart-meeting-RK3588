# -*- coding: utf-8 -*-
"""TTS Router — 统一 TTS 播报入口，委托给 Announcer 执行。

在 Announcer 之上增加:
  - 优先级队列（满时丢弃低优先级过期消息）
  - 全局冷却检查
  - 同 cooldown_key 冷却
  - 过期丢弃
  - DuplexController 状态检查
  - DB 审计日志（通过 on_spoken / on_suppressed 回调）

所有对 Announcer 的播报请求都应通过 TTSRouter.say() 发出。
"""

import logging
import time
from collections import deque

from PyQt5.QtCore import QObject

from core.tts_policy import (
    TTSRequest,
    TTSPriority,
    GLOBAL_MIN_INTERVAL_SEC,
    MAX_PENDING_TTS,
    PRIORITY_BYPASS_THRESHOLD,
)
from core.duplex_controller import DuplexController

logger = logging.getLogger(__name__)


class TTSRouter(QObject):
    """统一 TTS 播报路由器。

    委托给现有 Announcer 处理实际播报（缓存、合成、播放、Duplex 状态），
    本层只做策略决策：该不该说、什么时候说、是否排队/丢弃。

    用法:
        router = TTSRouter(announcer=announcer, duplex=duplex, parent=self)
        router.on_spoken = self._on_tts_spoken      # (text, source, reason)
        router.on_suppressed = self._on_tts_suppressed  # (request, reason_str)

        req = TTSRequest(text="...", source="fixed_rule", ...)
        router.say(req)
    """

    def __init__(self, announcer, duplex: DuplexController, parent=None):
        super().__init__(parent)
        self._announcer = announcer
        self._duplex = duplex

        # ── 待播报队列 ──
        self._pending: deque[TTSRequest] = deque()

        # ── 全局状态 ──
        self._last_spoken_at: float = 0.0
        self._suppressed_count: int = 0
        self._spoken_count: int = 0
        self._muted: bool = False

        # ── cooldown_key → 下一次允许播报的时间戳 ──
        self._cooldowns: dict[str, float] = {}

        # ── 外部回调 ──
        # on_spoken: Callable[[str, str, str], None]
        #   (text, source, reason) — 播报成功完成后调用，用于写 TTSEvent 到 DB
        self.on_spoken: object = None
        # on_suppressed: Callable[[TTSRequest, str], None]
        #   (request, suppressed_reason) — 播报被抑制时调用，用于写 AgentDecision 到 DB
        self.on_suppressed: object = None
        # on_log: Callable[[str], None] — 日志回调
        self.on_log = lambda msg: logger.info(msg)

        # ── 连接 Announcer 的 on_spoken 桥接 ──
        # 当 Announcer 完成播报后 → 更新 _last_spoken_at → 调用外部 on_spoken
        announcer.on_spoken = self._on_announcer_spoken

    # ── 公共 API ──────────────────────────────────────────────────────────

    def say(self, request: TTSRequest) -> bool:
        """提交一次 TTS 播报请求。

        经过全局限流、冷却检查、过期检查、Duplex 检查。
        通过检查 → 调用 announcer.announce() 播报。
        未通过 → 调用 on_suppressed 回调，返回 False。

        Returns:
            True = 已提交播报, False = 被抑制
        """
        if self._muted:
            self._log_suppressed(request, "muted")
            return False

        now = time.time()

        # 1. 过期检查
        if request.is_expired(now):
            self._log_suppressed(request, "expired")
            return False

        # 2. 冷却检查 (高优先级绕过)
        if request.priority < PRIORITY_BYPASS_THRESHOLD:
            # 全局冷却
            if self._last_spoken_at > 0 and (now - self._last_spoken_at) < GLOBAL_MIN_INTERVAL_SEC:
                # 尝试排队
                return self._enqueue(request)

            # 同类型冷却
            if self._is_in_cooldown(request.cooldown_key, now):
                self._log_suppressed(request, "cooldown:same_key")
                return False

        # 3. Duplex 状态检查
        if self._duplex.is_vad_suppressed():
            # 正在播报/冷却 → 排队
            return self._enqueue(request)

        # 4. 提交播报
        return self._speak(request)

    def can_speak_now(self) -> bool:
        """当前是否可以立即播报（用于决策前检查）。"""
        if self._muted:
            return False
        if self._duplex.is_vad_suppressed():
            return False
        if len(self._pending) >= MAX_PENDING_TTS:
            return False
        return True

    def get_pending_count(self) -> int:
        """获取待播报队列长度。"""
        return len(self._pending)

    def get_suppressed_count(self) -> int:
        """获取累计抑制次数。"""
        return self._suppressed_count

    def get_spoken_count(self) -> int:
        """获取累计播报次数。"""
        return self._spoken_count

    def get_last_spoken_at(self) -> float:
        """获取上次成功播报的时间戳。"""
        return self._last_spoken_at

    def is_muted(self) -> bool:
        """是否已静音。"""
        return self._muted

    def set_muted(self, muted: bool):
        """设置静音模式（不产生任何声音输出，但仍记录抑制决策）。"""
        self._muted = muted
        self._log(f"🔇 TTSRouter 静音={'开启' if muted else '关闭'}")

    def reset_meeting_state(self):
        """重置会议相关状态（新会议开始时调用）。"""
        self._last_spoken_at = 0.0
        self._suppressed_count = 0
        self._spoken_count = 0
        self._pending.clear()
        self._cooldowns.clear()

    def set_cooldown(self, cooldown_key: str, duration_sec: float):
        """设置某类型的冷却期。

        Args:
            cooldown_key: 去重键
            duration_sec: 冷却时长（秒）
        """
        self._cooldowns[cooldown_key] = time.time() + duration_sec

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _speak(self, request: TTSRequest) -> bool:
        """实际提交播报给 Announcer。"""
        now = time.time()
        self._last_spoken_at = now
        self._spoken_count += 1

        self._log(f"🔊 播报: \"{request.text[:40]}...\" "
                  f"(pri={request.priority} src={request.source})")

        self._announcer.announce(
            request.text,
            source=request.source,
            reason=request.reason,
        )
        return True

    def _enqueue(self, request: TTSRequest) -> bool:
        """尝试将请求加入待播报队列。

        队列满时丢弃优先级最低的过期请求。
        """
        if len(self._pending) >= MAX_PENDING_TTS:
            # 丢弃最低优先级的过期请求
            self._drop_lowest_priority()
            if len(self._pending) >= MAX_PENDING_TTS:
                self._log_suppressed(request, "queue_full")
                return False

        self._pending.append(request)
        self._log(f"  ⏳ 排队 (pending={len(self._pending)}): \"{request.text[:30]}...\"")
        return True

    def _drop_lowest_priority(self):
        """从队列中丢弃最低优先级的过期请求。"""
        if not self._pending:
            return
        now = time.time()
        # 找出最低优先级 + 已过期的请求
        best_idx = -1
        best_pri = 999
        for i, req in enumerate(self._pending):
            if req.is_expired(now) and req.priority < best_pri:
                best_pri = req.priority
                best_idx = i
        if best_idx >= 0:
            dropped = self._pending[best_idx]
            del self._pending[best_idx]
            self._log_suppressed(dropped, "queue_drop:expired_low_priority")
            self._suppressed_count += 1  # 可以在这里计数

    def _is_in_cooldown(self, cooldown_key: str, now: float) -> bool:
        """检查是否仍在冷却期。"""
        if not cooldown_key:
            return False
        next_at = self._cooldowns.get(cooldown_key)
        if next_at is not None and now < next_at:
            return True
        return False

    def _log_suppressed(self, request: TTSRequest, reason: str):
        """记录抑制决策。"""
        self._suppressed_count += 1
        self._log(f"  🚫 抑制 ({reason}): \"{request.text[:30]}...\"")

        if self.on_suppressed is not None:
            try:
                self.on_suppressed(request, reason)  # type: ignore
            except Exception:
                pass

    def _on_announcer_spoken(self, text: str, source: str, reason: str):
        """Announcer 播报完成 → 转发给外部 on_spoken。"""
        if self.on_spoken is not None:
            try:
                self.on_spoken(text, source, reason)  # type: ignore
            except Exception:
                pass

    def _log(self, msg: str):
        if self.on_log:
            self.on_log(msg)
