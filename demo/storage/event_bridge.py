"""EventBus → DB 异步事件持久化桥接器。

问题: EventBus 事件以 10-30 Hz 频率从实时控制循环中发出。
     逐个同步写入 SQLite 会给 100ms tick 循环增加磁盘 I/O 延迟。

解决方案: 异步后台写入器 + 有界队列:
  EventBus.publish() → EventBridge._on_event() → Queue(max 5000) → writer thread → batch INSERT

特性:
  - 有界队列: 满时静默丢弃事件 (安全阀)，不阻塞实时循环
  - 批量刷新: 每 1 秒或累积 100 条事件时写入
  - meeting_id_provider: 自动将事件关联到当前会议
  - Daemon 线程: 主进程退出时自动清理

用法:
    bridge = EventBridge(meeting_id_provider=lambda: self._meeting_id)
    bridge.start()
    # ... 系统运行 ...
    bridge.stop()
"""

import json
import queue
import threading
import time
from datetime import datetime
from typing import Callable


class EventBridge:
    """订阅 EventBus 的所有事件，异步批量写入数据库。"""

    FLUSH_INTERVAL_SEC = 1.0     # 最多每 1 秒刷新一次
    MAX_BATCH_SIZE     = 100     # 或累积到 100 条时立即刷新
    QUEUE_MAXSIZE      = 5000    # 队列满时丢弃事件 (安全阀)

    def __init__(self, meeting_id_provider: Callable[[], int | None] = None):
        """
        Args:
            meeting_id_provider: 可调用对象，返回当前会议 ID (或 None)。
                                 用于在事件没有 meeting_id 时自动注入。
        """
        self._queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._running = False
        self._meeting_id_provider = meeting_id_provider
        self._dropped_count = 0
        self._written_count = 0
        self._flush_count = 0

    # ── 公共 API ────────────────────────────────────────────────────────────

    def start(self):
        """启动事件桥接：订阅 EventBus + 启动后台写入线程。"""
        from core.event_bus import EventBus

        bus = EventBus()
        bus.subscribe("*", self._on_event)  # 通配符订阅所有事件

        self._running = True
        self._thread = threading.Thread(target=self._writer_loop, daemon=True,
                                        name="EventBridgeWriter")
        self._thread.start()

    def stop(self, timeout: float = 5.0):
        """停止事件桥接：取消订阅 + 等待后台线程退出 + 刷新残留事件。"""
        self._running = False

        # 尝试取消订阅
        try:
            from core.event_bus import EventBus
            EventBus().unsubscribe("*", self._on_event)
        except Exception:
            pass

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

        # 刷新队列中残留的事件
        self._drain_and_flush()

    # ── 统计信息 ────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        """返回桥接器统计信息。"""
        return {
            "queue_size": self._queue.qsize(),
            "dropped": self._dropped_count,
            "written": self._written_count,
            "flush_count": self._flush_count,
        }

    # ── 内部方法 ────────────────────────────────────────────────────────────

    def _on_event(self, event: dict):
        """EventBus 回调 — 在发布事件的线程上执行。

        将事件放入队列。队列满时静默丢弃。
        """
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self._dropped_count += 1
            # 前 3 条立即告警，之后每 100 条告警一次
            if self._dropped_count <= 3 or self._dropped_count % 100 == 0:
                import sys
                print(f"[EventBridge] WARNING: Queue full, dropped {self._dropped_count} events",
                      file=sys.stderr)

    def _writer_loop(self):
        """后台写入线程: 累积事件，定期批量刷新到数据库。"""
        batch = []
        last_flush = time.time()

        while self._running:
            try:
                event = self._queue.get(timeout=0.5)

                # 自动注入 meeting_id (如果事件中没有且 provider 可用)
                if "meeting_id" not in event and self._meeting_id_provider:
                    mid = self._meeting_id_provider()
                    if mid is not None:
                        event["meeting_id"] = mid

                batch.append(event)
            except queue.Empty:
                pass

            # 检查是否需要刷新
            should_flush = (
                len(batch) >= self.MAX_BATCH_SIZE or
                (batch and time.time() - last_flush >= self.FLUSH_INTERVAL_SEC)
            )
            if should_flush:
                self._flush_batch(batch)
                batch = []
                last_flush = time.time()

    def _drain_and_flush(self):
        """排空队列中所有未处理的事件并写入数据库。"""
        batch = []
        while True:
            try:
                event = self._queue.get_nowait()
                if "meeting_id" not in event and self._meeting_id_provider:
                    mid = self._meeting_id_provider()
                    if mid is not None:
                        event["meeting_id"] = mid
                batch.append(event)
            except queue.Empty:
                break

        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list[dict]):
        """将一批事件写入 events 表。在写入线程中执行。"""
        if not batch:
            return

        try:
            from storage.db import session_scope

            with session_scope() as session:
                from storage.models import Event

                for evt in batch:
                    event_type = evt.get("event_type", "unknown")
                    meeting_id = evt.get("meeting_id")
                    timestamp = evt.get("timestamp", time.time())

                    # 时间戳可能是 float (EventBus 的 time.time())，转为 datetime
                    if isinstance(timestamp, (int, float)):
                        timestamp = datetime.utcfromtimestamp(timestamp)
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.utcnow()

                    # 分离已知字段和 ad-hoc 载荷
                    known = {"event_type", "timestamp"}
                    payload = {k: v for k, v in evt.items()
                              if k not in known and k != "meeting_id"}
                    payload_json = json.dumps(payload, ensure_ascii=False) if payload else None

                    session.add(Event(
                        meeting_id=meeting_id,
                        event_type=event_type,
                        timestamp=timestamp,
                        payload_json=payload_json,
                    ))

            self._written_count += len(batch)
            self._flush_count += 1

        except Exception as e:
            import sys
            print(f"[EventBridge] Batch flush error: {e}", file=sys.stderr)
