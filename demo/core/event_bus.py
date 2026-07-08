"""轻量发布/订阅事件总线 — 单例模式，用于系统组件间解耦通信。

线程安全版本：subscribe/unsubscribe/publish 均受 threading.Lock 保护。
publish 在锁内复制订阅者列表，锁外执行回调 — 避免回调中的死锁。

用法:
  from core.event_bus import EventBus
  bus = EventBus()
  bus.subscribe("state_changed", lambda e: print(e))
  bus.publish("state_changed", from_state="IDLE", to_state="AWAIT")
"""

import threading
import time
from collections import defaultdict
from typing import Callable


class EventBus:
    """线程安全的轻量 pub/sub 事件总线 (Singleton)。

    支持简单通配符: subscribe("speaker_*", cb) 匹配 speaker_started 等。

    线程安全设计:
      - subscribe/unsubscribe 获取写锁
      - publish 在锁内复制订阅者列表，在锁外执行回调
        这防止了回调中再次 publish/subscribe 导致的死锁，
        同时避免 "dict changed size during iteration" 运行时错误。
    """

    _instance: "EventBus | None" = None

    def __new__(cls) -> "EventBus":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers = defaultdict(list)
            cls._instance._lock = threading.Lock()
        return cls._instance

    def subscribe(self, event_type: str, callback: Callable[[dict], None]):
        """订阅指定类型的事件。

        Args:
            event_type: 事件类型字符串。以 '*' 结尾表示前缀匹配，
                        例如 "speaker_*" 匹配 "speaker_started", "speaker_ended"。
            callback: 回调函数，接收一个 dict (事件数据，包含 event_type 和 timestamp)。
        """
        with self._lock:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[[dict], None]):
        """取消订阅。"""
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)

    def publish(self, event_type: str, **kwargs):
        """发布事件。自动注入 timestamp。

        线程安全: 在锁内复制订阅者列表，锁外执行回调。

        Args:
            event_type: 事件类型字符串。
            **kwargs: 事件载荷，会与 event_type 和 timestamp 合并。
        """
        event = {"event_type": event_type, "timestamp": time.time()}
        event.update(kwargs)

        # 在锁内复制订阅者列表，避免迭代时 dict 被修改
        with self._lock:
            # 精确匹配
            exact_subs = list(self._subscribers.get(event_type, []))
            # 通配符前缀匹配 (如 "speaker_*" 匹配 "speaker_started")
            wildcard_subs = []
            for pattern, cbs in self._subscribers.items():
                if pattern.endswith("*") and event_type.startswith(pattern[:-1]):
                    wildcard_subs.extend(cbs)

        # 锁外执行回调 — 防止回调中死锁
        for cb in exact_subs:
            self._safe_call(cb, event)
        for cb in wildcard_subs:
            self._safe_call(cb, event)

    @staticmethod
    def _safe_call(callback: Callable[[dict], None], event: dict):
        try:
            callback(event)
        except Exception as e:
            print(f"[EventBus] 回调异常 ({event.get('event_type')}): {e}")

    def clear(self):
        """清空所有订阅（用于测试或重置）。"""
        with self._lock:
            self._subscribers.clear()
