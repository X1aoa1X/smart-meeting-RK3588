"""非侵入式追踪指标采集器 + CSV 导出。

用法:
  metrics = MetricsCollector()
  # 在各观测点调用 on_* 钩子方法
  metrics.on_state_change(old, new, timestamp)
  metrics.on_trigger_move(coarse_target_h, timestamp, reason)
  metrics.on_yolo_fps(fps)
  metrics.on_yolo_latency(pre, inf, post, total)
  metrics.on_first_valid_person(timestamp)
  metrics.on_stable_achieved(timestamp)
  metrics.on_tracking_tick(dev_x, dev_y, servo_h, servo_v, moved, adj_h, adj_v, ts, in_cd)
  ...
  # 程序退出前导出
  metrics.set_params(params_dict)
  metrics.export_csvs(output_dir)
"""

import os
import csv
import time
import numpy as np
from datetime import datetime


class MetricsCollector:
    """非侵入式指标采集器 — 通过钩子方法记录追踪全过程指标，程序退出时导出 CSV。

    使用方式:
      - 在追踪窗口 __init__ 中创建实例: self._metrics = MetricsCollector()
      - 在各观测点调用 on_* 方法记录时间戳、偏差、舵机状态等
      - closeEvent 中调用 set_params() + export_csvs() 导出 CSV
    """

    def __init__(self):
        # ── YOLO 性能（全局累积，不按 episode 拆分）────────────────────────────
        self._yolo_fps_samples: list[float] = []
        self._yolo_total_latency: list[float] = []       # 端到端总时延 ms
        self._yolo_preprocess_latency: list[float] = []  # 预处理 ms
        self._yolo_inference_latency: list[float] = []   # RKNN 推理 ms
        self._yolo_postprocess_latency: list[float] = [] # 后处理 ms

        # ── Episode 列表 — 每个 episode = 一次完整的 IDLE→AWAIT→TRACKING→... 循环 ──
        self._episodes: list[dict] = []
        self._current_episode: dict | None = None

        # ── Episode 跨状态临时变量 ──────────────────────────────────────────────
        self._idle_to_await_time: float = 0.0      # IDLE→AWAIT 时刻
        self._trigger_move_time: float = 0.0       # AWAIT→TRACKING (servo move) 时刻
        self._coarse_target_h: float = 0.0         # 粗定位目标舵机 H 角度
        self._await_exit_reason: str = ""          # AWAIT 退出原因

        # ── TRACKING 阶段临时累积 ───────────────────────────────────────────────
        self._visual_lock_achieved: bool = False
        self._first_valid_dev_time: float | None = None
        self._steady_dev_x: list[float] = []       # 稳态 |dev_x| 采样
        self._steady_dev_y: list[float] = []       # 稳态 |dev_y| 采样
        self._adjustments_h: list[float] = []      # 动态 H 调整量
        self._adjustments_v: list[float] = []      # 动态 V 调整量
        self._overshoot_samples: list[float] = []  # |servo_H - coarse_target| 采样
        self._stable_time: float | None = None     # 稳定时刻
        self._stable_achieved: bool = False

        # ── 全局计数 — 用于汇总统计 ─────────────────────────────────────────────
        self._total_idle_to_await: int = 0          # IDLE→AWAIT 跳变总次数
        self._total_tracking_entries: int = 0       # 进入 TRACKING 总次数
        self._tracking_to_await_transitions: int = 0 # TRACKING→AWAIT 重捕获次数
        self._tracking_to_idle_events: int = 0       # TRACKING→IDLE 失锁次数
        self._successful_episodes: int = 0           # 成功追踪 episode 数
        self._false_triggers: int = 0                # 误触发数
        self._successful_reacquisitions: int = 0     # 成功重捕获数
        self._total_tracking_duration_sec: float = 0.0  # TRACKING 状态累计时间

        # ── 会话元信息 ──────────────────────────────────────────────────────────
        self._session_start: str = datetime.now().isoformat()
        self._session_start_ts: float = time.time()
        self._params_snapshot: dict = {}

    # ═════════════════════════════════════════════════════════════════════════
    # Public hook methods — 由追踪窗口调用
    # ═════════════════════════════════════════════════════════════════════════

    def on_state_change(self, old: int, new: int, timestamp: float):
        """在状态转换时调用，追踪 episode 边界。"""
        # IDLE → AWAIT: 新 episode 开始（也可能是重捕获的开始）
        if old == 0 and new == 1:   # IDLE → AWAIT
            self._total_idle_to_await += 1
            # 如果已有未完成的 episode（理论上不应出现），先终结它
            if self._current_episode is not None:
                self._current_episode["disposition"] = "interrupted"
                self._current_episode["end_ts"] = datetime.fromtimestamp(timestamp).isoformat()
                self._finalize_episode()
            # 创建新 episode
            self._current_episode = self._new_episode_dict(timestamp, reacquisition=False)
            self._idle_to_await_time = timestamp

        # TRACKING → AWAIT: 重捕获 — 当前 episode 继续，更新基准时间戳
        elif old == 2 and new == 1:  # TRACKING → AWAIT
            self._tracking_to_await_transitions += 1
            if self._current_episode:
                self._current_episode["reacquisition"] = True
                self._current_episode["idle_to_await_ts_float"] = timestamp
                self._current_episode["idle_to_await_ts"] = datetime.fromtimestamp(timestamp).isoformat()
            # 记录重捕获起始时刻
            self._idle_to_await_time = timestamp

        # TRACKING → IDLE: episode 结束
        elif old == 2 and new == 0:  # TRACKING → IDLE
            self._tracking_to_idle_events += 1
            if self._current_episode:
                elapsed = timestamp - self._current_episode.get("tracking_entry_ts_float", timestamp)
                self._total_tracking_duration_sec += max(0, elapsed)
                self._current_episode["disposition"] = "lock_lost"
                self._current_episode["end_ts"] = datetime.fromtimestamp(timestamp).isoformat()
                self._current_episode["tracking_duration_ms"] = int(max(0, elapsed) * 1000)
                self._finalize_episode()

    def on_trigger_move(self, coarse_target_h: float, timestamp: float, reason: str):
        """在触发舵机移动时调用，记录粗定位目标与 AWAIT 退出原因。"""
        self._trigger_move_time = timestamp
        self._coarse_target_h = coarse_target_h
        self._await_exit_reason = reason
        self._total_tracking_entries += 1
        # 重置 TRACKING 阶段临时变量
        self._visual_lock_achieved = False
        self._first_valid_dev_time = None
        self._steady_dev_x.clear()
        self._steady_dev_y.clear()
        self._adjustments_h.clear()
        self._adjustments_v.clear()
        self._overshoot_samples.clear()
        self._stable_time = None
        self._stable_achieved = False
        # 更新 episode 字典
        if self._current_episode:
            self._current_episode["trigger_move_ts"] = datetime.fromtimestamp(timestamp).isoformat()
            self._current_episode["trigger_move_ts_float"] = timestamp
            self._current_episode["tracking_entry_ts"] = datetime.fromtimestamp(timestamp).isoformat()
            self._current_episode["tracking_entry_ts_float"] = timestamp
            self._current_episode["trigger_reason"] = reason
            self._current_episode["coarse_target_h"] = round(coarse_target_h, 2)

    def on_yolo_fps(self, fps: float):
        """累积 YOLO FPS 样本。"""
        self._yolo_fps_samples.append(fps)

    def on_yolo_latency(self, preprocess_ms: float, inference_ms: float,
                        postprocess_ms: float, total_ms: float):
        """累积 YOLO 推理各阶段时延样本。"""
        self._yolo_preprocess_latency.append(preprocess_ms)
        self._yolo_inference_latency.append(inference_ms)
        self._yolo_postprocess_latency.append(postprocess_ms)
        self._yolo_total_latency.append(total_ms)

    def on_first_valid_person(self, timestamp: float):
        """记录首次检测到人体的时刻（视觉锁定）。"""
        if not self._visual_lock_achieved:
            self._visual_lock_achieved = True
            self._first_valid_dev_time = timestamp
            if self._current_episode:
                self._current_episode["visual_lock_ts"] = datetime.fromtimestamp(timestamp).isoformat()
                self._current_episode["visual_lock_ts_float"] = timestamp

    def on_stable_achieved(self, timestamp: float):
        """记录首次稳定（偏差进入死区且无舵机动作）的时刻。"""
        if not self._stable_achieved:
            self._stable_achieved = True
            self._stable_time = timestamp
            if self._current_episode:
                self._current_episode["stable_ts"] = datetime.fromtimestamp(timestamp).isoformat()
                self._current_episode["stable_ts_float"] = timestamp

    def on_tracking_tick(self, dev_x: float | None, dev_y: float | None,
                         servo_h: float, servo_v: float,
                         moved: bool, adjustment_h: float, adjustment_v: float,
                         timestamp: float, in_cooldown: bool):
        """每个 TRACKING tick (100ms) 调用，累积稳态偏差与动态误差。"""
        if self._current_episode is None:
            return

        # 动态角度误差：仅在有舵机动作时记录
        if moved:
            if adjustment_h != 0.0:
                self._adjustments_h.append(adjustment_h)
            if adjustment_v != 0.0:
                self._adjustments_v.append(adjustment_v)

        # 稳态中心偏差：不在冷却期、不在舵机动作中、有人体检测
        if not moved and not in_cooldown and dev_x is not None:
            self._steady_dev_x.append(abs(dev_x))
            if dev_y is not None:
                self._steady_dev_y.append(abs(dev_y))

    def on_reacquisition_start(self, timestamp: float):
        """TRACKING→AWAIT 重捕获开始的钩子（计数器已在 on_state_change 中更新）。"""
        if self._current_episode:
            self._current_episode["reacquisition"] = True

    def on_tracking_lost(self, timestamp: float):
        """TRACKING→IDLE 失锁的钩子（计数器已在 on_state_change 中处理）。"""
        pass  # 计数器在 on_state_change 中处理

    def set_params(self, params: dict):
        """保存参数快照，供汇总 CSV 使用。"""
        self._params_snapshot = dict(params)

    # ═════════════════════════════════════════════════════════════════════════
    # CSV 导出
    # ═════════════════════════════════════════════════════════════════════════

    def export_csvs(self, output_dir: str):
        """导出 episodes CSV 与 summary CSV 到指定目录。"""
        # 确保未完成的 episode 被终结
        if self._current_episode is not None:
            self._current_episode["disposition"] = \
                self._current_episode.get("disposition") or "interrupted"
            self._current_episode["end_ts"] = datetime.now().isoformat()
            if self._current_episode.get("tracking_entry_ts_float"):
                elapsed = time.time() - self._current_episode["tracking_entry_ts_float"]
                self._current_episode["tracking_duration_ms"] = int(max(0, elapsed) * 1000)
            self._finalize_episode()

        # ── Episodes CSV ──────────────────────────────────────────────────
        ep_path = os.path.join(output_dir, "fusion_metrics_episodes.csv")
        ep_fields = [
            "episode_index", "trigger_reason", "disposition",
            "idle_to_await_ts", "trigger_move_ts", "tracking_entry_ts",
            "visual_lock_ts", "stable_ts", "end_ts",
            "await_duration_ms", "audio_response_ms", "visual_lock_ms",
            "end_to_end_ms", "stable_time_ms",
            "coarse_target_h",
            "steady_dev_x_mean", "steady_dev_y_mean",
            "steady_dev_x_std", "steady_dev_y_std",
            "dynamic_adjustments_h_count", "dynamic_adjustments_h_mean",
            "dynamic_adjustments_h_max",
            "dynamic_adjustments_v_count", "dynamic_adjustments_v_mean",
            "dynamic_adjustments_v_max",
            "overshoot_max_h", "servo_corrections",
            "n_false_trigger", "tracking_duration_ms", "reacquisition",
        ]
        try:
            with open(ep_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=ep_fields, extrasaction='ignore')
                w.writeheader()
                for ep in self._episodes:
                    row = {k: ep.get(k, "") for k in ep_fields}
                    w.writerow(row)
            print(f"[Metrics] Episodes CSV 已导出: {ep_path} ({len(self._episodes)} episodes)")
        except Exception as e:
            print(f"[Metrics] Episodes CSV 导出失败: {e}")

        # ── Summary CSV ───────────────────────────────────────────────────
        summary_path = os.path.join(output_dir, "fusion_metrics_summary.csv")
        summary = self._compute_summary()
        if summary:
            try:
                fields = list(summary.keys())
                with open(summary_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
                    w.writeheader()
                    w.writerow(summary)
                print(f"[Metrics] Summary CSV 已导出: {summary_path}")
            except Exception as e:
                print(f"[Metrics] Summary CSV 导出失败: {e}")

    # ═════════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _new_episode_dict(self, timestamp: float, reacquisition: bool = False) -> dict:
        """创建新 episode 字典模板。"""
        return {
            "episode_index": len(self._episodes),
            "trigger_reason": "",
            "disposition": "",
            "idle_to_await_ts": datetime.fromtimestamp(timestamp).isoformat(),
            "idle_to_await_ts_float": timestamp,
            "trigger_move_ts": "",
            "trigger_move_ts_float": 0.0,
            "tracking_entry_ts": "",
            "tracking_entry_ts_float": 0.0,
            "visual_lock_ts": "",
            "visual_lock_ts_float": 0.0,
            "stable_ts": "",
            "stable_ts_float": 0.0,
            "end_ts": "",
            "await_duration_ms": 0,
            "audio_response_ms": 0,
            "visual_lock_ms": 0,
            "end_to_end_ms": 0,
            "stable_time_ms": 0,
            "coarse_target_h": 0.0,
            "steady_dev_x_mean": None,
            "steady_dev_y_mean": None,
            "steady_dev_x_std": None,
            "steady_dev_y_std": None,
            "dynamic_adjustments_h_count": 0,
            "dynamic_adjustments_h_mean": None,
            "dynamic_adjustments_h_max": None,
            "dynamic_adjustments_v_count": 0,
            "dynamic_adjustments_v_mean": None,
            "dynamic_adjustments_v_max": None,
            "overshoot_max_h": None,
            "servo_corrections": 0,
            "n_false_trigger": 0,
            "tracking_duration_ms": 0,
            "reacquisition": reacquisition,
        }

    def _finalize_episode(self):
        """计算 episode 派生指标，判断成功/误触发，追加到 _episodes 列表。"""
        ep = self._current_episode
        if ep is None:
            return

        # ── 时间差计算 ────────────────────────────────────────────────────
        idle_ts = ep.get("idle_to_await_ts_float", 0)
        move_ts = ep.get("trigger_move_ts_float", 0)
        track_ts = ep.get("tracking_entry_ts_float", 0)
        visual_ts = ep.get("visual_lock_ts_float", 0)
        stable_ts = ep.get("stable_ts_float", 0)

        if idle_ts and move_ts:
            ep["await_duration_ms"] = int(max(0, move_ts - idle_ts) * 1000)
        if idle_ts and track_ts:
            ep["audio_response_ms"] = int(max(0, track_ts - idle_ts) * 1000)
        if track_ts and visual_ts:
            ep["visual_lock_ms"] = int(max(0, visual_ts - track_ts) * 1000)
            ep["end_to_end_ms"] = (ep["audio_response_ms"] or 0) + (ep["visual_lock_ms"] or 0)
        if track_ts and stable_ts:
            ep["stable_time_ms"] = int(max(0, stable_ts - track_ts) * 1000)

        # ── 稳态中心偏差统计 ───────────────────────────────────────────────
        if self._steady_dev_x:
            ep["steady_dev_x_mean"] = round(float(np.mean(self._steady_dev_x)), 4)
            ep["steady_dev_x_std"] = round(float(np.std(self._steady_dev_x)), 4)
        if self._steady_dev_y:
            ep["steady_dev_y_mean"] = round(float(np.mean(self._steady_dev_y)), 4)
            ep["steady_dev_y_std"] = round(float(np.std(self._steady_dev_y)), 4)

        # ── 动态角度误差统计 ───────────────────────────────────────────────
        ep["dynamic_adjustments_h_count"] = len(self._adjustments_h)
        if self._adjustments_h:
            abs_h = np.abs(self._adjustments_h)
            ep["dynamic_adjustments_h_mean"] = round(float(np.mean(abs_h)), 2)
            ep["dynamic_adjustments_h_max"] = round(float(np.max(abs_h)), 2)
        ep["dynamic_adjustments_v_count"] = len(self._adjustments_v)
        if self._adjustments_v:
            abs_v = np.abs(self._adjustments_v)
            ep["dynamic_adjustments_v_mean"] = round(float(np.mean(abs_v)), 2)
            ep["dynamic_adjustments_v_max"] = round(float(np.max(abs_v)), 2)

        # ── 超调量 ─────────────────────────────────────────────────────────
        if self._overshoot_samples:
            ep["overshoot_max_h"] = round(max(self._overshoot_samples), 2)

        # ── 舵机校正次数 ───────────────────────────────────────────────────
        ep["servo_corrections"] = len(self._adjustments_h)

        # ── 误触发判定：进入 TRACKING 但从未视觉锁定 → 误触发 ─────────────
        if not self._visual_lock_achieved:
            ep["n_false_trigger"] = 1
            self._false_triggers += 1
            if ep.get("disposition") != "lock_lost":
                ep["disposition"] = "false_trigger"
        else:
            ep["n_false_trigger"] = 0

        # ── 成功判定：视觉锁定 + 至少一次舵机校正 ──────────────────────────
        if self._visual_lock_achieved and len(self._adjustments_h) > 0:
            if not ep.get("disposition"):
                ep["disposition"] = "success"
            self._successful_episodes += 1
            # 如果是重捕获 episode 且成功，计数
            if ep.get("reacquisition"):
                self._successful_reacquisitions += 1

        # ── 清理内部 float 时间戳 ──────────────────────────────────────────
        for key in list(ep.keys()):
            if key.endswith("_float"):
                del ep[key]

        self._episodes.append(ep)
        self._current_episode = None

    def _compute_summary(self) -> dict:
        """计算会话汇总指标。"""
        session_end = datetime.now().isoformat()
        session_duration = max(0.001, time.time() - self._session_start_ts)

        def _safe_mean(arr: list) -> float | None:
            return round(float(np.mean(arr)), 4) if arr else None

        def _safe_std(arr: list) -> float | None:
            return round(float(np.std(arr)), 4) if arr else None

        def _safe_min(arr: list) -> float | None:
            return round(float(np.min(arr)), 4) if arr else None

        def _safe_max(arr: list) -> float | None:
            return round(float(np.max(arr)), 4) if arr else None

        # YOLO 统计
        yolo_fps_mean = _safe_mean(self._yolo_fps_samples)
        yolo_latency_mean = _safe_mean(self._yolo_total_latency)

        # 追踪统计
        n_episodes = len(self._episodes)
        tracking_success_rate = (
            round(self._successful_episodes / max(1, self._total_tracking_entries), 4)
            if self._total_tracking_entries > 0 else None
        )
        false_trigger_rate = (
            round(self._false_triggers / max(1, self._total_idle_to_await), 4)
            if self._total_idle_to_await > 0 else None
        )
        lock_loss_rate = (
            round(self._tracking_to_idle_events / max(0.001, self._total_tracking_duration_sec / 60.0), 4)
            if self._total_tracking_duration_sec > 0 else None
        )
        reacq_rate = (
            round(self._successful_reacquisitions / max(1, self._tracking_to_await_transitions), 4)
            if self._tracking_to_await_transitions > 0 else None
        )

        # 逐 episode 聚合
        all_audio_resp = [e["audio_response_ms"] for e in self._episodes if e.get("audio_response_ms")]
        all_visual_lock = [e["visual_lock_ms"] for e in self._episodes if e.get("visual_lock_ms")]
        all_e2e = [e["end_to_end_ms"] for e in self._episodes if e.get("end_to_end_ms")]
        all_stable = [e["stable_time_ms"] for e in self._episodes if e.get("stable_time_ms")]
        all_steady_x = [e["steady_dev_x_mean"] for e in self._episodes if e.get("steady_dev_x_mean") is not None]
        all_dyn_h = [e["dynamic_adjustments_h_mean"] for e in self._episodes if e.get("dynamic_adjustments_h_mean") is not None]

        summary = {
            "session_start": self._session_start,
            "session_end": session_end,
            "session_duration_sec": round(session_duration, 1),
            "total_episodes": n_episodes,
            "total_tracking_entries": self._total_tracking_entries,
            "successful_episodes": self._successful_episodes,
            "tracking_success_rate": tracking_success_rate,
            "false_triggers": self._false_triggers,
            "false_trigger_rate": false_trigger_rate,
            "lock_loss_events": self._tracking_to_idle_events,
            "lock_loss_rate_per_min": lock_loss_rate,
            "reacquisition_attempts": self._tracking_to_await_transitions,
            "reacquisition_success_rate": reacq_rate,
            "total_tracking_duration_sec": round(self._total_tracking_duration_sec, 1),
            # YOLO 模块指标
            "yolo_fps_mean": yolo_fps_mean,
            "yolo_fps_min": _safe_min(self._yolo_fps_samples),
            "yolo_fps_max": _safe_max(self._yolo_fps_samples),
            "yolo_fps_std": _safe_std(self._yolo_fps_samples),
            "yolo_latency_mean_ms": yolo_latency_mean,
            "yolo_latency_std_ms": _safe_std(self._yolo_total_latency),
            "yolo_preprocess_mean_ms": _safe_mean(self._yolo_preprocess_latency),
            "yolo_inference_mean_ms": _safe_mean(self._yolo_inference_latency),
            "yolo_postprocess_mean_ms": _safe_mean(self._yolo_postprocess_latency),
            "yolo_latency_samples": len(self._yolo_total_latency),
            # 闭环跟踪 episode 聚合指标
            "avg_audio_response_ms": _safe_mean(all_audio_resp),
            "avg_visual_lock_ms": _safe_mean(all_visual_lock),
            "avg_end_to_end_ms": _safe_mean(all_e2e),
            "avg_stable_time_ms": _safe_mean(all_stable),
            "avg_steady_dev_x": _safe_mean(all_steady_x),
            "avg_dynamic_adjustment_h": _safe_mean(all_dyn_h),
        }

        # ── 参数快照 ─────────────────────────────────────────────────────────
        if self._params_snapshot:
            for k, v in self._params_snapshot.items():
                summary[f"param_{k}"] = v

        return summary
