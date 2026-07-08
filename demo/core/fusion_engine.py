"""视听融合追踪状态机 — 纯 Python，零 GUI 依赖。

将原始 FusionTrackerWindow._tracking_tick() 中的三态状态机逻辑
提取为独立的 FusionEngine 类，通过回调和 EngineOutput 与 GUI 交互。

用法:
  from core.calibration import CalibrationModel
  from core.fusion_engine import FusionEngine, EngineOutput

  model = CalibrationModel()
  model.fit_linear()  # or load from storage

  engine = FusionEngine(
      calibration_model=model,
      params={...},  # all tunable parameters
  )
  engine.set_servo_h = pwm_h.set_angle
  engine.set_servo_v = pwm_v.set_angle
  engine.get_servo_h = pwm_h.get_angle
  engine.get_servo_v = pwm_v.get_angle
  engine.on_log = lambda msg: print(msg)
  engine.on_state_change = lambda old, new, ts: bus.publish("state_changed", ...)

  engine.start()
  # Every 100ms:
  output = engine.tick(now=time.time(), raw_doa=doa, hardware_speech=speech,
                       silero_prob=0.8, silero_is_speech=True, silero_duration=1.5,
                       dev_x=-0.05, dev_y=0.02)
  # Apply output:
  if output.servo_h_target is not None: pwm_h.set_angle(output.servo_h_target)
  # Update UI from output.state_display, output.cooldown_text, etc.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from core.calibration import CalibrationModel
from core.event_bus import EventBus


# ═════════════════════════════════════════════════════════════════════════════════
# EngineOutput — tick() 返回值
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class EngineOutput:
    """FusionEngine.tick() 的返回值。

    包含新状态、舵机目标、UI 显示文本和事件列表。
    所有字段均为纯 Python 类型，零 Qt 依赖。
    """

    state: int = 0
    state_name: str = "IDLE"
    state_display: str = ""       # UI 状态标签文本
    state_style: str = ""         # UI 颜色样式 (CSS color)
    servo_h_target: float | None = None  # None = 不移动
    servo_v_target: float | None = None
    servo_h_current: float = 0.0
    servo_v_current: float = 0.0
    cooldown_text: str = ""       # 冷却/等待/状态文本
    cooldown_style: str = ""      # 颜色
    audio_delta_display: str = "" # 声源偏移显示
    audio_delta_style: str = ""   # 颜色
    log_message: str | None = None
    events: list[dict] = field(default_factory=list)
    moved: bool = False           # 本次 tick 是否执行了舵机移动
    adjustment_h: float = 0.0     # H 轴调整量
    adjustment_v: float = 0.0     # V 轴调整量
    in_cooldown: bool = False     # 是否在舵机冷却期


# ═════════════════════════════════════════════════════════════════════════════════
# FusionEngine
# ═════════════════════════════════════════════════════════════════════════════════

class FusionEngine:
    """视听融合追踪三态状态机 — 纯 Python，零 Qt 依赖。

    IDLE  ──(语音+DOA跳变)──→ AWAIT  ──(稳定/超时)──→ TRACKING ──(失锁)──→ IDLE
                                                    TRACKING ──(声源偏移)──→ AWAIT

    IDLE/AWAIT = 纯听觉 (仅H)    TRACKING = 纯视觉 (H+V)
    """

    # ── 状态枚举 ────────────────────────────────────────────────────────────
    STATE_IDLE     = 0
    STATE_AWAIT    = 1
    STATE_TRACKING = 2
    STATE_NAMES    = {0: "IDLE", 1: "AWAIT", 2: "TRACKING"}

    # ── 默认参数 ────────────────────────────────────────────────────────────
    DEFAULT_PARAMS = {
        "speech_frames":      3,
        "threshold_audio":    10.0,
        "await_duration":     0.5,
        "await_max":          2.0,
        "converged_thresh":   3.0,
        "motor_cooldown":     3.0,
        "visual_lost_frames": 10,
        "audio_jump_thresh":  40.0,
        "jump_cooldown":      1.5,
        "deadzone":           0.08,
        "gain_h":             0.7,
        "gain_v":             0.5,
        "max_angle_v":        10.0,
        "cooldown":           3.0,
        "max_angle_error_h":  30.0,
        "max_angle_error_v":  15.0,
        # 垂直方向偏置：使人物在画面下 2/3 处 (dev_y≈+0.33) 而非画面正中。
        # dev_y = (cy - img_cy) / img_cy，正值=人物在画面中线下方。
        # 控制器驱动 (dev_y - vertical_bias) → 0，即人物稳定在偏置位置。
        "vertical_bias":      0.33,
    }

    def __init__(self, calibration_model: CalibrationModel,
                 params: dict | None = None):
        self._calib = calibration_model

        # ── 合并参数 ────────────────────────────────────────────────────────
        p = dict(self.DEFAULT_PARAMS)
        if params:
            p.update(params)
        self._speech_frames      = p["speech_frames"]
        self._threshold_audio    = p["threshold_audio"]
        self._await_duration     = p["await_duration"]
        self._await_max          = p["await_max"]
        self._converged_thresh   = p["converged_thresh"]
        self._motor_cooldown     = p["motor_cooldown"]
        self._visual_lost_frames = p["visual_lost_frames"]
        self._audio_jump_thresh  = p["audio_jump_thresh"]
        self._jump_cooldown      = p["jump_cooldown"]
        self._deadzone           = p["deadzone"]
        self._gain_h             = p["gain_h"]
        self._gain_v             = p["gain_v"]
        self._max_angle_v        = p["max_angle_v"]
        self._cooldown           = p["cooldown"]
        self._max_angle_error_h  = p["max_angle_error_h"]
        self._max_angle_error_v  = p["max_angle_error_v"]
        self._vertical_bias      = p["vertical_bias"]

        # ── 回调 (由 GUI 注入) ──────────────────────────────────────────────
        self.set_servo_h: Callable[[float], None] = lambda a: None
        self.set_servo_v: Callable[[float], None] = lambda a: None
        self.get_servo_h: Callable[[], float] = lambda: 0.0
        self.get_servo_v: Callable[[], float] = lambda: 0.0
        self.on_log: Callable[[str], None] = lambda m: None
        self.on_state_change: Callable[[int, int, float], None] = lambda o, n, t: None

        # ── 内部状态 ────────────────────────────────────────────────────────
        self._state: int = self.STATE_IDLE
        self._state_enter_time: float = 0.0
        self._active: bool = False

        # 音频
        self._speech_count: int = 0
        self._prev_doa: float = 0.0

        # AWAIT 状态
        self._frozen_doa: float = 0.0
        self._await_countdown: float = 0.0
        self._await_start_time: float = 0.0  # 倒计时起始墙钟时间
        self._recent_deltas: deque = deque(maxlen=5)

        # TRACKING 状态
        self._lost_frame_count: int = 0
        self._last_move_time: float = 0.0
        self._in_cooldown: bool = False
        self._jump_cooldown_until: float = 0.0
        self._latest_audio_delta: float = 0.0

    # ═════════════════════════════════════════════════════════════════════════
    # 参数访问 (供 GUI 读取/写入当前值)
    # ═════════════════════════════════════════════════════════════════════════

    @property
    def state(self) -> int:
        return self._state

    @property
    def state_name(self) -> str:
        return self.STATE_NAMES.get(self._state, "UNKNOWN")

    @property
    def active(self) -> bool:
        return self._active

    def get_params_dict(self) -> dict:
        """返回当前所有参数的字典（供 GUI 持久化）。"""
        return {
            "speech_frames":      self._speech_frames,
            "threshold_audio":    self._threshold_audio,
            "await_duration":     self._await_duration,
            "await_max":          self._await_max,
            "converged_thresh":   self._converged_thresh,
            "motor_cooldown":     self._motor_cooldown,
            "visual_lost_frames": self._visual_lost_frames,
            "audio_jump_thresh":  self._audio_jump_thresh,
            "jump_cooldown":      self._jump_cooldown,
            "deadzone":           self._deadzone,
            "gain_h":             self._gain_h,
            "gain_v":             self._gain_v,
            "max_angle_v":        self._max_angle_v,
            "cooldown":           self._cooldown,
            "max_angle_error_h":  self._max_angle_error_h,
            "max_angle_error_v":  self._max_angle_error_v,
            "vertical_bias":      self._vertical_bias,
        }

    def update_params(self, **kwargs):
        """批量更新参数（从 GUI 控件同步）。"""
        for k, v in kwargs.items():
            if hasattr(self, f"_{k}"):
                setattr(self, f"_{k}", v)

    # ═════════════════════════════════════════════════════════════════════════
    # 生命周期
    # ═════════════════════════════════════════════════════════════════════════

    def start(self):
        """启动状态机，进入 IDLE。"""
        self._active = True
        self._clear_all_state()
        self._enter_state(self.STATE_IDLE)

    def stop(self):
        """停止状态机。"""
        self._active = False

    # ═════════════════════════════════════════════════════════════════════════
    # 核心 tick — 每 100ms 由 GUI QTimer 驱动
    # ═════════════════════════════════════════════════════════════════════════

    def tick(self, now: float,
             raw_doa: float, hardware_speech: bool,
             silero_prob: float, silero_is_speech: bool, silero_duration: float,
             dev_x: float | None, dev_y: float | None,
             vad_enabled_effective: bool = False,
             vad_enabled: bool = True,
             vad_speech_duration: float = 0.3,
             vad_suppressed_by_duplex: bool = False) -> EngineOutput:
        """执行一个状态机周期。

        Args:
            now: time.time() 时间戳
            raw_doa: ReSpeaker DOA 原始角度
            hardware_speech: ReSpeaker 硬件 VAD 标志
            silero_prob: Silero VAD 语音概率 [0, 1]
            silero_is_speech: Silero VAD 判定为语音
            silero_duration: Silero VAD 连续语音时长 (s)
            dev_x: YOLO 人体中心水平偏差 [-1, 1]，None 表示未检测到人
            dev_y: YOLO 人体中心垂直偏差 [-1, 1]
            vad_enabled_effective: Silero VAD 是否实际生效 (用户设置 + 双工抑制)
            vad_enabled: VAD 总开关（GUI 设置）
            vad_speech_duration: VAD 触发所需最小连续语音时长
            vad_suppressed_by_duplex: 双工控制器是否正在抑制 VAD
                (PRE_SPEAKING / SPEAKING / COOLDOWN)。与 vad_enabled_effective
                的区别: 前者表示用户关闭 VAD (允许触发)，后者表示系统正在说话
                (应阻止触发，防止 TTS 音频回声导致误触发)。

        Returns:
            EngineOutput 包含状态信息、舵机目标和 UI 显示文本
        """
        out = EngineOutput(state=self._state,
                           state_name=self.STATE_NAMES.get(self._state, "?"),
                           servo_h_current=self.get_servo_h(),
                           servo_v_current=self.get_servo_v())

        if not self._active:
            return out

        # ── 语音计数更新 ──────────────────────────────────────────────────
        # 双工抑制时强制清零，防止 TTS 回声累积的 speech_count 在状态恢复后误触发
        if vad_suppressed_by_duplex:
            self._speech_count = 0
        elif hardware_speech:
            self._speech_count = min(self._speech_count + 1, self._speech_frames + 5)
        else:
            self._speech_count = max(self._speech_count - 1, 0)

        # ═══════════════════════════════════════════════════════════════════
        # STATE: IDLE
        # ═══════════════════════════════════════════════════════════════════
        if self._state == self.STATE_IDLE:
            # ── 双工抑制: 系统正在播报 TTS，阻止所有触发 ─────────────────
            if vad_suppressed_by_duplex:
                out.cooldown_text = "🔇 双工抑制中 (系统播报)"
                out.cooldown_style = "color: gray;"
                return out

            # ── Silero VAD 语音门控 ──────────────────────────────────────
            if vad_enabled_effective and vad_enabled:
                if not silero_is_speech:
                    out.cooldown_text = f"🔇 等待语音 Silero ({silero_prob:.2f})"
                    out.cooldown_style = "color: gray;"
                    return out
                if silero_duration < vad_speech_duration:
                    out.cooldown_text = (f"🔇 语音累积 {silero_duration:.1f}s/"
                                         f"{vad_speech_duration}s")
                    out.cooldown_style = "color: #cc6600;"
                    return out
            else:
                # 回退：硬件 VAD 模式 (仅当用户禁用 Silero VAD 时)
                if self._speech_count < self._speech_frames:
                    out.cooldown_text = (f"🔇 等待语音 HW… "
                                         f"({self._speech_count}/{self._speech_frames})")
                    out.cooldown_style = "color: gray;"
                    return out

            delta = self._shortest_angular_distance(raw_doa, self.get_servo_h())
            if delta > self._threshold_audio:
                self._frozen_doa = raw_doa
                self._recent_deltas.clear()
                self._enter_state(self.STATE_AWAIT)
                out = self._populate_output(out)
                out.cooldown_text = f"🔔 跳变 {delta:.0f}°"
                out.cooldown_style = "color: orange;"
                out.log_message = f"跳变检测: delta={delta:.1f}°, 冻结={raw_doa:.1f}°"
                return out
            else:
                out.cooldown_text = f"👂 delta={delta:.1f}°≤{self._threshold_audio:.0f}°"
                out.cooldown_style = "color: green;"
                return out

        # ═══════════════════════════════════════════════════════════════════
        # STATE: AWAIT
        # ═══════════════════════════════════════════════════════════════════
        if self._state == self.STATE_AWAIT:
            state_elapsed = now - self._state_enter_time
            current_delta = self._shortest_angular_distance(raw_doa, self._frozen_doa)
            self._recent_deltas.append(current_delta)

            # 规则 1: 新突变 → 刷新
            if current_delta > self._threshold_audio:
                self._frozen_doa = raw_doa
                self._await_countdown = self._await_duration
                self._await_start_time = now
                self._recent_deltas.clear()
                out = self._populate_output(out)
                out.cooldown_text = f"🔄 刷新 (delta={current_delta:.0f}°)"
                out.cooldown_style = "color: orange;"
                out.log_message = f"  AWAIT 刷新: delta={current_delta:.1f}°"
                return out

            # 持续更新冻结参考
            self._frozen_doa = raw_doa
            self._await_countdown = max(0.0, self._await_duration - (now - self._await_start_time))

            # 规则 2: 收敛触发
            if (len(self._recent_deltas) >= 3 and
                    all(d < self._converged_thresh for d in list(self._recent_deltas)[-3:])):
                out = self._trigger_move(self._frozen_doa, "收敛", out)
                return out

            # 规则 3: 倒计时到期
            if self._await_countdown <= 0:
                out = self._trigger_move(self._frozen_doa, "倒计时", out)
                return out

            # 规则 4: 超时
            if state_elapsed >= self._await_max:
                out = self._trigger_move(self._frozen_doa, "超时", out)
                return out

            out.cooldown_text = f"⏳ {self._await_countdown:.1f}s | d={current_delta:.1f}°"
            out.cooldown_style = "color: orange;"
            return out

        # ═══════════════════════════════════════════════════════════════════
        # STATE: TRACKING
        # ═══════════════════════════════════════════════════════════════════
        if self._state == self.STATE_TRACKING:
            state_elapsed = now - self._state_enter_time

            # ── Phase 1: 初始冷却（隔离 AWAIT→TRACKING 的舵机移动噪音）───
            if state_elapsed < self._motor_cooldown:
                remaining = self._motor_cooldown - state_elapsed
                out.cooldown_text = f"🧊 冷却 {remaining:.1f}s …"
                out.cooldown_style = "color: #cc6600; font-weight: bold;"
                out.in_cooldown = True
                return out

            # ── Phase 2: 纯视觉追踪 ──────────────────────────────────────
            out.cooldown_text = ""

            # 声源偏移检测：DOA 显著偏离当前舵机指向 → 回 AWAIT 重捕获
            if self._calib.fitted:
                doa_servo_equiv = self._calib.predict_clamped(raw_doa, -180, 180)
                h_current = self.get_servo_h()
                audio_delta = self._shortest_angular_distance(doa_servo_equiv, h_current)
                self._latest_audio_delta = audio_delta
                in_jump_cooldown = now < self._jump_cooldown_until
                if in_jump_cooldown:
                    out.audio_delta_display = f"声源偏移: {audio_delta:.0f}° (冷却)"
                    out.audio_delta_style = "color: gray;"
                elif audio_delta > self._audio_jump_thresh:
                    # VAD 门控：确认是真正的说话人变更而非噪音
                    # 关键区别：
                    #   - 用户关闭 VAD → vad_ok=True (允许声源跳变)
                    #   - 双工抑制 (TTS 播报) → vad_ok=False (阻止，防回声误触发)
                    #   - VAD 正常 → 检查 Silero 语音确认
                    if vad_suppressed_by_duplex:
                        vad_ok = False
                    elif vad_enabled_effective and vad_enabled:
                        vad_ok = (silero_is_speech and
                                  silero_duration >= vad_speech_duration)
                    else:
                        vad_ok = True  # VAD 被用户关闭，总是允许跳变

                    if vad_ok:
                        out.audio_delta_display = f"声源偏移: {audio_delta:.0f}° ⚠"
                        out.audio_delta_style = "color: red; font-weight: bold;"
                        reason = ""
                        if vad_enabled_effective:
                            reason = (f" (Silero {silero_prob:.2f}, "
                                      f"{silero_duration:.1f}s)")
                        out.log_message = (f"声源偏移 {audio_delta:.0f}° > "
                                           f"{self._audio_jump_thresh:.0f}°"
                                           f"{reason} → AWAIT")
                        out.events.append({
                            "type": "state_changed",
                            "from_state": "TRACKING",
                            "to_state": "AWAIT",
                            "reason": "audio_jump",
                            "audio_delta": audio_delta,
                        })
                        self._clear_all_state()
                        self._frozen_doa = raw_doa
                        self._recent_deltas.clear()
                        self._enter_state(self.STATE_AWAIT)
                        out = self._populate_output(out)
                        return out
                    else:
                        suffix = "双工抑制" if vad_suppressed_by_duplex else "VAD抑制"
                        out.audio_delta_display = f"声源偏移: {audio_delta:.0f}° ({suffix})"
                        out.audio_delta_style = "color: gray;"
                else:
                    out.audio_delta_display = f"声源偏移: {audio_delta:.0f}°"
                    out.audio_delta_style = "color: #cc6600;"

            # 舵机动作间隔冷却
            if self._in_cooldown and (now - self._last_move_time) < self._cooldown:
                out.in_cooldown = True
                return out
            self._in_cooldown = False
            out.in_cooldown = False

            # 无人 → 计数，超阈值回 IDLE
            if dev_x is None:
                self._lost_frame_count += 1
                if self._lost_frame_count >= self._visual_lost_frames:
                    out.log_message = f"视觉失锁 ({self._lost_frame_count}帧) → IDLE"
                    out.events.append({
                        "type": "state_changed",
                        "from_state": "TRACKING",
                        "to_state": "IDLE",
                        "reason": "visual_lock_lost",
                    })
                    self._clear_all_state()
                    self._enter_state(self.STATE_IDLE)
                    out = self._populate_output(out)
                return out
            else:
                self._lost_frame_count = 0

            moved = False
            adjustment_h = 0.0
            adjustment_v = 0.0

            # ── H 方向：纯视觉比例控制 ────────────────────────────────────
            if abs(dev_x) >= self._deadzone:
                adjustment_h = dev_x * self._gain_h * self._max_angle_error_h
                target_h = self.get_servo_h() + adjustment_h
                target_h = max(-135.0, min(135.0, target_h))
                if abs(target_h - self.get_servo_h()) >= 0.5:
                    self.set_servo_h(target_h)
                    moved = True

            # ── V 方向：纯视觉 (带垂直偏置，使人物位于画面下 2/3) ─────────
            # 误差信号 = dev_y - vertical_bias：控制器驱动其 → 0，
            # 即人物稳定在 dev_y == vertical_bias 处（画面中线下方）。
            # bias>0 → 目标位置下移 → 舵机会把人物压低到画面下 2/3。
            if dev_y is not None:
                err_v = dev_y - self._vertical_bias
                if abs(err_v) >= self._deadzone:
                    adjustment_v = err_v * self._gain_v * self._max_angle_error_v
                    target_v = self.get_servo_v() + adjustment_v
                    limit_v = min(self._max_angle_v, 90.0)
                    target_v = max(-limit_v, min(limit_v, target_v))
                    if abs(target_v - self.get_servo_v()) >= 0.3:
                        self.set_servo_v(target_v)
                        moved = True

            if moved:
                self._last_move_time = now
                self._in_cooldown = True
                self._jump_cooldown_until = now + self._jump_cooldown
                out.events.append({
                    "type": "servo_moved",
                    "h_angle": self.get_servo_h(),
                    "v_angle": self.get_servo_v(),
                    "adjustment_h": adjustment_h,
                    "adjustment_v": adjustment_v,
                })

            out.moved = moved
            out.adjustment_h = adjustment_h
            out.adjustment_v = adjustment_v
            out.in_cooldown = self._in_cooldown
            out.servo_h_current = self.get_servo_h()
            out.servo_v_current = self.get_servo_v()
            return out

        return out

    # ═════════════════════════════════════════════════════════════════════════
    # 工具方法
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _shortest_angular_distance(a: float, b: float) -> float:
        """返回两个角度之间的最短距离（度），处理 ±180° 环绕。"""
        diff = abs(a - b) % 360.0
        return min(diff, 360.0 - diff)

    # ═════════════════════════════════════════════════════════════════════════
    # 内部方法
    # ═════════════════════════════════════════════════════════════════════════

    def _clear_all_state(self):
        """重置所有跨状态累积变量。"""
        self._speech_count = 0
        self._recent_deltas.clear()
        self._frozen_doa = 0.0
        self._await_countdown = 0.0
        self._await_start_time = 0.0
        self._lost_frame_count = 0
        self._prev_doa = 0.0
        self._last_move_time = 0.0
        self._in_cooldown = False
        self._jump_cooldown_until = 0.0

    def _enter_state(self, new_state: int):
        """进入新状态。发出日志和状态变更事件。"""
        old = self._state
        self._state = new_state
        self._state_enter_time = time.time()

        # ── AWAIT 进入时初始化倒计时 ──────────────────────────────────
        if new_state == self.STATE_AWAIT:
            self._await_countdown = self._await_duration
            self._await_start_time = self._state_enter_time

        self.on_state_change(old, new_state, self._state_enter_time)

        if old != new_state:
            self.on_log(f"状态: {self.STATE_NAMES[old]} → {self.STATE_NAMES[new_state]}")
            EventBus().publish("state_changed",
                               from_state=self.STATE_NAMES.get(old, "?"),
                               to_state=self.STATE_NAMES.get(new_state, "?"))

            # ── 追踪丢失事件 ──────────────────────────────────────────
            if old == self.STATE_TRACKING and new_state == self.STATE_IDLE:
                EventBus().publish("tracking_lost")

    def _trigger_move(self, target_doa: float, reason: str, out: EngineOutput) -> EngineOutput:
        """AWAIT 结束 → 移动舵机H + 进入 TRACKING。"""
        servo_target = self._calib.predict_clamped(target_doa, -135, 135)
        self.set_servo_h(servo_target)
        self._lost_frame_count = 0
        self._last_move_time = time.time()
        self._enter_state(self.STATE_TRACKING)

        out = self._populate_output(out)
        out.servo_h_target = servo_target
        out.log_message = (f"AWAIT→移动: XVF={target_doa:.1f}° → "
                           f"H={servo_target:.1f}° ({reason})")
        out.events.append({
            "type": "servo_moved",
            "axis": "H",
            "angle": servo_target,
            "reason": reason,
            "trigger": "await_exit",
        })
        return out

    def _populate_output(self, out: EngineOutput) -> EngineOutput:
        """根据当前状态填充 EngineOutput 的 UI 显示字段。"""
        if self._state == self.STATE_IDLE:
            out.state_display = "👂 侦听中"
            out.state_style = "color: green; font-weight: bold;"
            out.audio_delta_display = ""
        elif self._state == self.STATE_AWAIT:
            out.state_display = "⏳ 等待稳定…"
            out.state_style = "color: orange; font-weight: bold;"
            out.audio_delta_display = ""
        elif self._state == self.STATE_TRACKING:
            out.state_display = "🎯 追踪中"
            out.state_style = "color: blue; font-weight: bold;"

        out.state = self._state
        out.state_name = self.STATE_NAMES.get(self._state, "?")
        out.servo_h_current = self.get_servo_h()
        out.servo_v_current = self.get_servo_v()
        return out
