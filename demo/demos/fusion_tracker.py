#!/usr/bin/env python3
"""视听融合立体舵机追踪 Demo — 基于 core/ 模块的 PyQt5 GUI

状态机 (由 core.fusion_engine.FusionEngine 驱动):
  IDLE ──(语音+跳变)──→ AWAIT ──(稳定/超时)──→ TRACKING ──(失锁)──→ IDLE
                                            TRACKING ──(声源偏移)──→ AWAIT

用法:
  sudo env DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \\
    python3 demos/fusion_tracker.py

依赖:
  - core/ 下所有模块
  - xvf_calibration.json (标定文件，由 calibrate_gui.py 生成)
  - models/silero_vad.jit (Silero VAD 模型)
"""

import os
import sys
import time
import signal
import atexit
import json
import queue
from collections import deque
from datetime import datetime

# ── 确保能找到项目根目录的 core/ 模块 ────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 显示环境修复 ──────────────────────────────────────────────────────────
from core.display_env import fix_display_env
fix_display_env()

os.environ.setdefault("DISPLAY", ":0.0")

import cv2
import numpy as np
from core.display_env import fix_cv2_qt_conflict
fix_cv2_qt_conflict()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QGroupBox,
    QStatusBar, QTextEdit, QMessageBox, QCheckBox, QComboBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter

# ── Core 模块导入 ─────────────────────────────────────────────────────────
from core.pwm_controller import PWMController
from core.calibration import CalibrationModel, CalibrationStorage
from core.yolo_camera import YoloCameraThread
from core.respeaker import ReSpeakerReader
from core.audio_vad import AudioVadThread
from core.fusion_engine import FusionEngine, EngineOutput
from core.metrics_collector import MetricsCollector
from core.stream_publisher import StreamThread
from core.event_bus import EventBus
from core.network_utils import get_device_ip
from core.speaker_identity import SpeakerIdentifier, SpeakerIdentity, int_to_tag_id
from storage.models import SpeakerSegment
from core.tag_detect_worker import TagDetectWorker
from core.overlay_renderer import render_overlay
from core.control_api import ControlApiServer
from core.meeting_service import MeetingService

# ── TTS 播报系统 ───────────────────────────────────────────────────────────
from core.audio_player import AudioPlayer
from core.tts_cache import TTSCache
from core.tts_engine import TTSEngine
from core.announcer import Announcer
from core.duplex_controller import DuplexController
from core.tts_router import TTSRouter
from core.tts_policy import TTSRequest, TTSPriority
from core.agent_state import AgentState
from core.agent_rules import RuleEngine, load_agent_policy
from core.agent_worker import AgentWorker

# ── Storage 模块 ─────────────────────────────────────────────────────────────
from storage import init as storage_init
from storage.event_bridge import EventBridge

# ── 参数持久化路径 ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
FUSION_PARAMS_FILE = os.path.join(PROJECT_DIR, "fusion_params.json")


# ═════════════════════════════════════════════════════════════════════════════════
# SplashOverlay — 启动渐变动画 (纯白 → 白底 + 比赛logo)
# ═════════════════════════════════════════════════════════════════════════════════

class SplashOverlay(QWidget):
    """启动渐变动画覆盖层: 从纯白渐变到白底 + 比赛 logo。

    logo 透明度从 0 (纯白) 经 ease-in-out 曲线渐变到 1 (完整显示)，
    持续约 3 秒，覆盖整个父窗口。
    """

    LOGO_PATH = "/home/elf/rknn-dev/demo/competition_logo.png"
    DURATION_MS = 3000

    def __init__(self, parent=None):
        super().__init__(parent)
        self._logo_pixmap = QPixmap(self.LOGO_PATH)
        if self._logo_pixmap.isNull():
            print(f"[Splash] ⚠️ 无法加载 logo: {self.LOGO_PATH}")
        self._opacity = 0.0
        self._start_time = 0.0
        self._on_finished = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60fps
        self._timer.timeout.connect(self._on_tick)
        if parent is not None:
            self.setGeometry(parent.rect())

    def start(self, on_finished=None):
        """启动渐变动画。on_finished 在动画结束时被调用。"""
        self._on_finished = on_finished
        self._start_time = time.time()
        self._opacity = 0.0
        self.raise_()
        self.show()
        self.update()
        self._timer.start()

    def _on_tick(self):
        elapsed = time.time() - self._start_time
        progress = min(elapsed / (self.DURATION_MS / 1000.0), 1.0)
        # ease-in-out: smoothstep
        self._opacity = progress * progress * (3 - 2 * progress)
        self.update()
        if progress >= 1.0:
            self._timer.stop()
            if self._on_finished:
                self._on_finished()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        # 纯白底
        painter.fillRect(self.rect(), Qt.white)
        if self._logo_pixmap.isNull() or self._opacity <= 0:
            return
        # logo 缩放至窗口 80%，保持纵横比
        max_w = int(self.width() * 0.8)
        max_h = int(self.height() * 0.8)
        scaled = self._logo_pixmap.scaled(
            max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.setOpacity(self._opacity)
        painter.drawPixmap(x, y, scaled)

    def resizeEvent(self, event):
        # 跟随父窗口大小
        if self.parent() is not None:
            self.setGeometry(self.parent().rect())


# ═════════════════════════════════════════════════════════════════════════════════
# FusionTrackerWindow — 视听融合追踪主窗口 (UI + 胶水代码)
# ═════════════════════════════════════════════════════════════════════════════════

class FusionTrackerWindow(QMainWindow):
    """PyQt5 主窗口: 三态视听融合追踪 + 双轴舵机控制。

    状态机逻辑由 core.fusion_engine.FusionEngine 处理，
    本窗口仅负责 UI 渲染、线程管理和信号接线。
    """

    # ── 可调参数默认值 (与原始 fusion_tracker_demo.py 一致) ──────────────────
    THRESHOLD_AUDIO    = 10.0
    AWAIT_DURATION     = 0.5
    AWAIT_MAX          = 2.0
    CONVERGED_THRESH   = 3.0
    MOTOR_COOLDOWN     = 3.0
    AUDIO_JUMP_THRESH  = 40.0
    JUMP_COOLDOWN      = 0.5
    # TRACKING 状态下连续未检测到人体的帧数阈值，超过即判定为视觉失锁 → IDLE
    VISUAL_LOST_FRAMES = 10

    # ── Silero VAD 参数 ──────────────────────────────────────────────────────
    VAD_ENABLED         = True
    VAD_SPEECH_DURATION = 0.3
    VAD_THRESHOLD       = 0.05
    VAD_PREGAIN         = 50.0
    VAD_DEVICE          = "hw:1,0"
    VAD_CAPTURE_RATE    = 16000

    # ── 视觉控制参数 ─────────────────────────────────────────────────────────
    DEFAULT_DEADZONE    = 0.08
    DEFAULT_GAIN_H      = 0.7
    DEFAULT_GAIN_V      = 0.5
    DEFAULT_MAX_ANGLE_V = 10.0
    DEFAULT_COOLDOWN    = 3.0
    # 垂直方向偏置：使人物稳定在画面下 2/3 (dev_y≈+0.33)。正值=人物偏下。
    DEFAULT_VERTICAL_BIAS = 0.33

    def __init__(self, pwm_h: PWMController, pwm_v: PWMController | None,
                 model: CalibrationModel):
        super().__init__()
        self.pwm_h = pwm_h
        self.pwm_v = pwm_v
        self.model = model

        # ── 指标采集 ────────────────────────────────────────────────────────
        self._metrics = MetricsCollector()

        # ── 运行时变量 ────────────────────────────────────────────────────────
        self._tracking_active = False

        # 音频
        self._latest_doa    = 0.0
        self._latest_speech = False
        self._speech_count  = 0

        # 视觉
        self._latest_dev_x: float | None = None
        self._latest_dev_y: float | None = None

        # Silero VAD 状态（由 AudioVadThread 写入，主线程只读）
        self._silero_is_speech = False
        self._silero_prob = 0.0
        self._silero_duration = 0.0
        self._vad_enabled_effective = False

        # 视觉控制参数（必须在 _load_params() 之前初始化，因为加载会覆盖它们）
        self._deadzone     = self.DEFAULT_DEADZONE
        self._gain_h       = self.DEFAULT_GAIN_H
        self._gain_v       = self.DEFAULT_GAIN_V
        self._max_angle_v  = self.DEFAULT_MAX_ANGLE_V
        self._cooldown     = self.DEFAULT_COOLDOWN
        self._vertical_bias = self.DEFAULT_VERTICAL_BIAS
        self._fps = 0.0

        # ── 标签检测 + 身份识别 ──────────────────────────────────────────────
        self._latest_tags: list = []
        self._latest_person_box: dict | None = None
        self._latest_identity: SpeakerIdentity | None = None
        self._speaker_identifier: SpeakerIdentifier | None = None
        self._tag_worker: TagDetectWorker | None = None
        self._tag_queue: queue.Queue | None = None

        # ── RTSP 推流名片叠加 ────────────────────────────────────────────────
        # 由主线程 _tracking_tick() 写入，StreamThread 的 GLib 线程读取。
        # Python GIL 保证引用赋值原子性，无需显式锁。
        self._stream_overlay_info: dict | None = None
        self._stream_overlay_state: str = "IDLE"

        # ── 参数加载 (覆盖默认值，必须在引擎创建之前) ─────────────────────────
        self._load_params()

        # ── 融合引擎 ─────────────────────────────────────────────────────────
        engine_params = {
            "threshold_audio":    self.THRESHOLD_AUDIO,
            "await_duration":     self.AWAIT_DURATION,
            "await_max":          self.AWAIT_MAX,
            "converged_thresh":   self.CONVERGED_THRESH,
            "motor_cooldown":     self.MOTOR_COOLDOWN,
            "audio_jump_thresh":  self.AUDIO_JUMP_THRESH,
            "jump_cooldown":      self.JUMP_COOLDOWN,
            "visual_lost_frames": self.VISUAL_LOST_FRAMES,
            "deadzone":           self._deadzone,
            "gain_h":             self._gain_h,
            "gain_v":             self._gain_v,
            "max_angle_v":        self._max_angle_v,
            "cooldown":           self._cooldown,
            "vertical_bias":      self._vertical_bias,
        }
        self._engine = FusionEngine(model, params=engine_params)
        self._engine.set_servo_h = self.pwm_h.set_angle
        if self.pwm_v and self.pwm_v.initialized:
            self._engine.set_servo_v = self.pwm_v.set_angle
            self._engine.get_servo_v = self.pwm_v.get_angle
        self._engine.get_servo_h = self.pwm_h.get_angle
        self._engine.on_log = self._log
        self._engine.on_state_change = self._on_engine_state_change

        # ── 事件持久化桥接 ──────────────────────────────────────────────────
        self._meeting_id: int | None = None            # 当前会议 ID (由控制 API 设置)
        self._event_bridge = EventBridge(
            meeting_id_provider=lambda: self._meeting_id
        )
        self._event_bridge.start()

        # ── 会议服务 + 控制 API ───────────────────────────────────────────────
        self._meeting_service = MeetingService()
        self._speaker_locked = False
        self._tracking_paused = False
        self._overlay_enabled = True

        # ── DB 查询缓存（减少 100ms tick 中的 DB 访问频率）─────────────
        self._cached_participants = ([], None)
        self._participants_cache_ts = 0.0
        self._show_debug_overlay = False

        self._api_server = ControlApiServer(port=8800)
        self._api_server.start()
        # 种子初始状态（Streamlit 在追踪启动前也能看到基本信息）
        self._api_server.update_state({
            "runtime_state": "IDLE",
            "meeting_state": "no_meeting",
            "meeting_id": None,
            "meeting_name": "",
            "current_speaker": None,
            "tracking": {
                "tracking_active": False, "tracking_paused": False,
                "speaker_locked": False, "vad_enabled": self.VAD_ENABLED,
                "vad_is_speech": False,
                "vad_device": self.VAD_DEVICE, "doa_angle": 0.0,
                "pan_angle": self.pwm_h.get_angle(),
                "tilt_angle": self.pwm_v.get_angle() if self.pwm_v else 0.0,
                "yolo_fps": 0.0, "rtsp_status": "stopped", "rtsp_url": "",
            },
            "overlay": {"enabled": True, "show_debug": False},
            "participants": [],
            "timestamp": time.time(),
        })
        self._log(f"🌐 控制 API 已启动: {self._api_server.url}")

        # ── TTS 语音播报系统 (对象创建 — 日志在 _build_ui 之后) ────────
        self._player = AudioPlayer()
        self._player_ok = self._player.init(frequency=16000)

        self._tts_cache = TTSCache(
            cache_dir=os.path.join(PROJECT_DIR, "data", "tts_cache"))

        self._tts_engine = TTSEngine(self)

        self._duplex = DuplexController(parent=self)
        self._duplex.state_changed.connect(self._on_duplex_state_change)

        self._announcer = Announcer(
            cache=self._tts_cache, engine=self._tts_engine,
            player=self._player, duplex=self._duplex, parent=self)
        self._announcer.on_log = self._log

        # ★ 关键: 立即订阅 EventBus（在 meeting_service 发布事件之前）
        self._announcer.start()

        # ── TTS Router（在 Announcer 之上增加策略 + 审计）───────────────────
        self._tts_router = TTSRouter(
            announcer=self._announcer, duplex=self._duplex, parent=self)
        self._tts_router.on_log = self._log
        self._tts_router.on_spoken = self._on_tts_spoken
        self._tts_router.on_suppressed = self._on_tts_suppressed

        # ── Agent Worker（规则引擎 + EventBus 订阅 + 自动播报）──────────────
        self._agent_state = AgentState()
        policy = load_agent_policy()
        self._agent_policy = policy
        self._rule_engine = RuleEngine(policy=policy)
        self._agent_worker = AgentWorker(
            rule_engine=self._rule_engine,
            tts_router=self._tts_router,
            duplex=self._duplex,
            agent_state=self._agent_state,
            on_decision_log=self._on_agent_decision,
            parent=self,
        )
        self._agent_worker.on_log = self._log
        self._agent_worker.start()

        # ── 构建 UI ────────────────────────────────────────────────────────────
        self._build_ui()

        # ── TTS 信号连接（_build_ui 后 _log_text 才存在）─────────────────────
        self._tts_engine.tts_unavailable.connect(
            lambda msg: self._log(f"⚠️ TTS 不可用: {msg}"))
        self._tts_engine.tts_ready.connect(
            lambda: self._log("🔊 TTS 引擎就绪"))
        self._log("🔊 播报器已订阅 EventBus")

        # ── 定时器（100ms 状态机 tick）────────────────────────────────────────
        # 定时器在构造时就启动，确保 API 命令处理 + 状态更新始终运行，
        # 即使 tracking 尚未启动（否则 Streamlit 的命令永远无法被处理）。
        self._track_timer = QTimer(self)
        self._track_timer.timeout.connect(self._tracking_tick)
        self._track_timer.setInterval(100)
        self._track_timer.start()

        # ── 后台线程 (创建 + 信号连接；start 延迟到动画期间) ──────────────────
        self._camera_thread = YoloCameraThread()
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.raw_frame_ready.connect(self._on_raw_frame)
        self._camera_thread.deviation_data.connect(self._on_deviation)
        self._camera_thread.fps_update.connect(self._on_fps)
        self._camera_thread.status_msg.connect(self._on_status)
        self._camera_thread.inference_timing.connect(self._on_inference_timing)

        self._reader = ReSpeakerReader()
        self._reader.doa_update.connect(self._on_doa)
        self._reader.device_error.connect(lambda m: self._log(f"⚠️ {m}"))
        self._reader.device_ready.connect(lambda v: self._log(f"🎤 ReSpeaker 就绪 (v{v})"))

        self._vad_thread: AudioVadThread | None = None

        # ── RTSP 推流 ──────────────────────────────────────────────────────────
        self._stream_thread: StreamThread | None = None
        self._streaming_enabled = False
        self._device_ip = get_device_ip()
        self._stream_queue: queue.Queue | None = None

        # ── 启动渐变动画 + 延迟初始化 ──────────────────────────────────────────
        # 覆盖整个窗口的纯白→白底+logo 渐变动画（~3s），期间逐步执行
        # TTS 缓存预加载、音频设备枚举、摄像头/麦克风启动等可延后的初始化。
        self._splash = SplashOverlay(self)
        self._deferred_steps = [
            self._deferred_tts_diagnostics,
            self._deferred_tts_cache_preload,
            self._deferred_tts_preload,
            self._deferred_vad_devices,
            self._deferred_camera_start,
            self._deferred_reader_start,
        ]
        self._splash.start(on_finished=self._on_splash_finished)
        QTimer.singleShot(0, self._run_next_deferred_step)

    # ── 延迟初始化 (启动动画期间逐步执行) ─────────────────────────────────────

    def _run_next_deferred_step(self):
        """逐步执行延迟初始化，每步之间让出事件循环以保持动画流畅。"""
        if not self._deferred_steps:
            return
        step = self._deferred_steps.pop(0)
        try:
            step()
        except Exception as e:
            self._log(f"⚠️ 延迟初始化失败: {e}")
        if self._deferred_steps:
            QTimer.singleShot(50, self._run_next_deferred_step)

    def _deferred_tts_diagnostics(self):
        self._log(f"🔍 TTS 诊断: {self._tts_engine.diagnose()}")

    def _deferred_tts_cache_preload(self):
        loaded = self._tts_cache.preload_from_disk()
        if loaded:
            self._log(f"📦 TTS 磁盘缓存: {len(loaded)} 条")

    def _deferred_tts_preload(self):
        if self._tts_engine.is_available() and self._player_ok:
            missing = self._tts_cache.get_missing_preloads()
            if missing:
                self._log(f"🔊 预合成 {len(missing)} 条通知语音…")
                self._tts_engine.preload_cache(
                    self._tts_cache, missing,
                    on_progress=lambda cur, tot: self._log(
                        f"  预合成进度: {cur}/{tot}"))
            self._log("🔊 TTS 播报就绪")
        else:
            if not self._player_ok:
                self._log("⚠️ 音频播放器不可用，TTS 播报已禁用")
            if not self._tts_engine.is_available():
                self._log(f"⚠️ TTS 引擎不可用: {self._tts_engine.unavailable_reason}")

    def _deferred_vad_devices(self):
        self._populate_vad_devices()

    def _deferred_camera_start(self):
        self._camera_thread.start()

    def _deferred_reader_start(self):
        self._reader.start()

    def _on_splash_finished(self):
        """渐变动画结束时隐藏覆盖层。"""
        if self._splash is not None:
            self._splash.hide()
            self._splash.deleteLater()
            self._splash = None

    def resizeEvent(self, event):
        """窗口大小变化时同步启动动画覆盖层。"""
        super().resizeEvent(event)
        if getattr(self, '_splash', None) is not None and self._splash.isVisible():
            self._splash.setGeometry(self.rect())

    # ── UI 构建 ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("视听融合追踪 — ReSpeaker + YOLOv8")
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 画面
        self._video_label = QLabel("正在启动摄像头…")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(640, 360)
        self._video_label.setStyleSheet("background: black; color: white;")
        layout.addWidget(self._video_label, stretch=1)

        # 信息面板
        info_layout = QGridLayout()

        self._lbl_fps = QLabel("FPS: --")
        self._lbl_fps.setFont(QFont("", 11, QFont.Bold))
        info_layout.addWidget(QLabel("FPS:"), 0, 0)
        info_layout.addWidget(self._lbl_fps, 0, 1)

        self._lbl_doa = QLabel("XVF: --°")
        self._lbl_doa.setFont(QFont("", 11, QFont.Bold))
        info_layout.addWidget(QLabel("DOA:"), 0, 2)
        info_layout.addWidget(self._lbl_doa, 0, 3)

        self._lbl_speech = QLabel("🔇")
        self._lbl_speech.setFont(QFont("", 11))
        info_layout.addWidget(self._lbl_speech, 0, 4)

        self._lbl_dev = QLabel("偏差: --")
        self._lbl_dev.setFont(QFont("", 11, QFont.Bold))
        info_layout.addWidget(QLabel("视觉:"), 1, 0)
        info_layout.addWidget(self._lbl_dev, 1, 1)

        self._lbl_servo_h = QLabel(f"H: {self.pwm_h.get_angle():.1f}°")
        self._lbl_servo_h.setFont(QFont("", 11, QFont.Bold))
        info_layout.addWidget(QLabel("舵机H:"), 1, 2)
        info_layout.addWidget(self._lbl_servo_h, 1, 3)

        servo_v_text = f"V: {self.pwm_v.get_angle():.1f}°" if self.pwm_v else "V: --"
        self._lbl_servo_v = QLabel(servo_v_text)
        self._lbl_servo_v.setFont(QFont("", 11, QFont.Bold))
        info_layout.addWidget(QLabel("舵机V:"), 1, 4)
        info_layout.addWidget(self._lbl_servo_v, 1, 5)

        self._lbl_state = QLabel("⏸ 停止")
        self._lbl_state.setFont(QFont("", 14, QFont.Bold))
        self._lbl_state.setStyleSheet("color: gray;")
        self._lbl_audio_offset = QLabel("")
        self._lbl_audio_offset.setFont(QFont("", 12, QFont.Bold))
        self._lbl_audio_offset.setStyleSheet("color: #cc6600;")
        self._lbl_cooldown = QLabel("")
        self._lbl_cooldown.setFont(QFont("", 11))
        state_row = QHBoxLayout()
        state_row.addWidget(QLabel("状态:"))
        state_row.addWidget(self._lbl_state)
        state_row.addWidget(self._lbl_audio_offset)
        state_row.addStretch()
        state_row.addWidget(self._lbl_cooldown)
        info_layout.addWidget(QLabel(""), 2, 0)
        info_layout.addLayout(state_row, 2, 1, 1, 5)

        layout.addLayout(info_layout)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self._btn_toggle = QPushButton("▶ 开始追踪")
        self._btn_toggle.setFont(QFont("", 12, QFont.Bold))
        self._btn_toggle.clicked.connect(self._on_toggle)
        btn_layout.addWidget(self._btn_toggle)
        self._btn_center = QPushButton("⬅ 回中")
        self._btn_center.setFont(QFont("", 12))
        self._btn_center.clicked.connect(self._on_center)
        btn_layout.addWidget(self._btn_center)
        self._btn_stream = QPushButton("📡 开始推流")
        self._btn_stream.setFont(QFont("", 12))
        self._btn_stream.clicked.connect(self._on_toggle_stream)
        btn_layout.addWidget(self._btn_stream)
        self._btn_tts_test = QPushButton("🔊 测试 TTS")
        self._btn_tts_test.setFont(QFont("", 12))
        self._btn_tts_test.clicked.connect(self._on_tts_test)
        self._btn_tts_test.setToolTip("播放测试语音，验证 TTS 链路")
        btn_layout.addWidget(self._btn_tts_test)
        self._btn_reconnect_mic = QPushButton("🎤 重连麦克风")
        self._btn_reconnect_mic.setFont(QFont("", 12))
        self._btn_reconnect_mic.setToolTip("重启 ReSpeaker (DOA) + Silero VAD 输入流")
        self._btn_reconnect_mic.clicked.connect(self._on_reconnect_mic)
        btn_layout.addWidget(self._btn_reconnect_mic)
        self._btn_reconnect_cam = QPushButton("📷 重连摄像头")
        self._btn_reconnect_cam.setFont(QFont("", 12))
        self._btn_reconnect_cam.setToolTip("重启摄像头采集 + YOLO 推理线程")
        self._btn_reconnect_cam.clicked.connect(self._on_reconnect_camera)
        btn_layout.addWidget(self._btn_reconnect_cam)
        self._lbl_stream_status = QLabel("推流: 关闭")
        self._lbl_stream_status.setFont(QFont("", 11))
        btn_layout.addWidget(self._lbl_stream_status)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 参数 — 行1: 音频
        param1 = QHBoxLayout()
        param1.addWidget(QLabel("跳变阈值(°):"))
        self._spin_threshold = QDoubleSpinBox()
        self._spin_threshold.setRange(3.0, 60.0)
        self._spin_threshold.setValue(self.THRESHOLD_AUDIO)
        self._spin_threshold.setDecimals(1)
        self._spin_threshold.setSingleStep(5.0)
        self._spin_threshold.valueChanged.connect(lambda v: self._update_param('threshold_audio', v))
        param1.addWidget(self._spin_threshold)

        param1.addWidget(QLabel("稳定计时(s):"))
        self._spin_await = QDoubleSpinBox()
        self._spin_await.setRange(0.2, 2.0)
        self._spin_await.setValue(self.AWAIT_DURATION)
        self._spin_await.setDecimals(1)
        self._spin_await.setSingleStep(0.1)
        self._spin_await.valueChanged.connect(lambda v: self._update_param('await_duration', v))
        param1.addWidget(self._spin_await)

        param1.addWidget(QLabel("冷却(s):"))
        self._spin_cooldown = QDoubleSpinBox()
        self._spin_cooldown.setRange(0.1, 3.0)
        self._spin_cooldown.setValue(self._cooldown)
        self._spin_cooldown.setDecimals(2)
        self._spin_cooldown.setSingleStep(0.05)
        self._spin_cooldown.setSuffix("s")
        self._spin_cooldown.valueChanged.connect(lambda v: self._update_param('cooldown', v))
        param1.addWidget(self._spin_cooldown)

        param1.addWidget(QLabel("声源偏移(°):"))
        self._spin_audio_jump = QDoubleSpinBox()
        self._spin_audio_jump.setRange(10.0, 120.0)
        self._spin_audio_jump.setValue(self.AUDIO_JUMP_THRESH)
        self._spin_audio_jump.setDecimals(0)
        self._spin_audio_jump.setSingleStep(5.0)
        self._spin_audio_jump.setSuffix("°")
        self._spin_audio_jump.valueChanged.connect(lambda v: self._update_param('audio_jump_thresh', v))
        param1.addWidget(self._spin_audio_jump)

        param1.addWidget(QLabel("跳转冷却(s):"))
        self._spin_jump_cooldown = QDoubleSpinBox()
        self._spin_jump_cooldown.setRange(0.1, 3.0)
        self._spin_jump_cooldown.setValue(self.JUMP_COOLDOWN)
        self._spin_jump_cooldown.setDecimals(2)
        self._spin_jump_cooldown.setSingleStep(0.1)
        self._spin_jump_cooldown.setSuffix("s")
        self._spin_jump_cooldown.valueChanged.connect(lambda v: self._update_param('jump_cooldown', v))
        param1.addWidget(self._spin_jump_cooldown)

        param1.addStretch()
        layout.addLayout(param1)

        # 参数 — 行1.5: Silero VAD
        param_vad = QHBoxLayout()
        self._chk_vad_enabled = QCheckBox("VAD启用")
        self._chk_vad_enabled.setChecked(self.VAD_ENABLED)
        self._chk_vad_enabled.toggled.connect(self._on_vad_toggle)
        param_vad.addWidget(self._chk_vad_enabled)

        param_vad.addWidget(QLabel("语音时长(s):"))
        self._spin_vad_duration = QDoubleSpinBox()
        self._spin_vad_duration.setRange(0.1, 3.0)
        self._spin_vad_duration.setValue(self.VAD_SPEECH_DURATION)
        self._spin_vad_duration.setDecimals(1)
        self._spin_vad_duration.setSingleStep(0.1)
        self._spin_vad_duration.setSuffix("s")
        self._spin_vad_duration.valueChanged.connect(lambda v: setattr(self, 'VAD_SPEECH_DURATION', v))
        param_vad.addWidget(self._spin_vad_duration)

        param_vad.addWidget(QLabel("VAD阈值:"))
        self._spin_vad_threshold = QDoubleSpinBox()
        self._spin_vad_threshold.setRange(0.01, 1.0)
        self._spin_vad_threshold.setValue(self.VAD_THRESHOLD)
        self._spin_vad_threshold.setDecimals(3)
        self._spin_vad_threshold.setSingleStep(0.01)
        self._spin_vad_threshold.valueChanged.connect(lambda v: setattr(self, 'VAD_THRESHOLD', v))
        param_vad.addWidget(self._spin_vad_threshold)

        param_vad.addWidget(QLabel("前置增益:"))
        self._spin_vad_pregain = QDoubleSpinBox()
        self._spin_vad_pregain.setRange(1.0, 200.0)
        self._spin_vad_pregain.setValue(self.VAD_PREGAIN)
        self._spin_vad_pregain.setDecimals(1)
        self._spin_vad_pregain.setSingleStep(10.0)
        self._spin_vad_pregain.setSuffix("x")
        self._spin_vad_pregain.valueChanged.connect(lambda v: setattr(self, 'VAD_PREGAIN', v))
        param_vad.addWidget(self._spin_vad_pregain)

        param_vad.addWidget(QLabel("设备:"))
        self._cmb_vad_device = QComboBox()
        self._cmb_vad_device.setMinimumWidth(220)
        self._cmb_vad_device.currentTextChanged.connect(self._on_vad_device_changed)
        param_vad.addWidget(self._cmb_vad_device)

        self._btn_vad_refresh = QPushButton("⟳")
        self._btn_vad_refresh.setFixedWidth(30)
        self._btn_vad_refresh.setToolTip("刷新音频设备列表")
        self._btn_vad_refresh.clicked.connect(self._populate_vad_devices)
        param_vad.addWidget(self._btn_vad_refresh)

        param_vad.addStretch()
        layout.addLayout(param_vad)

        # 参数 — 行2: 视觉
        param2 = QHBoxLayout()
        param2.addWidget(QLabel("死区:"))
        self._spin_deadzone = QDoubleSpinBox()
        self._spin_deadzone.setRange(0.0, 0.5)
        self._spin_deadzone.setValue(self._deadzone)
        self._spin_deadzone.setDecimals(2)
        self._spin_deadzone.setSingleStep(0.01)
        self._spin_deadzone.valueChanged.connect(lambda v: self._update_param('deadzone', v))
        param2.addWidget(self._spin_deadzone)

        param2.addWidget(QLabel("增益H:"))
        self._spin_gain_h = QDoubleSpinBox()
        self._spin_gain_h.setRange(-3.0, 3.0)
        self._spin_gain_h.setValue(self._gain_h)
        self._spin_gain_h.setDecimals(1)
        self._spin_gain_h.setSingleStep(0.1)
        self._spin_gain_h.valueChanged.connect(lambda v: self._update_param('gain_h', v))
        param2.addWidget(self._spin_gain_h)

        param2.addWidget(QLabel("增益V:"))
        self._spin_gain_v = QDoubleSpinBox()
        self._spin_gain_v.setRange(-2.0, 2.0)
        self._spin_gain_v.setValue(self._gain_v)
        self._spin_gain_v.setDecimals(1)
        self._spin_gain_v.setSingleStep(0.1)
        self._spin_gain_v.valueChanged.connect(lambda v: self._update_param('gain_v', v))
        param2.addWidget(self._spin_gain_v)

        param2.addWidget(QLabel("V范围(°):"))
        self._spin_max_angle_v = QDoubleSpinBox()
        self._spin_max_angle_v.setRange(1.0, 90.0)
        self._spin_max_angle_v.setValue(self._max_angle_v)
        self._spin_max_angle_v.setDecimals(1)
        self._spin_max_angle_v.setSingleStep(5.0)
        self._spin_max_angle_v.valueChanged.connect(lambda v: self._update_param('max_angle_v', v))
        param2.addWidget(self._spin_max_angle_v)

        param2.addWidget(QLabel("垂直偏置:"))
        self._spin_vertical_bias = QDoubleSpinBox()
        self._spin_vertical_bias.setRange(-1.0, 1.0)
        self._spin_vertical_bias.setValue(self._vertical_bias)
        self._spin_vertical_bias.setDecimals(2)
        self._spin_vertical_bias.setSingleStep(0.05)
        self._spin_vertical_bias.setToolTip(
            "V轴目标位置偏置：正值使人物稳定在画面下方（下2/3）。\n"
            "dev_y = (人物中心y - 画面中心y) / 画面中心y，范围约[-1,+1]。\n"
            "0.33 ≈ 人物位于画面 2/3 高度处；0 = 居中。")
        self._spin_vertical_bias.valueChanged.connect(lambda v: self._update_param('vertical_bias', v))
        param2.addWidget(self._spin_vertical_bias)

        param2.addWidget(QLabel("失锁帧数:"))
        self._spin_lost_frames = QSpinBox()
        self._spin_lost_frames.setRange(1, 120)
        self._spin_lost_frames.setValue(self.VISUAL_LOST_FRAMES)
        self._spin_lost_frames.setSingleStep(1)
        self._spin_lost_frames.setSuffix(" 帧")
        self._spin_lost_frames.setToolTip(
            "TRACKING 状态下连续未检测到人体的帧数阈值。\n"
            "超过该值即判定为视觉失锁，状态机回到 IDLE。\n"
            "值越大越宽容（抗瞬时遮挡），值越小越灵敏（更快释放）。")
        self._spin_lost_frames.valueChanged.connect(
            lambda v: self._update_param('visual_lost_frames', v))
        param2.addWidget(self._spin_lost_frames)

        param2.addStretch()
        layout.addLayout(param2)

        # 日志
        log_group = QGroupBox("状态日志")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.document().setMaximumBlockCount(200)
        self._log_text.setMaximumHeight(120)
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group)

        # 状态栏
        self._status = QStatusBar()
        self._status.showMessage("就绪 — 点击「开始追踪」启动视听融合追踪")
        self.setStatusBar(self._status)

    # ── 参数同步 ──────────────────────────────────────────────────────────────

    def _update_param(self, key: str, value: float):
        """更新本地属性 + 同步到引擎。"""
        # 更新本地属性
        attr_map = {
            'threshold_audio': 'THRESHOLD_AUDIO',
            'await_duration': 'AWAIT_DURATION',
            'audio_jump_thresh': 'AUDIO_JUMP_THRESH',
            'jump_cooldown': 'JUMP_COOLDOWN',
            'visual_lost_frames': 'VISUAL_LOST_FRAMES',
            'deadzone': '_deadzone',
            'gain_h': '_gain_h',
            'gain_v': '_gain_v',
            'max_angle_v': '_max_angle_v',
            'cooldown': '_cooldown',
            'vertical_bias': '_vertical_bias',
        }
        if key in attr_map:
            setattr(self, attr_map[key], value)
        # 同步到引擎
        self._engine.update_params(**{key: value})

    def _sync_params_to_engine(self):
        """将所有参数同步到引擎。"""
        self._engine.update_params(
            threshold_audio=self.THRESHOLD_AUDIO,
            await_duration=self.AWAIT_DURATION,
            audio_jump_thresh=self.AUDIO_JUMP_THRESH,
            jump_cooldown=self.JUMP_COOLDOWN,
            visual_lost_frames=self.VISUAL_LOST_FRAMES,
            deadzone=self._deadzone,
            gain_h=self._gain_h,
            gain_v=self._gain_v,
            max_angle_v=self._max_angle_v,
            cooldown=self._cooldown,
            vertical_bias=self._vertical_bias,
        )

    # ── 参数持久化 ────────────────────────────────────────────────────────────

    def _load_params(self):
        """从 system_config 表加载已保存的参数（优先），回退到 JSON 文件。"""
        # 1. 尝试从 DB 加载
        try:
            from storage.db import session_scope
            from storage.repo import ConfigRepo
            with session_scope() as session:
                repo = ConfigRepo(session)
                fusion_params = repo.get_section("fusion")
                vad_params = repo.get_section("vad")
            if fusion_params or vad_params:
                if fusion_params:
                    self.THRESHOLD_AUDIO   = float(fusion_params.get("threshold_audio", self.THRESHOLD_AUDIO))
                    self.AWAIT_DURATION    = float(fusion_params.get("await_duration", self.AWAIT_DURATION))
                    self.AUDIO_JUMP_THRESH = float(fusion_params.get("audio_jump_thresh", self.AUDIO_JUMP_THRESH))
                    self.JUMP_COOLDOWN     = float(fusion_params.get("jump_cooldown", self.JUMP_COOLDOWN))
                    self.VISUAL_LOST_FRAMES = int(fusion_params.get("visual_lost_frames", self.VISUAL_LOST_FRAMES))
                    self._cooldown         = float(fusion_params.get("cooldown", self._cooldown))
                    self._deadzone         = float(fusion_params.get("deadzone", self._deadzone))
                    self._gain_h           = float(fusion_params.get("gain_h", self._gain_h))
                    self._gain_v           = float(fusion_params.get("gain_v", self._gain_v))
                    self._max_angle_v      = float(fusion_params.get("max_angle_v", self._max_angle_v))
                    self._vertical_bias    = float(fusion_params.get("vertical_bias", self._vertical_bias))
                if vad_params:
                    self.VAD_ENABLED        = bool(vad_params.get("vad_enabled", self.VAD_ENABLED))
                    self.VAD_SPEECH_DURATION = float(vad_params.get("vad_speech_duration", self.VAD_SPEECH_DURATION))
                    self.VAD_THRESHOLD      = float(vad_params.get("vad_threshold", self.VAD_THRESHOLD))
                    self.VAD_PREGAIN        = float(vad_params.get("vad_pregain", self.VAD_PREGAIN))
                    self.VAD_DEVICE         = str(vad_params.get("vad_device", self.VAD_DEVICE))
                    self.VAD_CAPTURE_RATE   = int(vad_params.get("vad_capture_rate", self.VAD_CAPTURE_RATE))
                print("[Params] 已从 DB 加载参数")
                return
        except Exception as e:
            print(f"[Params] DB 加载失败: {e}，回退到 JSON 文件")

        # 2. 回退到 JSON 文件 (兼容旧版本)
        if not os.path.exists(FUSION_PARAMS_FILE):
            return
        try:
            with open(FUSION_PARAMS_FILE, "r", encoding="utf-8") as f:
                p = json.load(f)
            self.THRESHOLD_AUDIO   = float(p.get("threshold_audio", self.THRESHOLD_AUDIO))
            self.AWAIT_DURATION    = float(p.get("await_duration", self.AWAIT_DURATION))
            self.AUDIO_JUMP_THRESH = float(p.get("audio_jump_thresh", self.AUDIO_JUMP_THRESH))
            self.JUMP_COOLDOWN     = float(p.get("jump_cooldown", self.JUMP_COOLDOWN))
            self.VISUAL_LOST_FRAMES = int(p.get("visual_lost_frames", self.VISUAL_LOST_FRAMES))
            self._cooldown         = float(p.get("cooldown", self._cooldown))
            self._deadzone         = float(p.get("deadzone", self._deadzone))
            self._gain_h           = float(p.get("gain_h", self._gain_h))
            self._gain_v           = float(p.get("gain_v", self._gain_v))
            self._max_angle_v      = float(p.get("max_angle_v", self._max_angle_v))
            self._vertical_bias    = float(p.get("vertical_bias", self._vertical_bias))
            self.VAD_ENABLED       = bool(p.get("vad_enabled", self.VAD_ENABLED))
            self.VAD_SPEECH_DURATION = float(p.get("vad_speech_duration", self.VAD_SPEECH_DURATION))
            self.VAD_THRESHOLD      = float(p.get("vad_threshold", self.VAD_THRESHOLD))
            self.VAD_PREGAIN        = float(p.get("vad_pregain", self.VAD_PREGAIN))
            self.VAD_DEVICE         = str(p.get("vad_device", self.VAD_DEVICE))
            self.VAD_CAPTURE_RATE   = int(p.get("vad_capture_rate", self.VAD_CAPTURE_RATE))
            print(f"[Params] 已从 {FUSION_PARAMS_FILE} 加载参数")
        except Exception as e:
            print(f"[Params] 加载失败: {e}")

    def _save_params(self):
        """保存当前参数到 system_config 表 (同时保留 JSON 作为备份)。"""
        fusion_params = {
            "threshold_audio":     self.THRESHOLD_AUDIO,
            "await_duration":      self.AWAIT_DURATION,
            "audio_jump_thresh":   self.AUDIO_JUMP_THRESH,
            "jump_cooldown":       self.JUMP_COOLDOWN,
            "visual_lost_frames":  self.VISUAL_LOST_FRAMES,
            "cooldown":            self._cooldown,
            "deadzone":            self._deadzone,
            "gain_h":              self._gain_h,
            "gain_v":              self._gain_v,
            "max_angle_v":         self._max_angle_v,
            "vertical_bias":       self._vertical_bias,
        }
        vad_params = {
            "vad_enabled":         self.VAD_ENABLED,
            "vad_speech_duration": self.VAD_SPEECH_DURATION,
            "vad_threshold":       self.VAD_THRESHOLD,
            "vad_pregain":         self.VAD_PREGAIN,
            "vad_device":          self.VAD_DEVICE,
            "vad_capture_rate":    self.VAD_CAPTURE_RATE,
        }

        # 1. 写入 DB
        try:
            from storage.db import session_scope
            from storage.repo import ConfigRepo
            with session_scope() as session:
                repo = ConfigRepo(session)
                repo.set_section("fusion", fusion_params)
                repo.set_section("vad", vad_params)
            print("[Params] 已保存到 DB")
        except Exception as e:
            print(f"[Params] DB 保存失败: {e}")

        # 2. 同时保留 JSON 备份 (便于手动查看和调试)
        try:
            p = {**fusion_params, **vad_params}
            with open(FUSION_PARAMS_FILE, "w", encoding="utf-8") as f:
                json.dump(p, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Params] JSON 备份保存失败: {e}")

    # ── 信号处理 ────────────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        # ── 构建发言人信息（用于叠加渲染）─────────────────────────────────
        speaker_info = None
        if (self._overlay_enabled
                and self._latest_identity is not None
                and self._latest_identity.is_confirmed):
            speaker_info = {
                "name": self._latest_identity.name
                        or f"Tag {self._latest_identity.tag_id}",
                "role": self._latest_identity.role or "",
                "duration": self._latest_identity.duration,
            }

        # ── 叠加渲染（发言人名片条 + 系统状态栏）───────────────────────────
        annotated = render_overlay(
            frame, speaker_info=speaker_info,
            system_state=self._engine.state_name,
            show_debug=self._show_debug_overlay)

        h, w, ch = annotated.shape
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self._video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_raw_frame(self, frame: np.ndarray):
        """接收原始摄像头画面（回退路径，仅当无直连共享队列时使用）。"""
        if self._streaming_enabled and self._stream_thread is not None:
            self._stream_thread.put_frame(frame)

    def _on_deviation(self, data):
        if data is None:
            self._latest_dev_x = None
            self._latest_dev_y = None
            return
        self._latest_dev_x, self._latest_dev_y = data

        # ── 指标：首次检测到人体 → 视觉锁定时刻 ──────────────────────────
        if (self._engine.state == FusionEngine.STATE_TRACKING
                and self._latest_dev_x is not None
                and not self._metrics._visual_lock_achieved):
            self._metrics.on_first_valid_person(time.time())

        if self._latest_dev_x is not None:
            self._lbl_dev.setText(f"H:{self._latest_dev_x:+.3f} V:{self._latest_dev_y:+.3f}"
                                  if self._latest_dev_y is not None else
                                  f"H:{self._latest_dev_x:+.3f}")
            self._lbl_dev.setStyleSheet(
                "color: red;" if abs(self._latest_dev_x) >= self._deadzone else "color: green;")
        else:
            self._lbl_dev.setText("无人")
            self._lbl_dev.setStyleSheet("color: gray;")

    def _on_doa(self, doa_angle: float, speech: bool):
        self._latest_doa    = doa_angle
        self._latest_speech = speech
        if speech:
            self._speech_count = min(self._speech_count + 1, 8)
        else:
            self._speech_count = max(self._speech_count - 1, 0)

        self._lbl_doa.setText(f"{doa_angle:.1f}°")
        if self._vad_enabled_effective and self.VAD_ENABLED:
            if self._silero_is_speech:
                self._lbl_speech.setText(f"🔊 Silero {self._silero_prob:.2f}")
            else:
                self._lbl_speech.setText(f"🔇 Silero {self._silero_prob:.2f}")
        else:
            self._lbl_speech.setText("🔊" if speech else "🔇")

    def _on_silero_speech(self, prob: float, is_speech: bool, duration: float):
        """接收 AudioVadThread 的 Silero VAD 推理结果。"""
        self._silero_prob = prob
        self._silero_is_speech = is_speech
        self._silero_duration = duration

    def _on_vad_toggle(self, checked: bool):
        """VAD 启用/禁用开关。"""
        self.VAD_ENABLED = checked
        if checked and not self._vad_enabled_effective:
            self._log("⚠ VAD 已启用但模型未加载，回退到硬件 VAD")
        elif checked:
            self._log("VAD 已启用 (Silero)")
        else:
            self._log("VAD 已禁用，回退到硬件 VAD")

    # ── VAD 音频设备选择 ──────────────────────────────────────────────────────

    def _populate_vad_devices(self):
        """枚举 ALSA 录音设备并填充下拉框。"""
        try:
            from core.alsa_device_list import list_capture_devices
            devices = list_capture_devices()
        except Exception as e:
            self._log(f"⚠️ 枚举音频设备失败: {e}")
            devices = []

        self._cmb_vad_device.blockSignals(True)
        self._cmb_vad_device.clear()

        if not devices:
            self._cmb_vad_device.addItem(self.VAD_DEVICE)  # 回退: 仅显示已保存的设备
        else:
            for dev in devices:
                label = f"{dev['card_name']} ({dev['name']})"
                self._cmb_vad_device.addItem(label, dev["name"])

            # 选中当前设备
            current = self.VAD_DEVICE
            idx = self._cmb_vad_device.findData(current)
            if idx >= 0:
                self._cmb_vad_device.setCurrentIndex(idx)
            elif self._cmb_vad_device.count() > 0:
                # 当前设备不在列表中 — 选第一个并更新
                self._cmb_vad_device.setCurrentIndex(0)
                new_device = self._cmb_vad_device.currentData()
                if new_device:
                    self.VAD_DEVICE = new_device

        self._cmb_vad_device.blockSignals(False)

    def _on_vad_device_changed(self, text: str):
        """VAD 设备下拉框切换 — 重启 VAD 线程。"""
        if not text or not hasattr(self, '_cmb_vad_device'):
            return
        new_device = self._cmb_vad_device.currentData()
        if new_device and new_device != self.VAD_DEVICE:
            self.VAD_DEVICE = new_device
            self._log(f"🔊 VAD 设备已切换: {new_device}")
            # 如果正在追踪且未暂停，重启 VAD 线程
            if self._tracking_active and not self._tracking_paused:
                self._start_vad()
            self._save_params()

    def _on_fps(self, fps: float):
        self._fps = fps
        self._lbl_fps.setText(f"{fps:.1f}")
        if self._tracking_active:
            self._metrics.on_yolo_fps(fps)

    def _on_inference_timing(self, timing: dict):
        """接收 CameraThread 的 YOLO 推理各阶段时延。"""
        self._metrics.on_yolo_latency(
            timing.get("preprocess_ms", 0.0),
            timing.get("inference_ms", 0.0),
            timing.get("postprocess_ms", 0.0),
            timing.get("total_ms", 0.0),
        )

    def _on_status(self, msg: str):
        self._status.showMessage(msg)
        print(f"[Camera] {msg}")

    def _on_engine_state_change(self, old: int, new: int, timestamp: float):
        """FusionEngine 状态变更回调 — 转发给 MetricsCollector。"""
        self._metrics.on_state_change(old, new, timestamp)

    def _on_duplex_state_change(self, old: int, new: int):
        """半双工音频状态变更回调。"""
        names = {0: "LISTENING", 1: "SPEAKING", 2: "COOLDOWN", 3: "RECORDING"}
        self._log(f"🔊 双工: {names.get(old, '?')} → {names.get(new, '?')}")

    # ── 标签检测 + 身份识别回调 ────────────────────────────────────────────────

    def _on_person_box(self, box: dict | None):
        """接收 YoloCameraThread 的人体边界框。"""
        self._latest_person_box = box

    def _on_apriltag_result(self, tags: list):
        """接收 TagDetectWorker 的 AprilTag 检测结果。"""
        self._latest_tags = tags

    @staticmethod
    def _lookup_participant(tag_id_str: str) -> dict | None:
        """在数据库中查找 tag_id 对应的参与人信息。

        在主线程中调用，使用 session_scope 确保线程安全。
        """
        try:
            from storage.db import session_scope
            from storage.repo import ParticipantRepo
            with session_scope() as session:
                repo = ParticipantRepo(session)
                p = repo.get_by_tag_id(tag_id_str)
                if p is None:
                    return None
                return {
                    "name": p.name,
                    "role": p.role,
                    "organization": p.organization,
                }
        except Exception as e:
            print(f"[Identity] 参与人查询失败 ({tag_id_str}): {e}")
            return None

    def _on_speaker_event(self, event: dict):
        """接收 SpeakerIdentifier 的发言人变更事件。

        1. 发布到 EventBus（由 EventBridge 异步写入 events 表）
        2. 同步写入 SpeakerSegment 表（开始/结束发言片段）
        """
        event_type = event.get("event_type", "")
        tag_id = event.get("tag_id") or event.get("new_tag_id")

        # ── 通过 EventBus 发布（EventBridge 自动写入 events 表）─────────────
        bus = EventBus()
        # event dict 已包含 "event_type"，需排除避免重复传参
        bus.publish(event_type,
                    **{k: v for k, v in event.items() if k != "event_type"})

        # ── 写入 SpeakerSegment 表 ──────────────────────────────────────────
        meeting_id = self._meeting_id
        if meeting_id is None:
            return

        try:
            from storage.db import session_scope
            from storage.repo import SpeakerSegmentRepo

            with session_scope() as session:
                repo = SpeakerSegmentRepo(session)

                if event_type in ("speaker_started", "speaker_switched"):
                    repo.start_segment(
                        meeting_id=meeting_id,
                        speaker_tag_id=tag_id,
                        speaker_name=event.get("name"),
                        role=event.get("role"),
                        source=event.get("source", "AprilTag"),
                        confidence=event.get("confidence", 0.0),
                    )
                    self._log(f"🎤 发言人: {event.get('name') or tag_id} "
                             f"({event.get('role', '')}) [{event.get('source', '')}]")

                elif event_type == "speaker_ended":
                    active = repo.get_active_segment(meeting_id)
                    if active is not None:
                        repo.end_segment(active)
                        self._log(f"🔇 发言结束: {event.get('name') or tag_id} "
                                 f"({active.duration_seconds:.1f}s)")
        except Exception as e:
            print(f"[Identity] DB 写入失败 ({event_type}): {e}")

    # ── 追踪启停 ───────────────────────────────────────────────────────────────

    def _on_toggle(self):
        if self._tracking_active:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        if not self.pwm_h.initialized:
            QMessageBox.warning(self, "无法追踪", "水平 PWM 未初始化（需 root 权限）。")
            return
        if not self.model.fitted and len(self.model.points) < 2:
            QMessageBox.warning(self, "无法追踪",
                                "请先用 calibrate_gui.py 完成 XVF-Servo 标定。")
            return

        self._tracking_active = True
        self.model.reset_unwrap()
        self._engine.start()

        # ── 启动 Silero VAD 线程 ────────────────────────────────────────────
        self._start_vad()

        # ── 启动标签检测 + 身份识别 ─────────────────────────────────────────
        self._start_identity()

        self._track_timer.start()
        self._btn_toggle.setText("⏹ 停止追踪")
        self._log("追踪已启动 (视听融合模式)")

    def _stop_tracking(self):
        self._tracking_active = False
        self._tracking_paused = False
        # 不停止 _track_timer — 保持命令处理和状态更新，避免 Streamlit 死锁
        # 每步独立 try/except：单点异常不应阻塞后续清理，避免状态不一致
        self._safe_cleanup("engine.stop",  self._engine.stop)

        # ── 停止 Silero VAD 线程 ────────────────────────────────────────────
        self._safe_cleanup("stop_vad",     self._stop_vad)

        # ── 停止标签检测 + 身份识别 ─────────────────────────────────────────
        self._safe_cleanup("stop_identity", self._stop_identity)

        self._lbl_state.setText("⏸ 停止")
        self._lbl_state.setStyleSheet("color: gray;")
        self._lbl_cooldown.setText("")
        self._lbl_audio_offset.setText("")
        self._btn_toggle.setText("▶ 开始追踪")
        self._safe_cleanup("duplex.reset", self._duplex.reset)
        self._log("追踪已停止")

    def _start_vad(self):
        """启动 Silero VAD 线程（如果启用）。"""
        if not self.VAD_ENABLED:
            self._vad_enabled_effective = False
            self._log("VAD: 已禁用，使用硬件 VAD")
            return

        self._stop_vad()

        self._vad_thread = AudioVadThread(
            device=self.VAD_DEVICE, capture_rate=self.VAD_CAPTURE_RATE,
            threshold=self.VAD_THRESHOLD,
            min_speech_duration=self.VAD_SPEECH_DURATION,
            pregain=self.VAD_PREGAIN)
        self._vad_thread.silero_speech.connect(self._on_silero_speech)
        self._vad_thread.vad_error.connect(self._on_vad_error)
        self._vad_thread.vad_ready.connect(self._on_vad_ready)
        self._vad_thread.start()

    def _on_vad_error(self, msg: str):
        """VAD 线程瞬时错误回调 — 仅日志，线程自动恢复中。

        Silero VAD 是唯一语音检测来源，不降级到硬件 VAD。
        """
        self._log(f"⚠️ VAD: {msg}")

    def _on_vad_ready(self, msg: str):
        """VAD 线程就绪/恢复成功回调。"""
        self._vad_enabled_effective = True
        self._log(f"✅ VAD 就绪: {msg}")

    def _stop_vad(self):
        """停止 Silero VAD 线程。"""
        self._vad_enabled_effective = False
        self._silero_is_speech = False
        self._silero_prob = 0.0
        self._silero_duration = 0.0
        if self._vad_thread is not None:
            if self._vad_thread.isRunning():
                self._vad_thread.stop()
            self._vad_thread = None

    # ── 标签检测 + 身份识别 启停 ──────────────────────────────────────────────

    def _start_identity(self):
        """启动标签检测工作线程和发言人身份识别器。"""
        # ── 创建标签检测帧队列 + 工作线程 ──────────────────────────────────
        self._tag_queue = queue.Queue(maxsize=2)
        self._tag_worker = TagDetectWorker(frame_queue=self._tag_queue)
        self._tag_worker.tags_ready.connect(self._on_apriltag_result)
        self._tag_worker.fps_update.connect(self._on_tag_fps)
        self._tag_worker.status_msg.connect(self._log)
        self._tag_worker.start()

        # ── 将队列注入 CameraThread ─────────────────────────────────────────
        self._camera_thread.set_tag_queue(self._tag_queue)

        # ── 连接人体框信号 ──────────────────────────────────────────────────
        self._camera_thread.person_box_ready.connect(self._on_person_box)

        # ── 创建身份识别器 ──────────────────────────────────────────────────
        self._speaker_identifier = SpeakerIdentifier(confirm_frames=3, lost_timeout=5.0)
        self._speaker_identifier.set_participant_lookup(self._lookup_participant)
        self._speaker_identifier.on_speaker_event = self._on_speaker_event

        self._latest_tags = []
        self._latest_person_box = None
        self._latest_identity = None

        self._log("🏷️ 标签检测 + 身份识别已启动")

    def _stop_identity(self):
        """停止标签检测工作线程和身份识别器。"""
        # ── 断开 CameraThread 的标签队列 ────────────────────────────────────
        if self._camera_thread is not None:
            self._camera_thread.set_tag_queue(None)
            try:
                self._camera_thread.person_box_ready.disconnect(self._on_person_box)
            except Exception:
                pass

        # ── 结束当前发言片段 ──────────────────────────────────────────────────
        if self._meeting_id is not None and self._latest_identity is not None:
            if self._latest_identity.is_confirmed:
                try:
                    from storage.db import session_scope
                    from storage.repo import SpeakerSegmentRepo
                    with session_scope() as session:
                        repo = SpeakerSegmentRepo(session)
                        active = repo.get_active_segment(self._meeting_id)
                        if active is not None:
                            repo.end_segment(active)
                except Exception as e:
                    print(f"[Identity] 结束发言片段失败: {e}")

        # ── 停止工作线程 ────────────────────────────────────────────────────
        if self._tag_worker is not None:
            if self._tag_worker.isRunning():
                self._tag_worker.stop()
            self._tag_worker = None

        self._tag_queue = None
        self._speaker_identifier = None
        self._latest_tags = []
        self._latest_person_box = None
        self._latest_identity = None

        self._log("🏷️ 标签检测 + 身份识别已停止")

    def _on_tag_fps(self, fps: float):
        """标签检测 FPS 更新 — 仅记录到状态栏。"""
        pass  # 可选: 后续添加到 UI

    def _on_center(self):
        self.pwm_h.set_angle(0.0)
        self._lbl_servo_h.setText("H: 0.0°")
        if self.pwm_v and self.pwm_v.initialized:
            self.pwm_v.set_angle(0.0)
            self._lbl_servo_v.setText("V: 0.0°")
        self._log("舵机已回中")

    # ── 推流控制 ────────────────────────────────────────────────────────────────

    def _get_stream_overlay(self) -> tuple:
        """StreamThread 的 overlay 回调 — 返回当前发言人信息。

        由 StreamThread 的 GLib 线程调用，读取主线程写入的原子引用。
        Returns:
            (speaker_info: dict|None, system_state: str)
        """
        return (self._stream_overlay_info, self._stream_overlay_state)

    def _on_toggle_stream(self):
        """推流按钮回调：切换 RTSP 推流启停。"""
        if self._streaming_enabled:
            self._stop_streaming()
        else:
            self._start_streaming()

    def _on_tts_test(self):
        """测试 TTS 按钮回调 — 播报测试语音。"""
        self._log("🧪 手动触发 TTS 测试…")
        self._announcer.test("这是测试语音")

    # ── 硬件重连 ────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_disconnect(signal, slot=None):
        """断开 Qt 信号连接，忽略未连接异常。"""
        try:
            if slot is not None:
                signal.disconnect(slot)
            else:
                signal.disconnect()
        except (TypeError, RuntimeError):
            pass

    def _on_reconnect_mic(self):
        """重连麦克风按钮回调 — 重建 ReSpeakerReader (DOA) + AudioVadThread (VAD)。

        QThread 不可重启，因此采用 stop + 新建实例 + start 的模式。
        VAD 线程仅在追踪中且未暂停时重建，否则只清理状态。
        """
        self._log("🎤 正在重连麦克风 (DOA + VAD)…")
        self._btn_reconnect_mic.setEnabled(False)

        # ── 1. 重建 ReSpeakerReader (DOA) ────────────────────────────────
        try:
            old_reader = self._reader
            self._safe_disconnect(old_reader.doa_update)
            self._safe_disconnect(old_reader.device_error)
            self._safe_disconnect(old_reader.device_ready)
            if old_reader.isRunning():
                old_reader.stop()
            self._reader = ReSpeakerReader()
            self._reader.doa_update.connect(self._on_doa)
            self._reader.device_error.connect(lambda m: self._log(f"⚠️ {m}"))
            self._reader.device_ready.connect(
                lambda v: self._log(f"🎤 ReSpeaker 就绪 (v{v})"))
            self._reader.start()
            self._log("🎤 DOA 线程已重建并启动")
        except Exception as e:
            self._log(f"⚠️ DOA 重连失败: {e}")

        # ── 2. 重建 AudioVadThread (VAD 输入流) ──────────────────────────
        try:
            if self._tracking_active and not self._tracking_paused and self.VAD_ENABLED:
                self._start_vad()  # 内部已 stop + 重建
                self._log("🔊 VAD 线程已重建并启动")
            else:
                self._stop_vad()
                self._log("🔊 VAD 线程未运行（追踪未激活），已清理状态")
        except Exception as e:
            self._log(f"⚠️ VAD 重连失败: {e}")

        self._btn_reconnect_mic.setEnabled(True)
        self._log("🎤 麦克风重连已触发")

    def _on_reconnect_camera(self):
        """重连摄像头按钮回调 — 重建 YoloCameraThread。

        QThread 不可重启，因此采用 stop + 新建实例 + start 的模式。
        保留当前 stream_queue / tag_queue 注入，并在追踪中重新连接 person_box_ready。
        """
        self._log("📷 正在重连摄像头…")
        self._btn_reconnect_cam.setEnabled(False)

        old_cam = self._camera_thread
        saved_stream_queue = self._stream_queue
        saved_tag_queue = self._tag_queue
        identity_active = (self._tracking_active
                           and self._speaker_identifier is not None)

        # ── 1. 断开旧信号（防止 GC 延迟导致重复触发）──────────────────────
        self._safe_disconnect(old_cam.frame_ready)
        self._safe_disconnect(old_cam.raw_frame_ready)
        self._safe_disconnect(old_cam.deviation_data)
        self._safe_disconnect(old_cam.fps_update)
        self._safe_disconnect(old_cam.status_msg)
        self._safe_disconnect(old_cam.inference_timing)
        self._safe_disconnect(old_cam.person_box_ready)

        # ── 2. 停止旧线程 ────────────────────────────────────────────────
        try:
            if old_cam.isRunning():
                old_cam.stop()
                if old_cam.isRunning():
                    self._log("⚠️ 旧摄像头线程仍在退出中（可能阻塞片刻）")
        except Exception as e:
            self._log(f"⚠️ 旧摄像头线程停止失败: {e}")

        # ── 3. 创建新实例 + 重新连接信号 ─────────────────────────────────
        try:
            self._camera_thread = YoloCameraThread()
            self._camera_thread.frame_ready.connect(self._on_frame)
            self._camera_thread.raw_frame_ready.connect(self._on_raw_frame)
            self._camera_thread.deviation_data.connect(self._on_deviation)
            self._camera_thread.fps_update.connect(self._on_fps)
            self._camera_thread.status_msg.connect(self._on_status)
            self._camera_thread.inference_timing.connect(self._on_inference_timing)
            if identity_active:
                self._camera_thread.person_box_ready.connect(self._on_person_box)

            # ── 4. 重新注入队列（追踪/推流可能正在使用）─────────────────
            if saved_stream_queue is not None and self._streaming_enabled:
                self._camera_thread.set_stream_queue(saved_stream_queue)
            if saved_tag_queue is not None and self._tracking_active:
                self._camera_thread.set_tag_queue(saved_tag_queue)

            self._camera_thread.start()
            self._log("📷 摄像头线程已重建并启动")
        except Exception as e:
            self._log(f"⚠️ 摄像头重建失败: {e}")

        self._btn_reconnect_cam.setEnabled(True)
        self._log("📷 摄像头重连已触发")

    def _start_streaming(self):
        """启动 RTSP 推流。"""
        self._stream_queue = queue.Queue(maxsize=2)
        self._camera_thread.set_stream_queue(self._stream_queue)

        self._stream_thread = StreamThread(
            device_ip=self._device_ip,
            width=960, height=540, fps=25, bitrate=2_000_000,
            frame_queue=self._stream_queue)
        self._stream_thread.set_overlay_callback(self._get_stream_overlay)
        self._stream_thread.stream_started.connect(self._on_stream_started)
        self._stream_thread.stream_error.connect(self._on_stream_error)
        self._stream_thread.stream_stopped.connect(self._on_stream_stopped)
        self._stream_thread.status_msg.connect(self._on_stream_status)
        self._stream_thread.start()
        self._streaming_enabled = True
        self._btn_stream.setText("📡 停止推流")
        self._log(f"推流启动中… rtsp://{self._device_ip}:8554/smartcam")

    def _stop_streaming(self):
        """停止 RTSP 推流。"""
        if not self._streaming_enabled and self._stream_thread is None:
            return
        self._streaming_enabled = False
        self._camera_thread.set_stream_queue(None)
        self._stream_queue = None
        if self._stream_thread is not None:
            if self._stream_thread.isRunning():
                self._stream_thread.stop()
            self._stream_thread = None
        self._btn_stream.setText("📡 开始推流")
        self._lbl_stream_status.setText("推流: 关闭")
        self._lbl_stream_status.setStyleSheet("color: gray;")

    def _on_stream_started(self, url: str):
        """推流就绪回调。"""
        self._lbl_stream_status.setText(f"推流: {url}")
        self._lbl_stream_status.setStyleSheet("color: green; font-weight: bold;")
        self._log(f"推流已启动: {url}")

    def _on_stream_error(self, msg: str):
        """推流错误回调。"""
        self._lbl_stream_status.setText("推流: 错误")
        self._lbl_stream_status.setStyleSheet("color: red; font-weight: bold;")
        self._log(f"推流错误: {msg}")
        self._streaming_enabled = False
        self._btn_stream.setText("📡 开始推流")

    def _on_stream_stopped(self):
        """推流已停止回调。"""
        self._lbl_stream_status.setText("推流: 已停止")
        self._lbl_stream_status.setStyleSheet("color: gray;")

    def _on_stream_status(self, msg: str):
        """推流状态消息回调。"""
        self._status.showMessage(msg)

    # ── 核心：状态机 tick (薄包装) ─────────────────────────────────────────────

    def _tracking_tick(self):
        """状态机主循环，每 100ms 执行一次。委托给 FusionEngine。

        始终处理 API 命令 + 更新状态快照，确保 Streamlit 在任何状态下都能通信。
        """
        if not self._tracking_active:
            # tracking 尚未启动 — 仍处理命令 + 更新状态（否则死锁）
            self._update_api_state_idle()
            self._process_api_commands()
            # Agent Worker 仍需处理命令
            self._agent_worker.tick()
            return

        # ── 暂停检查 ──────────────────────────────────────────────────────
        if self._tracking_paused:
            # 仍然更新 API 状态（让 Streamlit 看到暂停状态）
            self._update_api_state_tick()
            self._process_api_commands()
            self._agent_worker.tick()
            return

        now = time.time()
        self._speech_count = min(self._speech_count, 8)

        # ── 半双工: TTS 播报时抑制 VAD ───────────────────────────────────────
        vad_duplex_suppressed = self._duplex.is_vad_suppressed()
        vad_effective = self._vad_enabled_effective
        if vad_duplex_suppressed:
            vad_effective = False

        # ── 调用引擎 ──────────────────────────────────────────────────────
        output = self._engine.tick(
            now=now,
            raw_doa=self._latest_doa,
            hardware_speech=self._latest_speech,
            silero_prob=self._silero_prob,
            silero_is_speech=self._silero_is_speech,
            silero_duration=self._silero_duration,
            dev_x=self._latest_dev_x,
            dev_y=self._latest_dev_y,
            vad_enabled_effective=vad_effective,
            vad_enabled=self.VAD_ENABLED,
            vad_speech_duration=self.VAD_SPEECH_DURATION,
            vad_suppressed_by_duplex=vad_duplex_suppressed,
        )

        # ── 应用输出到 UI ─────────────────────────────────────────────────
        self._lbl_state.setText(output.state_display)
        self._lbl_state.setStyleSheet(output.state_style)
        self._lbl_cooldown.setText(output.cooldown_text)
        if output.cooldown_style:
            self._lbl_cooldown.setStyleSheet(output.cooldown_style)
        self._lbl_audio_offset.setText(output.audio_delta_display)
        if output.audio_delta_style:
            self._lbl_audio_offset.setStyleSheet(output.audio_delta_style)

        # 舵机位置显示
        self._lbl_servo_h.setText(f"H: {output.servo_h_current:.1f}°")
        if self.pwm_v and self.pwm_v.initialized:
            self._lbl_servo_v.setText(f"V: {output.servo_v_current:.1f}°")

        # 日志
        if output.log_message:
            self._log(output.log_message)

        # ── 指标采集 ──────────────────────────────────────────────────────
        trigger_move_happened = (output.servo_h_target is not None)
        if trigger_move_happened:
            self._metrics.on_trigger_move(
                output.servo_h_target, now,
                "trigger"  # reason is embedded in log_message
            )

        self._metrics.on_tracking_tick(
            self._latest_dev_x, self._latest_dev_y,
            output.servo_h_current, output.servo_v_current,
            output.moved, output.adjustment_h, output.adjustment_v,
            now, output.in_cooldown,
        )

        # 稳定检测
        if (not output.moved and not output.in_cooldown
                and self._latest_dev_x is not None
                and abs(self._latest_dev_x) < self._deadzone
                and not self._metrics._stable_achieved):
            self._metrics.on_stable_achieved(now)

        # ── 更新 API 状态 + 处理来自 Streamlit 的命令 ─────────────────────
        self._update_api_state(output, now)
        self._process_api_commands()

        # ── Agent Worker: 更新状态 + 时间驱动触发检查 ──────────────────────
        self._agent_state.update_silence(
            is_speech=bool(self._silero_is_speech),
            now=now,
        )
        self._agent_state.update_meeting_state(
            meeting_active=(self._meeting_id is not None),
            meeting_id=self._meeting_id,
            tracking_active=self._tracking_active and not self._tracking_paused,
        )
        if self._latest_identity is not None:
            self._agent_state.update_current_speaker(
                tag_id=self._latest_identity.tag_id,
                name=self._latest_identity.name,
                state=self._latest_identity.state,
                now=now,
            )
        self._agent_worker.tick()

        # ── 身份识别 — 在 TRACKING 状态下解析 AprilTag ────────────────────
        if (self._speaker_identifier is not None
                and output.state == FusionEngine.STATE_TRACKING):
            if not self._speaker_locked:
                self._latest_identity = self._speaker_identifier.tick(
                    now=now,
                    tags=self._latest_tags,
                    person_box=self._latest_person_box,
                    frame_width=1920, frame_height=1080,
                )
            # 锁定状态下: 保持 _latest_identity 不变，不调用 tick()

        # ── 更新 RTSP 推流名片叠加数据（原子引用赋值，GLib 线程安全读取）───
        self._stream_overlay_state = self._engine.state_name
        if self._latest_identity is not None and self._latest_identity.is_confirmed:
            self._stream_overlay_info = {
                "name": self._latest_identity.name
                        or f"Tag {self._latest_identity.tag_id}",
                "role": self._latest_identity.role or "",
                "duration": self._latest_identity.duration,
            }
        else:
            self._stream_overlay_info = None

    # ── API 状态更新 ──────────────────────────────────────────────────────────

    def _build_participants_list(self):
        """构建参会人员列表，包含发言统计和检测状态。

        可从任何 _update_api_state_* 方法调用（tracking 活跃/暂停/空闲均可）。
        每 2 秒刷新一次缓存，减少 100ms tick 中的 DB 访问频率。
        Returns:
            (participants_list, current_speaker) 元组
        """
        # ── 缓存: 2 秒内不重复查询 DB ─────────────────────────────────
        now = time.time()
        if now - self._participants_cache_ts < 2.0:
            return self._cached_participants

        # ── 当前发言人 ────────────────────────────────────────────────
        current_speaker = None
        if self._latest_identity is not None and self._latest_identity.is_confirmed:
            current_speaker = {
                "tag_id": self._latest_identity.tag_id,
                "name": self._latest_identity.name or f"Tag {self._latest_identity.tag_id}",
                "role": self._latest_identity.role or "",
                "organization": self._latest_identity.organization or "",
                "source": self._latest_identity.source,
                "confidence": self._latest_identity.confidence,
                "speaking_duration": self._latest_identity.duration,
                "state": self._latest_identity.state,
            }

        # ── 参会人统计 ────────────────────────────────────────────────
        participants_list = []
        try:
            from storage.db import session_scope
            from storage.repo import ParticipantRepo, SpeakerSegmentRepo

            with session_scope() as session:
                pr = ParticipantRepo(session)
                all_participants = pr.list_all()

                sr = SpeakerSegmentRepo(session)
                meeting_id = self._meeting_id

                # 检测到的 tag_id 集合（来自 SpeakerIdentifier 的稳定性追踪）
                detected_tags = set()
                if self._speaker_identifier is not None:
                    for tag_int, count in self._speaker_identifier._tag_stability.items():
                        if count > 0:
                            try:
                                detected_tags.add(int_to_tag_id(tag_int))
                            except Exception:
                                pass

                for p in all_participants:
                    is_current = (
                        current_speaker is not None
                        and p.tag_id == current_speaker.get("tag_id")
                    )
                    acc_duration = 0.0
                    speaking_count = 0
                    if meeting_id is not None:
                        acc_duration = sr.get_total_duration(meeting_id, speaker_tag_id=p.tag_id)
                        segs = session.query(SpeakerSegment).filter_by(
                            meeting_id=meeting_id, speaker_tag_id=p.tag_id
                        ).all()
                        speaking_count = len(segs)

                    participants_list.append({
                        "tag_id": p.tag_id,
                        "name": p.name,
                        "role": p.role,
                        "organization": p.organization or "",
                        "detected": p.tag_id in detected_tags,
                        "is_current_speaker": is_current,
                        "accumulated_duration": acc_duration,
                        "speaking_count": speaking_count,
                    })
        except Exception:
            # DB 读取失败不阻塞状态更新
            pass

        self._cached_participants = (participants_list, current_speaker)
        self._participants_cache_ts = time.time()
        return participants_list, current_speaker

    def _update_api_state(self, output: EngineOutput, now: float):
        """构建完整状态快照并推送到 ControlApiServer。

        由 _tracking_tick() 每 100ms 调用一次。
        """
        # ── 当前发言人 + 参会人统计 ─────────────────────────────────────────
        participants_list, current_speaker = self._build_participants_list()

        # ── RTSP 状态 ─────────────────────────────────────────────────────
        rtsp_status = "OK" if self._streaming_enabled else "stopped"
        rtsp_url = ""
        if self._streaming_enabled:
            rtsp_url = f"rtsp://{self._device_ip}:8554/smartcam"

        # ── 当前会议信息 ───────────────────────────────────────────────────
        meeting_id = self._meeting_service.get_active_meeting_id() or self._meeting_id
        meeting_state = "no_meeting"
        meeting_name = ""
        if meeting_id is not None:
            meeting_state = "in_progress"
            try:
                from storage.db import session_scope
                from storage.repo import MeetingRepo
                with session_scope() as session:
                    m = MeetingRepo(session).get_by_id(meeting_id)
                    if m:
                        meeting_name = m.name
                        meeting_state = m.status
            except Exception:
                pass

        # ── 组装状态快照 ───────────────────────────────────────────────────
        state = {
            "runtime_state": self._engine.state_name,
            "meeting_state": meeting_state,
            "meeting_id": meeting_id,
            "meeting_name": meeting_name,
            "current_speaker": current_speaker,
            "tracking": {
                "tracking_active": self._tracking_active,
                "tracking_paused": self._tracking_paused,
                "speaker_locked": self._speaker_locked,
                "vad_enabled": self.VAD_ENABLED and self._vad_enabled_effective,
                "vad_is_speech": self._silero_is_speech,
                "vad_device": self.VAD_DEVICE,
                "doa_angle": self._latest_doa,
                "pan_angle": self.pwm_h.get_angle(),
                "tilt_angle": self.pwm_v.get_angle() if self.pwm_v else 0.0,
                "yolo_fps": self._fps,
                "rtsp_status": rtsp_status,
                "rtsp_url": rtsp_url,
            },
            "overlay": {
                "enabled": self._overlay_enabled,
                "show_debug": self._show_debug_overlay,
            },
            "participants": participants_list,
            "timestamp": now,
            "agent": {
                "enabled": self._rule_engine.enabled,
                "muted": self._tts_router.is_muted(),
                "llm_calls_this_meeting": self._agent_state.llm_calls_this_meeting,
                "suppressed_count": self._tts_router.get_suppressed_count(),
                "spoken_count": self._tts_router.get_spoken_count(),
                "last_tts_at": self._tts_router.get_last_spoken_at(),
                "pending_count": self._tts_router.get_pending_count(),
            },
        }

        self._api_server.update_state(state)

    def _update_api_state_tick(self):
        """暂停状态下的轻量状态更新。"""
        participants_list, _current_speaker = self._build_participants_list()
        state = {
            "runtime_state": self._engine.state_name,
            "meeting_state": "in_progress" if self._meeting_id else "no_meeting",
            "meeting_id": self._meeting_id,
            "meeting_name": "",
            "current_speaker": None,
            "tracking": {
                "tracking_active": self._tracking_active,
                "tracking_paused": True,
                "speaker_locked": self._speaker_locked,
                "vad_enabled": False,
                "vad_is_speech": False,
                "vad_device": self.VAD_DEVICE,
                "doa_angle": self._latest_doa,
                "pan_angle": self.pwm_h.get_angle(),
                "tilt_angle": self.pwm_v.get_angle() if self.pwm_v else 0.0,
                "yolo_fps": 0.0,
                "rtsp_status": "OK" if self._streaming_enabled else "stopped",
                "rtsp_url": f"rtsp://{self._device_ip}:8554/smartcam" if self._streaming_enabled else "",
            },
            "overlay": {
                "enabled": self._overlay_enabled,
                "show_debug": self._show_debug_overlay,
            },
            "participants": participants_list,
            "timestamp": time.time(),
            "agent": {
                "enabled": self._rule_engine.enabled,
                "muted": self._tts_router.is_muted(),
                "llm_calls_this_meeting": self._agent_state.llm_calls_this_meeting,
                "suppressed_count": self._tts_router.get_suppressed_count(),
                "spoken_count": self._tts_router.get_spoken_count(),
                "last_tts_at": self._tts_router.get_last_spoken_at(),
                "pending_count": self._tts_router.get_pending_count(),
            },
        }
        self._api_server.update_state(state)

    def _update_api_state_idle(self):
        """tracking 未启动时的轻量状态更新。

        与 _update_api_state_tick 的区别: tracking_paused 正确地反映
        self._tracking_paused（而非硬编码为 True）。
        """
        participants_list, _current_speaker = self._build_participants_list()
        rtsp_status = "OK" if self._streaming_enabled else "stopped"
        state = {
            "runtime_state": self._engine.state_name,
            "meeting_state": "in_progress" if self._meeting_id else "no_meeting",
            "meeting_id": self._meeting_id,
            "meeting_name": "",
            "current_speaker": None,
            "tracking": {
                "tracking_active": self._tracking_active,
                "tracking_paused": self._tracking_paused,
                "speaker_locked": self._speaker_locked,
                "vad_enabled": False,
                "vad_is_speech": False,
                "vad_device": self.VAD_DEVICE,
                "doa_angle": self._latest_doa,
                "pan_angle": self.pwm_h.get_angle(),
                "tilt_angle": self.pwm_v.get_angle() if self.pwm_v else 0.0,
                "yolo_fps": 0.0,
                "rtsp_status": rtsp_status,
                "rtsp_url": f"rtsp://{self._device_ip}:8554/smartcam" if self._streaming_enabled else "",
            },
            "overlay": {
                "enabled": self._overlay_enabled,
                "show_debug": self._show_debug_overlay,
            },
            "participants": participants_list,
            "timestamp": time.time(),
            "agent": {
                "enabled": self._rule_engine.enabled,
                "muted": self._tts_router.is_muted(),
                "llm_calls_this_meeting": self._agent_state.llm_calls_this_meeting,
                "suppressed_count": self._tts_router.get_suppressed_count(),
                "spoken_count": self._tts_router.get_spoken_count(),
                "last_tts_at": self._tts_router.get_last_spoken_at(),
                "pending_count": self._tts_router.get_pending_count(),
            },
        }
        self._api_server.update_state(state)

    # ── API 命令处理 ──────────────────────────────────────────────────────────

    def _process_api_commands(self):
        """排空并处理所有来自 Streamlit 的命令。

        由 _tracking_tick() 每 100ms 调用一次。
        """
        commands = self._api_server.poll_commands()
        for cmd in commands:
            cmd_type = cmd.get("type", "")
            params = cmd.get("params", {})

            try:
                if cmd_type == "meeting_start":
                    self._handle_cmd_meeting_start(params)
                elif cmd_type == "meeting_end":
                    self._handle_cmd_meeting_end(params)
                elif cmd_type == "meeting_pause":
                    self._handle_cmd_meeting_pause(params)
                elif cmd_type == "meeting_resume":
                    self._handle_cmd_meeting_resume(params)
                elif cmd_type == "recenter":
                    self._handle_cmd_recenter(params)
                elif cmd_type == "start_tracking":
                    self._handle_cmd_start_tracking(params)
                elif cmd_type == "stop_tracking":
                    self._handle_cmd_stop_tracking(params)
                elif cmd_type == "lock_speaker":
                    self._handle_cmd_lock_speaker(params)
                elif cmd_type == "unlock_speaker":
                    self._handle_cmd_unlock_speaker(params)
                elif cmd_type == "manual_speaker":
                    self._handle_cmd_manual_speaker(params)
                elif cmd_type == "set_overlay":
                    self._handle_cmd_set_overlay(params)
                elif cmd_type == "start_stream":
                    self._handle_cmd_start_streaming(params)
                elif cmd_type == "stop_stream":
                    self._handle_cmd_stop_streaming(params)
                elif cmd_type == "set_vad_device":
                    self._handle_cmd_set_vad_device(params)
                elif cmd_type == "tts_test":
                    self._handle_cmd_tts_test(params)
                elif cmd_type == "agent_trigger":
                    self._handle_cmd_agent_trigger(params)
                elif cmd_type == "agent_agenda":
                    self._handle_cmd_agent_agenda(params)
                elif cmd_type == "agent_status":
                    self._handle_cmd_agent_status(params)
                elif cmd_type == "agent_custom_tts":
                    self._handle_cmd_agent_custom_tts(params)
                elif cmd_type == "agent_config":
                    self._handle_cmd_agent_config(params)
                else:
                    self._log(f"⚠️ 未知命令: {cmd_type}")
            except Exception as e:
                self._log(f"❌ 命令处理异常 ({cmd_type}): {e}")

    def _handle_cmd_meeting_start(self, params: dict):
        meeting_id = params.get("meeting_id")
        if meeting_id is None:
            return
        meeting = self._meeting_service.start_meeting_by_id(meeting_id)
        self._meeting_id = meeting.id
        self._log(f"📋 会议已开始: [{meeting.id}] {meeting.name}")

        # 重置 Agent 状态（新会议开始）
        self._agent_state.reset()
        self._tts_router.reset_meeting_state()

        # 自动启动追踪（如果尚未启动）
        if not self._tracking_active:
            self._start_tracking()

    def _handle_cmd_meeting_end(self, params: dict):
        meeting = self._meeting_service.end_active_meeting()
        if meeting:
            self._log(f"🏁 会议已结束: [{meeting.id}] {meeting.name}")

        # 停止追踪
        if self._tracking_active:
            self._stop_tracking()

        self._meeting_id = None
        # 重置 Agent 状态
        self._agent_state.reset()
        self._tts_router.reset_meeting_state()

    def _handle_cmd_meeting_pause(self, params: dict):
        self._tracking_paused = True
        self._meeting_service.pause_tracking()
        self._log("⏸ 追踪已暂停")

    def _handle_cmd_meeting_resume(self, params: dict):
        self._tracking_paused = False
        self._meeting_service.resume_tracking()
        self._log("▶ 追踪已恢复")

    def _handle_cmd_recenter(self, params: dict):
        self._on_center()
        self._log("⬅ 云台回中 (API)")

    def _handle_cmd_start_tracking(self, params: dict):
        if not self._tracking_active:
            self._start_tracking()
            self._log("▶ 追踪已启动 (API)")

    def _handle_cmd_stop_tracking(self, params: dict):
        if self._tracking_active:
            self._stop_tracking()
            self._log("⏹ 追踪已停止 (API)")

    def _handle_cmd_lock_speaker(self, params: dict):
        self._speaker_locked = True
        EventBus().publish("host_locked_speaker",
                           tag_id=self._latest_identity.tag_id if self._latest_identity else None,
                           name=self._latest_identity.name if self._latest_identity else None)
        self._log("🔒 发言人已锁定")

    def _handle_cmd_unlock_speaker(self, params: dict):
        self._speaker_locked = False
        EventBus().publish("host_unlocked_speaker")
        self._log("🔓 发言人已解锁")

    def _handle_cmd_manual_speaker(self, params: dict):
        tag_id = params.get("tag_id", "")
        if not tag_id:
            return
        if self._speaker_identifier is not None:
            self._speaker_identifier.manual_override(tag_id)
            self._speaker_locked = False  # 手动覆盖时自动解锁
            # 立即查询人员信息更新身份
            p = self._lookup_participant(tag_id)
            if p:
                self._log(f"👤 手动指定发言人: {p['name']} ({tag_id})")
        EventBus().publish("speaker_override", tag_id=tag_id)

    def _handle_cmd_set_overlay(self, params: dict):
        if "enabled" in params:
            self._overlay_enabled = params["enabled"]
            self._log(f"📺 名片叠加: {'显示' if self._overlay_enabled else '隐藏'}")
        if "show_debug" in params:
            self._show_debug_overlay = params["show_debug"]
            self._log(f"🐛 调试框: {'显示' if self._show_debug_overlay else '隐藏'}")

    def _handle_cmd_start_streaming(self, params: dict):
        if not self._streaming_enabled:
            self._start_streaming()
            self._log("📡 推流已启动 (API)")

    def _handle_cmd_stop_streaming(self, params: dict):
        if self._streaming_enabled:
            self._stop_streaming()
            self._log("📡 推流已停止 (API)")

    def _handle_cmd_set_vad_device(self, params: dict):
        """处理来自 Streamlit 的 VAD 设备切换命令。"""
        device = params.get("device", "")
        if not device:
            return
        # 验证设备存在于下拉列表中
        idx = self._cmb_vad_device.findData(device)
        if idx < 0:
            self._log(f"⚠️ 未知音频设备: {device}，已忽略")
            return
        # 更新下拉框（阻断信号避免递归重启）
        self._cmb_vad_device.blockSignals(True)
        self._cmb_vad_device.setCurrentIndex(idx)
        self._cmb_vad_device.blockSignals(False)
        self.VAD_DEVICE = device
        self._log(f"🔊 VAD 设备已切换 (API): {device}")
        if self._tracking_active and not self._tracking_paused:
            self._start_vad()
        self._save_params()

    def _handle_cmd_tts_test(self, params: dict):
        """处理 TTS 测试播报命令 — 来自 Streamlit 测试按钮或 Qt UI 按钮。"""
        text = params.get("text", "这是测试语音")
        self._log(f"🧪 TTS 测试: \"{text}\"")
        req = TTSRequest(
            text=text, source="host_manual",
            priority=TTSPriority.HOST_MANUAL,
            meeting_id=self._meeting_id,
            cooldown_key=f"tts_test:{time.time()}",
            reason="手动测试播报",
        )
        self._tts_router.say(req)

    def _handle_cmd_agent_trigger(self, params: dict):
        """处理 Agent 手动触发命令 — 阶段总结。"""
        meeting_id = params.get("meeting_id", self._meeting_id)
        minutes = params.get("minutes", 3)
        self._agent_worker.request_summary(meeting_id=meeting_id, minutes=minutes)

    def _handle_cmd_agent_agenda(self, params: dict):
        """处理 Agent 议题提醒命令。"""
        meeting_id = params.get("meeting_id", self._meeting_id)
        self._agent_worker.request_agenda_reminder(meeting_id=meeting_id)

    def _handle_cmd_agent_status(self, params: dict):
        """处理 Agent 系统状态播报命令。"""
        meeting_id = params.get("meeting_id", self._meeting_id)
        self._agent_worker.request_status_broadcast(meeting_id=meeting_id)

    def _handle_cmd_agent_custom_tts(self, params: dict):
        """处理 Agent 自定义文本播报命令。"""
        text = params.get("text", "")
        if not text:
            self._log("⚠️ agent_custom_tts: 缺少 text 参数")
            return
        meeting_id = params.get("meeting_id", self._meeting_id)
        self._agent_worker.request_custom_tts(text=text, meeting_id=meeting_id)

    def _handle_cmd_agent_config(self, params: dict):
        """处理 Agent 配置更新命令 — 持久化到 SystemConfig DB。"""
        section = params.get("section", "agent")
        key = params.get("key", "")
        value = params.get("value")
        if not key:
            self._log("⚠️ agent_config: 缺少 key 参数")
            return
        try:
            from storage.db import session_scope
            from storage.repo import ConfigRepo
            with session_scope() as session:
                ConfigRepo(session).set(section, key, value)
            self._log(f"📝 Agent 配置已更新: [{section}] {key} = {value}")
        except Exception as e:
            self._log(f"⚠️ Agent 配置写入失败: {e}")

    # ── TTS Router 回调 (DB 审计) ──────────────────────────────────────────────

    def _on_tts_spoken(self, text: str, source: str, reason: str):
        """TTS 播报完成 → 写入 TTSEvent 到 DB。"""
        try:
            from storage.db import session_scope
            from storage.repo import TTSEventRepo
            with session_scope() as session:
                TTSEventRepo(session).log(
                    meeting_id=self._meeting_id,
                    text=text, source=source,
                    status="spoken", reason=reason,
                )
        except Exception as e:
            self._log(f"⚠️ TTS 审计写入失败: {e}")

    def _on_tts_suppressed(self, request, reason_str: str):
        """TTS 播报被抑制 → 写入 AgentDecision 到 DB。"""
        try:
            from storage.db import session_scope
            from storage.repo import AgentDecisionRepo
            with session_scope() as session:
                AgentDecisionRepo(session).log_decision(
                    meeting_id=request.meeting_id,
                    trigger_type="tts_suppressed",
                    trigger_key=request.cooldown_key,
                    priority=request.priority,
                    rule_reason=request.reason,
                    decision="suppressed",
                    final_text=request.text,
                    suppressed_reason=reason_str,
                )
        except Exception as e:
            self._log(f"⚠️ 决策审计写入失败: {e}")

    def _on_agent_decision(self, candidate, spoken: bool,
                           suppressed_reason: str | None,
                           final_text: str | None,
                           llm_used: bool = False,
                           llm_prompt_tokens: int = 0,
                           llm_completion_tokens: int = 0):
        """Agent 决策回调 → 写入 AgentDecision 到 DB。"""
        try:
            from storage.db import session_scope
            from storage.repo import AgentDecisionRepo
            with session_scope() as session:
                AgentDecisionRepo(session).log_decision(
                    meeting_id=candidate.meeting_id,
                    trigger_type=candidate.trigger_type,
                    trigger_key=candidate.trigger_key,
                    priority=candidate.priority,
                    rule_reason=candidate.reason,
                    llm_used=1 if llm_used else 0,
                    llm_prompt_tokens=llm_prompt_tokens,
                    llm_completion_tokens=llm_completion_tokens,
                    decision="spoken" if spoken else "suppressed",
                    final_text=final_text,
                    suppressed_reason=suppressed_reason,
                )
        except Exception as e:
            self._log(f"⚠️ Agent 决策审计写入失败: {e}")

    # ── 日志 ────────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        log_text = getattr(self, "_log_text", None)
        if log_text is not None:
            log_text.append(f"[{ts}] {msg}")
        else:
            print(f"[{ts}] {msg}")

    def _safe_cleanup(self, label: str, fn, *args, **kwargs):
        """执行清理操作，捕获异常以避免阻塞后续清理步骤。

        在 closeEvent / _stop_tracking 中使用：单点异常不应阻止
        释放其他硬件资源（摄像头、PWM、DB 连接等）。
        """
        try:
            fn(*args, **kwargs)
        except Exception as e:
            try:
                self._log(f"⚠️ 清理失败 ({label}): {e}")
            except Exception:
                print(f"[Cleanup] {label}: {e}")

    # ── 窗口关闭 ────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # ── 取消延迟初始化 + 启动动画 ──────────────────────────────────────
        if hasattr(self, '_deferred_steps'):
            self._deferred_steps.clear()
        if getattr(self, '_splash', None) is not None:
            self._splash.hide()
            self._splash = None

        # ── 逐步清理：每步 try/except，避免单点异常阻塞后续清理 ────────────
        # 顺序: 追踪 → Agent → 定时器 → TTS/LLM workers → 推流 → 播报器
        #       → 播放器 → API → 事件桥 → 参数 → 摄像头 → 麦克风 → VAD
        self._safe_cleanup("tracking",      self._stop_tracking)

        if hasattr(self, '_agent_worker') and self._agent_worker is not None:
            self._safe_cleanup("agent_worker", self._agent_worker.stop)

        if hasattr(self, '_track_timer') and self._track_timer is not None:
            self._safe_cleanup("track_timer", self._track_timer.stop)

        # ── 清理 TTS Engine 的 synthesis workers ─────────────────────────
        if hasattr(self, '_tts_engine') and self._tts_engine is not None:
            try:
                for w in list(getattr(self._tts_engine, '_workers', [])):
                    if w.isRunning():
                        w.wait(2000)
                self._tts_engine._workers.clear()
            except Exception as e:
                print(f"[Cleanup] TTS workers: {e}")

        # ── 清理 AgentWorker 的 LLM workers ──────────────────────────────
        if hasattr(self, '_agent_worker') and self._agent_worker is not None:
            try:
                for w in list(getattr(self._agent_worker, '_workers', [])):
                    if w.isRunning():
                        w.wait(2000)
                self._agent_worker._workers.clear()
            except Exception as e:
                print(f"[Cleanup] AgentWorker LLM workers: {e}")

        self._safe_cleanup("streaming",     self._stop_streaming)
        self._safe_cleanup("announcer",     self._announcer.stop)
        self._safe_cleanup("player",        self._player.stop)
        self._safe_cleanup("api_server",    self._api_server.stop)
        self._safe_cleanup("event_bridge",  self._event_bridge.stop)
        self._safe_cleanup("save_params",   self._save_params)

        if self._camera_thread.isRunning():
            self._safe_cleanup("camera",    self._camera_thread.stop)
        if self._reader.isRunning():
            self._safe_cleanup("reader",    self._reader.stop)
        self._safe_cleanup("vad",           self._stop_vad)

        # ── 导出指标 CSV ───────────────────────────────────────────────────
        self._metrics.set_params({
            "threshold_audio":     self.THRESHOLD_AUDIO,
            "await_duration":      self.AWAIT_DURATION,
            "audio_jump_thresh":   self.AUDIO_JUMP_THRESH,
            "jump_cooldown":       self.JUMP_COOLDOWN,
            "motor_cooldown":      self.MOTOR_COOLDOWN,
            "visual_lost_frames":  self.VISUAL_LOST_FRAMES,
            "cooldown":            self._cooldown,
            "deadzone":            self._deadzone,
            "gain_h":              self._gain_h,
            "gain_v":              self._gain_v,
            "max_angle_v":         self._max_angle_v,
            "vertical_bias":       self._vertical_bias,
            "vad_enabled":         self.VAD_ENABLED,
            "vad_threshold":       self.VAD_THRESHOLD,
            "vad_pregain":         self.VAD_PREGAIN,
        })
        self._metrics.export_csvs(PROJECT_DIR)
        self._log(f"📊 指标已导出到 {PROJECT_DIR}/fusion_metrics_*.csv")

        event.accept()


# ═════════════════════════════════════════════════════════════════════════════════
# main()
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    # ── 初始化数据库 + 运行迁移 ──────────────────────────────────────────────
    try:
        storage_init()
    except Exception as e:
        print(f"[Storage] 数据库初始化失败: {e}")

    # ── PWM 初始化 ──────────────────────────────────────────────────────────
    pwm_h = PWMController(pwmchip=0, pwm_index="0",
                          angle_min=-135, angle_max=135, label="PWM-H")
    h_ok = pwm_h.init()
    if not h_ok:
        print("⚠️ 水平 PWM 初始化失败")

    pwm_v = PWMController(pwmchip=1, pwm_index="0",
                          angle_min=-90, angle_max=90,
                          duty_at_min=PWMController.PCT_5,
                          duty_at_max=PWMController.PCT_25,
                          label="PWM-V")
    v_ok = pwm_v.init()
    if not v_ok:
        print("⚠️ 垂直 PWM 初始化失败，仅水平追踪可用")

    atexit.register(pwm_h.cleanup)
    atexit.register(pwm_v.cleanup)

    # ── 加载标定 ────────────────────────────────────────────────────────────
    storage = CalibrationStorage()
    points, slope, intercept, r2, fitted, timestamp = storage.load()
    model = CalibrationModel()
    if points:
        model.points = points
    if fitted:
        model.slope = slope
        model.intercept = intercept
        model.r_squared = r2
        model.fitted = fitted
    elif len(points) >= 2:
        model.fit_linear()

    calib_status = "✅" if model.fitted else ("⚠️ 有数据未拟合" if model.points else "❌ 未标定")

    # ── PyQt5 应用 ───────────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = FusionTrackerWindow(pwm_h, pwm_v if v_ok else None, model)
    window.resize(1024, 800)
    window.show()

    # ── SIGTERM / SIGINT → 触发 closeEvent 实现干净退出 ────────────────────
    # systemd stop / kill 发送 SIGTERM 后，Python 默认的 SystemExit 无法
    # 穿透 Qt C++ 事件循环 (app.exec_())，导致进程忽略信号直到 systemd
    # 超时 (90s) 后 SIGKILL 强杀。此处理器调用 window.close() 显式触发
    # closeEvent → 释放摄像头/PWM/DB/CSV。
    import signal as _signal_module
    def _signal_handler(signum, frame):
        try:
            sig_name = _signal_module.Signals(signum).name
        except (AttributeError, ValueError):
            sig_name = str(signum)
        print(f"\n[Signal] 收到信号 {sig_name}，正在退出...")
        window.close()

    _signal_module.signal(_signal_module.SIGTERM, _signal_handler)
    _signal_module.signal(_signal_module.SIGINT, _signal_handler)

    print("=" * 50)
    print("视听融合追踪 Demo 已启动 (core/ 模块重构版)")
    print(f"  PWM-H:   {'✅' if h_ok else '❌'} pwmchip0 (-135°~+135°)")
    print(f"  PWM-V:   {'✅' if v_ok else '❌'} pwmchip1 (-90°~+90°)")
    print(f"  标定:    {calib_status} ({len(model.points)} 点, R²={model.r_squared:.4f})")
    print("  状态机:  IDLE → AWAIT → TRACKING → IDLE (FusionEngine)")
    print("  关闭窗口或 Ctrl+C 退出")
    print("=" * 50)

    try:
        sys.exit(app.exec_())
    finally:
        pwm_h.cleanup()
        pwm_v.cleanup()


if __name__ == "__main__":
    main()
