#!/usr/bin/env python3
"""Silero VAD 调试 Demo — 独立的 PyQt5 GUI，用于测试和调试语音活动检测。

功能:
  - 实时音频波形显示（pyqtgraph）
  - 语音概率历史曲线（pyqtgraph）
  - 音频电平表（VU Meter）
  - 语音/静音状态指示
  - 参数调节：设备选择、阈值、前置增益
  - 时间戳日志输出

用法:
  python3 demos/silero_vad_debug.py
  (如需 sudo，自动通过 fix_display_env 修复 X11 环境)

依赖:
  - PyQt5, pyqtgraph, numpy, torch
  - core/silero_vad, core/alsa_capture, core/alsa_device_list, core/display_env
"""

import os
import sys
import time
from collections import deque

# ── 确保能找到项目根目录的 core/ 模块 ────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 显示环境修复（必须在任何 Qt import 之前调用）──────────────────────────
from core.display_env import fix_display_env
fix_display_env()

os.environ.setdefault("DISPLAY", ":0.0")

# ── 注意：本 Demo 不使用 cv2，无需 fix_cv2_qt_conflict ──────────────────

import ctypes
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QTextEdit, QPushButton, QComboBox,
    QDoubleSpinBox, QProgressBar, QStatusBar, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont

import pyqtgraph as pg

from core.alsa_capture import AlsaAudioCapture
from core.alsa_device_list import list_capture_devices
from core.silero_vad import SileroVAD

# ═════════════════════════════════════════════════════════════════════════════════
# 常量
# ═════════════════════════════════════════════════════════════════════════════════

WAVEFORM_SECONDS = 3        # 波形窗口长度（秒）
PROB_HISTORY = 300          # 概率历史长度（chunk 数，~9.6s @ 32ms/chunk）
DISPLAY_TIMER_MS = 50       # 显示刷新间隔（ms）


# ═════════════════════════════════════════════════════════════════════════════════
# VadDebugThread — 后台音频采集 + Silero VAD
# ═════════════════════════════════════════════════════════════════════════════════

class VadDebugThread(QThread):
    """后台线程：ALSA 音频采集 → pregain → Silero VAD 推理。

    通过 PyQt5 信号将音频数据、VAD 结果、电平等信息发送到主线程。
    无自动恢复机制 — 出错即停止并通知主线程（调试工具，用户应可见错误）。
    """

    # ── 信号定义 ──────────────────────────────────────────────────────
    audio_chunk    = pyqtSignal(np.ndarray)       # 原始音频块 (512, float32)
    vad_probability = pyqtSignal(float, bool, float)  # (prob, is_speech, duration)
    audio_level    = pyqtSignal(float)             # RMS 电平 (dBFS)
    status_msg     = pyqtSignal(str)               # 状态/日志消息
    device_info    = pyqtSignal(str)               # 设备参数信息
    fatal_error    = pyqtSignal(str)               # 致命错误（线程即将退出）

    def __init__(self, device: str, threshold: float, pregain: float):
        super().__init__()
        self._device = device
        self._threshold = threshold
        self._pregain = pregain
        self._running = False
        self._capture: AlsaAudioCapture | None = None
        self._vad: SileroVAD | None = None

        # 线程安全的公开状态（主线程只读）
        self.is_speech = False
        self.speech_prob = 0.0
        self.speech_duration = 0.0

    def stop(self):
        """停止线程并等待退出（最多 3 秒）。"""
        self._running = False
        self.wait(3000)

    def _ensure_pcm_started(self):
        """显式启动 ALSA PCM 捕获流。

        部分 ALSA 驱动（尤其是 USB 音频设备）不会在首次
        snd_pcm_readi 时自动启动流，导致 read() 永久阻塞。
        此处通过 ctypes 直接调用 snd_pcm_start 确保流已启动。
        """
        if not self._capture or not self._capture.is_open:
            return
        try:
            alsa = self._capture._alsa
            pcm = self._capture._pcm_handle
            if alsa is None or pcm is None:
                return

            # 声明 snd_pcm_start 签名（AlsaAudioCapture 未声明）
            alsa.snd_pcm_start.restype = ctypes.c_int
            alsa.snd_pcm_start.argtypes = [ctypes.c_void_p]

            ret = alsa.snd_pcm_start(pcm)
            if ret < 0:
                err = alsa.snd_strerror(ret)
                err_msg = err.decode() if err else f"code={ret}"
                self.status_msg.emit(f"⚠️ snd_pcm_start 失败: {err_msg}")
            else:
                self.status_msg.emit("PCM 流已显式启动 (snd_pcm_start OK)")
        except Exception as e:
            self.status_msg.emit(f"⚠️ snd_pcm_start 异常: {e}")

    def run(self):
        self._running = True

        # ── 1. 加载 VAD 模型 ──────────────────────────────────────────
        self._vad = SileroVAD(threshold=self._threshold)
        if not self._vad.load():
            self.fatal_error.emit("Silero VAD 模型加载失败")
            return
        self.status_msg.emit("VAD 模型已加载")

        # ── 2. 打开 ALSA 录音设备 ─────────────────────────────────────
        self._capture = AlsaAudioCapture(
            device=self._device, rate=SileroVAD.SAMPLE_RATE,
            channels=1, chunk_size=SileroVAD.CHUNK_SIZE)
        if not self._capture.open():
            self.fatal_error.emit(f"无法打开 ALSA 设备: {self._device}")
            return

        self.device_info.emit(
            f"设备 {self._device} → 实际 {self._capture.actual_rate}Hz "
            f"{self._capture.actual_channels}ch S16_LE")

        # ── 显式启动 PCM 流（部分 ALSA 驱动不会在 readi 时自动启动）───
        self._ensure_pcm_started()

        self.status_msg.emit(
            f"VAD 就绪 — thresh={self._threshold:.2f} "
            f"pregain={self._pregain:.1f}×")

        # ── 3. 主循环：读音频 → 处理 → 发射信号 ──────────────────────
        read_count = 0
        consecutive_none = 0
        first_chunk_logged = False
        _NONE_LOG_INTERVAL = 31  # 约每秒报告一次连续读取失败

        while self._running:
            chunk = self._capture.read()
            if chunk is None:
                consecutive_none += 1
                # 首次失败：报告具体错误码
                if consecutive_none == 1:
                    self.status_msg.emit(
                        "⚠️ ALSA read 返回 None — "
                        "可能原因: 设备无数据流 / 驱动不兼容 / 需要 plughw")
                # 持续失败时每 ~1s 报告一次
                elif consecutive_none % _NONE_LOG_INTERVAL == 0:
                    self.status_msg.emit(
                        f"⚠️ 连续 {consecutive_none} 次 ALSA read 失败 "
                        f"(已阻塞约 {consecutive_none * 32}ms) — "
                        f"请确认设备支持音频流采集")
                # 长时间失败后给出建议
                if consecutive_none == 93:  # ~3s
                    self.status_msg.emit(
                        "💡 提示: XVF3800 在某些内核版本仅支持 DOA，"
                        "音频采集请尝试 hw:1,0 (NAU8822 板载 codec)")
                self.msleep(10)
                continue

            # 读取成功 — 重置错误计数
            if consecutive_none > 0:
                self.status_msg.emit(
                    f"✅ 读取恢复 — 之前连续 {consecutive_none} 次失败后恢复")
            consecutive_none = 0
            read_count += 1

            # 诊断：首帧输出振幅范围，确认音频数据有效
            if not first_chunk_logged:
                first_chunk_logged = True
                peak = float(np.max(np.abs(chunk)))
                self.status_msg.emit(
                    f"首帧就绪 — 峰峰幅值={peak:.4f} "
                    f"(read_count=1, 数据{'有信号' if peak > 0.001 else '接近静音'})")

            # 心跳日志（约每秒一次 = ~31 chunks @ 32ms）
            if read_count % 31 == 0:
                peak = float(np.max(np.abs(chunk)))
                self.status_msg.emit(
                    f"心跳 — 已读 {read_count} 帧 "
                    f"({read_count * 32}ms), 当前峰峰={peak:.4f}")

            # 应用前置增益
            np.multiply(chunk, self._pregain, out=chunk)
            np.clip(chunk, -1.0, 1.0, out=chunk)

            # 计算 RMS 电平 (dBFS)
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            db = 20.0 * np.log10(max(rms, 1e-10))
            self.audio_level.emit(max(db, -60.0))

            # 发射音频数据供波形显示（必须复制，防止缓冲区复用）
            self.audio_chunk.emit(chunk.copy())

            # VAD 推理
            prob = self._vad.process(chunk)
            self.speech_prob = prob

            was_speech = self.is_speech
            self.is_speech = prob >= self._threshold

            if self.is_speech:
                if was_speech:
                    self.speech_duration += (
                        SileroVAD.CHUNK_SIZE / SileroVAD.SAMPLE_RATE)
                else:
                    self.speech_duration = (
                        SileroVAD.CHUNK_SIZE / SileroVAD.SAMPLE_RATE)
                    self.status_msg.emit(
                        f"🔊 检测到语音 (prob={prob:.3f})")
            else:
                if was_speech and self.speech_duration > 0.1:
                    self.status_msg.emit(
                        f"🔇 语音结束 — 持续 {self.speech_duration:.1f}s "
                        f"(末次 prob={prob:.3f})")
                self.speech_duration = 0.0

            self.vad_probability.emit(prob, self.is_speech, self.speech_duration)

        # ── 4. 清理 ───────────────────────────────────────────────────
        if self._capture:
            self._capture.close()
        self._vad = None
        self.status_msg.emit("线程已退出")


# ═════════════════════════════════════════════════════════════════════════════════
# MainWindow — PyQt5 主窗口
# ═════════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Silero VAD 调试主窗口。"""

    WINDOW_TITLE = "Silero VAD 调试 Demo"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)

        # ── 后台线程引用 ──────────────────────────────────────────────
        self._thread: VadDebugThread | None = None

        # ── 波形环形缓冲区 ────────────────────────────────────────────
        waveform_samples = int(SileroVAD.SAMPLE_RATE * WAVEFORM_SECONDS)
        self._waveform_buf = np.zeros(waveform_samples, dtype=np.float32)
        self._waveform_idx = 0

        # ── 概率历史缓冲区 ────────────────────────────────────────────
        self._prob_buffer = deque(maxlen=PROB_HISTORY)

        # ── 语音状态跟踪（用于仅在状态切换时刷新指示器）────────────────
        self._last_is_speech = False

        # ── 构建界面 ──────────────────────────────────────────────────
        self._build_ui()
        self._populate_devices()

        # ── 显示刷新定时器 ────────────────────────────────────────────
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._update_displays)
        self._display_timer.setInterval(DISPLAY_TIMER_MS)
        # 先不启动，Start 点击后再启动

    # ── UI 构建 ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ── 上半部分：左侧图表 + 右侧控制面板 ────────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # ─── 左侧：图表区域 ───────────────────────────────────────────
        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)

        # 波形图
        waveform_group = QGroupBox("📈 音频波形")
        waveform_layout = QVBoxLayout(waveform_group)
        waveform_layout.setContentsMargins(4, 12, 4, 4)

        self._waveform_plot = pg.PlotWidget()
        self._waveform_plot.setBackground('#1e1e1e')
        self._waveform_plot.getPlotItem().hideAxis('bottom')
        self._waveform_plot.getPlotItem().setLabel('left', '振幅')
        self._waveform_plot.setYRange(-1.1, 1.1)
        self._waveform_plot.setMouseEnabled(x=False, y=True)
        self._waveform_curve = self._waveform_plot.plot(
            pen=pg.mkPen(color='#4ecdc4', width=1))
        # 零线
        self._waveform_plot.addItem(pg.InfiniteLine(
            pos=0, angle=0, pen=pg.mkPen(color='#555', width=1)))
        waveform_layout.addWidget(self._waveform_plot)
        left_panel.addWidget(waveform_group, stretch=2)

        # 概率历史图
        prob_group = QGroupBox("📊 语音概率历史")
        prob_layout = QVBoxLayout(prob_group)
        prob_layout.setContentsMargins(4, 12, 4, 4)

        self._prob_plot = pg.PlotWidget()
        self._prob_plot.setBackground('#1e1e1e')
        self._prob_plot.getPlotItem().setLabel('left', '概率')
        self._prob_plot.getPlotItem().setLabel('bottom', '时间（每格 ≈ 1.6s）')
        self._prob_plot.setYRange(0, 1.05)
        self._prob_plot.setMouseEnabled(x=True, y=False)
        self._prob_curve = self._prob_plot.plot(
            pen=pg.mkPen(color='#2ecc71', width=1))
        # 阈值线
        self._prob_threshold_line = pg.InfiniteLine(
            pos=0.5, angle=0,
            pen=pg.mkPen(color='#f1c40f', width=1, style=Qt.DashLine))
        self._prob_plot.addItem(self._prob_threshold_line)
        prob_layout.addWidget(self._prob_plot)
        left_panel.addWidget(prob_group, stretch=1)

        top_row.addLayout(left_panel, stretch=3)

        # ─── 右侧：状态 + 控制 ───────────────────────────────────────
        right_panel = QVBoxLayout()
        right_panel.setSpacing(8)

        # 状态面板
        status_group = QGroupBox("🔍 状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)

        # 语音指示器
        self._speech_label = QLabel("SILENCE")
        self._speech_label.setAlignment(Qt.AlignCenter)
        self._speech_label.setFont(QFont("Sans", 28, QFont.Bold))
        self._speech_label.setMinimumHeight(80)
        self._speech_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; color: #555; "
            "border: 2px solid #333; border-radius: 8px; }")
        status_layout.addWidget(self._speech_label)

        # 概率值
        self._prob_label = QLabel("Probability: --")
        self._prob_label.setFont(QFont("monospace", 13))
        self._prob_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self._prob_label)

        # 时长
        self._duration_label = QLabel("Duration: --")
        self._duration_label.setFont(QFont("monospace", 13))
        self._duration_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self._duration_label)

        # 电平表
        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Level:"))
        self._level_meter = QProgressBar()
        self._level_meter.setRange(-60, 0)
        self._level_meter.setValue(-60)
        self._level_meter.setTextVisible(True)
        self._level_meter.setFormat("%v dB")
        self._level_meter.setStyleSheet("""
            QProgressBar {
                background-color: #1e1e1e;
                border: 1px solid #555;
                border-radius: 4px;
                text-align: center;
                color: #ccc;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ecc71, stop:0.5 #f1c40f, stop:0.85 #e74c3c
                );
                border-radius: 3px;
            }
        """)
        level_layout.addWidget(self._level_meter, stretch=1)
        status_layout.addLayout(level_layout)

        right_panel.addWidget(status_group)

        # 控制面板
        control_group = QGroupBox("🎛️ 控制")
        control_layout = QGridLayout(control_group)
        control_layout.setSpacing(6)

        # 设备选择
        control_layout.addWidget(QLabel("录音设备:"), 0, 0)
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(180)
        self._device_combo.setToolTip("选择 ALSA 录音设备")
        control_layout.addWidget(self._device_combo, 0, 1)
        self._refresh_devices_btn = QPushButton("🔄")
        self._refresh_devices_btn.setToolTip("刷新设备列表")
        self._refresh_devices_btn.setFixedWidth(40)
        self._refresh_devices_btn.clicked.connect(self._populate_devices)
        control_layout.addWidget(self._refresh_devices_btn, 0, 2)

        # 阈值
        control_layout.addWidget(QLabel("阈值:"), 1, 0)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.05, 0.95)
        self._threshold_spin.setSingleStep(0.05)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(0.50)
        self._threshold_spin.setToolTip(
            "语音判定概率阈值（0.05=极度敏感, 0.95=极不敏感）")
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)
        control_layout.addWidget(self._threshold_spin, 1, 1, 1, 2)

        # 前置增益
        control_layout.addWidget(QLabel("前置增益:"), 2, 0)
        self._pregain_spin = QDoubleSpinBox()
        self._pregain_spin.setRange(1.0, 100.0)
        self._pregain_spin.setSingleStep(1.0)
        self._pregain_spin.setDecimals(1)
        self._pregain_spin.setValue(30.0)
        self._pregain_spin.setToolTip(
            "音频信号放大倍数（补偿低电平输入如 NAU8822）")
        control_layout.addWidget(self._pregain_spin, 2, 1, 1, 2)

        # Start / Stop 按钮
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("▶ 开始")
        self._start_btn.clicked.connect(self._on_start)
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #2d6a2d; }"
            "QPushButton:hover { background-color: #3a8a3a; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }")
        btn_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #6a2d2d; }"
            "QPushButton:hover { background-color: #8a3a3a; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }")
        btn_layout.addWidget(self._stop_btn)

        control_layout.addLayout(btn_layout, 3, 0, 1, 3)

        right_panel.addWidget(control_group)

        # 右侧底部留弹性空间
        right_panel.addStretch()

        top_row.addLayout(right_panel, stretch=2)

        main_layout.addLayout(top_row, stretch=3)

        # ── 日志面板 ──────────────────────────────────────────────────
        log_group = QGroupBox("📋 日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 12, 4, 4)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("monospace", 9))
        self._log_text.setMaximumHeight(160)
        self._log_text.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #aaa; "
            "border: 1px solid #333; border-radius: 4px; }")
        log_layout.addWidget(self._log_text)

        main_layout.addWidget(log_group, stretch=1)

        # ── 状态栏 ────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("就绪 — 请点击「开始」启动 VAD 调试")
        self._status_bar.addWidget(self._status_label)

        # 窗口默认大小
        self.resize(1400, 800)

    # ── 设备枚举 ──────────────────────────────────────────────────────

    def _populate_devices(self):
        """枚举录音设备并填充下拉框。"""
        self._device_combo.clear()
        try:
            devices = list_capture_devices()
        except Exception as e:
            self._log(f"设备枚举失败: {e}")
            return

        if not devices:
            self._log("⚠️ 未找到录音设备")
            return

        default_idx = 0
        for i, dev in enumerate(devices):
            label = f"{dev['name']} [{dev['card_name']}]"
            self._device_combo.addItem(label, dev['name'])
            # 默认选中 hw:1,0（NAU8822）
            if dev['name'] == "hw:1,0":
                default_idx = i

        self._device_combo.setCurrentIndex(default_idx)
        self._log(f"已枚举 {len(devices)} 个录音设备")

    # ── Start / Stop ──────────────────────────────────────────────────

    def _on_start(self):
        """启动 VAD 线程。"""
        # 防止重复启动
        if self._thread is not None and self._thread.isRunning():
            self._log("⚠️ 线程已在运行中")
            return

        device = self._device_combo.currentData()
        if not device:
            QMessageBox.warning(self, "无设备", "请先选择录音设备。")
            return

        threshold = self._threshold_spin.value()
        pregain = self._pregain_spin.value()

        # 禁用控件
        self._set_controls_enabled(False)

        # 清空缓冲区
        self._waveform_buf.fill(0.0)
        self._waveform_idx = 0
        self._prob_buffer.clear()
        self._last_is_speech = False

        # 清除图表
        self._waveform_curve.setData(np.zeros(1))
        self._prob_curve.setData(np.zeros(1))

        # 创建并启动线程
        self._thread = VadDebugThread(device, threshold, pregain)
        self._thread.audio_chunk.connect(self._on_audio_chunk)
        self._thread.vad_probability.connect(self._on_vad_probability)
        self._thread.audio_level.connect(self._on_audio_level)
        self._thread.status_msg.connect(self._on_status)
        self._thread.device_info.connect(self._on_device_info)
        self._thread.fatal_error.connect(self._on_fatal_error)
        self._thread.start()

        # 启动显示刷新
        self._display_timer.start()

        self._status_label.setText("运行中…")

    def _on_stop(self):
        """停止 VAD 线程。"""
        if self._thread is not None:
            if self._thread.isRunning():
                self._thread.stop()
            self._thread = None

        self._display_timer.stop()
        self._set_controls_enabled(True)
        self._update_speech_indicator(False, 0.0)
        self._prob_label.setText("Probability: --")
        self._duration_label.setText("Duration: --")
        self._level_meter.setValue(-60)
        self._status_label.setText("已停止")

    def _set_controls_enabled(self, enabled: bool):
        """批量设置控件启用状态。"""
        self._device_combo.setEnabled(enabled)
        self._refresh_devices_btn.setEnabled(enabled)
        self._threshold_spin.setEnabled(enabled)
        self._pregain_spin.setEnabled(enabled)
        self._start_btn.setEnabled(enabled)
        self._stop_btn.setEnabled(not enabled)

    # ── 阈值变更（更新图表中的阈值线）─────────────────────────────────

    def _on_threshold_changed(self, value: float):
        self._prob_threshold_line.setPos(value)

    # ── 信号槽：audio_chunk ───────────────────────────────────────────

    def _on_audio_chunk(self, chunk: np.ndarray):
        """接收原始音频块，写入环形缓冲区。"""
        n = len(chunk)
        buf_len = len(self._waveform_buf)
        end_idx = self._waveform_idx + n
        if end_idx <= buf_len:
            self._waveform_buf[self._waveform_idx:end_idx] = chunk
        else:
            first_part = buf_len - self._waveform_idx
            self._waveform_buf[self._waveform_idx:] = chunk[:first_part]
            self._waveform_buf[:end_idx - buf_len] = chunk[first_part:]
        self._waveform_idx = end_idx % buf_len

    # ── 信号槽：vad_probability ───────────────────────────────────────

    def _on_vad_probability(self, prob: float, is_speech: bool, duration: float):
        """接收 VAD 推理结果。"""
        self._prob_buffer.append(prob)

        # 仅在语音状态切换时刷新指示器（避免频繁 stylesheet 重绘）
        if is_speech != self._last_is_speech:
            self._update_speech_indicator(is_speech, prob)
            self._last_is_speech = is_speech
        elif is_speech:
            # 语音中可适度更新概率颜色强度
            self._update_speech_indicator(True, prob)

        self._prob_label.setText(f"Probability: {prob:.4f}")
        self._duration_label.setText(
            f"Duration: {duration:.1f}s" if is_speech else "Duration: --")

    # ── 信号槽：audio_level ───────────────────────────────────────────

    def _on_audio_level(self, db: float):
        """接收音频电平（dBFS）。"""
        self._level_meter.setValue(int(round(db)))

    # ── 信号槽：日志 / 状态 ────────────────────────────────────────────

    def _on_status(self, msg: str):
        """记录状态消息。"""
        self._log(msg)

    def _on_device_info(self, msg: str):
        """记录设备参数信息。"""
        self._log(f"📡 {msg}")

    def _on_fatal_error(self, msg: str):
        """处理致命错误 — 记录日志并恢复界面。"""
        self._log(f"❌ 致命错误: {msg}")
        self._on_stop()
        QMessageBox.critical(self, "VAD 错误", msg)

    # ── 显示刷新（定时器驱动）─────────────────────────────────────────

    def _update_displays(self):
        """由 DISPLAY_TIMER 周期调用，刷新波形图和概率图。"""
        # 波形图：展开环形缓冲区
        idx = self._waveform_idx
        buf = self._waveform_buf
        display_data = np.concatenate([buf[idx:], buf[:idx]]) if idx > 0 else buf
        self._waveform_curve.setData(display_data)

        # 概率历史图
        if self._prob_buffer:
            self._prob_curve.setData(list(self._prob_buffer))

    # ── 语音指示器刷新 ────────────────────────────────────────────────

    def _update_speech_indicator(self, is_speech: bool, prob: float):
        """更新语音/静音指示器的样式和文本。"""
        if is_speech:
            intensity = int(60 + prob * 175)  # 60–235
            text = f"🔊 SPEECH ({prob:.2f})"
            self._speech_label.setStyleSheet(
                f"QLabel {{ background-color: rgb(0,{intensity},0); "
                f"color: #fff; border: 2px solid #2ecc71; "
                f"border-radius: 8px; }}")
        else:
            text = "🔇 SILENCE"
            self._speech_label.setStyleSheet(
                "QLabel { background-color: #1a1a1a; color: #555; "
                "border: 2px solid #333; border-radius: 8px; }")
        self._speech_label.setText(text)

    # ── 日志辅助 ──────────────────────────────────────────────────────

    def _log(self, msg: str):
        """追加时间戳日志。"""
        timestamp = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{timestamp}] {msg}")
        # 限制日志行数
        if self._log_text.document().blockCount() > 200:
            cursor = self._log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 30)
            cursor.removeSelectedText()

    # ── 窗口关闭 ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        """关闭窗口时清理线程和定时器。"""
        self._display_timer.stop()
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局暗色样式
    app.setStyleSheet("""
        QMainWindow { background-color: #2b2b2b; }
        QGroupBox {
            font-size: 13px; font-weight: bold;
            color: #ddd; border: 1px solid #555;
            border-radius: 6px; margin-top: 10px; padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px; padding: 0 6px 0 6px;
        }
        QPushButton {
            background-color: #3a3a3a; color: #ddd;
            border: 1px solid #555; border-radius: 4px;
            padding: 6px 14px; font-size: 12px;
        }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:disabled { background-color: #2a2a2a; color: #666; }
        QComboBox {
            background-color: #3a3a3a; color: #ddd;
            border: 1px solid #555; border-radius: 4px;
            padding: 4px 8px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #3a3a3a; color: #ddd;
            selection-background-color: #4a4a4a;
        }
        QDoubleSpinBox {
            background-color: #3a3a3a; color: #ddd;
            border: 1px solid #555; border-radius: 4px;
            padding: 4px 8px;
        }
        QLabel { color: #ddd; }
        QTextEdit {
            background-color: #1e1e1e; color: #aaa;
            border: 1px solid #333; border-radius: 4px;
        }
        QStatusBar {
            background-color: #2b2b2b; color: #aaa;
            border-top: 1px solid #444;
        }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
