#!/usr/bin/env python3
"""Tencent Cloud TTS 测试 Demo — 基于 tencentcloud-speech-sdk-python 的 PyQt5 GUI

功能:
  - 支持 HTTP 流式 TTS（SpeechSynthesizer）和 WebSocket 流式 TTS（FlowingSpeechSynthesizer）
  - 语音参数调节：音色、编码、采样率、语速、音量
  - 异步合成（QThread），不阻塞 UI
  - 合成完成后自动保存 WAV 并播放（pygame / aplay 双引擎）

用法:
  python3 demos/tts_demo.py

前置条件:
  - git clone https://github.com/TencentCloud/tencentcloud-speech-sdk-python tts_sdk/
  - 设置环境变量: TENCENT_SECRET_ID, TENCENT_SECRET_KEY, TENCENT_APPID
    或在 UI 中手动填写

依赖:
  - PyQt5, requests, websocket-client, pygame (可选), aplay (可选)
  - tencentcloud-speech-sdk-python (clone 到 tts_sdk/)
"""

import os
import sys
import time
import struct
import tempfile
import subprocess
from collections import OrderedDict

# ── 确保能找到项目根目录的 core/ 模块 ────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 显示环境修复（必须在任何 Qt import 之前调用）──────────────────────────
from core.display_env import fix_display_env
fix_display_env()

os.environ.setdefault("DISPLAY", ":0.0")

# ── 本 Demo 不使用 cv2，无需 fix_cv2_qt_conflict ─────────────────────

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QTextEdit, QPushButton, QStatusBar, QMessageBox,
    QComboBox, QSlider, QCheckBox, QLineEdit, QPlainTextEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


# ═════════════════════════════════════════════════════════════════════════════════
# 常量
# ═════════════════════════════════════════════════════════════════════════════════

VOICE_TYPES = OrderedDict([
    (101001, "智瑜-情感女声"),
    (101002, "智聆-通用女声"),
    (101003, "智美-客服女声"),
    (101004, "智云-通用男声"),
    (101005, "智莉-通用女声"),
    (101006, "智言-助手女声"),
    (101007, "智娜-客服女声"),
    (101008, "智琪-客服女声"),
    (101009, "智芸-知性女声"),
    (101010, "智轩-通用男声"),
    (101011, "智宇-通用男声"),
    (101012, "智月-客服女声"),
    (101013, "智飞-激昂男声"),
    (101014, "智辉-新闻男声"),
    (101015, "智婷-新闻女声"),
    (101016, "智刚-通用男声"),
    (101017, "智瑞-通用男声"),
])

CODECS = OrderedDict([
    ("pcm", "PCM (raw)"),
    ("wav", "WAV (with header)"),
    ("mp3", "MP3"),
])

SAMPLE_RATES = OrderedDict([
    (8000,  "8 kHz"),
    (16000, "16 kHz"),
    (22050, "22.05 kHz"),
    (24000, "24 kHz"),
    (44100, "44.1 kHz"),
    (48000, "48 kHz"),
])

# 需安装的 Python 包
REQUIRED_PACKAGES = ["requests", "websocket-client"]


# ═════════════════════════════════════════════════════════════════════════════════
# SDK 路径解析
# ═════════════════════════════════════════════════════════════════════════════════

def _resolve_sdk_path() -> str:
    """返回 tencentcloud-speech-sdk-python 的路径。
    优先级: 环境变量 TENCENT_TTS_SDK_PATH > 默认 tts_sdk/ 目录
    """
    env_path = os.environ.get("TENCENT_TTS_SDK_PATH", "")
    if env_path and os.path.isdir(env_path):
        return env_path

    default_path = os.path.join(_PROJECT_ROOT, "tts_sdk")
    if os.path.isdir(default_path):
        return default_path

    raise FileNotFoundError(
        f"SDK 未找到。请克隆:\n"
        f"  git clone https://github.com/TencentCloud/tencentcloud-speech-sdk-python {default_path}\n"
        f"或设置 TENCENT_TTS_SDK_PATH 环境变量指向已有的 clone 目录。"
    )


# ═════════════════════════════════════════════════════════════════════════════════
# TtsWorker — 后台 TTS 合成线程
# ═════════════════════════════════════════════════════════════════════════════════

class TtsWorker(QThread):
    """后台线程：调用腾讯云 TTS API 合成语音。

    支持两种模式:
      - HTTP 流式 (默认): SpeechSynthesizer.synthesis() — 简单，阻塞 HTTP POST
      - WebSocket 流式: FlowingSpeechSynthesizer — 支持流式文本输入

    所有回调通过 pyqtSignal 发送到主线程，不直接操作 UI。
    """

    # ── 信号定义 ──────────────────────────────────────────────────────────
    log_message     = pyqtSignal(str)           # 日志消息
    audio_progress  = pyqtSignal(int)           # 已接收音频字节数
    synthesis_done  = pyqtSignal(bytes, float)  # (完整音频数据, 耗时秒)
    synthesis_error = pyqtSignal(str)           # 错误消息
    text_result     = pyqtSignal(str)           # WebSocket 字幕文本
    ws_ready        = pyqtSignal()              # WebSocket 连接就绪

    def __init__(self, appid, secret_id, secret_key,
                 voice_type, codec, sample_rate, speed, volume,
                 text, use_websocket=False):
        super().__init__()  # 无 parent，避免 "Cannot move to target thread" 警告
        self._appid = appid
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._voice_type = voice_type
        self._codec = codec
        self._sample_rate = sample_rate
        self._speed = speed
        self._volume = volume
        self._text = text
        self._use_websocket = use_websocket
        self._cancelled = False

    def cancel(self):
        """设置取消标志。注意: 无法真正中断进行中的网络请求。"""
        self._cancelled = True
        self.log_message.emit("⏹ 用户取消 — 正在等待当前请求结束…")

    def run(self):
        """主入口：根据模式选择 HTTP 或 WebSocket 合成。"""
        try:
            # SDK 路径已在 main() 中添加到 sys.path，此处可直接导入
            if self._use_websocket:
                self._run_websocket()
            else:
                self._run_http()
        except ImportError as e:
            self.synthesis_error.emit(f"SDK 导入失败: {e}")
        except Exception as e:
            self.synthesis_error.emit(f"未预期的错误: {e}")

    # ── HTTP 模式 ─────────────────────────────────────────────────────────

    def _run_http(self):
        from tts.speech_synthesizer import SpeechSynthesizer, SpeechSynthesisListener
        from common.credential import Credential

        listener = _HttpListener(self)
        credential = Credential(self._secret_id, self._secret_key)
        synth = SpeechSynthesizer(self._appid, credential, self._voice_type, listener)
        synth.set_codec(self._codec)
        synth.set_sample_rate(self._sample_rate)
        synth.set_speed(self._speed)
        synth.set_volume(self._volume)

        self.log_message.emit(
            f"HTTP 合成开始 — voice={self._voice_type} codec={self._codec} "
            f"rate={self._sample_rate} speed={self._speed} volume={self._volume}"
        )

        t_start = time.time()
        try:
            synth.synthesis(self._text)
        except Exception as e:
            if not self._cancelled:
                self.synthesis_error.emit(f"HTTP 请求异常: {e}")
            return

        if self._cancelled:
            self.log_message.emit("合成已取消，丢弃结果。")
            return

        # on_complete / on_fail 在 synthesis() 内部同步调用
        # 如果走到这里且没有错误，说明 listener 已处理完毕
        if listener._error:
            self.synthesis_error.emit(listener._error)
        elif listener._data:
            elapsed = time.time() - t_start
            self.synthesis_done.emit(bytes(listener._data), elapsed)
        else:
            self.synthesis_error.emit("未收到任何音频数据")

    # ── WebSocket 模式 ────────────────────────────────────────────────────

    def _run_websocket(self):
        from tts.flowing_speech_synthesizer import (
            FlowingSpeechSynthesizer, FlowingSpeechSynthesisListener,
        )
        from common.credential import Credential

        listener = _WsListener(self)
        credential = Credential(self._secret_id, self._secret_key)
        synth = FlowingSpeechSynthesizer(self._appid, credential, listener)
        synth.set_voice_type(self._voice_type)
        synth.set_codec(self._codec)
        synth.set_sample_rate(self._sample_rate)
        synth.set_speed(self._speed)
        synth.set_volume(self._volume)

        self.log_message.emit(
            f"WebSocket 合成开始 — voice={self._voice_type} codec={self._codec} "
            f"rate={self._sample_rate} speed={self._speed} volume={self._volume}"
        )

        t_start = time.time()
        synth.start()

        if not synth.wait_ready(10000):
            self.synthesis_error.emit("WebSocket 连接超时 (10s)")
            return

        self.ws_ready.emit()
        self.log_message.emit("WebSocket 已就绪，发送合成文本…")

        synth.process(self._text)
        synth.complete()
        synth.wait()  # 阻塞直到 WebSocket 线程结束

        if self._cancelled:
            self.log_message.emit("合成已取消，丢弃结果。")
            return

        if listener._error:
            self.synthesis_error.emit(listener._error)
        elif listener._data:
            elapsed = time.time() - t_start
            self.synthesis_done.emit(bytes(listener._data), elapsed)
        else:
            self.synthesis_error.emit("未收到任何音频数据")


# ═════════════════════════════════════════════════════════════════════════════════
# Listener 辅助类 (模块级，供 TtsWorker 使用)
# ═════════════════════════════════════════════════════════════════════════════════

class _HttpListener:
    """HTTP TTS 回调监听器 — 适配 SpeechSynthesisListener 接口。

    注意: SpeechSynthesizer.synthesis() 的 on_message 回调每次传入
    **累积**的音频数据（不是增量），因此只需记录最终数据。
    """

    def __init__(self, worker: TtsWorker):
        self._worker = worker
        self._data = bytearray()
        self._error = ""
        self._last_size = 0

    def on_message(self, response: dict):
        data = response.get("data", b"")
        new_bytes = len(data) - self._last_size
        self._last_size = len(data)
        self._data = bytearray(data)
        self._worker.audio_progress.emit(len(data))

    def on_complete(self, response: dict):
        self._data = bytearray(response.get("data", b""))
        self._worker.audio_progress.emit(len(self._data))
        self._worker.log_message.emit(
            f"HTTP on_complete — session={response.get('session_id', '?')} "
            f"size={len(self._data)} bytes"
        )

    def on_fail(self, response: dict):
        code = response.get("Code", "UNKNOWN")
        msg = response.get("Message", "未知错误")
        self._error = f"TTS API 错误 [{code}]: {msg}"


class _WsListener:
    """WebSocket TTS 回调监听器 — 适配 FlowingSpeechSynthesisListener 接口。

    音频数据通过 on_audio_result 逐帧回调（增量，需自行累积）。
    """

    def __init__(self, worker: TtsWorker):
        self._worker = worker
        self._data = bytearray()
        self._error = ""

    def on_synthesis_start(self, session_id: str):
        self._worker.log_message.emit(f"WS on_synthesis_start — session={session_id}")

    def on_audio_result(self, audio_bytes: bytes):
        self._data.extend(audio_bytes)
        self._worker.audio_progress.emit(len(self._data))

    def on_text_result(self, response: dict):
        result = response.get("result", {})
        subtitles = result.get("subtitles", [])
        if subtitles:
            texts = [s.get("Text", "") for s in subtitles]
            self._worker.text_result.emit(" | ".join(texts))

    def on_synthesis_end(self):
        self._worker.log_message.emit(
            f"WS on_synthesis_end — size={len(self._data)} bytes"
        )

    def on_synthesis_fail(self, response: dict):
        code = response.get("code", "UNKNOWN")
        msg = response.get("message", "未知错误")
        self._error = f"WebSocket TTS 错误 [{code}]: {msg}"


# ═════════════════════════════════════════════════════════════════════════════════
# MainWindow — PyQt5 主窗口
# ═════════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """腾讯云 TTS 测试主窗口。"""

    WINDOW_TITLE = "Tencent Cloud TTS 测试"

    def __init__(self, sdk_path: str = ""):
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)

        # ── SDK 路径 ───────────────────────────────────────────────────
        self._sdk_path = sdk_path

        # ── 后台线程引用 ──────────────────────────────────────────────
        self._worker: TtsWorker | None = None

        # ── 合成结果 ──────────────────────────────────────────────────
        self._audio_data: bytes | None = None
        self._last_wav_path: str = ""
        self._last_sample_rate: int = 16000
        self._last_codec: str = "pcm"
        self._pygame_initialized: bool = False

        # ── 构建界面 ──────────────────────────────────────────────────
        self._build_ui()
        self._load_env_credentials()

    # ── UI 构建 ───────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # ── Credentials ─────────────────────────────────────────────────
        cred_group = QGroupBox("🔑 腾讯云凭证")
        cred_layout = QGridLayout(cred_group)
        cred_layout.setSpacing(6)

        cred_layout.addWidget(QLabel("SecretId:"), 0, 0)
        self._secret_id_edit = QLineEdit()
        self._secret_id_edit.setPlaceholderText("从 https://console.cloud.tencent.com/cam/capi 获取")
        cred_layout.addWidget(self._secret_id_edit, 0, 1)

        cred_layout.addWidget(QLabel("SecretKey:"), 1, 0)
        self._secret_key_edit = QLineEdit()
        self._secret_key_edit.setEchoMode(QLineEdit.Password)
        self._secret_key_edit.setPlaceholderText("从 https://console.cloud.tencent.com/cam/capi 获取")
        cred_layout.addWidget(self._secret_key_edit, 1, 1)

        cred_layout.addWidget(QLabel("AppId:"), 2, 0)
        self._appid_edit = QLineEdit()
        self._appid_edit.setPlaceholderText("从 https://console.cloud.tencent.com/developer 获取")
        cred_layout.addWidget(self._appid_edit, 2, 1)

        layout.addWidget(cred_group)

        # ── TTS Parameters ──────────────────────────────────────────────
        param_group = QGroupBox("🎛️ TTS 参数")
        param_layout = QGridLayout(param_group)
        param_layout.setSpacing(6)

        # Voice
        param_layout.addWidget(QLabel("音色:"), 0, 0)
        self._voice_combo = QComboBox()
        for vid, vname in VOICE_TYPES.items():
            self._voice_combo.addItem(f"{vid} — {vname}", vid)
        param_layout.addWidget(self._voice_combo, 0, 1)

        # Codec
        param_layout.addWidget(QLabel("编码:"), 0, 2)
        self._codec_combo = QComboBox()
        for ck, cv in CODECS.items():
            self._codec_combo.addItem(cv, ck)
        self._codec_combo.setCurrentText(CODECS["pcm"])
        param_layout.addWidget(self._codec_combo, 0, 3)

        # Sample Rate
        param_layout.addWidget(QLabel("采样率:"), 1, 0)
        self._rate_combo = QComboBox()
        for rk, rv in SAMPLE_RATES.items():
            self._rate_combo.addItem(rv, rk)
        self._rate_combo.setCurrentIndex(1)  # 默认 16000
        param_layout.addWidget(self._rate_combo, 1, 1)

        # Speed slider
        param_layout.addWidget(QLabel("语速:"), 1, 2)
        speed_row = QHBoxLayout()
        self._speed_slider = QSlider(Qt.Horizontal)
        self._speed_slider.setRange(-20, 60)  # -2.0 ~ 6.0, 精度 0.1
        self._speed_slider.setValue(0)
        self._speed_slider.setTickPosition(QSlider.TicksBelow)
        self._speed_slider.setTickInterval(10)
        self._speed_label = QLabel("0.0")
        self._speed_label.setMinimumWidth(35)
        self._speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._speed_slider.valueChanged.connect(
            lambda v: self._speed_label.setText(f"{v / 10:.1f}")
        )
        speed_row.addWidget(self._speed_slider)
        speed_row.addWidget(self._speed_label)
        param_layout.addLayout(speed_row, 1, 3)

        # Volume slider
        param_layout.addWidget(QLabel("音量:"), 2, 0)
        vol_row = QHBoxLayout()
        self._volume_slider = QSlider(Qt.Horizontal)
        self._volume_slider.setRange(-10, 10)
        self._volume_slider.setValue(0)
        self._volume_slider.setTickPosition(QSlider.TicksBelow)
        self._volume_slider.setTickInterval(5)
        self._volume_label = QLabel("0")
        self._volume_label.setMinimumWidth(30)
        self._volume_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._volume_slider.valueChanged.connect(
            lambda v: self._volume_label.setText(str(v))
        )
        vol_row.addWidget(self._volume_slider)
        vol_row.addWidget(self._volume_label)
        param_layout.addLayout(vol_row, 2, 1)

        # WebSocket toggle
        self._ws_checkbox = QCheckBox("使用 WebSocket 流式合成")
        param_layout.addWidget(self._ws_checkbox, 2, 2, 1, 2)

        layout.addWidget(param_group)

        # ── SDK Path ────────────────────────────────────────────────────
        sdk_group = QGroupBox("📦 SDK 路径")
        sdk_layout = QHBoxLayout(sdk_group)
        self._sdk_path_edit = QLineEdit()
        self._sdk_path_edit.setPlaceholderText(
            "默认: tts_sdk/ 或 $TENCENT_TTS_SDK_PATH"
        )
        self._sdk_path_edit.setReadOnly(True)
        self._sdk_path_edit.setText(self._sdk_path)
        sdk_layout.addWidget(self._sdk_path_edit)
        layout.addWidget(sdk_group)

        # ── Input Text ──────────────────────────────────────────────────
        text_group = QGroupBox("📝 合成文本")
        text_layout = QVBoxLayout(text_group)
        self._text_edit = QPlainTextEdit()
        self._text_edit.setPlaceholderText("输入要合成的文本…（中文最多 150 字，英文最多 400 字母）")
        self._text_edit.setMaximumHeight(120)
        self._text_edit.setFont(QFont("Sans", 11))
        text_layout.addWidget(self._text_edit)

        # 字符计数
        self._char_count_label = QLabel("字数: 0")
        self._char_count_label.setStyleSheet("color: #888; font-size: 11px;")
        self._text_edit.textChanged.connect(self._on_text_changed)
        text_layout.addWidget(self._char_count_label, alignment=Qt.AlignRight)

        layout.addWidget(text_group)

        # ── Buttons ─────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self._synth_btn = QPushButton("🔊 合成")
        self._synth_btn.setStyleSheet(
            "QPushButton { background-color: #2d6a2d; }"
            "QPushButton:hover { background-color: #3a8a3a; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        self._synth_btn.clicked.connect(self._on_synthesize)
        btn_layout.addWidget(self._synth_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet(
            "QPushButton { background-color: #6a2d2d; }"
            "QPushButton:hover { background-color: #8a3a3a; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        self._stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self._stop_btn)

        self._play_btn = QPushButton("▶ 播放")
        self._play_btn.setEnabled(False)
        self._play_btn.setStyleSheet(
            "QPushButton { background-color: #2d4a6a; }"
            "QPushButton:hover { background-color: #3a5a8a; }"
            "QPushButton:disabled { background-color: #2a2a2a; color: #666; }"
        )
        self._play_btn.clicked.connect(self._on_play)
        btn_layout.addWidget(self._play_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # ── Log ─────────────────────────────────────────────────────────
        log_group = QGroupBox("📋 日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(4, 12, 4, 4)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("monospace", 9))
        self._log_text.setMaximumHeight(160)
        self._log_text.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #aaa; "
            "border: 1px solid #333; border-radius: 4px; }"
        )
        log_layout.addWidget(self._log_text)
        layout.addWidget(log_group)

        # ── Status Bar ──────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("就绪 — 请填写凭证并输入合成文本")
        self._status_bar.addWidget(self._status_label)

        # 窗口默认大小
        self.resize(820, 680)

    # ── 环境变量凭证自动填充 ──────────────────────────────────────────

    def _load_env_credentials(self):
        """从环境变量预填凭证。"""
        sid = os.environ.get("TENCENT_SECRET_ID", "")
        skey = os.environ.get("TENCENT_SECRET_KEY", "")
        appid = os.environ.get("TENCENT_APPID", "")

        if sid:
            self._secret_id_edit.setText(sid)
            self._log("已从 TENCENT_SECRET_ID 环境变量加载 SecretId")
        if skey:
            self._secret_key_edit.setText(skey)
            self._log("已从 TENCENT_SECRET_KEY 环境变量加载 SecretKey")
        if appid:
            self._appid_edit.setText(appid)
            self._log("已从 TENCENT_APPID 环境变量加载 AppId")

    # ── 字符计数 ──────────────────────────────────────────────────────

    def _on_text_changed(self):
        text = self._text_edit.toPlainText()
        # 混合文本: 汉字计 1 字，英文单词粗略估算
        chinese = sum(1 for c in text if '一' <= c <= '鿿')
        other = len(text) - chinese
        self._char_count_label.setText(
            f"字数: {len(text)}  (中文 {chinese} + 其他 {other})"
        )

    # ── 合成 ──────────────────────────────────────────────────────────

    def _on_synthesize(self):
        """验证输入并启动 TTS 合成线程。"""
        # 1. 验证凭证
        secret_id = self._secret_id_edit.text().strip()
        secret_key = self._secret_key_edit.text().strip()
        appid_str = self._appid_edit.text().strip()

        if not secret_id:
            QMessageBox.warning(self, "缺少 SecretId", "请输入腾讯云 SecretId。")
            return
        if not secret_key:
            QMessageBox.warning(self, "缺少 SecretKey", "请输入腾讯云 SecretKey。")
            return
        if not appid_str:
            QMessageBox.warning(self, "缺少 AppId", "请输入腾讯云 AppId。")
            return
        try:
            appid = int(appid_str)
        except ValueError:
            QMessageBox.warning(self, "AppId 格式错误", "AppId 必须为整数。")
            return

        # 2. 验证文本
        text = self._text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "缺少文本", "请输入要合成的文本。")
            return

        # 3. 防止重复点击
        if self._worker is not None and self._worker.isRunning():
            self._log("⚠️ 合成线程已在运行中")
            return

        # 4. 收集参数
        voice_type = self._voice_combo.currentData()
        codec = self._codec_combo.currentData()
        sample_rate = self._rate_combo.currentData()
        speed = self._speed_slider.value() / 10.0
        volume = self._volume_slider.value()
        use_ws = self._ws_checkbox.isChecked()

        self._last_sample_rate = sample_rate
        self._last_codec = codec

        # 5. 更新 UI 状态
        self._synth_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._play_btn.setEnabled(False)
        self._audio_data = None
        self._last_wav_path = ""
        self._log_text.clear()
        self._status_label.setText("合成中…")

        # 6. 启动合成线程
        self._worker = TtsWorker(
            appid=appid,
            secret_id=secret_id,
            secret_key=secret_key,
            voice_type=voice_type,
            codec=codec,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            text=text,
            use_websocket=use_ws,
        )
        self._worker.log_message.connect(self._on_log)
        self._worker.audio_progress.connect(self._on_audio_progress)
        self._worker.synthesis_done.connect(self._on_synthesis_done)
        self._worker.synthesis_error.connect(self._on_synthesis_error)
        self._worker.text_result.connect(self._on_text_result)
        self._worker.ws_ready.connect(self._on_ws_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    # ── 停止 ──────────────────────────────────────────────────────────

    def _on_stop(self):
        """取消正在进行中的合成。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
        self._status_label.setText("已取消")

    # ── 播放 ──────────────────────────────────────────────────────────

    def _on_play(self):
        """播放最后一次合成的音频。"""
        if not self._last_wav_path or not os.path.exists(self._last_wav_path):
            self._log("没有可播放的音频文件")
            return

        self._log(f"开始播放: {self._last_wav_path}")
        self._status_label.setText("播放中…")

        # 优先使用 pygame（非阻塞）
        try:
            import pygame
            if not self._pygame_initialized:
                pygame.mixer.init(
                    frequency=self._last_sample_rate, size=-16, channels=1
                )
                self._pygame_initialized = True
            sound = pygame.mixer.Sound(self._last_wav_path)
            sound.play()
            self._log("pygame 播放已启动")
            self._status_label.setText("pygame 播放中…")
            return
        except Exception as e:
            self._log(f"pygame 播放失败: {e}")

        # 回退到 aplay（非阻塞 subprocess）
        try:
            subprocess.Popen(
                ["aplay", self._last_wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._log("aplay 播放已启动")
            self._status_label.setText("aplay 播放中…")
            return
        except Exception as e:
            self._log(f"aplay 播放失败: {e}")

        self._status_label.setText(f"播放失败 — 文件: {self._last_wav_path}")

    # ── 信号槽：合成进度 ──────────────────────────────────────────────

    def _on_audio_progress(self, total_bytes: int):
        self._status_label.setText(f"合成中… 已接收 {total_bytes} bytes")

    def _on_ws_ready(self):
        self._log("WebSocket 连接已就绪")

    def _on_text_result(self, text: str):
        self._log(f"字幕: {text}")

    # ── 信号槽：合成完成 ──────────────────────────────────────────────

    def _on_synthesis_done(self, audio_data: bytes, elapsed: float):
        """合成成功完成 — 保存 WAV 并启用播放按钮。"""
        self._audio_data = audio_data
        self._log(
            f"✅ 合成完成 — {len(audio_data)} bytes, "
            f"耗时 {elapsed:.1f}s "
            f"({len(audio_data) / max(elapsed, 0.001) / 1000:.1f} KB/s)"
        )

        # 保存 WAV
        try:
            wav_path = os.path.join(tempfile.gettempdir(), "tts_output.wav")
            _save_wav(wav_path, audio_data, self._last_sample_rate, self._last_codec)
            self._last_wav_path = wav_path
            self._log(f"已保存: {wav_path}")
            self._play_btn.setEnabled(True)
            self._status_label.setText(
                f"合成完成 — {len(audio_data)} bytes, {elapsed:.1f}s"
            )
        except Exception as e:
            self._log(f"❌ WAV 保存失败: {e}")
            self._status_label.setText(f"WAV 保存失败: {e}")

    # ── 信号槽：合成失败 ──────────────────────────────────────────────

    def _on_synthesis_error(self, msg: str):
        """合成出错。"""
        self._log(f"❌ 合成错误: {msg}")
        self._status_label.setText(f"错误: {msg}")

    # ── 信号槽：线程结束 ──────────────────────────────────────────────

    def _on_worker_finished(self):
        """合成线程结束时恢复按钮状态。"""
        self._synth_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ── 日志 ──────────────────────────────────────────────────────────

    def _on_log(self, msg: str):
        """接收 TtsWorker 的信号并追加到日志面板。"""
        self._log(msg)

    def _log(self, msg: str):
        """追加时间戳日志到 QTextEdit。"""
        timestamp = time.strftime("%H:%M:%S")
        log_widget = getattr(self, "_log_text", None)
        if log_widget is not None:
            log_widget.append(f"[{timestamp}] {msg}")
            # 限制日志行数
            if log_widget.document().blockCount() > 200:
                cursor = log_widget.textCursor()
                cursor.movePosition(cursor.Start)
                cursor.movePosition(cursor.Down, cursor.KeepAnchor, 30)
                cursor.removeSelectedText()
        else:
            print(f"[{timestamp}] {msg}")

    # ── 窗口关闭 ──────────────────────────────────────────────────────

    def closeEvent(self, event):
        """关闭窗口时停止合成线程。"""
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════════════════════════

def _save_wav(filepath: str, audio_data: bytes, sample_rate: int, codec: str):
    """将音频数据保存为 WAV 文件。

    - PCM: 原始 16-bit 有符号 PCM，需添加 WAV 头
    - WAV: 已包含 WAV 头（SDK codec="wav"），直接写入
    - MP3: 保存为 .mp3 然后无法用 aplay 播放，仅保留原始数据
    """
    import wave

    if codec == "wav":
        with open(filepath, "wb") as f:
            f.write(audio_data)
        return

    if codec == "mp3":
        # MP3 无法简单转 WAV — 保存为 .mp3 并通知用户
        mp3_path = filepath.replace(".wav", ".mp3")
        with open(mp3_path, "wb") as f:
            f.write(audio_data)
        raise ValueError(f"MP3 已保存至 {mp3_path}，请用其他播放器播放")

    # PCM → WAV
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)


# ═════════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    # ── 解析 SDK 路径 ──────────────────────────────────────────────────
    sdk_path = ""
    try:
        sdk_path = _resolve_sdk_path()
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)
        # 验证 SDK 是否可导入
        from common.credential import Credential  # noqa: F401
    except FileNotFoundError as e:
        sdk_path = ""
    except ImportError as e:
        sdk_path = ""

    # ── 创建 QApplication ──────────────────────────────────────────────
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
        QSlider::groove:horizontal {
            border: 1px solid #555; height: 6px;
            background: #3a3a3a; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #6a6a6a; width: 14px; margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover { background: #8a8a8a; }
        QSlider::sub-page:horizontal {
            background: #4a7a9a; border-radius: 3px;
        }
        QCheckBox {
            color: #ddd; spacing: 6px;
        }
        QCheckBox::indicator {
            width: 16px; height: 16px;
        }
        QLineEdit, QPlainTextEdit {
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

    window = MainWindow(sdk_path=sdk_path)

    # SDK 不可用时禁用合成按钮
    if not sdk_path:
        window._log("❌ SDK 未找到！")
        window._log(
            "请执行: git clone "
            "https://github.com/TencentCloud/tencentcloud-speech-sdk-python "
            f"{os.path.join(_PROJECT_ROOT, 'tts_sdk')}"
        )
        window._log("或设置 TENCENT_TTS_SDK_PATH 环境变量。")
        window._synth_btn.setEnabled(False)
        window._synth_btn.setToolTip(
            "SDK 未找到。请克隆 tencentcloud-speech-sdk-python 到 tts_sdk/ 目录。"
        )

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
