"""后台 ALSA 录音 + Silero VAD 推理线程 (QThread)。

用法:
  thread = AudioVadThread(device="hw:1,0", capture_rate=16000, threshold=0.5)
  thread.silero_speech.connect(on_speech)   # (prob, is_speech, duration)
  thread.vad_error.connect(on_error)         # (msg) — 瞬时错误/恢复中（仅日志）
  thread.vad_ready.connect(on_ready)         # (msg) — VAD 就绪/恢复成功
  thread.start()
  ...
  thread.stop()

依赖: core.alsa_capture.AlsaAudioCapture, core.silero_vad.SileroVAD

稳健性设计:
  - 无硬件 VAD 回退 — Silero VAD 是唯一语音检测来源
  - ALSA 瞬时读取错误: 连续 N 次后触发自动恢复（关闭→重开 ALSA + 重置 VAD 状态）
  - 恢复成功: 发出 vad_ready
  - 恢复失败: 退避无限重试，永不放弃（线程持续运行直到 stop()）
  - 模型加载失败 / ALSA 初始打开失败: 同样无限重试
  - 信号源切换（如 USB 音频设备重枚举）后可自动恢复
"""

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.alsa_capture import AlsaAudioCapture
from core.silero_vad import SileroVAD


class AudioVadThread(QThread):
    """后台线程: ALSA 录音 → Silero VAD 推理 → 通过 Qt 信号发送语音状态。

    与 ReSpeakerReader 并行运行，互不依赖。
    音频通过 NAU8822 板载 codec (hw:1,0) 以 16kHz 单声道采集。

    内置无限自动恢复: ALSA 断连后无限重试重开设备 + 重置 VAD 状态，
    永不回退到硬件 VAD — Silero VAD 是唯一语音检测来源。
    """

    # ── 公开信号 ────────────────────────────────────────────────────
    silero_speech = pyqtSignal(float, bool, float)
    # (speech_probability, is_speech, cumulative_speech_duration_seconds)

    vad_error = pyqtSignal(str)
    # 瞬时/可恢复错误 — 仅用于日志，线程继续运行并自动恢复

    vad_ready = pyqtSignal(str)
    # VAD 初始化成功 或 从错误中恢复成功

    # ── 恢复参数 ────────────────────────────────────────────────────
    MAX_CONSECUTIVE_ERRORS = 30       # 连续读取错误 → 触发恢复（~960ms @ 32ms/chunk）
    RECOVERY_BACKOFF_BASE_MS = 400    # 恢复重试基础退避时间
    RECOVERY_BACKOFF_MAX_MS = 5000    # 恢复重试最大退避时间
    STABILITY_WINDOW = 150            # 稳定读取次数后重置恢复计数器（~4.8s）

    # ── 初始化 ──────────────────────────────────────────────────────

    def __init__(self, device: str = "hw:1,0", capture_rate: int = 16000,
                 model_path: str | None = None, threshold: float = 0.5,
                 min_speech_duration: float = 0.3, pregain: float = 30.0):
        super().__init__()
        self._device = device
        self._capture_rate = capture_rate
        self._model_path = model_path
        self._threshold = threshold
        self._min_speech_duration = min_speech_duration
        self._pregain = pregain
        self._running = False

        # 线程安全属性 — 仅本线程写入，主线程只读
        self.is_speech = False
        self.speech_prob = 0.0
        self.speech_duration = 0.0

        self._capture: AlsaAudioCapture | None = None
        self._vad: SileroVAD | None = None

    # ── 主循环 ──────────────────────────────────────────────────────

    def run(self):
        self._running = True

        # ── 状态变量 ────────────────────────────────────────────
        self.is_speech = False
        self.speech_prob = 0.0
        self.speech_duration = 0.0
        consecutive_errors = 0
        recovery_retries = 0
        stable_reads = 0
        was_ready = False  # 跟踪是否曾经就绪过

        while self._running:
            # ── 步骤 1: 确保 VAD 模型已加载 ─────────────────────
            if self._vad is None:
                self._vad = SileroVAD(
                    model_path=self._model_path, threshold=self._threshold)
                if not self._vad.load():
                    self._vad = None
                    self.vad_error.emit("Silero VAD 模型加载失败 — 重试中...")
                    self._recovery_sleep(recovery_retries)
                    recovery_retries += 1
                    continue

            # ── 步骤 2: 确保 ALSA 采集设备已打开 ────────────────
            if self._capture is None or not self._capture.is_open:
                if not self._try_reopen_capture():
                    self.vad_error.emit("ALSA 录音设备不可用 — 重试中...")
                    self._recovery_sleep(recovery_retries)
                    recovery_retries += 1
                    continue

                # 采集设备就绪
                recovery_retries = 0
                consecutive_errors = 0
                stable_reads = 0
                actual_rate = self._capture.actual_rate
                if was_ready:
                    self.vad_ready.emit(
                        f"ALSA 设备已恢复 {actual_rate}Hz, Silero VAD {self._threshold}")
                else:
                    was_ready = True
                    self.vad_ready.emit(
                        f"ALSA {actual_rate}Hz, Silero VAD {self._threshold}")
                continue

            # ── 步骤 3: 读取音频块 ──────────────────────────────
            chunk = self._capture.read()
            if chunk is None:
                consecutive_errors += 1
                stable_reads = 0

                if consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    # 触发恢复 — 清除语音状态
                    self.silero_speech.emit(0.0, False, 0.0)
                    self.vad_error.emit(
                        f"连续 {consecutive_errors} 次音频读取错误 — 尝试恢复 ALSA 设备...")
                    self._close_capture()
                    self._reset_vad_for_recovery()
                    continue
                elif consecutive_errors <= 3:
                    print(f"[AudioVadThread] 音频读取错误 (×{consecutive_errors})")
                QThread.msleep(10)
                continue

            # ── 步骤 4: 读取成功 → 处理音频 ──────────────────────
            consecutive_errors = 0
            stable_reads += 1
            if stable_reads >= self.STABILITY_WINDOW:
                # 长时间稳定运行 → 重置恢复预算
                recovery_retries = 0
                stable_reads = 0

            # ── 重采样（如需要）─────────────────────────────────
            if self._capture.actual_rate != SileroVAD.SAMPLE_RATE:
                chunk = self._resample(chunk, self._capture.actual_rate,
                                       SileroVAD.SAMPLE_RATE)

            # ── 前置增益（补偿 NAU8822 低电平输入）───────────────
            if self._pregain != 1.0:
                np.multiply(chunk, self._pregain, out=chunk)
                np.clip(chunk, -1.0, 1.0, out=chunk)

            # ── Silero VAD 推理 ──────────────────────────────────
            try:
                self.speech_prob = self._vad.process(chunk)
            except Exception as e:
                self.vad_error.emit(f"VAD 推理异常: {e}")
                self.speech_prob = 0.0

            was_speech = self.is_speech
            self.is_speech = self.speech_prob >= self._threshold

            if self.is_speech:
                if was_speech:
                    self.speech_duration += SileroVAD.CHUNK_SIZE / SileroVAD.SAMPLE_RATE
                else:
                    self.speech_duration = SileroVAD.CHUNK_SIZE / SileroVAD.SAMPLE_RATE
            else:
                self.speech_duration = 0.0

            self.silero_speech.emit(
                self.speech_prob, self.is_speech, self.speech_duration)

        # ── 清理 ────────────────────────────────────────────────
        self._close_capture()
        self._vad = None
        print("[AudioVadThread] 已停止")

    # ── 公开方法 ────────────────────────────────────────────────────

    def stop(self):
        self._running = False
        self.wait(8000)

    # ── 内部方法 ────────────────────────────────────────────────────

    def _close_capture(self):
        """安全关闭 ALSA 采集设备。"""
        if self._capture is not None:
            try:
                self._capture.close()
            except Exception:
                pass
            self._capture = None

    def _reset_vad_for_recovery(self):
        """重置 VAD 模型状态以准备恢复。"""
        if self._vad is not None:
            try:
                self._vad.reset_states()
            except Exception:
                pass

    def _try_reopen_capture(self) -> bool:
        """尝试（重新）打开 ALSA 采集设备。返回 True 表示成功。"""
        if self._capture is not None and self._capture.is_open:
            return True

        self._capture = AlsaAudioCapture(
            device=self._device, rate=self._capture_rate,
            channels=1, chunk_size=SileroVAD.CHUNK_SIZE)

        if self._capture.open():
            return True

        self._capture = None
        return False

    def _recovery_sleep(self, retry_count: int):
        """退避等待，上限封顶。"""
        delay = min(
            self.RECOVERY_BACKOFF_BASE_MS * (1 << min(retry_count, 4)),
            self.RECOVERY_BACKOFF_MAX_MS)
        QThread.msleep(delay)

    # ── 静态工具 ────────────────────────────────────────────────────

    @staticmethod
    def _resample(chunk: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """使用 scipy 将音频 chunk 从 src_rate 重采样到 dst_rate。"""
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(src_rate, dst_rate)
            up = dst_rate // g
            down = src_rate // g
            result = resample_poly(chunk.astype(np.float64), up, down)
            return result[:len(chunk) * dst_rate // src_rate].astype(np.float32)
        except ImportError:
            if src_rate % dst_rate == 0:
                factor = src_rate // dst_rate
                return chunk[::factor].copy()
            target_len = len(chunk) * dst_rate // src_rate
            if target_len <= len(chunk):
                return chunk[:target_len].copy()
            return np.pad(chunk, (0, target_len - len(chunk)))[:target_len]
