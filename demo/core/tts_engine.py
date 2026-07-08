# -*- coding: utf-8 -*-
"""腾讯云 TTS 引擎 — 异步语音合成 (QThread)，使用 WebSocket 流式 API。

封装 tencentcloud-speech-sdk-python 的 FlowingSpeechSynthesizer (WebSocket)。
与 demos/tts_demo.py 使用完全一致的 API 路径，已在 RK3588 上验证可行。

凭证来源 (优先级递减):
  1. 构造参数 explicit_secret_id/key/appid
  2. 环境变量 TENCENT_SECRET_ID / TENCENT_SECRET_KEY / TENCENT_APPID

SDK 路径: TENCENT_TTS_SDK_PATH 环境变量或项目根目录 tts_sdk/

用法:
  engine = TTSEngine()
  engine.diagnose()  # 打印可用性诊断
  if engine.is_available():
      engine.synthesize_async("你好", on_done=lambda pcm: ...)
"""

import os
import sys
import logging

from PyQt5.QtCore import QObject, QThread, pyqtSignal

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════════
# SDK 路径解析
# ═════════════════════════════════════════════════════════════════════════════════

def _resolve_sdk_path() -> str:
    """解析 tencentcloud-speech-sdk-python 路径。"""
    env_path = os.environ.get("TENCENT_TTS_SDK_PATH", "")
    if env_path and os.path.isdir(env_path):
        logger.info(f"SDK path from env: {env_path}")
        return env_path

    # 从项目根目录找 tts_sdk/
    project_root = os.environ.get(
        "SMART_MEETING_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_path = os.path.join(project_root, "tts_sdk")
    logger.info(f"SDK path computed: {default_path} (exists={os.path.isdir(default_path)})")
    if os.path.isdir(default_path):
        return default_path

    raise FileNotFoundError(
        f"tencentcloud-speech-sdk-python 未找到。"
        f"已检查: env TENCENT_TTS_SDK_PATH={env_path!r}, "
        f"default={default_path}")


# ═════════════════════════════════════════════════════════════════════════════════
# TtsSynthesisWorker — WebSocket 流式合成 (与 tts_demo.py 一致)
# ═════════════════════════════════════════════════════════════════════════════════

class TtsSynthesisWorker(QThread):
    """后台线程: 调用腾讯云 WebSocket TTS API 合成语音。

    使用 FlowingSpeechSynthesizer — 与 demos/tts_demo.py 验证通过的路径一致。
    """

    synthesis_done = pyqtSignal(bytes, float)   # (pcm_data, elapsed_seconds)
    synthesis_error = pyqtSignal(str)            # error message

    def __init__(self, appid: int, secret_id: str, secret_key: str,
                 text: str, voice_type: int = 101002,
                 codec: str = "pcm", sample_rate: int = 16000,
                 speed: float = 0.0, volume: int = 0,
                 sdk_path: str = ""):
        super().__init__()
        self._appid = appid
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._text = text
        self._voice_type = voice_type
        self._codec = codec
        self._sample_rate = sample_rate
        self._speed = speed
        self._volume = volume
        self._sdk_path = sdk_path

    def run(self):
        try:
            self._do_synthesis_ws()
        except ImportError as e:
            self.synthesis_error.emit(f"SDK 导入失败: {e}")
        except Exception as e:
            self.synthesis_error.emit(f"合成异常: {e}")

    def _do_synthesis_ws(self):
        """WebSocket 流式合成 — 与 demos/tts_demo.py 的 _run_websocket 一致。"""
        import time

        if self._sdk_path and self._sdk_path not in sys.path:
            sys.path.insert(0, self._sdk_path)

        from tts.flowing_speech_synthesizer import (
            FlowingSpeechSynthesizer, FlowingSpeechSynthesisListener,
        )
        from common.credential import Credential

        listener = _WsListener()
        credential = Credential(self._secret_id, self._secret_key)
        synth = FlowingSpeechSynthesizer(self._appid, credential, listener)
        synth.set_voice_type(self._voice_type)
        synth.set_codec(self._codec)
        synth.set_sample_rate(self._sample_rate)
        synth.set_speed(self._speed)
        synth.set_volume(self._volume)

        t_start = time.time()
        synth.start()

        if not synth.wait_ready(10000):
            self.synthesis_error.emit("WebSocket 连接超时 (10s)")
            return

        synth.process(self._text)
        synth.complete()
        synth.wait()

        if listener.error:
            self.synthesis_error.emit(listener.error)
        elif listener.data:
            elapsed = time.time() - t_start
            self.synthesis_done.emit(bytes(listener.data), elapsed)
        else:
            self.synthesis_error.emit("未收到音频数据")


class _WsListener:
    """WebSocket TTS 回调适配器 — 桥接 FlowingSpeechSynthesisListener 和 pyqtSignal。"""

    def __init__(self):
        self.data = bytearray()
        self.error = ""

    def on_synthesis_start(self, session_id: str):
        pass

    def on_audio_result(self, audio_bytes: bytes):
        self.data.extend(audio_bytes)

    def on_text_result(self, response: dict):
        pass

    def on_synthesis_end(self):
        pass

    def on_synthesis_fail(self, response: dict):
        code = response.get("code", "UNKNOWN")
        msg = response.get("message", "未知错误")
        self.error = f"WebSocket TTS 错误 [{code}]: {msg}"


# ═════════════════════════════════════════════════════════════════════════════════
# TTSEngine
# ═════════════════════════════════════════════════════════════════════════════════

class TTSEngine(QObject):
    """腾讯云 TTS 引擎 — 凭证管理 + 异步合成 (WebSocket)。

    构造参数可显式传入凭证，未提供则从环境变量读取。
    """

    tts_ready = pyqtSignal()             # 凭证就绪
    tts_unavailable = pyqtSignal(str)    # 凭证缺失原因

    def __init__(self, parent=None,
                 secret_id: str = "",
                 secret_key: str = "",
                 appid: int = 0,
                 sdk_path: str = ""):
        super().__init__(parent)

        # ── 活跃 worker 引用 (防止 GC 过早回收 QThread) ──────────────
        self._workers: list = []

        # ── 凭证: 参数 > 环境变量 ──────────────────────────────────────
        self._secret_id = secret_id or os.environ.get("TENCENT_SECRET_ID", "").strip()
        self._secret_key = secret_key or os.environ.get("TENCENT_SECRET_KEY", "").strip()
        appid_str = str(appid) if appid > 0 else os.environ.get("TENCENT_APPID", "").strip()

        # ── SDK 路径 ───────────────────────────────────────────────────
        self._sdk_path = sdk_path
        if not self._sdk_path:
            try:
                self._sdk_path = _resolve_sdk_path()
            except FileNotFoundError as e:
                self._sdk_path = ""
                self._unavailable_reason = str(e)

        # ── 验证 ───────────────────────────────────────────────────────
        self._unavailable_reason = ""
        self._appid = 0

        if not self._secret_id:
            self._unavailable_reason = "缺少 SecretId (TENCENT_SECRET_ID)"
        elif not self._secret_key:
            self._unavailable_reason = "缺少 SecretKey (TENCENT_SECRET_KEY)"
        elif not appid_str:
            self._unavailable_reason = "缺少 AppId (TENCENT_APPID)"
        elif not self._sdk_path:
            self._unavailable_reason = f"SDK 未找到 (已检查 tts_sdk/ 和 TENCENT_TTS_SDK_PATH)"
        else:
            try:
                self._appid = int(appid_str)
            except ValueError:
                self._unavailable_reason = f"AppId 格式错误: {appid_str!r}"

        # ── 发射信号 ───────────────────────────────────────────────────
        if self._unavailable_reason:
            logger.warning(f"TTSEngine 不可用: {self._unavailable_reason}")
            self.tts_unavailable.emit(self._unavailable_reason)
        else:
            logger.info(f"TTSEngine 就绪 (appid={self._appid}, sdk={self._sdk_path})")
            self.tts_ready.emit()

    # ── 公开属性 ──────────────────────────────────────────────────────

    @property
    def appid(self) -> int:
        return self._appid

    @property
    def sdk_path(self) -> str:
        return self._sdk_path

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def is_available(self) -> bool:
        return not self._unavailable_reason

    def diagnose(self) -> str:
        """返回可用性诊断字符串，用于日志排查。"""
        if self.is_available():
            return (f"TTSEngine OK: appid={self._appid}, "
                    f"secret_id={'***' if self._secret_id else '(empty)'}, "
                    f"sdk={self._sdk_path}")
        else:
            return (f"TTSEngine UNAVAILABLE: {self._unavailable_reason} | "
                    f"secret_id={'***' if self._secret_id else '(empty)'} "
                    f"secret_key={'***' if self._secret_key else '(empty)'} "
                    f"appid={self._appid} sdk={self._sdk_path or '(empty)'}")

    # ── 异步合成 ──────────────────────────────────────────────────────

    def synthesize_async(self, text: str,
                         on_done,  # Callable[[bytes], None]
                         on_error=None,  # Callable[[str], None] | None
                         voice_type: int = 101002,
                         sample_rate: int = 16000,
                         speed: float = 0.0,
                         volume: int = 0):
        """启动异步 TTS 合成 (WebSocket)。

        on_done(pcm_data: bytes) — 合成成功回调
        on_error(msg: str)        — 合成失败回调 (可选)
        """
        if not self.is_available():
            if on_error:
                on_error(f"TTS 引擎不可用: {self._unavailable_reason}")
            return None

        worker = TtsSynthesisWorker(
            appid=self._appid,
            secret_id=self._secret_id,
            secret_key=self._secret_key,
            text=text,
            voice_type=voice_type,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            sdk_path=self._sdk_path,
        )

        # ★ 关键: 保持 worker 引用，防止 Python GC 过早回收 QThread
        # QThread 的 Python wrapper 一旦被 GC 就会触发 C++ 析构，
        # 若线程仍在运行则导致 "QThread: Destroyed while thread is still running"
        # 并可能触发 SIGABRT 崩溃。
        self._workers.append(worker)

        def _cleanup():
            try:
                self._workers.remove(worker)
            except ValueError:
                pass
            worker.deleteLater()

        worker.finished.connect(_cleanup)

        if on_error:
            worker.synthesis_error.connect(on_error)
        worker.synthesis_done.connect(
            lambda data, elapsed: on_done(data))
        worker.start()
        return worker

    def preload_cache(self, cache, texts: list,
                      on_progress=None,
                      sample_rate: int = 16000):
        """批量预合成文本并写入缓存。

        Args:
            cache: TTSCache 实例
            texts: 要合成的文本列表
            on_progress: Callable[[int, int], None] — (current, total) 回调
            sample_rate: 采样率
        """
        total = len(texts)

        def _next(idx: int):
            if idx >= total:
                return
            text = texts[idx]

            def on_done(pcm_data: bytes):
                cache.put(text, pcm_data, sample_rate)
                if on_progress:
                    on_progress(idx + 1, total)
                _next(idx + 1)

            def on_error(msg: str):
                logger.warning(f"预合成失败 [{text}]: {msg}")
                if on_progress:
                    on_progress(idx + 1, total)
                _next(idx + 1)

            self.synthesize_async(
                text, on_done=on_done, on_error=on_error,
                sample_rate=sample_rate)

        _next(0)
