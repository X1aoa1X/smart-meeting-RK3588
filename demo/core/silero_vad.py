"""Silero VAD 语音活动检测器 — CPU 推理 (PyTorch JIT)。

用法:
  vad = SileroVAD(threshold=0.5)
  vad.load()                    # 加载 JIT 模型
  prob = vad.process(chunk)     # chunk: float32 numpy array shape=(512,), [-1, 1]
  vad.reset()                   # 重置 VAD 状态

模型路径: models/silero_vad.jit (相对于项目根目录或通过 model_path 参数指定)
"""

import os
import logging
import numpy as np

# ═════════════════════════════════════════════════════════════════════════════════
# PyTorch 2.7 logging bug 修复 — 必须在 import torch 之前执行
# ═════════════════════════════════════════════════════════════════════════════════

_torch = None

_orig_logger_setLevel = logging.Logger.setLevel
_orig_handler_setLevel = logging.Handler.setLevel


def _patched_setLevel(self, level):
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.WARNING
    return _orig_logger_setLevel(self, level)


def _patched_handler_setLevel(self, level):
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.WARNING
    return _orig_handler_setLevel(self, level)


logging.Logger.setLevel = _patched_setLevel
logging.Handler.setLevel = _patched_handler_setLevel

try:
    import torch as _torch
finally:
    # 恢复原始方法
    logging.Logger.setLevel = _orig_logger_setLevel
    logging.Handler.setLevel = _orig_handler_setLevel
    # 修复 PyTorch 2.7 可能破坏的 _nameToLevel / _levelToName 映射表
    # (torch._logging 可能调用 addLevelName 或用非法值污染内部字典)
    _std_levels = {
        'CRITICAL': 50, 'FATAL': 50, 'ERROR': 40,
        'WARN': 30, 'WARNING': 30, 'INFO': 20,
        'DEBUG': 10, 'NOTSET': 0,
    }
    for _name, _num in _std_levels.items():
        if _name not in logging._nameToLevel or logging._nameToLevel[_name] != _num:
            logging._nameToLevel[_name] = _num
        if _num not in logging._levelToName:
            logging._levelToName[_num] = _name


# ═════════════════════════════════════════════════════════════════════════════════
# SileroVAD
# ═════════════════════════════════════════════════════════════════════════════════

class SileroVAD:
    """Silero VAD 语音活动检测器 — 加载本地 JIT 模型，在 CPU 上推理。

    使用流式接口: 每 32ms (512 帧 @ 16kHz) 调用一次 process()。
    模型内部管理 LSTM 隐藏状态（调用方无需维护状态）。
    需要创建新实例来重置 VAD 状态。
    """

    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512          # 32ms @ 16kHz

    # 本地模型路径（相对于脚本所在目录）
    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "models", "silero_vad.jit")

    def __init__(self, model_path: str | None = None, threshold: float = 0.5):
        self.threshold = threshold
        self._model_path = model_path or self.DEFAULT_MODEL_PATH
        self._model = None
        self._loaded = False

    def load(self) -> bool:
        """加载 Silero VAD JIT 模型。"""
        if self._loaded:
            return True

        if _torch is None:
            print("[SileroVAD] PyTorch 未安装")
            return False

        if not os.path.exists(self._model_path):
            print(f"[SileroVAD] 模型文件不存在: {self._model_path}")
            return False

        try:
            self._model = _torch.jit.load(self._model_path)
            # 预热：前传一帧静音以初始化内部状态
            dummy = _torch.zeros(1, self.CHUNK_SIZE)
            with _torch.no_grad():
                self._model(dummy, self.SAMPLE_RATE)
            self._loaded = True
            print(f"[SileroVAD] 模型已加载: {self._model_path}")
            return True
        except Exception as e:
            print(f"[SileroVAD] 模型加载失败: {e}")
            return False

    def reset(self):
        """重置 VAD 状态（重新加载模型，完全重置）。"""
        if self._model is not None:
            self._loaded = False
            self._model = None
        self.load()

    def reset_states(self):
        """重置 VAD 内部 LSTM 状态（不重新加载模型，轻量操作）。

        用于从短暂的音频断连中恢复，避免重新读取模型文件。
        """
        if not self._loaded or self._model is None:
            return
        try:
            self._model.reset_states()
            # 预热一帧静音以确保状态稳定
            dummy = _torch.zeros(1, self.CHUNK_SIZE)
            with _torch.no_grad():
                self._model(dummy, self.SAMPLE_RATE)
        except Exception:
            # Fallback: 完全重新加载
            self._loaded = False
            self._model = None
            self.load()

    def process(self, chunk: np.ndarray) -> float:
        """处理一个音频块，返回语音概率 [0, 1]。

        Args:
            chunk: float32 numpy 数组，shape=(512,)，范围 [-1, 1]

        Returns:
            speech_prob: 语音概率 ∈ [0, 1]。模型未加载时返回 0.0。
            注意：这是逐帧概率，不含平滑。调用方应自行管理时间平滑。
        """
        if not self._loaded or self._model is None:
            return 0.0

        # 确保 chunk 大小正确
        if len(chunk) < self.CHUNK_SIZE:
            pad = np.zeros(self.CHUNK_SIZE, dtype=np.float32)
            pad[:len(chunk)] = chunk
            chunk = pad
        elif len(chunk) > self.CHUNK_SIZE:
            chunk = chunk[:self.CHUNK_SIZE]

        # 转换为 torch tensor: [1, 512]（无需 .copy()，from_numpy 安全处理）
        x = _torch.from_numpy(chunk).float().unsqueeze(0)

        # 模型推理 — 模型内部管理 LSTM 状态
        with _torch.no_grad():
            try:
                prob = self._model(x, self.SAMPLE_RATE)
            except Exception:
                return 0.0

        return float(prob.item())

    @property
    def is_loaded(self) -> bool:
        return self._loaded
