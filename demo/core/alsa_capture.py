"""ALSA PCM 音频采集 — 通过 ctypes 调用 libasound，零 pip 依赖。

用法:
  cap = AlsaAudioCapture(device="hw:1,0", rate=16000, channels=1, chunk_size=512)
  if cap.open():
      while True:
          chunk = cap.read()  # float32 numpy array, shape=(chunk_size,), range [-1, 1]
          if chunk is None:
              break
  cap.close()
"""

import ctypes
import ctypes.util
import numpy as np


class AlsaAudioCapture:
    """通过 ctypes 调用 libasound 进行 ALSA PCM 录音。

    无 pip 依赖 — 仅使用系统 libasound.so.2。
    配置为 S16_LE / 1ch / 16000Hz，读取 float32 [-1, 1] 的 numpy 数组。
    """

    # alsa-lib 常量
    SND_PCM_STREAM_CAPTURE = 1
    SND_PCM_ACCESS_RW_INTERLEAVED = 3
    SND_PCM_FORMAT_S16_LE = 2
    SND_PCM_FORMAT_S24_LE = 6
    SND_PCM_FORMAT_S32_LE = 10

    def __init__(self, device: str = "hw:1,0", rate: int = 16000,
                 channels: int = 1, chunk_size: int = 512):
        self.device = device
        self.target_rate = rate
        self.target_channels = channels
        self.chunk_size = chunk_size  # 帧数（32ms @ 16kHz = 512 帧）
        self.actual_rate = 0
        self.actual_channels = 0

        self._pcm_handle = None
        self._hw_params = None
        self._alsa = None
        self._opened = False

    def _load_lib(self) -> bool:
        if self._alsa is not None:
            return True
        libname = ctypes.util.find_library("asound")
        if not libname:
            print("[AlsaCapture] 找不到 libasound.so")
            return False
        try:
            self._alsa = ctypes.CDLL(libname)
        except OSError as e:
            print(f"[AlsaCapture] 加载 libasound 失败: {e}")
            return False
        return True

    def open(self) -> bool:
        if self._opened:
            return True
        if not self._load_lib():
            return False

        alsa = self._alsa

        # ── 函数签名 ──────────────────────────────────────────────
        alsa.snd_pcm_open.restype = ctypes.c_int
        alsa.snd_pcm_open.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int]

        alsa.snd_pcm_close.restype = ctypes.c_int
        alsa.snd_pcm_close.argtypes = [ctypes.c_void_p]

        alsa.snd_pcm_hw_params_malloc.restype = ctypes.c_int
        alsa.snd_pcm_hw_params_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

        alsa.snd_pcm_hw_params_free.restype = None
        alsa.snd_pcm_hw_params_free.argtypes = [ctypes.c_void_p]

        alsa.snd_pcm_hw_params_any.restype = ctypes.c_int
        alsa.snd_pcm_hw_params_any.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        alsa.snd_pcm_hw_params_set_access.restype = ctypes.c_int
        alsa.snd_pcm_hw_params_set_access.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

        alsa.snd_pcm_hw_params_set_format.restype = ctypes.c_int
        alsa.snd_pcm_hw_params_set_format.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

        alsa.snd_pcm_hw_params_set_rate_near.restype = ctypes.c_int
        alsa.snd_pcm_hw_params_set_rate_near.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_int)]

        alsa.snd_pcm_hw_params_set_channels_near.restype = ctypes.c_int
        alsa.snd_pcm_hw_params_set_channels_near.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint)]

        alsa.snd_pcm_hw_params.restype = ctypes.c_int
        alsa.snd_pcm_hw_params.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        alsa.snd_pcm_prepare.restype = ctypes.c_int
        alsa.snd_pcm_prepare.argtypes = [ctypes.c_void_p]

        alsa.snd_pcm_readi.restype = ctypes.c_long
        alsa.snd_pcm_readi.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]

        alsa.snd_pcm_drain.restype = ctypes.c_int
        alsa.snd_pcm_drain.argtypes = [ctypes.c_void_p]

        alsa.snd_strerror.restype = ctypes.c_char_p
        alsa.snd_strerror.argtypes = [ctypes.c_int]

        # ── 尝试打开设备 ──────────────────────────────────────────
        # 按优先级尝试: plughw → hw → default
        # USB 音频设备（ReSpeaker XVF3800 等）需要 plughw 插件层
        # 处理格式/声道转换，直接 hw 虽然能打开但无法正常输送数据。
        device_candidates = []
        if self.device.startswith("hw:"):
            device_candidates.append(self.device.replace("hw:", "plughw:", 1))
        device_candidates.append(self.device)
        if "default" not in device_candidates:
            device_candidates.append("default")

        pcm = ctypes.c_void_p()
        opened_device = None
        for dev in device_candidates:
            ret = alsa.snd_pcm_open(
                ctypes.byref(pcm), dev.encode("utf-8"),
                self.SND_PCM_STREAM_CAPTURE, 0)
            if ret == 0:
                opened_device = dev
                break
            else:
                err = alsa.snd_strerror(ret)
                print(f"[AlsaCapture] 打开 '{dev}' 失败: {err.decode() if err else ret}")

        if opened_device is None:
            print("[AlsaCapture] 所有 ALSA 设备均无法打开")
            return False

        self._pcm_handle = pcm
        print(f"[AlsaCapture] 已打开设备 '{opened_device}'")

        # ── 配置硬件参数 ──────────────────────────────────────────
        hw = ctypes.c_void_p()
        ret = alsa.snd_pcm_hw_params_malloc(ctypes.byref(hw))
        if ret < 0:
            print(f"[AlsaCapture] hw_params_malloc 失败: {ret}")
            alsa.snd_pcm_close(pcm)
            return False
        self._hw_params = hw

        ret = alsa.snd_pcm_hw_params_any(pcm, hw)
        if ret < 0:
            print(f"[AlsaCapture] hw_params_any 失败: {ret}")
            alsa.snd_pcm_hw_params_free(hw)
            alsa.snd_pcm_close(pcm)
            self._hw_params = None
            self._pcm_handle = None
            return False

        # 访问模式: 交错读写
        alsa.snd_pcm_hw_params_set_access(pcm, hw, self.SND_PCM_ACCESS_RW_INTERLEAVED)

        # 格式: S16_LE
        if alsa.snd_pcm_hw_params_set_format(pcm, hw, self.SND_PCM_FORMAT_S16_LE) < 0:
            print("[AlsaCapture] S16_LE 格式不支持")
            alsa.snd_pcm_hw_params_free(hw)
            alsa.snd_pcm_close(pcm)
            self._hw_params = None
            self._pcm_handle = None
            return False

        # 声道数
        ch = ctypes.c_uint(self.target_channels)
        alsa.snd_pcm_hw_params_set_channels_near(pcm, hw, ctypes.byref(ch))
        self.actual_channels = ch.value

        # 采样率
        rate = ctypes.c_uint(self.target_rate)
        dir_ref = ctypes.c_int(0)
        alsa.snd_pcm_hw_params_set_rate_near(pcm, hw, ctypes.byref(rate), ctypes.byref(dir_ref))
        self.actual_rate = rate.value

        # 应用参数
        ret = alsa.snd_pcm_hw_params(pcm, hw)
        if ret < 0:
            err = alsa.snd_strerror(ret)
            print(f"[AlsaCapture] hw_params 失败: {err.decode() if err else ret}")
            alsa.snd_pcm_hw_params_free(hw)
            alsa.snd_pcm_close(pcm)
            self._hw_params = None
            self._pcm_handle = None
            return False

        print(f"[AlsaCapture] 参数: {self.actual_rate}Hz, {self.actual_channels}ch, S16_LE")

        # 对于立体声设备（NAU8822 是 2ch），记录需要做声道混合
        self._need_channel_mix = (self.actual_channels == 2)

        self._opened = True
        return True

    def read(self) -> np.ndarray | None:
        """读取一帧音频数据。

        Returns:
            float32 numpy 数组 shape=(chunk_size,)，范围 [-1, 1]；出错则返回 None。
        """
        if not self._opened:
            return None

        alsa = self._alsa
        pcm = self._pcm_handle

        # 分配缓冲区: chunk_size 帧 × actual_channels × sizeof(int16)
        buf_size = self.chunk_size * max(self.actual_channels, 1)
        buf = (ctypes.c_int16 * buf_size)()

        frames_read = alsa.snd_pcm_readi(pcm, ctypes.byref(buf), self.chunk_size)

        if frames_read < 0:
            # XRUN 恢复: -EPIPE(-32), -ESTRPIPE(-86)
            if frames_read in (-32, -86):
                ret = alsa.snd_pcm_prepare(pcm)
                if ret < 0:
                    return None
                # 第二次尝试
                frames_read = alsa.snd_pcm_readi(pcm, ctypes.byref(buf), self.chunk_size)
                if frames_read < 0:
                    return None
            elif frames_read == -4:  # -EINTR: 被信号中断，重试一次
                frames_read = alsa.snd_pcm_readi(pcm, ctypes.byref(buf), self.chunk_size)
                if frames_read < 0:
                    return None
            else:
                return None

        if frames_read < self.chunk_size:
            # 部分读取 — 用零填充
            arr = np.zeros(self.chunk_size, dtype=np.float32)
            n = min(frames_read, self.chunk_size)
            if self._need_channel_mix:
                # 立体声 → 单声道: 取左声道
                raw = np.frombuffer(buf, dtype=np.int16, count=n * 2)
                arr[:n] = raw[::2].astype(np.float32) / 32768.0
            else:
                raw = np.frombuffer(buf, dtype=np.int16, count=n)
                arr[:n] = raw.astype(np.float32) / 32768.0
            return arr

        if self._need_channel_mix:
            raw = np.frombuffer(buf, dtype=np.int16, count=frames_read * 2)
            return raw[::2].astype(np.float32) / 32768.0
        else:
            raw = np.frombuffer(buf, dtype=np.int16, count=frames_read)
            return raw.astype(np.float32) / 32768.0

    def close(self):
        if not self._opened:
            return
        alsa = self._alsa
        pcm = self._pcm_handle
        try:
            alsa.snd_pcm_drain(pcm)
        except Exception:
            pass
        try:
            alsa.snd_pcm_close(pcm)
        except Exception:
            pass
        if self._hw_params is not None:
            try:
                alsa.snd_pcm_hw_params_free(self._hw_params)
            except Exception:
                pass
        self._pcm_handle = None
        self._hw_params = None
        self._opened = False
        print("[AlsaCapture] 已关闭")

    @property
    def is_open(self) -> bool:
        return self._opened
