# -*- coding: utf-8 -*-
"""本地音频播放器 — 基于 pygame mixer / aplay 的非阻塞 PCM/WAV 播放。

在 RK3588 上通过 ALSA 硬件设备直接播放（绕过 PulseAudio），
与 VAD 采集 (hw:1,0) 可并存。

关键: 使用 plughw:1,0 (NAU8822) 而非 "default"，
因为 root 用户无法访问 PulseAudio daemon (XDG_RUNTIME_DIR 权限错误)，
导致 "default" → pulse → 静默丢弃音频数据。

播放优先级 (与 demos/tts_demo.py 一致，均在 RK3588 验证):
  1. pygame.mixer.Sound(filepath) — 从文件加载，非阻塞
  2. subprocess aplay — 系统级 ALSA 直接播放

用法:
  player = AudioPlayer()
  player.init()
  player.play_pcm(pcm_bytes, sample_rate=16000)
"""

import os
import io
import wave
import subprocess
import tempfile
import threading
import logging

logger = logging.getLogger(__name__)

# ── ALSA 播放设备: NAU8822 板载 codec (plughw 层支持采样率/格式转换) ──
# card 1: rockchipnau8822 → plughw:1,0
# 不使用 "default" — root 用户下 PulseAudio 不可用
ALSA_PLAYBACK_DEVICE = "plughw:1,0"

# 后台清理线程追踪 (防 GC)
_cleanup_threads: list = []


def _cleanup_temp_file(tmp_path: str, proc: subprocess.Popen | None = None):
    """后台线程: 等待 aplay 结束，删除临时文件。"""
    if proc is not None:
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass
    try:
        _cleanup_threads.remove(threading.current_thread())
    except ValueError:
        pass


def _wait_and_cleanup_pygame_file(filepath: str):
    """后台线程: 等待 pygame 播放结束，删除临时文件。"""
    import pygame
    import time
    # 等待 pygame 开始播放（最多 500ms）
    for _ in range(10):
        if pygame.mixer.get_busy():
            break
        time.sleep(0.05)
    # 等待播放结束（最多 30s）
    for _ in range(600):
        if not pygame.mixer.get_busy():
            break
        time.sleep(0.05)
    time.sleep(0.1)
    try:
        os.unlink(filepath)
    except OSError:
        pass
    try:
        _cleanup_threads.remove(threading.current_thread())
    except ValueError:
        pass


class AudioPlayer:
    """本地音频播放器 — 文件路径加载 + 硬件设备直接播放。

    play_pcm() / play_wav_bytes() 均为非阻塞。
    所有 WAV 先写临时文件，再通过 pygame.Sound(filepath) 或 aplay 播放。
    """

    def __init__(self):
        self._initialized = False
        self._available = False
        self._aplay_available = self._check_aplay()
        self._device = self._detect_device()
        # 追踪当前 aplay 子进程 (pygame 不可用时的回退)，用于 is_busy()
        self._aplay_proc: subprocess.Popen | None = None
        self._aplay_lock = threading.Lock()

    # ── 初始化 ──────────────────────────────────────────────────────────

    def init(self, frequency: int = 16000) -> bool:
        """初始化 pygame mixer。使用 ALSA 硬件设备 (非 "default"/PulseAudio)。

        环境变量在 import pygame 之前设置，告知 SDL2 使用直连 ALSA:
          SDL_AUDIODRIVER=alsa   → 强制 ALSA 驱动 (不用 PulseAudio)
          AUDIODEV=plughw:1,0    → 指定 NAU8822 硬件设备
        """
        if self._initialized:
            return self._available

        self._initialized = True
        try:
            # ★ 关键: 必须在 import pygame 之前设置，SDL2 在初始化时读取
            os.environ.setdefault("SDL_AUDIODRIVER", "alsa")
            os.environ["AUDIODEV"] = self._device

            import pygame
            pygame.mixer.init(
                frequency=frequency,
                size=-16,
                channels=1,
            )
            self._available = True
            logger.info(
                "AudioPlayer: pygame mixer 初始化成功 "
                "(SDL_AUDIODRIVER=alsa, AUDIODEV=%s, frequency=%d)",
                self._device, frequency)
            return True
        except Exception as e:
            logger.warning(f"AudioPlayer: pygame mixer 初始化失败: {e}")
            self._available = False
            if self._aplay_available:
                logger.info("AudioPlayer: 将使用 aplay 回退 (device=%s)", self._device)
            return False

    def is_available(self) -> bool:
        if not self._initialized:
            self.init()
        return self._available or self._aplay_available

    # ── 播放 ────────────────────────────────────────────────────────────

    def play_pcm(self, pcm_data: bytes, sample_rate: int = 16000) -> bool:
        """播放原始 PCM 16-bit 单声道音频 (非阻塞)。"""
        if not pcm_data:
            return False
        try:
            wav_bytes = self._pcm_to_wav_bytes(pcm_data, sample_rate)
        except Exception as e:
            logger.error(f"AudioPlayer: PCM→WAV 转换失败: {e}")
            return False
        return self._play_wav(wav_bytes)

    def play_wav_bytes(self, wav_data: bytes) -> bool:
        """播放 WAV 字节 (已含 WAV 头)。"""
        if not wav_data:
            return False
        return self._play_wav(wav_data)

    def _play_wav(self, wav_data: bytes) -> bool:
        """写入临时文件 → 按优先级播放。"""
        # 1. 写入临时 WAV 文件
        try:
            fd, filepath = tempfile.mkstemp(suffix=".wav", prefix="meeting_tts_")
            with os.fdopen(fd, "wb") as f:
                f.write(wav_data)
        except Exception as e:
            logger.error(f"AudioPlayer: 临时文件创建失败: {e}")
            return False

        # 2. 尝试 pygame 播放 (文件路径, 与 tts_demo.py 一致)
        if self._available:
            try:
                ok = self._play_via_pygame_file(filepath)
                if ok:
                    t = threading.Thread(
                        target=_wait_and_cleanup_pygame_file,
                        args=(filepath,), daemon=True)
                    _cleanup_threads.append(t)
                    t.start()
                    return True
                logger.warning("AudioPlayer: pygame 播放失败，尝试 aplay 回退")
            except Exception as e:
                logger.warning(f"AudioPlayer: pygame 异常: {e}，尝试 aplay 回退")

        # 3. 降级到 aplay (指定硬件设备)
        if self._aplay_available:
            try:
                return self._play_via_aplay(filepath)
            except Exception as e:
                logger.error(f"AudioPlayer: aplay 回退也失败: {e}")
                try:
                    os.unlink(filepath)
                except OSError:
                    pass
                return False

        try:
            os.unlink(filepath)
        except OSError:
            pass
        logger.error("AudioPlayer: 无可用播放方式")
        return False

    # ── pygame 播放 ─────────────────────────────────────────────────────

    @staticmethod
    def _play_via_pygame_file(filepath: str) -> bool:
        """通过 pygame.mixer.Sound(filepath) 播放 — 与 tts_demo.py 一致。"""
        import pygame
        try:
            sound = pygame.mixer.Sound(filepath)
        except pygame.error as e:
            logger.error(f"AudioPlayer: pygame.Sound({filepath!r}) 失败: {e}")
            return False

        channel = sound.play()
        if channel is None:
            logger.warning("AudioPlayer: pygame mixer 所有通道忙，无法播放")
            return False
        logger.info("AudioPlayer: pygame 播放已启动 (%s)", filepath)
        return True

    # ── aplay 播放 ──────────────────────────────────────────────────────

    def _play_via_aplay(self, filepath: str) -> bool:
        """通过 aplay -D <device> 播放 WAV 文件 (非阻塞)。"""
        try:
            proc = subprocess.Popen(
                ["aplay", "-q", "-D", self._device, filepath],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # 存储进程引用，供 is_busy() 查询
            with self._aplay_lock:
                self._aplay_proc = proc
            t = threading.Thread(
                target=self._cleanup_aplay,
                args=(filepath, proc), daemon=True)
            _cleanup_threads.append(t)
            t.start()
            logger.info(
                "AudioPlayer: aplay 播放已启动 (device=%s, %s)",
                self._device, filepath)
            return True
        except FileNotFoundError:
            logger.error("AudioPlayer: aplay 命令未找到")
            try:
                os.unlink(filepath)
            except OSError:
                pass
            return False
        except Exception as e:
            logger.error(f"AudioPlayer: aplay 播放失败: {e}")
            try:
                os.unlink(filepath)
            except OSError:
                pass
            return False

    def _cleanup_aplay(self, tmp_path: str, proc: subprocess.Popen):
        """后台线程: 等待 aplay 结束，清理临时文件和进程引用。"""
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        finally:
            with self._aplay_lock:
                if self._aplay_proc is proc:
                    self._aplay_proc = None
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        try:
            _cleanup_threads.remove(threading.current_thread())
        except ValueError:
            pass

    @staticmethod
    def _check_aplay() -> bool:
        try:
            result = subprocess.run(
                ["which", "aplay"],
                capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    # ── 设备检测 ────────────────────────────────────────────────────────

    @staticmethod
    def _detect_device() -> str:
        """检测最佳 ALSA 播放设备。

        优先 plughw:1,0 (NAU8822)，通过 /proc/asound/card1 验证存在性，
        回退到 "default"。
        """
        # 检查 NAU8822 声卡是否存在
        if os.path.isdir("/proc/asound/card1"):
            logger.info(
                "AudioPlayer: 检测到 NAU8822 (card1) → 使用 %s", ALSA_PLAYBACK_DEVICE)
            return ALSA_PLAYBACK_DEVICE

        logger.warning(
            "AudioPlayer: card1 (NAU8822) 未检测到，回退到 'default'")
        return "default"

    # ── 状态 ────────────────────────────────────────────────────────────

    def is_busy(self) -> bool:
        """检查是否有音频正在播放。

        同时检查 pygame mixer 和 aplay 回退，防止 COOLDOWN 提前结束。
        """
        # 1. 检查 aplay 回退进程 (subprocess，不依赖 pygame)
        with self._aplay_lock:
            if self._aplay_proc is not None:
                poll_result = self._aplay_proc.poll()
                if poll_result is None:
                    return True  # aplay 仍在运行
                # 已退出但引用未清理 → 清理
                self._aplay_proc = None

        # 2. 检查 pygame mixer
        if self._available:
            try:
                import pygame
                return pygame.mixer.get_busy()
            except Exception:
                pass
        return False

    def stop(self):
        """停止所有播放 (pygame mixer + aplay 回退)。"""
        if self._available:
            try:
                import pygame
                pygame.mixer.stop()
            except Exception:
                pass
        with self._aplay_lock:
            if self._aplay_proc is not None:
                try:
                    self._aplay_proc.terminate()
                except Exception:
                    pass
                self._aplay_proc = None

    # ── 工具 ────────────────────────────────────────────────────────────

    @staticmethod
    def _pcm_to_wav_bytes(pcm_data: bytes, sample_rate: int = 16000,
                           channels: int = 1, sampwidth: int = 2) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()
