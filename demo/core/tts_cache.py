# -*- coding: utf-8 -*-
"""TTS 播报缓存 — 内存 + 磁盘双层缓存。

缓存 key = 播报文本的 SHA256 哈希，避免中文文件名编码问题。
磁盘格式: WAV 文件 (含完整 WAV 头)，可直接用 pygame 加载。

预置播报:
  - 会议已开始 / 会议已结束
  - 已锁定当前发言人 / 已解锁发言人
  - 目标丢失，正在重新搜索

用法:
  cache = TTSCache(cache_dir="data/tts_cache")
  cache.preload_from_disk()

  # 写入
  cache.put("会议已开始", pcm_bytes, sample_rate=16000)

  # 读取
  pcm = cache.get("会议已开始")  # -> bytes or None
"""

import os
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


class TTSCache:
    """双层 TTS 播报缓存: 内存 dict + 磁盘 WAV 文件。"""

    CACHE_DIR_DEFAULT = "data/tts_cache"
    INDEX_FILE = "index.json"

    # 启动时应预合成的播报文本列表
    # ★ 必须与 tts_templates.py 中的实际模板文本完全一致，否则缓存命中失败
    PRELOAD_TEXTS = [
        # ── 会议生命周期（高频，优先预加载） ──
        "会议已开始，系统将自动跟踪当前发言人。",
        "会议已结束，正在保存会议记录。",
        # ── 导播控制 ──
        "已锁定当前发言人。",
        "已解锁发言人。",
        "目标丢失，正在重新搜索。",
        "自动跟踪已开启。",
        "自动跟踪已暂停。",
        "已切换到手动指定发言人。",
        # ── 发言人身份 ──
        "身份识别暂时丢失，正在等待重新确认。",
        "当前身份识别暂时丢失，系统将继续保持视觉跟踪。",
        # ── 静默提醒 ──
        "当前讨论暂停了一会儿，可以进入下一议题或请成员补充。",
        "会议暂时无人发言。",
        # ── 系统异常 ──
        "摄像头暂时不可用，请检查视频输入。",
        "音频采集异常，请检查麦克风连接。",
        "语音播报服务暂时不可用。",
        "系统出现异常，请检查设备连接。",
        # ── 手动触发 ──
        "系统运行正常，正在跟踪当前发言人。",
        "检测到一条可能的待办事项，已记录到会议备注。",
    ]

    def __init__(self, cache_dir: str | None = None):
        self._cache_dir = cache_dir or self.CACHE_DIR_DEFAULT
        self._memory: dict[str, bytes] = {}  # text → PCM bytes
        os.makedirs(self._cache_dir, exist_ok=True)

    # ── 读取 ────────────────────────────────────────────────────────────

    def get(self, text: str) -> bytes | None:
        """查找缓存的 PCM 音频数据 (无 WAV 头)。"""
        # 内存优先
        if text in self._memory:
            return self._memory[text]
        # 磁盘回退
        pcm = self._load_from_disk(text)
        if pcm is not None:
            self._memory[text] = pcm
        return pcm

    def has(self, text: str) -> bool:
        return text in self._memory or self._disk_exists(text)

    # ── 写入 ────────────────────────────────────────────────────────────

    def put(self, text: str, pcm_data: bytes, sample_rate: int = 16000):
        """写入 PCM 数据到缓存 (内存 + 磁盘)。"""
        self._memory[text] = pcm_data
        self._save_to_disk(text, pcm_data, sample_rate)

    # ── 预加载 ──────────────────────────────────────────────────────────

    def preload_from_disk(self) -> list[str]:
        """从磁盘加载所有缓存条目到内存。返回已加载的文本列表。"""
        index = self._read_index()
        loaded = []
        for text, entry in index.get("entries", {}).items():
            try:
                pcm = self._load_wav_pcm(
                    os.path.join(self._cache_dir, entry["file"]))
                if pcm is not None:
                    self._memory[text] = pcm
                    loaded.append(text)
            except Exception as e:
                logger.warning(f"TTSCache: 加载 {text} 失败: {e}")
        logger.info(
            f"TTSCache: 从磁盘加载 {len(loaded)}/{len(index.get('entries', {}))} 条")
        return loaded

    def get_missing_preloads(self) -> list[str]:
        """返回 PRELOAD_TEXTS 中尚未缓存的文本列表。"""
        return [t for t in self.PRELOAD_TEXTS if not self.has(t)]

    def all_cached_count(self) -> int:
        """已缓存条目数 (含 PRELOAD_TEXTS 之外的用户文本)。"""
        index = self._read_index()
        return len(index.get("entries", {}))

    # ── 磁盘 I/O ────────────────────────────────────────────────────────

    def _key_for(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _disk_exists(self, text: str) -> bool:
        index = self._read_index()
        return text in index.get("entries", {})

    def _load_from_disk(self, text: str) -> bytes | None:
        index = self._read_index()
        entry = index.get("entries", {}).get(text)
        if not entry:
            return None
        filepath = os.path.join(self._cache_dir, entry["file"])
        return self._load_wav_pcm(filepath)

    @staticmethod
    def _load_wav_pcm(filepath: str) -> bytes | None:
        """从 WAV 文件读取 PCM 数据。"""
        import wave
        try:
            with wave.open(filepath, "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception as e:
            logger.warning(f"TTSCache: 读取 WAV 失败 {filepath}: {e}")
            return None

    def _save_to_disk(self, text: str, pcm_data: bytes, sample_rate: int):
        """写入 WAV 文件到磁盘并更新索引。"""
        import wave
        filename = f"{self._key_for(text)}.wav"
        filepath = os.path.join(self._cache_dir, filename)

        try:
            with wave.open(filepath, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm_data)

            index = self._read_index()
            if "entries" not in index:
                index["entries"] = {}
            index["entries"][text] = {
                "file": filename,
                "sample_rate": sample_rate,
                "bytes": len(pcm_data),
            }
            self._write_index(index)
            logger.info(f"TTSCache: 已缓存 '{text}' → {filename}")
        except Exception as e:
            logger.warning(f"TTSCache: 写入磁盘失败 {text}: {e}")

    def _read_index(self) -> dict:
        index_path = os.path.join(self._cache_dir, self.INDEX_FILE)
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _write_index(self, index: dict):
        index_path = os.path.join(self._cache_dir, self.INDEX_FILE)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
