# -*- coding: utf-8 -*-
"""Agent LLM 包装器 — 在现有 llm_service 之上增加 TTS 专用约束。

职责:
  - 构建 System Prompt + User Prompt
  - 调用 llm_service.call_deepseek()（5 秒超时）
  - 输出校验（长度、禁用词、多行）
  - 失败 → 返回模板文本

纯 Python，零 Qt 依赖。AgentLLMWorker 负责在 QThread 中调用。
"""

import json
import logging
import re
import time

from core.agent_context import AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# ── 输出禁用词 ──────────────────────────────────────────────────────────
BANNED_WORDS = [
    "根据系统", "Token", "大模型", "我认为", "可能你们", "必须",
    "LLM", "模型", "API", "deepseek", "DeepSeek",
]

# ── 默认最大输出字符数 ──────────────────────────────────────────────────
DEFAULT_MAX_CHARS = 60

# LLM 超时（秒）
LLM_TIMEOUT_SEC = 5.0


def validate_tts_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[bool, str | None]:
    """校验 LLM 输出的 TTS 文本。

    Returns:
        (is_valid, error_reason)
    """
    if not text or not text.strip():
        return False, "empty"

    text = text.strip()

    if len(text) > max_chars:
        return False, f"too_long ({len(text)} > {max_chars})"

    if "\n" in text:
        return False, "multi_line"

    for word in BANNED_WORDS:
        if word in text:
            return False, f"banned_word: {word}"

    return True, None


def estimate_tokens(text: str) -> int:
    """粗略估计 token 数（中文 ~1.5 字/token，英文 ~4 字/token）。"""
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


class AgentLLM:
    """Agent TTS 专用 LLM 包装器。

    用法:
        llm = AgentLLM(policy)
        text, used_llm, stats = llm.generate_speech(
            context={"meeting": {...}, "current_speaker": {...}, "policy": {...}},
            template_text="张三已连续发言三分钟，建议进入总结。",
            meeting_id=12,
        )
    """

    def __init__(self, policy: dict | None = None):
        self._policy = policy or {}
        self._speech_cfg = self._policy.get("speech", {})

    def is_available(self) -> bool:
        """检查 LLM 服务是否可用。"""
        try:
            from core.llm_service import LLMService
            return LLMService.is_available()
        except Exception:
            return False

    def generate_speech(
        self,
        context: dict,
        template_text: str,
        meeting_id: int,
    ) -> tuple[str, bool, dict]:
        """生成 TTS 播报文本。

        Args:
            context: AgentContextBuilder.build() 的输出
            template_text: LLM 不可用时的 fallback 模板
            meeting_id: 会议 ID

        Returns:
            (text, llm_used, stats_dict)
            text: 播报文本
            llm_used: 是否实际调用了 LLM
            stats_dict: {"prompt_tokens": int, "completion_tokens": int, "duration_ms": int}
        """
        max_chars = self._speech_cfg.get("max_chars", DEFAULT_MAX_CHARS)

        # 检查 LLM 可用性
        if not self.is_available():
            return template_text, False, {"reason": "llm_unavailable"}

        # 构建 user prompt
        try:
            context_json = json.dumps(context, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return template_text, False, {"reason": "context_serialization_error"}

        # 限制输入大小
        if len(context_json) > 1200:
            context_json = context_json[:1200]

        user_prompt = f"请根据以下 JSON 生成 TTS 播报：\n\n{context_json}"

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_tokens = estimate_tokens(AGENT_SYSTEM_PROMPT) + estimate_tokens(user_prompt)

        try:
            from core.llm_service import call_deepseek

            t0 = time.time()
            raw_output = call_deepseek(
                messages=messages,
                temperature=0.2,
                max_tokens=128,       # TTS 播报很短，不需要很多 token
                timeout=LLM_TIMEOUT_SEC,
            )
            duration_ms = int((time.time() - t0) * 1000)

            completion_tokens = estimate_tokens(raw_output)

            # 校验输出
            is_valid, error = validate_tts_text(raw_output, max_chars)
            if is_valid:
                return raw_output.strip(), True, {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "duration_ms": duration_ms,
                }
            else:
                logger.warning(f"LLM output validation failed: {error} → fallback to template")
                return template_text, True, {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "duration_ms": duration_ms,
                    "validation_error": error,
                }

        except Exception as e:
            logger.warning(f"LLM call failed: {e} → fallback to template")
            return template_text, False, {"reason": f"llm_error: {e}"}
