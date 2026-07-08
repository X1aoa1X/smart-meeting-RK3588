# -*- coding: utf-8 -*-
"""Agent LLM Worker — 后台 QThread 用于 LLM 调用。

与 TtsSynthesisWorker 模式一致：
  - 在后台线程中调用 LLM（避免阻塞主 Qt 线程）
  - 通过 pyqtSignal 返回结果
  - 必须存入 self._workers 列表防 GC 崩溃

用法:
    worker = AgentLLMWorker(context, template_text, policy, parent)
    worker.llm_done.connect(on_done)
    worker.llm_error.connect(on_error)
    self._workers.append(worker)
    worker.start()
"""

import logging

from PyQt5.QtCore import QThread, pyqtSignal

from core.agent_llm import AgentLLM

logger = logging.getLogger(__name__)


class AgentLLMWorker(QThread):
    """后台 LLM 调用工作线程。

    单次使用：run() 完成后发射信号，不可复用。
    必须由 AgentWorker 存入 self._workers 防止 GC 崩溃。
    """

    # 成功: (text: str, stats: dict)
    llm_done = pyqtSignal(str, dict)

    # 失败: (error_message: str)
    llm_error = pyqtSignal(str)

    def __init__(self, context: dict, template_text: str,
                 policy: dict | None = None, parent=None):
        super().__init__(parent)
        self._context = context
        self._template_text = template_text
        self._policy = policy

    def run(self):
        """在后台线程中调用 LLM。"""
        try:
            llm = AgentLLM(policy=self._policy)
            text, llm_used, stats = llm.generate_speech(
                context=self._context,
                template_text=self._template_text,
                meeting_id=0,  # meeting_id 仅用于日志
            )
            stats["llm_used"] = llm_used
            self.llm_done.emit(text, stats)
        except Exception as e:
            logger.error(f"AgentLLMWorker error: {e}")
            self.llm_error.emit(str(e))
