"""Control API Server — 轻量级 HTTP API 服务器，在 fusion_tracker 后台线程中运行。

提供 REST API 供 Streamlit 主持人控制台调用。使用 Python stdlib
http.server + ThreadingMixIn，零额外依赖。

线程安全设计:
  - 状态共享: update_state() + get_snapshot() 受 threading.Lock 保护
  - 命令队列: send_command() 放入 queue.Queue，主线程 poll_commands() 取出执行
  - DB 访问: GET /api/events 和 POST /api/host_note 在 HTTP 线程中
    直接使用 session_scope() — SQLite WAL 模式支持并发读写

用法:
  from core.control_api import ControlApiServer

  server = ControlApiServer(host="127.0.0.1", port=8800)
  server.start()

  # 主线程每个 tick (100ms):
  server.update_state({...})          # 更新共享状态
  for cmd in server.poll_commands():  # 处理来自 Streamlit 的命令
      dispatch(cmd)

  # 退出时:
  server.stop()
"""

import copy
import json
import queue
import threading
import time
import traceback
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from typing import Callable


# ═══════════════════════════════════════════════════════════════════════════════
# ThreadingHTTPServer
# ═══════════════════════════════════════════════════════════════════════════════

class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """多线程 HTTP 服务器 — 每个请求在独立线程中处理。"""
    daemon_threads = True  # 主线程退出时自动清理
    allow_reuse_address = True


# ═══════════════════════════════════════════════════════════════════════════════
# Request Handler
# ═══════════════════════════════════════════════════════════════════════════════

class _APIHandler(BaseHTTPRequestHandler):
    """HTTP 请求处理器 — 路由到 ControlApiServer 的方法。"""

    # 由 ControlApiServer 在创建时注入
    server_ref: "ControlApiServer | None" = None

    def log_message(self, format, *args):
        """抑制默认的 stderr 日志 — 由 ControlApiServer 统一管理。"""
        pass

    # ── 路由分发 ──────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = parse_qs(parsed.query)

        if path == "/api/status":
            self._handle_status()
        elif path == "/api/events":
            self._handle_get_events(params)
        elif path == "/api/audio/devices":
            self._handle_audio_devices()
        elif path == "/api/llm/status":
            self._handle_llm_status()
        elif path == "/api/agent/decisions":
            self._handle_agent_decisions(params)
        elif path == "/api/agent/tts_events":
            self._handle_agent_tts_events(params)
        elif path == "/api/agent/status":
            self._handle_agent_status()
        else:
            self._send_error(404, f"Unknown GET endpoint: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # 读取 JSON body
        body = None
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            try:
                raw = self.rfile.read(content_length)
                body = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                self._send_error(400, f"Invalid JSON body: {e}")
                return

        # 路由
        if path == "/api/meeting/start":
            self._handle_meeting_start(body)
        elif path == "/api/meeting/end":
            self._handle_meeting_end(body)
        elif path == "/api/meeting/pause":
            self._handle_meeting_pause(body)
        elif path == "/api/meeting/resume":
            self._handle_meeting_resume(body)
        elif path == "/api/control/recenter":
            self._handle_control("recenter")
        elif path == "/api/control/start_tracking":
            self._handle_control("start_tracking")
        elif path == "/api/control/stop_tracking":
            self._handle_control("stop_tracking")
        elif path == "/api/control/lock_speaker":
            self._handle_control("lock_speaker")
        elif path == "/api/control/unlock_speaker":
            self._handle_control("unlock_speaker")
        elif path == "/api/control/manual_speaker":
            self._handle_manual_speaker(body)
        elif path == "/api/control/set_overlay":
            self._handle_set_overlay(body)
        elif path == "/api/control/start_stream":
            self._handle_control("start_stream")
        elif path == "/api/control/stop_stream":
            self._handle_control("stop_stream")
        elif path == "/api/control/set_vad_device":
            self._handle_set_vad_device(body)
        elif path == "/api/host_note":
            self._handle_host_note(body)
        elif path == "/api/tts/test":
            self._handle_control("tts_test")
        elif path == "/api/llm/chat":
            self._handle_llm_chat(body)
        elif path == "/api/llm/summary":
            self._handle_llm_summary(body)
        elif path == "/api/llm/action_items":
            self._handle_llm_action_items(body)
        elif path == "/api/llm/diagnose":
            self._handle_llm_diagnose(body)
        elif path == "/api/agent/summarize":
            self._handle_agent_summarize(body)
        elif path == "/api/agent/agenda":
            self._handle_agent_agenda(body)
        elif path == "/api/agent/status":
            self._handle_agent_trigger_status(body)
        elif path == "/api/agent/custom_tts":
            self._handle_agent_custom_tts(body)
        elif path == "/api/agent/config":
            self._handle_agent_config(body)
        else:
            self._send_error(404, f"Unknown POST endpoint: {path}")

    # ── GET handlers ──────────────────────────────────────────────────────

    def _handle_status(self):
        """GET /api/status — 返回完整系统状态快照。"""
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        snapshot = self.server_ref.get_snapshot()
        self._send_ok(snapshot)

    def _handle_audio_devices(self):
        """GET /api/audio/devices — 返回可用的 ALSA 录音设备列表。"""
        try:
            from core.alsa_device_list import list_capture_devices
            devices = list_capture_devices()
            self._send_ok({"devices": devices})
        except Exception as e:
            self._send_error(500, f"Failed to enumerate ALSA devices: {e}")

    def _handle_get_events(self, params: dict):
        """GET /api/events?meeting_id=N&minutes=M — 查询事件日志。"""
        meeting_id_str = params.get("meeting_id", [None])[0]
        minutes_str = params.get("minutes", ["30"])[0]

        meeting_id = None
        if meeting_id_str:
            try:
                meeting_id = int(meeting_id_str)
            except ValueError:
                self._send_error(400, f"Invalid meeting_id: {meeting_id_str}")
                return

        try:
            minutes = float(minutes_str)
        except ValueError:
            self._send_error(400, f"Invalid minutes: {minutes_str}")
            return

        try:
            from storage.db import session_scope
            from storage.repo import EventRepo, HostNoteRepo

            with session_scope() as session:
                event_repo = EventRepo(session)
                events = event_repo.get_recent(
                    minutes=minutes,
                    meeting_id=meeting_id,
                )
                note_repo = HostNoteRepo(session)
                if meeting_id is not None:
                    notes = note_repo.get_notes_for_meeting(meeting_id)
                else:
                    notes = []

            events_data = []
            for evt in events:
                payload = None
                if evt.payload_json:
                    try:
                        payload = json.loads(evt.payload_json)
                    except json.JSONDecodeError:
                        payload = {"raw": evt.payload_json}
                events_data.append({
                    "id": evt.id,
                    "event_type": evt.event_type,
                    "timestamp": evt.timestamp.isoformat() if evt.timestamp else None,
                    "meeting_id": evt.meeting_id,
                    "payload": payload,
                })

            notes_data = []
            for note in notes:
                notes_data.append({
                    "id": note.id,
                    "note_type": note.note_type,
                    "content": note.content,
                    "related_speaker": note.related_speaker,
                    "timestamp": note.timestamp.isoformat() if note.timestamp else None,
                    "meeting_id": note.meeting_id,
                })

            self._send_ok({
                "events": events_data,
                "notes": notes_data,
            })
        except Exception as e:
            self._send_error(500, f"DB query failed: {e}")

    # ── POST handlers ─────────────────────────────────────────────────────

    def _handle_meeting_start(self, body: dict | None):
        """POST /api/meeting/start — 开始会议。"""
        if not body or "meeting_id" not in body:
            self._send_error(400, "Missing 'meeting_id' in request body")
            return

        meeting_id = body["meeting_id"]
        if not isinstance(meeting_id, int):
            self._send_error(400, "'meeting_id' must be an integer")
            return

        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return

        if not self.server_ref.send_command("meeting_start", meeting_id=meeting_id):
            self._send_error(503, "Command queue full")
            return

        self._send_ok({"message": f"Meeting start command queued (id={meeting_id})"})

    def _handle_meeting_end(self, body: dict | None):
        """POST /api/meeting/end — 结束会议。"""
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("meeting_end"):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Meeting end command queued"})

    def _handle_meeting_pause(self, body: dict | None):
        """POST /api/meeting/pause — 暂停追踪。"""
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("meeting_pause"):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Meeting pause command queued"})

    def _handle_meeting_resume(self, body: dict | None):
        """POST /api/meeting/resume — 恢复追踪。"""
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("meeting_resume"):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Meeting resume command queued"})

    def _handle_control(self, cmd_type: str):
        """POST /api/control/* — 通用导播控制命令。"""
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command(cmd_type):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": f"Command '{cmd_type}' queued"})

    def _handle_manual_speaker(self, body: dict | None):
        """POST /api/control/manual_speaker — 手动指定发言人。"""
        if not body or "tag_id" not in body:
            self._send_error(400, "Missing 'tag_id' in request body")
            return
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("manual_speaker", tag_id=body["tag_id"]):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": f"Manual speaker override queued (tag_id={body['tag_id']})"})

    def _handle_set_overlay(self, body: dict | None):
        """POST /api/control/set_overlay — 设置名片叠加。"""
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        params = {}
        if body:
            if "enabled" in body:
                params["enabled"] = bool(body["enabled"])
            if "show_debug" in body:
                params["show_debug"] = bool(body["show_debug"])
        if not self.server_ref.send_command("set_overlay", **params):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": f"Overlay settings queued: {params}"})

    def _handle_set_vad_device(self, body: dict | None):
        """POST /api/control/set_vad_device — 切换 VAD 录音设备。"""
        if not body or "device" not in body:
            self._send_error(400, "Missing 'device' in request body")
            return
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("set_vad_device", device=body["device"]):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": f"VAD device switch queued: {body['device']}"})

    def _handle_host_note(self, body: dict | None):
        """POST /api/host_note — 添加主持人备注 (直接写 DB)。"""
        if not body:
            self._send_error(400, "Missing request body")
            return

        required = ["meeting_id", "note_type", "content"]
        for field in required:
            if field not in body:
                self._send_error(400, f"Missing required field: '{field}'")
                return

        try:
            from storage.db import session_scope
            from storage.repo import HostNoteRepo

            with session_scope() as session:
                repo = HostNoteRepo(session)
                note = repo.create_note(
                    meeting_id=body["meeting_id"],
                    note_type=body["note_type"],
                    content=body["content"],
                    related_speaker=body.get("related_speaker"),
                    timestamp=datetime.utcnow(),
                )
                note_id = note.id

            # 发布事件到 EventBus（供 EventBridge 写入 events 表）
            try:
                from core.event_bus import EventBus
                EventBus().publish("host_note_added",
                                   meeting_id=body["meeting_id"],
                                   note_id=note_id,
                                   note_type=body["note_type"],
                                   related_speaker=body.get("related_speaker"),
                                   content=body["content"])
            except Exception:
                pass  # EventBus 发布失败不影响 DB 写入

            self._send_ok({"message": "Host note created", "note_id": note_id})
        except Exception as e:
            self._send_error(500, f"Failed to create host note: {e}")

    # ── LLM handlers ────────────────────────────────────────────────────────

    def _handle_llm_status(self):
        """GET /api/llm/status — 检查 LLM 服务是否可用。"""
        try:
            from core.llm_service import LLMService
            available = LLMService.is_available()
            self._send_ok({
                "available": available,
                "model": "deepseek-v4-flash",
            })
        except Exception as e:
            self._send_error(500, f"LLM status check failed: {e}")

    def _handle_llm_chat(self, body: dict | None):
        """POST /api/llm/chat — 会中问答。

        Body: {"meeting_id": int, "question": str}
        """
        if not body:
            self._send_error(400, "Missing request body")
            return

        meeting_id = body.get("meeting_id")
        question = body.get("question", "").strip()

        if not isinstance(meeting_id, int):
            self._send_error(400, "'meeting_id' must be an integer")
            return
        if not question:
            self._send_error(400, "'question' is required and must not be empty")
            return

        # 验证会议存在
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo
            with session_scope() as session:
                meeting = MeetingRepo(session).get_by_id(meeting_id)
                if meeting is None:
                    self._send_error(404, f"Meeting not found: id={meeting_id}")
                    return
        except Exception as e:
            self._send_error(500, f"DB query failed: {e}")
            return

        # 获取状态快照（如果可用）
        status_snapshot = None
        if self.server_ref is not None:
            try:
                status_snapshot = self.server_ref.get_snapshot()
            except Exception:
                pass

        # 调用 LLM
        try:
            from core.llm_service import LLMService
            result = LLMService.ask_question(meeting_id, question, status_snapshot)
            if result.get("ok"):
                self._send_ok({
                    "answer": result["answer"],
                    "meeting_id": meeting_id,
                })
            else:
                self._send_error(500, result.get("error", "LLM call failed"))
        except Exception as e:
            self._send_error(500, f"LLM service error: {e}")

    def _handle_llm_summary(self, body: dict | None):
        """POST /api/llm/summary — 生成会后摘要。

        Body: {"meeting_id": int}
        """
        if not body:
            self._send_error(400, "Missing request body")
            return

        meeting_id = body.get("meeting_id")
        if not isinstance(meeting_id, int):
            self._send_error(400, "'meeting_id' must be an integer")
            return

        # 验证会议存在
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo
            with session_scope() as session:
                meeting = MeetingRepo(session).get_by_id(meeting_id)
                if meeting is None:
                    self._send_error(404, f"Meeting not found: id={meeting_id}")
                    return
        except Exception as e:
            self._send_error(500, f"DB query failed: {e}")
            return

        try:
            from core.llm_service import LLMService
            result = LLMService.generate_summary(meeting_id)
            if result.get("ok"):
                self._send_ok({
                    "summary": result["summary"],
                    "meeting_id": meeting_id,
                })
            else:
                self._send_error(500, result.get("error", "LLM call failed"))
        except Exception as e:
            self._send_error(500, f"LLM service error: {e}")

    def _handle_llm_action_items(self, body: dict | None):
        """POST /api/llm/action_items — 提取待办事项。

        Body: {"meeting_id": int}
        """
        if not body:
            self._send_error(400, "Missing request body")
            return

        meeting_id = body.get("meeting_id")
        if not isinstance(meeting_id, int):
            self._send_error(400, "'meeting_id' must be an integer")
            return

        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo
            with session_scope() as session:
                meeting = MeetingRepo(session).get_by_id(meeting_id)
                if meeting is None:
                    self._send_error(404, f"Meeting not found: id={meeting_id}")
                    return
        except Exception as e:
            self._send_error(500, f"DB query failed: {e}")
            return

        try:
            from core.llm_service import LLMService
            result = LLMService.extract_action_items(meeting_id)
            if result.get("ok"):
                self._send_ok({
                    "action_items": result["action_items"],
                    "meeting_id": meeting_id,
                })
            else:
                self._send_error(500, result.get("error", "LLM call failed"))
        except Exception as e:
            self._send_error(500, f"LLM service error: {e}")

    def _handle_llm_diagnose(self, body: dict | None):
        """POST /api/llm/diagnose — 系统状态诊断。

        无需 body 参数，从 ControlApiServer 获取当前状态快照。
        """
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return

        try:
            status_snapshot = self.server_ref.get_snapshot()
        except Exception as e:
            self._send_error(500, f"Failed to get status snapshot: {e}")
            return

        try:
            from core.llm_service import LLMService
            result = LLMService.diagnose(status_snapshot)
            if result.get("ok"):
                self._send_ok({"diagnosis": result["diagnosis"]})
            else:
                self._send_error(500, result.get("error", "LLM call failed"))
        except Exception as e:
            self._send_error(500, f"LLM service error: {e}")

    # ── Agent handlers ─────────────────────────────────────────────────────

    def _handle_agent_decisions(self, params: dict):
        """GET /api/agent/decisions?meeting_id=N&limit=50 — 查询 Agent 决策。"""
        meeting_id_str = params.get("meeting_id", [None])[0]
        limit_str = params.get("limit", ["50"])[0]

        meeting_id = None
        if meeting_id_str:
            try:
                meeting_id = int(meeting_id_str)
            except ValueError:
                self._send_error(400, f"Invalid meeting_id: {meeting_id_str}")
                return

        try:
            limit = int(limit_str)
        except ValueError:
            limit = 50

        try:
            from storage.db import session_scope
            from storage.repo import AgentDecisionRepo

            with session_scope() as session:
                repo = AgentDecisionRepo(session)
                decisions = repo.get_recent(meeting_id=meeting_id, limit=limit)

            data = []
            for d in decisions:
                data.append({
                    "id": d.id,
                    "meeting_id": d.meeting_id,
                    "trigger_type": d.trigger_type,
                    "trigger_key": d.trigger_key,
                    "priority": d.priority,
                    "rule_reason": d.rule_reason,
                    "llm_used": d.llm_used,
                    "llm_prompt_tokens": d.llm_prompt_tokens,
                    "llm_completion_tokens": d.llm_completion_tokens,
                    "decision": d.decision,
                    "final_text": d.final_text,
                    "suppressed_reason": d.suppressed_reason,
                    "created_at": d.created_at.isoformat() if d.created_at else None,
                })
            self._send_ok({"decisions": data})
        except Exception as e:
            self._send_error(500, f"Query failed: {e}")

    def _handle_agent_tts_events(self, params: dict):
        """GET /api/agent/tts_events?meeting_id=N&limit=50 — 查询 TTS 播报事件。"""
        meeting_id_str = params.get("meeting_id", [None])[0]
        limit_str = params.get("limit", ["50"])[0]

        meeting_id = None
        if meeting_id_str:
            try:
                meeting_id = int(meeting_id_str)
            except ValueError:
                self._send_error(400, f"Invalid meeting_id: {meeting_id_str}")
                return

        try:
            limit = int(limit_str)
        except ValueError:
            limit = 50

        try:
            from storage.db import session_scope
            from storage.repo import TTSEventRepo

            with session_scope() as session:
                repo = TTSEventRepo(session)
                events = repo.get_recent(meeting_id=meeting_id, limit=limit)

            data = []
            for e in events:
                data.append({
                    "id": e.id,
                    "meeting_id": e.meeting_id,
                    "text": e.text,
                    "source": e.source,
                    "priority": e.priority,
                    "status": e.status,
                    "cooldown_key": e.cooldown_key,
                    "reason": e.reason,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "spoken_at": e.spoken_at.isoformat() if e.spoken_at else None,
                })
            self._send_ok({"tts_events": data})
        except Exception as e:
            self._send_error(500, f"Query failed: {e}")

    def _handle_agent_status(self):
        """GET /api/agent/status — 返回 Agent 状态摘要。"""
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        snapshot = self.server_ref.get_snapshot()
        agent_info = snapshot.get("agent", {})
        self._send_ok(agent_info)

    def _handle_agent_summarize(self, body: dict | None):
        """POST /api/agent/summarize — 手动触发阶段总结。"""
        if not body or "meeting_id" not in body:
            self._send_error(400, "Missing meeting_id")
            return
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("agent_trigger",
                                             meeting_id=body["meeting_id"],
                                             minutes=body.get("minutes", 3)):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Agent summary triggered"})

    def _handle_agent_agenda(self, body: dict | None):
        """POST /api/agent/agenda — 手动触发议题提醒。"""
        if not body or "meeting_id" not in body:
            self._send_error(400, "Missing meeting_id")
            return
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("agent_agenda",
                                             meeting_id=body["meeting_id"]):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Agent agenda reminder triggered"})

    def _handle_agent_trigger_status(self, body: dict | None):
        """POST /api/agent/status — 手动触发系统状态播报。"""
        meeting_id = body.get("meeting_id") if body else None
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("agent_status",
                                             meeting_id=meeting_id):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Agent status broadcast triggered"})

    def _handle_agent_custom_tts(self, body: dict | None):
        """POST /api/agent/custom_tts — 手动输入文字播报。"""
        if not body or "text" not in body:
            self._send_error(400, "Missing 'text' in request body")
            return
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("agent_custom_tts",
                                             text=body["text"],
                                             meeting_id=body.get("meeting_id")):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Custom TTS triggered"})

    def _handle_agent_config(self, body: dict | None):
        """POST /api/agent/config — 更新 Agent 配置。"""
        if not body or "key" not in body:
            self._send_error(400, "Missing 'key' in request body")
            return
        if self.server_ref is None:
            self._send_error(500, "Server not initialized")
            return
        if not self.server_ref.send_command("agent_config",
                                             section=body.get("section", "agent"),
                                             key=body["key"],
                                             value=body.get("value")):
            self._send_error(503, "Command queue full")
            return
        self._send_ok({"message": "Agent config updated"})

    # ── 响应辅助 ────────────────────────────────────────────────────────────

    def _send_ok(self, data):
        """发送 200 OK JSON 响应。"""
        body = json.dumps({"ok": True, "data": data}, ensure_ascii=False)
        self._send_json(200, body)

    def _send_error(self, code: int, message: str):
        """发送错误 JSON 响应。"""
        body = json.dumps({"ok": False, "error": message}, ensure_ascii=False)
        self._send_json(code, body)

    def _send_json(self, code: int, body: str):
        """发送 JSON 响应。"""
        encoded = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)


# ═══════════════════════════════════════════════════════════════════════════════
# ControlApiServer
# ═══════════════════════════════════════════════════════════════════════════════

class ControlApiServer:
    """轻量级 HTTP API 服务器 — 在后台 daemon 线程中运行。

    为 Streamlit 主持人控制台提供 REST API 接口。所有硬件操作通过
    命令队列异步传递给主线程执行，确保线程安全。

    Thread-safety:
      - 状态: threading.Lock 保护 update_state() / get_snapshot()
      - 命令: queue.Queue (无界, HTTP→主线程单向)
    """

    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8800

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self._host = host
        self._port = port
        self._running = False
        self._thread: threading.Thread | None = None
        self._httpd: _ThreadingHTTPServer | None = None

        # ── 共享状态 ──────────────────────────────────────────────────────
        self._state_lock = threading.Lock()
        self._state: dict = {}

        # ── 命令队列 ──────────────────────────────────────────────────────
        self._cmd_queue: queue.Queue = queue.Queue(maxsize=100)

        # ── 回调引用（可选）───────────────────────────────────────────────
        self._on_start: Callable[[], None] | None = None
        self._on_stop: Callable[[], None] | None = None

        # 标记 handler 的 server_ref
        _APIHandler.server_ref = self

    # ── 生命周期 ──────────────────────────────────────────────────────────

    def start(self):
        """启动 HTTP 服务器（后台 daemon 线程）。"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._serve_loop,
            daemon=True,
            name="ControlApiServer",
        )
        self._thread.start()

        if self._on_start:
            try:
                self._on_start()
            except Exception:
                pass

    def stop(self, timeout: float = 5.0):
        """停止 HTTP 服务器，等待线程退出。"""
        self._running = False

        # 关闭 HTTP 服务器 socket
        httpd = self._httpd
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:
                pass

        # 等待后台线程退出
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

        self._httpd = None
        self._thread = None

        if self._on_stop:
            try:
                self._on_stop()
            except Exception:
                pass

    # ── 回调设置 ──────────────────────────────────────────────────────────

    def set_on_start(self, callback: Callable[[], None]):
        """设置服务器启动回调。"""
        self._on_start = callback

    def set_on_stop(self, callback: Callable[[], None]):
        """设置服务器停止回调。"""
        self._on_stop = callback

    # ── 状态共享 (主线程→HTTP线程) ────────────────────────────────────────

    def update_state(self, state: dict):
        """主线程调用 — 更新共享状态快照 (线程安全)。

        每 100ms 从 _tracking_tick() 调用一次。

        Args:
            state: 状态字典，会被深拷贝后存储
        """
        with self._state_lock:
            self._state = copy.deepcopy(state)

    def get_snapshot(self) -> dict:
        """HTTP 线程调用 — 获取当前状态快照 (线程安全)。

        Returns:
            状态字典的深拷贝，不会持有锁引用
        """
        with self._state_lock:
            return copy.deepcopy(self._state)

    # ── 命令队列 (HTTP线程→主线程) ───────────────────────────────────────

    def send_command(self, cmd_type: str, **params) -> bool:
        """HTTP 线程调用 — 发送命令到主线程队列。

        Args:
            cmd_type: 命令类型 (如 "recenter", "lock_speaker")
            **params: 命令参数

        Returns:
            True = 成功入队, False = 队列满或异常
        """
        try:
            cmd = {"type": cmd_type, "params": params, "timestamp": time.time()}
            self._cmd_queue.put_nowait(cmd)
            return True
        except queue.Full:
            print(f"[ControlApi] 命令队列满，丢弃: {cmd_type}")
            return False
        except Exception:
            return False

    def poll_commands(self) -> list[dict]:
        """主线程调用 — 排空并返回所有待处理命令。

        每 100ms 从 _tracking_tick() 调用一次。

        Returns:
            命令列表 (按入队顺序)，队列为空时返回空列表
        """
        commands = []
        now = time.time()
        CMD_TTL = 10.0  # 丢弃超过 10 秒的陈旧命令
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
                # 丢弃陈旧命令
                if now - cmd.get("timestamp", 0) < CMD_TTL:
                    commands.append(cmd)
            except queue.Empty:
                break
        return commands

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def port(self) -> int:
        return self._port

    @property
    def host(self) -> str:
        return self._host

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self._port}"

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _serve_loop(self):
        """HTTP 服务器主循环 — 运行在后台线程中。"""
        try:
            self._httpd = _ThreadingHTTPServer((self._host, self._port), _APIHandler)
            # 更新 server_ref (可能在构造函数和实际运行之间变了)
            _APIHandler.server_ref = self
            print(f"[ControlApi] 服务器已启动: {self.url}")
            self._httpd.serve_forever(poll_interval=0.5)
        except OSError as e:
            print(f"[ControlApi] 启动失败 (端口 {self._port}): {e}")
            self._running = False
        except Exception as e:
            print(f"[ControlApi] 服务器异常: {e}")
            traceback.print_exc()
            self._running = False
        finally:
            print("[ControlApi] 服务器已停止")
