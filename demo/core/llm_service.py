"""LLM Service — DeepSeek V4 Flash 调用封装。

基于 OpenAI Python SDK，为智能会议系统提供:
  - 会中问答（实时会议状态查询）
  - 会后摘要生成
  - 待办事项提取
  - 系统故障诊断

环境变量:
  DEEPSEEK_API_KEY   — DeepSeek API 密钥（必需）
  DEEPSEEK_BASE_URL  — API 基础 URL（默认 https://api.deepseek.com）

用法:
  from core.llm_service import LLMService

  svc = LLMService()

  # 检查可用性
  if LLMService.is_available():
      result = LLMService.ask_question(1, "谁发言最多？")
      if result["ok"]:
          print(result["answer"])

  # 会后摘要
  summary = LLMService.generate_summary(1)
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI 异常类型（用于友好错误提示）
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from openai import (
        APIError,
        APIConnectionError,
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        RateLimitError,
    )
except ImportError:  # openai < 1.0 回退
    APIError = Exception
    APIConnectionError = Exception
    APITimeoutError = Exception
    AuthenticationError = Exception
    BadRequestError = Exception
    RateLimitError = Exception


# ═══════════════════════════════════════════════════════════════════════════════
# 客户端管理
# ═══════════════════════════════════════════════════════════════════════════════

_client: OpenAI | None = None
_client_checked: bool = False


def _get_client() -> OpenAI | None:
    """获取 OpenAI 客户端（指向 DeepSeek）。

    API key 缺失时返回 None。使用模块级惰性单例。
    """
    global _client, _client_checked

    if _client_checked:
        return _client

    _client_checked = True
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        _client = None
        return None

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    try:
        _client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        _client = None

    return _client


# ═══════════════════════════════════════════════════════════════════════════════
# 核心调用
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL = "deepseek-v4-flash"


def _map_error(e: Exception) -> str:
    """将 OpenAI 异常映射为中文友好错误消息。"""
    if isinstance(e, AuthenticationError):
        return "API 密钥无效，请检查 DEEPSEEK_API_KEY 环境变量"
    elif isinstance(e, RateLimitError):
        return "API 调用频率过高，请稍后重试"
    elif isinstance(e, APITimeoutError):
        return "API 请求超时，请稍后重试"
    elif isinstance(e, APIConnectionError):
        return "无法连接到 DeepSeek API，请检查网络和 DEEPSEEK_BASE_URL"
    elif isinstance(e, BadRequestError):
        return f"请求参数错误: {e}"
    else:
        msg = str(e)
        # 截断过长的错误消息
        if len(msg) > 200:
            msg = msg[:200] + "..."
        return f"API 调用失败: {msg}"


def call_deepseek(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    timeout: float = 60.0,
) -> str:
    """调用 DeepSeek V4 Flash Chat Completions。

    Args:
        messages: OpenAI 格式的消息列表
        temperature: 采样温度（默认 0.2，偏确定性）
        max_tokens: 最大输出 token 数
        timeout: 超时秒数

    Returns:
        LLM 响应文本

    Raises:
        RuntimeError: API key 未配置或 API 调用失败
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY 环境变量，请在启动前设置")

    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return resp.choices[0].message.content or ""
    except (APIError, APIConnectionError, APITimeoutError,
            AuthenticationError, BadRequestError, RateLimitError) as e:
        raise RuntimeError(_map_error(e)) from e
    except Exception as e:
        raise RuntimeError(f"LLM 调用异常: {e}") from e


# ═══════════════════════════════════════════════════════════════════════════════
# Context Builder — 会中上下文
# ═══════════════════════════════════════════════════════════════════════════════

def build_meeting_context(meeting_id: int) -> dict:
    """构建会中问答所需的会议上下文。

    在 session_scope 中查询数据库，收集:
      - 会议基本信息
      - 参会人员及发言统计
      - 当前发言人
      - 最近 30 分钟事件
      - 主持人备注

    Args:
        meeting_id: 会议 ID

    Returns:
        结构化上下文字典，可直接 JSON 序列化
    """
    from storage.db import session_scope
    from storage.repo import (
        MeetingRepo, ParticipantRepo, SpeakerSegmentRepo,
        EventRepo, HostNoteRepo,
    )

    ctx: dict[str, Any] = {}

    with session_scope() as session:
        # ── 会议信息 ──
        meeting = MeetingRepo(session).get_by_id(meeting_id)
        if meeting is None:
            return {"error": f"会议不存在 (id={meeting_id})"}

        ctx["会议信息"] = {
            "名称": meeting.name,
            "状态": meeting.status,
            "地点": meeting.location or "未指定",
            "开始时间": meeting.start_time.isoformat() if meeting.start_time else "未开始",
        }

        # ── 参会人员 + 发言统计 ──
        seg_repo = SpeakerSegmentRepo(session)
        participants = ParticipantRepo(session).list_all()

        participant_stats = []
        for p in participants:
            total_dur = seg_repo.get_total_duration(meeting_id, speaker_tag_id=p.tag_id) or 0
            seg_count = 0
            # 通过遍历 segments 统计发言次数
            all_segs = seg_repo.get_segments_for_meeting(meeting_id)
            seg_count = sum(1 for s in all_segs if s.speaker_tag_id == p.tag_id)

            participant_stats.append({
                "标签ID": p.tag_id,
                "姓名": p.name,
                "角色": p.role or "参会人员",
                "单位": p.organization or "",
                "累计发言秒数": round(total_dur, 1),
                "发言次数": seg_count,
            })

        ctx["参会人员"] = participant_stats

        # ── 当前发言人 ──
        active_seg = seg_repo.get_active_segment(meeting_id)
        if active_seg:
            ctx["当前发言人"] = {
                "姓名": active_seg.speaker_name or "未知",
                "角色": active_seg.role or "",
                "标签ID": active_seg.speaker_tag_id or "",
                "识别来源": active_seg.source or "unknown",
                "已发言秒数": round(
                    (datetime.utcnow() - active_seg.start_time).total_seconds(), 1
                ) if active_seg.start_time else 0,
            }
        else:
            ctx["当前发言人"] = None

        # ── 最近事件（30 分钟） ──
        event_repo = EventRepo(session)
        recent_events = event_repo.get_recent(minutes=30, meeting_id=meeting_id)
        events_list = []
        for evt in recent_events:
            payload = {}
            if evt.payload_json:
                try:
                    payload = json.loads(evt.payload_json)
                except (json.JSONDecodeError, TypeError):
                    payload = {}

            ts = evt.timestamp.isoformat() if evt.timestamp else ""
            events_list.append({
                "类型": evt.event_type,
                "时间": ts,
                "详情": _event_summary(evt.event_type, payload),
            })
        ctx["最近事件"] = events_list

        # ── 主持人备注 ──
        note_repo = HostNoteRepo(session)
        notes = note_repo.get_notes_for_meeting(meeting_id)
        notes_list = []
        for note in notes:
            notes_list.append({
                "类型": note.note_type,
                "内容": note.content,
                "相关发言人": note.related_speaker or "",
                "时间": note.timestamp.isoformat() if note.timestamp else "",
            })
        ctx["主持人备注"] = notes_list

    return ctx


def _event_summary(event_type: str, payload: dict) -> str:
    """将事件 payload 转为简短中文摘要。"""
    if event_type == "meeting_started":
        loc = payload.get("location", "")
        return f"会议开始 ({payload.get('meeting_name', '')}{' @ ' + loc if loc else ''})"
    elif event_type == "meeting_ended":
        dur = payload.get("duration_seconds")
        if dur:
            m = int(dur // 60)
            s = int(dur % 60)
            return f"会议结束 (时长 {m}分{s}秒)"
        return "会议结束"
    elif event_type == "speaker_started":
        return f"{payload.get('name', '?')} 开始发言 ({payload.get('source', '')})"
    elif event_type == "speaker_ended":
        dur = payload.get("duration")
        if dur:
            return f"{payload.get('name', '?')} 结束发言 (持续 {int(dur)}秒)"
        return f"{payload.get('name', '?')} 结束发言"
    elif event_type == "speaker_switched":
        prev = payload.get("prev_name", "?")
        curr = payload.get("name", "?")
        return f"发言人切换: {prev} → {curr}"
    elif event_type == "speaker_lost":
        return f"标签丢失: {payload.get('name', payload.get('tag_id', '?'))}"
    elif event_type == "speaker_reidentified":
        return f"标签恢复: {payload.get('name', payload.get('tag_id', '?'))}"
    elif event_type == "host_locked_speaker":
        return f"主持人锁定: {payload.get('name', payload.get('tag_id', '?'))}"
    elif event_type == "host_note_added":
        nt = payload.get("note_type", "")
        content = payload.get("content", "")
        short = content[:50] + "..." if len(content) > 50 else content
        return f"备注 [{nt}]: {short}"
    elif event_type == "state_changed":
        return f"状态变更: {payload.get('from_state', '?')} → {payload.get('to_state', '?')}"
    elif event_type == "tracking_started":
        return "开始追踪"
    elif event_type == "tracking_stopped":
        return "停止追踪"
    else:
        # 通用：尝试显示 payload 中有意义的值
        for key in ("reason", "message", "detail", "name"):
            if key in payload and payload[key]:
                return str(payload[key])[:80]
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Context Builder — 会后完整上下文
# ═══════════════════════════════════════════════════════════════════════════════

def build_post_meeting_context(meeting_id: int) -> str:
    """构建会后摘要所需的完整会议上下文。

    查询完整会议数据并格式化为文本块，供 LLM 生成摘要使用。
    包含截断策略：发言片段 > 200 时保留最近 200 条。

    Args:
        meeting_id: 会议 ID

    Returns:
        格式化的文本块（可直接嵌入 system prompt）
    """
    from storage.db import session_scope
    from storage.repo import (
        MeetingRepo, ParticipantRepo, SpeakerSegmentRepo,
        EventRepo, HostNoteRepo,
    )

    with session_scope() as session:
        meeting = MeetingRepo(session).get_by_id(meeting_id)
        if meeting is None:
            return f"[错误] 会议不存在 (id={meeting_id})"

        seg_repo = SpeakerSegmentRepo(session)
        segments = seg_repo.get_segments_for_meeting(meeting_id)
        events = EventRepo(session).get_for_meeting(meeting_id)
        notes = HostNoteRepo(session).get_notes_for_meeting(meeting_id)
        participants = ParticipantRepo(session).list_all()

    # ── 计算时长 ──
    total_duration = 0
    if meeting.start_time and meeting.end_time:
        total_duration = (meeting.end_time - meeting.start_time).total_seconds()
    elif meeting.start_time:
        total_duration = (datetime.utcnow() - meeting.start_time).total_seconds()

    # ── 组装文本 ──
    lines = []

    lines.append(f"会议名称: {meeting.name}")
    lines.append(f"地点: {meeting.location or '未指定'}")
    lines.append(f"状态: {meeting.status}")
    if meeting.start_time:
        lines.append(f"开始时间: {meeting.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if meeting.end_time:
        lines.append(f"结束时间: {meeting.end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    h = int(total_duration // 3600)
    m = int((total_duration % 3600) // 60)
    s = int(total_duration % 60)
    lines.append(f"总时长: {h}小时{m}分{s}秒")
    lines.append("")

    # ── 发言人统计 ──
    # 按 tag_id 聚合
    speaker_stats: dict[str, dict] = {}
    for seg in segments:
        tid = seg.speaker_tag_id or "__unknown__"
        if tid not in speaker_stats:
            # 查找参与者信息
            p = next((x for x in participants if x.tag_id == tid), None)
            speaker_stats[tid] = {
                "name": seg.speaker_name or (p.name if p else "未知"),
                "role": seg.role or (p.role if p else ""),
                "duration": 0.0,
                "count": 0,
            }
        speaker_stats[tid]["duration"] += seg.duration_seconds or 0
        speaker_stats[tid]["count"] += 1

    lines.append("=== 发言人统计 ===")
    for tid, stat in sorted(speaker_stats.items(),
                             key=lambda x: x[1]["duration"], reverse=True):
        dur_m = int(stat["duration"] // 60)
        dur_s = int(stat["duration"] % 60)
        role_str = f" ({stat['role']})" if stat["role"] else ""
        lines.append(
            f"  {stat['name']}{role_str}: "
            f"总发言 {dur_m}分{dur_s}秒, "
            f"发言 {stat['count']} 次"
        )
    lines.append("")

    # ── 发言时间轴 ──
    lines.append("=== 发言时间轴 ===")
    max_segments = 200
    if len(segments) > max_segments:
        lines.append(f"[共 {len(segments)} 条发言记录，以下显示最近 {max_segments} 条]")
        segments = segments[-max_segments:]

    for seg in segments:
        start_str = seg.start_time.strftime("%H:%M:%S") if seg.start_time else "--:--:--"
        end_str = seg.end_time.strftime("%H:%M:%S") if seg.end_time else "(发言中)"
        dur = seg.duration_seconds or 0
        dur_str = f"{int(dur // 60)}分{int(dur % 60)}秒"
        lines.append(
            f"  [{start_str} → {end_str}] {seg.speaker_name or '未知'} "
            f"({seg.source or 'unknown'}) 持续 {dur_str}"
        )
    lines.append("")

    # ── 系统事件 ──
    lines.append("=== 系统事件 ===")
    for evt in events:
        ts = evt.timestamp.strftime("%H:%M:%S") if evt.timestamp else "--"
        payload = {}
        if evt.payload_json:
            try:
                payload = json.loads(evt.payload_json)
            except (json.JSONDecodeError, TypeError):
                payload = {}
        detail = _event_summary(evt.event_type, payload)
        lines.append(f"  [{ts}] {detail}")

    lines.append("")

    # ── 主持人备注 ──
    lines.append("=== 主持人备注 ===")
    if notes:
        for note in notes:
            ts = note.timestamp.strftime("%H:%M:%S") if note.timestamp else "--"
            related = f" ({note.related_speaker})" if note.related_speaker else ""
            lines.append(f"  [{ts}] [{note.note_type}]{related} {note.content}")
    else:
        lines.append("  (无)")

    lines.append("")

    # ── 参会人员名册 ──
    lines.append("=== 参会人员名册 ===")
    for p in participants:
        lines.append(f"  {p.tag_id} | {p.name} | {p.role or '参会人员'} | {p.organization or ''}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 高层功能函数
# ═══════════════════════════════════════════════════════════════════════════════

def ask_meeting_question(
    meeting_id: int,
    question: str,
    status_snapshot: dict | None = None,
) -> dict:
    """会中问答 — 基于会议上下文回答用户问题。

    Args:
        meeting_id: 会议 ID
        question: 用户问题
        status_snapshot: 可选，来自 /api/status 的系统状态快照

    Returns:
        {"ok": True, "answer": str} 或 {"ok": False, "error": str}
    """
    if _get_client() is None:
        return {"ok": False, "error": "未配置 DEEPSEEK_API_KEY 环境变量，请在启动前设置"}

    # 构建上下文
    try:
        ctx = build_meeting_context(meeting_id)
    except Exception as e:
        return {"ok": False, "error": f"构建会议上下文失败: {e}"}

    if "error" in ctx:
        return {"ok": False, "error": ctx["error"]}

    # 注入系统状态
    if status_snapshot:
        tracking = status_snapshot.get("tracking", {})
        ctx["系统状态"] = {
            "运行状态": status_snapshot.get("runtime_state", "未知"),
            "追踪": "进行中" if tracking.get("tracking_active") else "已停止",
            "VAD语音检测": "有声" if tracking.get("vad_is_speech") else "无声",
            "DOA角度": tracking.get("doa_angle", "未知"),
            "云台水平角": tracking.get("pan_angle", "未知"),
            "云台俯仰角": tracking.get("tilt_angle", "未知"),
            "YOLO帧率": f"{tracking.get('yolo_fps', 0):.1f} fps" if tracking.get('yolo_fps') else "未知",
            "RTSP推流": tracking.get("rtsp_status", "未知"),
        }
    else:
        ctx["系统状态"] = "未提供（离线查询）"

    # 构建 system prompt
    system_prompt = (
        "你是「智会追声」智能会议追踪系统的 AI 助手。\n"
        "根据提供的会议上下文数据，回答用户关于当前会议的问题。\n\n"
        "要求:\n"
        "1. 回答简洁、准确，使用中文\n"
        "2. 如果数据不足以回答，诚实地说明缺少哪些信息\n"
        "3. 引用具体数据时注明来源（如发言人统计、事件日志等）\n"
        "4. 对于系统状态类问题，用通俗语言解释技术指标的含义\n\n"
        "会议上下文:\n"
        "```json\n"
        f"{json.dumps(ctx, ensure_ascii=False, indent=2)}\n"
        "```"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        answer = call_deepseek(messages, temperature=0.3, max_tokens=2048, timeout=60.0)
        return {"ok": True, "answer": answer}
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}


def generate_meeting_summary(meeting_id: int) -> dict:
    """会后摘要 — 根据完整会议数据生成中文摘要。

    Args:
        meeting_id: 会议 ID

    Returns:
        {"ok": True, "summary": str} 或 {"ok": False, "error": str}
    """
    if _get_client() is None:
        return {"ok": False, "error": "未配置 DEEPSEEK_API_KEY 环境变量，请在启动前设置"}

    try:
        ctx_text = build_post_meeting_context(meeting_id)
    except Exception as e:
        return {"ok": False, "error": f"构建会议上下文失败: {e}"}

    if ctx_text.startswith("[错误]"):
        return {"ok": False, "error": ctx_text}

    system_prompt = (
        "你是一位专业的会议记录秘书。请根据提供的完整会议数据，生成一份中文会议摘要。\n\n"
        "要求:\n"
        "1. 使用清晰的分段结构（用 Markdown 标题 ##）\n"
        "2. 包含以下内容:\n"
        "   - 会议基本信息（名称、时间、地点、时长）\n"
        "   - 参会人员概况\n"
        "   - 发言统计（每人累计时长、发言次数排名）\n"
        "   - 会议时间线摘要（关键发言切换节点）\n"
        "   - 主持人备注要点汇总\n"
        "   - 系统运行概况（事件统计）\n"
        "3. 基于实际数据，不要编造信息\n"
        "4. 如果某项数据为空，说明「无记录」即可\n\n"
        "会议数据:\n"
        "```\n"
        f"{ctx_text}\n"
        "```"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "请生成会议摘要。"},
    ]

    try:
        summary = call_deepseek(messages, temperature=0.3, max_tokens=4096, timeout=120.0)
        return {"ok": True, "summary": summary}
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}


def generate_action_items(meeting_id: int) -> dict:
    """待办提取 — 从会议备注和讨论中提取待办事项。

    Args:
        meeting_id: 会议 ID

    Returns:
        {"ok": True, "action_items": str} 或 {"ok": False, "error": str}
    """
    if _get_client() is None:
        return {"ok": False, "error": "未配置 DEEPSEEK_API_KEY 环境变量，请在启动前设置"}

    try:
        ctx_text = build_post_meeting_context(meeting_id)
    except Exception as e:
        return {"ok": False, "error": f"构建会议上下文失败: {e}"}

    if ctx_text.startswith("[错误]"):
        return {"ok": False, "error": ctx_text}

    system_prompt = (
        "你是一位高效的会议助理。请根据提供的会议数据，提取所有待办事项和行动项。\n\n"
        "提取来源:\n"
        "1. 主持人明确标记为「待办事项」的备注\n"
        "2. 评委问题中隐含的后续工作\n"
        "3. 会议讨论中可能产生的行动项\n\n"
        "要求:\n"
        "- 每条待办事项包含: 任务描述、相关人员（如有）、优先级建议（高/中/低）\n"
        "- 使用 Markdown 格式的复选框列表\n"
        "- 如果没有任何待办事项，说明「未发现待办事项」\n\n"
        "会议数据:\n"
        "```\n"
        f"{ctx_text}\n"
        "```"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "请提取待办事项。"},
    ]

    try:
        items = call_deepseek(messages, temperature=0.2, max_tokens=2048, timeout=90.0)
        return {"ok": True, "action_items": items}
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}


def diagnose_system(status_snapshot: dict) -> dict:
    """系统诊断 — 根据当前状态快照分析系统运行情况。

    Args:
        status_snapshot: 来自 ControlApiServer.get_snapshot() 的状态快照

    Returns:
        {"ok": True, "diagnosis": str} 或 {"ok": False, "error": str}
    """
    if _get_client() is None:
        return {"ok": False, "error": "未配置 DEEPSEEK_API_KEY 环境变量，请在启动前设置"}

    system_prompt = (
        "你是一位智能会议追踪系统的运维专家。「智会追声」系统运行在 RK3588 ARM 嵌入式板上，"
        "包含以下组件:\n"
        "- YOLO 人体检测 (NPU 推理)\n"
        "- AprilTag 视觉身份识别\n"
        "- ReSpeaker 麦克风阵列 DOA 声源定位\n"
        "- Silero VAD 语音活动检测\n"
        "- 双轴 PWM 舵机云台控制\n"
        "- GStreamer RTSP 视频推流\n"
        "- 视听融合追踪状态机 (IDLE → AWAIT → TRACKING)\n\n"
        "请根据提供的系统状态快照，分析当前运行情况，指出潜在问题并给出建议。\n"
        "用通俗易懂的中文回答，逐项分析各组件的健康状态。\n\n"
        "系统状态:\n"
        "```json\n"
        f"{json.dumps(status_snapshot, ensure_ascii=False, indent=2)}\n"
        "```"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "请诊断当前系统状态。"},
    ]

    try:
        diagnosis = call_deepseek(messages, temperature=0.3, max_tokens=2048, timeout=60.0)
        return {"ok": True, "diagnosis": diagnosis}
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# LLMService — 薄封装类
# ═══════════════════════════════════════════════════════════════════════════════

class LLMService:
    """DeepSeek LLM 服务封装 — 提供会中问答、会后摘要、待办提取、系统诊断。

    所有方法均为静态方法，无需实例化。
    与 MeetingService 风格一致。

    用法:
      if LLMService.is_available():
          result = LLMService.ask_question(1, "谁发言最多？")
    """

    @staticmethod
    def is_available() -> bool:
        """检查 LLM 服务是否可用（API key 已配置且客户端可创建）。"""
        return _get_client() is not None

    @staticmethod
    def ask_question(meeting_id: int, question: str,
                     status_snapshot: dict | None = None) -> dict:
        """会中问答。"""
        return ask_meeting_question(meeting_id, question, status_snapshot)

    @staticmethod
    def generate_summary(meeting_id: int) -> dict:
        """生成会议摘要。"""
        return generate_meeting_summary(meeting_id)

    @staticmethod
    def extract_action_items(meeting_id: int) -> dict:
        """提取待办事项。"""
        return generate_action_items(meeting_id)

    @staticmethod
    def diagnose(status_snapshot: dict) -> dict:
        """系统状态诊断。"""
        return diagnose_system(status_snapshot)
