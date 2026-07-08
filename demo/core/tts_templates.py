# -*- coding: utf-8 -*-
"""TTS 模板库 — 无需 LLM 的固定播报文本。

每个模板可包含 {name} {tag_id} {agenda_name} {next_agenda} 等占位符，
由 RuleEngine 在生成 CandidateSpeech 时填充。

纯 Python，零 Qt 依赖。
"""

# ═══════════════════════════════════════════════════════════════════════════════
# A 类：确定性系统播报（不用 LLM，每场会议次数极少）
# ═══════════════════════════════════════════════════════════════════════════════

TEMPLATES = {
    # ── 会议生命周期 ──
    "meeting_started":
        "会议已开始，系统将自动跟踪当前发言人。",
    "meeting_ended":
        "会议已结束，正在保存会议记录。",

    # ── 发言人身份 ──
    "speaker_confirmed":
        "已识别当前发言人：{name}，正在自动跟踪。",
    "speaker_confirmed_demo":
        "当前发言人：{name}，身份已确认，系统正在追踪。",

    # ── 发言超时 ──
    "speaker_overtime":
        "{name}已连续发言三分钟，建议进入总结。",
    "speaker_overtime_polite":
        "{name}已经发言较久，可以请他做个小结，或者邀请其他成员补充。",

    # ── 静默提醒 ──
    "silence_timeout":
        "当前讨论暂停了一会儿，可以进入下一议题或请成员补充。",
    "silence_timeout_short":
        "会议暂时无人发言。",

    # ── 议题超时 ──
    "agenda_timeout":
        "{agenda_name}已超出预计时间，建议进入下一议题：{next_agenda}。",
    "agenda_timeout_no_next":
        "{agenda_name}已超出预计时间，可以请{owner}进行总结。",

    # ── 身份丢失 ──
    "identity_lost":
        "身份识别暂时丢失，正在等待重新确认。",
    "identity_lost_tracking":
        "当前身份识别暂时丢失，系统将继续保持视觉跟踪。",

    # ── 系统异常 ──
    "system_error_camera":
        "摄像头暂时不可用，请检查视频输入。",
    "system_error_audio":
        "音频采集异常，请检查麦克风连接。",
    "system_error_tts":
        "语音播报服务暂时不可用。",
    "system_error_general":
        "系统出现异常，请检查设备连接。",

    # ── 导播控制 ──
    "tracking_started":
        "自动跟踪已开启。",
    "tracking_paused":
        "自动跟踪已暂停。",
    "tracking_lost":
        "目标丢失，正在重新搜索。",
    "manual_speaker":
        "已切换到手动指定发言人。",
    "host_locked_speaker":
        "已锁定当前发言人。",
    "host_unlocked_speaker":
        "已解锁发言人。",

    # ── 手动触发（Streamlit 按钮） ──
    "manual_system_status":
        "系统运行正常，正在跟踪当前发言人。",

    # ── 待办事项 ──
    "action_item_detected":
        "检测到一条可能的待办事项，已记录到会议备注。",
}


def get_template(template_id: str) -> str:
    """获取模板文本。

    Args:
        template_id: 模板 ID（如 "speaker_confirmed"）

    Returns:
        模板字符串，未匹配时返回空字符串
    """
    return TEMPLATES.get(template_id, "")


def format_template(template_id: str, **kwargs) -> str:
    """获取并填充模板。

    Args:
        template_id: 模板 ID
        **kwargs: 占位符值 (如 name="张三")

    Returns:
        填充后的播报文本，占位符缺失时保留原样
    """
    template = get_template(template_id)
    if not template:
        return ""
    try:
        return template.format(**kwargs)
    except KeyError:
        # 部分占位符缺失 — 安全回退，保留未填充的占位符
        return template
