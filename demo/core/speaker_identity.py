"""AprilTag-based speaker identity resolution — 纯 Python，零 Qt 依赖。

将 AprilTag 检测结果 + YOLO 人体框关联到参会人员，实现:
  - 标签→人员匹配（空间距离 + 稳定性评分）
  - 防抖状态机（连续帧确认 / 短暂丢失保持 / 超时遗忘）
  - 发言人变更事件回调
  - 手动覆盖 / 解除覆盖

用法:
  from core.speaker_identity import SpeakerIdentifier, SpeakerIdentity

  identifier = SpeakerIdentifier(confirm_frames=3, lost_timeout=5.0)
  identifier.set_participant_lookup(lambda tag_id: {"name": "王强", "role": "队长"})
  identifier.on_speaker_event = lambda evt: print(f"Speaker event: {evt}")

  # 每帧调用（由 GUI 的 100ms tick 驱动）
  identity = identifier.tick(now=time.time(), tags=detections,
                             person_box={"left":320,"top":100,"right":640,"bottom":600},
                             frame_width=1920, frame_height=1080)
"""

import time
from dataclasses import dataclass, field
from typing import Callable

# Re-export tag ID helpers from desk_card_generator (avoid circular import at module level)
# Users can also import directly from core.desk_card_generator


def tag_id_to_int(label: str) -> int:
    """Convert human-readable tag label to integer ID.

    "A001" -> 0, "A023" -> 22, "A100" -> 99
    """
    if not label or not label[0].isalpha():
        raise ValueError(f"Invalid tag label: {label!r}, expected like 'A001'")
    return int(label[1:]) - 1


def int_to_tag_id(n: int) -> str:
    """Convert integer ID to human-readable tag label.

    0 -> "A001", 99 -> "A100"
    """
    if n < 0 or n > 999:
        raise ValueError(f"Tag integer {n} out of range [0, 999]")
    return f"A{n + 1:03d}"


# ═════════════════════════════════════════════════════════════════════════════════
# SpeakerIdentity — 单帧身份解析结果
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class SpeakerIdentity:
    """当前发言人身份快照。"""

    tag_id: str | None = None          # 人类可读标签，如 "A001"
    tag_id_int: int | None = None      # AprilTag 整数 ID，如 0
    name: str | None = None            # 发言人姓名（来自 participant 表）
    role: str | None = None            # 发言人角色
    organization: str | None = None    # 所属组织
    source: str = "unknown"            # "april_tag" | "manual" | "unknown"
    confidence: float = 0.0            # 0.0 - 1.0
    is_confirmed: bool = False         # 是否通过防抖确认
    state: str = "unknown"             # "unknown" | "candidate" | "confirmed" | "lost" | "manual"
    confirmed_at: float = 0.0          # time.time() 确认时刻
    duration: float = 0.0              # 当前身份持续时间 (秒)


# ═════════════════════════════════════════════════════════════════════════════════
# SpeakerIdentifier — 防抖状态机
# ═════════════════════════════════════════════════════════════════════════════════

class SpeakerIdentifier:
    """纯 Python 发言人身份识别器 — 防抖 + 标签-人体关联 + 参与人查询。

    State machine:
      UNKNOWN ──(N consecutive frames same tag + in DB)──▶ CONFIRMED
      CONFIRMED ──(tag not seen for M frames)──▶ LOST (timer starts)
      LOST ──(same tag reappears for N frames)──▶ CONFIRMED
      LOST ──(timer > timeout)──▶ UNKNOWN (emit speaker_ended)
      CONFIRMED ──(different tag confirmed)──▶ CONFIRMED (emit speaker_switched)
      Any ──(manual_override)──▶ MANUAL
      MANUAL ──(clear_override)──▶ UNKNOWN

      N = confirm_frames (default 3), M = lost_confirm_frames (default 5).
    """

    # ── 状态枚举 ────────────────────────────────────────────────────────────
    STATE_UNKNOWN   = "unknown"
    STATE_CANDIDATE = "candidate"
    STATE_CONFIRMED = "confirmed"
    STATE_LOST      = "lost"
    STATE_MANUAL    = "manual"

    def __init__(self,
                 confirm_frames: int = 3,
                 lost_timeout: float = 5.0,
                 max_tag_distance_ratio: float = 3.0,
                 lost_confirm_frames: int = 5,
                 ):
        """
        Args:
            confirm_frames: 连续帧数阈值，同一 tag 出现 N 帧后确认身份
            lost_timeout: 丢失超时（秒），超过此时间未恢复则遗忘身份
            max_tag_distance_ratio: 标签到人体框中心的最大距离倍率
                                      (相对于人体框对角线长度)
            lost_confirm_frames: 连续丢失帧数阈值，tag 消失 N 帧后才
                                  触发 speaker_lost 事件（消除抖动）
        """
        self.confirm_frames = confirm_frames
        self.lost_timeout = lost_timeout
        self.max_tag_distance_ratio = max_tag_distance_ratio
        self.lost_confirm_frames = lost_confirm_frames

        # ── 回调 ──────────────────────────────────────────────────────────
        self._participant_lookup: Callable[[str], dict | None] | None = None
        self.on_speaker_event: Callable[[dict], None] | None = None

        # ── 内部状态 ──────────────────────────────────────────────────────
        self._state: str = self.STATE_UNKNOWN
        self._identity: SpeakerIdentity = SpeakerIdentity()

        # 候选标签累积 (UNKNOWN → CONFIRMED 的防抖)
        self._candidate_tag_int: int | None = None
        self._candidate_count: int = 0

        # 丢失计时 (CONFIRMED → LOST → UNKNOWN)
        self._lost_since: float | None = None

        # 丢失帧计数器（tag 连续丢失帧数，用于防抖）
        self._lost_frame_count: int = 0

        # 恢复帧计数器（LOST 状态下 tag 连续重现帧数，用于防抖）
        self._recovery_frame_count: int = 0

        # 标签稳定性追踪 (tag_id_int → 连续出现帧数)
        self._tag_stability: dict[int, int] = {}

        # 已确认的身份（LOST 期间保持）
        self._last_confirmed_tag_int: int | None = None
        self._last_confirmed_name: str | None = None
        self._last_confirmed_role: str | None = None
        self._last_confirmed_org: str | None = None

    # ── 公共 API ────────────────────────────────────────────────────────────

    def set_participant_lookup(self, fn: Callable[[str], dict | None]):
        """注入参与人查询函数。

        Args:
            fn: 接收 tag_id 字符串 (如 "A001")，返回 dict 或 None。
                dict 格式: {"name": str, "role": str, "organization": str}
        """
        self._participant_lookup = fn

    def tick(self, now: float, tags: list,
             person_box: dict | None = None,
             frame_width: int = 1920, frame_height: int = 1080) -> SpeakerIdentity:
        """处理一帧 AprilTag 检测结果，更新身份状态。

        Args:
            now: time.time() 时间戳
            tags: pupil_apriltags.Detection 对象列表
            person_box: {"left", "top", "right", "bottom"} 人体框（图像坐标），
                        None 表示未检测到人
            frame_width: 画面宽度（像素）
            frame_height: 画面高度（像素）

        Returns:
            当前 SpeakerIdentity（处理后）
        """
        # ── 手动覆盖模式：跳过所有自动处理 ──────────────────────────────────
        if self._state == self.STATE_MANUAL:
            self._identity.duration = now - self._identity.confirmed_at
            return self._identity

        # ── 更新标签稳定性计数器 ────────────────────────────────────────────
        seen_int_ids = {tag.tag_id for tag in tags} if tags else set()
        for tag_int in list(self._tag_stability.keys()):
            if tag_int in seen_int_ids:
                self._tag_stability[tag_int] += 1
            else:
                # 衰减：丢失的标签逐渐降低稳定性
                self._tag_stability[tag_int] -= 1
                if self._tag_stability[tag_int] <= 0:
                    del self._tag_stability[tag_int]
        for tag_int in seen_int_ids:
            if tag_int not in self._tag_stability:
                self._tag_stability[tag_int] = 1

        # ── 关联：找到最佳匹配的标签 ─────────────────────────────────────────
        best_tag_int, best_tag_str, best_confidence = self._associate_tags(
            tags, person_box, frame_width, frame_height)

        # ── 状态机 ──────────────────────────────────────────────────────────
        if self._state == self.STATE_UNKNOWN:
            self._tick_unknown(now, best_tag_int, best_tag_str, best_confidence)
        elif self._state == self.STATE_CONFIRMED:
            self._tick_confirmed(now, best_tag_int, best_tag_str, best_confidence)
        elif self._state == self.STATE_LOST:
            self._tick_lost(now, best_tag_int, best_tag_str, best_confidence)
        elif self._state == self.STATE_CANDIDATE:
            self._tick_unknown(now, best_tag_int, best_tag_str, best_confidence)

        # ── 更新持续时间 ────────────────────────────────────────────────────
        if self._identity.confirmed_at > 0:
            self._identity.duration = now - self._identity.confirmed_at
        self._identity.state = self._state

        return self._identity

    def manual_override(self, tag_id_str: str):
        """手动指定发言人（跳过自动识别）。

        Args:
            tag_id_str: tag_id 字符串，如 "A001"
        """
        self._state = self.STATE_MANUAL
        self._confirm_identity(tag_id_str, time.time(), source="manual")
        self._emit_event("speaker_override", {"tag_id": tag_id_str})

    def clear_override(self):
        """解除手动覆盖，回到 UNKNOWN 状态重新自动识别。"""
        prev_tag = self._identity.tag_id
        self._state = self.STATE_UNKNOWN
        self._identity = SpeakerIdentity()
        self._candidate_tag_int = None
        self._candidate_count = 0
        self._lost_since = None
        self._last_confirmed_tag_int = None
        self._emit_event("speaker_override_cleared", {"prev_tag_id": prev_tag})

    @property
    def state(self) -> str:
        return self._state

    @property
    def identity(self) -> SpeakerIdentity:
        return self._identity

    # ── 状态处理 ────────────────────────────────────────────────────────────

    def _tick_unknown(self, now: float, best_tag_int: int | None,
                      best_tag_str: str | None, confidence: float):
        """UNKNOWN 状态：累积防抖，等待连续 N 帧同一标签。"""
        if best_tag_str is None or best_tag_int is None:
            # 没有匹配的标签 → 重置候选
            self._candidate_tag_int = None
            self._candidate_count = 0
            return

        # 检查是否在参与人表中
        if self._participant_lookup:
            p = self._participant_lookup(best_tag_str)
            if p is None:
                # 标签不在参与人表中 → 忽略
                self._candidate_tag_int = None
                self._candidate_count = 0
                return

        # 累积计数
        if best_tag_int == self._candidate_tag_int:
            self._candidate_count += 1
        else:
            self._candidate_tag_int = best_tag_int
            self._candidate_count = 1

        if self._candidate_count >= self.confirm_frames:
            self._confirm_identity(best_tag_str, now, source="april_tag",
                                   confidence=confidence)
            self._emit_event("speaker_started", {
                "tag_id": best_tag_str,
                "name": self._identity.name,
                "role": self._identity.role,
                "source": "april_tag",
                "confidence": confidence,
            })

    def _tick_confirmed(self, now: float, best_tag_int: int | None,
                        best_tag_str: str | None, confidence: float):
        """CONFIRMED 状态：检查标签是否仍在，或切换到新发言人。"""
        current_tag_int = self._last_confirmed_tag_int
        current_tag_str = self._identity.tag_id

        # ── 同一标签仍在 → 保持 ──────────────────────────────────────────
        if best_tag_int is not None and best_tag_int == current_tag_int:
            self._lost_since = None
            self._lost_frame_count = 0
            # 更新置信度（平滑）
            self._identity.confidence = self._identity.confidence * 0.7 + confidence * 0.3
            return

        # ── 不同标签出现 → 启动候选防抖 ───────────────────────────────────
        if best_tag_str is not None and best_tag_int != current_tag_int:
            self._lost_frame_count = 0  # 有标签活动，重置丢失计数器
            # 检查新标签是否在参与人表中
            in_db = True
            if self._participant_lookup:
                p = self._participant_lookup(best_tag_str)
                in_db = p is not None

            if in_db:
                if best_tag_int == self._candidate_tag_int:
                    self._candidate_count += 1
                else:
                    self._candidate_tag_int = best_tag_int
                    self._candidate_count = 1

                if self._candidate_count >= self.confirm_frames:
                    prev_tag_id = current_tag_str
                    prev_name = self._identity.name
                    self._confirm_identity(best_tag_str, now, source="april_tag",
                                           confidence=confidence)
                    self._emit_event("speaker_switched", {
                        "prev_tag_id": prev_tag_id,
                        "prev_name": prev_name,
                        "new_tag_id": best_tag_str,
                        "name": self._identity.name,
                        "role": self._identity.role,
                        "source": "april_tag",
                        "confidence": confidence,
                    })
                return

        # ── 标签丢失 → 累积丢失帧数，防抖后进入 LOST 状态 ─────────────────
        self._lost_frame_count += 1
        if self._lost_frame_count >= self.lost_confirm_frames:
            if self._lost_since is None:
                self._lost_since = now
            self._state = self.STATE_LOST
            self._emit_event("speaker_lost", {
                "tag_id": current_tag_str,
                "name": self._identity.name,
            })

    def _tick_lost(self, now: float, best_tag_int: int | None,
                   best_tag_str: str | None, confidence: float):
        """LOST 状态：等待标签恢复，超时则遗忘。"""
        current_tag_int = self._last_confirmed_tag_int

        # ── 同一标签恢复 → 累积恢复帧数，防抖后回到 CONFIRMED ──────────────
        if best_tag_int is not None and best_tag_int == current_tag_int:
            self._recovery_frame_count += 1
            if self._recovery_frame_count >= self.confirm_frames:
                away_duration = now - (self._lost_since or now)
                self._state = self.STATE_CONFIRMED
                self._lost_since = None
                self._lost_frame_count = 0
                self._recovery_frame_count = 0
                self._identity.is_confirmed = True
                self._identity.state = self.STATE_CONFIRMED
                self._identity.confidence = confidence
                self._emit_event("speaker_reidentified", {
                    "tag_id": self._identity.tag_id,
                    "name": self._identity.name,
                    "away_duration": away_duration,
                })
            return
        else:
            # 标签不匹配或消失 → 重置恢复计数器
            self._recovery_frame_count = 0

        # ── 超时 → 遗忘，回到 UNKNOWN ──────────────────────────────────────
        if self._lost_since is not None and (now - self._lost_since) >= self.lost_timeout:
            prev_tag = self._identity.tag_id
            prev_name = self._identity.name
            duration = now - self._identity.confirmed_at
            self._emit_event("speaker_ended", {
                "tag_id": prev_tag,
                "name": prev_name,
                "duration": duration,
            })
            # 清空当前身份
            self._identity = SpeakerIdentity()
            self._state = self.STATE_UNKNOWN
            self._candidate_tag_int = None
            self._candidate_count = 0
            self._lost_since = None
            self._last_confirmed_tag_int = None
            self._last_confirmed_name = None
            self._last_confirmed_role = None
            self._last_confirmed_org = None

            # 如果同时有新标签出现，立即开始候选
            if best_tag_str is not None:
                in_db = True
                if self._participant_lookup:
                    p = self._participant_lookup(best_tag_str)
                    in_db = p is not None
                if in_db:
                    self._candidate_tag_int = best_tag_int
                    self._candidate_count = 1

    # ── 标签关联算法 ─────────────────────────────────────────────────────────

    def _associate_tags(self, tags: list,
                        person_box: dict | None,
                        frame_width: int, frame_height: int
                        ) -> tuple[int | None, str | None, float]:
        """评估所有检测到的标签，返回最佳匹配的 (tag_int, tag_str, confidence)。

        评分规则（优先级从高到低）：
        1. 靠近人体框中心 → 高分
        2. 靠近画面中心 → 中等分（无人框时的回退）
        3. 历史稳定性 → 加分
        4. 过滤：不在参与人表中的标签直接排除
        """
        if not tags:
            return None, None, 0.0

        # 计算人体框中心和尺寸
        box_center = None
        box_diag = None
        if person_box:
            box_center = (
                (person_box["left"] + person_box["right"]) / 2.0,
                (person_box["top"] + person_box["bottom"]) / 2.0,
            )
            box_diag = (
                (person_box["right"] - person_box["left"]) ** 2 +
                (person_box["bottom"] - person_box["top"]) ** 2
            ) ** 0.5

        # 画面中心
        img_center = (frame_width / 2.0, frame_height / 2.0)
        img_diag = (frame_width ** 2 + frame_height ** 2) ** 0.5

        scored = []
        for tag in tags:
            tag_int = tag.tag_id
            tag_str = int_to_tag_id(tag_int)

            # 规则 4: 必须在参与人表中（如果设置了 lookup）
            if self._participant_lookup:
                p = self._participant_lookup(tag_str)
                if p is None:
                    continue  # 排除非参与人标签

            score = 0.0

            # 标签中心坐标
            cx, cy = tag.center

            # 规则 1: 靠近人体框中心（主要评分维度）
            if box_center is not None and box_diag is not None and box_diag > 0:
                dist_to_box = ((cx - box_center[0]) ** 2 + (cy - box_center[1]) ** 2) ** 0.5
                norm_dist = dist_to_box / box_diag
                if norm_dist <= self.max_tag_distance_ratio:
                    # 距离越近分数越高，最高 10 分
                    proximity_score = max(0.0, 1.0 - norm_dist / self.max_tag_distance_ratio) * 10.0
                    score += proximity_score

            # 规则 2: 靠近画面中心（回退/辅助评分）
            dist_to_center = ((cx - img_center[0]) ** 2 + (cy - img_center[1]) ** 2) ** 0.5
            norm_center_dist = dist_to_center / img_diag
            center_score = max(0.0, 1.0 - norm_center_dist * 4.0) * 2.0
            score += center_score

            # 规则 3: 稳定性加分
            stability = self._tag_stability.get(tag_int, 0)
            score += min(stability, 10) * 0.5

            # decision_margin 作为置信度参考
            conf = getattr(tag, "decision_margin", 0.0)

            scored.append((score, conf, tag_int, tag_str))

        if not scored:
            return None, None, 0.0

        # 按分数降序排列
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_conf, best_int, best_str = scored[0]

        # 置信度归一化: decision_margin 通常在 0-100+ 范围
        norm_conf = min(1.0, max(0.0, best_conf / 50.0))

        return best_int, best_str, norm_conf

    # ── 内部辅助 ────────────────────────────────────────────────────────────

    def _confirm_identity(self, tag_id_str: str, now: float,
                          source: str = "april_tag", confidence: float = 0.0):
        """确认身份：查询参与人信息并更新内部状态。"""
        name = None
        role = None
        org = None
        if self._participant_lookup:
            p = self._participant_lookup(tag_id_str)
            if p:
                name = p.get("name")
                role = p.get("role")
                org = p.get("organization")

        tag_int = tag_id_to_int(tag_id_str)

        self._state = self.STATE_CONFIRMED
        self._identity = SpeakerIdentity(
            tag_id=tag_id_str,
            tag_id_int=tag_int,
            name=name,
            role=role,
            organization=org,
            source=source,
            confidence=confidence,
            is_confirmed=True,
            state=self.STATE_CONFIRMED,
            confirmed_at=now,
            duration=0.0,
        )
        self._candidate_tag_int = None
        self._candidate_count = 0
        self._lost_since = None
        self._last_confirmed_tag_int = tag_int
        self._last_confirmed_name = name
        self._last_confirmed_role = role
        self._last_confirmed_org = org

    def _emit_event(self, event_type: str, extra: dict | None = None):
        """触发回调（如果已设置）。"""
        if self.on_speaker_event is None:
            return
        event = {
            "event_type": event_type,
            "timestamp": time.time(),
            "state": self._state,
        }
        if extra:
            event.update(extra)
        try:
            self.on_speaker_event(event)
        except Exception as e:
            print(f"[SpeakerIdentifier] 事件回调异常 ({event_type}): {e}")

    def reset(self):
        """重置所有状态到初始。"""
        self._state = self.STATE_UNKNOWN
        self._identity = SpeakerIdentity()
        self._candidate_tag_int = None
        self._candidate_count = 0
        self._lost_since = None
        self._lost_frame_count = 0
        self._recovery_frame_count = 0
        self._tag_stability.clear()
        self._last_confirmed_tag_int = None
        self._last_confirmed_name = None
        self._last_confirmed_role = None
        self._last_confirmed_org = None
