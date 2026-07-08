"""视频帧叠加渲染 — 在画面上绘制名片条和系统状态。

使用 Pillow 渲染中文字体（OpenCV Hershey 字体仅支持 ASCII），
再合成回 OpenCV BGR 帧。

用法:
  from core.overlay_renderer import render_overlay
  annotated = render_overlay(frame, speaker_info={"name": "王强", "role": "队长", "duration": 35.2},
                             system_state="TRACKING", show_debug=False)
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── 中文字体路径（按优先级查找）──────────────────────────────────────────
def _find_chinese_font() -> str | None:
    """查找系统可用的中文字体，返回字体文件路径或 None。"""
    candidates = [
        # Noto Sans CJK SC (Simplified Chinese) — index 3 in TTC
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 3),
        # Droid Sans Fallback — single TTF
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", None),
        # Noto Sans CJK SC Bold — index 1 in TTC (Mono SC)
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 1),
        # WenQuanYi (fallback)
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
        # generic search
        *[(p, None) for p in [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ] if os.path.exists(p)],
    ]
    for path, idx in candidates:
        if os.path.exists(path):
            return path
    return None


_FONT_PATH = _find_chinese_font()

# ── 字体缓存（按尺寸懒加载）─────────────────────────────────────────────
_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont | None] = {}

# ── 名片文字渲染缓存 (单条目: 发言人通常持续多帧不变) ───────────────────
# cache 结构: ((name, role, bar_w, bar_h), pil_bgra_ndarray) 或 None
_speaker_text_cache: tuple[tuple, np.ndarray] | None = None


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | None:
    """获取指定大小的中文字体（带缓存）。"""
    if _FONT_PATH is None:
        return None
    key = (_FONT_PATH, size, bold)
    if key not in _font_cache:
        try:
            # TTC 文件需要指定 index；TTF 文件 index=0 也可正常工作
            _font_cache[key] = ImageFont.truetype(_FONT_PATH, size, index=3)
        except Exception:
            try:
                _font_cache[key] = ImageFont.truetype(_FONT_PATH, size, index=0)
            except Exception:
                _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


# ═════════════════════════════════════════════════════════════════════════
# 核心渲染函数
# ═════════════════════════════════════════════════════════════════════════

def _blend_rect_roi(display: np.ndarray, x: int, y: int, w: int, h: int,
                    bg_color: tuple, alpha: float):
    """在 display 的 ROI 区域绘制半透明矩形 (in-place)。

    用 ROI addWeighted 替代全帧 copy+addWeighted, 避免每帧 2 次 1.5MB 全帧拷贝。
    ROI 越小收益越大 (名片条 80×500=40KB vs 全帧 960×540=1.5MB)。

    Args:
        display: 目标 BGR 帧 (in-place 修改)
        x, y: ROI 左上角
        w, h: ROI 宽高
        bg_color: 矩形颜色 (B, G, R)
        alpha: 矩形不透明度 (0=完全透明, 1=不透明)
    """
    H, W = display.shape[:2]
    y1, y2 = max(0, y), min(H, y + h)
    x1, x2 = max(0, x), min(W, x + w)
    if y1 >= y2 or x1 >= x2:
        return
    roi = display[y1:y2, x1:x2]
    overlay_roi = roi.copy()
    cv2.rectangle(overlay_roi, (0, 0), (x2 - x1, y2 - y1), bg_color, -1)
    blended = cv2.addWeighted(roi, 1.0 - alpha, overlay_roi, alpha, 0)
    display[y1:y2, x1:x2] = blended


def render_overlay(frame: np.ndarray,
                   speaker_info: dict | None = None,
                   system_state: str = "",
                   show_debug: bool = False) -> np.ndarray:
    """在视频帧上叠加 lower-third 名片条和系统状态。

    中文字符通过 Pillow 渲染（支持 CJK），
    ASCII 字符（时长、系统状态）仍用 cv2.putText 以保持速度。

    Args:
        frame: BGR 视频帧 (numpy array, shape=(H, W, 3), uint8)
        speaker_info: 当前发言人信息，None 表示未知。
                      格式: {"name": str, "role": str, "duration": float (秒)}
        system_state: 系统状态文本（如 "IDLE", "AWAIT", "TRACKING"）
        show_debug: 是否显示调试信息

    Returns:
        叠加后的 BGR 帧（新数组，不修改原始 frame）
    """
    display = frame.copy()
    h, w = display.shape[:2]

    # ── 左下角 lower-third 名片条 ──────────────────────────────────────
    if speaker_info:
        name = speaker_info.get("name", "未知")
        role = speaker_info.get("role", "")
        duration = speaker_info.get("duration", 0.0)

        # 时长格式化
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration_str = f"{mins:02d}:{secs:02d}"

        # 名片条尺寸
        bar_h = 80
        bar_w = min(500, w - 40)
        bar_x = 20
        bar_y = h - bar_h - 20

        # 半透明背景 (ROI 混合: 只 blend 名片条区域, 不做全帧 copy+addWeighted)
        # 原 full-frame addWeighted 1.7ms → ROI 0.14ms, 省 ~1.6ms
        _blend_rect_roi(display, bar_x, bar_y, bar_w, bar_h,
                        bg_color=(30, 30, 30), alpha=0.7)
        # 左侧彩色竖条 (同 ROI 混合)
        _blend_rect_roi(display, bar_x, bar_y, 6, bar_h,
                        bg_color=(0, 180, 80), alpha=0.7)

        # ── 用 Pillow 渲染中文姓名 + 角色，再贴回 OpenCV ──────────────
        if _FONT_PATH is not None:
            _draw_speaker_text_pil(display, name, role,
                                   bar_x, bar_y, bar_w, bar_h)
        else:
            # 回退：OpenCV putText（仅支持 ASCII，中文会变 ????）
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display, name, (bar_x + 18, bar_y + 32), font,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)
            if role:
                cv2.putText(display, role, (bar_x + 18, bar_y + 56), font,
                            0.6, (0, 220, 100), 1, cv2.LINE_AA)

        # 发言时长（右侧 — ASCII 安全，用 OpenCV）
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(duration_str, font, 0.6, 1)
        cv2.putText(display, duration_str,
                    (bar_x + bar_w - tw - 16, bar_y + 44), font,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # ── 顶部状态栏 ────────────────────────────────────────────────────
    if system_state:
        # ROI 混合: 只 blend 顶部 30px 条, 不做全帧 copy+addWeighted
        # 原 full-frame addWeighted 1.7ms → ROI 0.1ms, 省 ~1.6ms
        _blend_rect_roi(display, 0, 0, w, 30, bg_color=(0, 0, 0), alpha=0.3)

        state_colors = {
            "IDLE": (100, 255, 100),
            "AWAIT": (100, 180, 255),
            "TRACKING": (255, 100, 100),
        }
        color = state_colors.get(system_state, (200, 200, 200))
        cv2.putText(display, system_state, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    return display


# ═════════════════════════════════════════════════════════════════════════
# Pillow 中文渲染 + 合成
# ═════════════════════════════════════════════════════════════════════════

def _draw_speaker_text_pil(display: np.ndarray,
                           name: str, role: str,
                           bar_x: int, bar_y: int,
                           bar_w: int, bar_h: int):
    """用 Pillow 渲染中文姓名和角色，合成到 OpenCV BGR 帧上。

    优化:
      1. PIL 渲染结果缓存 — 发言人姓名/角色通常持续多帧不变, 缓存 pil_bgra 避免重复渲染。
      2. uint16 整数 alpha 混合 — 替代原 float64 numpy 运算, ARM 上快 ~3x。

    Args:
        display: 目标 BGR 帧 (会在原位修改)
        name: 发言人姓名（可能含中文）
        role: 发言人角色（可能含中文）
        bar_x, bar_y: 名片条左上角
        bar_w, bar_h: 名片条宽高
    """
    global _speaker_text_cache
    h_display, w_display = display.shape[:2]

    # ── 1. PIL 渲染 (带缓存) ────────────────────────────────────
    cache_key = (name, role, bar_w, bar_h)
    pil_bgra = None
    if _speaker_text_cache is not None:
        ck, cached = _speaker_text_cache
        if ck == cache_key:
            pil_bgra = cached

    if pil_bgra is None:
        pil_img = Image.new("RGBA", (bar_w, bar_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(pil_img)
        name_font = _get_font(28, bold=True)
        role_font = _get_font(18, bold=False)
        if name_font:
            draw.text((18, 8), name, font=name_font, fill=(255, 255, 255, 255))
        if role_font and role:
            draw.text((18, 46), role, font=role_font, fill=(0, 220, 100, 255))
        pil_rgba = np.array(pil_img)                    # (bar_h, bar_w, 4) RGBA
        pil_bgra = pil_rgba[:, :, [2, 1, 0, 3]].copy()  # R,G,B,A → B,G,R,A (contiguous)
        _speaker_text_cache = (cache_key, pil_bgra)

    # ── 2. uint16 整数 alpha 混合 (替代 float64) ────────────────
    # result = (src * a + dst * (255 - a)) >> 8
    # 比 float64 运算快 ~3x on ARM (省 ~1.3ms)
    y1, y2 = bar_y, min(bar_y + bar_h, h_display)
    x1, x2 = bar_x, min(bar_x + bar_w, w_display)
    if y1 >= y2 or x1 >= x2:
        return

    a   = pil_bgra[:y2 - y1, :x2 - x1, 3:4].astype(np.uint16)
    rgb = pil_bgra[:y2 - y1, :x2 - x1, :3].astype(np.uint16)
    roi = display[y1:y2, x1:x2]
    display[y1:y2, x1:x2] = (
        (rgb * a + roi.astype(np.uint16) * (255 - a)) >> 8
    ).astype(np.uint8)
