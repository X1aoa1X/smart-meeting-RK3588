#!/usr/bin/env python3
"""
Horizontal desk-card generator — produces PNG desk cards and a combined PDF.

Visual target:
  - horizontal card, similar to a nameplate / desk card
  - left: large AprilTag
  - middle: large Chinese name + optional pinyin
  - right: event title + red rounded role pill
  - bottom: curved red footer with slogan

Uses Pillow for PNG generation and ReportLab for PDF assembly.
Loads pre-generated AprilTag images from the tagStandard41h12/ directory.
Zero PyQt5 dependency — safe for headless/Streamlit use.

Tag ID mapping:
  - Integer 0 -> filename "tag41_12_00000.png" -> human label "A001"
  - Integer N -> filename "tag41_12_{N:05d}.png" -> human label f"A{N+1:03d}"
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Tag ID <-> integer mapping
# ---------------------------------------------------------------------------


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



def tag_filename(tag_id_int: int) -> str:
    """Return the expected filename for a tag integer in tagStandard41h12/."""
    return f"tag41_12_{tag_id_int:05d}.png"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DeskCardConfig:
    """Layout configuration for the horizontal desk-card design."""

    # 210 x 100 mm has almost the same aspect ratio as the sample image.
    # At 300 dpi this produces about 2480 x 1181 px.
    card_size_mm: tuple = (210, 100)
    dpi: int = 300
    tag_display_mm: int = 58
    margin_mm: int = 12

    font_path_regular: str = ""
    font_path_bold: str = ""
    font_path_latin: str = ""

    name_font_size_pt: int = 200
    pinyin_font_size_pt: int = 50
    event_font_size_pt: int = 46
    role_font_size_pt: int = 52
    slogan_font_size_pt: int = 48
    info_font_size_pt: int = 24

    # Text / brand content.
    event_title: str = "嵌入式大赛"
    slogan: str = "AI for design, design for AI."

    # Colors.
    accent_red: str = "#d20b00"
    pill_red_left: str = "#ff5048"
    pill_red_right: str = "#d40e05"
    text_black: str = "#000000"
    subtle_gray: str = "#222222"

    # Derived pixel dimensions — computed properties so they react to dpi/mm changes.
    @property
    def _px_per_mm(self) -> float:
        return self.dpi / 25.4

    @property
    def _card_w_px(self) -> int:
        return round(self.card_size_mm[0] * self._px_per_mm)

    @property
    def _card_h_px(self) -> int:
        return round(self.card_size_mm[1] * self._px_per_mm)

    @property
    def _tag_size_px(self) -> int:
        return round(self.tag_display_mm * self._px_per_mm)

    @property
    def _margin_px(self) -> int:
        return round(self.margin_mm * self._px_per_mm)

    @property
    def card_size_px(self) -> tuple:
        return (self._card_w_px, self._card_h_px)

    def to_dict(self) -> dict:
        return {
            "card_size_mm": list(self.card_size_mm),
            "dpi": self.dpi,
            "tag_display_mm": self.tag_display_mm,
            "margin_mm": self.margin_mm,
            "font_path_regular": self.font_path_regular,
            "font_path_bold": self.font_path_bold,
            "font_path_latin": self.font_path_latin,
            "name_font_size_pt": self.name_font_size_pt,
            "pinyin_font_size_pt": self.pinyin_font_size_pt,
            "event_font_size_pt": self.event_font_size_pt,
            "role_font_size_pt": self.role_font_size_pt,
            "slogan_font_size_pt": self.slogan_font_size_pt,
            "info_font_size_pt": self.info_font_size_pt,
            "event_title": self.event_title,
            "slogan": self.slogan,
            "accent_red": self.accent_red,
            "pill_red_left": self.pill_red_left,
            "pill_red_right": self.pill_red_right,
            "text_black": self.text_black,
            "subtle_gray": self.subtle_gray,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeskCardConfig":
        card_size = tuple(d.get("card_size_mm", [210, 100]))
        return cls(
            card_size_mm=card_size,
            dpi=d.get("dpi", 300),
            tag_display_mm=d.get("tag_display_mm", 58),
            margin_mm=d.get("margin_mm", 12),
            font_path_regular=d.get("font_path_regular", ""),
            font_path_bold=d.get("font_path_bold", ""),
            font_path_latin=d.get("font_path_latin", ""),
            name_font_size_pt=d.get("name_font_size_pt", 150),
            pinyin_font_size_pt=d.get("pinyin_font_size_pt", 50),
            event_font_size_pt=d.get("event_font_size_pt", 46),
            role_font_size_pt=d.get("role_font_size_pt", 52),
            slogan_font_size_pt=d.get("slogan_font_size_pt", 48),
            info_font_size_pt=d.get("info_font_size_pt", 24),
            event_title=d.get("event_title", "嵌入式大赛"),
            slogan=d.get("slogan", "AI for design, design for AI."),
            accent_red=d.get("accent_red", "#d20b00"),
            pill_red_left=d.get("pill_red_left", "#ff5048"),
            pill_red_right=d.get("pill_red_right", "#d40e05"),
            text_black=d.get("text_black", "#000000"),
            subtle_gray=d.get("subtle_gray", "#222222"),
        )

    @classmethod
    def from_json(cls, path: str) -> "DeskCardConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------


_FONT_SEARCH_PATHS = [
    "assets/fonts",
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/noto-cjk",
    "/usr/share/fonts/google-noto-cjk",
    "/usr/share/fonts/truetype/wqy",
    "/usr/share/fonts/truetype/droid",
    "/usr/share/fonts/truetype/arphic-gbsn00lp",
    "/usr/share/fonts/truetype/arphic",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/open-sans",
    "/usr/share/fonts/truetype/roboto",
    "/usr/share/fonts",
    "/System/Library/Fonts",
    "/Library/Fonts",
]

_CJK_FONT_CANDIDATES = [
    "NotoSansCJKsc-Regular.otf",
    "NotoSansCJK-Regular.ttc",
    "NotoSansSC-Regular.otf",
    "SourceHanSansSC-Regular.otf",
    "SourceHanSerifSC-Regular.otf",
    "SimSun.ttf",
    "simsun.ttc",
    "STSong.ttf",
    "Songti.ttc",
    "gbsn00lp.ttf",
    "uming.ttc",
    "wqy-zenhei.ttc",
    "wqy-microhei.ttc",
    "DroidSansFallbackFull.ttf",
    "PingFang.ttc",
    "Hiragino Sans GB.ttc",
    "Microsoft YaHei.ttf",
    "msyh.ttc",
]

_BOLD_CANDIDATES = [
    "NotoSansCJKsc-Bold.otf",
    "NotoSansCJK-Bold.ttc",
    "NotoSansSC-Bold.otf",
    "SourceHanSansSC-Bold.otf",
    "SourceHanSerifSC-Bold.otf",
    "SimHei.ttf",
    "simhei.ttf",
    "Microsoft YaHei Bold.ttf",
    "msyhbd.ttc",
    "wqy-zenhei.ttc",
    "wqy-microhei.ttc",
    "DroidSansFallbackFull.ttf",
    "PingFang.ttc",
    "gbsn00lp.ttf",
    "uming.ttc",
]

_LATIN_FONT_CANDIDATES = [
    "DejaVuSerif-Bold.ttf",
    "DejaVuSerif.ttf",
    "Georgia.ttf",
    "Times New Roman.ttf",
    "Roboto-Bold.ttf",
    "OpenSans-Bold.ttf",
]



def _find_font(candidates: list[str], explicit_path: str = "") -> Optional[str]:
    """Find a font file by searching known paths."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    for search_dir in _FONT_SEARCH_PATHS:
        if not os.path.isdir(search_dir):
            continue
        for root, _dirs, files in os.walk(search_dir):
            lower_to_name = {fname.lower(): fname for fname in files}
            for candidate in candidates:
                match = lower_to_name.get(candidate.lower())
                if match:
                    return os.path.join(root, match)
    return None



def _load_font(size_pt: int, bold: bool = False, explicit_path: str = "") -> ImageFont.FreeTypeFont:
    """Load a CJK-capable font, falling back to Pillow default."""
    candidates = _BOLD_CANDIDATES if bold else _CJK_FONT_CANDIDATES
    path = _find_font(candidates, explicit_path)
    if path:
        return ImageFont.truetype(path, size_pt)
    return ImageFont.load_default()



def _load_latin_font(size_pt: int, explicit_path: str = "") -> ImageFont.FreeTypeFont:
    """Load a Latin display font for pinyin and slogan."""
    path = _find_font(_LATIN_FONT_CANDIDATES, explicit_path)
    if path:
        return ImageFont.truetype(path, size_pt)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------



def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))



def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]



def _tracked_text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, tracking_px: int) -> int:
    if not text:
        return 0
    widths = [_text_size(draw, ch, font)[0] for ch in text]
    return sum(widths) + tracking_px * max(0, len(text) - 1)



def _draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str | tuple,
    tracking_px: int = 0,
) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        ch_w, _ = _text_size(draw, ch, font)
        x += ch_w + tracking_px



def _draw_tracked_text_center(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: str | tuple,
    tracking_px: int = 0,
) -> None:
    text_w = _tracked_text_width(draw, text, font, tracking_px)
    _draw_tracked_text(draw, (int(center_x - text_w / 2), y), text, font, fill, tracking_px)



def _fit_font(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    size_px: int,
    bold: bool,
    explicit_path: str = "",
    latin: bool = False,
    min_size: int = 20,
) -> ImageFont.ImageFont:
    """Shrink font until text fits within max_width."""
    size = size_px
    while size >= min_size:
        font = _load_latin_font(size, explicit_path) if latin else _load_font(size, bold=bold, explicit_path=explicit_path)
        if _text_size(draw, text, font)[0] <= max_width:
            return font
        size -= 4
    return _load_latin_font(min_size, explicit_path) if latin else _load_font(min_size, bold=bold, explicit_path=explicit_path)



def _resize_nearest(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    try:
        return img.resize(size, Image.Resampling.NEAREST)
    except AttributeError:
        return img.resize(size, Image.NEAREST)



def _make_tag_rgb(tag_img: Image.Image, tag_size: int) -> Image.Image:
    """Scale AprilTag and force it to crisp black/white RGB."""
    scaled = _resize_nearest(tag_img.convert("L"), (tag_size, tag_size))
    bw = scaled.point(lambda p: 0 if p < 128 else 255, mode="1")
    return bw.convert("RGB")



def _draw_gradient_rounded_rect(
    base: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    left_color: str,
    right_color: str,
) -> None:
    """Draw a horizontal gradient rounded rectangle onto base image."""
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    left = _hex_to_rgb(left_color)
    right = _hex_to_rgb(right_color)

    gradient = Image.new("RGB", (width, height), left)
    gdraw = ImageDraw.Draw(gradient)
    for x in range(width):
        t = x / max(1, width - 1)
        color = tuple(round(left[i] * (1 - t) + right[i] * t) for i in range(3))
        gdraw.line([(x, 0), (x, height)], fill=color)

    mask = Image.new("L", (width, height), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle([0, 0, width - 1, height - 1], radius=radius, fill=255)
    base.paste(gradient, (x1, y1), mask)



def _draw_footer(card: Image.Image, config: DeskCardConfig) -> None:
    """Draw curved red footer and the slogan."""
    w, h = card.size
    red = _hex_to_rgb(config.accent_red)

    # Quadratic Bezier approximating the sample's sweeping footer curve.
    start = (-int(w * 0.05), int(h * 0.765))
    ctrl = (int(w * 0.42), int(h * 0.865))
    end = (int(w * 1.05), int(h * 0.660))
    points = []
    for i in range(90):
        t = i / 89
        x = (1 - t) * (1 - t) * start[0] + 2 * (1 - t) * t * ctrl[0] + t * t * end[0]
        y = (1 - t) * (1 - t) * start[1] + 2 * (1 - t) * t * ctrl[1] + t * t * end[1]
        points.append((int(x), int(y)))

    # Soft shadow above the red footer.
    shadow = Image.new("RGBA", card.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sdraw.line(points, fill=(0, 0, 0, 75), width=max(8, int(h * 0.020)))
    sdraw.line(points, fill=(255, 255, 255, 200), width=max(3, int(h * 0.006)))
    composited = Image.alpha_composite(card.convert("RGBA"), shadow).convert("RGB")
    card.paste(composited)

    draw = ImageDraw.Draw(card)
    polygon = points + [(w + 20, h + 20), (-20, h + 20)]
    draw.polygon(polygon, fill=red)

    # Slogan with wide tracking.
    slogan = config.slogan
    slogan_font_size = config.slogan_font_size_pt
    slogan_font = _load_latin_font(slogan_font_size, config.font_path_latin)
    max_slogan_w = int(w * 0.68)
    tracking = max(4, int(w * 0.010))
    while _tracked_text_width(draw, slogan, slogan_font, tracking) > max_slogan_w and slogan_font_size > 20:
        slogan_font_size -= 2
        slogan_font = _load_latin_font(slogan_font_size, config.font_path_latin)

    slogan_y = int(h * 0.890)
    _draw_tracked_text_center(draw, w // 2, slogan_y, slogan, slogan_font, "white", tracking)


# ---------------------------------------------------------------------------
# Tag image loading
# ---------------------------------------------------------------------------



def load_tag_image(tag_id_int: int, tag_dir: str = "tagStandard41h12") -> Image.Image:
    """Load a pre-generated AprilTag PNG image."""
    fname = tag_filename(tag_id_int)
    path = os.path.join(tag_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Tag image not found: {path}\n"
            f"Expected file for tag ID {tag_id_int} (label {int_to_tag_id(tag_id_int)})."
        )

    img = Image.open(path)
    if img.mode == "RGBA":
        bg = Image.new("L", img.size, 255)
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("L")


# ---------------------------------------------------------------------------
# Desk-card generation
# ---------------------------------------------------------------------------



def generate_desk_card_png(
    tag_label: str,
    name: str,
    role: str = "",
    organization: str = "",
    tag_dir: str = "tagStandard41h12",
    output_path: Optional[str] = None,
    config: Optional[DeskCardConfig] = None,
    pinyin: str = "",
) -> Image.Image:
    """Generate one horizontal desk card as a PIL Image.

    Args:
        tag_label: Human-readable tag ID, for example "A001".
        name: Participant name, Chinese supported if a CJK font is available.
        role: Participant role. Drawn inside the red rounded pill.
        organization: Event title. If empty, config.event_title is used.
        tag_dir: Path to the directory containing tag41_12_XXXXX.png files.
        output_path: If provided, save PNG to this path.
        config: Layout configuration.
        pinyin: Optional romanization shown below the Chinese name.

    Returns:
        PIL Image of the desk card.
    """
    if config is None:
        config = DeskCardConfig()

    w, h = config.card_size_px
    margin = config._margin_px
    tag_size = config._tag_size_px

    card = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(card)

    # Load fonts.
    font_name = _load_font(config.name_font_size_pt, bold=True, explicit_path=config.font_path_bold)
    font_pinyin = _load_latin_font(config.pinyin_font_size_pt, config.font_path_latin)
    font_event = _load_font(config.event_font_size_pt, bold=False, explicit_path=config.font_path_regular)
    font_role = _load_font(config.role_font_size_pt, bold=True, explicit_path=config.font_path_bold)
    font_info = _load_font(config.info_font_size_pt, bold=False, explicit_path=config.font_path_regular)

    # Footer first so foreground elements stay clean.
    _draw_footer(card, config)
    draw = ImageDraw.Draw(card)

    # Top-left AprilTag.
    tag_x = margin
    tag_y = margin
    try:
        tag_int = tag_id_to_int(tag_label)
        tag_img = load_tag_image(tag_int, tag_dir)
        tag_rgb = _make_tag_rgb(tag_img, tag_size)
        card.paste(tag_rgb, (tag_x, tag_y))
    except FileNotFoundError:
        # Missing tag: draw a clear placeholder but keep the layout intact.
        draw.rectangle([tag_x, tag_y, tag_x + tag_size, tag_y + tag_size], outline="red", width=max(4, tag_size // 90))
        label_font = _fit_font(draw, tag_label, int(tag_size * 0.70), max(24, tag_size // 9), False)
        tw, th = _text_size(draw, tag_label, label_font)
        draw.text((tag_x + (tag_size - tw) / 2, tag_y + (tag_size - th) / 2), tag_label, fill="red", font=label_font)

    # Keep a slim dark outer stroke like the sample image.
    border_w = max(6, int(w * 0.003))
    draw.rectangle(
        [tag_x, tag_y, tag_x + tag_size, tag_y + tag_size],
        outline="#202424",
        width=border_w,
    )

    # Main name block.
    divider_x = int(w * 0.690)
    name_center_x = int(w * 0.495)
    name_y = int(h * 0.245)
    name_tracking = max(14, int(w * 0.018))
    display_name = name.strip()
    if len(display_name) > 1 and " " not in display_name:
        # Track CJK names so they read visually like "肖 翔".
        display_name = "".join(display_name)
    _draw_tracked_text_center(
        draw,
        name_center_x,
        name_y,
        display_name,
        font_name,
        config.text_black,
        name_tracking if len(display_name) <= 4 else max(2, name_tracking // 3),
    )

    if pinyin:
        pinyin_text = pinyin.strip().lower()
        pinyin_y = int(h * 0.510)
        pinyin_tracking = max(10, int(w * 0.012))
        _draw_tracked_text_center(draw, name_center_x, pinyin_y, pinyin_text, font_pinyin, config.text_black, pinyin_tracking)

    # Vertical red separator.
    divider_top = int(h * 0.270)
    divider_bottom = int(h * 0.535)
    divider_w = max(4, int(w * 0.0022))
    draw.rounded_rectangle(
        [divider_x - divider_w // 2, divider_top, divider_x + divider_w // 2, divider_bottom],
        radius=divider_w,
        fill=config.accent_red,
    )

    # Right event title and role pill.
    right_center_x = int(w * 0.835)
    event_text = organization.strip() or config.event_title
    event_font = _fit_font(draw, event_text, int(w * 0.250), config.event_font_size_pt, False, config.font_path_regular)
    event_w, event_h = _text_size(draw, event_text, event_font)
    event_y = int(h * 0.305)
    draw.text((right_center_x - event_w / 2, event_y), event_text, fill=config.text_black, font=event_font)

    if role:
        role_font = _fit_font(draw, role, int(w * 0.170), config.role_font_size_pt, True, config.font_path_bold, min_size=28)
        role_w, role_h = _text_size(draw, role, role_font)
        pill_w = max(int(w * 0.185), role_w + int(w * 0.070))
        pill_h = int(h * 0.105)
        pill_x1 = int(right_center_x - pill_w / 2)
        pill_y1 = int(h * 0.420)
        pill_x2 = int(pill_x1 + pill_w)
        pill_y2 = int(pill_y1 + pill_h)
        _draw_gradient_rounded_rect(
            card,
            (pill_x1, pill_y1, pill_x2, pill_y2),
            radius=pill_h // 2,
            left_color=config.pill_red_left,
            right_color=config.pill_red_right,
        )
        draw = ImageDraw.Draw(card)
        draw.text(
            (pill_x1 + (pill_w - role_w) / 2, pill_y1 + (pill_h - role_h) / 2 - int(h * 0.004)),
            role,
            fill="white",
            font=role_font,
        )

    # Small tag ID, optional but useful in batch output / QA.
    if tag_label:
        tag_id_text = tag_label
        tag_id_w, tag_id_h = _text_size(draw, tag_id_text, font_info)
        draw.text((margin, h - margin - tag_id_h), tag_id_text, fill="white", font=font_info)

    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        card.save(output_path, "PNG")

    return card



def generate_desk_cards_for_participants(
    participants: list[dict],
    output_dir: str = "exports/desk_cards",
    tag_dir: str = "tagStandard41h12",
    config: Optional[DeskCardConfig] = None,
) -> dict:
    """Generate PNG desk cards for all participants plus a combined PDF.

    Expected participant keys:
        tag_id, name, role, organization, pinyin
    """
    if config is None:
        config = DeskCardConfig()

    os.makedirs(output_dir, exist_ok=True)
    generated = []
    errors = []

    for p in participants:
        tag_label = p.get("tag_id", "")
        name = p.get("name", "")
        if not tag_label or not name:
            errors.append(f"Missing tag_id or name: {p}")
            continue

        safe_name = name.replace("/", "_").replace(" ", "_")
        output_path = os.path.join(output_dir, f"{tag_label}_{safe_name}.png")

        try:
            generate_desk_card_png(
                tag_label=tag_label,
                name=name,
                role=p.get("role", ""),
                organization=p.get("organization", p.get("event_title", "")),
                pinyin=p.get("pinyin", ""),
                tag_dir=tag_dir,
                output_path=output_path,
                config=config,
            )
            generated.append(output_path)
        except Exception as e:
            errors.append(f"{tag_label} {name}: {e}")

    pdf_path = ""
    if generated:
        pdf_path = os.path.join(output_dir, "desk_cards_combined.pdf")
        try:
            generate_desk_card_pdf(generated, pdf_path, config=config)
        except Exception as e:
            errors.append(f"PDF generation: {e}")
            pdf_path = ""

    return {"generated": generated, "pdf": pdf_path, "errors": errors}


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------



def generate_desk_card_pdf(
    card_paths: list[str],
    output_pdf_path: str,
    cards_per_page: Optional[int] = None,
    config: Optional[DeskCardConfig] = None,
) -> str:
    """Combine individual PNG desk cards into a printable A4 PDF.

    One card per page, centered on the page.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    if config is None:
        config = DeskCardConfig()

    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)

    page_w, page_h = A4
    card_w_mm, card_h_mm = config.card_size_mm
    card_w_pt = card_w_mm * mm
    card_h_pt = card_h_mm * mm

    # One card per page, centered.
    c = canvas.Canvas(output_pdf_path, pagesize=A4)

    for i, png_path in enumerate(card_paths):
        if i > 0:
            c.showPage()

        x = (page_w - card_w_pt) / 2
        y = (page_h - card_h_pt) / 2

        c.drawImage(
            png_path,
            x,
            y,
            width=card_w_pt,
            height=card_h_pt,
            preserveAspectRatio=True,
            anchor="c",
        )

    c.showPage()
    c.save()
    return output_pdf_path


if __name__ == "__main__":
    # Minimal manual test. Requires tagStandard41h12/tag41_12_00000.png.
    generate_desk_card_png(
        tag_label="A001",
        name="肖翔",
        pinyin="xiao xiang",
        role="算法工程师",
        organization="嵌入式大赛",
        output_path="exports/desk_cards/A001_肖翔.png",
    )
