#!/usr/bin/env python3
"""
Desk card generator — produces individual PNG desk cards and combined PDF.

Uses Pillow for PNG generation and ReportLab for PDF assembly.
Loads pre-generated AprilTag images from the tagStandard41h12/ directory.
Zero PyQt5 dependency — safe for headless/Streamlit use.

Tag ID mapping:
  - Integer 0 → filename "tag41_12_00000.png" → human label "A001"
  - Integer N → filename "tag41_12_{N:05d}.png" → human label f"A{N+1:03d}"
"""

import json
import os
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Tag ID ↔ integer mapping
# ---------------------------------------------------------------------------

def tag_id_to_int(label: str) -> int:
    """Convert human-readable tag label to integer ID.

    "A001" → 0, "A023" → 22, "A100" → 99
    """
    if not label or not label[0].isalpha():
        raise ValueError(f"Invalid tag label: {label!r}, expected like 'A001'")
    return int(label[1:]) - 1


def int_to_tag_id(n: int) -> str:
    """Convert integer ID to human-readable tag label.

    0 → "A001", 99 → "A100"
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
    """Layout configuration for desk card generation."""

    card_size_mm: tuple = (105, 148)  # A6 portrait
    dpi: int = 300
    tag_display_mm: int = 35
    margin_mm: int = 5
    font_path_regular: str = ""
    font_path_bold: str = ""
    name_font_size_pt: int = 72
    role_font_size_pt: int = 36
    info_font_size_pt: int = 20

    # Derived pixel dimensions (computed in __post_init__ or lazily)
    _card_w_px: int = field(default=0, repr=False)
    _card_h_px: int = field(default=0, repr=False)
    _tag_size_px: int = field(default=0, repr=False)
    _margin_px: int = field(default=0, repr=False)

    def __post_init__(self):
        px_per_mm = self.dpi / 25.4
        self._card_w_px = round(self.card_size_mm[0] * px_per_mm)
        self._card_h_px = round(self.card_size_mm[1] * px_per_mm)
        self._tag_size_px = round(self.tag_display_mm * px_per_mm)
        self._margin_px = round(self.margin_mm * px_per_mm)

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
            "name_font_size_pt": self.name_font_size_pt,
            "role_font_size_pt": self.role_font_size_pt,
            "info_font_size_pt": self.info_font_size_pt,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeskCardConfig":
        card_size = tuple(d.get("card_size_mm", [105, 148]))
        return cls(
            card_size_mm=card_size,
            dpi=d.get("dpi", 300),
            tag_display_mm=d.get("tag_display_mm", 35),
            margin_mm=d.get("margin_mm", 5),
            font_path_regular=d.get("font_path_regular", ""),
            font_path_bold=d.get("font_path_bold", ""),
            name_font_size_pt=d.get("name_font_size_pt", 72),
            role_font_size_pt=d.get("role_font_size_pt", 36),
            info_font_size_pt=d.get("info_font_size_pt", 20),
        )

    @classmethod
    def from_json(cls, path: str) -> "DeskCardConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

_FONT_SEARCH_PATHS = [
    # Bundled fonts (if user installs them)
    "assets/fonts",
    # Common Linux CJK font paths
    "/usr/share/fonts/opentype/noto",
    "/usr/share/fonts/truetype/noto",
    "/usr/share/fonts/noto-cjk",
    "/usr/share/fonts/google-noto-cjk",
    "/usr/share/fonts/truetype/wqy",
    "/usr/share/fonts/truetype/droid",
    "/usr/share/fonts",
    # macOS
    "/System/Library/Fonts",
    "/Library/Fonts",
]

_CJK_FONT_CANDIDATES = [
    "NotoSansCJKsc-Regular.otf",
    "NotoSansCJK-Regular.ttc",
    "NotoSansSC-Regular.otf",
    "NotoSansMonoCJKsc-Regular.otf",
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
    "wqy-zenhei.ttc",
    "wqy-microhei.ttc",
    "DroidSansFallbackFull.ttf",
    "PingFang.ttc",
]


def _find_font(candidates: list[str], explicit_path: str = "") -> Optional[str]:
    """Find a font file by searching known paths."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    for search_dir in _FONT_SEARCH_PATHS:
        if not os.path.isdir(search_dir):
            continue
        for root, _dirs, files in os.walk(search_dir):
            for candidate in candidates:
                for fname in files:
                    if fname.lower() == candidate.lower():
                        return os.path.join(root, fname)
    return None


def _load_font(size_pt: int, bold: bool = False, explicit_path: str = "") -> ImageFont.FreeTypeFont:
    """Load a CJK-capable font, falling back to Pillow default."""
    candidates = _BOLD_CANDIDATES if bold else _CJK_FONT_CANDIDATES
    path = _find_font(candidates, explicit_path)
    if path:
        return ImageFont.truetype(path, size_pt)
    # Fallback: Pillow default font (no CJK support, but won't crash)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Tag image loading
# ---------------------------------------------------------------------------

def load_tag_image(
    tag_id_int: int,
    tag_dir: str = "tagStandard41h12",
) -> Image.Image:
    """Load a pre-generated AprilTag PNG image.

    Args:
        tag_id_int: Integer tag ID (0–2116 for tagStandard41h12).
        tag_dir: Path to the directory containing tag41_12_XXXXX.png files.

    Returns:
        PIL Image (grayscale "L" mode) of the tag bitmap.

    Raises:
        FileNotFoundError: If the tag image doesn't exist.
    """
    fname = tag_filename(tag_id_int)
    path = os.path.join(tag_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Tag image not found: {path}\n"
            f"Expected file for tag ID {tag_id_int} (label {int_to_tag_id(tag_id_int)})."
        )
    img = Image.open(path)
    # Convert RGBA → grayscale (L) for clean compositing
    if img.mode == "RGBA":
        # Composite onto white background to handle alpha
        bg = Image.new("L", img.size, 255)
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("L")


# ---------------------------------------------------------------------------
# Desk card generation
# ---------------------------------------------------------------------------

def generate_desk_card_png(
    tag_label: str,
    name: str,
    role: str = "",
    organization: str = "",
    tag_dir: str = "tagStandard41h12",
    output_path: Optional[str] = None,
    config: Optional[DeskCardConfig] = None,
) -> Image.Image:
    """Generate a single desk card as a PIL Image.

    Args:
        tag_label: Human-readable tag ID, e.g. "A001".
        name: Participant name (Chinese supported if CJK font available).
        role: Participant role/position.
        organization: Organization name.
        tag_dir: Path to tagStandard41h12/ directory.
        output_path: If provided, save PNG to this path.
        config: Layout configuration (defaults to A6 portrait).

    Returns:
        PIL Image of the desk card.
    """
    if config is None:
        config = DeskCardConfig()

    w, h = config.card_size_px
    margin = config._margin_px
    tag_size = config._tag_size_px

    # Create white canvas
    card = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(card)

    # Load fonts
    font_name = _load_font(config.name_font_size_pt, bold=True, explicit_path=config.font_path_bold)
    font_role = _load_font(config.role_font_size_pt, bold=False, explicit_path=config.font_path_regular)
    font_info = _load_font(config.info_font_size_pt, bold=False, explicit_path=config.font_path_regular)

    # --- Name ---
    bbox = draw.textbbox((0, 0), name, font=font_name)
    name_w = bbox[2] - bbox[0]
    name_h = bbox[3] - bbox[1]
    name_y = margin + 20  # small top offset
    draw.text(((w - name_w) / 2, name_y), name, fill="black", font=font_name)

    # --- Role ---
    if role:
        bbox = draw.textbbox((0, 0), role, font=font_role)
        role_w = bbox[2] - bbox[0]
        role_y = name_y + name_h + 10
        draw.text(((w - role_w) / 2, role_y), role, fill="#555555", font=font_role)
        next_y = role_y + (bbox[3] - bbox[1]) + 30
    else:
        next_y = name_y + name_h + 30

    # --- AprilTag image (NEAREST interpolation for sharp edges) ---
    try:
        tag_int = tag_id_to_int(tag_label)
        tag_img = load_tag_image(tag_int, tag_dir)
        # Scale with NEAREST — no interpolation, preserve hard black/white edges
        tag_scaled = tag_img.resize((tag_size, tag_size), Image.NEAREST)
        # Convert grayscale to RGB white-on-black tag
        tag_rgb = Image.new("RGB", (tag_size, tag_size), "white")
        for y in range(tag_size):
            for x in range(tag_size):
                pixel = tag_scaled.getpixel((x, y))
                # In the source 9x9 images: dark pixels are the tag marker
                # We draw black markers on white background
                if pixel < 128:
                    tag_rgb.putpixel((x, y), (0, 0, 0))

        tag_x = (w - tag_size) // 2
        tag_y = next_y
        card.paste(tag_rgb, (tag_x, tag_y))

        # Border around tag
        draw.rectangle(
            [tag_x - 2, tag_y - 2, tag_x + tag_size + 1, tag_y + tag_size + 1],
            outline="black",
            width=2,
        )
        tag_bottom = tag_y + tag_size
    except FileNotFoundError:
        # Tag image missing — draw a placeholder
        tag_bottom = next_y
        draw.rectangle(
            [(w - tag_size) // 2, next_y, (w + tag_size) // 2, next_y + tag_size],
            outline="red",
            width=2,
        )
        draw.text(
            ((w - tag_size) // 2 + 10, next_y + tag_size // 2 - 10),
            f"[{tag_label}]",
            fill="red",
            font=font_info,
        )
        tag_bottom = next_y + tag_size

    # --- Bottom info: tag_id | organization ---
    info_text = tag_label
    if organization:
        info_text += f"  |  {organization}"
    bbox = draw.textbbox((0, 0), info_text, font=font_info)
    info_w = bbox[2] - bbox[0]
    info_y = max(tag_bottom + 30, h - margin - (bbox[3] - bbox[1]) - 10)
    draw.text(((w - info_w) / 2, info_y), info_text, fill="#333333", font=font_info)

    # --- Thin border around entire card ---
    draw.rectangle([0, 0, w - 1, h - 1], outline="#cccccc", width=1)

    # Save if requested
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        card.save(output_path, "PNG")

    return card


def generate_desk_cards_for_participants(
    participants: list[dict],
    output_dir: str = "exports/desk_cards",
    tag_dir: str = "tagStandard41h12",
    config: Optional[DeskCardConfig] = None,
) -> dict:
    """Generate individual PNG desk cards for all participants + a combined PDF.

    Args:
        participants: List of dicts with keys: tag_id, name, role, organization.
        output_dir: Directory for generated PNGs and PDF.
        tag_dir: Path to tagStandard41h12/ directory.
        config: Layout configuration.

    Returns:
        {"generated": ["path/A001_王强.png", ...], "pdf": "path/...pdf", "errors": [...]}
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
                organization=p.get("organization", ""),
                tag_dir=tag_dir,
                output_path=output_path,
                config=config,
            )
            generated.append(output_path)
        except Exception as e:
            errors.append(f"{tag_label} {name}: {e}")

    # Generate combined PDF
    pdf_path = ""
    if generated:
        pdf_path = os.path.join(output_dir, "desk_cards_combined.pdf")
        try:
            generate_desk_card_pdf(generated, pdf_path)
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
    cards_per_page: int = 4,
) -> str:
    """Combine individual PNG desk cards into a single printable PDF.

    Uses ReportLab for PDF generation.
    A4 page (210×297mm). Four A6 cards (105×148mm each) fit in a 2×2 grid.

    Args:
        card_paths: List of paths to individual desk card PNG files.
        output_pdf_path: Path for the output PDF file.
        cards_per_page: Number of cards per A4 page (2 or 4; default 4).

    Returns:
        Path to the generated PDF.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)

    page_w, page_h = A4  # points (1 point = 1/72 inch)

    # A6 = 105×148 mm
    card_w_mm = 105.0
    card_h_mm = 148.0
    card_w_pt = card_w_mm * mm
    card_h_pt = card_h_mm * mm

    c = canvas.Canvas(output_pdf_path, pagesize=A4)

    # 2 columns × 2 rows
    cols = 2
    rows = cards_per_page // cols if cards_per_page >= cols else 1

    # Calculate offsets to center the grid on the page
    grid_w = cols * card_w_pt
    grid_h = rows * card_h_pt
    offset_x = (page_w - grid_w) / 2
    offset_y = (page_h - grid_h) / 2

    for i, png_path in enumerate(card_paths):
        page_idx = i // cards_per_page
        slot_idx = i % cards_per_page

        if slot_idx == 0 and i > 0:
            c.showPage()

        col = slot_idx % cols
        row = slot_idx % rows
        x = offset_x + col * card_w_pt
        y = page_h - offset_y - (row + 1) * card_h_pt  # PDF y-axis is bottom-up

        c.drawImage(
            png_path,
            x, y,
            width=card_w_pt,
            height=card_h_pt,
            preserveAspectRatio=True,
            anchor="c",
        )

    c.showPage()
    c.save()
    return output_pdf_path
