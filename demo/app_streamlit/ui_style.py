"""Shared Streamlit UI styling for the Smart Meeting Tracker app.

This module intentionally contains presentation-layer helpers only: CSS tokens,
page headers, sidebar navigation, and lightweight visual affordances.
"""

from __future__ import annotations

import streamlit as st


OFFICE_THEME_CSS = """
<style>
:root {
  --office-blue: #1664ff;
  --office-blue-soft: #e8f1ff;
  --office-bg: #f7f8fa;
  --office-surface: #ffffff;
  --office-surface-soft: #fbfcff;
  --office-border: #e5e6eb;
  --office-border-strong: #d8dadf;
  --office-text: #1f2329;
  --office-text-secondary: #646a73;
  --office-text-tertiary: #8f959e;
  --office-success: #00b578;
  --office-warning: #ff7d00;
  --office-danger: #f53f3f;
  --office-radius: 10px;
  --office-radius-lg: 14px;
  --office-shadow: 0 8px 24px rgba(31, 35, 41, 0.06);
}

html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
}

.stApp {
  background: var(--office-bg);
  color: var(--office-text);
}

.block-container {
  padding-top: 1.15rem;
  padding-bottom: 2.5rem;
  max-width: 1320px;
}

[data-testid="stHeader"] {
  background: rgba(247, 248, 250, 0.86);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(229, 230, 235, 0.72);
}

[data-testid="stSidebar"] {
  background: var(--office-surface);
  border-right: 1px solid var(--office-border);
}

[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding-top: 1.2rem;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  font-size: 0.98rem;
  font-weight: 650;
  color: var(--office-text);
  letter-spacing: -0.01em;
}

h1 {
  font-size: 1.72rem !important;
  line-height: 1.22 !important;
  font-weight: 680 !important;
  letter-spacing: -0.025em !important;
  color: var(--office-text) !important;
}

h2 {
  font-size: 1.28rem !important;
  font-weight: 660 !important;
  letter-spacing: -0.015em !important;
}

h3 {
  font-size: 1.08rem !important;
  font-weight: 640 !important;
}

p, li, label, .stCaption, [data-testid="stMarkdownContainer"] {
  color: var(--office-text-secondary);
}

small, [data-testid="stCaptionContainer"] {
  color: var(--office-text-tertiary) !important;
}

hr {
  margin: 1rem 0;
  border-color: var(--office-border);
}

.office-hero {
  background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
  border: 1px solid var(--office-border);
  border-radius: var(--office-radius-lg);
  padding: 22px 24px;
  box-shadow: var(--office-shadow);
  margin-bottom: 18px;
}

.office-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 9px;
  border-radius: 999px;
  background: var(--office-blue-soft);
  color: var(--office-blue);
  font-size: 12px;
  font-weight: 650;
  margin-bottom: 10px;
}

.office-title {
  margin: 0;
  color: var(--office-text);
  font-size: 27px;
  line-height: 1.2;
  font-weight: 700;
  letter-spacing: -0.03em;
}

.office-subtitle {
  max-width: 860px;
  margin: 8px 0 0 0;
  color: var(--office-text-secondary);
  font-size: 14px;
  line-height: 1.65;
}

.office-card {
  height: 100%;
  background: var(--office-surface);
  border: 1px solid var(--office-border);
  border-radius: var(--office-radius-lg);
  padding: 18px;
  box-shadow: 0 4px 18px rgba(31, 35, 41, 0.04);
}

.office-card:hover {
  border-color: rgba(22, 100, 255, 0.26);
  box-shadow: var(--office-shadow);
}

.office-card-title {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0 0 6px 0;
  color: var(--office-text);
  font-size: 16px;
  line-height: 1.3;
  font-weight: 680;
}

.office-card-desc {
  min-height: 42px;
  color: var(--office-text-secondary);
  font-size: 13px;
  line-height: 1.6;
  margin-bottom: 12px;
}

.office-card-meta {
  color: var(--office-text-tertiary);
  font-size: 12px;
  line-height: 1.45;
}

.office-note {
  background: #f2f6ff;
  border: 1px solid #d7e6ff;
  border-left: 3px solid var(--office-blue);
  border-radius: 10px;
  padding: 12px 14px;
  color: var(--office-text-secondary);
  font-size: 13px;
  line-height: 1.6;
}

.office-section-label {
  margin: 4px 0 10px 0;
  color: var(--office-text-tertiary);
  font-size: 12px;
  font-weight: 650;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

.office-stepper {
  display: grid;
  gap: 8px;
  margin: 8px 0 4px 0;
}

.office-step {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  border: 1px solid var(--office-border);
  border-radius: 10px;
  background: #fff;
  color: var(--office-text-secondary);
  font-size: 13px;
}

.office-step .office-step-index {
  width: 24px;
  height: 24px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #f2f3f5;
  color: var(--office-text-tertiary);
  font-size: 12px;
  font-weight: 700;
}

.office-step.active {
  border-color: rgba(22, 100, 255, 0.36);
  background: #f5f9ff;
  color: var(--office-text);
  font-weight: 650;
}

.office-step.active .office-step-index {
  background: var(--office-blue);
  color: #fff;
}

.office-step.done {
  background: #f4fbf8;
  border-color: #b9ead7;
  color: var(--office-success);
}

.office-step.done .office-step-index {
  background: var(--office-success);
  color: #fff;
}

.office-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid var(--office-border);
  padding: 2px 9px;
  background: #fff;
  color: var(--office-text-secondary);
  font-size: 12px;
  font-weight: 550;
}

div[data-testid="stMetric"] {
  background: var(--office-surface);
  border: 1px solid var(--office-border);
  border-radius: var(--office-radius-lg);
  padding: 14px 16px;
  box-shadow: 0 4px 18px rgba(31, 35, 41, 0.035);
}

[data-testid="stMetricLabel"] {
  color: var(--office-text-tertiary) !important;
  font-size: 12px !important;
}

[data-testid="stMetricValue"] {
  color: var(--office-text) !important;
  font-size: 1.3rem !important;
  font-weight: 700 !important;
}

.stButton > button,
.stDownloadButton > button,
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"] {
  border-radius: 9px !important;
  min-height: 36px;
  border: 1px solid var(--office-border-strong) !important;
  font-weight: 570 !important;
  box-shadow: none !important;
}

button[data-testid="baseButton-primary"],
.stButton > button[kind="primary"] {
  background: var(--office-blue) !important;
  border-color: var(--office-blue) !important;
  color: #fff !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color: var(--office-blue) !important;
  color: var(--office-blue) !important;
}

button[data-testid="baseButton-primary"]:hover {
  filter: brightness(0.96);
  color: #fff !important;
}

div[data-testid="stForm"] button[kind="primary"],
div[data-testid="stForm"] button[data-testid="baseButton-primary"],
div[data-testid="stForm"] button[data-testid="stFormSubmitButton"],
div[data-testid="stForm"] div[data-testid="stButton"] button,
div[data-testid="stForm"] > div > div > button {
  background: var(--office-blue) !important;
  border-color: var(--office-blue) !important;
  color: #fff !important;
}

div[data-testid="stForm"] button[data-testid="baseButton-primary"]:hover,
div[data-testid="stForm"] button[data-testid="stFormSubmitButton"]:hover,
div[data-testid="stForm"] div[data-testid="stButton"] button:hover,
div[data-testid="stForm"] > div > div > button:hover {
  filter: brightness(0.96);
  color: #fff !important;
}

button [data-testid="stMarkdownContainer"],
button [data-testid="stMarkdownContainer"] p {
  color: inherit !important;
}

[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input {
  border-radius: 9px !important;
  border-color: var(--office-border) !important;
  background: #fff !important;
}

[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-testid="stNumberInput"] input:focus {
  border-color: var(--office-blue) !important;
  box-shadow: 0 0 0 3px rgba(22, 100, 255, 0.12) !important;
}

[data-testid="stFileUploader"] {
  background: var(--office-surface-soft);
  border: 1px dashed var(--office-border-strong);
  border-radius: var(--office-radius-lg);
  padding: 12px;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border: 1px solid var(--office-border);
  border-radius: var(--office-radius-lg);
  overflow: hidden;
  background: #fff;
}

div[data-testid="stExpander"] {
  border: 1px solid var(--office-border) !important;
  border-radius: var(--office-radius-lg) !important;
  background: var(--office-surface) !important;
  box-shadow: 0 3px 14px rgba(31, 35, 41, 0.03);
}

[data-testid="stAlert"] {
  border-radius: var(--office-radius) !important;
  border: 1px solid var(--office-border) !important;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  border-bottom: 1px solid var(--office-border);
}

.stTabs [data-baseweb="tab"] {
  height: 38px;
  padding: 8px 12px;
  border-radius: 9px 9px 0 0;
  color: var(--office-text-secondary);
  font-weight: 560;
}

.stTabs [aria-selected="true"] {
  background: #fff;
  color: var(--office-blue) !important;
}

div[data-testid="stForm"] {
  background: var(--office-surface);
  border: 1px solid var(--office-border);
  border-radius: var(--office-radius-lg);
  padding: 18px;
  box-shadow: 0 4px 18px rgba(31, 35, 41, 0.035);
}

code {
  color: #2454a6 !important;
  background: #eef4ff !important;
  border-radius: 6px;
  padding: 1px 5px;
}

@media (max-width: 900px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .office-hero { padding: 18px; }
  .office-title { font-size: 23px; }
}
</style>
"""


def apply_office_theme() -> None:
    """Apply the concise enterprise-office visual style."""
    st.markdown(OFFICE_THEME_CSS, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "", eyebrow: str = "智会追声") -> None:
    """Render a compact Feishu/DingTalk-like page header."""
    subtitle_html = f'<p class="office-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <section class="office-hero">
          <div class="office-eyebrow">{eyebrow}</div>
          <h1 class="office-title">{title}</h1>
          {subtitle_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_stepper(steps: list[str], current_step: int) -> None:
    """Render a compact vertical stepper for the preparation workflow."""
    items = []
    for index, name in enumerate(steps):
        state = "active" if index == current_step else "done" if index < current_step else ""
        marker = "✓" if index < current_step else str(index + 1)
        items.append(
            f'<div class="office-step {state}">'
            f'<span class="office-step-index">{marker}</span>'
            f'<span>{name}</span></div>'
        )
    st.markdown('<div class="office-stepper">' + ''.join(items) + '</div>', unsafe_allow_html=True)


def render_sidebar_nav(active: str | None = None) -> None:
    """Render app-level navigation links in a consistent order."""
    st.markdown('<div class="office-section-label">导航</div>', unsafe_allow_html=True)
    st.page_link("Home.py", label="首页", icon="🏠")
    st.page_link("pages/01_会前准备.py", label="会前准备", icon="📋")
    st.page_link("pages/02_会议控制台.py", label="会议控制台", icon="🎛️")
    st.page_link("pages/03_会议记录与总结.py", label="会议记录与总结", icon="📊")
    st.page_link("pages/05_Agent控制台.py", label="Agent 控制台", icon="🤖")
    st.page_link("pages/04_数据库调试.py", label="数据库调试", icon="🗄️")


def render_office_note(text: str) -> None:
    """Render a concise informational note."""
    st.markdown(f'<div class="office-note">{text}</div>', unsafe_allow_html=True)
