"""智会追声 — 主持人控制台 Home。"""

import streamlit as st

from ui_style import apply_office_theme, render_office_note, render_page_header

st.set_page_config(
    page_title="智会追声",
    page_icon="🎤",
    layout="wide",
)
apply_office_theme()

render_page_header(
    "智会追声 · 智能会议追踪系统",
    "面向会前准备、实时导播、会后复盘和 Agent 播报的一体化会议工作台。界面已收敛为更轻量、清晰、办公软件化的控制台风格。",
    "Smart Meeting Tracker",
)

st.markdown('<div class="office-section-label">功能模块</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="office-card">
      <div class="office-card-title">📋 会前准备</div>
      <div class="office-card-desc">创建会议、导入人员名单、验证标签绑定、生成桌牌并完成摄像头扫描自检。</div>
      <div class="office-card-meta">建议先完成此模块，再进入实时控制台。</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/01_会前准备.py", label="进入会前准备", icon="📋")

with col2:
    st.markdown("""
    <div class="office-card">
      <div class="office-card-title">🎛️ 会议控制台</div>
      <div class="office-card-desc">实时状态监控、发言人识别、导播控制、事件时间轴与主持人备注。</div>
      <div class="office-card-meta">会议进行中主要工作区。</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/02_会议控制台.py", label="进入会议控制台", icon="🎛️")

with col3:
    st.markdown("""
    <div class="office-card">
      <div class="office-card-title">📊 会议记录与总结</div>
      <div class="office-card-desc">导出时间轴、查看发言统计、生成会后摘要、评委问题整理和系统运行报告。</div>
      <div class="office-card-meta">会后复盘与归档区域。</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/03_会议记录与总结.py", label="查看会议记录", icon="📊")

st.markdown("")
col4, col5 = st.columns(2)
with col4:
    st.markdown("""
    <div class="office-card">
      <div class="office-card-title">🤖 Agent 控制台</div>
      <div class="office-card-desc">查看 LLM Agent 决策审计、TTS 播报状态，并手动触发阶段总结或议题提醒。</div>
      <div class="office-card-meta">适合主持人或系统管理员使用。</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/05_Agent控制台.py", label="进入 Agent 控制台", icon="🤖")

with col5:
    st.markdown("""
    <div class="office-card">
      <div class="office-card-title">🗄️ 数据库调试</div>
      <div class="office-card-desc">查看业务表、调试 CRUD 和原始 SQL。该模块仅建议在开发环境使用。</div>
      <div class="office-card-meta">高风险操作已保留原有确认逻辑。</div>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/04_数据库调试.py", label="打开数据库调试", icon="🗄️")

st.divider()
render_office_note(
    "快速开始：先启动 fusion_tracker 控制 API，再运行 <code>streamlit run app_streamlit/Home.py --server.port 8501</code>。"
)

st.caption("智会追声 · Concise Office UI")
