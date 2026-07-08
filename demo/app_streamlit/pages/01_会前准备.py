"""会前准备 — 创建会议、导入人员、验证绑定、生成桌牌、摄像头自检。

五步向导流程，使用 st.session_state.prep_step 导航。
"""

import streamlit as st
import pandas as pd
import os
import sys
import io
import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.environ.get(
    "MEETING_TRACKER_ROOT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Shared presentation layer — concise office UI styling
# ---------------------------------------------------------------------------
_APP_STREAMLIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _APP_STREAMLIT_DIR not in sys.path:
    sys.path.insert(0, _APP_STREAMLIT_DIR)
from ui_style import apply_office_theme, render_page_header, render_sidebar_nav, render_stepper


# ---------------------------------------------------------------------------
# Database initialization (cached once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def _init_storage():
    """Initialize the storage module once. Must be called before any DB access."""
    from storage import init
    init()
    return True


def _get_db_path() -> str:
    from storage.db import db_path
    return db_path()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="会前准备", page_icon="📋", layout="wide")
apply_office_theme()

_storage_ready = _init_storage()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "prep_step": 0,
    "current_meeting_id": None,
    "current_meeting_name": "",
    "imported_participants": [],
    "import_result": None,
    "generated_cards": None,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
STEPS = ["创建会议", "导入人员", "验证绑定", "生成桌牌", "扫描测试"]

with st.sidebar:
    st.markdown('<div class="office-section-label">系统状态</div>', unsafe_allow_html=True)
    st.caption(f"数据库: `{_get_db_path()}`")

    # Current meeting indicator
    if st.session_state.current_meeting_id:
        try:
            from storage.db import session_scope
            from storage.repo import MeetingRepo
            with session_scope() as s:
                mr = MeetingRepo(s)
                m = mr.get_by_id(st.session_state.current_meeting_id)
                if m:
                    st.success(f"📌 {m.name}")
        except Exception:
            pass

    st.divider()
    st.markdown('<div class="office-section-label">操作流程</div>', unsafe_allow_html=True)
    render_stepper(STEPS, st.session_state.prep_step)

    st.divider()
    render_sidebar_nav("prep")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 0: Create Meeting
# ═══════════════════════════════════════════════════════════════════════════════

def _render_create_meeting():
    st.subheader("Step 1/5: 创建会议")

    col_form, col_list = st.columns([1, 1])

    with col_form:
        with st.form("create_meeting_form"):
            name = st.text_input("会议名称 *", placeholder="例如: 智会追声项目路演")
            location = st.text_input("会议地点", placeholder="会议室 A")
            description = st.text_area("会议描述", placeholder="会议主题、议程等 (可选)")
            submitted = st.form_submit_button("✅ 创建会议", type="primary")

            if submitted:
                if not name.strip():
                    st.error("会议名称不能为空")
                else:
                    from storage.db import session_scope
                    from storage.repo import MeetingRepo
                    with session_scope() as s:
                        mr = MeetingRepo(s)
                        meeting = mr.create(
                            name=name.strip(),
                            location=location.strip(),
                            description=description.strip(),
                        )
                        st.session_state.current_meeting_id = meeting.id
                        st.session_state.current_meeting_name = meeting.name
                    st.success(f"会议 '{name}' 创建成功!")
                    st.session_state.prep_step = 1
                    st.rerun()

    with col_list:
        _show_existing_meetings()


def _show_existing_meetings():
    """Show table of existing meetings and allow selection."""
    st.subheader("已有会议")
    from storage.db import session_scope
    from storage.repo import MeetingRepo

    with session_scope() as s:
        mr = MeetingRepo(s)
        meetings = mr.list_all()

    if not meetings:
        st.info("暂无会议记录，请创建新会议")
        return

    df = pd.DataFrame([{
        "ID": m.id,
        "名称": m.name,
        "状态": m.status,
        "地点": m.location or "",
        "创建时间": m.created_at.strftime("%m-%d %H:%M") if m.created_at else "",
    } for m in meetings])

    st.dataframe(df, width="stretch", hide_index=True)

    # Select existing meeting
    meeting_options = {f"#{m.id} {m.name}": m.id for m in meetings}
    selected = st.selectbox("选择已有会议继续操作:", ["--"] + list(meeting_options.keys()))
    if selected != "--":
        st.session_state.current_meeting_id = meeting_options[selected]
        st.session_state.current_meeting_name = selected[selected.find(" ") + 1:]
        st.session_state.prep_step = 1
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Import Participants
# ═══════════════════════════════════════════════════════════════════════════════

CSV_TEMPLATE = "tag_id,name,organization,role,title\nA001,王强,XX大学,项目负责人,队长\nA002,李老师,XX学院,评委,教授\nA003,张老师,XX实验室,主持人,老师\n"


def _render_import_participants():
    st.subheader("Step 2/5: 导入人员名单")

    if not st.session_state.current_meeting_id:
        st.warning("请先创建或选择一个会议")
        if st.button("← 返回创建会议"):
            st.session_state.prep_step = 0
            st.rerun()
        return

    # Template download
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("#### 上传 CSV 或 Excel 文件")
    with col2:
        st.download_button(
            "📥 下载模板 CSV",
            data=CSV_TEMPLATE,
            file_name="participants_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader(
        "选择文件",
        type=["csv", "xlsx"],
        help="CSV 或 Excel 文件，必须包含列: tag_id, name",
    )

    if uploaded is None:
        # Show currently imported participants
        if st.session_state.imported_participants:
            st.divider()
            st.markdown("#### 当前已导入人员")
            df = pd.DataFrame(st.session_state.imported_participants)
            st.dataframe(df, width="stretch", hide_index=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("→ 下一步：验证绑定"):
                    st.session_state.prep_step = 2
                    st.rerun()
            with c2:
                if st.button("🔄 重新导入"):
                    st.session_state.imported_participants = []
                    st.session_state.import_result = None
                    st.rerun()
        return

    # Parse file
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"文件解析失败: {e}")
        return

    # Ensure required columns
    for col in ["tag_id", "name"]:
        if col not in df.columns:
            st.error(f"缺少必要列: `{col}`")
            return

    # Fill defaults
    if "role" not in df.columns:
        df["role"] = "参会人员"
    else:
        df["role"] = df["role"].fillna("参会人员")
    if "organization" not in df.columns:
        df["organization"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    st.markdown("#### 预览与编辑")
    st.caption("双击单元格可编辑。确认无误后点击导入。")

    edited = st.data_editor(
        df[["tag_id", "name", "organization", "role", "title"]],
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📥 导入到数据库", type="primary"):
            rows = edited.dropna(subset=["tag_id", "name"]).to_dict("records")
            from storage.db import session_scope
            from storage.repo import ParticipantRepo

            with session_scope() as s:
                pr = ParticipantRepo(s)
                result = pr.bulk_import(rows)

            st.session_state.imported_participants = rows
            st.session_state.import_result = result
            st.success(
                f"导入完成: 新增 {result['created']} 人, "
                f"更新 {result['updated']} 人, "
                f"错误 {len(result['errors'])} 条"
            )
            if result["errors"]:
                for e in result["errors"]:
                    st.warning(e)
            st.rerun()

    with c2:
        if st.session_state.imported_participants:
            if st.button("→ 下一步"):
                st.session_state.prep_step = 2
                st.rerun()

    with c3:
        if st.button("🔄 重置"):
            st.session_state.imported_participants = []
            st.session_state.import_result = None
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Validate Bindings
# ═══════════════════════════════════════════════════════════════════════════════

def _render_validate_bindings():
    st.subheader("Step 3/5: 验证标签绑定")

    from storage.db import session_scope
    from storage.repo import ParticipantRepo

    with session_scope() as s:
        pr = ParticipantRepo(s)
        participants = pr.list_all()

    if not participants:
        st.warning("尚未导入任何人员，请先导入")
        if st.button("← 返回导入"):
            st.session_state.prep_step = 1
            st.rerun()
        return

    # Run validations
    tag_dir = os.path.join(_PROJECT_ROOT, "tagStandard41h12")

    results = []
    seen_tags = {}
    has_errors = False

    for p in participants:
        issues = []
        status = "ok"

        # Check duplicate
        if p.tag_id in seen_tags:
            issues.append(f"重复: 与 {seen_tags[p.tag_id]} 冲突")
            status = "error"
        else:
            seen_tags[p.tag_id] = p.name

        # Check empty name
        if not p.name.strip():
            issues.append("姓名为空")
            status = "error"

        # Check tag image existence
        if tag_dir and os.path.isdir(tag_dir):
            from core.desk_card_generator import tag_id_to_int, tag_filename
            try:
                tid = tag_id_to_int(p.tag_id)
                fname = tag_filename(tid)
                if not os.path.isfile(os.path.join(tag_dir, fname)):
                    issues.append(f"标签图片不存在: {fname}")
                    status = "warning" if status == "ok" else status
            except ValueError:
                issues.append(f"tag_id 格式无效: {p.tag_id}")
                status = "error"

        if not issues:
            issues.append("正常")

        results.append({
            "tag_id": p.tag_id,
            "姓名": p.name,
            "角色": p.role or "",
            "状态": status,
            "详情": "; ".join(issues),
        })

    # Display
    df = pd.DataFrame(results)

    def _color_status(val):
        if val == "ok":
            return "background-color: #d4edda; color: #155724"
        elif val == "warning":
            return "background-color: #fff3cd; color: #856404"
        elif val == "error":
            return "background-color: #f8d7da; color: #721c24"
        return ""

    styled = df.style.map(_color_status, subset=["状态"])
    st.dataframe(styled, width="stretch", hide_index=True)

    # Summary
    ok_count = sum(1 for r in results if r["状态"] == "ok")
    warn_count = sum(1 for r in results if r["状态"] == "warning")
    err_count = sum(1 for r in results if r["状态"] == "error")

    cols = st.columns(3)
    cols[0].metric("✅ 正常", ok_count)
    cols[1].metric("⚠️ 警告", warn_count)
    cols[2].metric("❌ 错误", err_count)

    if err_count > 0:
        st.error("请修复上述错误后再继续")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("← 返回导入"):
                st.session_state.prep_step = 1
                st.rerun()
        with c2:
            if st.button("→ 下一步：生成桌牌", type="primary"):
                st.session_state.prep_step = 3
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Generate Desk Cards
# ═══════════════════════════════════════════════════════════════════════════════

def _render_generate_cards():
    st.subheader("Step 4/5: 生成桌牌")

    from storage.db import session_scope
    from storage.repo import ParticipantRepo

    with session_scope() as s:
        pr = ParticipantRepo(s)
        participants = pr.list_all()

    if not participants:
        st.warning("暂无参会人员")
        if st.button("← 返回导入"):
            st.session_state.prep_step = 1
            st.rerun()
        return

    st.markdown(f"共 **{len(participants)}** 名参会人员")

    # Config
    with st.expander("⚙️ 桌牌设置", expanded=False):
        config_path = os.path.join(_PROJECT_ROOT, "configs", "desk_card_config.json")
        if os.path.isfile(config_path):
            from core.desk_card_generator import DeskCardConfig
            config = DeskCardConfig.from_json(config_path)
        else:
            from core.desk_card_generator import DeskCardConfig
            config = DeskCardConfig()

        card_size_options = {
            "横版 (210×100mm)": (210, 100),
            "A5 竖版 (148×210mm)": (148, 210),
            "A6 竖版 (105×148mm)": (105, 148),
        }
        # Find current card size in options (default to first)
        current_mm = config.card_size_mm
        default_idx = 0
        for i, (label, size) in enumerate(card_size_options.items()):
            if size == current_mm:
                default_idx = i
                break
        selected = st.selectbox(
            "桌牌尺寸",
            list(card_size_options.keys()),
            index=default_idx,
        )
        config.card_size_mm = card_size_options[selected]
        config.dpi = st.slider("DPI", 150, 600, config.dpi, 75)
        config.tag_display_mm = st.slider("标签显示大小 (mm)", 20, 80, config.tag_display_mm)

    # Preview one card
    st.markdown("#### 预览")
    preview_person = participants[0]
    try:
        from core.desk_card_generator import generate_desk_card_png
        preview_img = generate_desk_card_png(
            tag_label=preview_person.tag_id,
            name=preview_person.name,
            role=preview_person.role or "",
            organization=preview_person.organization or "",
            tag_dir=os.path.join(_PROJECT_ROOT, "tagStandard41h12"),
            config=config,
        )
        st.image(preview_img, caption=f"{preview_person.tag_id} {preview_person.name}", width=300)
    except Exception as e:
        st.error(f"预览生成失败: {e}")
        return

    # Generate all
    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🎨 批量生成桌牌", type="primary"):
            from core.desk_card_generator import generate_desk_cards_for_participants

            part_dicts = [{
                "tag_id": p.tag_id,
                "name": p.name,
                "role": p.role or "",
                "organization": p.organization or "",
            } for p in participants]

            output_dir = os.path.join(_PROJECT_ROOT, "exports", "desk_cards")

            with st.spinner("正在生成桌牌..."):
                result = generate_desk_cards_for_participants(
                    part_dicts,
                    output_dir=output_dir,
                    tag_dir=os.path.join(_PROJECT_ROOT, "tagStandard41h12"),
                    config=config,
                )

            st.session_state.generated_cards = result
            st.success(f"生成完成: {len(result['generated'])} 张桌牌")
            if result["errors"]:
                for e in result["errors"]:
                    st.warning(e)
            st.rerun()

    with c2:
        if st.button("← 返回验证"):
            st.session_state.prep_step = 2
            st.rerun()

    # Show downloads
    cards = st.session_state.generated_cards
    if cards and cards.get("generated"):
        st.divider()
        st.markdown("#### 📥 下载")

        # Combined PDF
        if cards.get("pdf") and os.path.isfile(cards["pdf"]):
            with open(cards["pdf"], "rb") as f:
                st.download_button(
                    "📄 下载合并 PDF",
                    data=f,
                    file_name="desk_cards_combined.pdf",
                    mime="application/pdf",
                )

        # Individual PNGs
        for path in cards["generated"]:
            if os.path.isfile(path):
                fname = os.path.basename(path)
                with open(path, "rb") as f:
                    st.download_button(
                        f"🖼️ {fname}",
                        data=f,
                        file_name=fname,
                        mime="image/png",
                    )

        # Next step
        st.divider()
        if st.button("→ 下一步：扫描测试"):
            st.session_state.prep_step = 4
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Camera Scan Test
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_and_display_apriltags(img_data, all_participants):
    """Run AprilTag detection on an image and display results.

    Shared by both camera capture and file upload paths.
    Returns True if detection succeeded (even if no tags found), False on error.
    """
    try:
        import numpy as np
        from PIL import Image as PILImage
        import pupil_apriltags

        # Decode image
        pil_img = PILImage.open(img_data)
        gray = np.array(pil_img.convert("L"))

        # Run detector
        detector = pupil_apriltags.Detector(
            families="tagStandard41h12",
            nthreads=2,
            quad_decimate=2.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )
        detections = detector.detect(gray)

        from core.desk_card_generator import int_to_tag_id

        if not detections:
            st.info("未检测到任何 AprilTag")
            return True

        st.success(f"检测到 {len(detections)} 个标签")

        # Build results table
        rows = []
        for d in detections:
            label = int_to_tag_id(d.tag_id)
            p = all_participants.get(label)
            if p:
                status = "✅ 已绑定"
                name = p.name
                role = p.role or ""
            else:
                status = "⚠️ 未绑定"
                name = "—"
                role = "—"

            rows.append({
                "标签 ID": f"{label} (int: {d.tag_id})",
                "姓名": name,
                "角色": role,
                "绑定状态": status,
                "置信度": f"{d.decision_margin:.2f}",
                "中心坐标": f"({d.center[0]:.0f}, {d.center[1]:.0f})",
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        # Highlight unbound tags
        unbound = [r for r in rows if "未绑定" in r["绑定状态"]]
        if unbound:
            st.warning(
                f"{len(unbound)} 个标签未绑定到参会人员: "
                f"{', '.join(r['标签 ID'].split()[0] for r in unbound)}"
            )
        return True

    except ImportError:
        st.warning("⚠️ `pupil-apriltags` 未安装，无法进行标签检测。请在终端运行: `pip install pupil-apriltags`")
        return False
    except Exception as e:
        st.error(f"检测失败: {e}")
        return False


def _render_camera_scan():
    st.subheader("Step 5/5: 摄像头扫描测试")

    st.markdown("""
    拍摄或上传包含 AprilTag 的照片，验证标签是否能被正确识别及绑定。

    **依赖**: 此功能需要 `pupil-apriltags` 库。如果未安装，请运行:
    ```
    pip install pupil-apriltags
    ```
    """)

    from storage.db import session_scope
    from storage.repo import ParticipantRepo

    with session_scope() as s:
        pr = ParticipantRepo(s)
        all_participants = {p.tag_id: p for p in pr.list_all()}

    # ── Two input methods: camera (localhost/HTTPS only) + file upload (always works) ──
    tab_camera, tab_upload = st.tabs(["📷 摄像头拍照", "📤 上传照片"])

    with tab_camera:
        # st.camera_input relies on getUserMedia(), which browsers only allow on
        # localhost or HTTPS. When accessing Streamlit remotely via
        # http://<board-ip>:8501, the browser blocks it silently — you see
        # "This app would like to use your camera" without a real prompt.
        st.caption(
            "⚠️ 浏览器摄像头仅支持 **localhost** 或 **HTTPS** 访问。"
            "如果无法唤起摄像头，请切换到「📤 上传照片」标签页。"
        )
        img_data = st.camera_input(
            "拍照扫描 AprilTag",
            help="如需使用摄像头，请通过 https:// 或 localhost 访问本页面",
        )
        if img_data is not None:
            _detect_and_display_apriltags(img_data, all_participants)

    with tab_upload:
        st.caption("上传包含 AprilTag 的照片（支持 JPG/PNG）。适用于远程 HTTP 访问。")
        uploaded_img = st.file_uploader(
            "选择图片文件",
            type=["jpg", "jpeg", "png"],
            help="从本地选择一张包含 AprilTag 的照片",
        )
        if uploaded_img is not None:
            st.image(uploaded_img, caption="上传的图片", width=400)
            _detect_and_display_apriltags(uploaded_img, all_participants)

    # Navigation
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← 返回生成桌牌"):
            st.session_state.prep_step = 3
            st.rerun()
    with c2:
        st.success("🎉 会前准备完成!")


# ═══════════════════════════════════════════════════════════════════════════════
# Main dispatch (must be after all function definitions)
# ═══════════════════════════════════════════════════════════════════════════════
render_page_header("会前准备", "按照创建会议、导入人员、验证标签、生成桌牌、扫描测试的顺序完成会前配置。", "Preparation")

if st.session_state.prep_step == 0:
    _render_create_meeting()
elif st.session_state.prep_step == 1:
    _render_import_participants()
elif st.session_state.prep_step == 2:
    _render_validate_bindings()
elif st.session_state.prep_step == 3:
    _render_generate_cards()
elif st.session_state.prep_step == 4:
    _render_camera_scan()


# ═══════════════════════════════════════════════════════════════════════════════
# Main guard
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # When run directly, the Streamlit script runner will render the page
    pass
