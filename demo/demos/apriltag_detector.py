#!/usr/bin/env python3
"""AprilTag 实时检测 Demo — 基于 core/ 模块的 PyQt5 GUI

用法:
  python3 demos/apriltag_detector.py
  (如需 sudo，添加环境变量修复即可)

依赖:
  - opencv-python (cv2)
  - PyQt5
  - numpy
  - pupil-apriltags
  - core/apriltag_camera, core/display_env
"""

import os
import sys
import time

# ── 确保能找到项目根目录的 core/ 模块 ────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── 显示环境修复 ──────────────────────────────────────────────────────────
from core.display_env import fix_display_env
fix_display_env()

os.environ.setdefault("DISPLAY", ":0.0")

import cv2
from core.display_env import fix_cv2_qt_conflict
fix_cv2_qt_conflict()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTextEdit, QPushButton, QStatusBar,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

import numpy as np

from core.apriltag_camera import ApriltagCameraThread


# ═════════════════════════════════════════════════════════════════════════════════
# MainWindow — PyQt5 主窗口
# ═════════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """AprilTag 检测器主窗口。"""

    WINDOW_TITLE = "AprilTag 实时检测 — tagStandard41h12"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)

        # ── 相机线程 ────────────────────────────────────────────────────────
        self._cam_thread = ApriltagCameraThread()
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.detection_result.connect(self._on_detections)
        self._cam_thread.fps_update.connect(self._on_fps)
        self._cam_thread.status_msg.connect(self._on_status)

        # ── 构建 UI ──────────────────────────────────────────────────────────
        self._build_ui()

        # ── 启动相机 ──────────────────────────────────────────────────────────
        self._cam_thread.start()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)

        # ── 左侧: 视频画面 ───────────────────────────────────────────────────
        left_panel = QVBoxLayout()

        video_group = QGroupBox("📹 摄像头画面")
        video_layout = QVBoxLayout(video_group)
        self._video_label = QLabel("等待摄像头…")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(640, 360)
        self._video_label.setStyleSheet(
            "QLabel { background-color: #1a1a1a; color: #888; "
            "border: 1px solid #333; border-radius: 4px; }")
        video_layout.addWidget(self._video_label)
        left_panel.addWidget(video_group)

        # ── 帧率标签 ──────────────────────────────────────────────────────────
        self._fps_label = QLabel("FPS: --")
        self._fps_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        left_panel.addWidget(self._fps_label)

        layout.addLayout(left_panel, stretch=3)

        # ── 右侧: 检测结果面板 ──────────────────────────────────────────────
        right_panel = QVBoxLayout()

        # 检测详情
        detail_group = QGroupBox("🏷️ 检测详情")
        detail_layout = QVBoxLayout(detail_group)
        self._detail_text = QTextEdit()
        self._detail_text.setReadOnly(True)
        self._detail_text.setFont(QFont("monospace", 10))
        self._detail_text.setMinimumWidth(380)
        self._detail_text.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #ccc; "
            "border: 1px solid #333; border-radius: 4px; }")
        self._detail_text.setPlaceholderText("等待摄像头检测到 AprilTag…")
        detail_layout.addWidget(self._detail_text)
        right_panel.addWidget(detail_group, stretch=3)

        # 日志区域
        log_group = QGroupBox("📋 日志")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setFont(QFont("monospace", 9))
        self._log_text.setMaximumHeight(150)
        self._log_text.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #aaa; "
            "border: 1px solid #333; border-radius: 4px; }")
        log_layout.addWidget(self._log_text)
        right_panel.addWidget(log_group, stretch=1)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self._pause_btn = QPushButton("⏸ 暂停/恢复")
        self._pause_btn.setCheckable(True)
        self._pause_btn.toggled.connect(self._on_pause_toggled)
        btn_layout.addWidget(self._pause_btn)

        self._reset_btn = QPushButton("🔄 重置检测器")
        self._reset_btn.clicked.connect(self._on_reset)
        btn_layout.addWidget(self._reset_btn)

        right_panel.addLayout(btn_layout)

        layout.addLayout(right_panel, stretch=2)

        # ── 状态栏 ───────────────────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("就绪")
        self._status_bar.addWidget(self._status_label)

        # 窗口默认大小
        self.resize(1400, 720)

    # ── 槽函数 ──────────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        """接收标注后的视频帧。"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimg)
        self._video_label.setPixmap(
            pixmap.scaled(self._video_label.size(),
                          Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_detections(self, tags: list):
        """接收检测结果，更新右侧详情面板。"""
        if not tags:
            self._detail_text.setPlainText("(未检测到 AprilTag)")
            return

        lines = []
        for i, tag in enumerate(tags):
            lines.append(f"━━━ Tag #{i + 1} ━━━")
            lines.append(f"  标签ID (tag_id):      {tag.tag_id}")
            tag_family = (tag.tag_family.decode()
                          if isinstance(tag.tag_family, bytes)
                          else tag.tag_family)
            lines.append(f"  标签族 (tag_family):   {tag_family}")
            lines.append(f"  海明距 (hamming):      {tag.hamming}")
            lines.append(f"  判决裕度 (margin):     {tag.decision_margin:.4f}")
            cx, cy = tag.center
            lines.append(f"  中心坐标 (center):     ({cx:.1f}, {cy:.1f})")
            lines.append(f"  四角坐标 (corners):")
            for j, corner in enumerate(tag.corners):
                lines.append(f"    [{j}] ({corner[0]:.1f}, {corner[1]:.1f})")
            lines.append("")

        self._detail_text.setPlainText("\n".join(lines))

    def _on_fps(self, fps: float):
        """更新 FPS 显示。"""
        self._fps_label.setText(
            f"FPS: {fps:.1f}  |  Family: {ApriltagCameraThread.TAG_FAMILY}")

    def _on_status(self, msg: str):
        """追加日志消息并更新状态栏。"""
        timestamp = time.strftime("%H:%M:%S")
        self._log_text.append(f"[{timestamp}] {msg}")
        self._status_label.setText(msg)
        # 限制日志行数
        if self._log_text.document().blockCount() > 100:
            cursor = self._log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 20)
            cursor.removeSelectedText()

    def _on_pause_toggled(self, checked: bool):
        """暂停/恢复检测显示。"""
        if checked:
            self._cam_thread.frame_ready.disconnect(self._on_frame)
            self._cam_thread.detection_result.disconnect(self._on_detections)
            self._pause_btn.setText("▶ 恢复")
            self._status_label.setText("⏸ 已暂停")
        else:
            self._cam_thread.frame_ready.connect(self._on_frame)
            self._cam_thread.detection_result.connect(self._on_detections)
            self._pause_btn.setText("⏸ 暂停")
            self._status_label.setText("▶ 已恢复")

    def _on_reset(self):
        """重新初始化检测器。"""
        self._cam_thread.stop()
        # 重新创建并启动
        self._cam_thread = ApriltagCameraThread()
        self._cam_thread.frame_ready.connect(self._on_frame)
        self._cam_thread.detection_result.connect(self._on_detections)
        self._cam_thread.fps_update.connect(self._on_fps)
        self._cam_thread.status_msg.connect(self._on_status)
        self._cam_thread.start()
        self._status_label.setText("🔄 检测器已重置")

    # ── 窗口关闭 ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._cam_thread.stop()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局样式
    app.setStyleSheet("""
        QMainWindow { background-color: #2b2b2b; }
        QGroupBox {
            font-size: 13px; font-weight: bold;
            color: #ddd; border: 1px solid #555;
            border-radius: 6px; margin-top: 10px; padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px; padding: 0 6px 0 6px;
        }
        QPushButton {
            background-color: #3a3a3a; color: #ddd;
            border: 1px solid #555; border-radius: 4px;
            padding: 6px 14px; font-size: 12px;
        }
        QPushButton:hover { background-color: #4a4a4a; }
        QPushButton:checked { background-color: #505050; }
        QLabel { color: #ddd; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
