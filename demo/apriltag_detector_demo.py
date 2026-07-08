#!/usr/bin/env python3
"""
AprilTag 实时检测 PyQt5 Demo — 从摄像头画面识别 tagStandard41h12 标签

用法:
  python3 apriltag_detector_demo.py
  (如需 sudo，添加环境变量修复即可)

依赖:
  - opencv-python (cv2)
  - PyQt5
  - numpy
  - pupil-apriltags
"""

import os
import sys
import time
import pwd

# ── 自动修复 sudo 丢失的显示环境变量 ────────────────────────────────────────────
def _fix_display_env():
    """自动检测并设置本地显示所需的 DISPLAY / XAUTHORITY 环境变量。"""
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print(f"[Display] DISPLAY 未设置，自动设为 {os.environ['DISPLAY']}")

    if not os.environ.get("XAUTHORITY"):
        uid = os.getuid()
        candidates = []
        candidates.append(f"/run/user/{uid}/gdm/Xauthority")
        candidates.append(f"/run/user/{uid}/Xauthority")
        try:
            home = pwd.getpwuid(uid).pw_dir
            candidates.append(os.path.join(home, ".Xauthority"))
        except KeyError:
            pass

        for path in candidates:
            if os.path.isfile(path) and os.access(path, os.R_OK):
                os.environ["XAUTHORITY"] = path
                print(f"[Display] XAUTHORITY 自动设为 {path}")
                break
        else:
            print("[Display] ⚠️ 未找到 XAUTHORITY 文件，尝试无授权连接")

_fix_display_env()

# ── 修复 OpenCV 内建 Qt 插件与 PyQt5 冲突 ────────────────────────────────────
os.environ.setdefault("DISPLAY", ":0.0")

import cv2
import numpy as np

_CV2_QT_PLUGIN_PATH = "/usr/local/lib/python3.10/dist-packages/cv2/qt/plugins"
if os.environ.get("QT_PLUGIN_PATH") == _CV2_QT_PLUGIN_PATH:
    del os.environ["QT_PLUGIN_PATH"]

_SYS_QT5_PLATFORM_PATH = "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms"
if os.path.isdir(_SYS_QT5_PLATFORM_PATH):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _SYS_QT5_PLATFORM_PATH

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QTextEdit, QPushButton, QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

import pupil_apriltags


# ═════════════════════════════════════════════════════════════════════════════════
# CameraThread — 后台摄像头采集 + AprilTag 检测
# ═════════════════════════════════════════════════════════════════════════════════

class CameraThread(QThread):
    """后台线程: 读取摄像头 → AprilTag 检测 → 发送带标注帧 + 检测结果。"""

    frame_ready      = pyqtSignal(np.ndarray)        # 标注后的画面帧
    detection_result = pyqtSignal(list)              # 检测结果列表 (list[Detection])
    fps_update       = pyqtSignal(float)             # FPS
    status_msg       = pyqtSignal(str)               # 状态消息

    # ── 可配置参数 ──────────────────────────────────────────────────────────
    CAP_DEVICE  = "/dev/video21"
    CAM_WIDTH   = 1280
    CAM_HEIGHT  = 720
    FOURCC      = cv2.VideoWriter_fourcc(*'MJPG')
    BUFFER_SIZE = 1

    # AprilTag 检测器参数
    TAG_FAMILY         = "tagStandard41h12"
    QUAD_DECIMATE      = 2.0
    QUAD_SIGMA         = 0.0
    REFINE_EDGES       = 1
    DECODE_SHARPENING  = 0.25
    NTHREADS           = 2

    # 显示缩放
    DISPLAY_MAX_WIDTH = 960

    def __init__(self):
        super().__init__()
        self._running = False
        self._detector: pupil_apriltags.Detector | None = None

    def run(self):
        self._running = True

        # ── 初始化 AprilTag 检测器 ──────────────────────────────────────────
        try:
            self._detector = pupil_apriltags.Detector(
                families=self.TAG_FAMILY,
                nthreads=self.NTHREADS,
                quad_decimate=self.QUAD_DECIMATE,
                quad_sigma=self.QUAD_SIGMA,
                refine_edges=self.REFINE_EDGES,
                decode_sharpening=self.DECODE_SHARPENING,
            )
            self.status_msg.emit(f"✅ AprilTag 检测器就绪 (family={self.TAG_FAMILY})")
        except Exception as e:
            self.status_msg.emit(f"❌ AprilTag 检测器初始化失败: {e}")
            return

        # ── 打开摄像头 ──────────────────────────────────────────────────────
        cap = cv2.VideoCapture(self.CAP_DEVICE, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.status_msg.emit(f"❌ 无法打开 {self.CAP_DEVICE}")
            return

        cap.set(cv2.CAP_PROP_FOURCC, self.FOURCC)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.BUFFER_SIZE)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.status_msg.emit(f"📷 摄像头: {actual_w}×{actual_h} @ {self.CAP_DEVICE}")

        # ── 帧率统计 ──────────────────────────────────────────────────────────
        frames = 0
        t_start = time.time()
        t_last_fps = time.time()

        # ── 主循环 ─────────────────────────────────────────────────────────
        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                QThread.msleep(1)
                continue

            # ── AprilTag 检测（在灰度图上运行）────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                tags = self._detector.detect(gray)
            except Exception:
                tags = []

            # ── 在彩色帧上绘制检测结果 ─────────────────────────────────────────
            annotated = self._draw_detections(frame, tags)

            # ── 发送信号 ──────────────────────────────────────────────────────
            self.detection_result.emit(tags)

            # 缩放显示帧
            h, w = annotated.shape[:2]
            if w > self.DISPLAY_MAX_WIDTH:
                scale = self.DISPLAY_MAX_WIDTH / w
                new_w, new_h = int(w * scale), int(h * scale)
                display_frame = cv2.resize(annotated, (new_w, new_h),
                                           interpolation=cv2.INTER_LINEAR)
            else:
                display_frame = annotated
            self.frame_ready.emit(display_frame)

            # ── FPS 统计 ──────────────────────────────────────────────────────
            frames += 1
            if frames >= 15:
                now = time.time()
                elapsed = now - t_last_fps
                if elapsed > 0:
                    fps = frames / max(now - t_start, 0.001)
                    self.fps_update.emit(fps)
                t_last_fps = now
                frames = 0
                t_start = now

        # ── 清理 ─────────────────────────────────────────────────────────────
        cap.release()
        self._detector = None
        self.status_msg.emit("📷 摄像头已释放")

    def stop(self):
        self._running = False
        self.wait(3000)

    # ── 可视化绘制 ──────────────────────────────────────────────────────────

    @staticmethod
    def _draw_detections(frame: np.ndarray, tags: list) -> np.ndarray:
        """在帧上绘制 AprilTag 检测结果。"""
        display = frame.copy()

        # 颜色表（按 tag_id 轮换，使不同标签颜色不同）
        COLORS = [
            (0, 255, 0),     # 绿
            (255, 0, 0),     # 蓝
            (0, 0, 255),     # 红
            (255, 255, 0),   # 青
            (255, 0, 255),   # 紫
            (0, 255, 255),   # 黄
            (128, 255, 0),   # 黄绿
            (255, 128, 0),   # 橙
        ]

        for tag in tags:
            color = COLORS[tag.tag_id % len(COLORS)]
            corners = tag.corners.astype(int)   # (4, 2)

            # 绘制边框
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(display, pt1, pt2, color, thickness=2)

            # 绘制中心十字
            cx, cy = tag.center.astype(int)
            cross_size = 8
            cv2.line(display, (cx - cross_size, cy), (cx + cross_size, cy),
                     (0, 0, 255), thickness=2)
            cv2.line(display, (cx, cy - cross_size), (cx, cy + cross_size),
                     (0, 0, 255), thickness=2)

            # 绘制 tag_id 标签
            text = f"ID:{tag.tag_id}"
            text_pos = (corners[0, 0], corners[0, 1] - 10)
            # 白色背景
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display,
                          (text_pos[0], text_pos[1] - th - 4),
                          (text_pos[0] + tw + 4, text_pos[1] + 2),
                          (255, 255, 255), -1)
            cv2.putText(display, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 绘制 corner 序号小圆点
            for idx, corner in enumerate(corners):
                cv2.circle(display, tuple(corner), 4, (0, 0, 255), -1)

        # 顶部信息栏背景
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (display.shape[1], 36), (0, 0, 0), -1)
        display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

        # 左上角显示检测数量
        n_tags = len(tags)
        cv2.putText(display, f"AprilTags Detected: {n_tags}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0) if n_tags > 0 else (100, 100, 255), 2)

        return display


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
        self._cam_thread = CameraThread()
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
            lines.append(f"  标签族 (tag_family):   {tag.tag_family.decode() if isinstance(tag.tag_family, bytes) else tag.tag_family}")
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
        self._fps_label.setText(f"FPS: {fps:.1f}  |  Family: {CameraThread.TAG_FAMILY}")

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
        self._cam_thread = CameraThread()
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

    def resizeEvent(self, event):
        """窗口大小变化时适配视频标签尺寸。"""
        super().resizeEvent(event)
        # 保持宽高比缩放（通过 Qt.KeepAspectRatio 自动处理）


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
