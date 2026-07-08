"""后台摄像头 + AprilTag 检测线程 (QThread)。

用法:
  thread = ApriltagCameraThread()
  thread.frame_ready.connect(on_frame)           # (annotated_frame)
  thread.detection_result.connect(on_tags)       # (list of Detection)
  thread.fps_update.connect(on_fps)              # (fps)
  thread.status_msg.connect(on_status)           # (msg)
  thread.start()

依赖: cv2, numpy, PyQt5, pupil_apriltags
"""

import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import pupil_apriltags


class ApriltagCameraThread(QThread):
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
        _consecutive_read_failures = 0
        _MAX_READ_FAILURES = 10

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                _consecutive_read_failures += 1
                if _consecutive_read_failures >= _MAX_READ_FAILURES:
                    self.status_msg.emit("❌ 摄像头读取连续失败，线程退出")
                    break
                QThread.msleep(10)
                continue
            _consecutive_read_failures = 0

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
