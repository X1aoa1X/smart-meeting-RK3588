"""轻量级 AprilTag 检测工作线程 — 从 Queue 接收帧、运行检测、发射结果。

与 ApriltagCameraThread 不同: 本线程**不打开摄像头**，而是从 YoloCameraThread
提供的帧队列中读取。这解决了 V4L2 设备不能同时打开两次的问题。

用法:
  import queue
  from core.tag_detect_worker import TagDetectWorker

  tag_queue = queue.Queue(maxsize=2)
  worker = TagDetectWorker(tag_queue)
  worker.tags_ready.connect(on_tags)
  worker.fps_update.connect(on_fps)
  worker.status_msg.connect(print)
  worker.start()

  # YoloCameraThread 向 tag_queue 写入帧
  camera_thread.set_tag_queue(tag_queue)

依赖: cv2, numpy, PyQt5, pupil_apriltags
"""

import time
import queue
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import pupil_apriltags


class TagDetectWorker(QThread):
    """后台线程: 从 Queue 接收帧 → AprilTag 检测 → 发射结果。

    不拥有摄像头，帧由外部生产者（通常是 YoloCameraThread）提供。
    """

    tags_ready   = pyqtSignal(list)       # list[Detection] — 检测结果
    fps_update   = pyqtSignal(float)      # 检测 FPS
    status_msg   = pyqtSignal(str)        # 状态消息

    # ── AprilTag 检测器参数 (与 ApriltagCameraThread 保持一致) ────────────
    TAG_FAMILY         = "tagStandard41h12"
    QUAD_DECIMATE      = 2.0
    QUAD_SIGMA         = 0.0
    REFINE_EDGES       = 1
    DECODE_SHARPENING  = 0.25
    NTHREADS           = 1               # 单线程: 与 YOLO NPU 并行，避免过度订阅 CPU

    # 检测帧处理尺寸 (从 1920×1080 缩放到此分辨率做检测)
    DETECT_WIDTH  = 640
    DETECT_HEIGHT = 480

    def __init__(self, frame_queue: queue.Queue):
        """
        Args:
            frame_queue: 接收帧的队列（通常由 YoloCameraThread 写入）。
                         帧应为 BGR uint8 numpy array。
        """
        super().__init__()
        self._queue = frame_queue
        self._running = False
        self._detector: pupil_apriltags.Detector | None = None

    def run(self):
        self._running = True

        # ── 初始化检测器 ──────────────────────────────────────────────────
        try:
            self._detector = pupil_apriltags.Detector(
                families=self.TAG_FAMILY,
                nthreads=self.NTHREADS,
                quad_decimate=self.QUAD_DECIMATE,
                quad_sigma=self.QUAD_SIGMA,
                refine_edges=self.REFINE_EDGES,
                decode_sharpening=self.DECODE_SHARPENING,
            )
            self.status_msg.emit(
                f"🏷️ TagDetect 就绪 (family={self.TAG_FAMILY}, "
                f"{self.DETECT_WIDTH}×{self.DETECT_HEIGHT})")
        except Exception as e:
            self.status_msg.emit(f"❌ TagDetect 初始化失败: {e}")
            return

        # ── 帧率统计 ──────────────────────────────────────────────────────
        frames = 0
        t_start = time.time()
        t_last_fps = time.time()

        # ── 主循环 ────────────────────────────────────────────────────────
        while self._running:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if frame is None or frame.size == 0:
                continue

            # ── 缩放到检测尺寸（如已缩放则跳过）──────────────────────────
            h, w = frame.shape[:2]
            if w != self.DETECT_WIDTH or h != self.DETECT_HEIGHT:
                detect_frame = cv2.resize(frame, (self.DETECT_WIDTH, self.DETECT_HEIGHT),
                                          interpolation=cv2.INTER_LINEAR)
            else:
                detect_frame = frame

            # ── 灰度转换 + 检测 ────────────────────────────────────────────
            gray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)
            try:
                tags = self._detector.detect(gray)
            except Exception:
                tags = []

            self.tags_ready.emit(tags)

            # ── FPS 统计 ──────────────────────────────────────────────────
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

        # ── 清理 ──────────────────────────────────────────────────────────
        self._detector = None
        self.status_msg.emit("🏷️ TagDetect 已停止")

    def stop(self):
        """停止检测线程。"""
        self._running = False
        self.wait(3000)
