"""后台摄像头 + RKNN YOLO 推理线程 (QThread)。

用法:
  thread = YoloCameraThread(stream_queue=None)
  thread.frame_ready.connect(on_frame)         # (annotated_frame)
  thread.raw_frame_ready.connect(on_raw)       # (raw_frame) — 仅在无直连队列时发射
  thread.deviation_data.connect(on_dev)        # ((dev_x, dev_y) or None)
  thread.inference_timing.connect(on_timing)   # (dict: preprocess_ms, inference_ms, postprocess_ms, total_ms)
  thread.fps_update.connect(on_fps)            # (fps)
  thread.status_msg.connect(on_status)         # (msg)
  thread.start()

  # 低延迟 RTSP 推流路径: 传入共享队列，CameraThread 直接入队，跳过 Qt 信号
  stream_queue = queue.Queue(maxsize=2)
  thread.set_stream_queue(stream_queue)

依赖:
  - rknnpool.rknnpool_ld.rknnPoolExecutor (RKNN 推理池)
  - func.func_yolov8_optimize.myFunc, yolov8_post_process, letterbox
  - cv2, numpy, PyQt5
"""

import time
import queue
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


class YoloCameraThread(QThread):
    """后台线程: 读取摄像头 → RKNN 推理 → 发送带标注帧 + 人体偏差。"""

    frame_ready    = pyqtSignal(np.ndarray)
    raw_frame_ready = pyqtSignal(np.ndarray)   # 原始摄像头画面（无 YOLO 标注），用于 RTSP 推流
    deviation_data = pyqtSignal(object)
    person_box_ready = pyqtSignal(object)      # dict {left,top,right,bottom} 或 None
    inference_timing = pyqtSignal(object)  # dict: {preprocess_ms, inference_ms, postprocess_ms, total_ms}
    fps_update     = pyqtSignal(float)
    status_msg     = pyqtSignal(str)

    CAP_DEVICE  = "/dev/video21"
    CAM_WIDTH   = 1920
    CAM_HEIGHT  = 1080
    FOURCC      = cv2.VideoWriter_fourcc(*'MJPG')
    BUFFER_SIZE = 1
    TPEs        = 4
    MODEL_PATH  = "./rknnModel/best.rknn"

    def __init__(self, stream_queue: queue.Queue | None = None):
        super().__init__()
        self._running = False
        self._pool = None
        self._stream_queue: queue.Queue | None = stream_queue
        # RTSP 推流帧的预缩放分辨率（与 StreamThread 保持同步）
        self._stream_width = 960
        self._stream_height = 540
        # 标签检测帧队列（供 TagDetectWorker 消费）
        self._tag_queue: queue.Queue | None = None
        self._tag_width = 640
        self._tag_height = 480

    def set_stream_queue(self, q: queue.Queue | None):
        """设置/清除直连 RTSP 推流队列。设为 None 则回退到 Qt 信号模式。"""
        self._stream_queue = q

    def set_tag_queue(self, q: queue.Queue | None):
        """设置/清除标签检测帧队列（供 TagDetectWorker 消费）。"""
        self._tag_queue = q

    def run(self):
        self._running = True

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
        self.status_msg.emit(f"📷 摄像头: {actual_w}×{actual_h}")

        # ── 延迟导入 RKNN 模块（避免 NPU 过早初始化）──────────────────
        from rknnpool.rknnpool_ld import rknnPoolExecutor
        from func.func_yolov8_optimize import myFunc

        try:
            self._pool = rknnPoolExecutor(
                rknnModel=self.MODEL_PATH, TPEs=self.TPEs, func=myFunc)
        except Exception as e:
            self.status_msg.emit(f"❌ RKNN 初始化失败: {e}")
            cap.release()
            return

        for i in range(self.TPEs + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                self.status_msg.emit(f"❌ 预加载第{i+1}帧失败")
                cap.release()
                self._pool.release()
                return
            self._pool.put(frame)
        self.status_msg.emit("✅ RKNN 池就绪")

        frames = 0
        t_start = time.time()
        t_last_fps = time.time()

        _rknn_for_dev = self._pool.rknnPool[0]
        from func.func_yolov8_optimize import yolov8_post_process, letterbox

        def compute_person_deviation(frame_bgr):
            t0 = time.perf_counter()
            try:
                img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_rgb, ratio, padding = letterbox(img_rgb)
                img_in = np.expand_dims(img_rgb, 0)
                img_in = np.ascontiguousarray(img_in)
                t1 = time.perf_counter()

                outputs = _rknn_for_dev.inference(inputs=[img_in], data_format=['nhwc'])
                t2 = time.perf_counter()

                boxes, classes, scores = yolov8_post_process(outputs)
                if boxes is None:
                    self.inference_timing.emit({
                        "preprocess_ms": (t1 - t0) * 1000,
                        "inference_ms": (t2 - t1) * 1000,
                        "postprocess_ms": 0.0,
                        "total_ms": (time.perf_counter() - t0) * 1000,
                    })
                    return None, None, None

                person_boxes = []
                for box, score, cl in zip(boxes, scores, classes):
                    if int(cl) == 0:
                        left, top, right, bottom = box
                        left   = int((left   - padding[0]) / ratio[0])
                        top    = int((top    - padding[1]) / ratio[1])
                        right  = int((right  - padding[0]) / ratio[0])
                        bottom = int((bottom - padding[1]) / ratio[1])
                        area = (bottom - top) * (right - left)
                        center_x = (left + right) / 2.0
                        center_y = (top + bottom) / 2.0
                        person_boxes.append((area, center_x, center_y, left, top, right, bottom))

                if not person_boxes:
                    self.inference_timing.emit({
                        "preprocess_ms": (t1 - t0) * 1000,
                        "inference_ms": (t2 - t1) * 1000,
                        "postprocess_ms": 0.0,
                        "total_ms": (time.perf_counter() - t0) * 1000,
                    })
                    return None, None, None

                person_boxes.sort(key=lambda x: x[0], reverse=True)
                _, cx, cy, left, top, right, bottom = person_boxes[0]
                img_cx = frame_bgr.shape[1] / 2.0
                img_cy = frame_bgr.shape[0] / 2.0
                dev_x = (cx - img_cx) / img_cx
                dev_y = (cy - img_cy) / img_cy
                person_box = {"left": left, "top": top, "right": right, "bottom": bottom}
                t3 = time.perf_counter()
                self.inference_timing.emit({
                    "preprocess_ms": (t1 - t0) * 1000,
                    "inference_ms": (t2 - t1) * 1000,
                    "postprocess_ms": (t3 - t2) * 1000,
                    "total_ms": (t3 - t0) * 1000,
                })
                return dev_x, dev_y, person_box
            except Exception:
                self.inference_timing.emit({
                    "preprocess_ms": 0.0,
                    "inference_ms": 0.0,
                    "postprocess_ms": 0.0,
                    "total_ms": 0.0,
                })
                return None, None, None

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # ── 发送原始帧（用于 RTSP 推流）─────────────────────────────
            # 直连队列模式（低延迟）：CameraThread 直接入队，跳过 Qt 信号
            if self._stream_queue is not None:
                stream_frame = cv2.resize(frame,
                                          (self._stream_width, self._stream_height),
                                          interpolation=cv2.INTER_LINEAR)
                # drain 旧帧，仅保留最新帧（等价于原 put_frame 逻辑）
                while True:
                    try:
                        self._stream_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self._stream_queue.put_nowait(stream_frame)
                except queue.Full:
                    pass
            else:
                # 回退：Qt 信号模式（用于非推流场景或其他订阅者）
                self.raw_frame_ready.emit(frame.copy())

            # ── 发送帧给标签检测（供 TagDetectWorker 消费）───────────────
            if self._tag_queue is not None:
                tag_frame = cv2.resize(frame,
                                       (self._tag_width, self._tag_height),
                                       interpolation=cv2.INTER_LINEAR)
                # drain 旧帧，仅保留最新帧
                while True:
                    try:
                        self._tag_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self._tag_queue.put_nowait(tag_frame)
                except queue.Full:
                    pass

            self._pool.put(frame)

            result = self._pool.get()
            if result[0] is not None:
                annotated_frame, flag = result
                if not flag:
                    self.status_msg.emit("❌ 推理失败")
                    break

                frames += 1

                if frame is not None and len(frame.shape) == 3:
                    dev_x, dev_y, person_box = compute_person_deviation(frame)
                    self.deviation_data.emit((dev_x, dev_y))
                    self.person_box_ready.emit(person_box)

                if annotated_frame is not None and len(annotated_frame.shape) == 3:
                    h, w = annotated_frame.shape[:2]
                    if w > 960:
                        scale = 960.0 / w
                        new_w, new_h = int(w * scale), int(h * scale)
                        display_frame = cv2.resize(annotated_frame, (new_w, new_h),
                                                   interpolation=cv2.INTER_LINEAR)
                    else:
                        display_frame = annotated_frame
                    self.frame_ready.emit(display_frame)
            else:
                QThread.msleep(1)

            if frames >= 15:
                now = time.time()
                if now - t_last_fps > 0:
                    fps = frames / max(now - t_start, 0.001)
                    self.fps_update.emit(fps)
                t_last_fps = now

        if self._pool is not None:
            self._pool.release()
        cap.release()
        self.status_msg.emit("📷 摄像头已释放")

    def stop(self):
        self._running = False
        self.wait(3000)
