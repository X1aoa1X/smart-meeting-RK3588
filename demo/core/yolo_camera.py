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
  - yolo_core (C++ pybind11 扩展, 4-way NPU 池 + DFL/NMS/letterbox/绘制)
  - cv2, numpy, PyQt5

说明:
  原 Python 实现 (rknnpool_ld.rknnPoolExecutor + func_yolov8_optimize.myFunc)
  已被 C++ 扩展 yolo_core.YoloInferEngine 替换, 信号契约与公共 API 不变,
  demos/fusion_tracker.py 无需任何修改。构建扩展见
  c_realtime_infer_demo/yolo_core/build.sh。
"""

import time
import queue
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# C++ pybind11 扩展 (build.sh 会把 .so 拷贝到 demo/ 目录)
# 延迟导入在 run() 中进行, 避免 NPU 在 import 时过早初始化


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
    CAM_WIDTH = 1280
    CAM_HEIGHT = 720
    FOURCC      = cv2.VideoWriter_fourcc(*'MJPG')
    CAM_FPS     = 30
    # BUFFER_SIZE=1 会让 OpenCV V4L2 后端只用 1 个缓冲区, 摄像头必须等
    # OpenCV 归还缓冲区才能捕获下一帧, 实测帧率减半 (30fps→15fps)。
    # BUFFER_SIZE=2 让内核可并行排队下一帧, 实测恢复到 31.2fps。
    # 延迟代价: 最多 1 帧 (~33ms), 对实时追踪可接受。
    BUFFER_SIZE = 2
    # 推理池深度越大吞吐越高，但端到端延迟也会增加约 N 帧。
    # 若更重视操控实时性，可降到 2；若更重视 FPS，保持 4。
    TPEs        = 4
    MODEL_PATH  = "./rknnModel/best.rknn"

    def __init__(self, stream_queue: queue.Queue | None = None):
        super().__init__()
        self._running = False
        self._engine = None  # yolo_core.YoloInferEngine (C++ 扩展)
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

    @staticmethod
    def _fourcc_to_str(v: float) -> str:
        """把 OpenCV 返回的 FOURCC 数值转成可读字符串。"""
        try:
            iv = int(v)
            return "".join(chr((iv >> (8 * i)) & 0xFF) for i in range(4))
        except Exception:
            return str(v)

    def _open_capture(self):
        """打开并配置 V4L2 摄像头。

        关键点：
        1. 先设置 MJPG，再设置分辨率/FPS；不少 UVC 摄像头在 YUYV@1080p 下只有 5~15FPS。
        2. 显式设置 FPS，避免驱动落到 15FPS 左右导致 cap.read() 约 60~70ms。
        3. 回读实际参数，便于判断 set() 是否被驱动接受。
        """
        cap = cv2.VideoCapture(self.CAP_DEVICE, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.status_msg.emit(f"❌ 无法打开 {self.CAP_DEVICE}")
            return None

        # 设置顺序对 V4L2/UVC 很重要：格式 -> 分辨率 -> FPS -> buffer。
        cap.set(cv2.CAP_PROP_FOURCC, self.FOURCC)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.CAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.BUFFER_SIZE)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = self._fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        actual_buf = cap.get(cv2.CAP_PROP_BUFFERSIZE)

        self.status_msg.emit(
            f"📷 摄像头: {actual_w}×{actual_h} "
            f"{actual_fps:.1f}fps fourcc={actual_fourcc} buffer={actual_buf:.0f}")

        if actual_fps > 0 and actual_fps < self.CAM_FPS * 0.8:
            self.status_msg.emit(
                f"⚠️ 摄像头实际 FPS={actual_fps:.1f} 低于请求 {self.CAM_FPS}; "
                f"cap.read() 可能被帧周期限制。请用 v4l2-ctl --device={self.CAP_DEVICE} --list-formats-ext 确认 1080p MJPG 支持的 FPS。")

        return cap

    def _read_frame(self, cap):
        """读取一帧并返回 (ret, frame, read_ms)。

        OpenCV 文档说明 read() 等价于 grab()+retrieve()：grab 捕获下一帧，
        retrieve 解码并返回图像。这里保留 read()，因为单摄像头场景下 split 调用通常
        不会降低阻塞帧周期；真正影响 60ms 的通常是实际 FPS/曝光/驱动格式。
        """
        t0 = time.perf_counter()
        ret, frame = cap.read()
        read_ms = (time.perf_counter() - t0) * 1000.0
        return ret, frame, read_ms

    def run(self):
        self._running = True

        cap = self._open_capture()
        if cap is None:
            return

        # ── 延迟导入 C++ 扩展 yolo_core（避免 NPU 在 import 时过早初始化）──
        try:
            import yolo_core
        except ImportError as e:
            self.status_msg.emit(
                f"❌ 无法导入 yolo_core 扩展，请先运行 "
                f"c_realtime_infer_demo/yolo_core/build.sh: {e}")
            cap.release()
            return

        try:
            self._engine = yolo_core.YoloInferEngine(
                self.MODEL_PATH, self.TPEs,
                obj_thresh=0.75, nms_thresh=0.6)
            if not self._engine.init():
                raise RuntimeError("YoloInferEngine.init() returned false")
        except Exception as e:
            self.status_msg.emit(f"❌ RKNN 初始化失败: {e}")
            cap.release()
            return

        # 预填充推理池。原来是 TPEs+1，会额外增加约 1 帧端到端延迟；
        # 这里改成 TPEs，保持 NPU 并行度，同时少排队一帧。
        for i in range(self.TPEs):
            ret, frame, read_ms = self._read_frame(cap)
            if not ret or frame is None:
                self.status_msg.emit(f"❌ 预加载第{i+1}帧失败")
                cap.release()
                self._engine.release()
                return
            self._engine.submit_frame(frame)
        self.status_msg.emit("✅ RKNN 池就绪 (C++ yolo_core)")

        frames = 0
        t_start = time.time()
        t_last_fps = time.time()

        _consecutive_read_failures = 0
        _MAX_READ_FAILURES = 10
        _read_ms_sum = 0.0
        _read_ms_max = 0.0
        _read_ms_n = 0

        while self._running and cap.isOpened():
            ret, frame, read_ms = self._read_frame(cap)
            _read_ms_sum += read_ms
            _read_ms_max = max(_read_ms_max, read_ms)
            _read_ms_n += 1
            if not ret or frame is None:
                _consecutive_read_failures += 1
                if _consecutive_read_failures >= _MAX_READ_FAILURES:
                    self.status_msg.emit("❌ 摄像头读取连续失败，线程退出")
                    break
                QThread.msleep(10)
                continue
            _consecutive_read_failures = 0

            # ── 派生帧: RTSP 推流 + 标签检测 (合并 resize) ──────────────
            # 先缩到 stream 尺寸 (960×540), tag 从它再缩到 640×480;
            # 比对 1920×1080 原图做两次独立 resize 省 ~0.5ms/帧。
            # 本地快照队列引用：set_stream_queue/set_tag_queue 可能从主线程
            # 在本帧处理期间置 None（停止追踪时），若 run() 内再次读取
            # self._*_queue 会拿到 None 并触发 AttributeError 把线程打死，
            # frame_ready 永久停发 → 视频卡死。用局部引用保证一次迭代内一致。
            stream_q = self._stream_queue
            tag_q = self._tag_queue
            need_stream = stream_q is not None
            need_tag = tag_q is not None
            shared_small = None
            if need_stream or need_tag:
                shared_small = cv2.resize(
                    frame,
                    (self._stream_width, self._stream_height),
                    interpolation=cv2.INTER_LINEAR)

            if need_stream:
                # drain 旧帧，仅保留最新帧（等价于原 put_frame 逻辑）
                while True:
                    try:
                        stream_q.get_nowait()
                    except queue.Empty:
                        break
                try:
                    stream_q.put_nowait(shared_small)
                except queue.Full:
                    pass
            else:
                # 回退：Qt 信号模式（用于非推流场景或其他订阅者）
                self.raw_frame_ready.emit(frame.copy())

            if need_tag:
                # 从 960×540 再缩到 tag 尺寸, 比从 1920×1080 原图缩更快
                tag_src = shared_small if shared_small is not None else frame
                tag_frame = cv2.resize(
                    tag_src,
                    (self._tag_width, self._tag_height),
                    interpolation=cv2.INTER_LINEAR)
                # drain 旧帧，仅保留最新帧
                while True:
                    try:
                        tag_q.get_nowait()
                    except queue.Empty:
                        break
                try:
                    tag_q.put_nowait(tag_frame)
                except queue.Full:
                    pass

            # ── 异步提交一帧到 C++ 推理池 (GIL 在 C++ 内释放) ──────────
            self._engine.submit_frame(frame)

            # ── 阻塞取回最早提交且已完成的结果 (FIFO) ────────────────────
            try:
                ok, annotated_frame, _boxes, timing = self._engine.get_result(
                    timeout_ms=10000)
            except Exception as e:
                self.status_msg.emit(f"❌ 推理异常: {e}")
                QThread.msleep(50)
                continue

            if not ok:
                QThread.msleep(1)
                continue

            frames += 1

            # ── 发送主推理阶段的耗时 (来自 get_result) ──────────────────
            self.inference_timing.emit(timing)

            # ── 人体偏差: 复用主检测结果 (与 annotated_frame 严格同帧) ──
            # 原 C++ compute_deviation 会触发第二次完整 NPU 推理 (pool[0]),
            # 是端到端延迟翻倍的主因; 主推理已返回 person 框, 直接复用即可。
            # 附带好处: 偏差与标注帧同帧, 比原版 (偏差=当前帧/标注=旧帧) 更对齐。
            dev = None
            best_box = None
            best_area = -1.0
            if frame is not None and len(frame.shape) == 3:
                img_h, img_w = frame.shape[:2]
                img_cx = img_w * 0.5
                img_cy = img_h * 0.5
                for b in _boxes:
                    if b.get("cls_id") != 0:
                        continue
                    if b.get("confidence", 0.0) < 0.75:
                        continue
                    bw = b["right"] - b["left"]
                    bh = b["bottom"] - b["top"]
                    area = bw * bh
                    if area > best_area:
                        best_area = area
                        best_box = b
                if best_box is not None:
                    cx = (best_box["left"] + best_box["right"]) * 0.5
                    cy = (best_box["top"] + best_box["bottom"]) * 0.5
                    dev = ((cx - img_cx) / img_cx, (cy - img_cy) / img_cy)

            if dev is None:
                self.deviation_data.emit((None, None))
                self.person_box_ready.emit(None)
            else:
                self.deviation_data.emit(dev)
                self.person_box_ready.emit({
                    "left":   best_box["left"],
                    "top":    best_box["top"],
                    "right":  best_box["right"],
                    "bottom": best_box["bottom"],
                })

            # ── 发送带标注帧 (C++ get_result 已缩放到 ≤960px 宽, 直接 emit) ──
            if annotated_frame is not None and len(annotated_frame.shape) == 3:
                self.frame_ready.emit(annotated_frame)

            if frames >= 15:
                now = time.time()
                elapsed = now - t_last_fps
                if elapsed > 0:
                    fps = frames / max(now - t_start, 0.001)
                    self.fps_update.emit(fps)
                    if _read_ms_n > 0:
                        avg_read_ms = _read_ms_sum / _read_ms_n
                        timing_with_read = dict(timing) if isinstance(timing, dict) else {}
                        timing_with_read["camera_read_ms"] = avg_read_ms
                        timing_with_read["camera_read_max_ms"] = _read_ms_max
                        self.inference_timing.emit(timing_with_read)
                        if avg_read_ms > 45.0:
                            self.status_msg.emit(
                                f"⚠️ 摄像头 read 平均 {avg_read_ms:.1f}ms, max {_read_ms_max:.1f}ms；"
                                f"请检查实际 FPS/曝光/USB 带宽，或降低 CAM_WIDTH/CAM_HEIGHT/CAM_FPS。")
                t_last_fps = now
                frames = 0
                t_start = now
                _read_ms_sum = 0.0
                _read_ms_max = 0.0
                _read_ms_n = 0

        if self._engine is not None:
            self._engine.release()
            self._engine = None
        cap.release()
        self.status_msg.emit("📷 摄像头已释放")

    def stop(self):
        self._running = False
        self.wait(3000)
