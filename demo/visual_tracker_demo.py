#!/usr/bin/env python3
"""
视觉追踪 Demo — 基于 YOLOv8 人体检测 + 舵机水平追踪 (PyQt5 单文件)

用法:
  sudo env DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority python3 visual_tracker_demo.py

依赖:
  - rknnlite (Rockchip NPU)
  - opencv-python (cv2)
  - PyQt5
  - numpy
"""

import os
import sys
import time
import atexit

# ── 修复 OpenCV 内建 Qt 插件与 PyQt5 冲突 ────────────────────────────────────
# cv2 导入后会设置 QT_PLUGIN_PATH 指向自带的 Qt 插件目录，
# 导致 PyQt5 加载到不兼容的 xcb 平台插件而崩溃。
os.environ.setdefault("DISPLAY", ":0.0")

import cv2
import numpy as np

# 清除 cv2 设置的 QT_PLUGIN_PATH，改为使用系统 Qt5 插件
_CV2_QT_PLUGIN_PATH = "/usr/local/lib/python3.10/dist-packages/cv2/qt/plugins"
if os.environ.get("QT_PLUGIN_PATH") == _CV2_QT_PLUGIN_PATH:
    del os.environ["QT_PLUGIN_PATH"]

# 显式指定系统 Qt5 平台插件路径 (Rockchip ARM Linux)
_SYS_QT5_PLATFORM_PATH = "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms"
if os.path.isdir(_SYS_QT5_PLATFORM_PATH):
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _SYS_QT5_PLATFORM_PATH

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QStatusBar,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont

from rknnpool.rknnpool_ld import rknnPoolExecutor
from func.func_yolov8_optimize import myFunc

# ═════════════════════════════════════════════════════════════════════════════════
# 1. PWMController (from servo_test.py / calibrate_gui.py)
# ═════════════════════════════════════════════════════════════════════════════════

class PWMController:
    """PWM 舵机控制器 — 与 servo_test.py 使用相同常量。

    支持通过构造函数参数配置不同 PWM 通道和角度范围。
    """

    PERIOD_NS = 10_000_000                                       # 100 Hz
    PCT_5  = int(PERIOD_NS * (1 - 0.05))                         # 9_500_000 ns
    PCT_25 = int(PERIOD_NS * (1 - 0.25))                         # 7_500_000 ns
    PCT_15 = int(PERIOD_NS * (1 - 0.15))                         # 8_500_000 ns (中间)

    def __init__(self, pwmchip: int, pwm_index: str = "0",
                 angle_min: float = -135, angle_max: float = 135,
                 duty_at_min: int | None = None,
                 duty_at_max: int | None = None,
                 label: str = "PWM"):
        """
        pwmchip:  PWM 控制器编号 (0 → /sys/class/pwm/pwmchip0)
        pwm_index: PWM 通道索引，默认 "0"
        angle_min: 最小角度 (°)
        angle_max: 最大角度 (°)
        duty_at_min: angle_min 对应的占空比 (ns)。None=默认按 25%→min, 5%→max
        duty_at_max: angle_max 对应的占空比 (ns)。垂直舵机需要翻转此参数。
        label: 日志标签
        """
        self.PWMCHIP_PATH = f"/sys/class/pwm/pwmchip{pwmchip}"
        self.PWM_INDEX = pwm_index
        self.ANGLE_MIN = float(angle_min)
        self.ANGLE_MAX = float(angle_max)
        self.label = label

        # 占空比→角度 映射: 默认 5%→angle_max, 25%→angle_min (与 servo_test.py 一致)
        if duty_at_max is None:
            self.DUTY_AT_MAX = self.PCT_5     # 9.5M → angle_max (水平: +135°)
        else:
            self.DUTY_AT_MAX = int(duty_at_max)
        if duty_at_min is None:
            self.DUTY_AT_MIN = self.PCT_25    # 7.5M → angle_min (水平: -135°)
        else:
            self.DUTY_AT_MIN = int(duty_at_min)
        self.DUTY_MID = self.PCT_15           # 8.5M → 0°

        self._pwm_base = os.path.join(self.PWMCHIP_PATH, f"pwm{self.PWM_INDEX}")
        self._current_angle = 0.0
        self._initialized = False

    @staticmethod
    def _write_file(path: str, value):
        with open(path, "w") as f:
            f.write(str(value))

    def init(self) -> bool:
        """导出 PWM 通道, 设置周期, 使能。"""
        try:
            export_path = os.path.join(self.PWMCHIP_PATH, "export")
            if not os.path.exists(self._pwm_base):
                self._write_file(export_path, self.PWM_INDEX)
                time.sleep(0.1)

            self._write_file(os.path.join(self._pwm_base, "period"), self.PERIOD_NS)
            self._write_file(os.path.join(self._pwm_base, "duty_cycle"), self.DUTY_MID)
            self._write_file(os.path.join(self._pwm_base, "enable"), "1")
            self._initialized = True
            self._current_angle = 0.0
            print(f"[{self.label}] 初始化成功 ({self.ANGLE_MIN}°~{self.ANGLE_MAX}°)")
            return True
        except PermissionError:
            print(f"[{self.label}] 权限不足，请使用 sudo 运行")
            return False
        except Exception as e:
            print(f"[{self.label}] 初始化失败: {e}")
            return False

    def cleanup(self):
        """失能并反导出 PWM。"""
        if not self._initialized:
            return
        try:
            self._write_file(os.path.join(self._pwm_base, "enable"), "0")
            unexport_path = os.path.join(self.PWMCHIP_PATH, "unexport")
            self._write_file(unexport_path, self.PWM_INDEX)
            print(f"[{self.label}] 已清理")
        except Exception as e:
            print(f"[{self.label}] 清理警告: {e}")
        self._initialized = False

    def angle_to_duty(self, angle: float) -> int:
        """角度 → duty_cycle (ns)。

        映射规则: angle_min → DUTY_AT_MIN, 0° → DUTY_MID, angle_max → DUTY_AT_MAX。
        """
        angle = max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle))
        if angle >= 0:
            ratio = angle / float(self.ANGLE_MAX)          # 0.0 ~ 1.0
            duty = self.DUTY_MID + ratio * (self.DUTY_AT_MAX - self.DUTY_MID)
        else:
            ratio = angle / float(self.ANGLE_MIN)           # 0.0 ~ 1.0 (angle 为负, 除 min 得正)
            duty = self.DUTY_MID + ratio * (self.DUTY_AT_MIN - self.DUTY_MID)
        return int(duty)

    def set_angle(self, angle: float):
        """设置舵机到指定角度。"""
        if not self._initialized:
            return
        angle = max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle))
        duty = self.angle_to_duty(angle)
        self._write_file(os.path.join(self._pwm_base, "duty_cycle"), duty)
        self._current_angle = angle

    def get_angle(self) -> float:
        return self._current_angle

    @property
    def initialized(self) -> bool:
        return self._initialized


# ═════════════════════════════════════════════════════════════════════════════════
# 2. CameraThread — 后台摄像头 + NPU 推理线程
# ═════════════════════════════════════════════════════════════════════════════════

class CameraThread(QThread):
    """后台线程: 读取摄像头 → RKNN 推理 → 发送带标注帧 + 人体偏差。

    使用 rknnPoolExecutor 做异步推理（与 cam1.py 完全一致），
    通过 Qt 信号将结果发送到主线程。
    """

    # 信号
    frame_ready    = pyqtSignal(np.ndarray)              # 带检测框标注的帧
    deviation_data = pyqtSignal(object)                  # (dev_x, dev_y) 或 None
    fps_update     = pyqtSignal(float)                    # 推理帧率
    status_msg     = pyqtSignal(str)                      # 状态信息

    CAP_DEVICE   = "/dev/video21"
    CAM_WIDTH    = 1920
    CAM_HEIGHT   = 1080
    FOURCC       = cv2.VideoWriter_fourcc(*'MJPG')
    BUFFER_SIZE  = 1
    TPEs         = 4                                     # RKNN 线程数
    MODEL_PATH   = "./rknnModel/best.rknn"

    def __init__(self):
        super().__init__()  # 不传 parent，避免 moveToThread 警告
        self._running = False
        self._pool = None

    def run(self):
        self._running = True

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
        self.status_msg.emit(f"📷 摄像头: {actual_w}×{actual_h}")

        # ── 初始化 RKNN 池 ──────────────────────────────────────────────────
        try:
            self._pool = rknnPoolExecutor(
                rknnModel=self.MODEL_PATH,
                TPEs=self.TPEs,
                func=myFunc,
            )
        except Exception as e:
            self.status_msg.emit(f"❌ RKNN 初始化失败: {e}")
            cap.release()
            return

        # ── 预加载帧 ────────────────────────────────────────────────────────
        for i in range(self.TPEs + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                self.status_msg.emit(f"❌ 预加载第{i+1}帧失败")
                cap.release()
                self._pool.release()
                return
            self._pool.put(frame)
        self.status_msg.emit("✅ RKNN 池就绪")

        # ── 主循环 ──────────────────────────────────────────────────────────
        frames = 0
        t_start = time.time()
        t_last_fps = time.time()

        # ---------- 人体偏差计算 (同 cam1.py:77-106) ----------
        # 缓存一份 rknn 实例用于偏差计算
        _rknn_for_dev = self._pool.rknnPool[0]

        from func.func_yolov8_optimize import yolov8_post_process, letterbox

        def compute_person_deviation(frame_bgr):
            """在给定帧上做一次独立的 YOLOv8 推理，返回最大人体的中心偏差。"""
            try:
                img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img_rgb, ratio, padding = letterbox(img_rgb)
                img_in = np.expand_dims(img_rgb, 0)
                img_in = np.ascontiguousarray(img_in)
                outputs = _rknn_for_dev.inference(inputs=[img_in], data_format=['nhwc'])
                boxes, classes, scores = yolov8_post_process(outputs)
                if boxes is None:
                    return None, None

                person_boxes = []
                for box, score, cl in zip(boxes, scores, classes):
                    if int(cl) == 0:               # 仅 person
                        left, top, right, bottom = box          # box 格式: [left, top, right, bottom]
                        left   = int((left   - padding[0]) / ratio[0])
                        top    = int((top    - padding[1]) / ratio[1])
                        right  = int((right  - padding[0]) / ratio[0])
                        bottom = int((bottom - padding[1]) / ratio[1])
                        area = (bottom - top) * (right - left)
                        center_x = (left + right) / 2.0
                        center_y = (top + bottom) / 2.0
                        person_boxes.append((area, center_x, center_y))

                if not person_boxes:
                    return None, None

                person_boxes.sort(key=lambda x: x[0], reverse=True)
                _, cx, cy = person_boxes[0]
                img_cx = frame_bgr.shape[1] / 2.0
                img_cy = frame_bgr.shape[0] / 2.0
                dev_x = (cx - img_cx) / img_cx
                dev_y = (cy - img_cy) / img_cy
                return dev_x, dev_y
            except Exception:
                return None, None
        # ------------------------------------------------------

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            self._pool.put(frame)

            result = self._pool.get()
            if result[0] is not None:
                annotated_frame, flag = result
                if not flag:
                    self.status_msg.emit("❌ 推理失败")
                    break

                frames += 1

                # 计算人体偏差（在原始帧 run 上做一次额外推理，与 cam1.py 一致）
                if frame is not None and len(frame.shape) == 3:
                    dev_x, dev_y = compute_person_deviation(frame)
                    self.deviation_data.emit((dev_x, dev_y))

                # 缩放后发送到主线程显示
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
                # 队列为空，短暂等待
                QThread.msleep(1)

            # 每秒更新一次 FPS
            if frames >= 15:
                now = time.time()
                dt = now - t_last_fps
                if dt > 0:
                    fps = frames / max(now - t_start, 0.001)
                    self.fps_update.emit(fps)
                t_last_fps = now

        # ── 清理 ────────────────────────────────────────────────────────────
        if self._pool is not None:
            self._pool.release()
        cap.release()
        self.status_msg.emit("📷 摄像头已释放")

    def stop(self):
        self._running = False
        self.wait(3000)


# ═════════════════════════════════════════════════════════════════════════════════
# 3. TrackerWindow — 主窗口 (含 UI + 追踪逻辑)
# ═════════════════════════════════════════════════════════════════════════════════

class TrackerWindow(QMainWindow):
    """PyQt5 主窗口: 摄像头预览 + 追踪控制 + 舵机驱动。"""

    # 追踪参数默认值
    DEFAULT_DEADZONE    = 0.08     # 死区 (归一化坐标)
    DEFAULT_GAIN        = 0.7      # 比例增益 (水平)
    DEFAULT_GAIN_V      = 0.5      # 比例增益 (垂直，较小避免频繁抬头)
    DEFAULT_COOLDOWN     = 0.35    # 两次动作最短间隔 (秒)
    DEFAULT_MAX_ANGLE_V = 10.0     # 垂直追踪角度范围限制 (±10°)
    MAX_ANGLE_ERROR_H   = 30.0     # 水平最大追踪角度偏移 (度)
    MAX_ANGLE_ERROR_V   = 15.0     # 垂直最大追踪角度偏移 (度)

    def __init__(self, pwm_h: PWMController, pwm_v: PWMController | None = None):
        super().__init__()
        self.pwm_h = pwm_h
        self.pwm_v = pwm_v                     # None = 无垂直舵机

        # ── 运行时状态 ────────────────────────────────────────────────────
        self._tracking_active = False
        self._latest_dev_x: float | None = None
        self._latest_dev_y: float | None = None
        self._last_move_time  = 0.0
        self._in_cooldown     = False
        self._fps             = 0.0

        # ── UI 参数 ────────────────────────────────────────────────────────
        self._deadzone  = self.DEFAULT_DEADZONE
        self._gain_h    = self.DEFAULT_GAIN
        self._gain_v    = self.DEFAULT_GAIN_V
        self._max_angle_v = self.DEFAULT_MAX_ANGLE_V
        self._cooldown  = self.DEFAULT_COOLDOWN

        # ── 构建 UI ────────────────────────────────────────────────────────
        self._build_ui()

        # ── 启动后台线程 ───────────────────────────────────────────────────
        self._camera_thread = CameraThread()
        self._camera_thread.frame_ready.connect(self._on_frame)
        self._camera_thread.deviation_data.connect(self._on_deviation)
        self._camera_thread.fps_update.connect(self._on_fps)
        self._camera_thread.status_msg.connect(self._on_status)
        self._camera_thread.start()

    # ── UI 构建 ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("视觉追踪 Demo — YOLOv8 + 舵机")
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── 画面预览 ──────────────────────────────────────────────────────
        self._video_label = QLabel("正在启动摄像头…")
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setMinimumSize(640, 360)
        self._video_label.setStyleSheet("background: black; color: white;")
        layout.addWidget(self._video_label, stretch=1)

        # ── 信息面板 ──────────────────────────────────────────────────────
        info_layout = QGridLayout()

        self._lbl_fps = QLabel("FPS: --")
        self._lbl_fps.setFont(QFont("", 12, QFont.Bold))
        info_layout.addWidget(QLabel("推理帧率:"), 0, 0)
        info_layout.addWidget(self._lbl_fps, 0, 1)

        self._lbl_dev = QLabel("偏差: --")
        self._lbl_dev.setFont(QFont("", 12, QFont.Bold))
        info_layout.addWidget(QLabel("水平/垂直偏差:"), 1, 0)
        info_layout.addWidget(self._lbl_dev, 1, 1)

        self._lbl_servo_h = QLabel(f"舵机H: {self.pwm_h.get_angle():.1f}°")
        self._lbl_servo_h.setFont(QFont("", 12, QFont.Bold))
        info_layout.addWidget(QLabel("水平舵机:"), 2, 0)
        info_layout.addWidget(self._lbl_servo_h, 2, 1)

        servo_v_text = f"舵机V: {self.pwm_v.get_angle():.1f}°" if self.pwm_v else "舵机V: (未安装)"
        self._lbl_servo_v = QLabel(servo_v_text)
        self._lbl_servo_v.setFont(QFont("", 12, QFont.Bold))
        info_layout.addWidget(QLabel("垂直舵机:"), 3, 0)
        info_layout.addWidget(self._lbl_servo_v, 3, 1)

        self._lbl_state = QLabel("⏸ 停止")
        self._lbl_state.setFont(QFont("", 14, QFont.Bold))
        self._lbl_state.setStyleSheet("color: gray;")
        info_layout.addWidget(QLabel("状态:"), 4, 0)
        info_layout.addWidget(self._lbl_state, 4, 1)

        layout.addLayout(info_layout)

        # ── 控制按钮 ──────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self._btn_toggle = QPushButton("▶ 开始追踪")
        self._btn_toggle.setFont(QFont("", 12, QFont.Bold))
        self._btn_toggle.clicked.connect(self._on_toggle)
        btn_layout.addWidget(self._btn_toggle)

        self._btn_center = QPushButton("⬅ 回中")
        self._btn_center.setFont(QFont("", 12))
        self._btn_center.clicked.connect(self._on_center)
        btn_layout.addWidget(self._btn_center)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # ── 参数调节 ──────────────────────────────────────────────────────
        param_group = QGroupBox("追踪参数")
        param_layout = QHBoxLayout(param_group)

        # 死区
        param_layout.addWidget(QLabel("死区:"))
        self._spin_deadzone = QDoubleSpinBox()
        self._spin_deadzone.setRange(0.0, 0.5)
        self._spin_deadzone.setSingleStep(0.01)
        self._spin_deadzone.setDecimals(2)
        self._spin_deadzone.setValue(self._deadzone)
        self._spin_deadzone.setToolTip("偏差小于此值不动作 (归一化坐标)")
        self._spin_deadzone.valueChanged.connect(self._on_deadzone_changed)
        param_layout.addWidget(self._spin_deadzone)

        # 增益 (水平) — 负值反转追踪方向
        param_layout.addWidget(QLabel("增益H:"))
        self._spin_gain_h = QDoubleSpinBox()
        self._spin_gain_h.setRange(-3.0, 3.0)
        self._spin_gain_h.setSingleStep(0.1)
        self._spin_gain_h.setDecimals(1)
        self._spin_gain_h.setValue(self._gain_h)
        self._spin_gain_h.setToolTip("水平增益: 正值=跟随, 负值=反向")
        self._spin_gain_h.valueChanged.connect(self._on_gain_h_changed)
        param_layout.addWidget(self._spin_gain_h)

        # 增益 (垂直) — 负值反转追踪方向
        param_layout.addWidget(QLabel("增益V:"))
        self._spin_gain_v = QDoubleSpinBox()
        self._spin_gain_v.setRange(-2.0, 2.0)
        self._spin_gain_v.setSingleStep(0.1)
        self._spin_gain_v.setDecimals(1)
        self._spin_gain_v.setValue(self._gain_v)
        self._spin_gain_v.setToolTip("垂直增益: 正值=跟随, 负值=反向")
        self._spin_gain_v.valueChanged.connect(self._on_gain_v_changed)
        param_layout.addWidget(self._spin_gain_v)

        # 垂直角度范围限制
        param_layout.addWidget(QLabel("V范围(°):"))
        self._spin_max_angle_v = QDoubleSpinBox()
        self._spin_max_angle_v.setRange(1.0, 90.0)
        self._spin_max_angle_v.setSingleStep(5.0)
        self._spin_max_angle_v.setDecimals(1)
        self._spin_max_angle_v.setValue(self._max_angle_v)
        self._spin_max_angle_v.setToolTip("垂直舵机角度范围 ±N° (对称限制)")
        self._spin_max_angle_v.valueChanged.connect(self._on_max_angle_v_changed)
        param_layout.addWidget(self._spin_max_angle_v)

        # 冷却
        param_layout.addWidget(QLabel("冷却(s):"))
        self._spin_cooldown = QDoubleSpinBox()
        self._spin_cooldown.setRange(0.1, 3.0)
        self._spin_cooldown.setSingleStep(0.05)
        self._spin_cooldown.setDecimals(2)
        self._spin_cooldown.setSuffix("s")
        self._spin_cooldown.setValue(self._cooldown)
        self._spin_cooldown.setToolTip("两次动作最短间隔")
        self._spin_cooldown.valueChanged.connect(self._on_cooldown_changed)
        param_layout.addWidget(self._spin_cooldown)

        param_layout.addStretch()
        layout.addWidget(param_group)

        # ── 状态栏 ────────────────────────────────────────────────────────
        self._status = QStatusBar()
        self._status.showMessage("就绪 — 点击「开始追踪」启动视觉追踪")
        self.setStatusBar(self._status)

    # ── 信号处理 ────────────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray):
        """接收摄像头帧并显示。"""
        h, w, ch = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._video_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self._video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def _on_deviation(self, data):
        """接收人体偏差数据。"""
        if data is None:
            self._latest_dev_x = None
            self._latest_dev_y = None
            return
        dev_x, dev_y = data
        self._latest_dev_x = dev_x
        self._latest_dev_y = dev_y

        if dev_x is not None:
            h_str = f"H:{dev_x:+.3f}"
            v_str = f" V:{dev_y:+.3f}" if dev_y is not None else " V:--"
            self._lbl_dev.setText(h_str + v_str)
            outside_deadzone = abs(dev_x) >= self._deadzone
            self._lbl_dev.setStyleSheet("color: red;" if outside_deadzone else "color: green;")
        else:
            self._lbl_dev.setText("无人")
            self._lbl_dev.setStyleSheet("color: gray;")

        # 如果正在追踪，立即执行追踪逻辑
        if self._tracking_active:
            self._tracking_tick()

    def _on_fps(self, fps: float):
        self._fps = fps
        self._lbl_fps.setText(f"{fps:.1f}")

    def _on_status(self, msg: str):
        self._status.showMessage(msg)
        print(f"[CameraThread] {msg}")

    def _on_deadzone_changed(self, val):
        self._deadzone = val

    def _on_gain_h_changed(self, val):
        self._gain_h = val

    def _on_gain_v_changed(self, val):
        self._gain_v = val

    def _on_max_angle_v_changed(self, val):
        self._max_angle_v = val

    def _on_cooldown_changed(self, val):
        self._cooldown = val

    # ── 追踪逻辑 ───────────────────────────────────────────────────────────────

    def _on_toggle(self):
        if self._tracking_active:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        if not self.pwm_h.initialized:
            self._status.showMessage("⚠️ 水平 PWM 未初始化 (需 root 权限)")
            return

        self._tracking_active = True
        self._last_move_time = 0.0
        self._in_cooldown    = False
        self._lbl_state.setText("🎯 追踪中")
        self._lbl_state.setStyleSheet("color: blue; font-weight: bold;")
        self._btn_toggle.setText("⏹ 停止追踪")
        v_status = "H+V" if (self.pwm_v and self.pwm_v.initialized) else "仅水平"
        self._status.showMessage(f"追踪已启动 ({v_status})")
        print(f"[Tracker] 追踪已启动 ({v_status})")

    def _stop_tracking(self):
        self._tracking_active = False
        self._lbl_state.setText("⏸ 停止")
        self._lbl_state.setStyleSheet("color: gray;")
        self._btn_toggle.setText("▶ 开始追踪")
        self._status.showMessage("追踪已停止")
        print("[Tracker] 追踪已停止")

    def _on_center(self):
        """手动回到中间位置（双轴）。"""
        self.pwm_h.set_angle(0.0)
        self._lbl_servo_h.setText("舵机H: 0.0°")
        if self.pwm_v and self.pwm_v.initialized:
            self.pwm_v.set_angle(0.0)
            self._lbl_servo_v.setText("舵机V: 0.0°")
            self._status.showMessage("舵机已回中 (H:0° V:0°)")
        else:
            self._status.showMessage("舵机已回中 (H:0°)")
        print("[Tracker] 舵机回中")

    def _tracking_tick(self):
        """追踪主逻辑 — 双轴比例控制 + 死区 + 冷却。"""
        now = time.time()

        # 冷却检查
        if self._in_cooldown and (now - self._last_move_time) < self._cooldown:
            return
        self._in_cooldown = False

        dev_x = self._latest_dev_x
        dev_y = self._latest_dev_y

        # 无人 → 不动作
        if dev_x is None:
            return

        moved = False

        # ── 水平追踪 ─────────────────────────────────────────────────────
        if abs(dev_x) >= self._deadzone:
            adjustment = dev_x * self._gain_h * self.MAX_ANGLE_ERROR_H
            target_h = self.pwm_h.get_angle() + adjustment
            target_h = max(self.pwm_h.ANGLE_MIN, min(self.pwm_h.ANGLE_MAX, target_h))
            self.pwm_h.set_angle(target_h)
            self._lbl_servo_h.setText(f"舵机H: {target_h:.1f}°")
            moved = True

        # ── 垂直追踪 ─────────────────────────────────────────────────────
        if (self.pwm_v and self.pwm_v.initialized
                and dev_y is not None and abs(dev_y) >= self._deadzone):
            adjustment_v = dev_y * self._gain_v * self.MAX_ANGLE_ERROR_V
            target_v = self.pwm_v.get_angle() + adjustment_v
            # 软件限位: ±max_angle_v (同时不超过硬件边界)
            limit_v = min(self._max_angle_v, self.pwm_v.ANGLE_MAX)
            target_v = max(-limit_v, min(limit_v, target_v))
            self.pwm_v.set_angle(target_v)
            self._lbl_servo_v.setText(f"舵机V: {target_v:.1f}°")
            moved = True

        if moved:
            self._last_move_time = now
            self._in_cooldown = True

    # ── 窗口关闭 ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        """窗口关闭时清理资源。"""
        self._stop_tracking()
        if self._camera_thread.isRunning():
            self._camera_thread.stop()
        event.accept()


# ═════════════════════════════════════════════════════════════════════════════════
# 4. main()
# ═════════════════════════════════════════════════════════════════════════════════

def main():
    # ── PWM 初始化 ──────────────────────────────────────────────────────────
    # 水平舵机: pwmchip0, -135°~+135°, 5%→+135°, 25%→-135° (默认映射)
    pwm_h = PWMController(pwmchip=0, pwm_index="0",
                          angle_min=-135, angle_max=135,
                          label="PWM-H")
    h_ok = pwm_h.init()
    if not h_ok:
        print("⚠️ 水平 PWM 初始化失败，追踪功能不可用（继续运行以展示摄像头预览）")

    # 垂直舵机: pwmchip1, -90°~+90°, 5%→-90°, 25%→+90° (翻转映射)
    pwm_v = PWMController(pwmchip=1, pwm_index="0",
                          angle_min=-90, angle_max=90,
                          duty_at_min=PWMController.PCT_5,     # 5% (9.5M) → -90°
                          duty_at_max=PWMController.PCT_25,    # 25% (7.5M) → +90°
                          label="PWM-V")
    v_ok = pwm_v.init()
    if not v_ok:
        print("⚠️ 垂直 PWM 初始化失败，仅水平追踪可用")

    # 注册退出清理
    atexit.register(pwm_h.cleanup)
    atexit.register(pwm_v.cleanup)

    # ── PyQt5 应用 ───────────────────────────────────────────────────────────
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = TrackerWindow(pwm_h, pwm_v if v_ok else None)
    window.resize(1024, 768)
    window.show()

    print("=" * 50)
    print("视觉追踪 Demo 已启动")
    print(f"  - 水平舵机: {'✅' if h_ok else '❌'} pwmchip0 (-135°~+135°)")
    print(f"  - 垂直舵机: {'✅' if v_ok else '❌'} pwmchip1 (-90°~+90°)")
    print("  - 开始追踪: 点击「开始追踪」")
    print("  - 关闭窗口或 Ctrl+C 退出")
    print("=" * 50)

    try:
        sys.exit(app.exec_())
    finally:
        pwm_h.cleanup()
        pwm_v.cleanup()


if __name__ == "__main__":
    main()
