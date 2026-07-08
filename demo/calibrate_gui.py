#!/usr/bin/env python3
"""
XVF-Servo 标定程序 (PyQt5 GUI)

功能：
  - 标定 ReSpeaker (XVF) DOA 角度 与 舵机角度的对应关系
  - 支持多点线性拟合并可视化
  - 实时声源跟踪测试（带角度 unwrap）
  - 标定参数双持久化：DB (SystemConfig) + JSON (xvf_calibration.json)
  - 自动修复 sudo 下的 DISPLAY / XAUTHORITY 环境变量

用法:
  sudo env PYTHONPATH=~/.local/lib/python3.10/site-packages python3 calibrate_gui.py
"""

import os
import sys
import time
import json
import struct
import threading
import math
from collections import deque
from datetime import datetime

import numpy as np

# ── Display 环境修复 (必须在 PyQt5 之前) ─────────────────────────────────────
from core.display_env import fix_display_env
fix_display_env()

# ── Storage (DB persistence) ──────────────────────────────────────────────────
from storage.db import session_scope, init_db, db_path
from storage.repo import ConfigRepo

# ── PyQt5 ──────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QSlider, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QTextEdit, QHeaderView,
    QMenuBar, QMenu, QAction, QStatusBar, QMessageBox, QSplitter,
    QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QMutex
from PyQt5.QtGui import QFont, QColor, QPalette

# ── matplotlib (embedded in PyQt5) ─────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ── USB (ReSpeaker) ────────────────────────────────────────────────────────
import usb.core
import usb.util

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. PWMController — wraps servo_test.py PWM sysfs logic                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class PWMController:
    """PWM 舵机控制器，直接写入 sysfs。与 servo_test.py 使用相同常量。

    舵机水平范围: -135° (左) ~ +135° (右), 总计 270°。
    PWM 边界不变: 5% 占空比 = 最右端 (+135°), 25% 占空比 = 最左端 (-135°)。
    """

    PWMCHIP_PATH = "/sys/class/pwm/pwmchip0"
    PWM_INDEX    = "0"

    PERIOD_NS          = 10_000_000            # 100 Hz
    DUTY_CYCLE_RIGHT   = int(PERIOD_NS * (1 - 0.05))   # 9_500_000  → +135° (最右)
    DUTY_CYCLE_LEFT    = int(PERIOD_NS * (1 - 0.25))   # 7_500_000  → -135° (最左)
    DUTY_CYCLE_MID     = int(PERIOD_NS * (1 - 0.15))   # 8_500_000  →   0° (中间)

    ANGLE_MIN = -135
    ANGLE_MAX =  135

    def __init__(self):
        self._pwm_base = os.path.join(self.PWMCHIP_PATH, f"pwm{self.PWM_INDEX}")
        self._current_angle = 0.0
        self._initialized = False

    # ── low-level sysfs helpers ──────────────────────────────────────────

    @staticmethod
    def _write_file(path: str, value):
        with open(path, "w") as f:
            f.write(str(value))

    # ── public API ───────────────────────────────────────────────────────

    def init(self) -> bool:
        """导出 PWM 通道, 设置周期, 使能。返回是否成功。"""
        try:
            export_path = os.path.join(self.PWMCHIP_PATH, "export")
            if not os.path.exists(self._pwm_base):
                self._write_file(export_path, self.PWM_INDEX)
                time.sleep(0.1)

            self._write_file(os.path.join(self._pwm_base, "period"), self.PERIOD_NS)
            self._write_file(os.path.join(self._pwm_base, "duty_cycle"), self.DUTY_CYCLE_MID)
            self._write_file(os.path.join(self._pwm_base, "enable"), "1")
            self._initialized = True
            self._current_angle = 0.0
            return True
        except PermissionError:
            print("[PWM] 权限不足，请使用 sudo 运行")
            return False
        except Exception as e:
            print(f"[PWM] 初始化失败: {e}")
            return False

    def cleanup(self):
        """失能并反导出 PWM。"""
        if not self._initialized:
            return
        try:
            self._write_file(os.path.join(self._pwm_base, "enable"), "0")
            unexport_path = os.path.join(self.PWMCHIP_PATH, "unexport")
            self._write_file(unexport_path, self.PWM_INDEX)
        except Exception as e:
            print(f"[PWM] 清理警告: {e}")
        self._initialized = False

    def angle_to_duty(self, angle: float) -> int:
        """角度 [-135, 135] → duty_cycle (ns)。

        线性映射: 0° → MID (8.5M), +135° → RIGHT (9.5M), -135° → LEFT (7.5M)
        """
        angle = max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle))
        ratio = angle / float(self.ANGLE_MAX)          # -1.0 ~ +1.0
        # 从 MID 向 RIGHT(+135) 或 LEFT(-135) 偏移
        if ratio >= 0:
            duty = self.DUTY_CYCLE_MID + ratio * (self.DUTY_CYCLE_RIGHT - self.DUTY_CYCLE_MID)
        else:
            duty = self.DUTY_CYCLE_MID + ratio * (self.DUTY_CYCLE_MID - self.DUTY_CYCLE_LEFT)
        return int(duty)

    def duty_to_angle(self, duty_ns: int) -> float:
        """duty_cycle (ns) → 角度 [-135, 135]。"""
        d_mid = float(self.DUTY_CYCLE_MID)
        if duty_ns >= d_mid:
            ratio = (duty_ns - d_mid) / (self.DUTY_CYCLE_RIGHT - d_mid)
        else:
            ratio = (duty_ns - d_mid) / (d_mid - self.DUTY_CYCLE_LEFT)
        return ratio * float(self.ANGLE_MAX)

    def set_angle(self, angle: float):
        """设置舵机到指定角度 [-135, 135]。"""
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. ReSpeaker USB 驱动 (from xvf_test.py)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

_REASPEAKER_VID = 0x2886
_REASPEAKER_PID = 0x001A

_PARAMETERS = {
    "VERSION":             (48,  0,  3,  "ro", "uint8"),
    "AEC_AZIMUTH_VALUES":  (33, 75, 16, "ro", "radians"),
    "DOA_VALUE":           (20, 18,  4,  "ro", "uint16"),
    "REBOOT":              (48,  7,  1,  "wo", "uint8"),
}


class ReSpeaker:
    """ReSpeaker USB 麦克风阵列驱动 (from xvf_test.py)"""

    TIMEOUT = 100_000  # USB timeout (ms)

    def __init__(self, dev):
        self.dev = dev

    def read(self, name: str):
        try:
            meta = _PARAMETERS[name]
        except KeyError:
            return None

        resid  = meta[0]
        cmdid  = 0x80 | meta[1]
        length = meta[2] + 1          # +1 for status byte

        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmdid, resid, length, self.TIMEOUT)

        byte_data = response.tobytes()

        if meta[4] == "uint8":
            return response.tolist()
        elif meta[4] == "radians":
            num_floats = (length - 1) // 4
            fmt = "<" + "f" * num_floats
            return list(struct.unpack(fmt, byte_data[1:1 + num_floats * 4]))
        elif meta[4] == "uint16":
            num_words = meta[2] // 2
            fmt = "<" + "H" * num_words
            return list(struct.unpack(fmt, byte_data[1:1 + num_words * 2]))
        return None

    def close(self):
        usb.util.dispose_resources(self.dev)


def find_respeaker() -> ReSpeaker | None:
    """查找 ReSpeaker USB 设备。"""
    dev = usb.core.find(idVendor=_REASPEAKER_VID, idProduct=_REASPEAKER_PID)
    if dev is None:
        return None
    return ReSpeaker(dev)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. ReSpeakerReader(QThread) — 后台轮询 XVF DOA 数据                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class ReSpeakerReader(QThread):
    """在后台线程中连续读取 ReSpeaker DOA 数据，通过 Qt 信号发送给主线程。"""

    doa_update    = pyqtSignal(float, bool)   # doa_angle, speech_detected
    device_error  = pyqtSignal(str)
    device_ready  = pyqtSignal(str)           # 版本字符串

    POLL_INTERVAL = 0.08   # 秒 (~12.5 Hz)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._respeaker: ReSpeaker | None = None
        self._mutex = QMutex()

    def run(self):
        self._running = True

        # 尝试连接设备
        while self._running:
            dev = find_respeaker()
            if dev is not None:
                self._respeaker = dev
                try:
                    ver = dev.read("VERSION")
                    self.device_ready.emit(str(ver))
                except Exception:
                    self.device_ready.emit("unknown")
                break
            else:
                self.device_error.emit("未找到 ReSpeaker (VID:0x2886 PID:0x001A)")
                # 每 2 秒重试一次
                for _ in range(20):
                    if not self._running:
                        return
                    time.sleep(0.1)

        # 主读取循环
        consecutive_errors = 0
        while self._running:
            try:
                result = self._respeaker.read("DOA_VALUE")
                if result and len(result) >= 2:
                    doa_raw        = result[0]         # 0–359  (uint16)
                    speech_detected = bool(result[1])   # VAD flag

                    # 与 xvf_test.py 完全一致的取反
                    doa_angle = -float(doa_raw)

                    self.doa_update.emit(doa_angle, speech_detected)
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1
            except usb.core.USBError as e:
                consecutive_errors += 1
                if consecutive_errors == 1:
                    self.device_error.emit(f"USB 错误: {e}")
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors == 1:
                    self.device_error.emit(f"读取错误: {e}")

            # 设备断开检测
            if consecutive_errors > 30:
                self.device_error.emit("ReSpeaker 连接丢失，尝试重连…")
                self._respeaker = None
                # 重连循环
                while self._running:
                    dev = find_respeaker()
                    if dev is not None:
                        self._respeaker = dev
                        self.device_ready.emit("reconnected")
                        consecutive_errors = 0
                        break
                    time.sleep(1.0)

            time.sleep(self.POLL_INTERVAL)

        # 清理
        if self._respeaker:
            try:
                self._respeaker.close()
            except Exception:
                pass
            self._respeaker = None

    def stop(self):
        self._running = False
        self.wait(2000)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. CalibrationModel — 标定数据模型与线性拟合                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CalibrationModel:
    """存储标定点并支持线性拟合与预测。"""

    def __init__(self):
        self.points: list[tuple[float, float]] = []   # [(xvf_angle, servo_angle), …]
        self.slope: float      = 0.0
        self.intercept: float  = 0.0
        self.r_squared: float  = 0.0
        self.fitted: bool      = False

        # 角度 unwrap 状态（用于跟踪）
        self._prev_raw_xvf: float = 0.0
        self._accumulated_xvf: float = 0.0
        self._unwrap_initialized: bool = False

    # ── 标定点管理 ────────────────────────────────────────────────────────

    def add_point(self, xvf: float, servo: float):
        self.points.append((float(xvf), float(servo)))
        self.fitted = False

    def remove_point(self, index: int):
        if 0 <= index < len(self.points):
            del self.points[index]
            self.fitted = False

    def clear(self):
        self.points.clear()
        self.fitted = False
        self.slope = 0.0
        self.intercept = 0.0
        self.r_squared = 0.0

    # ── 线性拟合 ──────────────────────────────────────────────────────────

    def fit_linear(self) -> tuple[float, float, float] | None:
        """最小二乘拟合: servo = slope * xvf + intercept。返回 (slope, intercept, r²)。"""
        if len(self.points) < 2:
            self.fitted = False
            return None

        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])

        # 预防全等 x 导致奇异矩阵
        if np.std(x) < 1e-9:
            self.slope = 0.0
            self.intercept = float(np.mean(y))
            self.r_squared = 0.0
            self.fitted = True
            return (self.slope, self.intercept, self.r_squared)

        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        y_pred = m * x + c
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0

        self.slope = float(m)
        self.intercept = float(c)
        self.r_squared = float(r2)
        self.fitted = True
        return (self.slope, self.intercept, self.r_squared)

    # ── 预测 ──────────────────────────────────────────────────────────────

    def predict(self, xvf_angle: float) -> float:
        """将 XVF 角度映射为舵机角度。"""
        if self.fitted:
            return self.slope * xvf_angle + self.intercept
        # 未标定时返回正前方
        return 0.0

    def predict_clamped(self, xvf_angle: float, lo=-135.0, hi=135.0) -> float:
        raw = self.predict(xvf_angle)
        return max(lo, min(hi, raw))

    # ── 角度 unwrap ───────────────────────────────────────────────────────

    def unwrap_angle(self, current: float, previous: float) -> float:
        """消除 0°/360° 跳变。"""
        diff = current - previous
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return previous + diff

    def reset_unwrap(self):
        """重置 unwrap 状态。"""
        self._unwrap_initialized = False
        self._accumulated_xvf = 0.0

    def predict_unwrapped(self, xvf_raw: float) -> float:
        """先 unwrap XVF 角度，再映射到舵机角度。"""
        if not self._unwrap_initialized:
            self._prev_raw_xvf = xvf_raw
            self._accumulated_xvf = xvf_raw
            self._unwrap_initialized = True
        else:
            unwrapped = self.unwrap_angle(xvf_raw, self._prev_raw_xvf)
            self._prev_raw_xvf = xvf_raw
            self._accumulated_xvf = unwrapped

        return self.predict_clamped(self._accumulated_xvf)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. CalibrationStorage — JSON + DB 双持久化                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DEFAULT_CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "xvf_calibration.json")

# SystemConfig section/key 常量
CALIB_SECTION = "calibration"
CALIB_KEYS = {
    "points":           "calibration_points",
    "slope":            "slope",
    "intercept":        "intercept",
    "r_squared":        "r_squared",
    "fitted":           "fitted",
    "last_calibrated":  "last_calibrated",
}


class CalibrationStorage:
    """标定数据的 JSON + DB 双持久化。

    读取: DB 优先，JSON 回退
    写入: DB + JSON 双写
    """

    def __init__(self, filepath: str = DEFAULT_CALIB_FILE):
        self.filepath = filepath

    # ── 公开 API ────────────────────────────────────────────────────────────

    def save(self, model: CalibrationModel, last_calibrated: str | None = None) -> bool:
        """将当前标定同时写入 DB 和 JSON 文件。"""
        if last_calibrated is None:
            last_calibrated = datetime.now().isoformat(timespec="seconds")

        ok_json = self._save_json(model, last_calibrated)
        ok_db   = self._save_db(model, last_calibrated)
        return ok_json and ok_db

    def load(self) -> tuple[list[tuple[float, float]], float, float, float, bool, str]:
        """加载标定数据: DB 优先，JSON 回退。

        Returns:
            (points, slope, intercept, r2, fitted, timestamp)
        """
        # 1) 尝试 DB
        db_result = self._load_db()
        if db_result[0] or db_result[5]:
            # 有点或有时戳，认为 DB 中有有效数据
            return db_result

        # 2) 回退到 JSON
        json_result = self._load_json()

        # 3) 如果 JSON 有数据而 DB 为空，自动回写到 DB
        if json_result[0]:
            self._save_json_to_db(*json_result)

        return json_result

    # ── JSON 读写 ────────────────────────────────────────────────────────────

    def _save_json(self, model: CalibrationModel, last_calibrated: str) -> bool:
        data = {
            "calibration_points": [[float(p[0]), float(p[1])] for p in model.points],
            "slope":             model.slope,
            "intercept":         model.intercept,
            "r_squared":         model.r_squared,
            "fitted":            model.fitted,
            "last_calibrated":   last_calibrated,
        }
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[Storage] JSON 保存失败: {e}")
            return False

    def _load_json(self) -> tuple[list, float, float, float, bool, str]:
        if not os.path.exists(self.filepath):
            return ([], 0.0, 0.0, 0.0, False, "")

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            points_raw = data.get("calibration_points", [])
            points = [(float(p[0]), float(p[1])) for p in points_raw]

            return (
                points,
                float(data.get("slope", 0.0)),
                float(data.get("intercept", 0.0)),
                float(data.get("r_squared", 0.0)),
                bool(data.get("fitted", False)),
                str(data.get("last_calibrated", "")),
            )
        except Exception as e:
            print(f"[Storage] JSON 加载失败: {e}")
            return ([], 0.0, 0.0, 0.0, False, "")

    # ── DB 读写 ──────────────────────────────────────────────────────────────

    def _save_db(self, model: CalibrationModel, last_calibrated: str) -> bool:
        """将标定数据写入 SystemConfig 表 (section='calibration')。"""
        try:
            with session_scope() as sess:
                repo = ConfigRepo(sess)

                repo.set(CALIB_SECTION, CALIB_KEYS["points"],
                         [[float(p[0]), float(p[1])] for p in model.points])
                repo.set(CALIB_SECTION, CALIB_KEYS["slope"], model.slope)
                repo.set(CALIB_SECTION, CALIB_KEYS["intercept"], model.intercept)
                repo.set(CALIB_SECTION, CALIB_KEYS["r_squared"], model.r_squared)
                repo.set(CALIB_SECTION, CALIB_KEYS["fitted"], model.fitted)
                repo.set(CALIB_SECTION, CALIB_KEYS["last_calibrated"], last_calibrated)
            return True
        except Exception as e:
            print(f"[Storage] DB 保存失败: {e}")
            return False

    def _load_db(self) -> tuple[list, float, float, float, bool, str]:
        """从 SystemConfig 表读取标定数据。"""
        try:
            with session_scope() as sess:
                repo = ConfigRepo(sess)
                section = repo.get_section(CALIB_SECTION)
                if not section:
                    return ([], 0.0, 0.0, 0.0, False, "")

                points_raw = section.get(CALIB_KEYS["points"], [])
                points = [(float(p[0]), float(p[1])) for p in points_raw]

                return (
                    points,
                    float(section.get(CALIB_KEYS["slope"], 0.0)),
                    float(section.get(CALIB_KEYS["intercept"], 0.0)),
                    float(section.get(CALIB_KEYS["r_squared"], 0.0)),
                    bool(section.get(CALIB_KEYS["fitted"], False)),
                    str(section.get(CALIB_KEYS["last_calibrated"], "")),
                )
        except Exception as e:
            print(f"[Storage] DB 加载失败: {e}")
            return ([], 0.0, 0.0, 0.0, 0.0, False, "")

    def _save_json_to_db(self, points, slope, intercept, r2, fitted, timestamp):
        """将 JSON 中的数据回写到 DB（用于首次迁移）。"""
        try:
            with session_scope() as sess:
                repo = ConfigRepo(sess)
                repo.set(CALIB_SECTION, CALIB_KEYS["points"],
                         [[float(p[0]), float(p[1])] for p in points])
                repo.set(CALIB_SECTION, CALIB_KEYS["slope"], slope)
                repo.set(CALIB_SECTION, CALIB_KEYS["intercept"], intercept)
                repo.set(CALIB_SECTION, CALIB_KEYS["r_squared"], r2)
                repo.set(CALIB_SECTION, CALIB_KEYS["fitted"], fitted)
                repo.set(CALIB_SECTION, CALIB_KEYS["last_calibrated"], timestamp)
            print("[Storage] 已将 JSON 标定数据迁移到 DB")
        except Exception as e:
            print(f"[Storage] JSON→DB 迁移失败: {e}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. Matplotlib 控件 — 标定曲线可视化                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CalibrationPlot(FigureCanvas):
    """嵌入 PyQt5 的 matplotlib 标定绘图。"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._setup_axes()

    def _setup_axes(self):
        self.ax.set_xlabel("XVF DOA Angle (°)")
        self.ax.set_ylabel("Servo Angle (°)")
        self.ax.set_title("Calibration Mapping")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlim(-200, 200)
        self.ax.set_ylim(-150, 150)
        (self.scatter_pts,) = self.ax.plot([], [], "ro", label="Calibration Points")
        (self.fit_line,)   = self.ax.plot([], [], "b-", label="Fitted Line")
        self.ax.legend(loc="upper left")
        self.fig.tight_layout()

    def update_plot(self, model: CalibrationModel):
        """用当前标定数据更新图形。"""
        self.ax.clear()
        self._setup_axes()

        if model.points:
            xs = [p[0] for p in model.points]
            ys = [p[1] for p in model.points]
            self.ax.plot(xs, ys, "ro", label="Calibration Points")

            if model.fitted:
                x_min, x_max = min(xs), max(xs)
                margin = max((x_max - x_min) * 0.1, 10)
                x_plot = np.linspace(x_min - margin, x_max + margin, 200)
                y_plot = model.slope * x_plot + model.intercept
                y_plot = np.clip(y_plot, -135, 135)
                self.ax.plot(x_plot, y_plot, "b-", label="Fitted Line")

                eq = f"Servo = {model.slope:.4f} * XVF + {model.intercept:.2f}"
                r2 = f"R² = {model.r_squared:.4f}"
                self.ax.set_title(f"Calibration Mapping\n{eq}  |  {r2}")

            self.ax.set_xlim(min(xs) - 20, max(xs) + 20)
            self.ax.legend(loc="upper left")

        self.fig.tight_layout()
        self.draw()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. CalibrationTab — 标定界面                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CalibrationTab(QWidget):
    """标定 Tab：实时 XVF 显示、舵机手动控制、标定点管理、拟合曲线。"""

    # 信号：通知 TrackingTab 标定模型已更新
    model_updated  = pyqtSignal()
    servo_set_requested = pyqtSignal(float)   # 请求设置舵机角度（也更新 TrackingTab 的显示）

    def __init__(self, model: CalibrationModel, pwm: PWMController, parent=None):
        super().__init__(parent)
        self.model = model
        self.pwm   = pwm

        self._latest_xvf_angle   = 0.0
        self._latest_speech      = False
        self._captured_xvf       = None   # 冻结的 XVF 角度（用于标定）

        self._build_ui()

    # ── UI 构建 ───────────────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # ── 左侧：控制面板 ───────────────────────────────────────────────
        left_panel = QVBoxLayout()

        # XVF 实时状态
        xvf_group = QGroupBox("XVF 实时状态")
        xvf_layout = QVBoxLayout(xvf_group)
        self.lbl_xvf_angle = QLabel("DOA: --°")
        self.lbl_xvf_angle.setFont(QFont("", 18, QFont.Bold))
        self.lbl_speech = QLabel("语音: --")
        self.lbl_speech.setFont(QFont("", 12))
        self.lbl_speech.setStyleSheet("color: gray;")
        self.btn_capture_xvf = QPushButton("📷 捕获当前 XVF 角度")
        self.btn_capture_xvf.clicked.connect(self._on_capture_xvf)
        self.btn_capture_xvf.setEnabled(False)

        xvf_layout.addWidget(self.lbl_xvf_angle)
        xvf_layout.addWidget(self.lbl_speech)

        # 捕获值显示
        self.lbl_captured_xvf = QLabel("已捕获: --")
        self.lbl_captured_xvf.setFont(QFont("monospace", 11))
        self.lbl_captured_xvf.setStyleSheet("color: orange;")
        xvf_layout.addWidget(self.lbl_captured_xvf)
        xvf_layout.addWidget(self.btn_capture_xvf)
        left_panel.addWidget(xvf_group)

        # 手动舵机控制
        servo_group = QGroupBox("舵机手动控制")
        servo_layout = QVBoxLayout(servo_group)

        slider_row = QHBoxLayout()
        self.servo_slider = QSlider(Qt.Horizontal)
        self.servo_slider.setRange(-135, 135)
        self.servo_slider.setValue(0)
        self.servo_slider.setTickPosition(QSlider.TicksBelow)
        self.servo_slider.setTickInterval(15)
        slider_row.addWidget(QLabel("-135°"))
        slider_row.addWidget(self.servo_slider)
        slider_row.addWidget(QLabel("135°"))

        spin_row = QHBoxLayout()
        self.servo_spin = QDoubleSpinBox()
        self.servo_spin.setRange(-135, 135)
        self.servo_spin.setValue(0)
        self.servo_spin.setDecimals(1)
        self.servo_spin.setSuffix("°")
        self.servo_spin.setSingleStep(5.0)
        spin_row.addWidget(QLabel("角度:"))
        spin_row.addWidget(self.servo_spin)

        self.btn_set_servo = QPushButton("⚡ 设置舵机")
        self.btn_set_servo.clicked.connect(self._on_set_servo)

        # 滑块 ↔ 输入框 双向绑定
        self.servo_slider.valueChanged.connect(
            lambda v: self.servo_spin.setValue(float(v)))
        self.servo_slider.valueChanged.connect(
            lambda v: self._update_duty_preview(float(v)))
        self.servo_spin.valueChanged.connect(
            lambda v: self.servo_slider.blockSignals(True) or
                       self.servo_slider.setValue(int(v)) or
                       self.servo_slider.blockSignals(False))
        self.servo_spin.valueChanged.connect(
            lambda v: self._update_duty_preview(v))

        servo_layout.addLayout(slider_row)
        servo_layout.addLayout(spin_row)
        servo_layout.addWidget(self.btn_set_servo)

        # 舵机预设按钮
        preset_row = QHBoxLayout()
        for name, ang in [("左端 -135°", -135), ("中间 0°", 0), ("右端 +135°", 135)]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, a=ang: self._preset_servo(a))
            preset_row.addWidget(btn)
        servo_layout.addLayout(preset_row)

        # 当前占空比显示
        self.lbl_duty = QLabel("占空比: -- ns")
        servo_layout.addWidget(self.lbl_duty)

        if not self.pwm.initialized:
            self.lbl_duty.setText("⚠ PWM 未初始化（需 root）")
            self.lbl_duty.setStyleSheet("color: red;")

        left_panel.addWidget(servo_group)

        # 标定点操作
        pt_group = QGroupBox("标定点操作")
        pt_layout = QVBoxLayout(pt_group)
        self.btn_add_point = QPushButton("➕ 添加标定点 (当前 XVF + 当前舵机)")
        self.btn_add_point.clicked.connect(self._on_add_point)
        self.btn_add_point.setEnabled(False)
        pt_layout.addWidget(self.btn_add_point)
        left_panel.addWidget(pt_group)

        left_panel.addStretch()

        # ── 中间：标定点表格 ─────────────────────────────────────────────
        center_panel = QVBoxLayout()

        table_group = QGroupBox("标定点列表")
        table_layout = QVBoxLayout(table_group)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["XVF 角度 (°)", "舵机角度 (°)", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_fit = QPushButton("📈 拟合标定")
        self.btn_fit.clicked.connect(self._on_fit)
        self.btn_clear = QPushButton("🗑 清空全部")
        self.btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self.btn_fit)
        btn_row.addWidget(self.btn_clear)
        table_layout.addLayout(btn_row)

        center_panel.addWidget(table_group)
        center_panel.addStretch()

        # ── 右侧：拟合曲线 ───────────────────────────────────────────────
        right_panel = QVBoxLayout()
        self.plot = CalibrationPlot(self)
        right_panel.addWidget(self.plot)

        # 拟合参数标签
        self.lbl_fit_info = QLabel("尚未拟合")
        self.lbl_fit_info.setFont(QFont("", 10))
        self.lbl_fit_info.setWordWrap(True)
        right_panel.addWidget(self.lbl_fit_info)

        # ── 组装 ─────────────────────────────────────────────────────────
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(center_panel, 3)
        main_layout.addLayout(right_panel, 5)

    # ── 从 Reader 线程接收数据 ────────────────────────────────────────────

    def on_doa_update(self, doa_angle: float, speech: bool):
        self._latest_xvf_angle = doa_angle
        self._latest_speech    = speech

        self.lbl_xvf_angle.setText(f"DOA: {doa_angle:.1f}°")
        if speech:
            self.lbl_speech.setText("语音: 检测到 🔊")
            self.lbl_speech.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.lbl_speech.setText("语音: 无声 🔇")
            self.lbl_speech.setStyleSheet("color: gray;")

        # 设备就绪后启用按钮
        self.btn_capture_xvf.setEnabled(True)
        self.btn_add_point.setEnabled(True)

    def on_device_error(self, msg: str):
        self.lbl_xvf_angle.setText("DOA: --°")
        self.lbl_speech.setText(f"⚠ {msg}")
        self.lbl_speech.setStyleSheet("color: red;")
        self.btn_capture_xvf.setEnabled(False)

    def on_device_ready(self, version: str):
        self.lbl_speech.setText(f"设备就绪 (v{version})")
        self.lbl_speech.setStyleSheet("color: blue;")

    # ── 按钮回调 ─────────────────────────────────────────────────────────

    def _on_capture_xvf(self):
        """冻结当前 XVF DOA 角度（用于标定配对）。"""
        self._captured_xvf = self._latest_xvf_angle
        self.lbl_captured_xvf.setText(
            f"已捕获: {self._captured_xvf:.1f}°"
        )
        self.lbl_captured_xvf.setStyleSheet("color: orange; font-weight: bold;")

    def _preset_servo(self, angle: float):
        self.servo_spin.setValue(angle)
        self._on_set_servo()

    def _on_set_servo(self):
        angle = self.servo_spin.value()
        self.pwm.set_angle(angle)
        duty = self.pwm.angle_to_duty(angle)
        duty_pct = duty / self.pwm.PERIOD_NS * 100
        self.lbl_duty.setText(f"占空比: {duty} ns ({duty_pct:.1f}%)")
        self.servo_set_requested.emit(angle)

    def _update_duty_preview(self, angle: float):
        """实时更新占空比预览（不发送 PWM 指令）。"""
        if self.pwm.initialized:
            duty = self.pwm.angle_to_duty(angle)
            duty_pct = duty / self.pwm.PERIOD_NS * 100
            self.lbl_duty.setText(f"占空比预览: {duty} ns ({duty_pct:.1f}%)")

    def _on_add_point(self):
        """将捕获的 XVF 角度和当前舵机角度配对添加到标定点列表。"""
        if self._captured_xvf is None:
            QMessageBox.warning(self, "未捕获", "请先点击「捕获当前 XVF 角度」冻结 DOA 读数。")
            return

        xvf_angle   = self._captured_xvf
        servo_angle = self.servo_spin.value()

        self.model.add_point(xvf_angle, servo_angle)
        self._refresh_table()
        self.plot.update_plot(self.model)

        # 清除捕获状态
        self._captured_xvf = None
        self.lbl_captured_xvf.setText("已捕获: --")
        self.lbl_captured_xvf.setStyleSheet("color: orange;")

        self.model_updated.emit()

    def _on_fit(self):
        if len(self.model.points) < 2:
            QMessageBox.warning(self, "拟合失败", "至少需要 2 个标定点才能拟合。")
            return
        result = self.model.fit_linear()
        if result:
            m, c, r2 = result
            self.lbl_fit_info.setText(
                f"斜率 (slope): {m:.6f}\n"
                f"截距 (intercept): {c:.4f}°\n"
                f"R²: {r2:.6f}\n"
                f"点数: {len(self.model.points)}"
            )
            self.plot.update_plot(self.model)
            self.model_updated.emit()

    def _on_clear(self):
        reply = QMessageBox.question(self, "确认", "确定要清空所有标定点吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.model.clear()
            self._refresh_table()
            self.plot.update_plot(self.model)
            self.lbl_fit_info.setText("尚未拟合")
            self.model_updated.emit()

    def _refresh_table(self):
        """用 model.points 刷新表格。"""
        self.table.setRowCount(0)
        for i, (xvf, servo) in enumerate(self.model.points):
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(f"{xvf:.2f}"))
            self.table.setItem(i, 1, QTableWidgetItem(f"{servo:.2f}"))

            btn_del = QPushButton("✕")
            btn_del.setFixedWidth(30)
            btn_del.clicked.connect(lambda checked, idx=i: self._delete_point(idx))
            self.table.setCellWidget(i, 2, btn_del)

    def _delete_point(self, index: int):
        self.model.remove_point(index)
        self._refresh_table()
        self.plot.update_plot(self.model)
        self.model_updated.emit()

    def refresh_from_model(self):
        """外部调用：用 model 数据刷新 UI。"""
        self._refresh_table()
        self.plot.update_plot(self.model)
        if self.model.fitted:
            self.lbl_fit_info.setText(
                f"斜率 (slope): {self.model.slope:.6f}\n"
                f"截距 (intercept): {self.model.intercept:.4f}°\n"
                f"R²: {self.model.r_squared:.6f}\n"
                f"点数: {len(self.model.points)}"
            )
        else:
            self.lbl_fit_info.setText("尚未拟合（已加载数据点）" if self.model.points else "尚未拟合")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. TrackingTab — 实时声源跟踪测试                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class TrackingTab(QWidget):
    """跟踪测试 Tab：实时 DOA → 校准 → 舵机跟踪。

    状态机（三层串行）：
      IDLE ──(语音稳定 + 跳变检测)──→ AWAITING ──(稳定确认/超时)──→ COOLDOWN ──(冷却到期)──→ IDLE

    设计目的：
      - XVF 估计器在声源跳变时会经历一段 ramp（0→90° 渐变），
        如果在 ramp 期间移动舵机会导致多次抖动到错误位置。
      - AWAITING 状态等待 ramp 结束、角度稳定后，一次精准移动到目标。
      - 移动后进入 COOLDOWN 隔离电机噪音。
    """

    # ── 可调参数 (GUI) ──────────────────────────────────────────────────────
    THRESHOLD         = 10.0     # 角度跳变阈值 (°)，超过此值判定为突变
    AWAIT_DURATION    = 0.5      # 稳定倒计时时长 (s)
    CONVERGED_THRESH  = 3.0      # 连续帧 delta 低于此值认为已收敛 (可提前触发)
    MAX_AWAIT         = 2.0      # 最大等待超时 (s)，防永远不触发
    MOTOR_COOLDOWN    = 1.5      # 舵机移动后忽略音频的冷却时间 (s)
    SPEECH_FRAMES     = 3        # 连续语音帧数才认为"有声音"

    # ── 内部状态枚举 ────────────────────────────────────────────────────────
    _STATE_IDLE      = 0
    _STATE_AWAITING  = 1
    _STATE_COOLDOWN  = 2
    _STATE_NAMES     = {0: "IDLE", 1: "AWAITING", 2: "COOLDOWN"}

    def __init__(self, model: CalibrationModel, pwm: PWMController, parent=None):
        super().__init__(parent)
        self.model = model
        self.pwm   = pwm

        self._tracking_active    = False
        self._latest_doa         = 0.0
        self._latest_speech      = False

        # 语音累积
        self._speech_count       = 0

        # 状态机变量
        self._state              = self._STATE_IDLE
        self._state_enter_time   = 0.0
        self._frozen_doa         = 0.0       # 进入 AWAITING 时的参考角度
        self._await_countdown    = 0.0       # AWAITING 剩余倒计时
        self._recent_deltas      = deque(maxlen=5)  # 最近几帧的 raw delta

        self._build_ui()

        # 定时器：100 ms 运行一次跟踪循环
        self._track_timer = QTimer(self)
        self._track_timer.timeout.connect(self._tracking_tick)
        self._track_timer.setInterval(100)

        # 显示刷新定时器
        self._display_timer = QTimer(self)
        self._display_timer.timeout.connect(self._refresh_display)
        self._display_timer.setInterval(100)
        self._display_timer.start()

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ── 状态面板 ─────────────────────────────────────────────────────
        status_grid = QGridLayout()
        status_grid.setColumnStretch(1, 1)

        self.lbl_doa = QLabel("XVF DOA: --°")
        self.lbl_doa.setFont(QFont("", 16, QFont.Bold))
        status_grid.addWidget(QLabel("原始 XVF:"), 0, 0)
        status_grid.addWidget(self.lbl_doa, 0, 1)

        self.lbl_speech = QLabel("语音: --")
        self.lbl_speech.setFont(QFont("", 12))
        status_grid.addWidget(QLabel("语音检测:"), 1, 0)
        status_grid.addWidget(self.lbl_speech, 1, 1)

        self.lbl_predicted = QLabel("预测舵机: --°")
        self.lbl_predicted.setFont(QFont("", 16, QFont.Bold))
        self.lbl_predicted.setStyleSheet("color: blue;")
        status_grid.addWidget(QLabel("预测舵机:"), 2, 0)
        status_grid.addWidget(self.lbl_predicted, 2, 1)

        self.lbl_current_servo = QLabel("当前舵机: --°")
        self.lbl_current_servo.setFont(QFont("", 12))
        status_grid.addWidget(QLabel("实际舵机:"), 3, 0)
        status_grid.addWidget(self.lbl_current_servo, 3, 1)

        # 状态 + 冷却合一行
        row = QHBoxLayout()
        self.lbl_tracking_status = QLabel("⏸ 停止")
        self.lbl_tracking_status.setFont(QFont("", 14, QFont.Bold))
        self.lbl_tracking_status.setStyleSheet("color: gray;")
        row.addWidget(QLabel("状态:"))
        row.addWidget(self.lbl_tracking_status)
        row.addStretch()
        self.lbl_cooldown = QLabel("")
        self.lbl_cooldown.setFont(QFont("", 11))
        row.addWidget(self.lbl_cooldown)
        status_grid.addWidget(QLabel(""), 4, 0)
        status_grid.addLayout(row, 4, 1)

        layout.addLayout(status_grid)

        # ── 控制面板 ─────────────────────────────────────────────────────
        ctrl_layout = QHBoxLayout()

        self.btn_toggle = QPushButton("▶ 开始跟踪")
        self.btn_toggle.setFont(QFont("", 12, QFont.Bold))
        self.btn_toggle.clicked.connect(self._on_toggle_tracking)
        ctrl_layout.addWidget(self.btn_toggle)

        ctrl_layout.addStretch()

        # 参数调节区 (第二行)
        param_row = QHBoxLayout()

        param_row.addWidget(QLabel("跳变阈值(°):"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(3.0, 60.0)
        self.spin_threshold.setValue(self.THRESHOLD)
        self.spin_threshold.setDecimals(1)
        self.spin_threshold.setSingleStep(5.0)
        self.spin_threshold.valueChanged.connect(self._on_threshold_changed)
        param_row.addWidget(self.spin_threshold)

        param_row.addWidget(QLabel("稳定计时(s):"))
        self.spin_await = QDoubleSpinBox()
        self.spin_await.setRange(0.2, 2.0)
        self.spin_await.setValue(self.AWAIT_DURATION)
        self.spin_await.setDecimals(1)
        self.spin_await.setSingleStep(0.1)
        self.spin_await.setSuffix("s")
        self.spin_await.valueChanged.connect(self._on_await_changed)
        param_row.addWidget(self.spin_await)

        param_row.addWidget(QLabel("收敛阈值(°):"))
        self.spin_converged = QDoubleSpinBox()
        self.spin_converged.setRange(1.0, 15.0)
        self.spin_converged.setValue(self.CONVERGED_THRESH)
        self.spin_converged.setDecimals(1)
        self.spin_converged.setSingleStep(1.0)
        self.spin_converged.valueChanged.connect(self._on_converged_changed)
        param_row.addWidget(self.spin_converged)

        param_row.addWidget(QLabel("最大等待(s):"))
        self.spin_max_await = QDoubleSpinBox()
        self.spin_max_await.setRange(0.5, 5.0)
        self.spin_max_await.setValue(self.MAX_AWAIT)
        self.spin_max_await.setDecimals(1)
        self.spin_max_await.setSingleStep(0.2)
        self.spin_max_await.setSuffix("s")
        self.spin_max_await.valueChanged.connect(self._on_max_await_changed)
        param_row.addWidget(self.spin_max_await)

        param_row.addWidget(QLabel("冷却(s):"))
        self.spin_cooldown = QDoubleSpinBox()
        self.spin_cooldown.setRange(0.3, 5.0)
        self.spin_cooldown.setValue(self.MOTOR_COOLDOWN)
        self.spin_cooldown.setDecimals(1)
        self.spin_cooldown.setSuffix("s")
        self.spin_cooldown.setSingleStep(0.1)
        self.spin_cooldown.valueChanged.connect(self._on_cooldown_changed)
        param_row.addWidget(self.spin_cooldown)

        param_row.addStretch()

        self.btn_reset_unwrap = QPushButton("🔄 重置")
        self.btn_reset_unwrap.clicked.connect(self._on_reset_unwrap)
        param_row.addWidget(self.btn_reset_unwrap)

        layout.addLayout(ctrl_layout)
        layout.addLayout(param_row)

        # ── 日志 ─────────────────────────────────────────────────────────
        log_group = QGroupBox("跟踪日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.document().setMaximumBlockCount(200)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

    # ── 数据输入 ──────────────────────────────────────────────────────────

    def on_doa_update(self, doa_angle: float, speech: bool):
        self._latest_doa    = doa_angle
        self._latest_speech = speech

        # 累积语音帧数 (用于 IDLE 状态判断)
        if speech:
            self._speech_count = min(self._speech_count + 1, self.SPEECH_FRAMES + 5)
        else:
            self._speech_count = max(self._speech_count - 1, 0)

    def on_servo_set(self, angle: float):
        """当舵机被 CalibrationTab 手动设置时更新。"""
        self.lbl_current_servo.setText(f"{angle:.1f}°")

    # ── 跟踪启停 ──────────────────────────────────────────────────────────

    def _on_toggle_tracking(self):
        if self._tracking_active:
            self._stop_tracking()
        else:
            self._start_tracking()

    def _start_tracking(self):
        if not self.model.fitted and len(self.model.points) < 2:
            QMessageBox.warning(self, "无法跟踪", "请先在「标定」页完成拟合。")
            return
        if not self.pwm.initialized:
            QMessageBox.warning(self, "无法跟踪", "PWM 未初始化（需要 root 权限）。")
            return

        self._tracking_active = True
        self.model.reset_unwrap()
        self._clear_audio_state()
        self._enter_state(self._STATE_IDLE)

        self.lbl_cooldown.setText("")

        self.btn_toggle.setText("⏹ 停止跟踪")
        self._track_timer.start()
        self._log(f"跟踪已启动 (阈值={self.THRESHOLD}° 稳定={self.AWAIT_DURATION}s 冷却={self.MOTOR_COOLDOWN}s)")

    def _stop_tracking(self):
        self._tracking_active = False
        self._track_timer.stop()
        self.lbl_cooldown.setText("")
        self.lbl_tracking_status.setText("⏸ 停止")
        self.lbl_tracking_status.setStyleSheet("color: gray;")
        self.btn_toggle.setText("▶ 开始跟踪")
        self._log("跟踪已停止")

    # ── 状态机核心 ────────────────────────────────────────────────────────

    def _enter_state(self, new_state: int):
        old = self._state
        self._state = new_state
        self._state_enter_time = time.time()

        if new_state == self._STATE_IDLE:
            self.lbl_tracking_status.setText("👂 侦听中")
            self.lbl_tracking_status.setStyleSheet("color: green; font-weight: bold;")
        elif new_state == self._STATE_AWAITING:
            self._await_countdown = self.AWAIT_DURATION
            self.lbl_tracking_status.setText("⏳ 等待稳定…")
            self.lbl_tracking_status.setStyleSheet("color: orange; font-weight: bold;")
        elif new_state == self._STATE_COOLDOWN:
            self.lbl_tracking_status.setText("🧊 冷却中")
            self.lbl_tracking_status.setStyleSheet("color: #cc6600; font-weight: bold;")

        if old != new_state:
            self._log(f"状态: {self._STATE_NAMES[old]} → {self._STATE_NAMES[new_state]}")

    def _clear_audio_state(self):
        """重置所有音频相关状态。"""
        self._speech_count = 0
        self._recent_deltas.clear()
        self._frozen_doa = 0.0
        self._await_countdown = 0.0

    def _tracking_tick(self):
        """状态机主循环，每 100ms 执行一次。"""
        if not self._tracking_active:
            return

        now = time.time()
        raw_doa = self._latest_doa
        speech  = self._latest_speech

        # ═════════════════════════════════════════════════════════════════════
        # STATE: COOLDOWN — 电机噪音隔离
        # ═════════════════════════════════════════════════════════════════════
        if self._state == self._STATE_COOLDOWN:
            elapsed = now - self._state_enter_time
            if elapsed >= self.MOTOR_COOLDOWN:
                self._clear_audio_state()
                self._enter_state(self._STATE_IDLE)
                self.lbl_cooldown.setText("")
                return
            else:
                remaining = self.MOTOR_COOLDOWN - elapsed
                self.lbl_cooldown.setText(f"⏳ {remaining:.1f}s …")
                self.lbl_cooldown.setStyleSheet("color: orange; font-weight: bold;")
                return  # 冷却期内一切都忽略

        # ═════════════════════════════════════════════════════════════════════
        # STATE: IDLE — 等待声音 + 角度跳变
        # ═════════════════════════════════════════════════════════════════════
        if self._state == self._STATE_IDLE:
            # 需要连续 SPEECH_FRAMES 帧语音
            if self._speech_count < self.SPEECH_FRAMES:
                self.lbl_cooldown.setText("🔇 等待语音…")
                self.lbl_cooldown.setStyleSheet("color: gray;")
                return

            # 语音稳定后检测角度跳变：与当前舵机位置差超过阈值
            delta = abs(raw_doa - self.pwm.get_angle())
            # 同时使用 XVF 原始值与自身变化做辅助判断
            if delta > self.THRESHOLD:
                self._frozen_doa = raw_doa
                self._recent_deltas.clear()
                self._enter_state(self._STATE_AWAITING)
                self.lbl_cooldown.setText(f"🔔 检测到跳变 {delta:.0f}°")
                self.lbl_cooldown.setStyleSheet("color: orange;")
                self._log(f"跳变检测: delta={delta:.1f}°, 冻结参考={raw_doa:.1f}°")
                return
            else:
                self.lbl_cooldown.setText("👂")
                self.lbl_cooldown.setStyleSheet("color: green;")
                return

        # ═════════════════════════════════════════════════════════════════════
        # STATE: AWAITING — 等待 XVF 估计器收敛到稳定值
        # ═════════════════════════════════════════════════════════════════════
        if self._state == self._STATE_AWAITING:
            state_elapsed = now - self._state_enter_time

            # 计算当前 raw 与冻结参考的差值
            current_delta = abs(raw_doa - self._frozen_doa)
            self._recent_deltas.append(current_delta)

            # ══ 规则 1: 有新突变 → 刷新倒计时，更新冻结参考 ═══════════════
            if current_delta > self.THRESHOLD:
                self._frozen_doa = raw_doa
                self._await_countdown = self.AWAIT_DURATION
                self._recent_deltas.clear()
                self.lbl_cooldown.setText(f"🔄 刷新计时 (delta={current_delta:.0f}°)")
                self.lbl_cooldown.setStyleSheet("color: orange;")
                self._log(f"  AWAIT 刷新: 新突变 delta={current_delta:.1f}°, 重置倒计时 {self.AWAIT_DURATION}s")
                self.lbl_predicted.setText(f"--° (刷新中)")
                return

            # ══ 倒计时期间：持续跟踪最新 raw 值 ════════════════════════════
            # delta 在阈值以内 → 更新冻结参考到最新的稳定值
            self._frozen_doa = raw_doa
            self._await_countdown -= 0.1  # timer interval

            # ══ 规则 2: 已收敛 → 立即触发移动（提前于倒计时） ═════════════
            if (len(self._recent_deltas) >= 3 and
                all(d < self.CONVERGED_THRESH for d in list(self._recent_deltas)[-3:])):
                self._trigger_move(self._frozen_doa, reason="收敛")
                return

            # ══ 规则 3: 倒计时归零 → 稳定触发 ═════════════════════════════
            if self._await_countdown <= 0:
                self._trigger_move(self._frozen_doa, reason="倒计时到期")
                return

            # ══ 规则 4: 超时保护 → 强制移动 ══════════════════════════════
            if state_elapsed >= self.MAX_AWAIT:
                self._trigger_move(self._frozen_doa, reason="超时强制")
                return

            # 仍在等待
            self.lbl_cooldown.setText(f"⏳ 稳定中 {self._await_countdown:.1f}s | delta={current_delta:.1f}°")
            self.lbl_cooldown.setStyleSheet("color: orange;")
            self.lbl_predicted.setText(f"--° (等待)")
            return

    def _trigger_move(self, target_doa: float, reason: str = ""):
        """执行一次舵机移动，然后进入 COOLDOWN 状态。"""
        # 用 XVF 角度预测舵机角度（一次到位）
        servo_target = self.model.predict_clamped(target_doa)
        self.pwm.set_angle(servo_target)
        self._enter_state(self._STATE_COOLDOWN)

        current = self.pwm.get_angle()
        self._log(
            f"移动: XVF={target_doa:.1f}° → 舵机={servo_target:.1f}° "
            f"(移动={abs(servo_target - current):.1f}°) "
            f"原因={reason} → 进入 {self.MOTOR_COOLDOWN}s 冷却"
        )

        self.lbl_predicted.setText(f"{servo_target:.1f}°")
        self.lbl_current_servo.setText(f"{servo_target:.1f}°")

    # ── 显示刷新 ──────────────────────────────────────────────────────────

    def _refresh_display(self):
        """高频刷新 XVF 原始值显示。"""
        self.lbl_doa.setText(f"{self._latest_doa:.1f}°")
        if self._latest_speech:
            self.lbl_speech.setText(f"语音: 检测到 🔊 (×{self._speech_count})")
            self.lbl_speech.setStyleSheet("color: green;")
        else:
            self.lbl_speech.setText("语音: 无声 🔇")
            self.lbl_speech.setStyleSheet("color: gray;")

        # 非 AWAITING 状态下更新舵机实际位置
        if self._state != self._STATE_AWAITING and self._tracking_active:
            self.lbl_current_servo.setText(f"{self.pwm.get_angle():.1f}°")

    # ── 参数回调 ──────────────────────────────────────────────────────────

    def _on_threshold_changed(self, val: float):
        self.THRESHOLD = val
        self._log(f"跳变阈值已设为 {val:.1f}°")

    def _on_await_changed(self, val: float):
        self.AWAIT_DURATION = val
        self._log(f"稳定计时已设为 {val:.1f}s")

    def _on_converged_changed(self, val: float):
        self.CONVERGED_THRESH = val
        self._log(f"收敛阈值已设为 {val:.1f}°")

    def _on_max_await_changed(self, val: float):
        self.MAX_AWAIT = val
        self._log(f"最大等待已设为 {val:.1f}s")

    def _on_cooldown_changed(self, val: float):
        self.MOTOR_COOLDOWN = val
        self._log(f"冷却时间已设为 {val:.1f}s")

    def _on_reset_unwrap(self):
        self.model.reset_unwrap()
        self._clear_audio_state()
        if self._state == self._STATE_AWAITING:
            self._enter_state(self._STATE_IDLE)
        self._log("状态已重置")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {msg}")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  9. MainWindow — 主窗口                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class MainWindow(QMainWindow):
    """主窗口：菜单栏、状态栏、Tab 容器。"""

    def __init__(self):
        super().__init__()

        # ── 核心组件创建 ──────────────────────────────────────────────────
        self.model   = CalibrationModel()
        self.storage = CalibrationStorage()
        self.pwm     = PWMController()

        # 初始化 DB (确保表存在)
        try:
            init_db()
            print(f"[DB] 数据库已就绪 → {db_path()}")
        except Exception as e:
            print(f"[DB] 初始化失败 (标定仍可保存为 JSON): {e}")

        # 初始化 PWM
        pwm_ok = self.pwm.init()
        if not pwm_ok:
            print("[WARN] PWM 初始化失败，舵机控制不可用（需要 sudo）")

        # ── 加载已有标定 ──────────────────────────────────────────────────
        points, slope, intercept, r2, fitted, timestamp = self.storage.load()
        if points:
            self.model.points = points
        if fitted:
            self.model.slope     = slope
            self.model.intercept = intercept
            self.model.r_squared = r2
            self.model.fitted    = fitted
            # 如果没有保存拟合值但有足够点，自动拟合
            if not fitted and len(points) >= 2:
                self.model.fit_linear()

        self._last_calibrated = timestamp

        # ── ReSpeaker 后台线程 ────────────────────────────────────────────
        self.reader = ReSpeakerReader(self)

        # ── UI ────────────────────────────────────────────────────────────
        self.setWindowTitle("XVF-Servo 标定与跟踪")
        self.resize(1200, 750)

        self._build_menu()
        self._build_statusbar()

        # Tab
        self.tabs = QTabWidget()

        self.calib_tab = CalibrationTab(self.model, self.pwm)
        self.track_tab = TrackingTab(self.model, self.pwm)

        self.tabs.addTab(self.calib_tab, "📐 标定")
        self.tabs.addTab(self.track_tab, "🎯 实时跟踪")

        self.setCentralWidget(self.tabs)

        # ── 信号连接 ──────────────────────────────────────────────────────
        # ReSpeaker → 两个 Tab
        self.reader.doa_update.connect(self._on_doa_update)
        self.reader.device_error.connect(self._on_device_error)
        self.reader.device_ready.connect(self._on_device_ready)

        # CalibTab → 外部
        self.calib_tab.model_updated.connect(self._on_model_updated)
        self.calib_tab.servo_set_requested.connect(self.track_tab.on_servo_set)

        # ── 启动 ──────────────────────────────────────────────────────────
        self.reader.start()

        self._update_statusbar()

    def closeEvent(self, event):
        """关闭窗口时检查未保存的标定点。"""
        # 检查是否需要保存
        if self.model.points:
            saved_points, _, _, _, _, saved_ts = self.storage.load()
            # 简单判断：点数和之前不同 或 从未保存过
            if not saved_ts or len(self.model.points) != len(saved_points):
                reply = QMessageBox.question(
                    self, "保存标定？",
                    "当前标定点尚未保存，是否保存后退出？\n"
                    "(将同时保存到 DB 和 JSON 文件)",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                if reply == QMessageBox.Yes:
                    self._save_calibration()
                elif reply == QMessageBox.Cancel:
                    event.ignore()
                    return

        # 清理
        self.reader.stop()
        self.pwm.cleanup()
        event.accept()

    # ── 菜单栏 ────────────────────────────────────────────────────────────

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("文件(&F)")

        act_save = QAction("💾 保存标定", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_calibration)
        file_menu.addAction(act_save)

        act_load = QAction("📂 加载标定", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self._load_calibration)
        file_menu.addAction(act_load)

        file_menu.addSeparator()

        act_exit = QAction("退出", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

    # ── 状态栏 ────────────────────────────────────────────────────────────

    def _build_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self._lbl_xvf_status = QLabel("XVF: 连接中…")
        self._lbl_pwm_status = QLabel("PWM: " + ("就绪" if self.pwm.initialized else "无权限"))
        self._lbl_calib_status = QLabel("标定: --")
        self.statusbar.addPermanentWidget(self._lbl_xvf_status)
        self.statusbar.addPermanentWidget(self._lbl_pwm_status)
        self.statusbar.addPermanentWidget(self._lbl_calib_status)

    def _update_statusbar(self):
        self._lbl_pwm_status.setText("PWM: " + ("就绪 ✅" if self.pwm.initialized else "无权限 ❌"))
        if self._last_calibrated:
            self._lbl_calib_status.setText(f"标定: {self._last_calibrated}")
        else:
            self._lbl_calib_status.setText("标定: 无")

    # ── 槽：Reader 信号 ───────────────────────────────────────────────────

    def _on_doa_update(self, doa_angle: float, speech: bool):
        self.calib_tab.on_doa_update(doa_angle, speech)
        self.track_tab.on_doa_update(doa_angle, speech)

    def _on_device_error(self, msg: str):
        self._lbl_xvf_status.setText(f"XVF: {msg[:40]}")
        self.calib_tab.on_device_error(msg)

    def _on_device_ready(self, version: str):
        self._lbl_xvf_status.setText(f"XVF: 就绪 (v{version})")
        self.calib_tab.on_device_ready(version)

    def _on_model_updated(self):
        self._last_calibrated = datetime.now().isoformat(timespec="seconds")
        self._update_statusbar()

    # ── 保存 / 加载 ───────────────────────────────────────────────────────

    def _save_calibration(self):
        ts = datetime.now().isoformat(timespec="seconds")
        ok = self.storage.save(self.model, ts)
        if ok:
            self._last_calibrated = ts
            self._update_statusbar()
            self.statusbar.showMessage(
                f"标定已保存 → DB + {self.storage.filepath}", 3000)
        else:
            QMessageBox.critical(self, "保存失败",
                                 f"无法写入 DB 或 JSON:\n{self.storage.filepath}")

    def _load_calibration(self):
        points, slope, intercept, r2, fitted, timestamp = self.storage.load()
        if not points:
            QMessageBox.information(self, "加载",
                                    "未找到已保存的标定数据（DB 和 JSON 均为空）。")
            return

        self.model.points     = points
        self.model.slope      = slope
        self.model.intercept  = intercept
        self.model.r_squared  = r2
        self.model.fitted     = fitted
        self._last_calibrated = timestamp

        if not fitted and len(points) >= 2:
            self.model.fit_linear()

        self.calib_tab.refresh_from_model()
        self._update_statusbar()
        self.statusbar.showMessage(f"已加载 {len(points)} 个标定点 (DB/JSON)", 3000)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  10. main()                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main():
    # 检查 root 权限（仅警告）
    if os.geteuid() != 0:
        print("⚠️  警告: 未以 root 运行。PWM 舵机控制将不可用。")
        print("    请使用: sudo python3 calibrate_gui.py")
        print()

    # 高 DPI 支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 全局字体
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
