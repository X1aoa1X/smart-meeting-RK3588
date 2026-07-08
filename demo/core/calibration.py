"""XVF→舵机 标定模型 + JSON 持久化。

用法:
  # 加载标定
  storage = CalibrationStorage("xvf_calibration.json")
  points, slope, intercept, r2, fitted, ts = storage.load()

  model = CalibrationModel()
  model.points = points
  if fitted:
      model.slope, model.intercept, model.r_squared = slope, intercept, r2
      model.fitted = True
  elif len(points) >= 2:
      model.fit_linear()

  # 预测
  servo_angle = model.predict_clamped(xvf_angle, lo=-135, hi=135)
"""

import os
import math
import json
import numpy as np

# 默认标定文件路径（相对于项目根目录）
DEFAULT_CALIB_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "xvf_calibration.json")


class CalibrationModel:
    """存储 XVF→舵机 标定点，支持线性拟合与预测。"""

    def __init__(self):
        self.points: list[tuple[float, float]] = []
        self.slope: float     = 0.0
        self.intercept: float = 0.0
        self.r_squared: float = 0.0
        self.fitted: bool     = False

        self._prev_raw_xvf: float = 0.0
        self._accumulated_xvf: float = 0.0
        self._unwrap_initialized: bool = False

    def fit_linear(self) -> tuple[float, float, float] | None:
        if len(self.points) < 2:
            self.fitted = False
            return None
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])
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

    def predict(self, xvf_angle: float) -> float:
        if self.fitted and math.isfinite(self.slope) and math.isfinite(self.intercept):
            return self.slope * xvf_angle + self.intercept
        return 0.0

    def predict_clamped(self, xvf_angle: float, lo=-135.0, hi=135.0) -> float:
        raw = self.predict(xvf_angle)
        return max(lo, min(hi, raw))

    def unwrap_angle(self, current: float, previous: float) -> float:
        diff = current - previous
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return previous + diff

    def reset_unwrap(self):
        self._unwrap_initialized = False
        self._accumulated_xvf = 0.0

    def predict_unwrapped(self, xvf_raw: float) -> float:
        if not self._unwrap_initialized:
            self._prev_raw_xvf = xvf_raw
            self._accumulated_xvf = xvf_raw
            self._unwrap_initialized = True
        else:
            unwrapped = self.unwrap_angle(xvf_raw, self._prev_raw_xvf)
            self._prev_raw_xvf = xvf_raw
            self._accumulated_xvf = unwrapped
        return self.predict_clamped(self._accumulated_xvf)


class CalibrationStorage:
    """标定数据的 JSON 保存 / 加载。"""

    def __init__(self, filepath: str = DEFAULT_CALIB_FILE):
        self.filepath = filepath

    def load(self) -> tuple[list, float, float, float, bool, str]:
        if not os.path.exists(self.filepath):
            return ([], 0.0, 0.0, 0.0, False, "")
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            points_raw = data.get("calibration_points", [])
            points = [(float(p[0]), float(p[1])) for p in points_raw]
            slope = float(data.get("slope", 0.0))
            intercept = float(data.get("intercept", 0.0))
            r_squared = float(data.get("r_squared", 0.0))
            fitted = bool(data.get("fitted", False))
            # 校验: NaN/Inf 数据应丢弃 fitted 标志
            if fitted and not (math.isfinite(slope) and math.isfinite(intercept)):
                print("[Storage] 标定数据包含 NaN/Inf，丢弃 fitted 标志")
                fitted = False
            return (
                points,
                slope,
                intercept,
                r_squared,
                fitted,
                str(data.get("last_calibrated", "")),
            )
        except Exception as e:
            print(f"[Storage] 加载失败: {e}")
            return ([], 0.0, 0.0, 0.0, False, "")

    def save(self, model: CalibrationModel):
        """保存标定数据到 JSON 文件。"""
        from datetime import datetime
        data = {
            "calibration_points": [[p[0], p[1]] for p in model.points],
            "slope": model.slope,
            "intercept": model.intercept,
            "r_squared": model.r_squared,
            "fitted": model.fitted,
            "last_calibrated": datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
