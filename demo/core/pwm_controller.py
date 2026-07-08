"""PWM 舵机控制器 — 通过 Linux sysfs 接口控制舵机。

用法:
  pwm = PWMController(pwmchip=0, pwm_index="0",
                      angle_min=-135, angle_max=135, label="PWM-H")
  pwm.init()
  pwm.set_angle(45.0)
  pwm.cleanup()
"""

import os
import time


class PWMController:
    """PWM 舵机控制器 — 支持通过构造函数参数配置不同 PWM 通道和角度范围。"""

    PERIOD_NS = 10_000_000
    PCT_5  = int(PERIOD_NS * (1 - 0.05))     # 9_500_000 ns
    PCT_25 = int(PERIOD_NS * (1 - 0.25))     # 7_500_000 ns
    PCT_15 = int(PERIOD_NS * (1 - 0.15))     # 8_500_000 ns (中间)

    def __init__(self, pwmchip: int, pwm_index: str = "0",
                 angle_min: float = -135, angle_max: float = 135,
                 duty_at_min: int | None = None,
                 duty_at_max: int | None = None,
                 label: str = "PWM"):
        self.PWMCHIP_PATH = f"/sys/class/pwm/pwmchip{pwmchip}"
        self.PWM_INDEX = pwm_index
        self.ANGLE_MIN = float(angle_min)
        self.ANGLE_MAX = float(angle_max)
        self.label = label

        if duty_at_max is None:
            self.DUTY_AT_MAX = self.PCT_5
        else:
            self.DUTY_AT_MAX = int(duty_at_max)
        if duty_at_min is None:
            self.DUTY_AT_MIN = self.PCT_25
        else:
            self.DUTY_AT_MIN = int(duty_at_min)
        self.DUTY_MID = self.PCT_15

        self._pwm_base = os.path.join(self.PWMCHIP_PATH, f"pwm{self.PWM_INDEX}")
        self._current_angle = 0.0
        self._initialized = False

    @staticmethod
    def _write_file(path: str, value):
        with open(path, "w") as f:
            f.write(str(value))

    def init(self) -> bool:
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
        if not self._initialized:
            return
        try:
            # 先回中舵机再断电，防止物理损坏
            self.set_angle(0.0)
            time.sleep(0.5)
            self._write_file(os.path.join(self._pwm_base, "enable"), "0")
            unexport_path = os.path.join(self.PWMCHIP_PATH, "unexport")
            self._write_file(unexport_path, self.PWM_INDEX)
            print(f"[{self.label}] 已清理")
        except Exception as e:
            print(f"[{self.label}] 清理警告: {e}")
        self._initialized = False

    def angle_to_duty(self, angle: float) -> int:
        angle = max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle))
        if angle >= 0:
            ratio = angle / float(self.ANGLE_MAX)
            duty = self.DUTY_MID + ratio * (self.DUTY_AT_MAX - self.DUTY_MID)
        else:
            ratio = angle / float(self.ANGLE_MIN)
            duty = self.DUTY_MID + ratio * (self.DUTY_AT_MIN - self.DUTY_MID)
        return int(duty)

    def set_angle(self, angle: float):
        if not self._initialized:
            return
        angle = max(self.ANGLE_MIN, min(self.ANGLE_MAX, angle))
        duty = self.angle_to_duty(angle)
        try:
            self._write_file(os.path.join(self._pwm_base, "duty_cycle"), duty)
            self._current_angle = angle
        except Exception as e:
            print(f"[{self.label}] set_angle({angle}) 写入失败: {e}")

    def get_angle(self) -> float:
        return self._current_angle

    @property
    def initialized(self) -> bool:
        return self._initialized
