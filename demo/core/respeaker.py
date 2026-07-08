"""ReSpeaker USB 麦克风阵列驱动 + DOA 后台读取线程。

用法:
  from core.respeaker import ReSpeaker, ReSpeakerReader, find_respeaker

  # 直接使用 (适用于非 Qt 脚本)
  dev = find_respeaker()
  if dev:
      result = dev.read("DOA_VALUE")
      doa_angle = -float(result[0])
      speech = bool(result[1])

  # Qt 后台线程 (适用于 PyQt5 GUI)
  reader = ReSpeakerReader()
  reader.doa_update.connect(on_doa)   # (doa_angle, speech)
  reader.start()

依赖: pyusb, PyQt5
"""

import time
import struct
import usb.core
import usb.util
from PyQt5.QtCore import QThread, pyqtSignal

_REASPEAKER_VID = 0x2886
_REASPEAKER_PID = 0x001A

_PARAMETERS = {
    "VERSION":             (48,  0,  3,  "ro", "uint8"),
    "AEC_AZIMUTH_VALUES":  (33, 75, 16, "ro", "radians"),
    "DOA_VALUE":           (20, 18,  4,  "ro", "uint16"),
    "REBOOT":              (48,  7,  1,  "wo", "uint8"),
}


class ReSpeaker:
    """ReSpeaker USB 麦克风阵列驱动 — 通过 vendor control transfer 读取 DOA 数据。"""

    TIMEOUT = 3000

    def __init__(self, dev):
        self.dev = dev

    def read(self, name: str):
        try:
            meta = _PARAMETERS[name]
        except KeyError:
            return None

        resid  = meta[0]
        cmdid  = 0x80 | meta[1]
        length = meta[2] + 1

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
    """查找并返回 ReSpeaker 设备，未找到返回 None。"""
    dev = usb.core.find(idVendor=_REASPEAKER_VID, idProduct=_REASPEAKER_PID)
    if dev is None:
        return None
    return ReSpeaker(dev)


class ReSpeakerReader(QThread):
    """后台线程中连续读取 ReSpeaker DOA 数据，通过 Qt 信号发送给主线程。"""

    doa_update   = pyqtSignal(float, bool)   # doa_angle, speech_detected
    device_error = pyqtSignal(str)
    device_ready = pyqtSignal(str)

    POLL_INTERVAL = 0.08  # ~12.5 Hz

    def __init__(self):
        super().__init__()
        self._running = False
        self._respeaker: ReSpeaker | None = None

    def run(self):
        self._running = True

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
                for _ in range(20):
                    if not self._running:
                        return
                    time.sleep(0.1)

        consecutive_errors = 0
        while self._running:
            try:
                result = self._respeaker.read("DOA_VALUE")
                if result and len(result) >= 2:
                    doa_raw   = result[0]
                    speech    = bool(result[1])
                    doa_angle = -float(doa_raw)          # 取反（与 xvf_test.py 一致）
                    self.doa_update.emit(doa_angle, speech)
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

            if consecutive_errors > 30:
                self.device_error.emit("ReSpeaker 连接丢失，尝试重连…")
                if self._respeaker is not None:
                    try:
                        self._respeaker.close()
                    except Exception:
                        pass
                self._respeaker = None
                while self._running:
                    dev = find_respeaker()
                    if dev is not None:
                        self._respeaker = dev
                        self.device_ready.emit("reconnected")
                        consecutive_errors = 0
                        break
                    time.sleep(1.0)

            time.sleep(self.POLL_INTERVAL)

        if self._respeaker:
            try:
                self._respeaker.close()
            except Exception:
                pass
            self._respeaker = None

    def stop(self):
        self._running = False
        self.wait(2000)
