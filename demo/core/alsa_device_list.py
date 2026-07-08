"""ALSA 录音设备枚举 — 通过 ctypes 调用 libasound，零 pip 依赖。

用法:
  from core.alsa_device_list import list_capture_devices
  devices = list_capture_devices()
  # [{"name": "hw:1,0", "card_name": "rockchip-nau8822",
  #   "device_name": "nau8822-hifi-0", "card_index": 1, "device_index": 0}, ...]
"""

import ctypes
import ctypes.util
import os


def list_capture_devices() -> list[dict]:
    """枚举所有支持录音的 ALSA PCM 设备。

    使用 ctypes 调用 libasound：
    1. snd_card_next 遍历声卡 → snd_ctl_open 获取声卡名称
    2. snd_ctl_pcm_next_device 遍历 PCM 设备
    3. snd_pcm_open(..., CAPTURE) 实测每个设备是否支持录音（最可靠的方法）

    Returns:
        list[dict]: 按声卡编号排序的设备列表，每项包含:
            - name (str): ALSA 设备名，如 "hw:1,0"
            - card_name (str): 声卡名称，如 "rockchip-nau8822"
            - device_name (str): PCM 设备名称
            - card_index (int): 声卡编号
            - device_index (int): 设备编号
    """
    libname = ctypes.util.find_library("asound")
    if not libname:
        print("[AlsaDeviceList] 找不到 libasound.so")
        return _fallback_parse_proc()

    try:
        alsa = ctypes.CDLL(libname)
    except OSError as e:
        print(f"[AlsaDeviceList] 加载 libasound 失败: {e}")
        return _fallback_parse_proc()

    # ── 公共函数签名 ──────────────────────────────────────────
    alsa.snd_card_next.restype = ctypes.c_int
    alsa.snd_card_next.argtypes = [ctypes.POINTER(ctypes.c_int)]

    alsa.snd_ctl_open.restype = ctypes.c_int
    alsa.snd_ctl_open.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_int]

    alsa.snd_ctl_close.restype = ctypes.c_int
    alsa.snd_ctl_close.argtypes = [ctypes.c_void_p]

    alsa.snd_ctl_card_info.restype = ctypes.c_int
    alsa.snd_ctl_card_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    alsa.snd_ctl_card_info_get_name.restype = ctypes.c_char_p
    alsa.snd_ctl_card_info_get_name.argtypes = [ctypes.c_void_p]

    alsa.snd_ctl_card_info_sizeof.restype = ctypes.c_size_t
    alsa.snd_ctl_card_info_sizeof.argtypes = []

    alsa.snd_ctl_pcm_next_device.restype = ctypes.c_int
    alsa.snd_ctl_pcm_next_device.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    # PCM 打开/关闭 — 与 AlsaAudioCapture 保持一致
    alsa.snd_pcm_open.restype = ctypes.c_int
    alsa.snd_pcm_open.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
        ctypes.c_int, ctypes.c_int]

    alsa.snd_pcm_close.restype = ctypes.c_int
    alsa.snd_pcm_close.argtypes = [ctypes.c_void_p]

    # snd_pcm_info — 用于获取 PCM 设备名称
    alsa.snd_pcm_info_sizeof.restype = ctypes.c_size_t
    alsa.snd_pcm_info_sizeof.argtypes = []

    alsa.snd_pcm_info.restype = ctypes.c_int
    alsa.snd_pcm_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

    alsa.snd_pcm_info_get_name.restype = ctypes.c_char_p
    alsa.snd_pcm_info_get_name.argtypes = [ctypes.c_void_p]

    alsa.snd_strerror.restype = ctypes.c_char_p
    alsa.snd_strerror.argtypes = [ctypes.c_int]

    SND_PCM_STREAM_CAPTURE = 1

    card_info_size = alsa.snd_ctl_card_info_sizeof()
    pcm_info_size = alsa.snd_pcm_info_sizeof()

    devices = []

    # ── 遍历所有声卡 ──────────────────────────────────────────
    card_ptr = ctypes.c_int(-1)
    while True:
        ret = alsa.snd_card_next(ctypes.byref(card_ptr))
        card = card_ptr.value
        if ret < 0 or card < 0:
            break

        # 打开控制接口（获取声卡名称）
        ctl_name = f"hw:{card}"
        ctl = ctypes.c_void_p()
        ret = alsa.snd_ctl_open(ctypes.byref(ctl), ctl_name.encode("utf-8"), 0)
        if ret < 0:
            continue

        card_info = (ctypes.c_uint8 * card_info_size)()
        card_name = f"card_{card}"
        if alsa.snd_ctl_card_info(ctl, ctypes.cast(card_info, ctypes.c_void_p)) >= 0:
            name_ptr = alsa.snd_ctl_card_info_get_name(
                ctypes.cast(card_info, ctypes.c_void_p))
            if name_ptr:
                card_name = name_ptr.decode("utf-8", errors="replace")

        # ── 遍历 PCM 设备，实测 CAPTURE 是否可用 ────────────────
        dev_ptr = ctypes.c_int(-1)
        while True:
            ret = alsa.snd_ctl_pcm_next_device(ctl, ctypes.byref(dev_ptr))
            dev = dev_ptr.value
            if ret < 0 or dev < 0:
                break

            dev_name = f"hw:{card},{dev}"
            pcm = ctypes.c_void_p()

            # 实际尝试打开 CAPTURE 流 — 唯一可靠的检测方式
            ret = alsa.snd_pcm_open(
                ctypes.byref(pcm), dev_name.encode("utf-8"),
                SND_PCM_STREAM_CAPTURE, 0)
            if ret != 0 and ret != -16:  # -16 = EBUSY (设备存在但被占用)
                continue  # 不支持录音 (-2 = ENOENT, 等)
            if ret == 0:
                # 打开成功 — 获取名称后关闭
                pcm_device_name = ""
                pcm_info = (ctypes.c_uint8 * pcm_info_size)()
                if alsa.snd_pcm_info(pcm, ctypes.cast(pcm_info, ctypes.c_void_p)) >= 0:
                    name_ptr = alsa.snd_pcm_info_get_name(
                        ctypes.cast(pcm_info, ctypes.c_void_p))
                    if name_ptr:
                        pcm_device_name = name_ptr.decode("utf-8", errors="replace")
                alsa.snd_pcm_close(pcm)
            else:
                # EBUSY — PCM 被占用（进程已在录音），仍标记为可用设备
                pcm_device_name = f"busy:{card},{dev}"

            devices.append({
                "name": dev_name,
                "card_name": card_name,
                "device_name": pcm_device_name,
                "card_index": card,
                "device_index": dev,
            })

        alsa.snd_ctl_close(ctl)

    # ── 按声卡编号排序 ────────────────────────────────────────
    devices.sort(key=lambda d: (d["card_index"], d["device_index"]))

    if not devices:
        return _fallback_parse_proc()

    return devices


def _fallback_parse_proc() -> list[dict]:
    """解析 /proc/asound/cards 获取声卡列表（ctypes 不可用时的回退方案）。

    不验证设备是否支持 capture — 调用方需自行处理。
    """
    devices = []
    cards_path = "/proc/asound/cards"
    if not os.path.exists(cards_path):
        return devices

    try:
        with open(cards_path, "r") as f:
            content = f.read()
    except (IOError, PermissionError):
        return devices

    import re
    # 解析格式:
    #  0 [rockchipdp0    ]: rockchip_dp0 - rockchip,dp0
    #                       rockchip,dp0
    #  1 [rockchipnau8822]: rockchip-nau882 - rockchip-nau8822
    #                       rockchip-nau8822
    for match in re.finditer(
            r'^\s*(\d+)\s+\[([^\]]+)\]\s*:\s*(.+?)(?=\n\s*\d+\s+\[|\Z)',
            content, re.MULTILINE | re.DOTALL):
        card_idx = int(match.group(1))
        card_id = match.group(2).strip()
        description = match.group(3).strip()
        # 描述的第一行通常是 "driver - card_name"
        lines = description.split("\n")
        card_name = card_id
        dev_name = ""
        if lines:
            first_line = lines[0].strip()
            if " - " in first_line:
                card_name = first_line.split(" - ", 1)[1].strip()
            dev_name = first_line

        # 对于回退方案，假设 device 0 可用
        devices.append({
            "name": f"hw:{card_idx},0",
            "card_name": card_name,
            "device_name": dev_name,
            "card_index": card_idx,
            "device_index": 0,
        })

    return devices
