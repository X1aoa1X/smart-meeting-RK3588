"""X11 显示环境修复 — 解决 sudo 丢失 DISPLAY/XAUTHORITY 及 OpenCV-Qt5 冲突。

用法:
  from core.display_env import fix_display_env
  fix_display_env()          # 必须在任何 Qt/X11 相关 import 之前调用

  import cv2                 # 然后导入 cv2
  from core.display_env import fix_cv2_qt_conflict
  fix_cv2_qt_conflict()      # 在 import cv2 之后, import PyQt5 之前调用

  from PyQt5.QtWidgets import ...
"""

import os
import pwd


def _resolve_real_uid():
    """返回"真实用户"的 UID，处理 sudo/doas 场景。

    sudo 下 os.getuid() 返回 0，但 Xauthority 属于原始用户。
    通过 SUDO_UID、DOAS_UID、SUDO_USER 环境变量逆推。
    """
    # 1) 直接尝试环境变量中的 UID
    for var in ("SUDO_UID", "DOAS_UID", "PKEXEC_UID"):
        val = os.environ.get(var)
        if val and val.isdigit():
            return int(val)

    # 2) 通过 SUDO_USER / DOAS_USER 用户名查询
    for var in ("SUDO_USER", "DOAS_USER", "USER"):
        username = os.environ.get(var)
        if username and username != "root":
            try:
                return pwd.getpwnam(username).pw_uid
            except KeyError:
                pass

    # 3) 回退：当前进程 UID
    return os.getuid()


def _resolve_real_home():
    """返回真实用户的家目录。"""
    uid = _resolve_real_uid()
    try:
        return pwd.getpwuid(uid).pw_dir
    except KeyError:
        return os.path.expanduser("~")


def fix_display_env():
    """自动检测并设置本地显示所需的 DISPLAY / XAUTHORITY 环境变量。

    必须在任何可能触发 Qt/X11 初始化的 import 之前调用。
    sudo 会清空 DISPLAY / XAUTHORITY / XDG_RUNTIME_DIR，导致 Qt 无法连接本地 X 服务器。
    此函数通过 SUDO_UID / SUDO_USER 找到原始用户的 Xauthority。
    """
    # 1. DISPLAY — 如果未设置或为空，默认使用本地主控台 :0
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"
        print(f"[Display] DISPLAY 未设置，自动设为 {os.environ['DISPLAY']}")

    # 2. XAUTHORITY — 尝试自动查找
    if not os.environ.get("XAUTHORITY"):
        uid = _resolve_real_uid()
        candidates = []

        # GDM 登录会话 (Ubuntu 默认)
        candidates.append(f"/run/user/{uid}/gdm/Xauthority")
        # 跨 display manager 通用 XDG 路径
        candidates.append(f"/run/user/{uid}/Xauthority")
        # 传统 ~/.Xauthority
        candidates.append(os.path.join(_resolve_real_home(), ".Xauthority"))
        # LightDM / SDDM 等可能使用 /tmp/xauth_* 或 /var/run/lightdm/root/:0
        # 最后兜底 — 用 xauth 提取当前 display 的 cookie
        candidates.append(f"/tmp/xauth_{uid}")

        for path in candidates:
            if os.path.isfile(path) and os.access(path, os.R_OK):
                os.environ["XAUTHORITY"] = path
                print(f"[Display] XAUTHORITY 自动设为 {path}")
                break
        else:
            print("[Display] ⚠️ 未找到 XAUTHORITY 文件，尝试无授权连接")


def fix_cv2_qt_conflict():
    """修复 OpenCV 内建 Qt 插件与系统 PyQt5 的冲突。

    必须在 import cv2 之后、from PyQt5 import ... 之前调用。
    OpenCV (cv2) 设置 QT_PLUGIN_PATH 指向自带的 Qt 插件目录，
    这会覆盖系统 PyQt5 的插件搜索路径，导致 PyQt5 平台插件找不到。
    """
    _CV2_QT_PLUGIN_PATH = "/usr/local/lib/python3.10/dist-packages/cv2/qt/plugins"
    if os.environ.get("QT_PLUGIN_PATH") == _CV2_QT_PLUGIN_PATH:
        del os.environ["QT_PLUGIN_PATH"]

    _SYS_QT5_PLATFORM_PATH = "/usr/lib/aarch64-linux-gnu/qt5/plugins/platforms"
    if os.path.isdir(_SYS_QT5_PLATFORM_PATH):
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = _SYS_QT5_PLATFORM_PATH
