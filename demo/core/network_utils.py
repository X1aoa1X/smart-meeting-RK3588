"""网络工具函数。"""

import socket
import subprocess


def get_device_ip() -> str:
    """获取本机主要 LAN IP 地址。

    通过 UDP socket 连接外部地址的方式获取本机实际使用的网络接口 IP，
    避免返回 localhost。回退方案使用 hostname -I。
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        # 不实际发送数据，仅通过路由表确定出口 IP
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        pass

    # 回退: 解析 hostname -I 输出
    try:
        result = subprocess.run(
            ['hostname', '-I'], capture_output=True, text=True, timeout=2)
        ips = result.stdout.strip().split()
        if ips:
            return ips[0]
    except Exception:
        pass

    return '127.0.0.1'
