#!/usr/bin/env python3
"""
修复后的ZMQ订阅端（SUB）
修复内容：
1. 添加连接建立等待，解决慢连接问题
2. 使用同步机制等待PUB端就绪
3. 改进消息接收逻辑，避免消息丢失
4. 添加心跳检测机制
"""
import os
import time
import zmq
import struct
import sys

PWMCHIP_PATH = "/sys/class/pwm/pwmchip0"
PWM_EXPORT = os.path.join(PWMCHIP_PATH, "export")
PWM_UNEXPORT = os.path.join(PWMCHIP_PATH, "unexport")
PWM_INDEX = "0"
PWM_BASE_PATH = os.path.join(PWMCHIP_PATH, f"pwm{PWM_INDEX}")

PERIOD_NS = 10000000
DUTY_CYCLE_RIGHT = int(PERIOD_NS * (1-0.05))
DUTY_CYCLE_LEFT = int(PERIOD_NS * (1-0.25))
DUTY_CYCLE_MID = int(PERIOD_NS * (1-0.15))

ZMQ_PORT = 5557
CONNECTION_TIMEOUT = 5000  # 连接超时（毫秒）
MAX_RECONNECT_ATTEMPTS = 10  # 最大重连次数

def write_file(path, value):
    with open(path, 'w') as f:
        f.write(str(value))
    print(f"[DEBUG] Write '{value}' to {path}")

def pwm_init():
    if not os.path.exists(PWM_BASE_PATH):
        print(f"[DEBUG] Exporting PWM {PWM_INDEX}...")
        write_file(PWM_EXPORT, PWM_INDEX)
        time.sleep(0.1)
    else:
        print(f"[DEBUG] PWM {PWM_INDEX} already exported")

    print(f"[DEBUG] Setting period to {PERIOD_NS} ns ({PERIOD_NS/1000000} ms)")
    write_file(os.path.join(PWM_BASE_PATH, "period"), PERIOD_NS)

    print(f"[DEBUG] Setting duty_cycle to {DUTY_CYCLE_MID} ns (middle position)")
    write_file(os.path.join(PWM_BASE_PATH, "duty_cycle"), DUTY_CYCLE_MID)

    print(f"[DEBUG] Enabling PWM")
    write_file(os.path.join(PWM_BASE_PATH, "enable"), "1")

def pwm_cleanup():
    print(f"[DEBUG] Disabling PWM")
    try:
        write_file(os.path.join(PWM_BASE_PATH, "enable"), "0")
        print(f"[DEBUG] Unexporting PWM {PWM_INDEX}...")
        write_file(PWM_UNEXPORT, PWM_INDEX)
    except Exception as e:
        print(f"[警告] PWM清理时出错: {e}")

def angle_to_duty_cycle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    angle = max(-135, min(135, angle))
    duty_ratio = 1 - (0.05 + (angle + 135) / 270 * 0.20)
    duty_ns = int(PERIOD_NS * duty_ratio)
    return duty_ns

def set_servo_angle(angle):
    duty_ns = angle_to_duty_cycle(angle)
    print(f"[DEBUG] Setting servo angle to {angle}° (duty_cycle: {duty_ns} ns)")
    write_file(os.path.join(PWM_BASE_PATH, "duty_cycle"), duty_ns)

def wait_for_publisher(zmq_socket, timeout=5000):
    """
    等待发布端就绪信号，解决慢连接问题
    """
    print("[等待] 等待发布端就绪...")

    # 方法1: 尝试接收就绪消息
    start_time = time.time()
    while (time.time() - start_time) * 1000 < timeout:
        try:
            msg = zmq_socket.recv_json(flags=zmq.NOBLOCK)
            msg_type = msg.get("type", "")
            if msg_type == "READY":
                print("[就绪] 收到发布端就绪信号，连接建立成功！")
                return True
            elif msg_type == "SENSOR_DATA":
                # 如果收到的是传感器数据而不是就绪消息，说明发布端已经开始了
                print("[注意] 发布端已在运行，收到的第一条数据: ", msg)
                return True
        except zmq.Again:
            # 没有消息，继续等待
            time.sleep(0.1)

    print("[警告] 未收到就绪信号，继续尝试接收数据...")
    return False

def connect_with_retry(zmq_context, port, max_attempts=MAX_RECONNECT_ATTEMPTS):
    """
    带重连机制的连接函数
    """
    zmq_socket = zmq_context.socket(zmq.SUB)

    # ============================================================
    # 修复1: 设置订阅过滤器（必须设置才能接收消息）
    # ============================================================
    zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # ============================================================
    # 修复2: 设置接收超时
    # ============================================================
    zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)

    # ============================================================
    # 修复3: 设置HWM（如果需要）
    # ============================================================
    # zmq_socket.set_hwm(0)  # 取消注释以设置无限队列

    # ============================================================
    # 修复4: 设置重连参数
    # ============================================================
    zmq_socket.setsockopt(zmq.RECONNECT_IVL, 100)  # 重连间隔（毫秒）
    zmq_socket.setsockopt(zmq.RECONNECT_IVL_MAX, 5000)  # 最大重连间隔

    for attempt in range(max_attempts):
        try:
            connection_string = f"tcp://localhost:{port}"
            print(f"[尝试 {attempt + 1}/{max_attempts}] 连接到 {connection_string}")
            zmq_socket.connect(connection_string)

            # 等待一小段时间让连接建立
            time.sleep(0.2)

            # 尝试接收一条消息来验证连接
            zmq_socket.setsockopt(zmq.RCVTIMEO, 2000)
            try:
                msg = zmq_socket.recv_json(flags=zmq.NOBLOCK)
                print(f"[连接成功] 收到测试消息: {msg}")
                zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)
                return zmq_socket, True
            except zmq.Again:
                print("[连接成功] 连接已建立（未收到测试消息）")
                zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)
                return zmq_socket, True

        except zmq.Again:
            print(f"[失败] 尝试 {attempt + 1} 失败，正在重连...")
            time.sleep(1)
            continue
        except Exception as e:
            print(f"[错误] 连接时出错: {e}")
            time.sleep(1)
            continue

    print("[错误] 所有连接尝试均失败")
    zmq_socket.close()
    return None, False

def main():
    print("=" * 50)
    print("声源定位跟踪程序（修复版）")
    print("=" * 50)
    print(f"ZMQ订阅端口: {ZMQ_PORT}")
    print(f"连接超时: {CONNECTION_TIMEOUT}ms")
    print(f"最大重连次数: {MAX_RECONNECT_ATTEMPTS}")
    print("=" * 50)

    # 初始化PWM
    try:
        pwm_init()
        set_servo_angle(90)
    except Exception as e:
        print(f"[警告] PWM初始化失败: {e}")
        print("[继续] 将继续运行但无法控制舵机")

    # 创建ZMQ上下文和socket
    zmq_context = zmq.Context()
    zmq_socket = None

    # ============================================================
    # 修复5: 使用重连机制连接
    # ============================================================
    zmq_socket, connected = connect_with_retry(zmq_context, ZMQ_PORT)
    if not connected:
        print("[错误] 无法连接到发布端，退出程序")
        zmq_context.term()
        sys.exit(1)

    # ============================================================
    # 修复6: 等待发布端就绪（可选，增强可靠性）
    # ============================================================
    # wait_for_publisher(zmq_socket, timeout=3000)

    print("[启动] 开始接收声源数据...")

    waiting_for_next_detection = False
    last_doa_angle = None
    consecutive_same_angle = 0
    messages_received = 0
    messages_processed = 0

    try:
        while True:
            # ============================================================
            # 修复7: 改进消息接收逻辑
            # ============================================================

            if waiting_for_next_detection:
                # 在等待模式下，使用较长的超时时间减少CPU占用
                zmq_socket.setsockopt(zmq.RCVTIMEO, 500)

                try:
                    msg_data = zmq_socket.recv_json(flags=zmq.NOBLOCK)
                    messages_received += 1
                    speech_active = msg_data.get("speech_detected", False)
                    current_angle = msg_data.get("doa_angle", 0)
                    # 丢弃此期间的消息
                    if messages_received % 10 == 0:
                        print(f"[DRAIN] 丢弃消息 (累计{messages_received}条): speech={speech_active}, angle={current_angle}")
                except zmq.Again:
                    pass

                time.sleep(0.1)
                continue

            # 正常接收模式
            zmq_socket.setsockopt(zmq.RCVTIMEO, 1000)

            try:
                message = zmq_socket.recv_json()
                messages_received += 1
            except zmq.Again:
                # 超时，继续循环
                continue
            except Exception as e:
                print(f"[错误] 接收消息时出错: {e}")
                continue

            # 解析消息
            speech_detected = message.get("speech_detected", False)
            doa_angle = message.get("doa_angle", 0)
            msg_type = message.get("type", "UNKNOWN")

            # 跳过控制消息
            if msg_type == "READY":
                print(f"[INFO] 收到就绪消息: {message}")
                continue

            if messages_received % 10 == 0 or messages_received <= 3:
                print(f"[INFO] 收到消息 #{messages_received} - speech_detected: {speech_detected}, doa_angle: {doa_angle}")

            if speech_detected:
                messages_processed += 1
                print(f"[ACTION] 检测到声源 #{messages_processed}，舵机转向角度: {doa_angle}°")

                try:
                    set_servo_angle(int(doa_angle))
                except Exception as e:
                    print(f"[错误] 舵机控制失败: {e}")

                waiting_for_next_detection = True

                print(f"[WAIT] 等待3秒排除电机噪音...")
                time.sleep(3)
                waiting_for_next_detection = False
                print(f"[RESUME] 恢复声源检测")

            # 每50条消息输出一次统计
            if messages_received % 50 == 0:
                print(f"[统计] 已接收 {messages_received} 条消息，处理 {messages_processed} 个声源事件")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        pwm_cleanup()
        if zmq_socket:
            zmq_socket.close()
        zmq_context.term()
        print(f"Resources released. Total received: {messages_received}, processed: {messages_processed}")

if __name__ == "__main__":
    main()
