# core/ — 视听融合追踪系统可复用核心模块
#
# 模块列表:
#   display_env        — sudo/X11 显示环境修复 + OpenCV-Qt5 冲突修复
#   network_utils      — 网络工具 (get_device_ip)
#   event_bus          — 轻量发布/订阅事件总线
#   pwm_controller     — PWM 舵机控制 (sysfs)
#   alsa_capture       — ALSA PCM 音频采集 (ctypes)
#   silero_vad         — Silero VAD 语音活动检测 (PyTorch JIT)
#   calibration        — XVF→舵机 线性标定模型 + JSON 持久化
#   audio_vad          — 后台 ALSA 录音 + Silero VAD 推理线程
#   respeaker          — ReSpeaker USB 麦克风阵列驱动 + DOA 读取线程
#   metrics_collector  — 非侵入式追踪指标采集 + CSV 导出
#   overlay_renderer   — 视频帧叠加渲染 (名片条)
#   apriltag_camera    — 后台摄像头 + AprilTag 检测线程
#   yolo_camera        — 后台摄像头 + RKNN YOLO 推理线程
#   stream_publisher   — GStreamer RTSP 推流线程
#   fusion_engine      — 视听融合状态机 (纯 Python, 零 GUI 依赖)
