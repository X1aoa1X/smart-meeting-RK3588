# 智会追声 — 嵌入式智能会议辅助系统

基于 Rockchip RK3588 NPU 的实时会议辅助系统，通过音视频融合追踪、AprilTag 身份识别与 LLM 驱动的语音播报，为线下会议提供自动追声、发言者定位、桌牌生成与会议记录等能力。

本目录包含两个相互协作的子项目：

| 子项目 | 路径 | 说明 |
|--------|------|------|
| 主程序 | `demo/` | Python 主程序（PyQt5 GUI + Streamlit 控制台 + HTTP API + 持久化层） |
| C++ 推理扩展 | `c_realtime_infer_demo/` | C++ pybind11 扩展，封装 RKNN YOLOv8 多核 NPU 推理池，供主程序调用 |

## 系统能力

- **音视频融合追踪**：3 态状态机（IDLE → AWAIT → TRACKING）以声源方向粗调，再以视觉比例控制精调，驱动双轴 PWM 舵机云台。
- **NPU 加速目标检测**：YOLOv8 模型经 RKNN 编译后运行于 RK3588 三核 NPU，C++ 扩展实现 4 路推理池，保障实时帧率。
- **AprilTag 身份识别**：5 态去抖状态机将 AprilTag 检测结果映射到参会人身份，避免误判。
- **GStreamer RTSP 推流**：在视频流上叠加姓名/角色字幕条，远程即可查看会议画面。
- **Streamlit 控制台**：5 步会前准备向导 + 实时主持控制台 + 会议记录与总结 + 数据库调试 + Agent 控制台。
- **LLM 智能播报**：DeepSeek 驱动的会议 Agent，在发言人切换、超时、静默等事件触发时生成简短播报并通过腾讯云 TTS 播放。
- **SQLite 持久化**：参会人、会议、发言片段、事件、主持人备注、Agent 决策、TTS 事件全量入库，支持 WAL 模式与异步写入。
- **桌牌自动生成**：根据参会人信息生成 PNG/PDF 桌牌，含 AprilTag 标签图。

## 系统架构

```
硬件层                处理层                       服务层                  Web 层
┌──────────────────┐  ┌─────────────────────┐    ┌─────────────────┐    ┌──────────────┐
│ PWM 舵机 (H+V)    │  │ core/fusion_engine   │    │ core/control_api │◄───│ Streamlit    │
│ ReSpeaker USB 麦 │  │ core/speaker_identity│    │ (HTTP :8800)    │    │ (:8501)      │
│ 摄像头 V4L2       │◄─┤ core/audio_vad        │    │ core/meeting_    │    │ - 会前准备    │
│ ALSA 音频         │  │ core/yolo_camera      │    │   service        │    │ - 会议控制台  │
│ NPU (RKNN)        │  │ core/apriltag_camera  │    └─────────────────┘    │ - 会议记录    │
└──────────────────┘  │ core/stream_publisher  │                           │ - 数据库调试  │
                      │ core/agent_worker      │                           │ - Agent 控制台│
                      └─────────────────────┘                           └──────────────┘
                                │
                      ┌─────────────────────┐
                      │ core/event_bus       │──► storage/event_bridge ──► SQLite
                      │ storage/ (ORM)       │
                      └─────────────────────┘
```

**设计原则**：`demos/fusion_tracker.py` 是唯一的硬件控制器，以 root 权限运行。Streamlit 控制台通过 HTTP API（`127.0.0.1:8800`）远程下发指令，所有硬件命令经 `queue.Queue` 投递到主 Qt 线程统一执行。

## 目录结构

```
submission/
├── README.md                      # 本说明文件
├── demo/                          # 智会追声 Python 主程序
│   ├── CLAUDE.md                  # 主程序架构与开发说明
│   ├── requirements.txt           # Python 依赖清单
│   ├── fusion_params.json         # 融合引擎调参配置
│   ├── xvf_calibration.json       # XVF→舵机标定参数
│   ├── competition_logo.png       # 桌牌水印图标
│   ├── core/                      # 核心业务模块
│   │   ├── fusion_engine.py       # 音视频融合 3 态状态机
│   │   ├── speaker_identity.py    # AprilTag→发言人去抖状态机
│   │   ├── yolo_camera.py         # 摄像头 + RKNN 推理 QThread（调用 C++ 扩展）
│   │   ├── apriltag_camera.py     # AprilTag 检测 QThread
│   │   ├── audio_vad.py           # ALSA 采集 + Silero VAD 推理
│   │   ├── silero_vad.py          # Silero VAD 模型封装
│   │   ├── alsa_capture.py        # ALSA PCM 采集（ctypes 零依赖）
│   │   ├── respeaker.py           # ReSpeaker USB 麦克风阵列驱动
│   │   ├── pwm_controller.py      # PWM 舵机 sysfs 控制
│   │   ├── calibration.py         # XVF→舵机线性拟合模型
│   │   ├── control_api.py         # HTTP API 服务器（:8800）
│   │   ├── meeting_service.py     # 会议生命周期管理
│   │   ├── stream_publisher.py    # GStreamer RTSP 推流（:8554）
│   │   ├── overlay_renderer.py    # RTSP 字幕条叠加渲染
│   │   ├── event_bus.py           # 轻量级发布订阅总线
│   │   ├── agent_worker.py        # LLM Agent 工作线程
│   │   ├── agent_llm.py           # LLM 服务调用
│   │   ├── agent_rules.py         # Agent 触发规则与策略
│   │   ├── tts_engine.py          # 腾讯云 TTS 引擎
│   │   ├── tts_router.py          # TTS 路由策略
│   │   ├── tts_cache.py           # TTS 音频缓存
│   │   ├── announcer.py           # 播报器（统一调度 LLM + TTS）
│   │   ├── duplex_controller.py   # 双工控制（避免播报自我触发）
│   │   ├── audio_player.py        # 音频播放（绕过 PulseAudio）
│   │   ├── desk_card_generator.py # 桌牌 PNG/PDF 生成
│   │   ├── metrics_collector.py   # 追踪指标采集与 CSV 导出
│   │   ├── display_env.py         # X11 环境与 cv2/Qt 冲突修复
│   │   └── network_utils.py       # 网络工具
│   ├── demos/                     # 集成入口
│   │   ├── fusion_tracker.py      # 主程序入口（PyQt5 GUI）
│   │   └── apriltag_detector.py   # AprilTag 独立演示
│   ├── app_streamlit/             # Streamlit Web 控制台
│   │   ├── Home.py                # 仪表盘
│   │   ├── ui_style.py            # UI 样式
│   │   └── pages/                 # 5 个功能页面
│   ├── storage/                   # 持久化层
│   │   ├── db.py                  # 引擎、WAL、会话管理
│   │   ├── models.py              # ORM 模型（9 张表）
│   │   ├── repo.py                # Repository/DAO 层
│   │   ├── event_bridge.py        # EventBus→DB 异步写入
│   │   ├── migration.py           # 迁移框架
│   │   └── migrations/            # 迁移脚本
│   ├── configs/                   # 业务策略与配置
│   │   ├── agent_policy.json      # Agent 触发策略
│   │   └── desk_card_config.json  # 桌牌样式配置
│   ├── scripts/                   # 部署与启动脚本
│   │   ├── start_all.sh           # 一键启动主程序 + Streamlit
│   │   ├── stop_all.sh            # 一键停止
│   │   ├── start_fusion_tracker.sh# 启动主程序
│   │   ├── start_streamlit.sh     # 启动 Streamlit
│   │   ├── install_services.sh    # 安装 systemd 服务
│   │   ├── smart-meeting-runtime.service
│   │   ├── smart-meeting-streamlit.service
│   │   └── fusion_tracker.desktop # 桌面自启动
│   ├── models/silero_vad.jit      # Silero VAD 模型权重
│   ├── rknnModel/                 # RKNN 编译后模型
│   │   ├── best.rknn              # YOLOv8 检测模型
│   │   └── yolo_world_v2.rknn     # YOLO World 模型
│   ├── rknnpool/rknnpool_ld.py    # RKNN 推理池加载器
│   ├── func/                      # YOLO 后处理函数（Python 版，已被 C++ 扩展替代）
│   ├── tagStandard41h12/          # AprilTag 41h12 标签图集（项目运行所需）
│   ├── data/                      # 运行时数据目录（运行时自动生成）
│   └── exports/desk_cards/        # 桌牌导出目录（运行时自动生成）
│
└── c_realtime_infer_demo/         # C++ RKNN 实时推理扩展（pybind11）
    ├── CLAUDE.md                  # 扩展模块说明
    ├── yolo_camera.py             # Python 摄像头推理入口（旧版参考实现）
    ├── func_yolov8_optimize.py    # YOLOv8 后处理 Python 参考实现
    ├── best.rknn                  # RKNN 模型文件
    ├── camera_infer/              # 纯 C++ 摄像头实时推理 demo
    │   ├── CMakeLists.txt
    │   ├── build.sh
    │   ├── coco_80_labels_list.txt
    │   ├── yolo_camera.cc         # 单核 NPU 独立 demo
    │   └── yolo_camera_fast.cc    # 单核 NPU 优化版 demo
    ├── yolo_core/                 # pybind11 C++ 扩展模块源码
    │   ├── CMakeLists.txt         # 构建配置
    │   ├── build.sh               # 一键构建脚本
    │   ├── include/               # 头文件
    │   │   ├── types.h            # 公共数据结构
    │   │   ├── rknn_context.h     # RKNN 上下文 RAII 封装
    │   │   ├── rknn_pool.h        # 多核 NPU 推理池
    │   │   ├── post_process.h     # YOLOv8 后处理
    │   │   └── yolo_engine.h      # 高层推理引擎
    │   ├── src/                   # 实现文件
    │   │   ├── rknn_context.cc
    │   │   ├── rknn_pool.cc
    │   │   ├── post_process.cc
    │   │   └── yolo_engine.cc
    │   └── bindings/
    │       └── yolo_module.cc     # pybind11 绑定层
    └── .trae/documents/
        └── c_plus_plus_replacement_for_python_yolo_module.md  # C++ 替换方案设计文档
```

## 两个项目的关系

`c_realtime_infer_demo/yolo_core/` 是 `demo/core/yolo_camera.py` 所调用的 C++ pybind11 推理扩展源码。构建 `yolo_core` 产生的 `.so` 文件需放置到 `demo/` 根目录下，由 `yolo_camera.py` 以 `import yolo_core` 方式加载，实现 NPU 加速的目标检测。

C++ 扩展提供以下能力，完全替代原 Python 推理路径：

- 4 路 RKNN 上下文组成的推理池，按 round-robin 分配到 RK3588 的三个 NPU 核心
- YOLOv8 后处理（letterbox + DFL + per-class NMS）
- 异步 `submit_frame` / `get_result` 接口，不阻塞 Qt 事件循环
- 同步 `compute_deviation` 接口，复用主推理的检测框计算人体相对画面中心的归一化偏移
- `cv::Mat` 与 `numpy.ndarray` 零拷贝互转，关键接口在 C++ 侧释放 GIL

接口契约与原 Python 实现保持一致，`demo/demos/fusion_tracker.py` 无需任何修改即可完成替换。

## 硬件要求

| 硬件 | 规格 | 用途 |
|------|------|------|
| 主板 | Rockchip RK3588（3 核 NPU） | 运行主程序与 RKNN 推理 |
| 摄像头 | V4L2 UVC，`/dev/video21`，1920×1080 MJPG | 画面采集与目标检测 |
| 麦克风阵列 | ReSpeaker USB（VID:0x2886, PID:0x001A），接 USB3.0 | 声源方向（DOA）与硬件 VAD |
| 板载 codec | NAU8822，`plughw:1,0`，16kHz S16_LE | Silero VAD 音频采集 |
| 舵机 | 双轴 PWM 云台，`pwmchip0`（水平）/ `pwmchip1`（垂直） | 摄像头朝向控制，需 root 权限 |

## 构建与运行

### 1. 构建 C++ 推理扩展

```bash
cd c_realtime_infer_demo/yolo_core
./build.sh
# 产物：build/yolo_core.cpython-310-aarch64-linux-gnu.so
# 自动拷贝到 ../../demo/ 目录
```

构建依赖：

- `librknnrt.so` 与 `rknn_api.h`（RKNN 运行时库，系统安装或来自 `rknn_model_zoo-2.1.0/3rdparty/`）
- OpenCV 4.x（`core`、`imgproc` 组件）
- `pybind11`（`pip install pybind11`）
- `cmake`（`pip install cmake`）
- Python3 开发头文件与 NumPy 头文件

### 2. 安装 Python 依赖

```bash
cd demo
pip install -r requirements.txt
```

系统级依赖（板端预装）：

- `rknnlite`（Rockchip RKNPU2 SDK `.whl`）
- `opencv-python`、`PyQt5`、`torch` ≥ 2.0、`pyusb`、`numpy`、`scipy`

### 3. 标定

首次部署需完成 XVF→舵机线性标定：

```bash
PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages sudo -E python3 calibrate_gui.py
```

标定结果写入 `xvf_calibration.json` 与数据库 `system_config` 表的 `calibration` 段，数据库优先读取，JSON 作为回退。

### 4. 启动服务

**一键启动**：

```bash
sudo ./scripts/start_all.sh       # 启动主程序 + Streamlit
sudo ./scripts/stop_all.sh        # 停止两者
```

**systemd 服务**（推荐生产部署）：

```bash
sudo ./scripts/install_services.sh                                          # 一次性安装
sudo systemctl start smart-meeting-runtime smart-meeting-streamlit          # 启动
sudo systemctl enable smart-meeting-runtime smart-meeting-streamlit         # 开机自启
```

### 5. 验证

```bash
# 主程序状态
curl http://127.0.0.1:8800/api/status | python3 -m json.tool

# Streamlit 控制台
# 浏览器访问 http://<板端IP>:8501
```

### 6. 查看日志

```bash
journalctl -u smart-meeting-runtime -f       # systemd 日志
tail -f demo/data/fusion_tracker.log         # 主程序日志
tail -f demo/data/streamlit.log              # Streamlit 日志
```

## 配置

### 环境变量

启动前需配置以下环境变量（通过 `scripts/*.sh` 或 systemd unit 注入）：

| 变量 | 用途 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek LLM API Key |
| `SECRET_ID` / `SECRET_KEY` / `APPID` | 腾讯云语音合成凭证 |
| `DISPLAY` / `XAUTHORITY` | PyQt5 GUI 显示授权 |
| `PYTHONPATH` | 用户级 Python 包路径 |

### 关键配置文件

| 文件 | 说明 |
|------|------|
| `demo/fusion_params.json` | 融合引擎参数（VAD 阈值、死区、增益、垂直偏置等），启动加载、关闭自动保存 |
| `demo/xvf_calibration.json` | XVF→舵机标定参数（斜率、截距、R²），数据库优先 |
| `demo/configs/agent_policy.json` | Agent 触发策略（会议开始/结束、发言人切换、超时、静默等） |
| `demo/configs/desk_card_config.json` | 桌牌样式（尺寸、字体、配色、标语） |
| `demo/app_streamlit/.streamlit/config.toml` | Streamlit 服务配置 |

## HTTP API 参考

所有接口相对于 `http://127.0.0.1:8800`。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/status` | 系统全状态快照 |
| POST | `/api/meeting/start` | 启动会议（`{"meeting_id": N}`） |
| POST | `/api/meeting/end` | 结束当前会议 |
| POST | `/api/meeting/pause` | 暂停追踪 |
| POST | `/api/meeting/resume` | 恢复追踪 |
| POST | `/api/control/recenter` | 舵机回中（0°） |
| POST | `/api/control/start_tracking` | 启动自动追踪 |
| POST | `/api/control/stop_tracking` | 停止自动追踪 |
| POST | `/api/control/lock_speaker` | 锁定当前发言人身份 |
| POST | `/api/control/unlock_speaker` | 恢复自动身份识别 |
| POST | `/api/control/manual_speaker` | 手动指定发言人（`{"tag_id": "A001"}`） |
| POST | `/api/control/set_overlay` | 设置字幕叠加（`{"enabled": bool, "show_debug": bool}`） |
| POST | `/api/control/start_stream` | 启动 RTSP 推流 |
| POST | `/api/control/stop_stream` | 停止 RTSP 推流 |
| GET | `/api/events?meeting_id=N&minutes=M` | 事件时间线与主持人备注 |
| POST | `/api/host_note` | 提交主持人备注 |

## 数据持久化

采用 SQLite + SQLAlchemy ORM，WAL 模式 + QueuePool 连接池。

核心 ORM 模型：

- `Participant` — 参会人（含 AprilTag ID、姓名、角色）
- `Meeting` — 会议（含状态机：草稿/进行中/已结束）
- `SpeakerSegment` — 发言片段（含来源：AprilTag/手动/追踪）
- `Event` — 系统事件流水
- `HostNote` — 主持人备注
- `SystemConfig` — 系统配置（含标定参数）
- `AgentDecision` — Agent 决策记录
- `TTSEvent` — TTS 播报事件
- `SchemaVersion` — 数据库迁移版本记录

`EventBridge` 以异步后台线程写入，有界队列（max 5000）每 1 秒或 100 条事件批量刷盘，避免磁盘 I/O 阻塞主循环。
