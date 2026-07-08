# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **smart meeting tracking system** running on a Rockchip ARM Linux board (RK3588) with NPU:

- **Camera + RKNN YOLO** → person detection + visual tracking
- **ReSpeaker USB mic array** → Direction of Arrival (DOA) for sound source localization
- **Dual-axis PWM servo** → pan-tilt camera mount, driven by a 3-state audio-visual fusion state machine
- **AprilTag** → per-speaker identity resolution with debounce
- **GStreamer RTSP** → video streaming with nameplate overlay
- **Streamlit web console** → pre-meeting prep + live host control via HTTP API

The primary entry point is `demos/fusion_tracker.py` (PyQt5 GUI). Requires `root` for PWM sysfs access.

## Architecture

```
Hardware Layer               Processing Layer           Service Layer          Web Layer
┌──────────────────┐    ┌─────────────────────┐    ┌─────────────────┐    ┌──────────────┐
│ PWM servo (H+V)  │    │ core/fusion_engine   │    │ core/control_api │◄───│ Streamlit    │
│ ReSpeaker USB    │    │ core/speaker_identity│    │ (HTTP :8800)    │    │ (:8501)      │
│ Camera V4L2      │◄───│ core/audio_vad        │    │ core/meeting_    │    │ - 会前准备    │
│ ALSA audio       │    │ core/yolo_camera      │    │   service        │    │ - 会议控制台  │
│ NPU (RKNN)       │    │ core/apriltag_camera  │    └─────────────────┘    └──────────────┘
└──────────────────┘    │ core/stream_publisher  │
                        └─────────────────────┘
                                  │
                        ┌─────────────────────┐
                        │ core/event_bus       │──► storage/event_bridge ──► SQLite
                        │ storage/ (ORM)       │
                        └─────────────────────┘
```

**Design principle**: `demos/fusion_tracker.py` is the ONLY hardware controller. Streamlit is a remote control panel communicating via HTTP API (`127.0.0.1:8800`). All hardware commands flow through `queue.Queue` → main Qt thread → hardware.

## Key Dependencies

| Dependency | Purpose |
|---|---|
| `rknnlite` (Rockchip SDK `.whl`) | NPU inference runtime |
| `opencv-python` (cv2) | Camera capture, video rendering |
| `PyQt5` | Main tracker GUI (`demos/fusion_tracker.py`, `calibrate_gui.py`) |
| `torch` ≥ 2.0 | Silero VAD CPU inference (`models/silero_vad.jit`) |
| `pyusb` | ReSpeaker USB vendor control transfers |
| `sqlalchemy` ≥ 2.0 | ORM for SQLite persistence (`storage/`) |
| `streamlit`, `pandas`, `openpyxl`, `Pillow`, `reportlab` | Web console + desk card generation |
| `numpy`, `scipy` | Math, audio resampling (fallback) |
| `pyzmq` | Legacy: inter-process pub/sub (not used in current main path) |

## Directory Layout

```
demo/
├── core/                          # Reusable modules (22 files)
│   ├── fusion_engine.py           # ★ Pure-Python 3-state audio→visual fusion
│   ├── speaker_identity.py        # AprilTag→speaker debounce state machine
│   ├── control_api.py             # HTTP API server (stdlib, ThreadingMixIn)
│   ├── meeting_service.py         # Meeting lifecycle business logic
│   ├── yolo_camera.py             # Camera+RKNN YOLO inference (QThread)
│   ├── apriltag_camera.py         # AprilTag detection (QThread)
│   ├── tag_detect_worker.py       # Tag detection worker (QThread, no Qt deps)
│   ├── stream_publisher.py        # GStreamer RTSP server (QThread)
│   ├── pwm_controller.py          # PWM servo via Linux sysfs
│   ├── respeaker.py               # ReSpeaker USB driver (QThread)
│   ├── alsa_capture.py            # ALSA PCM audio capture (ctypes, zero deps)
│   ├── silero_vad.py              # Silero VAD model + PyTorch log monkey-patch
│   ├── audio_vad.py               # AudioVadThread (ALSA + Silero VAD)
│   ├── calibration.py             # XVF→servo linear fit model
│   ├── event_bus.py               # Lightweight pub/sub (singleton)
│   ├── overlay_renderer.py        # RTSP nameplate overlay renderer
│   ├── metrics_collector.py       # Tracking metrics + CSV export
│   ├── desk_card_generator.py     # Desk card PNG/PDF generation
│   ├── display_env.py             # X11 env + cv2→PyQt5 Qt plugin conflict fix
│   └── network_utils.py           # get_device_ip()
├── storage/                       # SQLite persistence layer
│   ├── db.py                      # Engine, WAL pragmas, scoped_session
│   ├── models.py                  # 6 ORM models (Participant, Meeting, etc.)
│   ├── repo.py                    # Repository/DAO classes (DI pattern)
│   ├── event_bridge.py            # EventBus→DB async persistence
│   └── migration.py               # Minimal migration framework
├── app_streamlit/                  # Web management console
│   ├── Home.py                    # Dashboard entry
│   └── pages/
│       ├── 01_会前准备.py          # 5-step pre-meeting wizard
│       └── 02_会议控制台.py        # Live host control console
├── demos/                         # Refactored entry points
│   ├── fusion_tracker.py          # ★ Main app (~1500 lines PyQt5)
│   └── apriltag_detector.py       # AprilTag demo
├── calibrate_gui.py               # XVF-Servo calibration GUI (PyQt5, DB+JSON persistence)
├── scripts/                       # Startup & systemd scripts
│   ├── start_all.sh / stop_all.sh
│   ├── install_services.sh
│   └── smart-meeting-*.service
├── func/                          # YOLO postprocessing (numpy port)
├── rknnpool/                      # RKNN inference thread pool
├── rknnModel/                     # RKNN model files (.rknn)
├── models/                        # silero_vad.jit
├── configs/                       # desk_card_config.json
├── data/                          # Runtime DB + logs (gitignored)
├── exports/desk_cards/            # Generated PNGs + PDF (gitignored)
└── tagStandard41h12/              # Pre-generated AprilTag library (2117 files)
```

**Legacy files** (preserved for reference, do not modify):
- `fusion_tracker_demo.py` — original monolithic script
- `apriltag_detector_demo.py` — original monolithic script
- `visual_tracker_demo.py` — visual-only tracker (no audio)
- `sound_source_tracker.py` — audio-only tracker (no camera)

## Getting Started

### Prerequisites

1. Complete XVF→servo calibration: `PYTHONPATH=/home/elf/.local/lib/python3.10/site-packages sudo -E python3 calibrate_gui.py` → save (DB + `xvf_calibration.json`)
2. Ensure `root` access for PWM; `DISPLAY` + `XAUTHORITY` for PyQt5 GUI

### Start / Stop Services

```bash
# One-click
sudo ./scripts/start_all.sh       # Start fusion_tracker + Streamlit
sudo ./scripts/stop_all.sh         # Stop both

# Or via systemd (install once: sudo ./scripts/install_services.sh)
sudo systemctl start smart-meeting-runtime smart-meeting-streamlit
sudo systemctl stop smart-meeting-runtime smart-meeting-streamlit

# Enable auto-start on boot
sudo systemctl enable smart-meeting-runtime smart-meeting-streamlit
```

### Verify

```bash
curl http://127.0.0.1:8800/api/status | python3 -m json.tool
# Streamlit: http://<board-ip>:8501
```

### View Logs

```bash
journalctl -u smart-meeting-runtime -f
tail -f data/fusion_tracker.log
tail -f data/streamlit.log
```

## Key Modules

### Hardware I/O

- **`core/pwm_controller.py`** — PWM servo via sysfs. Configurable angle range, duty mapping, chip/index. Supports dual-axis (pan H: `pwmchip0`, tilt V: `pwmchip1` with inverted duty).
- **`core/respeaker.py`** — ReSpeaker USB mic array (VID:0x2886, PID:0x001A). Polls `DOA_VALUE` register via pyusb control transfers. Returns `(doa_angle, vad_flag)`. DOA is **negated** to align with servo coordinate system.
- **`core/alsa_capture.py`** — ALSA PCM capture via ctypes (zero deps). Used by Silero VAD for NAU8822 on-board codec at 16kHz S16_LE mono.
- **`core/yolo_camera.py`** — Camera + RKNN YOLO inference in a QThread. Emits `frame_ready`, `deviation_data`, `person_box_ready` signals.
- **`core/apriltag_camera.py`** — AprilTag detection QThread. Feeds tag list to `SpeakerIdentifier`.

### Processing

- **`core/fusion_engine.py`** — 3-state audio→visual fusion state machine (IDLE → AWAIT → TRACKING). Zero PyQt5. Audio for coarse acquisition; pure visual proportional control for tracking. Parameters persisted to `fusion_params.json`. See source for state transition details.
- **`core/speaker_identity.py`** — 5-state debounce machine (`UNKNOWN → CANDIDATE → CONFIRMED → LOST → MANUAL`). Associates AprilTag detections to participant identities via spatial scoring.
- **`core/audio_vad.py`** — `AudioVadThread(QThread)`: captures ALSA audio → Silero VAD inference → emits `silero_speech` signal. Falls back to ReSpeaker hardware VAD if model unavailable.
- **`core/silero_vad.py`** — Silero VAD model wrapper. **Includes a monkey-patch for PyTorch 2.x `Logger.setLevel` bug** — must be imported before torch.

### Infrastructure

- **`core/event_bus.py`** — Singleton pub/sub with prefix-matching wildcards. Core events: `state_changed`, `servo_moved`, `speaker_*`, `meeting_*`.
- **`core/stream_publisher.py`** — GStreamer RTSP server on port 8554. Accepts overlay callback for nameplate rendering.
- **`core/overlay_renderer.py`** — Lower-third name bar overlay (`姓名 · 角色 | 发言时长`) rendered onto video frames.
- **`core/display_env.py`** — `fix_display_env()` + `fix_cv2_qt_conflict()`. **Critical**: cv2 sets `QT_PLUGIN_PATH` to incompatible Qt plugins — this fix must run after `import cv2` and before `from PyQt5 import ...`.
- **`core/network_utils.py`** — `get_device_ip()`.

### Storage (`storage/`)

- **SQLite WAL mode** + QueuePool (pool_size=5) + `scoped_session` per thread.
- **ORM models**: `Participant`, `Meeting`, `SpeakerSegment`, `Event`, `HostNote`, `SystemConfig`.
- **`session_scope()`** — preferred context manager for DB access. Handles commit/rollback + session cleanup.
- **`EventBridge`** — async background writer: bounded queue (max 5000), flushes every 1s or 100 events. Prevents disk I/O from blocking the 100ms tick.
- **`SystemConfig` calibration section** — `calibrate_gui.py` writes calibration parameters (points, slope, intercept, R²) to `system_config` table under `config_section='calibration'`. On load, DB is preferred; JSON (`xvf_calibration.json`) serves as fallback. If JSON has data and DB is empty, auto-migration runs.

```python
from storage import init
init()  # Once at startup

from storage.db import session_scope
from storage.repo import MeetingRepo

with session_scope() as session:
    meeting = MeetingRepo(session).create("项目路演")
```

### Services

- **`core/control_api.py`** — HTTP API server on `127.0.0.1:8800`. Thread safety: `threading.Lock` for state snapshots; `queue.Queue` for commands; direct `session_scope()` for read-only DB queries.
- **`core/meeting_service.py`** — Single-active-meeting enforcement; auto-starts tracking on meeting start; publishes events to EventBus.

## API Reference

All endpoints relative to `http://127.0.0.1:8800`.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Full system state snapshot |
| POST | `/api/meeting/start` | `{"meeting_id": N}` |
| POST | `/api/meeting/end` | End current meeting |
| POST | `/api/meeting/pause` | Pause tracking |
| POST | `/api/meeting/resume` | Resume tracking |
| POST | `/api/control/recenter` | Center servo to 0° |
| POST | `/api/control/start_tracking` | Start auto tracking |
| POST | `/api/control/stop_tracking` | Stop auto tracking |
| POST | `/api/control/lock_speaker` | Freeze current speaker identity |
| POST | `/api/control/unlock_speaker` | Resume auto-identification |
| POST | `/api/control/manual_speaker` | `{"tag_id": "A001"}` |
| POST | `/api/control/set_overlay` | `{"enabled": bool, "show_debug": bool}` |
| POST | `/api/control/start_stream` | Start RTSP push |
| POST | `/api/control/stop_stream` | Stop RTSP push |
| GET | `/api/events?meeting_id=N&minutes=M` | Event timeline + host notes |
| POST | `/api/host_note` | `{"meeting_id", "note_type", "content", "related_speaker"}` |

## Hardware Notes

### PWM Servo

- Chip 0 (pan H): `[-135°, 135°]`, 15% duty = 0° center, 5% = right, 25% = left
- Chip 1 (tilt V): `[-90°, 90°]`, **inverted**: 5% = min (-90°), 25% = max (+90°)
- Period: 10,000,000 ns (100 Hz). Requires `root`.

### ReSpeaker DOA

- `doa_angle`: 0–359°, **negated** on read (`angle = -result[0]`)
- XVF estimator produces a slow ramp (0→target over ~0.5s). The AWAIT state in `fusion_engine.py` waits for this ramp to settle.
- **Motor noise feedback loop**: servo noise → mic pickup → false DOA reading. All trackers implement a motor quiet period after each move.

### Camera

- `/dev/video21` (RKISP mainpath), 1920×1080 MJPG V4L2

### NPU

- RKNN pool (`rknnpool/rknnpool_ld.py`): round-robin across cores 0/1/2, one core per instance
- YOLOv8 model: `rknnModel/best.rknn`, postprocessing in `func/func_yolov8_optimize.py`
- Only class 0 (person) used. Largest bbox by area selected.

## Gotchas & Conventions

1. **OpenCV + PyQt5 conflict**: Always call `fix_display_env()` before imports, `fix_cv2_qt_conflict()` after `import cv2` and before `from PyQt5 import ...`. See `core/display_env.py`.

2. **DOA angle negation**: `angle = -raw_doa` — required because mic and servo coordinate systems are inverted.

3. **YOLOv8 box format**: `[left, top, right, bottom]` — NOT `top, left, right, bottom`.

4. **QThread parent**: Create with `QThread()` (no parent) — `QThread(parent=self)` triggers "Cannot move to target thread" warning.

5. **Import pattern for `demos/`**: Both demos prepend project root to `sys.path`:
   ```python
   _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
   if _PROJECT_ROOT not in sys.path:
       sys.path.insert(0, _PROJECT_ROOT)
   ```

6. **Timer always running**: `_track_timer` starts in the constructor and never stops — ensures API command processing runs even when tracking is inactive (avoids Streamlit deadlock).

7. **`session_scope()` over `get_session()`**: Always prefer `session_scope()` for safe connection return to QueuePool. Raw `get_session()` + manual `s.close()` leaks sessions under Streamlit reruns.

8. **Parameter persistence**: `fusion_params.json` auto-saved on window close, auto-loaded on startup. `SystemConfig` table is the canonical store but backward-compat with JSON is maintained. Calibration data (`xvf_calibration.json` + `SystemConfig` section `"calibration"`) uses dual persistence: DB preferred on read, both written on save. See `calibrate_gui.py:CalibrationStorage`.

9. **Backward compatibility**: Original monolithic scripts (`fusion_tracker_demo.py`, `apriltag_detector_demo.py`) are preserved unchanged. New development targets `core/` + `demos/`.

10. **Log robustness**: `_log()` uses `getattr(self, '_log_text', None)` fallback → `print()` — safe before `_build_ui()` creates the log widget.

11. XVF3800 必须接在 USB3.0 (蓝色)接口：防止和视频争抢带宽

12. **QThread 生命周期 — 必须保持 Python 引用**: PyQt5 中 `QThread` 的 Python wrapper 一旦失去所有引用就会被 GC 回收，触发 C++ 析构。若线程仍在运行，Qt 会打印 `"QThread: Destroyed while thread is still running"` 并调用 `abort()` → **SIGABRT**。`worker.finished.connect(worker.deleteLater)` 的信号连接不足以保证 Python 侧存活。正确做法：将 worker 存入 `self._workers: list`，在 `finished` 信号中移除。参见 `core/tts_engine.py:TTSEngine._workers`。

13. **DuplexController 构造函数 — 必须使用关键字参数**: `DuplexController.__init__(self, cooldown_ms: int = 500, parent=None)` — 第一个位置参数是 `cooldown_ms`，不是 `parent`。误写 `DuplexController(widget)` 会把 QWidget 传为 `cooldown_ms`，导致 `finish_speaking()` 中 `QTimer.start(widget)` 抛出 `TypeError`，此时 `_transition_to(COOLDOWN)` 已执行但定时器未启动，状态永久卡死无法退出。正确：`DuplexController(parent=self)`。`__init__` 内有 `isinstance(cooldown_ms, QObject)` 防御性检查。

14. **SIGTERM / Qt 事件循环 — 必须安装信号处理器**: Python 默认的 `SIGTERM` 处理器尝试抛出 `SystemExit`，但此异常无法穿透 Qt C++ 事件循环 (`app.exec_()`)。systemd 发送 SIGTERM 后进程无响应，只能等待 `TimeoutStopSec` 超时后 SIGKILL 强杀 → PWM 舵机不归位、DB 不刷新、CSV 不导出。fix: 在 `main()` 中 `signal.signal(signal.SIGTERM, handler)` 调用 `QApplication.quit()`。参见 `demos/fusion_tracker.py:_signal_handler`。

15. **root 用户 + PulseAudio → 静默无声**: `fusion_tracker.py` 以 root 运行（PWM 需要），但 PulseAudio daemon 属于用户 `elf` (uid 1000)，root 无权访问。ALSA "default" 设备路由到 PulseAudio → 连接被拒 → 音频数据被**静默丢弃**，pygame 和 aplay 返回 rc=0 但实际无声。**必须绕过 PulseAudio**：
   - pygame: 在 `import pygame` 之前设置 `SDL_AUDIODRIVER=alsa` + `AUDIODEV=plughw:1,0`（NAU8822 板载 codec）
   - aplay: 使用 `aplay -D plughw:1,0` 而非 `aplay default`
   - 注意 `hw:1,0` 会因 VAD 采集占用而报 "Device or resource busy"，必须用 `plughw:1,0`
   - `tts_demo.py` 以普通用户运行所以用 `default` 正常，不可照搬。参见 `core/audio_player.py:_detect_device()` 和 `init()`

16. **pygame 播放：文件路径 优于 内存 buffer**: 在 ARM/SDL2 上，`pygame.mixer.Sound(filepath)`（从文件加载，调用 `Mix_LoadWAV`）比 `pygame.mixer.Sound(buffer=bytes)`（从内存加载，调用 `Mix_LoadWAV_RW`）更可靠。`tts_demo.py` 先写 WAV 到临时文件再播放，`audio_player.py` 也采用同样策略。不要使用 `pygame.mixer.init(buffer=1024)` — 1024 samples 在 RK3588 NAU8822 上太小（64ms），用 SDL2 默认值（4096 = 256ms）。

## Development Workflow

```bash
# After code changes, restart services
sudo systemctl restart smart-meeting-runtime smart-meeting-streamlit

# Check service health
sudo systemctl status smart-meeting-runtime smart-meeting-streamlit

# Watch logs
journalctl -u smart-meeting-runtime -f

# Test DB (headless)
python3 test_storage.py
```
