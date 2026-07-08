# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Real-time YOLOv8 object detection demo running on Rockchip RK3588 NPU. Captures frames from a V4L2 camera, performs inference using a pooled multi-core RKNN backend, and emits annotated frames plus person-deviation data via Qt signals.

## Key Dependencies

- **RKNN runtime**: `rknnlite.api.RKNNLite` — Rockchip NPU API (RKNPU2 SDK). Installed system-wide on the target board.
- **rknnpool**: `rknnpool.rknnpool_ld.rknnPoolExecutor` — Thread-pooled RKNN executor that round-robins frames across multiple NPU core instances. Located at `../demo/rknnpool/rknnpool_ld.py` relative to this project.
- **func**: `func.func_yolov8_optimize.myFunc` — The inference callback invoked per-frame by the pool. Located at `../demo/func/` relative to this project. The copy at `func_yolov8_optimize.py` in this repo is the same code vendored locally.
- OpenCV (`cv2`), NumPy, PyQt5 — all on the target board via pip/system packages.

## Running

This project runs **on-device** (RK3588 board). It is not a standalone Python file — it depends on `rknnpool` and `func` modules imported from the parent `demo/` project.

```bash
# On the RK3588 board, from the parent demo/ directory:
python -c "from c_realtime_infer_demo.yolo_camera import YoloCameraThread; ..."
```

There is no standalone `python yolo_camera.py` entrypoint. The `YoloCameraThread` is a QThread component designed to be instantiated from a larger Qt application.

## Architecture

### Inference Pipeline

```
Camera (V4L2 /dev/video21, 1920×1080, MJPG)
  → rknnPoolExecutor.put(frame)   [round-robin across TPEs=4 NPU instances]
  → myFunc(rknn_lite, frame)       [BGR→RGB → letterbox(640×640) → inference → post-process → draw boxes]
  → rknnPoolExecutor.get()         [blocking get from completion queue]
  → annotated_frame emitted via frame_ready Qt signal (resized to ≤960px wide for display)
```

### NPU Core Pinning

On RK3588 (3 NPU cores), `initRKNN` in the rknnpool module cycles through cores:
- `TPEs % 3 == 0` → `NPU_CORE_0`
- `TPEs % 3 == 1` → `NPU_CORE_1`
- `TPEs % 3 == 2` → `NPU_CORE_2`

Each `TPE` is a separate `RKNNLite` instance with its own `ThreadPoolExecutor` worker.

### Person Deviation

After pool inference returns, a **second** inference runs synchronously on the calling thread using `_rknn_for_dev` (the pool's first NPU instance) to compute person bounding boxes, then finds the largest person and calculates their offset from the frame center as normalized `(dev_x, dev_y)`.

### YOLOv8 Post-Processing (`func_yolov8_optimize.py`)

- `letterbox()` — Resize/pad image to 640×640 while preserving aspect ratio, returning ratio and padding offsets for coordinate mapping.
- `dfl()` — Distribution Focal Loss: converts raw position output to bounding box coordinates using softmax-weighted regression values.
- `box_process()` — Combines grid/stride lookup with DFL to produce xyxy boxes.
- `yolov8_post_process()` — Processes 3-branch YOLOv8 output (boxes + class conf + scores), applies object threshold (`OBJ_THRESH=0.75`), per-class NMS (`NMS_THRESH=0.6`).
- `myFunc()` — Full per-frame callback: preprocess → inference → postprocess → draw.

### rknn_model_zoo-2.1.0/

Vendored reference code from Rockchip's RKNN Model Zoo v2.1.0. Not required for the demo to run. Contains C++ and Python examples for many models (YOLOv5/v6/v7/v8/v10, PPOCR, Whisper, CLIP, etc.). Build with `build-linux.sh` / `build-android.sh` for cross-compilation to target boards.

## Important Details

- **Model**: `best.rknn` — pre-compiled RKNN model binary. The code in `yolo_camera.py` references `./rknnModel/best.rknn`, but the model is at the project root as `best.rknn`. Path resolution depends on the working directory of the parent application.
- **Camera device**: `/dev/video21` with MJPG fourcc at 1920×1080. Requires V4L2 support.
- **Warmup**: The pool pre-loads `TPEs+1` frames before entering the main loop.
- **Stream queues**: `YoloCameraThread` supports zero-copy frame delivery to RTSP streaming and AprilTag detection consumers via `queue.Queue`, bypassing Qt signal overhead. Uses drain-on-put pattern (drops old frames, keeps only latest).
- **FPS reporting**: Updates every 15 frames.
