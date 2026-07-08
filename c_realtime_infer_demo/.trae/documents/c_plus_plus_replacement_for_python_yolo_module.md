# Plan: C++ PyBind11 Replacement for Python `YoloCameraThread` Module

## Summary

Replace the Python `demo/core/yolo_camera.py` `YoloCameraThread` internals with a C++ PyBind11 extension module that preserves the **full Qt-signal contract** (all 7 signals, both queues, person-deviation second pass, 4-way NPU pool). The C++ core reuses the RKNN/DFL/NMS/letterbox algorithms from `c_realtime_infer_demo/camera_infer/yolo_camera.cc` but is restructured as a **library** (not a `main()` CLI), extended with an NPU pool and a deviation pass, and bound to Python via pybind11. A thin Python `QThread` shell wraps the extension and emits the unchanged signals — `demo/demos/fusion_tracker.py` needs **no edits**.

- **Approach**: PyBind11 extension module (`.so`) + thin Python QThread shell.
- **Parity**: Full — all 7 signals, `stream_queue`, `tag_queue`, person-deviation second pass, `TPEs=4` NPU pool.
- **Params**: Match Python — `OBJ_THRESH=0.75`, `NMS_THRESH=0.6`, camera 1920×1080 MJPG, `MODEL_PATH=./rknnModel/best.rknn`.

---

## Current State Analysis

### C++ source (`c_realtime_infer_demo/camera_infer/yolo_camera.cc`)
- 807-line **standalone `main()` CLI** with `cv::imshow` loop. No library API, no header.
- Single RKNN context, single NPU core. No thread pool.
- Pipeline: V4L2 → preprocess (BGR→RGB + letterbox 640×640) → `rknn_run` → YOLOv8 post-process (DFL + per-class NMS) → draw → `imshow`.
- Algorithms to **reuse**: `init_rknn()` (lines 157-253), `compute_dfl()` (260-273), `process_branch_i8/fp32` (289-410), `nms_per_class()` (446-465), `preprocess()` letterbox (563-587).
- Defaults (lines 38-53): `BOX_THRESH=0.25`, `NMS_THRESH=0.45`, 1280×720@30, `./best.rknn` — **will be overridden** to match Python.
- COCO labels inlined (lines 56-70); identical content to Python.
- Build: `camera_infer/CMakeLists.txt` (3 targets: `yolo_camera`, `yolo_camera_fast`, `yolo_camera_hw`).

### Python target (`demo/core/yolo_camera.py`)
- 288-line `YoloCameraThread(QThread)` with 7 `pyqtSignal`s (lines 33-39):
  - `frame_ready(np.ndarray)` — annotated BGR frame ≤960 px wide.
  - `raw_frame_ready(np.ndarray)` — raw frame (only when no `stream_queue`).
  - `deviation_data(object)` — `(dev_x, dev_y)` normalized [-1,1] or `None`.
  - `person_box_ready(object)` — `{"left","top","right","bottom"}` or `None`.
  - `inference_timing(object)` — `{"preprocess_ms","inference_ms","postprocess_ms","total_ms"}`.
  - `fps_update(float)` — every 15 frames.
  - `status_msg(str)` — human-readable status.
- Constants (lines 41-47): `CAP_DEVICE="/dev/video21"`, `CAM_WIDTH=1920`, `CAM_HEIGHT=1080`, `FOURCC=MJPG`, `BUFFER_SIZE=1`, `TPEs=4`, `MODEL_PATH="./rknnModel/best.rknn"`.
- Public API: `__init__(stream_queue=None)`, `set_stream_queue(q)`, `set_tag_queue(q)`, `run()`, `stop()`.
- Internal pipeline uses `rknnpool/rknnpool_ld.py` (`rknnPoolExecutor`, 4 contexts round-robin NPU cores 0/1/2) and `func/func_yolov8_optimize.py` (`myFunc` callback: letterbox → inference → post-process → draw).
- **Person deviation** (lines 116-183): second synchronous inference on `rknnPool[0]` under `_dev_lock`, picks largest class-0 (person) box, computes `dev_x = (cx - W/2)/(W/2)`, `dev_y = (cy - H/2)/(H/2)`.
- **Drain-on-put** for `stream_queue` (960×540) and `tag_queue` (640×480): keep only latest frame.
- Thresholds (from `func_yolov8_optimize.py` line 5): `OBJ_THRESH=0.75`, `NMS_THRESH=0.6`, `IMG_SIZE=640`.

### Consumers
- `demo/demos/fusion_tracker.py` instantiates `YoloCameraThread` at line 334, wires 6 signals (335-340), later attaches `person_box_ready` (line 1099-1102), `set_tag_queue` (1191), `set_stream_queue` (1212). The replacement must preserve this call surface exactly.

### Functional gaps (C++ → Python parity)
| Gap | Resolution |
|---|---|
| `main()` CLI vs QThread library | Restructure as `YoloInferEngine` class; no `main()`. |
| Single NPU core vs 4-way pool | New `RknnPool` class: 4 contexts, core-pin via `rknn_set_core_mask`, `ThreadPoolExecutor`-style submit/get. |
| No person deviation | Add `YoloInferEngine::infer_deviation()` reusing the same post-process; returns `(dev_x, dev_y, person_box)` or `None`. |
| No Qt signals | Signals stay in Python shell; C++ returns plain structs. |
| No queues | Queues stay in Python shell. |
| No timing dict | Instrument C++ `infer()` with 4 `std::chrono` checkpoints; return `TimingMs` struct. |
| Thresholds 0.25/0.45 vs 0.75/0.6 | Make thresholds constructor params; Python shell passes `0.75`/`0.6`. |
| Camera 1280×720 vs 1920×1080 | Camera capture stays in Python (`cv2.VideoCapture`); C++ receives BGR frames. |

---

## Proposed Changes

### New C++ library + bindings: `c_realtime_infer_demo/yolo_core/`

A new directory alongside `camera_infer/`. Contains a CMake-built shared library `libyolo_core.so` plus a pybind11 module `yolo_core.so` importable from Python.

#### File: `yolo_core/include/types.h`
Common structs:
```cpp
struct DetectBox { float left, top, right, bottom, confidence; int cls_id; };
struct DetectResult { int count; std::vector<DetectBox> boxes; };
struct TimingMs { double preprocess, inference, postprocess, total; };
struct DeviationResult { bool has_person; float dev_x, dev_y; float left, top, right, bottom; };
```
- *Why*: clean C++ data contract; pybind11 maps to Python dicts/tuples.

#### File: `yolo_core/include/rknn_pool.h` + `src/rknn_pool.cc`
Class `RknnPool`:
- `RknnPool(const std::string& model_path, int tpes, int obj_class_num, int dfl_len)`
- Holds `std::vector<RKNNContext>` (the RAII wrapper from `yolo_camera.cc` lines 110-128, extracted).
- `init()`: for each of `tpes` instances, load model, query I/O attrs, warmup (1 fake pass). Core-pin instance `i` to `NPU_CORE_0/1/2` cyclically via `rknn_set_core_mask(RKNN_NPU_CORE_0_1_2)` + auto-round-robin (matches `rknnpool_ld.py` lines 7-27).
- `submit(cv::Mat bgr_frame) -> int`: pick `instance = next_id % tpes`, push `{frame, instance}` to internal `ThreadPoolExecutor`-style queue (use `std::thread` + `std::queue` + `std::mutex` + `std::condition_variable`, since C++17 has no built-in pool). Returns task id.
- `get(DetectResult& out, TimingMs& timing, cv::Mat& annotated) -> bool`: pop next completed result in FIFO order.
- `infer_sync(int instance_idx, cv::Mat bgr_frame, DetectResult& out, TimingMs& timing)`: synchronous inference on a specific instance — used by the deviation second pass under a `std::mutex` (mirrors Python's `_dev_lock`).
- *Why*: replaces `rknnpool/rknnpool_ld.py` + the per-frame `myFunc` callback. Reuses `init_rknn`/`preprocess`/`process_branch_*`/`nms_per_class` from `yolo_camera.cc` (lines 157-465), refactored to be instance-scoped.

#### File: `yolo_core/include/yolo_engine.h` + `src/yolo_engine.cc`
Class `YoloInferEngine`:
- `YoloInferEngine(std::string model_path, int tpes=4, float obj_thresh=0.75f, float nms_thresh=0.6f)`
- `init()`: construct `RknnPool`, load COCO labels (inlined, same as `yolo_camera.cc` lines 56-70).
- `submit_frame(cv::Mat bgr) -> int`: delegates to `RknnPool::submit`, draws boxes on a copy of the frame (port `draw_results` lines 592-618 but use **magenta + green corner accents** to match `func_yolov8_optimize.py::draw` lines 232-243).
- `get_result(cv::Mat& annotated, DetectResult& boxes, TimingMs& timing) -> bool`: delegates to `RknnPool::get`.
- `compute_deviation(cv::Mat bgr, DeviationResult& out)`: synchronous second pass on pool instance 0 under lock; picks largest class-0 box; computes `dev_x = (cx - W/2)/(W/2)`, `dev_y = (cy - H/2)/(H/2)`. Returns `has_person=false` if no class-0 detection above `obj_thresh`.
- *Why*: the high-level API the Python shell calls. Encapsulates pool + drawing + deviation so the Python side stays trivial.

#### File: `yolo_core/src/post_process.cc`
Standalone functions ported from `yolo_camera.cc`:
- `letterbox(cv::Mat src, cv::Mat& dst, float& scale, int& x_pad, int& y_pad, int target=640)` — from lines 563-587.
- `compute_dfl(float* src, int dfl_len)` — from lines 260-273. **Add max-subtraction** for numerical stability (matches Python `dfl()` lines 111-125).
- `process_branch_i8(...)` / `process_branch_fp32(...)` — from lines 289-410, parameterized by `obj_thresh`.
- `nms_per_class(...)` — from lines 446-465, parameterized by `nms_thresh`.
- *Why*: algorithmic core reused verbatim where possible; only threshold params differ.

#### File: `yolo_core/src/draw.cc`
`draw_results(cv::Mat& img, const DetectResult& result, const std::vector<std::string>& labels)` — port of `yolo_camera.cc` lines 592-618 with the **Python styling**: magenta boxes (`(255, 0, 255)`) + green corner accents + filled label background. Matches `func_yolov8_optimize.py::draw` lines 232-243.

#### File: `yolo_core/bindings/yolo_module.cc`
pybind11 module `yolo_core`:
- Bind `YoloInferEngine`, `RknnPool`, all structs.
- **`cv::Mat` ↔ `np.ndarray`**: use the `pybind11_opencv` header pattern (or the canonical `ndarray`-via-buffer-protocol helper). Convert BGR `cv::Mat` to a `np.ndarray` (contiguous, dtype=uint8) without copying when possible; accept `np.ndarray` inputs and wrap as `cv::Mat` (rows, cols, CV_8UC3).
- Struct conversions: `DetectBox` → Python `dict` with keys `left/top/right/bottom/confidence/cls_id`; `TimingMs` → dict; `DeviationResult` → dict or `None`.
- **Release GIL** during `submit_frame`, `get_result`, `compute_deviation` via `py::gil_scoped_release`.
- Module init: `PYBIND11_MODULE(yolo_core, m) { ... }`.
- *Why*: the bridge enabling Python to call C++ with zero behavioral change.

#### File: `yolo_core/CMakeLists.txt`
- `cmake_minimum_required(VERSION 3.14)`, `project(yolo_core LANGUAGES CXX)`, `set(CMAKE_CXX_STANDARD 17)`.
- `find_package(OpenCV REQUIRED)`, `find_package(Python3 COMPONENTS Interpreter Development REQUIRED)`, `find_package(pybind11 CONFIG REQUIRED)` (or `pybind11_add_module`).
- Auto-detect `librknnrt.so` (reuse logic from `camera_infer/CMakeLists.txt` lines 33-37).
- `pybind11_add_module(yolo_core bindings/yolo_module.cc)`.
- `target_sources(yolo_core PRIVATE src/yolo_engine.cc src/rknn_pool.cc src/post_process.cc src/draw.cc bindings/yolo_module.cc)`.
- `target_include_directories(yolo_core PRIVATE include ${OpenCV_INCLUDE_DIRS} ${RKNN_HEADER_DIR})`.
- `target_link_libraries(yolo_core PRIVATE ${OpenCV_LIBS} ${RKNN_LIB} pthread dl)`.
- Output: `yolo_core.cpython-3XX-aarch64-linux-gnu.so` in `build/`.
- Install target: copy `.so` to `demo/` (or add `build/` to `PYTHONPATH`).

#### File: `yolo_core/build.sh`
Convenience wrapper: `mkdir -p build && cd build && cmake .. && make -j4`. Mirrors `camera_infer/build.sh`.

### Modified Python files

#### File: `demo/core/yolo_camera.py` (modified in place)
- Keep the class signature, all 7 signals, all public methods (`__init__`, `set_stream_queue`, `set_tag_queue`, `run`, `stop`), all class constants.
- In `__init__`: replace `initRKNNs(self.MODEL_PATH, self.TPEs)` with `self._engine = yolo_core.YoloInferEngine(self.MODEL_PATH, self.TPEs, obj_thresh=0.75, nms_thresh=0.6); self._engine.init()`.
- Remove imports of `rknnpool.rknnpool_ld` and `func.func_yolov8_optimize` (keep them on disk for other consumers).
- In `run()` loop:
  - `cap.read()` unchanged.
  - `stream_queue` drain-and-put (960×540) **unchanged**.
  - `tag_queue` drain-and-put (640×480) **unchanged**.
  - Replace `self._pool.put(frame)` with `task_id = self._engine.submit_frame(frame_bgr)`.
  - Replace `self._pool.get()` with `self._engine.get_result(annotated, boxes, timing)` (GIL released inside C++).
  - Replace the synchronous deviation call with `self._engine.compute_deviation(frame, dev_result)` (GIL released).
  - Build `inference_timing` dict from the returned `TimingMs`.
  - Build `person_box` dict / `None` from `DeviationResult`.
  - Resize annotated frame to ≤960 px wide, emit `frame_ready`.
  - FPS counter (every 15 frames) unchanged.
- **No changes** to `demos/fusion_tracker.py`.

#### File: `demo/requirements.txt` or `demo/CLAUDE.md` (docs only)
- Note that `yolo_core` extension must be built and on `PYTHONPATH` before running `fusion_tracker.py`.
- Note that `rknnpool`/`func_yolov8_optimize` are no longer used by the camera path.

### Untouched files (explicitly)
- `c_realtime_infer_demo/camera_infer/yolo_camera.cc` — kept as reference; not modified.
- `demo/demos/fusion_tracker.py` — no edits (drop-in guarantee).
- `demo/func/func_yolov8_optimize.py`, `demo/rknnpool/rknnpool_ld.py` — kept on disk (may be used elsewhere).

---

## Architecture Diagram (data flow)

```
┌──────────────────────────── demo/demos/fusion_tracker.py (UNCHANGED) ────────────────────────────┐
│   YoloCameraThread ──signals──> PyQt5 main window                                                │
└─────────────────────────────────────┬─────────────────────────────────────────────────────────────┘
                                      │  (Qt signals: 7, unchanged)
┌─────────────────────────────────────▼─────────────────────────────────────────────────────────────┐
│  demo/core/yolo_camera.py  (THIN SHELL — QThread + queue mgmt + signal emission)                  │
│    cv2.VideoCapture(1920×1080 MJPG) → BGR frame                                                   │
│    stream_queue (drain-and-put 960×540)    tag_queue (drain-and-put 640×480)                      │
│    self._engine.submit_frame(bgr)  →  self._engine.get_result(annotated, boxes, timing)           │
│    self._engine.compute_deviation(bgr, &dev_result)                                               │
│    emit frame_ready / deviation_data / person_box_ready / inference_timing / fps_update / status  │
└─────────────────────────────────────┬─────────────────────────────────────────────────────────────┘
                                      │  (pybind11 call, GIL released during NPU work)
┌─────────────────────────────────────▼─────────────────────────────────────────────────────────────┐
│  c_realtime_infer_demo/yolo_core/  (C++ SHARED LIB + pybind11 module: yolo_core.so)               │
│  ┌───────────────────────── YoloInferEngine ─────────────────────────┐                            │
│  │  RknnPool (4× RKNNContext, core-pinned 0/1/2 round-robin)         │                            │
│  │    submit() ──> ThreadPool (std::thread+queue+mutex+cv)           │                            │
│  │    get()    <── FIFO result queue                                  │                           │
│  │    infer_sync(0, ...) under _dev_lock  (deviation second pass)    │                            │
│  │  post_process: letterbox + DFL + process_branch_i8/fp32 + NMS     │                            │
│  │  draw: magenta boxes + green corners + label bg                    │                           │
│  │  timing: 4× std::chrono checkpoints                                │                           │
│  └───────────────────────────────────────────────────────────────────┘                            │
│  Links: librknnrt.so, OpenCV, pthread, dl                                                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Assumptions & Decisions

1. **Target platform**: aarch64 Linux (Rockchip RK3588), Python 3.x, OpenCV 4.x, `librknnrt.so` already installed at `/usr/lib` or `/usr/lib/aarch64-linux-gnu`. Verified by `camera_infer/CMakeLists.txt` lines 33-37.
2. **pybind11 availability**: assumed installable via `pip install pybind11` (or system package). CMake uses `find_package(pybind11 CONFIG)`.
3. **OpenCV ↔ numpy**: implemented via the buffer-protocol helper (no external `pybind11_opencv` package dependency). BGR contiguous uint8 arrays only.
4. **NPU core affinity**: `rknn_set_core_mask` with `RKNN_NPU_CORE_0_1_2` allows the runtime to auto-assign; the C++ pool additionally round-robins by selecting instance index, matching the Python behavior. If `rknn_set_core_mask` is unavailable, fall back to default core assignment (still 4 contexts).
5. **Model file**: single shared `demo/rknnModel/best.rknn`. The C++ engine opens it 4 times (once per pool instance), like Python `initRKNNs` does.
6. **Thresholds**: passed as constructor args from Python (`0.75`, `0.6`); the C++ source's `0.25`/`0.45` defaults are **not** used.
7. **Camera capture**: stays in Python (`cv2.VideoCapture`). C++ never touches V4L2 directly. This keeps the camera-retry logic (10 failures → exit) and `BUFFER_SIZE=1` behavior identical.
8. **Drawing style**: matches Python (`func_yolov8_optimize.py::draw`) — magenta + green corners — to keep `frame_ready` output visually consistent.
9. **GIL**: released during NPU inference and post-processing; reacquired for the return value marshalling. The QThread's event loop is not blocked.
10. **No new dependencies** added to `demos/fusion_tracker.py` — it still imports only `core.yolo_camera`.
11. **Drop-in guarantee**: `fusion_tracker.py` lines 334-340, 1099-1102, 1191, 1212 remain valid without edits.

---

## Verification Steps

1. **Build the extension**:
   ```bash
   cd c_realtime_infer_demo/yolo_core && bash build.sh
   ```
   Expect `build/yolo_core.cpython-3XX-aarch64-linux-gnu.so`.

2. **Smoke test the module** (Python):
   ```python
   import yolo_core, cv2
   e = yolo_core.YoloInferEngine("./rknnModel/best.rknn", 4, 0.75, 0.6)
   e.init()
   f = cv2.imread("test.jpg")
   tid = e.submit_frame(f)
   annotated, boxes, timing = e.get_result()
   dev = e.compute_deviation(f)
   print(timing, dev)
   ```
   Expect: annotated frame has magenta boxes, `timing` dict has 4 keys in ms, `dev` is a dict or `None`.

3. **Unit-test the deviation logic**: feed a frame with a known person box; verify `dev_x`, `dev_y` match the Python formula `(cx - W/2)/(W/2)`, `(cy - H/2)/(H/2)`.

4. **Drop-in integration test**: build the extension, set `PYTHONPATH` to include it, run `python demo/demos/fusion_tracker.py`. Verify:
   - GUI shows live annotated camera feed.
   - `fps_update` updates the status bar.
   - `inference_timing` panel shows non-zero ms.
   - Person deviation values change as a person moves in frame.
   - RTSP stream (if configured) shows the 960×540 annotated frames.
   - AprilTag worker still receives 640×480 frames from `tag_queue`.

5. **Parity comparison**: run the Python-only `YoloCameraThread` and the C++-backed version on the same recorded video (replay via `cv2.VideoCapture("test.mp4")`); compare `frame_ready` outputs frame-by-frame. Box coordinates should match within ±1 px; timing should be strictly lower for the C++ version.

6. **Stability**: run for 10 minutes; no NPU core leaks, no segfaults, FPS stable.

7. **No-regression**: `git diff demo/demos/fusion_tracker.py` is empty.

---

## Implementation Order (recommended)

1. Create `yolo_core/` skeleton: `CMakeLists.txt`, `build.sh`, empty source files.
2. Port `types.h`, `post_process.cc` (letterbox/DFL/branch/NMS) from `yolo_camera.cc`.
3. Port `RKNNContext` + `RknnPool` (extract from `yolo_camera.cc` lines 110-253), add thread pool + core pinning.
4. Implement `YoloInferEngine` (init, submit, get, compute_deviation, draw).
5. Write `bindings/yolo_module.cc` (pybind11 + numpy/ Mat interop + GIL release).
6. Build and run the smoke test (step 2 above).
7. Modify `demo/core/yolo_camera.py` to use the extension.
8. Run the integration test (step 4).
9. Run parity + stability tests (steps 5-6).
