// types.h — 公共数据结构 (yolo_core 库)
//
// 定义 DetectBox / DetectResult / TimingMs / DeviationResult 等数据契约,
// 供 C++ 推理引擎与 pybind11 绑定层共享。坐标一律使用 float (与 Python
// func_yolov8_optimize.py 的输出一致), 便于直接映射到 Python dict。
#pragma once

#include <cstdint>
#include <vector>

namespace yolo {

// ── 模型 / 推理常量 ────────────────────────────────────────────
constexpr int   OBJ_CLASS_NUM     = 80;     // COCO 80 类
constexpr int   OBJ_NUMB_MAX_SIZE = 128;    // 每帧最多保留目标数
constexpr int   MODEL_WIDTH       = 640;    // YOLOv8 输入宽
constexpr int   MODEL_HEIGHT      = 640;    // YOLOv8 输入高

// ── Letterbox 参数 ─────────────────────────────────────────────
struct LetterBox {
    int   x_pad = 0;   // 水平 padding (left)
    int   y_pad = 0;   // 垂直 padding (top)
    float scale = 1.0f; // 缩放因子
};

// ── 单个检测框 ─────────────────────────────────────────────────
// 注意: left/top/right/bottom 为原图像素坐标 (已反 letterbox)。
struct DetectBox {
    float left       = 0.0f;
    float top        = 0.0f;
    float right      = 0.0f;
    float bottom     = 0.0f;
    float confidence = 0.0f;
    int   cls_id     = 0;
};

// ── 一帧的检测结果 ─────────────────────────────────────────────
struct DetectResult {
    int                     count = 0;
    std::vector<DetectBox>  boxes;   // 长度 == count
};

// ── 各阶段耗时 (毫秒) ──────────────────────────────────────────
struct TimingMs {
    double preprocess  = 0.0;
    double inference   = 0.0;
    double postprocess = 0.0;
    double total       = 0.0;
};

// ── 人体偏差二次推理结果 ───────────────────────────────────────
// has_person=false 时其余字段无意义。
struct DeviationResult {
    bool  has_person = false;
    float dev_x      = 0.0f;   // [-1, 1], 正值表示人在画面右侧
    float dev_y      = 0.0f;   // [-1, 1], 正值表示人在画面下方
    float left       = 0.0f;
    float top        = 0.0f;
    float right      = 0.0f;
    float bottom     = 0.0f;
};

}  // namespace yolo
