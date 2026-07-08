// yolo_engine.cc — YoloInferEngine 实现 + 绘制函数
//
// 绘制样式对齐 Python func_yolov8_optimize.py::draw (lines 232-243):
//   - 紫红色主框 (255, 0, 255)
//   - 绿色四角加粗 (corner length=15)
//   - 紫红色填充标签背景 + 黑色文字
//
// 引擎封装 RknnPool: submit_frame → get_result (含绘制) → compute_deviation。
#include "yolo_engine.h"

#include <chrono>
#include <cstdio>
#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "post_process.h"

namespace yolo {

// ── COCO 80 类标签 (与 yolo_camera.cc / func_yolov8_optimize.py 一致) ──
static const char* kCocoLabels[OBJ_CLASS_NUM] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

// ── 绘制: 四角加粗 (对齐 Python draw_box_corner) ────────────────
static void draw_box_corner(cv::Mat& img, int left, int top, int right,
                            int bottom, int length,
                            const cv::Scalar& corner_color) {
    std::vector<std::vector<cv::Point>> corners(4);
    corners[0] = {{left, top}, {left + length, top}, {left, top + length}};
    corners[1] = {{right, top}, {right - length, top}, {right, top + length}};
    corners[2] = {{left, bottom}, {left + length, bottom}, {left, bottom - length}};
    corners[3] = {{right, bottom}, {right - length, bottom}, {right, bottom - length}};
    for (auto& c : corners) {
        cv::polylines(img, c, true, corner_color, 3);
    }
}

// ── 绘制: 标签背景 (对齐 Python draw_label_type) ───────────────
static void draw_label_type(cv::Mat& img, int left, int top,
                            const std::string& label,
                            const cv::Scalar& label_color) {
    int base = 0;
    cv::Size txt = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
    int x2 = left + txt.width;
    int y2, text_y;
    if (top - txt.height - 3 < 0) {
        y2 = top + txt.height + 8;
        text_y = top + txt.height + 5;
    } else {
        y2 = top - 3;
        text_y = top - 5;
    }
    cv::rectangle(img, cv::Point(left, y2 - txt.height - 3),
                  cv::Point(x2, y2), label_color, cv::FILLED);
    cv::putText(img, label, cv::Point(left, text_y),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
}

// ── 绘制主入口: 紫红框 + 绿色四角 + 标签 (对齐 Python draw) ─────
static void draw_results(cv::Mat& img, const DetectResult& res,
                         const std::vector<std::string>& labels) {
    const cv::Scalar magenta(255, 0, 255);
    const cv::Scalar green(0, 255, 0);
    for (int i = 0; i < res.count; i++) {
        const DetectBox& d = res.boxes[i];
        int l = (int)d.left, t = (int)d.top, r = (int)d.right, b = (int)d.bottom;
        cv::rectangle(img, cv::Point(l, t), cv::Point(r, b), magenta, 2);
        draw_box_corner(img, l, t, r, b, 15, green);
        std::string name = (d.cls_id >= 0 && d.cls_id < (int)labels.size())
                           ? labels[d.cls_id] : std::to_string(d.cls_id);
        char buf[160];
        snprintf(buf, sizeof(buf), "%s %.2f", name.c_str(), d.confidence);
        draw_label_type(img, l, t, buf, magenta);
    }
}

// ============================================================
// YoloInferEngine
// ============================================================

YoloInferEngine::YoloInferEngine(const std::string& model_path, int tpes,
                                 float obj_thresh, float nms_thresh)
    : model_path_(model_path),
      tpes_(tpes > 0 ? tpes : 1),
      obj_thresh_(obj_thresh),
      nms_thresh_(nms_thresh) {}

YoloInferEngine::~YoloInferEngine() { release(); }

bool YoloInferEngine::init() {
    // 加载 COCO 标签
    labels_.clear();
    labels_.reserve(OBJ_CLASS_NUM);
    for (int i = 0; i < OBJ_CLASS_NUM; i++) labels_.emplace_back(kCocoLabels[i]);

    // 构造并初始化 pool
    pool_ = std::make_unique<RknnPool>(model_path_, tpes_, obj_thresh_, nms_thresh_);
    if (!pool_->init()) {
        printf("[YoloInferEngine] RknnPool init failed\n");
        pool_.reset();
        return false;
    }
    return true;
}

int YoloInferEngine::submit_frame(const cv::Mat& bgr) {
    if (!pool_) return -1;
    return pool_->submit(bgr);
}

bool YoloInferEngine::get_result(cv::Mat& annotated, DetectResult& boxes,
                                 TimingMs& timing, int timeout_ms) {
    if (!pool_) return false;
    cv::Mat raw_annotated;
    if (!pool_->get(raw_annotated, boxes, timing, timeout_ms)) return false;
    // 先缩放到 ≤960, 再在小帧上 draw: draw_results 含 rectangle/polylines/
    // putText/getTextSize, 开销随帧面积增长; 在 960×540 上画比 1920×1080 快 ~4x。
    // boxes 坐标按原图尺寸返回给 Python (用于偏差计算), 仅 draw 时缩放。
    constexpr int kMaxDisplayWidth = 960;
    int w = raw_annotated.cols, h = raw_annotated.rows;
    cv::Mat canvas;
    float sx = 1.0f, sy = 1.0f;
    if (w > kMaxDisplayWidth) {
        double s = (double)kMaxDisplayWidth / w;
        cv::resize(raw_annotated, canvas,
                   cv::Size(kMaxDisplayWidth, (int)(h * s)),
                   0, 0, cv::INTER_LINEAR);
        sx = (float)s; sy = (float)s;
    } else {
        canvas = raw_annotated;
    }
    // 在 canvas 上绘制 (boxes 坐标缩放到 canvas 尺寸; 原始 boxes 不变)
    DetectResult scaled = boxes;
    for (int i = 0; i < scaled.count; i++) {
        scaled.boxes[i].left   *= sx;
        scaled.boxes[i].top    *= sy;
        scaled.boxes[i].right  *= sx;
        scaled.boxes[i].bottom *= sy;
    }
    draw_results(canvas, scaled, labels_);
    annotated = canvas;
    return true;
}

DeviationResult YoloInferEngine::compute_deviation(const cv::Mat& bgr) {
    DeviationResult out;
    if (!pool_) return out;

    DetectResult det;
    TimingMs timing;
    if (!pool_->infer_sync(0, bgr, det, timing)) return out;

    // 找最大 person 框 (cls_id == 0)
    float best_area = -1.0f;
    int   best_idx  = -1;
    for (int i = 0; i < det.count; i++) {
        if (det.boxes[i].cls_id != 0) continue;
        if (det.boxes[i].confidence < obj_thresh_) continue;
        float w = det.boxes[i].right - det.boxes[i].left;
        float h = det.boxes[i].bottom - det.boxes[i].top;
        float area = w * h;
        if (area > best_area) {
            best_area = area;
            best_idx  = i;
        }
    }
    if (best_idx < 0) return out;

    const DetectBox& b = det.boxes[best_idx];
    float cx = (b.left + b.right) * 0.5f;
    float cy = (b.top + b.bottom) * 0.5f;
    float img_cx = bgr.cols * 0.5f;
    float img_cy = bgr.rows * 0.5f;
    out.has_person = true;
    out.dev_x = (cx - img_cx) / img_cx;
    out.dev_y = (cy - img_cy) / img_cy;
    out.left   = b.left;
    out.top    = b.top;
    out.right  = b.right;
    out.bottom = b.bottom;
    return out;
}

void YoloInferEngine::release() {
    if (pool_) pool_->release();
    pool_.reset();
}

}  // namespace yolo
