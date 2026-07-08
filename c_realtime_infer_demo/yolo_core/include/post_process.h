// post_process.h — YOLOv8 后处理函数声明
//
// 从 c_realtime_infer_demo/camera_infer/yolo_camera.cc 移植:
//   - letterbox:        BGR → RGB letterbox 640×640 (lines 563-587)
//   - compute_dfl:      DFL 解码 (lines 260-273), 增加 max 减法提升数值稳定性
//   - process_branch_*: 单检测头解析 (i8 量化 / fp32, lines 289-410)
//   - nms_per_class:    按类别 NMS (lines 446-465)
//   - yolov8_post_process: 完整后处理入口 (lines 468-558)
//
// 阈值通过参数传入 (不再用全局常量), 与 Python 侧 0.75 / 0.6 对齐。
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include "types.h"
#include "rknn_context.h"

namespace yolo {

// BGR 摄像头帧 → RGB letterbox (target_w × target_h, 默认 640×640)。
// 输出 rgb_letterbox 为连续 CV_8UC3, 可直接喂给 RKNN。
void letterbox(const cv::Mat& bgr, cv::Mat& rgb_letterbox, LetterBox& lb,
               int target_w = MODEL_WIDTH, int target_h = MODEL_HEIGHT);

// Distribution Focal Loss: 把 reg_max 个分布值 → 4 个标量 box 偏移。
// 输入 tensor 长度 = dfl_len * 4, 输出 box_out 长度 = 4。
// 与 Python dfl() (func_yolov8_optimize.py:111-125) 一致, 减去 max 提升稳定性。
void compute_dfl(const float* tensor, int dfl_len, float* box_out);

// INT8 反量化
inline float deqnt_i8_to_f32(int8_t q, int32_t zp, float scale) {
    return ((float)q - (float)zp) * scale;
}

// FP32 → INT8 量化
inline int8_t qnt_f32_to_i8(float f32, int32_t zp, float scale) {
    float dst = f32 / scale + (float)zp;
    if (dst <= -128.f) return -128;
    if (dst >= 127.f)  return 127;
    return (int8_t)dst;
}

// 处理一个检测头 (INT8 量化模型)。返回该头检出的目标数。
int process_branch_i8(
    int8_t*  box_tensor,   int32_t box_zp,   float box_scale,
    int8_t*  score_tensor, int32_t score_zp, float score_scale,
    int8_t*  score_sum_tensor, int32_t ssum_zp, float ssum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& obj_probs,
    std::vector<int>&   class_ids,
    float threshold);

// 处理一个检测头 (FP32 非量化模型)。
int process_branch_fp32(
    float* box_tensor, float* score_tensor, float* score_sum_tensor,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& obj_probs,
    std::vector<int>&   class_ids,
    float threshold);

// IoU 计算 (xywh 格式)
float calc_iou(float x0, float y0, float w0, float h0,
               float x1, float y1, float w1, float h1);

// 按置信度降序排序 (快速排序), indices 同步重排
void sort_desc(std::vector<float>& scores, int l, int r,
               std::vector<int>& indices);

// 按类别做 NMS: order 为排序后的索引, filter_cls 指定类别,
// 命中 (iou > nms_thresh) 的元素 order[j] 置 -1。
void nms_per_class(int valid_count,
                   std::vector<float>& boxes,    // xywh
                   std::vector<int>&   cls_ids,
                   std::vector<int>&   order,
                   int filter_cls, float nms_thresh);

// 完整后处理入口: 3 个检测头 → DFL → 阈值过滤 → 按类 NMS → 映射回原图坐标。
// 阈值 obj_thresh / nms_thresh 由调用方传入。
int yolov8_post_process(RKNNContext* ctx, rknn_output* outputs,
                        const LetterBox& lb, DetectResult* result,
                        float obj_thresh, float nms_thresh);

}  // namespace yolo
