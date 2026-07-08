// post_process.cc — YOLOv8 后处理实现
//
// 从 c_realtime_infer_demo/camera_infer/yolo_camera.cc 移植:
//   letterbox (563-587), compute_dfl (260-273, 增加 max 减法),
//   process_branch_i8 (289-352), process_branch_fp32 (355-410),
//   calc_iou / sort_desc / nms_per_class (413-465),
//   yolov8_post_process (468-558, 阈值参数化)。
//
// 算法保持与 C++ 原版一致; DFL 增加 max-subtraction 以对齐 Python dfl()。
#include "post_process.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

#include <opencv2/imgproc.hpp>

namespace yolo {

void letterbox(const cv::Mat& bgr, cv::Mat& rgb_letterbox, LetterBox& lb,
               int target_w, int target_h) {
    int iw = bgr.cols;
    int ih = bgr.rows;

    float scale = std::min(
        (float)target_w / (float)iw,
        (float)target_h / (float)ih);
    lb.scale = scale;

    int new_w = (int)std::round(iw * scale);
    int new_h = (int)std::round(ih * scale);
    lb.x_pad = (target_w - new_w) / 2;
    lb.y_pad = (target_h - new_h) / 2;

    cv::Mat resized, rgb;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rgb_letterbox = cv::Mat(target_h, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
    rgb.copyTo(rgb_letterbox(cv::Rect(lb.x_pad, lb.y_pad, new_w, new_h)));
}

void compute_dfl(const float* tensor, int dfl_len, float* box_out) {
    // 与 Python dfl() (func_yolov8_optimize.py:111-125) 一致:
    // 对每个 b in [0,4), 先求 dfl_len 个值的 max, 再做 softmax, 再加权求和。
    // 栈分配替代 std::vector, 避免每个 valid cell 都堆分配 (YOLOv8 dfl_len≤16)。
    constexpr int kMaxDflLen = 64;
    if (dfl_len > kMaxDflLen) dfl_len = kMaxDflLen;  // 防御性 clamp
    for (int b = 0; b < 4; b++) {
        const float* p = tensor + b * dfl_len;
        float max_v = p[0];
        for (int i = 1; i < dfl_len; i++) {
            if (p[i] > max_v) max_v = p[i];
        }
        float exp_sum = 0.0f, acc_sum = 0.0f;
        float exp_t[kMaxDflLen];
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = expf(p[i] - max_v);
            exp_sum += exp_t[i];
        }
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += (exp_t[i] / exp_sum) * (float)i;
        }
        box_out[b] = acc_sum;
    }
}

int process_branch_i8(
    int8_t*  box_tensor,   int32_t box_zp,   float box_scale,
    int8_t*  score_tensor, int32_t score_zp, float score_scale,
    int8_t*  score_sum_tensor, int32_t ssum_zp, float ssum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& obj_probs,
    std::vector<int>&   class_ids,
    float threshold) {
    int grid_len    = grid_h * grid_w;
    int valid_count = 0;
    int8_t score_thres_i8 = qnt_f32_to_i8(threshold, score_zp, score_scale);
    int8_t ssum_thres_i8  = qnt_f32_to_i8(threshold, ssum_zp,  ssum_scale);

    std::vector<float> before_dfl(dfl_len * 4);

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            int cell_offset = y * grid_w + x;

            if (score_sum_tensor && score_sum_tensor[cell_offset] < ssum_thres_i8) {
                continue;
            }

            int    max_cls_id = -1;
            int8_t max_score  = -score_zp;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                int idx = cell_offset + c * grid_len;
                if (score_tensor[idx] > score_thres_i8
                    && score_tensor[idx] > max_score) {
                    max_score  = score_tensor[idx];
                    max_cls_id = c;
                }
            }

            if (max_score <= score_thres_i8) continue;

            for (int k = 0; k < dfl_len * 4; k++) {
                int idx = cell_offset + k * grid_len;
                before_dfl[k] = deqnt_i8_to_f32(box_tensor[idx], box_zp, box_scale);
            }
            float dfl_box[4];
            compute_dfl(before_dfl.data(), dfl_len, dfl_box);

            float x1 = (-dfl_box[0] + (float)x + 0.5f) * (float)stride;
            float y1 = (-dfl_box[1] + (float)y + 0.5f) * (float)stride;
            float x2 = ( dfl_box[2] + (float)x + 0.5f) * (float)stride;
            float y2 = ( dfl_box[3] + (float)y + 0.5f) * (float)stride;

            boxes.push_back(x1);
            boxes.push_back(y1);
            boxes.push_back(x2 - x1);  // w
            boxes.push_back(y2 - y1);  // h

            obj_probs.push_back(deqnt_i8_to_f32(max_score, score_zp, score_scale));
            class_ids.push_back(max_cls_id);
            valid_count++;
        }
    }
    return valid_count;
}

int process_branch_fp32(
    float* box_tensor, float* score_tensor, float* score_sum_tensor,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& obj_probs,
    std::vector<int>&   class_ids,
    float threshold) {
    int grid_len    = grid_h * grid_w;
    int valid_count = 0;

    std::vector<float> before_dfl(dfl_len * 4);

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            int cell_offset = y * grid_w + x;

            if (score_sum_tensor && score_sum_tensor[cell_offset] < threshold) {
                continue;
            }

            int   max_cls_id = -1;
            float max_score  = 0.f;
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                int idx = cell_offset + c * grid_len;
                if (score_tensor[idx] > threshold
                    && score_tensor[idx] > max_score) {
                    max_score  = score_tensor[idx];
                    max_cls_id = c;
                }
            }

            if (max_score <= threshold) continue;

            for (int k = 0; k < dfl_len * 4; k++) {
                before_dfl[k] = box_tensor[cell_offset + k * grid_len];
            }
            float dfl_box[4];
            compute_dfl(before_dfl.data(), dfl_len, dfl_box);

            float x1 = (-dfl_box[0] + (float)x + 0.5f) * (float)stride;
            float y1 = (-dfl_box[1] + (float)y + 0.5f) * (float)stride;
            float x2 = ( dfl_box[2] + (float)x + 0.5f) * (float)stride;
            float y2 = ( dfl_box[3] + (float)y + 0.5f) * (float)stride;

            boxes.push_back(x1);
            boxes.push_back(y1);
            boxes.push_back(x2 - x1);
            boxes.push_back(y2 - y1);

            obj_probs.push_back(max_score);
            class_ids.push_back(max_cls_id);
            valid_count++;
        }
    }
    return valid_count;
}

float calc_iou(float x0, float y0, float w0, float h0,
               float x1, float y1, float w1, float h1) {
    float ix = fmaxf(0.f, fminf(x0 + w0, x1 + w1) - fmaxf(x0, x1) + 1.f);
    float iy = fmaxf(0.f, fminf(y0 + h0, y1 + h1) - fmaxf(y0, y1) + 1.f);
    float inter  = ix * iy;
    float area0  = w0 * h0;
    float area1  = w1 * h1;
    float un     = area0 + area1 - inter;
    return un <= 0.f ? 0.f : inter / un;
}

void sort_desc(std::vector<float>& scores, int l, int r,
               std::vector<int>& indices) {
    if (l >= r) return;
    float key   = scores[l];
    int   key_i = indices[l];
    int   low = l, high = r;
    while (low < high) {
        while (low < high && scores[high] <= key) high--;
        scores[low]  = scores[high];
        indices[low] = indices[high];
        while (low < high && scores[low] >= key) low++;
        scores[high]  = scores[low];
        indices[high] = indices[low];
    }
    scores[low]  = key;
    indices[low] = key_i;
    sort_desc(scores, l, low - 1, indices);
    sort_desc(scores, low + 1, r, indices);
}

void nms_per_class(int valid_count,
                   std::vector<float>& boxes,
                   std::vector<int>&   cls_ids,
                   std::vector<int>&   order,
                   int filter_cls, float nms_thresh) {
    for (int i = 0; i < valid_count; i++) {
        int n = order[i];
        if (n == -1 || cls_ids[n] != filter_cls) continue;
        for (int j = i + 1; j < valid_count; j++) {
            int m = order[j];
            if (m == -1 || cls_ids[m] != filter_cls) continue;
            float iou = calc_iou(
                boxes[n * 4], boxes[n * 4 + 1], boxes[n * 4 + 2], boxes[n * 4 + 3],
                boxes[m * 4], boxes[m * 4 + 1], boxes[m * 4 + 2], boxes[m * 4 + 3]);
            if (iou > nms_thresh) {
                order[j] = -1;
            }
        }
    }
}

static inline int clamp_i(float v, int lo, int hi) {
    if (v <= lo) return lo;
    if (v >= hi) return hi;
    return (int)v;
}

int yolov8_post_process(RKNNContext* ctx, rknn_output* outputs,
                        const LetterBox& lb, DetectResult* result,
                        float obj_thresh, float nms_thresh) {
    result->count = 0;
    result->boxes.clear();

    int output_per_branch = ctx->output_per_branch;
    int dfl_len           = ctx->dfl_len;

    std::vector<float> all_boxes;
    std::vector<float> all_probs;
    std::vector<int>   all_cls_ids;
    int total_valid = 0;

    for (int branch = 0; branch < 3; branch++) {
        void*   ssum_buf   = nullptr;
        int32_t ssum_zp    = 0;
        float   ssum_scale = 1.f;
        if (output_per_branch == 3) {
            int ssum_idx = branch * output_per_branch + 2;
            ssum_buf   = outputs[ssum_idx].buf;
            ssum_zp    = ctx->output_attrs[ssum_idx].zp;
            ssum_scale = ctx->output_attrs[ssum_idx].scale;
        }

        int box_idx   = branch * output_per_branch;
        int score_idx = branch * output_per_branch + 1;

        int grid_h = ctx->output_attrs[box_idx].dims[2];
        int grid_w = ctx->output_attrs[box_idx].dims[3];
        int stride = ctx->model_height / grid_h;

        int branch_valid;
        if (ctx->is_quant) {
            branch_valid = process_branch_i8(
                (int8_t*)outputs[box_idx].buf,
                ctx->output_attrs[box_idx].zp,
                ctx->output_attrs[box_idx].scale,
                (int8_t*)outputs[score_idx].buf,
                ctx->output_attrs[score_idx].zp,
                ctx->output_attrs[score_idx].scale,
                (int8_t*)ssum_buf, ssum_zp, ssum_scale,
                grid_h, grid_w, stride, dfl_len,
                all_boxes, all_probs, all_cls_ids, obj_thresh);
        } else {
            branch_valid = process_branch_fp32(
                (float*)outputs[box_idx].buf,
                (float*)outputs[score_idx].buf,
                (float*)ssum_buf,
                grid_h, grid_w, stride, dfl_len,
                all_boxes, all_probs, all_cls_ids, obj_thresh);
        }
        total_valid += branch_valid;
    }

    if (total_valid <= 0) return 0;

    std::vector<int> indices(total_valid);
    for (int i = 0; i < total_valid; i++) indices[i] = i;
    sort_desc(all_probs, 0, total_valid - 1, indices);

    std::set<int> cls_set(all_cls_ids.begin(), all_cls_ids.end());
    for (int c : cls_set) {
        nms_per_class(total_valid, all_boxes, all_cls_ids, indices, c, nms_thresh);
    }

    int out_count = 0;
    for (int i = 0; i < total_valid && out_count < OBJ_NUMB_MAX_SIZE; i++) {
        if (indices[i] == -1) continue;
        int n = indices[i];

        float x1 = all_boxes[n * 4 + 0] - (float)lb.x_pad;
        float y1 = all_boxes[n * 4 + 1] - (float)lb.y_pad;
        float x2 = x1 + all_boxes[n * 4 + 2];
        float y2 = y1 + all_boxes[n * 4 + 3];

        DetectBox d;
        d.left   = (float)(clamp_i(x1, 0, ctx->model_width)  / lb.scale);
        d.top    = (float)(clamp_i(y1, 0, ctx->model_height) / lb.scale);
        d.right  = (float)(clamp_i(x2, 0, ctx->model_width)  / lb.scale);
        d.bottom = (float)(clamp_i(y2, 0, ctx->model_height) / lb.scale);
        d.confidence = all_probs[n];
        d.cls_id     = all_cls_ids[n];
        result->boxes.push_back(d);
        out_count++;
    }
    result->count = out_count;
    return 0;
}

}  // namespace yolo
