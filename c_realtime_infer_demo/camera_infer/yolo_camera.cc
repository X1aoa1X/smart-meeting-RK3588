// yolo_camera.cc — C++ 实时摄像头 YOLOv8n 推理示例 (RK3588 NPU)
//
// 功能: V4L2 摄像头 → BGR→RGB → Letterbox(640×640) → RKNN NPU 推理
//       → YOLOv8 后处理(DFL+NMS) → 画框 → OpenCV 显示
//
// 依赖: librknnrt.so (Rockchip NPU SDK), OpenCV 4.x, pthread
//
// 编译 (板端):
//   g++ -O2 -Wall -std=c++17 yolo_camera.cc -o yolo_camera
//        -I/usr/include/opencv4 -I/usr/include/rknn
//        -L/usr/lib -lrknnrt -lopencv_core -lopencv_videoio
//        -lopencv_imgproc -lopencv_highgui -lpthread
//
// 运行:
//   ./yolo_camera ./best.rknn [/dev/video21]

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <set>
#include <vector>

#include "rknn_api.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

/* ============================================================
 *  配置常量
 * ============================================================ */
static constexpr int   MODEL_WIDTH       = 640;
static constexpr int   MODEL_HEIGHT      = 640;
static constexpr int   OBJ_CLASS_NUM     = 80;
static constexpr int   OBJ_NUMB_MAX_SIZE = 128;   // 每帧最多检出目标
// DFL_LEN 将在 init_rknn 后根据模型输出 tensor 动态计算
// YOLOv8n reg_max=16 → dfl_len=64/4=16; YOLOv8s reg_max=16 同

// 阈值 (可调整)
static constexpr float BOX_THRESH        = 0.25f; // 置信度阈值
static constexpr float NMS_THRESH        = 0.45f; // NMS IoU 阈值

// 摄像头默认参数
static const char*    DEFAULT_CAM_DEV    = "/dev/video21";
static constexpr int  CAM_WIDTH          = 1280;  // 摄像头最高支持 1280×720
static constexpr int  CAM_HEIGHT         = 720;
static constexpr int  CAM_FPS            = 30;    // MJPG 支持 30fps

// COCO 类别标签 (内联, 也可从文件加载)
static const char* COCO_CLASSES[OBJ_CLASS_NUM] = {
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

/* ============================================================
 *  数据结构
 * ============================================================ */
struct LetterBox {
    int   x_pad;  // 水平 padding (left)
    int   y_pad;  // 垂直 padding (top)
    float scale;  // 缩放因子
};

struct DetectBox {
    int   left, top, right, bottom;
    float confidence;
    int   cls_id;
};

struct DetectResult {
    int        count;
    DetectBox  results[OBJ_NUMB_MAX_SIZE];
};

/* ============================================================
 *  工具函数
 * ============================================================ */
static inline int clamp_i(float v, int lo, int hi) {
    if (v <= lo) return lo;
    if (v >= hi) return hi;
    return (int)v;
}

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ============================================================
 *  RKNN 模型加载 / 释放
 * ============================================================ */
struct RKNNContext {
    rknn_context       ctx       = 0;
    rknn_input_output_num io_num;
    rknn_tensor_attr*  input_attrs  = nullptr;
    rknn_tensor_attr*  output_attrs = nullptr;
    rknn_tensor_format input_fmt    = RKNN_TENSOR_NHWC;
    int                model_width  = MODEL_WIDTH;
    int                model_height = MODEL_HEIGHT;
    int                model_channel = 3;
    int                dfl_len      = 16;   // 由模型输出 tensor 决定
    int                output_per_branch = 2; // 每分支输出 tensor 数
    bool               is_quant     = false;

    ~RKNNContext() {
        free(input_attrs);
        free(output_attrs);
        if (ctx) rknn_destroy(ctx);
    }
};

// 读取文件全部内容到内存
static int load_model_file(const char* path, unsigned char** out_data) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    *out_data = (unsigned char*)malloc(sz);
    if (!*out_data) { fclose(f); return -1; }
    if (fread(*out_data, 1, (size_t)sz, f) != (size_t)sz) {
        free(*out_data);
        fclose(f);
        return -1;
    }
    fclose(f);
    return (int)sz;
}

static void dump_attr(rknn_tensor_attr* attr) {
    printf("  index=%d name=%s dims=[%d,%d,%d,%d] n_elems=%d size=%d "
           "fmt=%d type=%d qnt=%d zp=%d scale=%f\n",
           attr->index, attr->name,
           attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size,
           attr->fmt, attr->type, attr->qnt_type, attr->zp, attr->scale);
}

static int init_rknn(const char* model_path, RKNNContext* ctx) {
    unsigned char* model = nullptr;
    int model_len = load_model_file(model_path, &model);
    if (model_len < 0) { printf("load model fail!\n"); return -1; }

    int ret = rknn_init(&ctx->ctx, model, (uint32_t)model_len, 0, nullptr);
    free(model);
    if (ret < 0) { printf("rknn_init fail! ret=%d\n", ret); return -1; }

    // 查询输入输出数量
    ret = rknn_query(ctx->ctx, RKNN_QUERY_IN_OUT_NUM, &ctx->io_num,
                     sizeof(ctx->io_num));
    if (ret != RKNN_SUCC) { printf("query in/out num fail!\n"); return -1; }
    printf("model input num=%d, output num=%d\n",
           ctx->io_num.n_input, ctx->io_num.n_output);

    // 输入属性
    printf("input tensors:\n");
    ctx->input_attrs = (rknn_tensor_attr*)malloc(
        ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    memset(ctx->input_attrs, 0,
           ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < ctx->io_num.n_input; i++) {
        ctx->input_attrs[i].index = i;
        rknn_query(ctx->ctx, RKNN_QUERY_INPUT_ATTR, &ctx->input_attrs[i],
                   sizeof(rknn_tensor_attr));
        dump_attr(&ctx->input_attrs[i]);
    }

    // 输出属性
    printf("output tensors:\n");
    ctx->output_attrs = (rknn_tensor_attr*)malloc(
        ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    memset(ctx->output_attrs, 0,
           ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < ctx->io_num.n_output; i++) {
        ctx->output_attrs[i].index = i;
        rknn_query(ctx->ctx, RKNN_QUERY_OUTPUT_ATTR, &ctx->output_attrs[i],
                   sizeof(rknn_tensor_attr));
        dump_attr(&ctx->output_attrs[i]);
    }

    // 量化判断
    if (ctx->output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
        && ctx->output_attrs[0].type == RKNN_TENSOR_INT8) {
        ctx->is_quant = true;
    }
    printf("model is_quant=%d\n", (int)ctx->is_quant);

    // 计算 DFL 长度和每分支输出数
    // YOLOv8: box tensor dims[1]=4*reg_max, 所以 dfl_len = dims[1]/4
    ctx->dfl_len = ctx->output_attrs[0].dims[1] / 4;
    ctx->output_per_branch = ctx->io_num.n_output / 3;
    printf("dfl_len=%d, output_per_branch=%d\n",
           ctx->dfl_len, ctx->output_per_branch);

    // 输入格式
    ctx->input_fmt = ctx->input_attrs[0].fmt;
    if (ctx->input_fmt == RKNN_TENSOR_NCHW) {
        ctx->model_channel = ctx->input_attrs[0].dims[1];
        ctx->model_height  = ctx->input_attrs[0].dims[2];
        ctx->model_width   = ctx->input_attrs[0].dims[3];
    } else {
        ctx->model_height  = ctx->input_attrs[0].dims[1];
        ctx->model_width   = ctx->input_attrs[0].dims[2];
        ctx->model_channel = ctx->input_attrs[0].dims[3];
    }
    printf("model: %dx%d ch=%d (fmt=%s)\n",
           ctx->model_width, ctx->model_height, ctx->model_channel,
           ctx->input_fmt == RKNN_TENSOR_NCHW ? "NCHW" : "NHWC");

    // 预热一次
    printf("warming up...\n");
    rknn_input  warm_in[1];
    std::vector<rknn_output> warm_out(ctx->io_num.n_output);
    memset(warm_in, 0, sizeof(warm_in));
    memset(warm_out.data(), 0, warm_out.size() * sizeof(rknn_output));
    size_t in_size = ctx->model_width * ctx->model_height * ctx->model_channel;
    uint8_t* fake_buf = (uint8_t*)malloc(in_size);
    memset(fake_buf, 128, in_size);
    warm_in[0].index = 0;
    warm_in[0].type  = RKNN_TENSOR_UINT8;
    warm_in[0].fmt   = ctx->input_fmt;
    warm_in[0].size  = in_size;
    warm_in[0].buf   = fake_buf;
    rknn_inputs_set(ctx->ctx, 1, warm_in);
    rknn_run(ctx->ctx, nullptr);
    for (uint32_t i = 0; i < ctx->io_num.n_output; i++) {
        warm_out[i].index = i;
        warm_out[i].want_float = ctx->is_quant ? false : true;
    }
    rknn_outputs_get(ctx->ctx, ctx->io_num.n_output, warm_out.data(), nullptr);
    rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, warm_out.data());
    free(fake_buf);
    printf("warmup done.\n");
    return 0;
}

/* ============================================================
 *  YOLOv8 后处理
 * ============================================================ */

// Distribution Focal Loss: 把 reg_max 个分布值 → 标量 box 偏移
static void compute_dfl(const float* tensor, int dfl_len, float* box_out) {
    for (int b = 0; b < 4; b++) {
        float exp_t[dfl_len];
        float exp_sum = 0, acc_sum = 0;
        for (int i = 0; i < dfl_len; i++) {
            exp_t[i] = expf(tensor[i + b * dfl_len]);
            exp_sum += exp_t[i];
        }
        for (int i = 0; i < dfl_len; i++) {
            acc_sum += (exp_t[i] / exp_sum) * i;
        }
        box_out[b] = acc_sum;
    }
}

// INT8 反量化
static inline float deqnt_i8_to_f32(int8_t q, int32_t zp, float scale) {
    return ((float)q - (float)zp) * scale;
}

// FP32 → INT8 量化阈值
static inline int8_t qnt_f32_to_i8(float f32, int32_t zp, float scale) {
    float dst = f32 / scale + (float)zp;
    if (dst <= -128.f) return -128;
    if (dst >= 127.f)  return 127;
    return (int8_t)dst;
}

// 处理一个检测头 (INT8 量化模型)
static int process_branch_i8(
    int8_t*  box_tensor,   int32_t box_zp,   float box_scale,
    int8_t*  score_tensor, int32_t score_zp, float score_scale,
    int8_t*  score_sum_tensor, int32_t ssum_zp, float ssum_scale,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& obj_probs,
    std::vector<int>&   class_ids,
    float threshold)
{
    int grid_len    = grid_h * grid_w;
    int valid_count = 0;
    int8_t score_thres_i8   = qnt_f32_to_i8(threshold, score_zp, score_scale);
    int8_t ssum_thres_i8    = qnt_f32_to_i8(threshold, ssum_zp,  ssum_scale);

    for (int y = 0; y < grid_h; y++) {
        for (int x = 0; x < grid_w; x++) {
            int cell_offset = y * grid_w + x;

            // 快速跳过: score_sum 预过滤
            if (score_sum_tensor && score_sum_tensor[cell_offset] < ssum_thres_i8) {
                continue;
            }

            // 找分数最高的类别
            int   max_cls_id = -1;
            int8_t max_score = -score_zp;  // 相当于 0 置信度
            for (int c = 0; c < OBJ_CLASS_NUM; c++) {
                int idx = cell_offset + c * grid_len;
                if (score_tensor[idx] > score_thres_i8
                    && score_tensor[idx] > max_score) {
                    max_score  = score_tensor[idx];
                    max_cls_id = c;
                }
            }

            if (max_score <= score_thres_i8) continue;

            // DFL → bounding box
            float before_dfl[dfl_len * 4];
            for (int k = 0; k < dfl_len * 4; k++) {
                int idx = cell_offset + k * grid_len;
                before_dfl[k] = deqnt_i8_to_f32(box_tensor[idx], box_zp, box_scale);
            }
            float dfl_box[4];
            compute_dfl(before_dfl, dfl_len, dfl_box);

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

// 处理一个检测头 (FP32 非量化模型)
static int process_branch_fp32(
    float* box_tensor, float* score_tensor, float* score_sum_tensor,
    int grid_h, int grid_w, int stride, int dfl_len,
    std::vector<float>& boxes,
    std::vector<float>& obj_probs,
    std::vector<int>&   class_ids,
    float threshold)
{
    int grid_len    = grid_h * grid_w;
    int valid_count = 0;

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

            float before_dfl[dfl_len * 4];
            for (int k = 0; k < dfl_len * 4; k++) {
                before_dfl[k] = box_tensor[cell_offset + k * grid_len];
            }
            float dfl_box[4];
            compute_dfl(before_dfl, dfl_len, dfl_box);

            float x1 = (-dfl_box[0] + (float)x + 0.5f) * (float)stride;
            float y1 = (-dfl_box[1] + (float)y + 0.5f) * (float)stride;
            float x2 = ( dfl_box[2] + (float)x + 0.5f) * (float)stride;
            float y2 = ( dfl_box[3] + (float)y + 0.5f) * (float)stride;

            boxes.push_back(x1);
            boxes.push_back(y1);
            boxes.push_back(x2 - x1);  // w
            boxes.push_back(y2 - y1);  // h

            obj_probs.push_back(max_score);
            class_ids.push_back(max_cls_id);
            valid_count++;
        }
    }
    return valid_count;
}

// IoU 计算 (xywh 格式)
static float calc_iou(float x0, float y0, float w0, float h0,
                      float x1, float y1, float w1, float h1) {
    float ix = fmaxf(0.f, fminf(x0 + w0, x1 + w1) - fmaxf(x0, x1) + 1.f);
    float iy = fmaxf(0.f, fminf(y0 + h0, y1 + h1) - fmaxf(y0, y1) + 1.f);
    float inter = ix * iy;
    float area0 = w0 * h0;
    float area1 = w1 * h1;
    float un = area0 + area1 - inter;
    return un <= 0.f ? 0.f : inter / un;
}

// 快速排序 (按置信度降序)
static void sort_desc(std::vector<float>& scores, int l, int r,
                      std::vector<int>& indices) {
    if (l >= r) return;
    float key    = scores[l];
    int   key_i  = indices[l];
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

// 按类别做 NMS
static void nms_per_class(int valid_count,
                          std::vector<float>& boxes,    // xywh
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

// 完整后处理入口
static int yolov8_post_process(RKNNContext* ctx, rknn_output* outputs,
                               LetterBox* lb,
                               DetectResult* result) {
    memset(result, 0, sizeof(*result));

    int output_per_branch = ctx->output_per_branch;
    int dfl_len           = ctx->dfl_len;

    std::vector<float> all_boxes;
    std::vector<float> all_probs;
    std::vector<int>   all_cls_ids;
    int total_valid = 0;

    for (int branch = 0; branch < 3; branch++) {
        // 确定 score_sum (可能不存在)
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
                all_boxes, all_probs, all_cls_ids, BOX_THRESH);
        } else {
            branch_valid = process_branch_fp32(
                (float*)outputs[box_idx].buf,
                (float*)outputs[score_idx].buf,
                (float*)ssum_buf,
                grid_h, grid_w, stride, dfl_len,
                all_boxes, all_probs, all_cls_ids, BOX_THRESH);
        }
        total_valid += branch_valid;
    }

    if (total_valid <= 0) return 0;

    // 按置信度降序排序
    std::vector<int> indices(total_valid);
    for (int i = 0; i < total_valid; i++) indices[i] = i;
    sort_desc(all_probs, 0, total_valid - 1, indices);

    // 按类别 NMS
    std::set<int> cls_set(all_cls_ids.begin(), all_cls_ids.end());
    for (int c : cls_set) {
        nms_per_class(total_valid, all_boxes, all_cls_ids, indices, c, NMS_THRESH);
    }

    // 组装结果 (转换到原图坐标)
    int out_count = 0;
    for (int i = 0; i < total_valid; i++) {
        if (indices[i] == -1 || out_count >= OBJ_NUMB_MAX_SIZE) continue;
        int n = indices[i];

        // 先 clamp 到模型空间 [0, 640], 再除以 scale 映射回原图坐标
        float x1 = all_boxes[n * 4 + 0] - (float)lb->x_pad;
        float y1 = all_boxes[n * 4 + 1] - (float)lb->y_pad;
        float x2 = x1 + all_boxes[n * 4 + 2];
        float y2 = y1 + all_boxes[n * 4 + 3];

        result->results[out_count].left   = (int)(clamp_i(x1, 0, MODEL_WIDTH)  / lb->scale);
        result->results[out_count].top    = (int)(clamp_i(y1, 0, MODEL_HEIGHT) / lb->scale);
        result->results[out_count].right  = (int)(clamp_i(x2, 0, MODEL_WIDTH)  / lb->scale);
        result->results[out_count].bottom = (int)(clamp_i(y2, 0, MODEL_HEIGHT) / lb->scale);
        result->results[out_count].confidence = all_probs[n];
        result->results[out_count].cls_id     = all_cls_ids[n];
        out_count++;
    }
    result->count = out_count;
    return 0;
}

/* ============================================================
 *  预处理: BGR camera frame → RGB letterbox 640×640
 * ============================================================ */
static void preprocess(const cv::Mat& bgr_frame,
                       cv::Mat&       rgb_letterbox,
                       LetterBox&     lb) {
    int iw = bgr_frame.cols;
    int ih = bgr_frame.rows;

    // 计算 letterbox 参数
    float scale = std::min(
        (float)MODEL_WIDTH  / (float)iw,
        (float)MODEL_HEIGHT / (float)ih);
    lb.scale = scale;

    int new_w = (int)std::round(iw * scale);
    int new_h = (int)std::round(ih * scale);
    lb.x_pad = (MODEL_WIDTH  - new_w) / 2;
    lb.y_pad = (MODEL_HEIGHT - new_h) / 2;

    // BGR → RGB + resize → letterbox pad (黑边填充 0)
    cv::Mat resized, rgb;
    cv::resize(bgr_frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rgb_letterbox = cv::Mat(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    rgb.copyTo(rgb_letterbox(cv::Rect(lb.x_pad, lb.y_pad, new_w, new_h)));
}

/* ============================================================
 *  画框 (OpenCV)
 * ============================================================ */
static void draw_results(cv::Mat& frame, const DetectResult& res) {
    for (int i = 0; i < res.count; i++) {
        const DetectBox& d = res.results[i];
        cv::Scalar color(0, 255, 0);  // 绿色
        cv::rectangle(frame,
                      cv::Point(d.left, d.top),
                      cv::Point(d.right, d.bottom),
                      color, 2);

        char label[128];
        snprintf(label, sizeof(label), "%s %.1f%%",
                 COCO_CLASSES[d.cls_id], d.confidence * 100.f);
        int base;
        cv::Size txt = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                       0.5, 1, &base);
        int ty = d.top - txt.height - 3;
        if (ty < 0) ty = d.top + 5;
        cv::rectangle(frame,
                      cv::Point(d.left, ty),
                      cv::Point(d.left + txt.width, ty + txt.height + 3),
                      color, cv::FILLED);
        cv::putText(frame, label,
                    cv::Point(d.left, ty + txt.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1);
    }
}

/* ============================================================
 *  主函数
 * ============================================================ */
int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);   // 禁用缓冲，确保日志实时输出
    const char* model_path = (argc >= 2) ? argv[1] : "./best.rknn";
    const char* cam_dev    = (argc >= 3) ? argv[2] : DEFAULT_CAM_DEV;

    printf("========================================\n");
    printf("  YOLOv8n Camera Inference (C++ RKNN)\n");
    printf("========================================\n");
    printf("Model:    %s\n", model_path);
    printf("Camera:   %s\n", cam_dev);

    // ── 1. 加载 RKNN 模型 ──────────────────────────────────
    RKNNContext rknn_ctx;
    if (init_rknn(model_path, &rknn_ctx) < 0) {
        fprintf(stderr, "Failed to init RKNN model!\n");
        return -1;
    }

    // ── 2. 打开 V4L2 摄像头 ────────────────────────────────
    cv::VideoCapture cap(cam_dev, cv::CAP_V4L2);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open camera: %s\n", cam_dev);
        return -1;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  CAM_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT);
    cap.set(cv::CAP_PROP_FPS,          CAM_FPS);
    cap.set(cv::CAP_PROP_BUFFERSIZE,   1);

    int actual_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int actual_fps = (int)cap.get(cv::CAP_PROP_FPS);
    printf("Camera:    %dx%d @ MJPG, target %d fps (actual %d)\n",
           actual_w, actual_h, CAM_FPS, actual_fps);

    // ── 3. 创建显示窗口 ────────────────────────────────────
    cv::namedWindow("YOLOv8n Camera", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLOv8n Camera", 960, 540);

    // ── 4. 主循环 ──────────────────────────────────────────
    cv::Mat            frame, rgb_input;
    LetterBox          letter_box;
    DetectResult       det_result;
    double             last_fps_time = now_sec();
    int                frame_count   = 0;
    float              fps           = 0.f;
    rknn_input                rk_in[1];
    std::vector<rknn_output>  rk_out(rknn_ctx.io_num.n_output);

    // 用于帧率统计
    double sum_cap = 0, sum_pre = 0, sum_inf = 0, sum_post = 0;
    double cap_ms = 0, pre_ms = 0, inf_ms = 0, post_ms = 0;
    double t_last_post = 0;  // 上一帧 post 完成时间

    printf("\nPress ESC to exit.\n\n");

    while (true) {
        // 读取帧
        if (!cap.read(frame) || frame.empty()) {
            fprintf(stderr, "read frame failed, retrying...\n");
            usleep(10000);
            continue;
        }

        double t_cap = now_sec();

        // ── 预处理: BGR → RGB letterbox 640×640 ────────────
        preprocess(frame, rgb_input, letter_box);

        double t_pre = now_sec();

        // ── 设置输入 ────────────────────────────────────────
        memset(rk_in, 0, sizeof(rk_in));
        rk_in[0].index = 0;
        rk_in[0].type  = RKNN_TENSOR_UINT8;
        rk_in[0].fmt   = rknn_ctx.input_fmt;
        rk_in[0].size  = rknn_ctx.model_width
                       * rknn_ctx.model_height
                       * rknn_ctx.model_channel;
        rk_in[0].buf   = rgb_input.data;

        if (rknn_inputs_set(rknn_ctx.ctx, 1, rk_in) < 0) {
            fprintf(stderr, "rknn_inputs_set failed!\n");
            break;
        }

        // ── NPU 推理 ────────────────────────────────────────
        if (rknn_run(rknn_ctx.ctx, nullptr) < 0) {
            fprintf(stderr, "rknn_run failed!\n");
            break;
        }

        double t_inf = now_sec();

        // ── 获取输出 ────────────────────────────────────────
        memset(rk_out.data(), 0, rk_out.size() * sizeof(rknn_output));
        for (uint32_t i = 0; i < rknn_ctx.io_num.n_output; i++) {
            rk_out[i].index      = i;
            rk_out[i].want_float = rknn_ctx.is_quant ? false : true;
        }
        if (rknn_outputs_get(rknn_ctx.ctx, rknn_ctx.io_num.n_output,
                             rk_out.data(), nullptr) < 0) {
            fprintf(stderr, "rknn_outputs_get failed!\n");
            break;
        }

        // ── 后处理 ──────────────────────────────────────────
        yolov8_post_process(&rknn_ctx, rk_out.data(), &letter_box, &det_result);

        double t_post = now_sec();

        // ── 释放输出 buffer ─────────────────────────────────
        rknn_outputs_release(rknn_ctx.ctx, rknn_ctx.io_num.n_output,
                             rk_out.data());

        // ── 画框并显示 ──────────────────────────────────────
        draw_results(frame, det_result);

        // FPS 叠加
        char fps_str[96];
        snprintf(fps_str, sizeof(fps_str),
                 "FPS: %.1f | detect: %d | cap:%.1f pre:%.1f inf:%.1f post:%.1f ms",
                 fps, det_result.count, cap_ms, pre_ms, inf_ms, post_ms);
        cv::putText(frame, fps_str, cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);

        // 缩放到适合显示
        cv::Mat display;
        if (frame.cols > 960) {
            double ds = 960.0 / frame.cols;
            cv::resize(frame, display,
                       cv::Size(960, (int)(frame.rows * ds)));
        } else {
            display = frame;
        }

        cv::imshow("YOLOv8n Camera", display);

        // ── 按键处理 ────────────────────────────────────────
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') {  // ESC / q → 退出
            printf("\nUser exit.\n");
            break;
        }

        // ── FPS 统计 ────────────────────────────────────────
        // 累加本帧各阶段耗时
        double cur_cap  = t_cap  - t_last_post;
        double cur_pre  = t_pre  - t_cap;
        double cur_inf  = t_inf  - t_pre;
        double cur_post = t_post - t_inf;
        if (t_last_post > 0) {
            sum_cap  += cur_cap;
            sum_pre  += cur_pre;
            sum_inf  += cur_inf;
            sum_post += cur_post;
        }
        t_last_post = t_post;

        frame_count++;
        if (frame_count >= 15) {
            double now = now_sec();
            fps = (float)frame_count / (float)(now - last_fps_time);
            // 输出平均值 (ms)
            cap_ms  = sum_cap  / (float)frame_count * 1000.0;
            pre_ms  = sum_pre  / (float)frame_count * 1000.0;
            inf_ms  = sum_inf  / (float)frame_count * 1000.0;
            post_ms = sum_post / (float)frame_count * 1000.0;
            printf("  FPS: %5.1f | cap: %5.1fms | pre: %5.1fms | inf: %5.1fms | "
                   "post: %5.1fms | detect: %d\n",
                   fps, cap_ms, pre_ms, inf_ms, post_ms, det_result.count);
            frame_count   = 0;
            sum_cap = sum_pre = sum_inf = sum_post = 0;
            last_fps_time = now;
        }
    }

    // ── 5. 清理 ─────────────────────────────────────────────
    cv::destroyAllWindows();
    cap.release();
    printf("Exit. Bye!\n");
    return 0;
}
