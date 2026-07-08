// rknn_context.h — RKNN 模型上下文 (RAII 封装)
//
// 从 c_realtime_infer_demo/camera_infer/yolo_camera.cc 的 RKNNContext +
// init_rknn() 抽取, 改为可被 RknnPool 多实例持有的形式。每个上下文绑定
// 一个 NPU 核心 (通过 rknn_set_core_mask), 实现多核并行。
#pragma once

#include <cstdint>
#include <string>
#include "rknn_api.h"

namespace yolo {

struct RKNNContext {
    rknn_context        ctx           = 0;
    rknn_input_output_num io_num      = {0, 0};
    rknn_tensor_attr*   input_attrs   = nullptr;
    rknn_tensor_attr*   output_attrs  = nullptr;
    rknn_tensor_format  input_fmt     = RKNN_TENSOR_NHWC;
    int                 model_width   = 640;
    int                 model_height  = 640;
    int                 model_channel = 3;
    int                 dfl_len       = 16;   // 由模型输出 tensor 决定
    int                 output_per_branch = 2; // 每分支输出 tensor 数
    bool                is_quant      = false;

    RKNNContext() = default;
    ~RKNNContext();
    RKNNContext(const RKNNContext&) = delete;
    RKNNContext& operator=(const RKNNContext&) = delete;
};

// 加载模型文件并初始化上下文。core_mask 指定 NPU 核心 (RKNN_NPU_CORE_0/1/2/...),
// 传 0 表示 RKNN_NPU_CORE_AUTO。成功返回 0, 失败返回负值。
int init_rknn(const std::string& model_path, RKNNContext* ctx,
              rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO);

}  // namespace yolo
