// rknn_context.cc — RKNNContext 析构 + init_rknn 实现
//
// 从 c_realtime_infer_demo/camera_infer/yolo_camera.cc 移植 (lines 110-253),
// 改为: ① RAII 析构; ② core_mask 参数 (支持 NPU 多核绑定); ③ 模型路径用 std::string。
#include "rknn_context.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "rknn_api.h"

namespace yolo {

RKNNContext::~RKNNContext() {
    if (input_attrs)  { free(input_attrs);  input_attrs  = nullptr; }
    if (output_attrs) { free(output_attrs); output_attrs = nullptr; }
    if (ctx)          { rknn_destroy(ctx);  ctx = 0; }
}

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

int init_rknn(const std::string& model_path, RKNNContext* ctx,
              rknn_core_mask core_mask) {
    unsigned char* model = nullptr;
    int model_len = load_model_file(model_path.c_str(), &model);
    if (model_len < 0) { printf("load model fail!\n"); return -1; }

    int ret = rknn_init(&ctx->ctx, model, (uint32_t)model_len, 0, nullptr);
    free(model);
    if (ret < 0) { printf("rknn_init fail! ret=%d\n", ret); return -1; }

    // 设置 NPU 核心亲和 (Python rknnpool_ld.py 中 NPU_CORE_0/1/2 的等价物)
    if (core_mask != RKNN_NPU_CORE_AUTO) {
        ret = rknn_set_core_mask(ctx->ctx, core_mask);
        if (ret < 0) {
            printf("rknn_set_core_mask(mask=%d) fail! ret=%d (continue)\n",
                   (int)core_mask, ret);
        }
    }

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

}  // namespace yolo
