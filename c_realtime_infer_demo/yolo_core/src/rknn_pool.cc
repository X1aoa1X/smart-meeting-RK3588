// rknn_pool.cc — RknnPool 实现
//
// 设计要点:
//   - TPEs 个 worker 线程, 每个线程独占一个 RKNNContext (无锁并行)
//   - submit() 把 {task_id, frame} 入队, 任意空闲 worker 取走
//   - worker 用自己的 context 处理任务, 把结果写入 results_ map (按 task_id 索引)
//   - get() 阻塞等待 next_get_id_ 出现 (FIFO 取回)
//   - infer_sync(0,...) 与 worker 0 通过 dev_mtx_ 互斥 (偏差二次推理用)
//
// 这等价于 Python rknnpool_ld.py 的 rknnPoolExecutor + ThreadPoolExecutor 组合,
// 但把 RKNN 实例与线程绑定, 避免跨线程切换 context 的开销。
#include "rknn_pool.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <utility>

#include <opencv2/imgproc.hpp>

#include "post_process.h"
#include "rknn_api.h"

namespace yolo {

RknnPool::RknnPool(const std::string& model_path, int tpes,
                   float obj_thresh, float nms_thresh)
    : model_path_(model_path),
      tpes_(tpes > 0 ? tpes : 1),
      obj_thresh_(obj_thresh),
      nms_thresh_(nms_thresh) {}

RknnPool::~RknnPool() { release(); }

bool RknnPool::init() {
    contexts_.clear();
    contexts_.reserve(tpes_);
    for (int i = 0; i < tpes_; i++) {
        auto ctx = std::make_unique<RKNNContext>();
        // NPU 核心 0/1/2 轮转 (对齐 rknnpool_ld.py: i % 3)
        rknn_core_mask mask;
        switch (i % 3) {
            case 0: mask = RKNN_NPU_CORE_0; break;
            case 1: mask = RKNN_NPU_CORE_1; break;
            case 2: mask = RKNN_NPU_CORE_2; break;
            default: mask = RKNN_NPU_CORE_AUTO; break;
        }
        if (init_rknn(model_path_, ctx.get(), mask) < 0) {
            printf("[RknnPool] init_rknn failed on instance %d\n", i);
            return false;
        }
        contexts_.push_back(std::move(ctx));
    }

    // 启动 worker 线程 (每个 worker 拥有 contexts_[i])
    stop_ = false;
    workers_.clear();
    for (int i = 0; i < tpes_; i++) {
        workers_.emplace_back(&RknnPool::worker_loop, this, i);
    }
    return true;
}

int RknnPool::submit(cv::Mat bgr_frame) {
    int task_id = next_task_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(task_mtx_);
        task_queue_.emplace(task_id, std::move(bgr_frame));
    }
    task_cv_.notify_one();
    return task_id;
}

bool RknnPool::get(cv::Mat& annotated, DetectResult& result, TimingMs& timing,
                   int timeout_ms) {
    std::unique_lock<std::mutex> lk(results_mtx_);
    auto pred = [this]() {
        return results_.count(next_get_id_) > 0 || (stop_ && results_.empty());
    };
    if (timeout_ms < 0) {
        results_cv_.wait(lk, pred);
    } else {
        if (!results_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred)) {
            return false;  // 超时
        }
    }
    auto it = results_.find(next_get_id_);
    if (it == results_.end()) return false;  // 池已停止且无结果
    TaskResult tr = std::move(it->second);
    results_.erase(it);
    next_get_id_++;
    annotated = std::move(tr.annotated);
    result    = std::move(tr.result);
    timing    = std::move(tr.timing);
    return true;
}

bool RknnPool::infer_sync(int instance_idx, const cv::Mat& bgr_frame,
                          DetectResult& result, TimingMs& timing) {
    if (instance_idx < 0 || instance_idx >= (int)contexts_.size()) return false;
    cv::Mat annotated;
    // instance 0 与 worker 0 共享 context, 需互斥
    std::unique_lock<std::mutex> lk(dev_mtx_, std::defer_lock);
    if (instance_idx == 0) lk.lock();
    return process_on_context(instance_idx, bgr_frame, annotated, result, timing);
}

int RknnPool::pending() const {
    std::lock_guard<std::mutex> lk(const_cast<std::mutex&>(results_mtx_));
    return (int)results_.size();
}

void RknnPool::release() {
    if (stop_.exchange(true)) return;  // 已停止
    task_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
    contexts_.clear();
    {
        std::lock_guard<std::mutex> lk(task_mtx_);
        std::queue<std::pair<int, cv::Mat>> empty;
        task_queue_.swap(empty);
    }
    {
        std::lock_guard<std::mutex> lk(results_mtx_);
        results_.clear();
    }
    results_cv_.notify_all();
}

// ── 在指定 context 上跑一帧完整推理 ──────────────────────────
bool RknnPool::process_on_context(int instance_idx, const cv::Mat& bgr_frame,
                                  cv::Mat& annotated, DetectResult& result,
                                  TimingMs& timing) {
    RKNNContext* ctx = contexts_[instance_idx].get();
    auto t0 = std::chrono::steady_clock::now();

    // 1) 预处理: BGR → RGB letterbox 640×640
    cv::Mat rgb_input;
    LetterBox lb;
    letterbox(bgr_frame, rgb_input, lb);
    auto t1 = std::chrono::steady_clock::now();

    // 2) 设置输入
    rknn_input rk_in[1];
    memset(rk_in, 0, sizeof(rk_in));
    rk_in[0].index = 0;
    rk_in[0].type  = RKNN_TENSOR_UINT8;
    rk_in[0].fmt   = ctx->input_fmt;
    rk_in[0].size  = (size_t)ctx->model_width
                   * ctx->model_height * ctx->model_channel;
    rk_in[0].buf   = rgb_input.data;
    if (rknn_inputs_set(ctx->ctx, 1, rk_in) < 0) {
        printf("[RknnPool] rknn_inputs_set failed\n");
        return false;
    }

    // 3) NPU 推理
    if (rknn_run(ctx->ctx, nullptr) < 0) {
        printf("[RknnPool] rknn_run failed\n");
        return false;
    }

    // 4) 获取输出
    std::vector<rknn_output> rk_out(ctx->io_num.n_output);
    memset(rk_out.data(), 0, rk_out.size() * sizeof(rknn_output));
    for (uint32_t i = 0; i < ctx->io_num.n_output; i++) {
        rk_out[i].index      = i;
        rk_out[i].want_float = ctx->is_quant ? false : true;
    }
    if (rknn_outputs_get(ctx->ctx, ctx->io_num.n_output, rk_out.data(), nullptr) < 0) {
        printf("[RknnPool] rknn_outputs_get failed\n");
        return false;
    }
    auto t2 = std::chrono::steady_clock::now();

    // 5) 后处理
    yolov8_post_process(ctx, rk_out.data(), lb, &result, obj_thresh_, nms_thresh_);
    rknn_outputs_release(ctx->ctx, ctx->io_num.n_output, rk_out.data());
    auto t3 = std::chrono::steady_clock::now();

    // 6) 输出原图 (浅拷贝, 不 clone): bgr_frame 由 submit() move 入队,
    //    worker 独占该帧; engine 在 get_result 时才 draw_results 画框,
    //    此时 frame 引用已转移到 annotated, 不会污染调用方。
    //    原 clone() 每帧 6MB 堆拷贝是显著开销, 已移除。
    annotated = bgr_frame;

    timing.preprocess  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing.inference   = std::chrono::duration<double, std::milli>(t2 - t1).count();
    timing.postprocess = std::chrono::duration<double, std::milli>(t3 - t2).count();
    timing.total       = std::chrono::duration<double, std::milli>(t3 - t0).count();
    return true;
}

// ── worker 线程主循环 ─────────────────────────────────────────
void RknnPool::worker_loop(int worker_id) {
    while (true) {
        std::pair<int, cv::Mat> task;
        {
            std::unique_lock<std::mutex> lk(task_mtx_);
            task_cv_.wait(lk, [this]() { return stop_ || !task_queue_.empty(); });
            if (stop_ && task_queue_.empty()) return;
            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        TaskResult tr;
        tr.task_id = task.first;
        // worker_id 即 context 索引; instance 0 需与 infer_sync(0,...) 互斥
        std::unique_lock<std::mutex> dev_lk(dev_mtx_, std::defer_lock);
        if (worker_id == 0) dev_lk.lock();
        bool ok = process_on_context(worker_id, task.second,
                                     tr.annotated, tr.result, tr.timing);
        (void)ok;
        // worker 释放锁后再写结果
        if (worker_id == 0) dev_lk.unlock();

        {
            std::lock_guard<std::mutex> lk(results_mtx_);
            results_.emplace(tr.task_id, std::move(tr));
        }
        results_cv_.notify_all();
    }
}

}  // namespace yolo
