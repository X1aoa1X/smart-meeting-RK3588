// rknn_pool.h — RKNN 推理池 (多上下文 + 多线程)
//
// 替代 Python 侧 rknnpool/rknnpool_ld.py 的 rknnPoolExecutor:
//   - 持有 TPEs 个 RKNNContext, 每个 pin 到一个 NPU 核心 (0/1/2 轮转)
//   - TPEs 个 worker 线程, 每个线程独占一个 context (无锁并行)
//   - submit() 异步入队, get() 按 FIFO 取回结果 (内部用 task_id 重排)
//   - infer_sync() 同步推理指定实例 (供偏差二次推理使用, 加锁保护 ctx[0])
//
// 与 Python 行为一致: 最多 TPEs 个推理任务并行, get() 阻塞等待最早提交的任务。
#pragma once

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>

#include "types.h"
#include "rknn_context.h"

namespace yolo {

class RknnPool {
public:
    RknnPool(const std::string& model_path, int tpes,
             float obj_thresh, float nms_thresh);
    ~RknnPool();

    RknnPool(const RknnPool&) = delete;
    RknnPool& operator=(const RknnPool&) = delete;

    // 加载模型 TPEs 份, 每个 pin 到 NPU 核心 0/1/2 轮转, 各预热一次。
    // 成功返回 true。
    bool init();

    // 异步提交一帧 (BGR), 返回 task_id (单调递增)。
    // 内部 round-robin 选择 worker, 但任意空闲 worker 均可处理任意任务。
    int submit(cv::Mat bgr_frame);

    // 阻塞等待最早提交且尚未取回的任务完成, 取回结果。
    // timeout_ms < 0 表示无限等待; 返回 false 表示超时或池已停止且无结果。
    bool get(cv::Mat& annotated, DetectResult& result, TimingMs& timing,
             int timeout_ms = -1);

    // 同步推理: 在指定 instance_idx 的 context 上跑一帧, 加 dev_mtx_ 保护
    // (instance 0 与 worker 0 共享, 需互斥)。供偏差二次推理使用。
    bool infer_sync(int instance_idx, const cv::Mat& bgr_frame,
                    DetectResult& result, TimingMs& timing);

    // 当前已提交但尚未取回的任务数
    int pending() const;

    // 停止 worker 并释放 RKNN 资源 (可重复调用)
    void release();

    float obj_thresh() const { return obj_thresh_; }
    float nms_thresh() const { return nms_thresh_; }
    int   tpes() const { return tpes_; }

private:
    struct TaskResult {
        int          task_id = -1;
        cv::Mat      annotated;
        DetectResult result;
        TimingMs     timing;
    };

    // worker 线程主循环: 每个 worker 拥有 contexts_[worker_id]
    void worker_loop(int worker_id);

    // 在指定 context 上完成一帧完整推理 (preprocess→run→post→draw),
    // 把结果写入 annotated / result / timing。
    bool process_on_context(int instance_idx, const cv::Mat& bgr_frame,
                            cv::Mat& annotated, DetectResult& result,
                            TimingMs& timing);

    std::string model_path_;
    int         tpes_;
    float       obj_thresh_;
    float       nms_thresh_;

    std::vector<std::unique_ptr<RKNNContext>> contexts_;
    std::vector<std::thread>                  workers_;

    // 任务队列: {task_id, BGR frame}
    std::queue<std::pair<int, cv::Mat>> task_queue_;
    std::mutex                          task_mtx_;
    std::condition_variable             task_cv_;
    std::atomic<bool>                   stop_{false};

    // 结果集合: 按 task_id 索引; get() 等待 next_get_id_ 出现
    std::map<int, TaskResult> results_;
    std::mutex                results_mtx_;
    std::condition_variable   results_cv_;
    int                       next_get_id_   = 0;
    std::atomic<int>          next_task_id_{0};

    // 保护 contexts_[0] 的互斥锁 (worker 0 与 infer_sync(0,...) 共用)
    std::mutex dev_mtx_;
};

}  // namespace yolo
