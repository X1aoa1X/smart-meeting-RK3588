// yolo_engine.h — 高层推理引擎 (供 pybind11 直接绑定)
//
// 封装 RknnPool + 绘制 + 人体偏差二次推理, 对 Python 侧暴露最简接口。
// 替代 Python 侧 func_yolov8_optimize.py + rknnpool_ld.py 的组合。
//
// 接口契约 (与 demo/core/yolo_camera.py 的 C++ 替换方案对齐):
//   - init():                       构造池 + 加载 COCO 标签
//   - submit_frame(bgr) -> task_id: 异步入队
//   - get_result(annotated, boxes, timing) -> bool: 取回一帧结果
//   - compute_deviation(bgr) -> DeviationResult: 同步二次推理取人体偏差
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "types.h"
#include "rknn_pool.h"

namespace yolo {

class YoloInferEngine {
public:
    YoloInferEngine(const std::string& model_path, int tpes = 4,
                    float obj_thresh = 0.75f, float nms_thresh = 0.6f);
    ~YoloInferEngine();

    YoloInferEngine(const YoloInferEngine&) = delete;
    YoloInferEngine& operator=(const YoloInferEngine&) = delete;

    // 加载模型 + 预热。成功返回 true。
    bool init();

    // 异步提交一帧 BGR 图像, 返回 task_id。
    int submit_frame(const cv::Mat& bgr);

    // 阻塞取回最早提交且尚未取回的结果。
    // annotated: 带标注的 BGR 帧 (原图尺寸); boxes: 检测框; timing: 各阶段耗时。
    // 超时返回 false (timeout_ms < 0 表示无限等待)。
    bool get_result(cv::Mat& annotated, DetectResult& boxes, TimingMs& timing,
                    int timeout_ms = -1);

    // 同步二次推理: 在 pool 实例 0 上跑一帧, 取最大 person 框,
    // 计算 dev_x/dev_y (中心点相对画面中心的归一化偏移)。
    // 无 person 时 has_person=false。
    DeviationResult compute_deviation(const cv::Mat& bgr);

    // 释放资源 (可重复调用)
    void release();

    int   tpes()       const { return tpes_; }
    float obj_thresh() const { return obj_thresh_; }
    float nms_thresh() const { return nms_thresh_; }
    const std::vector<std::string>& labels() const { return labels_; }

private:
    std::string model_path_;
    int         tpes_;
    float       obj_thresh_;
    float       nms_thresh_;
    std::unique_ptr<RknnPool>         pool_;
    std::vector<std::string>          labels_;
};

}  // namespace yolo
