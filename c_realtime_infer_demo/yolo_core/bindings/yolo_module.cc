// yolo_module.cc — pybind11 绑定层
//
// 模块名: yolo_core (导入: `import yolo_core`)
//
// 绑定:
//   - YoloInferEngine: init / submit_frame / get_result / compute_deviation / release
//   - DetectBox / DetectResult / TimingMs / DeviationResult (转 Python dict)
//   - cv::Mat ↔ numpy.ndarray (uint8, BGR) 零拷贝互转
//
// GIL: submit_frame / get_result / compute_deviation 在 C++ 内部释放 GIL,
//      使 QThread 的事件循环不被阻塞 (对齐方案文档 §"GIL" 假设)。
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstring>
#include <opencv2/core.hpp>

#include "yolo_engine.h"
#include "types.h"

namespace py = pybind11;

// ── cv::Mat ↔ numpy.ndarray 转换辅助 ──────────────────────────
// 仅支持连续 CV_8UC3 (BGR), 与 Python 侧 cv2.imread / cap.read() 输出一致。
static cv::Mat ndarray_to_mat_bgr(py::array_t<uint8_t> arr) {
    py::buffer_info buf = arr.request();
    if (buf.format != py::format_descriptor<uint8_t>::format()) {
        throw std::runtime_error("yolo_core: expected uint8 ndarray");
    }
    if (buf.ndim != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("yolo_core: expected HxWx3 uint8 ndarray (BGR)");
    }
    int rows = (int)buf.shape[0];
    int cols = (int)buf.shape[1];
    // 注意: numpy 数组可能是非连续的, 用 cv::Mat 步长构造确保正确
    cv::Mat mat(rows, cols, CV_8UC3, (unsigned char*)buf.ptr,
                (size_t)buf.strides[0]);
    return mat.clone();  // 拷贝, 避免生命周期问题
}

static py::array_t<uint8_t> mat_to_ndarray_copy(const cv::Mat& mat) {
    if (mat.empty()) return py::array_t<uint8_t>();
    cv::Mat cont = mat.isContinuous() ? mat : mat.clone();
    int rows = cont.rows, cols = cont.cols, ch = cont.channels();
    auto arr = py::array_t<uint8_t>({rows, cols, ch});
    auto buf = arr.request();
    std::memcpy(buf.ptr, cont.data, (size_t)rows * cols * ch * sizeof(uint8_t));
    return arr;
}

// ── struct → dict 转换 ────────────────────────────────────────
static py::dict detect_box_to_dict(const yolo::DetectBox& b) {
    py::dict d;
    d["left"]       = b.left;
    d["top"]        = b.top;
    d["right"]      = b.right;
    d["bottom"]     = b.bottom;
    d["confidence"] = b.confidence;
    d["cls_id"]     = b.cls_id;
    return d;
}

static py::dict timing_to_dict(const yolo::TimingMs& t) {
    py::dict d;
    d["preprocess_ms"]  = t.preprocess;
    d["inference_ms"]   = t.inference;
    d["postprocess_ms"] = t.postprocess;
    d["total_ms"]       = t.total;
    return d;
}

static py::object deviation_to_object(const yolo::DeviationResult& r) {
    if (!r.has_person) return py::none();
    py::dict d;
    d["dev_x"]  = r.dev_x;
    d["dev_y"]  = r.dev_y;
    d["left"]   = r.left;
    d["top"]    = r.top;
    d["right"]  = r.right;
    d["bottom"] = r.bottom;
    return d;
}

PYBIND11_MODULE(yolo_core, m) {
    m.doc() = "C++ RKNN YOLOv8 inference engine (pybind11 binding)";

    // ── DetectBox ─────────────────────────────────────────────
    py::class_<yolo::DetectBox>(m, "DetectBox")
        .def_readwrite("left",       &yolo::DetectBox::left)
        .def_readwrite("top",        &yolo::DetectBox::top)
        .def_readwrite("right",      &yolo::DetectBox::right)
        .def_readwrite("bottom",     &yolo::DetectBox::bottom)
        .def_readwrite("confidence", &yolo::DetectBox::confidence)
        .def_readwrite("cls_id",     &yolo::DetectBox::cls_id);

    // ── DetectResult ──────────────────────────────────────────
    py::class_<yolo::DetectResult>(m, "DetectResult")
        .def_readwrite("count", &yolo::DetectResult::count)
        .def_readwrite("boxes", &yolo::DetectResult::boxes);

    // ── TimingMs ──────────────────────────────────────────────
    py::class_<yolo::TimingMs>(m, "TimingMs")
        .def_readwrite("preprocess",  &yolo::TimingMs::preprocess)
        .def_readwrite("inference",   &yolo::TimingMs::inference)
        .def_readwrite("postprocess", &yolo::TimingMs::postprocess)
        .def_readwrite("total",       &yolo::TimingMs::total);

    // ── DeviationResult ───────────────────────────────────────
    py::class_<yolo::DeviationResult>(m, "DeviationResult")
        .def_readwrite("has_person", &yolo::DeviationResult::has_person)
        .def_readwrite("dev_x",      &yolo::DeviationResult::dev_x)
        .def_readwrite("dev_y",      &yolo::DeviationResult::dev_y)
        .def_readwrite("left",       &yolo::DeviationResult::left)
        .def_readwrite("top",        &yolo::DeviationResult::top)
        .def_readwrite("right",      &yolo::DeviationResult::right)
        .def_readwrite("bottom",     &yolo::DeviationResult::bottom);

    // ── YoloInferEngine ───────────────────────────────────────
    py::class_<yolo::YoloInferEngine>(m, "YoloInferEngine")
        .def(py::init<const std::string&, int, float, float>(),
             py::arg("model_path"),
             py::arg("tpes") = 4,
             py::arg("obj_thresh") = 0.75f,
             py::arg("nms_thresh") = 0.6f)
        .def("init", [](yolo::YoloInferEngine& self) { return self.init(); })
        .def("submit_frame",
             [](yolo::YoloInferEngine& self, py::array_t<uint8_t> frame) {
                 cv::Mat bgr = ndarray_to_mat_bgr(frame);
                 py::gil_scoped_release release;
                 return self.submit_frame(bgr);
             },
             py::arg("frame").noconvert(),
             "Submit a BGR frame (HxWx3 uint8 ndarray). Returns task_id.")
        .def("get_result",
             [](yolo::YoloInferEngine& self, int timeout_ms) -> py::object {
                 cv::Mat annotated;
                 yolo::DetectResult boxes;
                 yolo::TimingMs timing;
                 bool ok;
                 {
                     py::gil_scoped_release release;
                     ok = self.get_result(annotated, boxes, timing, timeout_ms);
                 }
                 if (!ok) {
                     return py::make_tuple(false, py::none(), py::none(), py::none());
                 }
                 // 把 boxes 转成 list[dict]
                 py::list box_list;
                 for (int i = 0; i < boxes.count; i++) {
                     box_list.append(detect_box_to_dict(boxes.boxes[i]));
                 }
                 return py::make_tuple(true, mat_to_ndarray_copy(annotated),
                                       box_list, timing_to_dict(timing));
             },
             py::arg("timeout_ms") = -1,
             "Get next result. Returns (ok, annotated_ndarray, boxes_list, timing_dict).")
        .def("compute_deviation",
             [](yolo::YoloInferEngine& self, py::array_t<uint8_t> frame) {
                 cv::Mat bgr = ndarray_to_mat_bgr(frame);
                 yolo::DeviationResult r;
                 {
                     py::gil_scoped_release release;
                     r = self.compute_deviation(bgr);
                 }
                 return deviation_to_object(r);
             },
             py::arg("frame").noconvert(),
             "Synchronous second-pass inference for person deviation. "
             "Returns dict {dev_x,dev_y,left,top,right,bottom} or None.")
        .def("release", [](yolo::YoloInferEngine& self) { self.release(); })
        .def("tpes",       &yolo::YoloInferEngine::tpes)
        .def("obj_thresh", &yolo::YoloInferEngine::obj_thresh)
        .def("nms_thresh", &yolo::YoloInferEngine::nms_thresh);

    m.attr("__version__") = "1.0.0";
}
