// yolo_camera_fast.cc — 多线程 + 硬解优化版 YOLOv8n 摄像头推理
//
// 优化:
//   1. 可选分辨率: -r 640x480|960x540|1280x720 (默认 640×480)
//   2. 多 NPU 核并行: -t N (默认 3)
//   3. MPP 硬件 JPEG 解码 + RGA 零拷贝: --hw-decode
//
// 用法:
//   ./yolo_camera_fast ./best.rknn [-r 640x480] [-t 3] [--hw-decode] [--no-display]

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <errno.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <linux/videodev2.h>

extern "C" {
#include "rknn_api.h"
}

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// ── 硬件加速头 ────────────────────────────────────────────
// conditionally included in HW path section

/* ============================================================
 *  配置常量
 * ============================================================ */
static constexpr int   MODEL_WIDTH       = 640;
static constexpr int   MODEL_HEIGHT      = 640;
static constexpr int   OBJ_CLASS_NUM     = 80;
static constexpr int   OBJ_NUMB_MAX_SIZE = 128;
static constexpr float BOX_THRESH        = 0.25f;
static constexpr float NMS_THRESH        = 0.45f;
static constexpr int   MAX_NPU_CORES     = 3;

static const char* DEFAULT_CAM_DEV = "/dev/video21";

// ── 分辨率预设 ────────────────────────────────────────────
struct ResConfig { int w, h, fps; };
static const ResConfig RES_PRESETS[] = {
    {640,  480, 30},
    {960,  540, 30},
    {1280, 720, 30},
};
static constexpr int NUM_RES_PRESETS = sizeof(RES_PRESETS)/sizeof(ResConfig);

// ── COCO 类别 ──────────────────────────────────────────────
static const char* COCO_CLASSES[OBJ_CLASS_NUM] = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
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
    struct timeval tv; gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
/* ============================================================
 *  数据结构
 * ============================================================ */
struct LetterBox { int x_pad, y_pad; float scale; };
struct DetectBox { int left, top, right, bottom; float confidence; int cls_id; };
struct DetectResult { int count; DetectBox results[OBJ_NUMB_MAX_SIZE]; };

// ── 线程间帧数据 ──────────────────────────────────────────
struct InFrame {
    int    frame_id;
    int    cam_w, cam_h;
    cv::Mat bgr;            // CPU 路径: 解码后的 BGR; HW 路径: 显示用
    // HW 路径
    bool   use_hw = false;
    void*  jpeg_data = nullptr;
    size_t jpeg_size = 0;
};

struct InferResult {
    int          frame_id;
    bool         valid = false;
    DetectResult det;
    cv::Mat      orig;       // 显示用原图
    LetterBox    lb;
    double       t_pre = 0, t_inf = 0, t_post = 0;
};

/* ============================================================
 *  线程安全有界队列
 * ============================================================ */
template <typename T>
class SafeQueue {
    std::queue<T>           q_;
    mutable std::mutex      mtx_;
    std::condition_variable cv_;
    size_t                  max_size_;
    bool                    done_ = false;
public:
    explicit SafeQueue(size_t max_sz = 4) : max_size_(max_sz) {}

    void push(T val) {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [this]{ return q_.size() < max_size_ || done_; });
        if (done_) return;
        q_.push(std::move(val));
        lk.unlock();
        cv_.notify_all();
    }

    bool pop(T& val, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lk(mtx_);
        bool ok = cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                               [this]{ return !q_.empty() || done_; });
        if (!ok || q_.empty()) return false;
        val = std::move(q_.front()); q_.pop();
        lk.unlock();
        cv_.notify_all();
        return true;
    }

    void set_done() {
        { std::lock_guard<std::mutex> lk(mtx_); done_ = true; }
        cv_.notify_all();
    }
};

/* ============================================================
 *  NPU 单核心上下文
 * ============================================================ */
struct RKNNPerCore {
    rknn_context           ctx = 0;
    int                    core_id = 0;
    rknn_input_output_num  io_num;
    rknn_tensor_attr*      input_attrs = nullptr;
    rknn_tensor_attr*      output_attrs = nullptr;
    rknn_tensor_format     input_fmt = RKNN_TENSOR_NHWC;
    int                    model_w = MODEL_WIDTH;
    int                    model_h = MODEL_HEIGHT;
    int                    model_c = 3;
    int                    dfl_len = 16;
    int                    out_per_branch = 2;
    bool                   is_quant = false;

    ~RKNNPerCore() {
        free(input_attrs); free(output_attrs);
        if (ctx) rknn_destroy(ctx);
    }
};

/* ============================================================
 *  RNPP 模型加载 (per-core)
 * ============================================================ */
static int load_model_file(const char* path, unsigned char** out, int* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    *out = (unsigned char*)malloc(sz);
    if (!*out) { fclose(f); return -1; }
    if (fread(*out, 1, (size_t)sz, f) != (size_t)sz) { free(*out); fclose(f); return -1; }
    fclose(f);
    *out_len = (int)sz;
    return 0;
}

static int init_rknn_core(const char* model_path, RKNNPerCore* c, int core_id) {
    unsigned char* model = nullptr; int model_len;
    if (load_model_file(model_path, &model, &model_len) < 0) return -1;

    int ret = rknn_init(&c->ctx, model, (uint32_t)model_len, 0, nullptr);
    free(model);
    if (ret < 0) { fprintf(stderr,"rknn_init core%d fail! ret=%d\n",core_id,ret); return -1; }

    // 绑定 NPU 核心
    rknn_core_mask mask;
    switch (core_id) {
        case 0: mask = RKNN_NPU_CORE_0; break;
        case 1: mask = RKNN_NPU_CORE_1; break;
        case 2: mask = RKNN_NPU_CORE_2; break;
        default: mask = RKNN_NPU_CORE_AUTO; break;
    }
    ret = rknn_set_core_mask(c->ctx, mask);
    if (ret != RKNN_SUCC) fprintf(stderr,"rknn_set_core_mask core%d fail\n",core_id);
    c->core_id = core_id;

    ret = rknn_query(c->ctx, RKNN_QUERY_IN_OUT_NUM, &c->io_num, sizeof(c->io_num));
    if (ret != RKNN_SUCC) return -1;

    c->input_attrs = (rknn_tensor_attr*)malloc(c->io_num.n_input * sizeof(rknn_tensor_attr));
    memset(c->input_attrs, 0, c->io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < c->io_num.n_input; i++) {
        c->input_attrs[i].index = i;
        rknn_query(c->ctx, RKNN_QUERY_INPUT_ATTR, &c->input_attrs[i], sizeof(rknn_tensor_attr));
    }

    c->output_attrs = (rknn_tensor_attr*)malloc(c->io_num.n_output * sizeof(rknn_tensor_attr));
    memset(c->output_attrs, 0, c->io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < c->io_num.n_output; i++) {
        c->output_attrs[i].index = i;
        rknn_query(c->ctx, RKNN_QUERY_OUTPUT_ATTR, &c->output_attrs[i], sizeof(rknn_tensor_attr));
    }

    c->is_quant = (c->output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
                   && c->output_attrs[0].type == RKNN_TENSOR_INT8);
    c->dfl_len = c->output_attrs[0].dims[1] / 4;
    c->out_per_branch = c->io_num.n_output / 3;

    c->input_fmt = c->input_attrs[0].fmt;
    if (c->input_fmt == RKNN_TENSOR_NCHW) {
        c->model_c = c->input_attrs[0].dims[1];
        c->model_h = c->input_attrs[0].dims[2];
        c->model_w = c->input_attrs[0].dims[3];
    } else {
        c->model_h = c->input_attrs[0].dims[1];
        c->model_w = c->input_attrs[0].dims[2];
        c->model_c = c->input_attrs[0].dims[3];
    }

    // Warmup
    size_t in_size = c->model_w * c->model_h * c->model_c;
    uint8_t* buf = (uint8_t*)malloc(in_size);
    memset(buf, 128, in_size);
    rknn_input in; memset(&in,0,sizeof(in));
    in.index=0; in.type=RKNN_TENSOR_UINT8; in.fmt=c->input_fmt; in.size=in_size; in.buf=buf;
    rknn_inputs_set(c->ctx,1,&in);
    rknn_run(c->ctx,nullptr);
    std::vector<rknn_output> out(c->io_num.n_output);
    for (uint32_t i=0;i<c->io_num.n_output;i++){out[i].index=i;out[i].want_float=!c->is_quant;}
    rknn_outputs_get(c->ctx,c->io_num.n_output,out.data(),nullptr);
    rknn_outputs_release(c->ctx,c->io_num.n_output,out.data());
    free(buf);

    printf("  NPU core%d init OK (quant=%d, dfl=%d, in=%dx%dx%d %s)\n",
           core_id, c->is_quant, c->dfl_len, c->model_w, c->model_h, c->model_c,
           c->input_fmt==RKNN_TENSOR_NCHW?"NCHW":"NHWC");
    return 0;
}

/* ============================================================
 *  CPU 预处理 (BGR → RGB letterbox)
 * ============================================================ */
static void preprocess_cpu(const cv::Mat& bgr, uint8_t* rgb_out,
                           LetterBox& lb, int mw, int mh) {
    int iw=bgr.cols, ih=bgr.rows;
    float scale = std::min((float)mw/iw, (float)mh/ih);
    lb.scale = scale;
    int nw=(int)std::round(iw*scale), nh=(int)std::round(ih*scale);
    lb.x_pad=(mw-nw)/2; lb.y_pad=(mh-nh)/2;

    cv::Mat rs, rgb;
    cv::resize(bgr, rs, cv::Size(nw,nh), 0,0, cv::INTER_LINEAR);
    cv::cvtColor(rs, rgb, cv::COLOR_BGR2RGB);
    cv::Mat lb_img(mh, mw, CV_8UC3, cv::Scalar(0,0,0));
    rgb.copyTo(lb_img(cv::Rect(lb.x_pad, lb.y_pad, nw, nh)));
    memcpy(rgb_out, lb_img.data, mw*mh*3);
}

/* ============================================================
 *  YOLOv8 后处理 (DFL + NMS)
 * ============================================================ */
static void compute_dfl(const float* t, int dfl_len, float* box) {
    for (int b=0;b<4;b++){
        float exp_sum=0, acc=0;
        float ex[dfl_len];
        for(int i=0;i<dfl_len;i++){ex[i]=expf(t[i+b*dfl_len]);exp_sum+=ex[i];}
        for(int i=0;i<dfl_len;i++) acc+=(ex[i]/exp_sum)*i;
        box[b]=acc;
    }
}
static inline float deqnt_i8(int8_t q, int32_t zp, float s){return((float)q-(float)zp)*s;}
static inline int8_t qnt_f32_i8(float f,int32_t zp,float s){float d=f/s+zp;if(d<=-128)return-128;if(d>=127)return 127;return(int8_t)d;}

static int process_branch_i8(
    int8_t* box_t, int32_t box_zp, float box_s,
    int8_t* score_t, int32_t score_zp, float score_s,
    int8_t* ssum_t, int32_t ssum_zp, float ssum_s,
    int gh, int gw, int stride, int dfl_len,
    std::vector<float>& boxes, std::vector<float>& probs,
    std::vector<int>& cls_ids, float thresh)
{
    int gl=gh*gw, vc=0;
    int8_t st_i8=qnt_f32_i8(thresh,score_zp,score_s);
    int8_t sst_i8=qnt_f32_i8(thresh,ssum_zp,ssum_s);
    for(int y=0;y<gh;y++) for(int x=0;x<gw;x++){
        int off=y*gw+x;
        if(ssum_t&&ssum_t[off]<sst_i8)continue;
        int mc=-1; int8_t ms=-score_zp;
        for(int c=0;c<OBJ_CLASS_NUM;c++){
            int idx=off+c*gl;
            if(score_t[idx]>st_i8&&score_t[idx]>ms){ms=score_t[idx];mc=c;}
        }
        if(ms<=st_i8)continue;
        float bf[dfl_len*4];
        for(int k=0;k<dfl_len*4;k++)bf[k]=deqnt_i8(box_t[off+k*gl],box_zp,box_s);
        float db[4];compute_dfl(bf,dfl_len,db);
        float x1=(-db[0]+(float)x+0.5f)*stride;
        float y1=(-db[1]+(float)y+0.5f)*stride;
        float x2=( db[2]+(float)x+0.5f)*stride;
        float y2=( db[3]+(float)y+0.5f)*stride;
        boxes.push_back(x1);boxes.push_back(y1);boxes.push_back(x2-x1);boxes.push_back(y2-y1);
        probs.push_back(deqnt_i8(ms,score_zp,score_s));cls_ids.push_back(mc);vc++;
    }
    return vc;
}

static int process_branch_fp32(
    float* box_t, float* score_t, float* ssum_t,
    int gh, int gw, int stride, int dfl_len,
    std::vector<float>& boxes, std::vector<float>& probs,
    std::vector<int>& cls_ids, float thresh)
{
    int gl=gh*gw,vc=0;
    for(int y=0;y<gh;y++)for(int x=0;x<gw;x++){
        int off=y*gw+x;
        if(ssum_t&&ssum_t[off]<thresh)continue;
        int mc=-1;float ms=0;
        for(int c=0;c<OBJ_CLASS_NUM;c++){
            int idx=off+c*gl;
            if(score_t[idx]>thresh&&score_t[idx]>ms){ms=score_t[idx];mc=c;}
        }
        if(ms<=thresh)continue;
        float bf[dfl_len*4];
        for(int k=0;k<dfl_len*4;k++)bf[k]=box_t[off+k*gl];
        float db[4];compute_dfl(bf,dfl_len,db);
        float x1=(-db[0]+(float)x+0.5f)*stride;
        float y1=(-db[1]+(float)y+0.5f)*stride;
        float x2=( db[2]+(float)x+0.5f)*stride;
        float y2=( db[3]+(float)y+0.5f)*stride;
        boxes.push_back(x1);boxes.push_back(y1);boxes.push_back(x2-x1);boxes.push_back(y2-y1);
        probs.push_back(ms);cls_ids.push_back(mc);vc++;
    }
    return vc;
}

static float calc_iou(float x0,float y0,float w0,float h0,float x1,float y1,float w1,float h1){
    float ix=fmaxf(0,fminf(x0+w0,x1+w1)-fmaxf(x0,x1)+1);
    float iy=fmaxf(0,fminf(y0+h0,y1+h1)-fmaxf(y0,y1)+1);
    float inter=ix*iy,un=w0*h0+w1*h1-inter;
    return un<=0?0:inter/un;
}
static void sort_desc(std::vector<float>& s,int l,int r,std::vector<int>& idx){
    if(l>=r)return;
    float k=s[l];int ki=idx[l];int lo=l,hi=r;
    while(lo<hi){
        while(lo<hi&&s[hi]<=k)hi--;
        s[lo]=s[hi];idx[lo]=idx[hi];
        while(lo<hi&&s[lo]>=k)lo++;
        s[hi]=s[lo];idx[hi]=idx[lo];
    }
    s[lo]=k;idx[lo]=ki;sort_desc(s,l,lo-1,idx);sort_desc(s,lo+1,r,idx);
}
static void nms_per_cls(int vc,std::vector<float>& b,std::vector<int>& c,
    std::vector<int>& ord,int fc,float nms_t){
    for(int i=0;i<vc;i++){
        int n=ord[i];if(n==-1||c[n]!=fc)continue;
        for(int j=i+1;j<vc;j++){
            int m=ord[j];if(m==-1||c[m]!=fc)continue;
            if(calc_iou(b[n*4],b[n*4+1],b[n*4+2],b[n*4+3],
                        b[m*4],b[m*4+1],b[m*4+2],b[m*4+3])>nms_t)ord[j]=-1;
        }
    }
}

static int yolov8_post_process_fast(RKNNPerCore* c, rknn_output* outputs,
    LetterBox* lb, DetectResult* res, int mw, int mh) {
    memset(res,0,sizeof(*res));
    int opb=c->out_per_branch, dl=c->dfl_len;
    std::vector<float> boxes,probs;std::vector<int> cls_ids;int tv=0;
    for(int br=0;br<3;br++){
        void *ssum=nullptr;int32_t sszp=0;float sss=1;
        if(opb==3){int si=br*opb+2;ssum=outputs[si].buf;sszp=c->output_attrs[si].zp;sss=c->output_attrs[si].scale;}
        int bi=br*opb,si=br*opb+1;
        int gh=c->output_attrs[bi].dims[2],gw=c->output_attrs[bi].dims[3];
        int stride=mh/gh;
        if(c->is_quant)
            tv+=process_branch_i8((int8_t*)outputs[bi].buf,c->output_attrs[bi].zp,c->output_attrs[bi].scale,
                (int8_t*)outputs[si].buf,c->output_attrs[si].zp,c->output_attrs[si].scale,
                (int8_t*)ssum,sszp,sss,gh,gw,stride,dl,boxes,probs,cls_ids,BOX_THRESH);
        else tv+=process_branch_fp32((float*)outputs[bi].buf,(float*)outputs[si].buf,(float*)ssum,
            gh,gw,stride,dl,boxes,probs,cls_ids,BOX_THRESH);
    }
    if(tv<=0)return 0;
    std::vector<int> idx(tv);for(int i=0;i<tv;i++)idx[i]=i;
    sort_desc(probs,0,tv-1,idx);
    std::set<int> cs(cls_ids.begin(),cls_ids.end());
    for(int cl:cs)nms_per_cls(tv,boxes,cls_ids,idx,cl,NMS_THRESH);
    int oc=0;
    for(int i=0;i<tv;i++){
        if(idx[i]==-1||oc>=OBJ_NUMB_MAX_SIZE)continue;
        int n=idx[i];
        float x1=boxes[n*4+0]-(float)lb->x_pad;
        float y1=boxes[n*4+1]-(float)lb->y_pad;
        float x2=x1+boxes[n*4+2],y2=y1+boxes[n*4+3];
        res->results[oc].left  =(int)(clamp_i(x1,0,mw)/lb->scale);
        res->results[oc].top   =(int)(clamp_i(y1,0,mh)/lb->scale);
        res->results[oc].right =(int)(clamp_i(x2,0,mw)/lb->scale);
        res->results[oc].bottom=(int)(clamp_i(y2,0,mh)/lb->scale);
        res->results[oc].confidence=probs[n];res->results[oc].cls_id=cls_ids[n];oc++;
    }
    res->count=oc;
    return 0;
}

/* ============================================================
 *  画框
 * ============================================================ */
static void draw_results(cv::Mat& frame, const DetectResult& res) {
    for(int i=0;i<res.count;i++){
        const DetectBox& d=res.results[i];
        cv::Scalar color(0,255,0);
        cv::rectangle(frame,cv::Point(d.left,d.top),cv::Point(d.right,d.bottom),color,2);
        char lb[128];snprintf(lb,sizeof(lb),"%s %.1f%%",COCO_CLASSES[d.cls_id],d.confidence*100.f);
        int base;cv::Size ts=cv::getTextSize(lb,cv::FONT_HERSHEY_SIMPLEX,0.5,1,&base);
        int ty=d.top-ts.height-3;if(ty<0)ty=d.top+5;
        cv::rectangle(frame,cv::Point(d.left,ty),cv::Point(d.left+ts.width,ty+ts.height+3),color,cv::FILLED);
        cv::putText(frame,lb,cv::Point(d.left,ty+ts.height),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,0),1);
    }
}

/* ============================================================
 *  V4L2 直读 MJPEG (替代 OpenCV VideoCapture，获取真实 30fps)
 * ============================================================ */
class V4L2MjpegCapture {
    int fd_=-1;
    struct Buf { void* ptr; size_t len; };
    std::vector<Buf> bufs_;
    bool streaming_=false;
public:
    bool open(const char* dev, int w, int h) {
        fd_=::open(dev,O_RDWR);if(fd_<0){perror("V4L2 open");return false;}
        v4l2_format fmt={};fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_,VIDIOC_G_FMT,&fmt);
        fmt.fmt.pix.width=w;fmt.fmt.pix.height=h;
        fmt.fmt.pix.pixelformat=V4L2_PIX_FMT_MJPEG;fmt.fmt.pix.field=V4L2_FIELD_NONE;
        if(ioctl(fd_,VIDIOC_S_FMT,&fmt)<0){perror("VIDIOC_S_FMT");return false;}
        // 设置 30fps
        v4l2_streamparm parm={};parm.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parm.parm.capture.timeperframe.numerator=1;
        parm.parm.capture.timeperframe.denominator=30;
        ioctl(fd_,VIDIOC_S_PARM,&parm);
        printf("  V4L2: %dx%d MJPEG @ %d/%d fps\n",fmt.fmt.pix.width,fmt.fmt.pix.height,
               parm.parm.capture.timeperframe.denominator,
               parm.parm.capture.timeperframe.numerator);
        v4l2_requestbuffers req={};req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory=V4L2_MEMORY_MMAP;req.count=4;
        if(ioctl(fd_,VIDIOC_REQBUFS,&req)<0){perror("VIDIOC_REQBUFS");return false;}
        bufs_.resize(req.count);
        for(uint32_t i=0;i<req.count;i++){
            v4l2_buffer b={};b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;b.memory=V4L2_MEMORY_MMAP;b.index=i;
            ioctl(fd_,VIDIOC_QUERYBUF,&b);
            bufs_[i].ptr=mmap(nullptr,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,b.m.offset);
            bufs_[i].len=b.length;
            ioctl(fd_,VIDIOC_QBUF,&b);
        }
        int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;ioctl(fd_,VIDIOC_STREAMON,&type);
        streaming_=true;
        return true;
    }
    bool grab(const void** data, size_t* size) {
        v4l2_buffer b={};b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;b.memory=V4L2_MEMORY_MMAP;
        if(ioctl(fd_,VIDIOC_DQBUF,&b)<0)return false;
        *data=bufs_[b.index].ptr;*size=b.bytesused;
        ioctl(fd_,VIDIOC_QBUF,&b);
        return true;
    }
    void close() {
        if(streaming_){int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;ioctl(fd_,VIDIOC_STREAMOFF,&type);}
        for(auto& b:bufs_){if(b.ptr)munmap(b.ptr,b.len);}
        if(fd_>=0){::close(fd_);fd_=-1;}
    }
    ~V4L2MjpegCapture(){close();}
};

/* ============================================================
 *  线程函数 (CPU 路径)
 * ============================================================ */

// ── V4L2直接采集线程 (绕过 OpenCV V4L2 后端, 获取真实 30fps) ──
static void capture_thread_v4l2(V4L2MjpegCapture* v4l2, SafeQueue<InFrame>* fq,
                                 std::atomic<bool>* running, int cam_w, int cam_h) {
    int fid=0;
    while(*running){
        InFrame f; f.frame_id=fid++; f.cam_w=cam_w; f.cam_h=cam_h; f.use_hw=false;
        const void* data=nullptr; size_t sz=0;
        if(!v4l2->grab(&data,&sz)){usleep(5000);continue;}
        cv::Mat jpg_buf(1, sz, CV_8UC1, (void*)data);
        f.bgr = cv::imdecode(jpg_buf, cv::IMREAD_COLOR);
        if(f.bgr.empty()) continue;
        fq->push(std::move(f));
    }
    fq->set_done();
}

// ── Worker 线程 ────────────────────────────────────────────
struct WorkerCtx {
    RKNNPerCore* c;
    int mw,mh;
};

static void worker_thread_cpu(WorkerCtx* wctx, SafeQueue<InFrame>* fq,
                               SafeQueue<InferResult>* rq, std::atomic<bool>* running) {
    RKNNPerCore* c=wctx->c;
    int mw=wctx->mw, mh=wctx->mh;
    size_t in_sz=mw*mh*c->model_c;
    std::vector<rknn_output> rk_out(c->io_num.n_output);
    std::vector<uint8_t> rgb_buf(in_sz);

    while(*running){
        InFrame f;
        if(!fq->pop(f,100))continue;

        InferResult res; res.frame_id=f.frame_id; res.orig=f.bgr.clone();
        res.valid=false;

        double t0=now_sec();
        // 预处理
        preprocess_cpu(f.bgr, rgb_buf.data(), res.lb, mw, mh);
        double t1=now_sec(); res.t_pre=t1-t0;

        // 推理
        rknn_input in; memset(&in,0,sizeof(in));
        in.index=0;in.type=RKNN_TENSOR_UINT8;in.fmt=c->input_fmt;in.size=in_sz;in.buf=rgb_buf.data();
        if(rknn_inputs_set(c->ctx,1,&in)<0)continue;
        if(rknn_run(c->ctx,nullptr)<0)continue;
        double t2=now_sec(); res.t_inf=t2-t1;

        // 获取输出
        memset(rk_out.data(),0,rk_out.size()*sizeof(rknn_output));
        for(uint32_t i=0;i<c->io_num.n_output;i++){rk_out[i].index=i;rk_out[i].want_float=!c->is_quant;}
        if(rknn_outputs_get(c->ctx,c->io_num.n_output,rk_out.data(),nullptr)<0)continue;

        // 后处理
        yolov8_post_process_fast(c, rk_out.data(), &res.lb, &res.det, mw, mh);
        rknn_outputs_release(c->ctx,c->io_num.n_output,rk_out.data());
        double t3=now_sec(); res.t_post=t3-t2;
        res.valid=true;

        rq->push(std::move(res));
    }
}

// ── 显示线程 ───────────────────────────────────────────────
static void display_thread_func(SafeQueue<InferResult>* rq, std::atomic<bool>* running,
                                 bool no_display) {
    cv::namedWindow("YOLOv8n Fast", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLOv8n Fast", 960, 540);

    double last_t=now_sec(); int fc=0; float fps=0;
    double sum_pre=0,sum_inf=0,sum_post=0;

    while(*running){
        InferResult res;
        if(!rq->pop(res,100))continue;
        if(!res.valid)continue;

        if(!no_display && !res.orig.empty()){
            draw_results(res.orig, res.det);
            char fs[128]; snprintf(fs,sizeof(fs),
                "FPS:%.1f|d:%d|pre:%.1f inf:%.1f post:%.1f ms",
                fps,res.det.count,res.t_pre*1000,res.t_inf*1000,res.t_post*1000);
            cv::putText(res.orig,fs,cv::Point(10,25),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(0,255,255),2);
            cv::Mat disp;
            if(res.orig.cols>960){double ds=960.0/res.orig.cols;cv::resize(res.orig,disp,cv::Size(960,(int)(res.orig.rows*ds)));}
            else disp=res.orig;
            cv::imshow("YOLOv8n Fast",disp);
            int key=cv::waitKey(1)&0xFF;
            if(key==27||key=='q'){*running=false;break;}
        }

        fc++; sum_pre+=res.t_pre; sum_inf+=res.t_inf; sum_post+=res.t_post;
        if(fc>=15){
            double now=now_sec();fps=(float)fc/(float)(now-last_t);
            printf("  FPS:%5.1f|pre:%5.1fms inf:%5.1fms post:%5.1fms|d:%d\n",
                   fps,sum_pre/fc*1000,sum_inf/fc*1000,sum_post/fc*1000,res.det.count);
            fc=0;last_t=now;sum_pre=sum_inf=sum_post=0;
        }
    }
    if(!no_display)cv::destroyAllWindows();
}

/* ============================================================
 *  Phase 4: 硬件 MPP JPEG 解码 + RGA 预处理
 * ============================================================ */
#ifdef ENABLE_HW_DECODE

#include <rk_mpi.h>
#include <mpp_frame.h>
#include <mpp_packet.h>
#include <mpp_buffer.h>
#include <rk_type.h>

#include <im2d.h>
#include <RgaUtils.h>
#include <rga.h>

#include <linux/dma-heap.h>

// ── dma_heap 缓冲区分配 ─────────────────────────────────────
struct dma_heap_alloc_data {
    __u64 len;
    __u32 fd;
    __u32 fd_flags;
    __u64 heap_flags;
};
#define DMA_HEAP_IOC_ALLOC _IOWR('H', 0x00, struct dma_heap_alloc_data)

static int dma_heap_alloc(size_t size) {
    const char* paths[] = {
        "/dev/dma_heap/system-uncached",
        "/dev/dma_heap/system",
        "/dev/dma_heap/cma-uncached",
        "/dev/dma_heap/cma",
    };
    for (auto p : paths) {
        int hfd = ::open(p, O_RDWR);
        if (hfd < 0) continue;
        struct dma_heap_alloc_data alloc = {};
        alloc.len = size;
        alloc.fd_flags = O_RDWR | O_CLOEXEC;
        if (::ioctl(hfd, DMA_HEAP_IOC_ALLOC, &alloc) < 0) {
            ::close(hfd);
            continue;
        }
        ::close(hfd);
        return alloc.fd;
    }
    return -1;
}

// ── V4L2 直读 MJPEG ────────────────────────────────────────
class V4L2MjpegCapture {
    int fd_=-1;
    struct Buf { void* ptr; size_t len; };
    std::vector<Buf> bufs_;
    bool streaming_=false;
public:
    bool open(const char* dev, int w, int h) {
        fd_=::open(dev,O_RDWR);if(fd_<0){perror("V4L2 open");return false;}
        v4l2_format fmt={};fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_,VIDIOC_G_FMT,&fmt);
        fmt.fmt.pix.width=w;fmt.fmt.pix.height=h;
        fmt.fmt.pix.pixelformat=V4L2_PIX_FMT_MJPEG;fmt.fmt.pix.field=V4L2_FIELD_NONE;
        if(ioctl(fd_,VIDIOC_S_FMT,&fmt)<0){perror("VIDIOC_S_FMT");return false;}
        printf("  V4L2 HW: %dx%d MJPEG\n",fmt.fmt.pix.width,fmt.fmt.pix.height);

        v4l2_requestbuffers req={};req.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory=V4L2_MEMORY_MMAP;req.count=4;
        if(ioctl(fd_,VIDIOC_REQBUFS,&req)<0){perror("VIDIOC_REQBUFS");return false;}
        bufs_.resize(req.count);
        for(uint32_t i=0;i<req.count;i++){
            v4l2_buffer b={};b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;b.memory=V4L2_MEMORY_MMAP;b.index=i;
            ioctl(fd_,VIDIOC_QUERYBUF,&b);
            bufs_[i].ptr=mmap(nullptr,b.length,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,b.m.offset);
            bufs_[i].len=b.length;
            ioctl(fd_,VIDIOC_QBUF,&b);
        }
        int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;ioctl(fd_,VIDIOC_STREAMON,&type);
        streaming_=true;
        return true;
    }
    bool grab(const void** data, size_t* size) {
        v4l2_buffer b={};b.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;b.memory=V4L2_MEMORY_MMAP;
        if(ioctl(fd_,VIDIOC_DQBUF,&b)<0)return false;
        *data=bufs_[b.index].ptr;*size=b.bytesused;
        // requeue immediately so driver can fill next frame
        ioctl(fd_,VIDIOC_QBUF,&b);
        return true;
    }
    void close() {
        if(streaming_){int type=V4L2_BUF_TYPE_VIDEO_CAPTURE;ioctl(fd_,VIDIOC_STREAMOFF,&type);}
        for(auto& b:bufs_){if(b.ptr)munmap(b.ptr,b.len);}
        if(fd_>=0){::close(fd_);fd_=-1;}
    }
    ~V4L2MjpegCapture(){close();}
};

// ── MPP JPEG 解码器 ────────────────────────────────────────
class MppJpegDecoder {
    MppCtx ctx_=nullptr;MppApi* api_=nullptr;
    std::mutex mtx_;
public:
    bool init() {
        MPP_RET r=mpp_create(&ctx_,&api_);
        if(r!=MPP_OK){fprintf(stderr,"mpp_create fail\n");return false;}
        r=mpp_init(ctx_,MPP_CTX_DEC,MPP_VIDEO_CodingMJPEG);
        if(r!=MPP_OK){fprintf(stderr,"mpp_init MJPEG fail ret=%d\n",r);return false;}
        // Send a dummy packet to complete decoder init (may trigger info_change)
        // Some MPP decoders need this to finalize the init sequence
        uint8_t dummy[]={0xFF,0xD8,0xFF,0xD9}; // minimal JPEG (SOI+EOI)
        MppPacket init_pkt;
        mpp_packet_init(&init_pkt,dummy,sizeof(dummy));
        mpp_packet_set_pts(init_pkt,0);
        r=api_->decode_put_packet(ctx_,init_pkt);
        if(r==MPP_OK){
            MppFrame frame=nullptr;
            r=api_->decode_get_frame(ctx_,&frame);
            if(frame){mpp_frame_deinit(&frame);}
        }
        mpp_packet_deinit(&init_pkt);
        // Flush/reset for clean state
        api_->reset(ctx_);
        return true;
    }
    // 返回 NV12 dma_buf fd, -1 失败
    // frame 需在 RGA 处理完后由调用者释放
    bool decode(const void* jpeg, size_t sz, int* fd, int* w, int* h,
                int* hs, int* vs, MppFrame* out_frame) {
        std::lock_guard<std::mutex> lk(mtx_);
        MppPacket pkt;mpp_packet_init(&pkt,(void*)jpeg,sz);
        mpp_packet_set_pts(pkt,0);
        MPP_RET r=api_->decode_put_packet(ctx_,pkt);
        if(r!=MPP_OK){fprintf(stderr,"[MPP] decode_put_packet err=%d\n",r);mpp_packet_deinit(&pkt);return false;}

        MppFrame frame=nullptr;
        r=api_->decode_get_frame(ctx_,&frame);
        mpp_packet_deinit(&pkt);  // safe to release packet after get_frame
        if(r!=MPP_OK||!frame){fprintf(stderr,"[MPP] decode_get_frame err=%d frame=%p\n",r,(void*)frame);return false;}

        if(mpp_frame_get_info_change(frame)){
            fprintf(stderr,"[MPP] info_change, re-sending packet\n");
            mpp_frame_deinit(&frame);
            MppPacket pkt2;mpp_packet_init(&pkt2,(void*)jpeg,sz);
            mpp_packet_set_pts(pkt2,0);
            r=api_->decode_put_packet(ctx_,pkt2);
            if(r!=MPP_OK){fprintf(stderr,"[MPP] decode_put_packet(2) err=%d\n",r);mpp_packet_deinit(&pkt2);return false;}
            r=api_->decode_get_frame(ctx_,&frame);
            mpp_packet_deinit(&pkt2);
            if(r!=MPP_OK||!frame){fprintf(stderr,"[MPP] decode_get_frame(2) err=%d frame=%p\n",r,(void*)frame);return false;}
        }

        MppBuffer buf=mpp_frame_get_buffer(frame);
        if(!buf){fprintf(stderr,"[MPP] null buffer\n");mpp_frame_deinit(&frame);return false;}
        *fd=mpp_buffer_get_fd(buf);
        *w=mpp_frame_get_width(frame);
        *h=mpp_frame_get_height(frame);
        *hs=mpp_frame_get_hor_stride(frame);
        *vs=mpp_frame_get_ver_stride(frame);
        *out_frame=frame;
        fprintf(stderr,"[MPP] decode OK: %dx%d stride=%d,%d fd=%d\n",*w,*h,*hs,*vs,*fd);
        return true;
    }
    void deinit() {
        if(api_&&ctx_){api_->reset(ctx_);mpp_destroy(ctx_);ctx_=nullptr;api_=nullptr;}
    }
    ~MppJpegDecoder(){deinit();}
};

// ── RGA 预处理 (NV12→RGB letterbox) ────────────────────────
class RgaPreprocessor {
    int mw_, mh_;
public:
    bool init(int mw, int mh) { mw_=mw; mh_=mh; return true; }

    // NV12 dma_buf → RGB dma_buf (letterbox 640×640)
    // 调用者负责 close 返回的 fd
    int process(int nv12_fd, int sw, int sh, int hs, int vs,
                int* out_lb_x, int* out_lb_y, float* out_scale) {
        // letterbox 参数
        float sc = std::min((float)mw_/sw, (float)mh_/sh);
        *out_scale = sc;
        int nw=(int)(sw*sc), nh=(int)(sh*sc);
        *out_lb_x=(mw_-nw)/2; *out_lb_y=(mh_-nh)/2;

        // 1. 导入 NV12 source fd
        rga_buffer_handle_t src_h = importbuffer_fd(nv12_fd, hs, vs, RK_FORMAT_YCbCr_420_SP);
        if (!src_h) return -1;

        // 2. 分配 RGB 输出 dma_buf
        int rgb_fd = dma_heap_alloc(mw_ * mh_ * 3);
        if (rgb_fd < 0) { releasebuffer_handle(src_h); return -1; }

        rga_buffer_handle_t dst_h = importbuffer_fd(rgb_fd, mw_, mh_, RK_FORMAT_RGB_888);
        if (!dst_h) { releasebuffer_handle(src_h); close(rgb_fd); return -1; }

        // 3. 分配临时 RGB buffer (resized, nw×nh)
        int tmp_fd = dma_heap_alloc(nw * nh * 3);
        rga_buffer_handle_t tmp_h = 0;
        if (tmp_fd >= 0) {
            tmp_h = importbuffer_fd(tmp_fd, nw, nh, RK_FORMAT_RGB_888);
        }
        if (!tmp_h) {
            // fallback: use virtual address
            void* tmp_va = malloc(nw * nh * 3);
            tmp_h = importbuffer_virtualaddr(tmp_va, nw, nh, RK_FORMAT_RGB_888);
            if (!tmp_h) {
                releasebuffer_handle(src_h); releasebuffer_handle(dst_h);
                close(rgb_fd); if (tmp_fd>=0) close(tmp_fd);
                return -1;
            }
        }

        rga_buffer_t src = wrapbuffer_handle(src_h, sw, sh, RK_FORMAT_YCbCr_420_SP, hs, vs);
        rga_buffer_t tmp = wrapbuffer_handle(tmp_h, nw, nh, RK_FORMAT_RGB_888, nw, nh);
        rga_buffer_t dst = wrapbuffer_handle(dst_h, mw_, mh_, RK_FORMAT_RGB_888, mw_, mh_);

        // Step 1: resize + NV12→RGB → tmp (nw×nh)
        im_rect trect={0,0,nw,nh};
        IM_STATUS st = imresize(src, tmp, 0, 0, 0, 0);
        if (st == IM_STATUS_SUCCESS) {
            // Step 2: letterbox fill black + place tmp at offset
            imfill(dst, {0,0,mw_,mh_}, 0x000000);
            im_rect drect={*out_lb_x, *out_lb_y, nw, nh};
            rga_buffer_t pat_buf = {};  // empty pattern
            im_rect prect = {0,0,0,0};
            st = improcess(tmp, dst, pat_buf, trect, drect, prect, -1);
        } else {
            imfill(dst, {0,0,mw_,mh_}, 0x000000);
        }

        releasebuffer_handle(src_h);
        releasebuffer_handle(tmp_h);
        releasebuffer_handle(dst_h);
        if (tmp_fd >= 0) close(tmp_fd);

        if (st != IM_STATUS_SUCCESS) {
            close(rgb_fd);
            return -1;
        }

        return rgb_fd;  // caller closes after NPU use
    }
};

// ── HW 采集线程 ────────────────────────────────────────────
static void capture_thread_hw(V4L2MjpegCapture* v4l2, SafeQueue<InFrame>* fq,
                               std::atomic<bool>* running, int cam_w, int cam_h) {
    int fid=0;
    while(*running){
        InFrame f;f.frame_id=fid++;f.cam_w=cam_w;f.cam_h=cam_h;f.use_hw=true;
        const void* data=nullptr;size_t sz=0;
        if(!v4l2->grab(&data,&sz)){usleep(5000);continue;}
        f.jpeg_data=malloc(sz);memcpy(f.jpeg_data,data,sz);f.jpeg_size=sz;
        cv::Mat jpg_buf(1, sz, CV_8UC1, (void*)data);
        f.bgr = cv::imdecode(jpg_buf, cv::IMREAD_COLOR);
        fq->push(std::move(f));
    }
    fq->set_done();
}

// ── HW Worker 线程 ─────────────────────────────────────────
struct HWWorkerCtx {
    RKNNPerCore*    c;
    MppJpegDecoder* mpp;
    RgaPreprocessor* rga;
    int mw,mh;
};

static void worker_thread_hw(HWWorkerCtx* wctx, SafeQueue<InFrame>* fq,
                              SafeQueue<InferResult>* rq, std::atomic<bool>* running) {
    RKNNPerCore* c=wctx->c;
    while(*running){
        InFrame f;
        if(!fq->pop(f,100))continue;
        InferResult res;res.frame_id=f.frame_id;res.valid=false;
        if(!f.use_hw){free(f.jpeg_data);continue;}

        double t0=now_sec();

        // 保存显示用帧 (capture 线程已通过 imdecode 解码)
        res.orig = f.bgr.clone();

        // MPP decode
        int nv12_fd=-1,nv12_w,nv12_h,nv12_hs,nv12_vs;
        MppFrame mpp_frame=nullptr;
        if(!wctx->mpp->decode(f.jpeg_data,f.jpeg_size,&nv12_fd,&nv12_w,&nv12_h,&nv12_hs,&nv12_vs,&mpp_frame)){
            fprintf(stderr,"[worker_hw] MPP decode failed for frame %d\n", f.frame_id);
            free(f.jpeg_data);continue;
        }
        free(f.jpeg_data);
        fprintf(stderr,"[worker_hw] MPP decoded frame %d: %dx%d NV12 fd=%d\n",
                f.frame_id, nv12_w, nv12_h, nv12_fd);

        // RGA NV12→RGB letterbox
        int lb_x,lb_y;float lb_scale;
        int rgb_fd=wctx->rga->process(nv12_fd,nv12_w,nv12_h,nv12_hs,nv12_vs,&lb_x,&lb_y,&lb_scale);
        mpp_frame_deinit(&mpp_frame);  // release MPP frame
        if(rgb_fd<0){
            fprintf(stderr,"[worker_hw] RGA process failed for frame %d\n", f.frame_id);
            continue;
        }
        fprintf(stderr,"[worker_hw] RGA done frame %d: rgb_fd=%d lb=(%d,%d) scale=%.3f\n",
                f.frame_id, rgb_fd, lb_x, lb_y, lb_scale);
        res.lb.x_pad=lb_x;res.lb.y_pad=lb_y;res.lb.scale=lb_scale;

        double t1=now_sec();res.t_pre=t1-t0;

        // 零拷贝 NPU 输入
        size_t in_size = wctx->mw * wctx->mh * c->model_c;
        rknn_tensor_mem* in_mem = rknn_create_mem_from_fd(c->ctx, rgb_fd, nullptr,
            (uint32_t)in_size, RKNN_FLAG_MEMORY_NON_CACHEABLE);
        close(rgb_fd);  // fd no longer needed after create_mem
        if (!in_mem) continue;
        in_mem->size = in_size;

        // 通过 set_io_mem 零拷贝设置输入 (替代 rknn_inputs_set)
        int ret = rknn_set_io_mem(c->ctx, in_mem, &c->input_attrs[0]);
        if (ret < 0) {
            fprintf(stderr,"[worker_hw] rknn_set_io_mem failed ret=%d\n", ret);
            rknn_destroy_mem(c->ctx, in_mem); continue;
        }

        ret = rknn_run(c->ctx, nullptr);
        if (ret < 0) {
            fprintf(stderr,"[worker_hw] rknn_run failed ret=%d\n", ret);
            rknn_destroy_mem(c->ctx, in_mem); continue;
        }
        rknn_destroy_mem(c->ctx, in_mem);
        double t2=now_sec();res.t_inf=t2-t1;

        std::vector<rknn_output> rk_out(c->io_num.n_output);
        memset(rk_out.data(),0,rk_out.size()*sizeof(rknn_output));
        for(uint32_t i=0;i<c->io_num.n_output;i++){rk_out[i].index=i;rk_out[i].want_float=!c->is_quant;}
        if(rknn_outputs_get(c->ctx,c->io_num.n_output,rk_out.data(),nullptr)<0)continue;
        yolov8_post_process_fast(c,rk_out.data(),&res.lb,&res.det,wctx->mw,wctx->mh);
        rknn_outputs_release(c->ctx,c->io_num.n_output,rk_out.data());
        double t3=now_sec();res.t_post=t3-t2;
        res.valid=true;

        rq->push(std::move(res));
    }
}

#endif // ENABLE_HW_DECODE

/* ============================================================
 *  主函数
 * ============================================================ */
static void usage(const char* prog) {
    printf("Usage: %s <model.rknn> [opts]\n",prog);
    printf("  -r WxH    Resolution: 640x480(default) 960x540 1280x720\n");
    printf("  -t N      NPU threads (default 3, max %d)\n",MAX_NPU_CORES);
#ifdef ENABLE_HW_DECODE
    printf("  --hw-decode   Use MPP hardware JPEG decode + RGA\n");
#endif
    printf("  --no-display  Headless mode (FPS only)\n");
    printf("  -h         Help\n");
}

int main(int argc, char** argv) {
    setvbuf(stdout,NULL,_IONBF,0);

    const char* model_path=nullptr;
    const char* cam_dev=DEFAULT_CAM_DEV;

    ResConfig res=RES_PRESETS[0];  // default 640×480
    int num_workers=MAX_NPU_CORES;
    bool no_display=false;
#ifdef ENABLE_HW_DECODE
    bool use_hw=false;
#endif

    // ── CLI 解析 ────────────────────────────────────────────
    static struct option long_opts[]={
#ifdef ENABLE_HW_DECODE
        {"hw-decode",no_argument,0,'H'},
#endif
        {"no-display",no_argument,0,'D'},
        {"help",no_argument,0,'h'},
        {0,0,0,0}
    };
    int opt;
    while((opt=getopt_long(argc,argv,"r:t:h",long_opts,nullptr))!=-1){
        switch(opt){
            case 'r':{
                int w,h;if(sscanf(optarg,"%dx%d",&w,&h)==2){
                    bool found=false;
                    for(int i=0;i<NUM_RES_PRESETS;i++){if(RES_PRESETS[i].w==w&&RES_PRESETS[i].h==h){res=RES_PRESETS[i];found=true;break;}}
                    if(!found)fprintf(stderr,"Unknown resolution: %s, using default\n",optarg);
                }break;}
            case 't':num_workers=atoi(optarg);
                if(num_workers<1)num_workers=1;
                if(num_workers>MAX_NPU_CORES)num_workers=MAX_NPU_CORES;
                break;
#ifdef ENABLE_HW_DECODE
            case 'H':use_hw=true;break;
#endif
            case 'D':no_display=true;break;
            case 'h':default:usage(argv[0]);return 0;
        }
    }

    // 剩余参数: model_path [cam_dev]
    if(optind<argc){
        model_path=argv[optind++];
    }
    if(optind<argc){
        cam_dev=argv[optind];
    }
    if(!model_path){usage(argv[0]);return 1;}

    printf("========================================\n");
    printf("  YOLOv8n Camera Inference FAST\n");
    printf("========================================\n");
    printf("Model:     %s\n",model_path);
    printf("Camera:    %s\n",cam_dev);
    printf("Resolution:%dx%d @ %dfps\n",res.w,res.h,res.fps);
    printf("NPU cores: %d\n",num_workers);
#ifdef ENABLE_HW_DECODE
    printf("HW decode: %s\n",use_hw?"YES":"no (CPU)");
#else
    printf("HW decode: no (CPU only)\n");
#endif
    printf("Display:   %s\n",no_display?"headless":"on");
    printf("========================================\n");

    // ── 初始化多 NPU 核心 ──────────────────────────────────
    std::vector<RKNNPerCore> cores(num_workers);
    for(int i=0;i<num_workers;i++){
        printf("Init NPU core %d...\n",i);
        if(init_rknn_core(model_path,&cores[i],i)<0){
            fprintf(stderr,"Failed to init NPU core %d\n",i);
            return -1;
        }
    }

    // ── 打开摄像头 ──────────────────────────────────────────
    cv::VideoCapture cap_cpu;
#ifdef ENABLE_HW_DECODE
    if(!use_hw)
#endif
    {
        cap_cpu.open(cam_dev,cv::CAP_V4L2);
        if(!cap_cpu.isOpened()){fprintf(stderr,"Cannot open camera\n");return -1;}
        cap_cpu.set(cv::CAP_PROP_FOURCC,cv::VideoWriter::fourcc('M','J','P','G'));
        cap_cpu.set(cv::CAP_PROP_FRAME_WIDTH,res.w);
        cap_cpu.set(cv::CAP_PROP_FRAME_HEIGHT,res.h);
        cap_cpu.set(cv::CAP_PROP_FPS,res.fps);
        cap_cpu.set(cv::CAP_PROP_BUFFERSIZE,1);
        printf("Camera: %dx%d @ MJPG\n",(int)cap_cpu.get(cv::CAP_PROP_FRAME_WIDTH),
               (int)cap_cpu.get(cv::CAP_PROP_FRAME_HEIGHT));
    }

#ifdef ENABLE_HW_DECODE
    V4L2MjpegCapture v4l2_cap;
    MppJpegDecoder    mpp_dec;
    RgaPreprocessor   rga_pre;
    if(use_hw){
        if(!v4l2_cap.open(cam_dev,res.w,res.h)){
            fprintf(stderr,"V4L2 HW capture init fail, fallback to CPU\n");
            use_hw=false;
        }
        else if(!mpp_dec.init()){
            fprintf(stderr,"MPP init fail, fallback to CPU\n");
            use_hw=false;
        }
        else if(!rga_pre.init(MODEL_WIDTH,MODEL_HEIGHT)){
            fprintf(stderr,"RGA init fail, fallback to CPU\n");
            use_hw=false;
        }else{
            printf("HW decode pipeline: V4L2→MPP→RGA→RKNN OK\n");
        }
    }
#endif

    // ── 创建队列 ────────────────────────────────────────────
    SafeQueue<InFrame>     fq(2);
    SafeQueue<InferResult> rq(num_workers+1);
    std::atomic<bool>      running{true};

    // ── 启动线程 ────────────────────────────────────────────
    std::thread cap_t;
    std::vector<std::thread> workers;

    // Worker context 必须和 worker 线程生命周期一致
    std::vector<WorkerCtx> cpu_ctxs;
#ifdef ENABLE_HW_DECODE
    std::vector<HWWorkerCtx> hw_ctxs;
#endif

#ifdef ENABLE_HW_DECODE
    if(use_hw){
        cap_t=std::thread(capture_thread_hw,&v4l2_cap,&fq,&running,res.w,res.h);
        hw_ctxs.resize(num_workers);
        for(int i=0;i<num_workers;i++){
            hw_ctxs[i].c=&cores[i];hw_ctxs[i].mpp=&mpp_dec;
            hw_ctxs[i].rga=&rga_pre;hw_ctxs[i].mw=MODEL_WIDTH;hw_ctxs[i].mh=MODEL_HEIGHT;
            workers.emplace_back(worker_thread_hw,&hw_ctxs[i],&fq,&rq,&running);
        }
    }else
#endif
    {
        cap_t=std::thread(capture_thread_cpu,&cap_cpu,&fq,&running,res.w,res.h);
        cpu_ctxs.resize(num_workers);
        for(int i=0;i<num_workers;i++){
            cpu_ctxs[i].c=&cores[i];cpu_ctxs[i].mw=MODEL_WIDTH;cpu_ctxs[i].mh=MODEL_HEIGHT;
            workers.emplace_back(worker_thread_cpu,&cpu_ctxs[i],&fq,&rq,&running);
        }
    }

    std::thread disp_t(display_thread_func,&rq,&running,no_display);

    printf("\nPress ESC/q to exit.\n\n");

    // ── 等待退出 ────────────────────────────────────────────
    disp_t.join();
    running=false;fq.set_done();rq.set_done();
    cap_t.join();
    for(auto& w:workers)w.join();

    // ── 清理 ────────────────────────────────────────────────
#ifdef ENABLE_HW_DECODE
    if(use_hw){v4l2_cap.close();mpp_dec.deinit();}
#endif
    cap_cpu.release();
    printf("Exit.\n");
    return 0;
}
