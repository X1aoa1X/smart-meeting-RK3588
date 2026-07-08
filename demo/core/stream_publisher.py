"""GStreamer RTSP 推流后台线程。

用法:
  stream = StreamThread(device_ip="192.168.1.100",
                        width=960, height=540, fps=25, bitrate=2_000_000,
                        frame_queue=shared_queue)
  stream.stream_started.connect(on_started)   # (rtsp_url)
  stream.stream_error.connect(on_error)       # (msg)
  stream.stream_stopped.connect(on_stopped)   # ()
  stream.status_msg.connect(on_status)        # (msg)
  stream.start()

  # 帧喂送: 通过共享队列（CameraThread 直连）或 put_frame()（Qt 信号回退）
  stream.put_frame(frame_bgr)

  # 名片叠加（可选）:
  stream.set_overlay_callback(lambda: ({"name":"王强","role":"队长","duration":35.0}, "TRACKING"))

依赖: PyQt5, cv2, numpy, GStreamer (gi: Gst, GstRtspServer, GstApp, GLib)
"""

import time
import queue
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Callable


class StreamThread(QThread):
    """后台 RTSP 推流线程 — 使用 GStreamer RTSP Server。

    帧喂送路径 (低延迟直连模式):
      CameraThread [工作线程] → queue.Queue(共享) → need-data 回调 [GLib线程]
        → appsrc → videoconvert → NV12 → mpph264enc(baseline) → h264parse → rtph264pay
        → GstRtspServer :8554/smartcam

    回退路径 (无共享队列时):
      CameraThread → raw_frame_ready(Qt信号) → _on_raw_frame [GUI线程]
        → put_frame() → queue.Queue → need-data 回调 [GLib线程] → ...

    PTS 由 GStreamer pipeline clock 自动管理 (do-timestamp=true)，
    不再使用 frame_index 累加，消除 PTS→PCR 漂移和时间戳溢出问题。

    GstRtspServer 处理全部 RTSP 协议 (OPTIONS/DESCRIBE/SETUP/PLAY/TEARDOWN)，
    客户端使用标准 VLC/ffplay 即可播放。
    """

    # ── Qt 信号 ──────────────────────────────────────────────────────────
    stream_started = pyqtSignal(str)   # 推流就绪，携带 RTSP URL
    stream_error   = pyqtSignal(str)   # 推流错误
    stream_stopped = pyqtSignal()      # 推流已停止
    status_msg     = pyqtSignal(str)   # 状态消息

    # ── 默认配置 ─────────────────────────────────────────────────────────
    DEFAULT_WIDTH   = 960
    DEFAULT_HEIGHT  = 540
    DEFAULT_FPS     = 25
    DEFAULT_BITRATE = 2_000_000   # 2 Mbps
    RTSP_PORT       = 8554

    def __init__(self, device_ip: str = "127.0.0.1",
                 width: int = DEFAULT_WIDTH,
                 height: int = DEFAULT_HEIGHT,
                 fps: int = DEFAULT_FPS,
                 bitrate: int = DEFAULT_BITRATE,
                 frame_queue: queue.Queue | None = None):
        super().__init__()
        self._device_ip = device_ip
        self._width = width
        self._height = height
        self._fps = fps
        self._bitrate = bitrate

        self._running = False
        self._loop = None           # GLib.MainLoop
        self._server = None         # GstRtspServer.RTSPServer
        self._server_id = 0         # GLib source id (server.attach 返回值)，cleanup 时用于释放端口
        self._appsrc = None         # GstApp.AppSrc
        # 接受外部共享队列或自建 — 共享队列使 CameraThread 可绕过 Qt 信号直接喂帧
        self._frame_queue: queue.Queue = (
            frame_queue if frame_queue is not None else queue.Queue(maxsize=2))
        self._owns_queue = frame_queue is None  # 仅自建队列在 cleanup 时清空

        # 单调帧计数器 — 仅用于 buf.offset 调试；PTS 由 GStreamer pipeline clock 管理
        self._frame_count = 0
        self._clock_base_ns = 0
        self._frame_duration_ns = 0
        self._last_stream_frame: np.ndarray | None = None
        self._launch_description = ""
        self._push_source_id = 0       # GLib timeout source id, used for active FPS-paced pushing
        self._last_real_frame_time = 0.0
        self._last_push_status_time = 0.0

        # 名片叠加回调（可选）
        # callback() → (speaker_info: dict|None, system_state: str)
        self._overlay_callback: Callable[[], tuple] | None = None

    # ── 公共接口 ────────────────────────────────────────────────────────────

    def set_overlay_callback(self, callback: Callable[[], tuple] | None):
        """设置名片叠加回调（可选）。

        Args:
            callback: 无参数可调用对象，返回 (speaker_info, system_state)。
                      speaker_info 为 None 时不绘制名片条。
                      设为 None 则禁用叠加（输出干净画面）。
        """
        self._overlay_callback = callback

    def put_frame(self, frame_bgr: np.ndarray):
        """线程安全: 向推流队列投递一帧。

        由主/GUI 线程调用。非阻塞 — 队列满时静默丢弃旧帧。
        """
        if not self._running:
            return
        try:
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            self._frame_queue.put_nowait(frame_bgr)
        except queue.Full:
            pass

    # ── QThread 主循环 ──────────────────────────────────────────────────────

    def run(self):
        """启动 GStreamer RTSP Server 并运行 GLib 主循环。"""
        import gi
        gi.require_version('Gst', '1.0')
        gi.require_version('GstRtspServer', '1.0')
        gi.require_version('GstApp', '1.0')
        from gi.repository import Gst, GstRtspServer, GstApp, GLib

        # 保存引用供回调使用
        self._Gst = Gst
        self._GstApp = GstApp
        self._GLib = GLib
        self._GstRtspServer = GstRtspServer

        Gst.init(None)
        self._running = True

        # ── 1. 创建 RTSP 服务器 ─────────────────────────────────────────
        server = GstRtspServer.RTSPServer()
        server.set_service(str(self.RTSP_PORT))
        server.set_address("0.0.0.0")

        # ── 2. 创建媒体工厂 ────────────────────────────────────────────
        factory = GstRtspServer.RTSPMediaFactory()
        factory.set_shared(True)
        # VLC/ffplay 对 0ms RTSP jitterbuffer 较敏感，给少量缓冲更稳。
        factory.set_latency(100)

        try:
            launch = self._build_launch(Gst)
            # 关键：在真正挂载 RTSP 前先验证 launch 字符串。
            # 原实现把错误延迟到客户端 DESCRIBE/SETUP 阶段，VLC 只会显示"无法打开/无法解析"。
            test_bin = Gst.parse_launch(launch)
            test_bin.set_state(Gst.State.NULL)
            self._launch_description = launch
        except Exception as e:
            self.stream_error.emit(f"GStreamer 管道解析失败: {e}")
            self.status_msg.emit(f"失败的管道: {launch if 'launch' in locals() else '<未生成>'}")
            self._running = False
            return

        factory.set_launch(launch)
        factory.connect("media-configure", self._on_media_configure)
        self.status_msg.emit(
            f"GStreamer 管道: {self._width}x{self._height} @ {self._fps}fps, "
            f"{self._bitrate // 1000}kbps")
        self.status_msg.emit(f"Launch: {launch}")

        # ── 3. 挂载路径 ────────────────────────────────────────────────
        mount_points = server.get_mount_points()
        mount_points.add_factory("/smartcam", factory)

        # ── 4. 附加服务器到 GLib 主上下文 ─────────────────────────────
        server_id = server.attach(None)
        if server_id == 0:
            self.stream_error.emit("GStreamer RTSP 服务器附加失败 (端口 8554 可能被占用)")
            self._running = False
            self._cleanup()
            return
        self._server = server
        self._server_id = server_id   # 保存 source id，用于 _cleanup() 正确释放端口

        rtsp_url = f"rtsp://{self._device_ip}:{self.RTSP_PORT}/smartcam"
        self.status_msg.emit(f"RTSP 服务已启动: {rtsp_url}")
        self.stream_started.emit(rtsp_url)

        # ── 5. 运行 GLib 主循环 ────────────────────────────────────────
        self._loop = GLib.MainLoop()
        try:
            self._loop.run()
        except Exception as e:
            self.stream_error.emit(f"GLib 主循环异常: {e}")

        # ── 6. 清理 ────────────────────────────────────────────────────
        self._cleanup()
        self.stream_stopped.emit()

    # ── 管道构建 ────────────────────────────────────────────────────────────

    def _build_launch(self, Gst=None) -> str:
        """构建 GstRtspServer 媒体管道字符串 (gst-launch 语法)。

        修复点：
          1. 不再盲目硬编码单一 mpph264enc 属性组合；先根据本机插件选择编码器。
          2. 明确输出 H.264 byte-stream/AU alignment，便于 h264parse/rtph264pay 生成 VLC 可识别的 SDP。
          3. rtph264pay 周期性携带 SPS/PPS，避免 VLC 首帧/关键帧前无法解码。
          4. appsrc 使用 live/time caps，并关闭 block，防止客户端连接时队列空导致 RTSP prepare 卡死。
        """
        encoder = self._build_h264_encoder_launch(Gst)
        return (
            f"( appsrc name=src format=time is-live=true block=false "
            f"do-timestamp=false emit-signals=false max-bytes=0 "
            f"! video/x-raw,format=BGR,width={self._width},height={self._height},"
            f"framerate={self._fps}/1 "
            f"! queue max-size-buffers=2 leaky=downstream "
            f"! videoconvert "
            f"! video/x-raw,format=NV12,width={self._width},height={self._height},"
            f"framerate={self._fps}/1 "
            f"! queue max-size-buffers=2 leaky=downstream "
            f"! {encoder} "
            f"! h264parse config-interval=-1 "
            f"! video/x-h264,stream-format=byte-stream,alignment=au "
            f"! rtph264pay name=pay0 pt=96 config-interval=1 aggregate-mode=none )"
        )

    def _element_has_props(self, Gst, factory_name: str, props: list[str]) -> bool:
        """检查 GStreamer 元素是否存在并具备指定属性。"""
        if Gst is None or Gst.ElementFactory.find(factory_name) is None:
            return False
        elem = None
        try:
            elem = Gst.ElementFactory.make(factory_name, None)
            if elem is None:
                return False
            available = {p.name for p in elem.list_properties()}
            return all(p in available for p in props)
        except Exception:
            return False
        finally:
            if elem is not None:
                elem.set_state(Gst.State.NULL)

    def _build_h264_encoder_launch(self, Gst=None) -> str:
        """选择 H.264 编码器。优先 Rockchip MPP，缺失/属性不兼容时回退 x264enc。"""
        gop = max(1, min(self._fps, 60))
        bitrate_kbps = max(256, int(self._bitrate // 1000))

        # Rockchip gstreamer-rockchip 的 mpph264enc 常见属性是 bps/gop/rc-mode/header-mode。
        # 不强制写 profile caps 到编码器后面，避免部分版本 caps negotiation 失败。
        if self._element_has_props(Gst, "mpph264enc", ["bps", "gop"]):
            parts = [f"mpph264enc bps={self._bitrate} gop={gop}"]
            if self._element_has_props(Gst, "mpph264enc", ["header-mode"]):
                # 1 通常表示把 SPS/PPS header 插入码流，配合 rtph264pay config-interval 提高 VLC 兼容性。
                parts.append("header-mode=1")
            return " ".join(parts)

        # 某些发行版/板卡插件使用另一套属性名。
        if self._element_has_props(Gst, "mpph264enc", ["bitrate"]):
            parts = [f"mpph264enc bitrate={bitrate_kbps}"]
            if self._element_has_props(Gst, "mpph264enc", ["gop-size"]):
                parts.append(f"gop-size={gop}")
            return " ".join(parts)

        # 软件回退：牺牲 CPU，占用更高，但可验证 RTSP/VLC 链路是否正常。
        if Gst is None or Gst.ElementFactory.find("x264enc") is not None:
            return (
                f"x264enc tune=zerolatency speed-preset=ultrafast "
                f"bitrate={bitrate_kbps} key-int-max={gop} byte-stream=true "
                f"bframes=0 aud=true"
            )

        # 最后兜底：让 parse_launch 抛出清晰错误。
        return "x264enc tune=zerolatency byte-stream=true"

    # ── 媒体配置回调 ────────────────────────────────────────────────────────

    def _on_media_configure(self, factory, media):
        """客户端连接时 GstRtspServer 创建媒体管道后的回调。

        这版不再依赖 appsrc 的 need-data 信号持续触发。部分 GStreamer/RTSP
        组合下，VLC 能拿到首帧但 need-data 不再连续回调，表现就是停在
        "network caching" 后的第一帧。这里改为 GLib timeout 按 FPS 主动 push。
        """
        element = media.get_element()
        if element is None:
            return
        appsrc = element.get_by_name("src")
        if appsrc is None:
            self.status_msg.emit("⚠ 未找到 appsrc 元素")
            return

        self._appsrc = appsrc
        appsrc.set_property("format", self._Gst.Format.TIME)
        appsrc.set_property("is-live", True)
        appsrc.set_property("do-timestamp", False)
        appsrc.set_property("block", False)
        try:
            appsrc.set_property("emit-signals", False)
        except Exception:
            pass
        try:
            appsrc.set_property("max-bytes", 0)
        except Exception:
            pass
        try:
            caps = self._Gst.Caps.from_string(
                f"video/x-raw,format=BGR,width={self._width},height={self._height},"
                f"framerate={self._fps}/1")
            appsrc.set_property("caps", caps)
        except Exception:
            pass

        # 每次媒体重新 prepare 时重置时间戳。固定步进 PTS 对 RTSP/RTP 最稳。
        self._clock_base_ns = time.monotonic_ns()
        self._frame_duration_ns = int(self._Gst.SECOND // max(1, self._fps))
        self._frame_count = 0

        try:
            media.connect("unprepared", self._on_media_unprepared)
        except Exception:
            pass

        self._start_push_timer()

    def _on_media_unprepared(self, media):
        """客户端断开/媒体释放时停止主动推帧定时器。"""
        self._stop_push_timer()
        self._appsrc = None

    def _start_push_timer(self):
        """启动按 FPS 主动推帧的 GLib 定时器。"""
        self._stop_push_timer()
        interval_ms = max(1, int(round(1000.0 / max(1, self._fps))))
        self._push_source_id = self._GLib.timeout_add(interval_ms, self._on_push_timer)
        self.status_msg.emit(f"RTSP appsrc 主动推帧已启动: every {interval_ms} ms")

    def _stop_push_timer(self):
        """停止 GLib 推帧定时器。"""
        if self._push_source_id:
            try:
                self._GLib.source_remove(self._push_source_id)
            except Exception:
                pass
            self._push_source_id = 0

    def _on_push_timer(self):
        """GLib timeout 回调：每个周期推送一帧。返回 False 会移除定时器。"""
        if not self._running or self._appsrc is None:
            self._push_source_id = 0
            return False
        ok = self._push_one_frame(self._appsrc)
        if not ok:
            self._push_source_id = 0
        return ok

    def _read_latest_stream_frame(self) -> np.ndarray | None:
        """非阻塞读取最新帧；队列为空时复用上一帧，确保 VLC 端时间戳持续推进。"""
        newest = None
        while True:
            try:
                item = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                return None
            newest = item

        if newest is not None:
            self._last_stream_frame = newest
            self._last_real_frame_time = time.time()
            return newest

        if self._last_stream_frame is not None:
            return self._last_stream_frame

        frame = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._last_stream_frame = frame
        return frame

    def _push_one_frame(self, appsrc) -> bool:
        """向 appsrc 主动推送一帧。"""
        frame = self._read_latest_stream_frame()
        if frame is None:
            return False

        try:
            if frame.shape[1] != self._width or frame.shape[0] != self._height:
                frame = cv2.resize(frame, (self._width, self._height),
                                   interpolation=cv2.INTER_LINEAR)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8, copy=False)
            frame = np.ascontiguousarray(frame)

            # ── 名片叠加（可选）──────────────────────────────────────────
            if self._overlay_callback is not None:
                try:
                    speaker_info, system_state = self._overlay_callback()
                    if speaker_info or system_state:
                        from core.overlay_renderer import render_overlay
                        frame = render_overlay(
                            frame, speaker_info=speaker_info,
                            system_state=system_state)
                except Exception:
                    pass  # 叠加失败不中断推流

            data = frame.tobytes()

            buf = self._Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            if self._frame_duration_ns <= 0:
                self._frame_duration_ns = int(self._Gst.SECOND // max(1, self._fps))

            pts = self._frame_count * self._frame_duration_ns
            buf.pts = pts
            buf.dts = pts
            buf.duration = self._frame_duration_ns
            buf.offset = self._frame_count
            self._frame_count += 1

            ret = appsrc.emit("push-buffer", buf)
            if ret == self._Gst.FlowReturn.OK:
                # 降噪：最多每 2 秒打一条状态，便于确认推帧仍在继续。
                now = time.time()
                if now - self._last_push_status_time > 2.0:
                    self._last_push_status_time = now
                    stale = now - self._last_real_frame_time if self._last_real_frame_time else 0.0
                    self.status_msg.emit(
                        f"RTSP 推帧中: #{self._frame_count}, stale={stale:.1f}s")
                return True
            if ret == self._Gst.FlowReturn.FLUSHING:
                return False
            self.status_msg.emit(f"推流帧失败: {ret}")
            return True
        except Exception as e:
            self.status_msg.emit(f"推流帧错误: {e}")
            return True

    def _on_need_data(self, appsrc, size):
        """兼容保留：当前版本不再连接 need-data，避免只触发首帧后 VLC 卡住。"""
        self._push_one_frame(appsrc)

    # ── 清理 ────────────────────────────────────────────────────────────────

    def _cleanup(self):
        """停止 GStreamer RTSP 服务器并清理资源。

        关键：必须先调用 GLib.Source.remove(server_id) 从主上下文移除
        服务器的事件源，否则底层 socket 不会释放，导致下次 server.attach()
        端口绑定失败（"GStreamer RTSP 服务器附加失败"）。
        """
        self._stop_push_timer()

        # 断开 appsrc 信号
        if self._appsrc is not None:
            try:
                self._appsrc.disconnect_by_func(self._on_need_data)
            except Exception:
                pass
            self._appsrc = None

        # ── 正确释放 RTSP 服务器（必须移除 GLib source，否则端口不释放）──
        if hasattr(self, '_server_id') and self._server_id:
            try:
                self._GLib.Source.remove(self._server_id)
            except Exception:
                pass
            self._server_id = 0
        self._server = None

        # 仅清空自建队列（共享队列由外部管理）
        if self._owns_queue:
            while True:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break

        self._loop = None
        self.status_msg.emit("推流已停止")

    def stop(self):
        """优雅停止推流线程。"""
        self._running = False
        self._stop_push_timer()
        try:
            self._frame_queue.put_nowait(None)
        except Exception:
            pass
        if self._loop is not None:
            self._loop.quit()
        self.wait(5000)
