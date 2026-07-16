"""FLIR Grasshopper3 camera control + AVI recording via PySpin (Spinnaker 4.3).

Wraps the Spinnaker Python bindings to: enumerate/connect the camera, set frame
rate and pixel format, and record an M-JPEG AVI while emitting a per-frame
callback carrying the frame index, the camera's embedded timestamp, and a host
timestamp. Dropped/incomplete frames are counted and reported because a mismatch
between frames written to the AVI and rows in the timestamps CSV would silently
break alignment with SLEAP output downstream.

Video always flows over USB3; the camera GPIO strobe (hardware-sync mode) is a
separate concern handled in led_daq.py.

Importing this module requires ``PySpin`` (the Spinnaker 4.3 Python wheel built
for the active interpreter). It is only importable in the dedicated Python 3.10
venv, not the rig's system 3.13.
"""

from __future__ import annotations

import glob
import os
import threading
import time
from typing import Callable, Optional

import PySpin  # type: ignore[import-not-found]

#: Per-frame callback: ``(frame_index, camera_timestamp_ns, host_time_s, incomplete)``.
FrameCallback = Callable[[int, int, float, bool], None]
#: End-of-recording callback: ``(completed, frames_written, frames_incomplete)``.
FinishedCallback = Callable[[bool, int, int], None]


class CameraError(RuntimeError):
    """Raised on camera enumeration/configuration/recording failure."""


# --- low-level GenICam node helpers -----------------------------------------
# GS3-series nodes differ across firmware (e.g. AcquisitionFrameRateEnable vs
# AcquisitionFrameRateEnabled, and an AcquisitionFrameRateAuto that must be Off).
# These helpers set nodes defensively and stay quiet when a node is absent.


def _set_enum(nodemap: "PySpin.INodeMap", name: str, entry: str) -> bool:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return False
    entry_node = node.GetEntryByName(entry)
    if not PySpin.IsAvailable(entry_node) or not PySpin.IsReadable(entry_node):
        return False
    node.SetIntValue(entry_node.GetValue())
    return True


def _set_bool(nodemap: "PySpin.INodeMap", name: str, value: bool) -> bool:
    node = PySpin.CBooleanPtr(nodemap.GetNode(name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return False
    node.SetValue(value)
    return True


def _set_float(nodemap: "PySpin.INodeMap", name: str, value: float) -> bool:
    node = PySpin.CFloatPtr(nodemap.GetNode(name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return False
    # Respect device limits to avoid an out-of-range exception.
    value = max(node.GetMin(), min(node.GetMax(), value))
    node.SetValue(value)
    return True


class Camera:
    """A single FLIR camera, configured for opto-behavior recording.

    Typical use::

        with Camera() as cam:
            cam.configure(frame_rate=30.0)
            cam.record(n_frames=500, output_basepath=r"C:\\...\\session",
                       on_frame=logger.log_frame)
    """

    def __init__(self, index: int = 0, record_pixel_format: str = "Mono8") -> None:
        self.index = index
        # Recording format. Mono8 keeps files small and is ideal for SLEAP
        # tracking of a single fly; switch to a color format here if needed.
        self.record_pixel_format = record_pixel_format

        self._system: Optional["PySpin.System"] = None
        self._cam_list: Optional["PySpin.CameraList"] = None
        self._cam: Optional["PySpin.CameraPtr"] = None
        self._frame_rate: float = 0.0

        self._stop_event = threading.Event()
        self._record_thread: Optional[threading.Thread] = None
        # Actual video file(s) produced by the last recording (see the note in
        # _record_worker about SpinVideo's -NNNN chunk suffix).
        self._video_files: list[str] = []

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> str:
        """Connect to the camera and return its model name."""
        self._system = PySpin.System.GetInstance()
        self._cam_list = self._system.GetCameras()
        if self._cam_list.GetSize() <= self.index:
            self._release_system()
            raise CameraError(
                f"no camera at index {self.index} "
                f"(found {self._cam_list.GetSize() if self._cam_list else 0})"
            )
        self._cam = self._cam_list.GetByIndex(self.index)
        self._cam.Init()
        return self.model_name()

    def close(self) -> None:
        """Stop recording, deinit the camera, and release Spinnaker handles."""
        self.stop()
        if self._cam is not None:
            try:
                if self._cam.IsStreaming():
                    self._cam.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
            try:
                self._cam.DeInit()
            except PySpin.SpinnakerException:
                pass
            self._cam = None
        self._release_system()

    def _release_system(self) -> None:
        if self._cam_list is not None:
            self._cam_list.Clear()
            self._cam_list = None
        if self._system is not None:
            self._system.ReleaseInstance()
            self._system = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- info ----------------------------------------------------------------

    def model_name(self) -> str:
        if self._cam is None:
            raise CameraError("camera not open()")
        nodemap_tl = self._cam.GetTLDeviceNodeMap()
        node = PySpin.CStringPtr(nodemap_tl.GetNode("DeviceModelName"))
        if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
            return node.GetValue()
        return "unknown"

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    @staticmethod
    def list_cameras() -> list[str]:
        """Return model names of connected cameras (opens/closes the system)."""
        system = PySpin.System.GetInstance()
        try:
            cams = system.GetCameras()
            names = []
            for i in range(cams.GetSize()):
                cam = cams.GetByIndex(i)
                tl = cam.GetTLDeviceNodeMap()
                node = PySpin.CStringPtr(tl.GetNode("DeviceModelName"))
                names.append(
                    node.GetValue()
                    if PySpin.IsAvailable(node) and PySpin.IsReadable(node)
                    else f"camera{i}"
                )
                del cam
            cams.Clear()
            return names
        finally:
            system.ReleaseInstance()

    # -- configuration -------------------------------------------------------

    def configure(
        self, frame_rate: float, pixel_format: Optional[str] = None
    ) -> None:
        """Set continuous acquisition, pixel format, and a fixed frame rate."""
        if self._cam is None:
            raise CameraError("camera not open()")
        nodemap = self._cam.GetNodeMap()

        _set_enum(nodemap, "AcquisitionMode", "Continuous")

        pf = pixel_format or self.record_pixel_format
        _set_enum(nodemap, "PixelFormat", pf)

        # Disable any automatic frame-rate control, then pin the requested rate.
        _set_enum(nodemap, "AcquisitionFrameRateAuto", "Off")
        # Different firmware spell the enable node differently; try both.
        if not _set_bool(nodemap, "AcquisitionFrameRateEnable", True):
            _set_bool(nodemap, "AcquisitionFrameRateEnabled", True)
        _set_float(nodemap, "AcquisitionFrameRate", frame_rate)

        self._frame_rate = self._read_frame_rate(nodemap) or frame_rate

    def _read_frame_rate(self, nodemap: "PySpin.INodeMap") -> float:
        node = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
        if PySpin.IsAvailable(node) and PySpin.IsReadable(node):
            return node.GetValue()
        return 0.0

    # -- recording -----------------------------------------------------------

    def record(
        self,
        output_basepath: str,
        n_frames: int = 0,
        on_frame: Optional[FrameCallback] = None,
        on_finished: Optional[FinishedCallback] = None,
    ) -> None:
        """Start recording on a background thread.

        Args:
            output_basepath: Path WITHOUT extension; SpinVideo appends ``.avi``.
            n_frames: Stop after this many good frames; 0 means run until
                :meth:`stop` is called.
            on_frame: Called for each grabbed frame (including incomplete ones,
                flagged via the callback's ``incomplete`` argument).
            on_finished: Called once when recording ends.
        """
        if self._cam is None:
            raise CameraError("camera not open()")
        if self._record_thread and self._record_thread.is_alive():
            raise CameraError("a recording is already in progress")
        self._stop_event.clear()
        self._record_thread = threading.Thread(
            target=self._record_worker,
            args=(output_basepath, n_frames, on_frame, on_finished),
            name="camera-record",
            daemon=True,
        )
        self._record_thread.start()

    def _record_worker(
        self,
        output_basepath: str,
        n_frames: int,
        on_frame: Optional[FrameCallback],
        on_finished: Optional[FinishedCallback],
    ) -> None:
        assert self._cam is not None
        written = 0
        incomplete = 0
        completed = False

        option = PySpin.MJPGOption()
        option.frameRate = self._frame_rate or 15.0
        option.quality = 80  # matches the existing FlyCap2 workflow

        video = PySpin.SpinVideo()
        processor = PySpin.ImageProcessor()
        processor.SetColorProcessing(
            PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR
        )
        target_pf = getattr(PySpin, f"PixelFormat_{self.record_pixel_format}")

        try:
            video.Open(output_basepath, option)
            self._cam.BeginAcquisition()
            frame_index = 0
            while not self._stop_event.is_set():
                if n_frames and written >= n_frames:
                    completed = True
                    break
                try:
                    image = self._cam.GetNextImage(1000)  # 1 s timeout
                except PySpin.SpinnakerException:
                    # Timeout with no frame: loop back and re-check stop flag.
                    continue

                host_ts = time.perf_counter()
                is_incomplete = image.IsIncomplete()
                camera_ts = int(image.GetTimeStamp())

                if is_incomplete:
                    incomplete += 1
                else:
                    converted = processor.Convert(image, target_pf)
                    video.Append(converted)
                    written += 1

                if on_frame is not None:
                    on_frame(frame_index, camera_ts, host_ts, is_incomplete)
                frame_index += 1
                image.Release()
        except PySpin.SpinnakerException as exc:
            raise CameraError(f"recording failed: {exc}") from exc
        finally:
            try:
                if self._cam is not None and self._cam.IsStreaming():
                    self._cam.EndAcquisition()
            except PySpin.SpinnakerException:
                pass
            try:
                video.Close()
            except PySpin.SpinnakerException:
                pass
            self._video_files = self._resolve_video_files(output_basepath)
            if on_finished is not None:
                on_finished(completed, written, incomplete)

    def _resolve_video_files(self, output_basepath: str) -> list[str]:
        """Return the AVI file(s) SpinVideo actually wrote.

        SpinVideo appends a ``-NNNN`` chunk index and ``.avi`` to the base name
        (splitting into multiple files past its max file size). For the common
        single-chunk case we rename ``<base>-0000.avi`` to a clean
        ``<base>.avi`` so the video filename is predictable for SLEAP; multiple
        chunks are left as-is and all reported.
        """
        chunks = sorted(glob.glob(f"{output_basepath}-*.avi"))
        if len(chunks) == 1:
            final = f"{output_basepath}.avi"
            try:
                os.replace(chunks[0], final)
                return [final]
            except OSError:
                return chunks
        if not chunks and os.path.exists(f"{output_basepath}.avi"):
            return [f"{output_basepath}.avi"]
        return chunks

    @property
    def video_files(self) -> list[str]:
        return list(self._video_files)

    def stop(self) -> None:
        """Signal the recording thread to stop and wait for it to finish."""
        self._stop_event.set()
        thread = self._record_thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=5.0)

    def is_recording(self) -> bool:
        return bool(self._record_thread and self._record_thread.is_alive())
