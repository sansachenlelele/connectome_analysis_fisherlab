"""PySide6 GUI for the opto-camera control application.

Three panels (per DESIGN_NOTES): Camera, Stimulus timeline, and Sync settings.
Camera acquisition and the LED stimulus timeline run on background worker
threads inside ``RecordingController``; the GUI never touches hardware objects
from those threads. Instead a QTimer polls thread-safe counters a few times a
second to refresh the status labels, which keeps the window responsive during a
recording without cross-thread widget access.

The hardware modules (camera.py -> PySpin, led_daq.py -> nidaqmx) are imported
lazily so the window still opens for layout inspection on an interpreter without
those SDKs, showing a clear error instead of failing at import.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QSlider,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from stimulus import Interval, StimulusError, StimulusTimeline, ma_to_voltage

MAX_LED_MA = 1000

DEFAULT_OUTPUT_DIR = str(Path.home() / "Documents" / "OptoRecordings")


class RecordingController:
    """Coordinates camera + LED + logger for one recording session.

    Lives on the GUI thread but launches hardware work on worker threads (owned
    by Camera and LedController). All state the GUI polls is plain attributes
    updated under a lock or by the worker callbacks.
    """

    def __init__(self) -> None:
        self._camera = None
        self._led = None
        self._logger = None
        self._lock = threading.Lock()
        self._finalized = False
        self.running = False
        self.error: Optional[str] = None
        self.result_summary: Optional[str] = None

    def start(
        self,
        *,
        camera,
        output_dir: str,
        session_name: str,
        frame_rate: float,
        n_frames: int,
        timeline: StimulusTimeline,
        device: str,
        ao_channel: str,
        sync_mode: str,
    ) -> None:
        # Lazy imports so the GUI can open without PySpin/nidaqmx present.
        from led_daq import LedController
        from session_logger import SessionLogger

        self.error = None
        self.result_summary = None
        self._finalized = False

        led = LedController(device=device, ao_channel=ao_channel)
        led.open()

        # Camera is owned by the window (shared with live preview) and is
        # already open + configured to the requested frame rate.
        model = camera.model_name()

        logger = SessionLogger(
            output_dir=output_dir,
            session_name=session_name,
            current_state_fn=led.current_state,
            metadata={
                "camera_model": model,
                "requested_frame_rate": frame_rate,
                "actual_frame_rate": camera.frame_rate,
                "n_frames": n_frames,
                "daq_device": device,
                "ao_channel": ao_channel,
                "sync_mode": sync_mode,
                "timeline": {
                    "repeat_count": timeline.repeat_count,
                    "intervals": [
                        {
                            "state": iv.state,
                            "duration_ms": iv.duration_ms,
                            "intensity_ma": iv.intensity_ma,
                        }
                        for iv in timeline.intervals
                    ],
                },
            },
        )
        logger.open()

        self._camera = camera
        self._led = led
        self._logger = logger
        self.running = True

        # Camera drives session length (frame count / stop button); LED timeline
        # runs alongside and returns the LED to 0 mA when it completes.
        camera.record(
            output_basepath=logger.video_basepath,
            n_frames=n_frames,
            on_frame=logger.log_frame,
            on_finished=self._on_camera_finished,
        )
        led.run_timeline(timeline, on_transition=logger.log_transition)

    def _on_camera_finished(
        self, completed: bool, written: int, incomplete: int
    ) -> None:
        # Runs on the camera worker thread. Stop the LED and finalize once.
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
        try:
            if self._led is not None:
                self._led.stop()
            if self._logger is not None:
                video_files = (
                    [os.path.basename(p) for p in self._camera.video_files]
                    if self._camera is not None
                    else []
                )
                meta_path = self._logger.finalize(
                    extra_meta={"completed": completed, "video_files": video_files}
                )
                self.result_summary = (
                    f"{'Completed' if completed else 'Stopped'}: "
                    f"{written} frames written, {incomplete} incomplete. "
                    f"Meta: {meta_path.name}"
                )
        except Exception as exc:  # noqa: BLE001 - surfaced to GUI status
            self.error = str(exc)
        finally:
            self.running = False

    def stop(self) -> None:
        """User-initiated stop; camera stop triggers coordinated finalize."""
        if self._camera is not None:
            self._camera.stop()

    def close(self) -> None:
        # The camera is owned by the window (shared with preview), so only the
        # LED task is released here.
        if self._led is not None:
            self._led.close()
            self._led = None

    # -- live status (polled by the GUI timer) -------------------------------

    def status(self) -> dict:
        led_state, led_ma = (False, 0.0)
        if self._led is not None:
            led_state, led_ma = self._led.current_state()
        return {
            "running": self.running,
            "frames": self._logger.frames_logged if self._logger else 0,
            "incomplete": self._logger.frames_incomplete if self._logger else 0,
            "led_state": led_state,
            "led_ma": led_ma,
            "error": self.error,
            "summary": self.result_summary,
        }


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Opto-Camera Control")
        self.controller = RecordingController()
        self._start_time = 0.0

        # Shared camera (used by both live preview and recording), plus the
        # latest preview frame handed off from the camera thread under a lock.
        self.camera = None
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._resume_preview_after_record = False

        # Transient LED controller for the manual "LED On (test)" control. Held
        # only while testing; released on Off and before any recording so it
        # never contends with the recording's own AO task.
        self._manual_led = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.addWidget(self._build_preview_panel())
        layout.addWidget(self._build_camera_panel())
        layout.addWidget(self._build_stimulus_panel())
        layout.addWidget(self._build_sync_panel())

        self.setStatusBar(QStatusBar())
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)  # 5 Hz
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

        # Separate, faster timer that paints the latest preview frame.
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(50)  # 20 Hz display
        self._preview_timer.timeout.connect(self._update_preview)

    # -- Preview panel -------------------------------------------------------

    def _build_preview_panel(self) -> QWidget:
        box = QGroupBox("Live preview")
        v = QVBoxLayout(box)

        self.preview_label = QLabel("preview off")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(480, 480)
        self.preview_label.setStyleSheet(
            "background-color: #202020; color: #888; border: 1px solid #444;"
        )
        v.addWidget(self.preview_label)

        self.preview_btn = QPushButton("Start Preview")
        self.preview_btn.setCheckable(True)
        self.preview_btn.clicked.connect(self._on_toggle_preview)
        v.addWidget(self.preview_btn)

        self.preview_status = QLabel("Aim the camera before recording.")
        v.addWidget(self.preview_status)
        return box

    def _on_toggle_preview(self) -> None:
        if self.preview_btn.isChecked():
            self._start_preview()
        else:
            self._stop_preview()

    def _ensure_camera(self):
        """Open the shared camera and apply all current settings.

        Geometry (ROI/binning) and frame rate can only change when not
        streaming, so they are applied here at each stream start. Exposure /
        gain / gamma are also (re)applied so a fresh stream matches the UI.
        """
        from camera import Camera

        if self.camera is None:
            cam = Camera()
            model = cam.open()
            self.camera = cam
            self.camera_status.setText(f"connected: {model}")
            self._populate_camera_settings()  # seed widgets from the camera
        cam = self.camera
        if not (cam.is_recording() or cam.is_previewing()):
            binning = 2 if self.binning_chk.isChecked() else 1
            cam.set_geometry(
                binning=binning,
                width=self.roi_w.value(),
                height=self.roi_h.value(),
                offset_x=self.roi_x.value(),
                offset_y=self.roi_y.value(),
            )
            cam.configure(frame_rate=self.frame_rate.value())
            cam.set_exposure(
                auto=self.exp_auto.isChecked(),
                microseconds=float(self.exp_spin.value()),
            )
            cam.set_gain(
                auto=self.gain_auto.isChecked(), db=float(self.gain_spin.value())
            )
            cam.set_gamma(float(self.gamma_spin.value()), enable=True)
        return cam

    def _start_preview(self) -> None:
        try:
            cam = self._ensure_camera()
            cam.start_preview(self._on_preview_frame)
        except ImportError as exc:
            self.preview_btn.setChecked(False)
            self._error(
                "Camera SDK not importable in this interpreter.\n"
                "Run from the Python 3.10 venv with PySpin installed.\n\n"
                f"{exc}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            self.preview_btn.setChecked(False)
            self._error(f"Could not start preview:\n{exc}")
            return
        self.preview_btn.setText("Stop Preview")
        self.preview_status.setText("Preview running.")
        self._preview_timer.start()

    def _stop_preview(self) -> None:
        self._preview_timer.stop()
        if self.camera is not None:
            self.camera.stop_preview()
        with self._frame_lock:
            self._latest_frame = None
        self.preview_btn.setChecked(False)
        self.preview_btn.setText("Start Preview")
        self.preview_status.setText("Preview stopped.")
        self.preview_label.setText("preview off")

    def _on_preview_frame(self, arr) -> None:
        # Called on the camera preview thread; just stash the frame.
        with self._frame_lock:
            self._latest_frame = arr

    def _update_preview(self) -> None:
        with self._frame_lock:
            arr = self._latest_frame
        if arr is None:
            return
        h, w = arr.shape[:2]
        data = arr.tobytes()
        image = QImage(data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(image).scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)

    # -- Camera panel --------------------------------------------------------

    def _build_camera_panel(self) -> QWidget:
        box = QGroupBox("Camera")
        form = QFormLayout(box)

        self.camera_status = QLabel("not connected")
        self.connect_btn = QPushButton("Connect / Detect")
        self.connect_btn.clicked.connect(self._on_connect)

        self.frame_rate = QDoubleSpinBox()
        self.frame_rate.setRange(0.1, 200.0)
        self.frame_rate.setValue(30.0)
        self.frame_rate.setSuffix(" fps")

        self.frame_count = QSpinBox()
        self.frame_count.setRange(0, 1_000_000)
        self.frame_count.setValue(500)
        self.frame_count.setToolTip("0 = record until Stop is pressed")

        self.session_name = QLineEdit(self._default_session_name())

        self.output_dir = QLineEdit(DEFAULT_OUTPUT_DIR)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._on_browse)
        out_row = QHBoxLayout()
        out_row.addWidget(self.output_dir)
        out_row.addWidget(browse)
        out_widget = QWidget()
        out_widget.setLayout(out_row)

        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_widget = QWidget()
        btn_widget.setLayout(btn_row)

        self.frames_label = QLabel("frames: 0  (incomplete: 0)")
        self.led_label = QLabel("LED: OFF")
        self.elapsed_label = QLabel("elapsed: 0.0 s")

        form.addRow("Status:", self.camera_status)
        form.addRow(self.connect_btn)
        form.addRow("Frame rate:", self.frame_rate)
        form.addRow("Frame count:", self.frame_count)
        form.addRow("Session name:", self.session_name)
        form.addRow("Output folder:", out_widget)
        form.addRow(btn_widget)
        form.addRow(self.frames_label)
        form.addRow(self.led_label)
        form.addRow(self.elapsed_label)
        form.addRow(self._build_image_settings())
        return box

    # -- Image settings (exposure / gain / gamma / ROI / binning) ------------

    def _build_image_settings(self) -> QWidget:
        box = QGroupBox("Image settings")
        form = QFormLayout(box)

        # Exposure: auto toggle + live slider/spinbox (microseconds).
        self.exp_auto = QCheckBox("Auto exposure")
        self.exp_auto.setChecked(True)
        self.exp_auto.toggled.connect(self._on_exposure_auto)
        form.addRow(self.exp_auto)
        self.exp_slider = QSlider(Qt.Orientation.Horizontal)
        self.exp_slider.setRange(16, 33257)
        self.exp_slider.setValue(5000)
        self.exp_spin = QSpinBox()
        self.exp_spin.setRange(16, 33257)
        self.exp_spin.setSuffix(" us")
        self.exp_spin.setValue(5000)
        self.exp_slider.valueChanged.connect(
            lambda v: self._on_exposure_changed(v, "slider")
        )
        self.exp_spin.valueChanged.connect(
            lambda v: self._on_exposure_changed(v, "spin")
        )
        form.addRow("Exposure:", self.exp_slider)
        form.addRow("", self.exp_spin)

        # Gain: auto toggle + live slider/spinbox (dB; slider is dB x10).
        self.gain_auto = QCheckBox("Auto gain")
        self.gain_auto.setChecked(True)
        self.gain_auto.toggled.connect(self._on_gain_auto)
        form.addRow(self.gain_auto)
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, 98)
        self.gain_slider.setValue(0)
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.0, 9.8)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setSuffix(" dB")
        self.gain_slider.valueChanged.connect(
            lambda v: self._on_gain_changed(v / 10.0, "slider")
        )
        self.gain_spin.valueChanged.connect(
            lambda v: self._on_gain_changed(v, "spin")
        )
        form.addRow("Gain:", self.gain_slider)
        form.addRow("", self.gain_spin)

        # Gamma (live).
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.5, 4.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.valueChanged.connect(self._on_gamma_changed)
        form.addRow("Gamma:", self.gamma_spin)

        # ROI + binning (applied when a stream starts; restarts preview).
        self.binning_chk = QCheckBox("2x2 binning (half resolution, full view)")
        self.binning_chk.toggled.connect(self._on_geometry_changed)
        form.addRow(self.binning_chk)
        self.roi_w = QSpinBox()
        self.roi_w.setRange(32, 2048)
        self.roi_w.setSingleStep(32)
        self.roi_w.setValue(2048)
        self.roi_h = QSpinBox()
        self.roi_h.setRange(2, 2048)
        self.roi_h.setSingleStep(2)
        self.roi_h.setValue(2048)
        self.roi_x = QSpinBox()
        self.roi_x.setRange(0, 2046)
        self.roi_x.setSingleStep(2)
        self.roi_y = QSpinBox()
        self.roi_y.setRange(0, 2046)
        self.roi_y.setSingleStep(2)
        for w in (self.roi_w, self.roi_h, self.roi_x, self.roi_y):
            w.editingFinished.connect(self._on_geometry_changed)
        form.addRow("ROI width:", self.roi_w)
        form.addRow("ROI height:", self.roi_h)
        form.addRow("ROI offset X:", self.roi_x)
        form.addRow("ROI offset Y:", self.roi_y)
        self.full_frame_btn = QPushButton("Full frame")
        self.full_frame_btn.clicked.connect(self._on_full_frame)
        form.addRow(self.full_frame_btn)

        # Auto on -> manual widgets disabled until the user unticks auto.
        for w in (self.exp_slider, self.exp_spin, self.gain_slider, self.gain_spin):
            w.setEnabled(False)
        self._last_geometry = None
        return box

    def _on_exposure_auto(self, auto: bool) -> None:
        self.exp_slider.setEnabled(not auto)
        self.exp_spin.setEnabled(not auto)
        if self.camera is not None:
            try:
                self.camera.set_exposure(
                    auto=auto, microseconds=float(self.exp_spin.value())
                )
            except Exception:  # noqa: BLE001
                pass

    def _on_exposure_changed(self, value: float, source: str) -> None:
        other = self.exp_spin if source == "slider" else self.exp_slider
        other.blockSignals(True)
        other.setValue(int(value))
        other.blockSignals(False)
        if self.camera is not None and not self.exp_auto.isChecked():
            try:
                self.camera.set_exposure(auto=False, microseconds=float(value))
            except Exception:  # noqa: BLE001
                pass

    def _on_gain_auto(self, auto: bool) -> None:
        self.gain_slider.setEnabled(not auto)
        self.gain_spin.setEnabled(not auto)
        if self.camera is not None:
            try:
                self.camera.set_gain(auto=auto, db=float(self.gain_spin.value()))
            except Exception:  # noqa: BLE001
                pass

    def _on_gain_changed(self, db: float, source: str) -> None:
        if source == "slider":
            self.gain_spin.blockSignals(True)
            self.gain_spin.setValue(db)
            self.gain_spin.blockSignals(False)
        else:
            self.gain_slider.blockSignals(True)
            self.gain_slider.setValue(int(round(db * 10)))
            self.gain_slider.blockSignals(False)
        if self.camera is not None and not self.gain_auto.isChecked():
            try:
                self.camera.set_gain(auto=False, db=float(db))
            except Exception:  # noqa: BLE001
                pass

    def _on_gamma_changed(self, value: float) -> None:
        if self.camera is not None:
            try:
                self.camera.set_gamma(float(value), enable=True)
            except Exception:  # noqa: BLE001
                pass

    def _on_full_frame(self) -> None:
        for w in (self.roi_x, self.roi_y):
            w.blockSignals(True)
            w.setValue(0)
            w.blockSignals(False)
        self.binning_chk.blockSignals(True)
        self.binning_chk.setChecked(False)
        self.binning_chk.blockSignals(False)
        self.roi_w.blockSignals(True)
        self.roi_w.setValue(self.roi_w.maximum())
        self.roi_w.blockSignals(False)
        self.roi_h.blockSignals(True)
        self.roi_h.setValue(self.roi_h.maximum())
        self.roi_h.blockSignals(False)
        self._on_geometry_changed()

    def _on_geometry_changed(self, *args: object) -> None:
        # ROI/binning can only change while not streaming; if a preview is
        # running, restart it to apply. Skip if the geometry is unchanged so
        # a spinbox losing focus doesn't needlessly restart the preview.
        geom = (
            2 if self.binning_chk.isChecked() else 1,
            self.roi_w.value(),
            self.roi_h.value(),
            self.roi_x.value(),
            self.roi_y.value(),
        )
        if geom == self._last_geometry:
            return
        self._last_geometry = geom
        if self.camera is None or self.camera.is_recording():
            return
        if self.camera.is_previewing():
            self._stop_preview()
            self.preview_btn.setChecked(True)
            self._start_preview()

    def _populate_camera_settings(self) -> None:
        """Read current values + ranges from the camera into the widgets."""
        info = self.camera.get_settings_info()

        d = info.get("exposure_us")
        if d:
            for w in (self.exp_slider, self.exp_spin):
                w.blockSignals(True)
                w.setRange(int(d["min"]), int(d["max"]))
                w.setValue(int(d["value"]))
                w.blockSignals(False)
        auto = info.get("exposure_auto") != "Off"
        self.exp_auto.blockSignals(True)
        self.exp_auto.setChecked(auto)
        self.exp_auto.blockSignals(False)
        self.exp_slider.setEnabled(not auto)
        self.exp_spin.setEnabled(not auto)

        d = info.get("gain_db")
        if d:
            self.gain_slider.blockSignals(True)
            self.gain_slider.setRange(0, int(round(d["max"] * 10)))
            self.gain_slider.setValue(int(round(d["value"] * 10)))
            self.gain_slider.blockSignals(False)
            self.gain_spin.blockSignals(True)
            self.gain_spin.setRange(d["min"], d["max"])
            self.gain_spin.setValue(d["value"])
            self.gain_spin.blockSignals(False)
        gauto = info.get("gain_auto") != "Off"
        self.gain_auto.blockSignals(True)
        self.gain_auto.setChecked(gauto)
        self.gain_auto.blockSignals(False)
        self.gain_slider.setEnabled(not gauto)
        self.gain_spin.setEnabled(not gauto)

        d = info.get("gamma")
        if d:
            self.gamma_spin.blockSignals(True)
            self.gamma_spin.setRange(d["min"], d["max"])
            self.gamma_spin.setValue(d["value"])
            self.gamma_spin.blockSignals(False)

        for spin, key in (
            (self.roi_w, "width"),
            (self.roi_h, "height"),
        ):
            d = info.get(key)
            if d:
                spin.blockSignals(True)
                spin.setRange(int(d["min"]), int(d["max"]))
                spin.setSingleStep(max(1, int(d["inc"])))
                spin.setValue(int(d["value"]))
                spin.blockSignals(False)
        for spin, key in ((self.roi_x, "offset_x"), (self.roi_y, "offset_y")):
            d = info.get(key)
            if d:
                spin.blockSignals(True)
                spin.setValue(int(d["value"]))
                spin.blockSignals(False)
        d = info.get("binning")
        if d:
            self.binning_chk.blockSignals(True)
            self.binning_chk.setChecked(int(d["value"]) >= 2)
            self.binning_chk.blockSignals(False)
        self._last_geometry = (
            2 if self.binning_chk.isChecked() else 1,
            self.roi_w.value(),
            self.roi_h.value(),
            self.roi_x.value(),
            self.roi_y.value(),
        )

    # -- Stimulus panel ------------------------------------------------------

    def _build_stimulus_panel(self) -> QWidget:
        box = QGroupBox("Stimulus timeline")
        v = QVBoxLayout(box)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["State", "Duration (ms)", "Intensity (mA)"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.cellChanged.connect(self._on_cell_changed)
        v.addWidget(self.table)

        row_btns = QHBoxLayout()
        for label, slot in (
            ("Add", self._add_row),
            ("Remove", self._remove_row),
            ("Up", lambda: self._move_row(-1)),
            ("Down", lambda: self._move_row(1)),
        ):
            b = QPushButton(label)
            b.clicked.connect(slot)
            row_btns.addWidget(b)
        row_btns_widget = QWidget()
        row_btns_widget.setLayout(row_btns)
        v.addWidget(row_btns_widget)

        rc_row = QHBoxLayout()
        rc_row.addWidget(QLabel("Repeat count:"))
        self.repeat_count = QSpinBox()
        self.repeat_count.setRange(1, 1_000_000)
        self.repeat_count.setValue(1)
        self.repeat_count.valueChanged.connect(self._update_total_duration)
        rc_row.addWidget(self.repeat_count)
        rc_row.addStretch()
        rc_widget = QWidget()
        rc_widget.setLayout(rc_row)
        v.addWidget(rc_widget)

        self.total_duration_label = QLabel("Total duration: 0 ms")
        v.addWidget(self.total_duration_label)

        self.mod_mode_note = QLabel("⚠  LED driver (LEDD1B) must be in MOD mode")
        self.mod_mode_note.setStyleSheet("color: #e0a000; font-weight: bold;")
        v.addWidget(self.mod_mode_note)

        # Seed with a simple ON/OFF example.
        self._add_row(state=True, duration_ms=1000, intensity_ma=500.0)
        self._add_row(state=False, duration_ms=1000, intensity_ma=0.0)
        return box

    def _add_row(
        self,
        checked: bool = False,
        state: bool = True,
        duration_ms: int = 1000,
        intensity_ma: float = 500.0,
    ) -> None:
        r = self.table.rowCount()
        self.table.blockSignals(True)
        self.table.insertRow(r)

        combo = QComboBox()
        combo.addItems(["ON", "OFF"])
        combo.setCurrentIndex(0 if state else 1)
        combo.currentIndexChanged.connect(self._on_row_state_changed)
        self.table.setCellWidget(r, 0, combo)

        dur = QTableWidgetItem(str(duration_ms))
        self.table.setItem(r, 1, dur)
        # The ON intensity is stored in the item's UserRole so it survives being
        # blanked while the row is OFF; the visible text is derived from state.
        inten = QTableWidgetItem()
        inten.setData(Qt.ItemDataRole.UserRole, float(intensity_ma))
        self.table.setItem(r, 2, inten)
        self.table.blockSignals(False)
        self._refresh_intensity_cell(r)
        self._update_total_duration()

    def _refresh_intensity_cell(self, r: int) -> None:
        """Show/enable the Intensity cell for ON rows; blank+grey it for OFF."""
        combo = self.table.cellWidget(r, 0)
        item = self.table.item(r, 2)
        if combo is None or item is None:
            return
        on = combo.currentIndex() == 0
        self.table.blockSignals(True)
        if on:
            val = item.data(Qt.ItemDataRole.UserRole)
            item.setText(f"{float(val or 0):g}")
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QBrush())
        else:
            # Preserve any number currently shown as the row's ON value.
            txt = item.text().strip()
            if txt:
                try:
                    item.setData(Qt.ItemDataRole.UserRole, float(txt))
                except ValueError:
                    pass
            item.setText("")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QBrush(QColor("#3a3a3a")))
        self.table.blockSignals(False)

    def _row_of_widget(self, w) -> Optional[int]:
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 0) is w:
                return r
        return None

    def _on_row_state_changed(self, *args: object) -> None:
        r = self._row_of_widget(self.sender())
        if r is not None:
            self._refresh_intensity_cell(r)
            self._update_total_duration()

    def _on_cell_changed(self, row: int, col: int) -> None:
        # Keep the stored ON value in sync when the user edits an ON row's mA.
        if col == 2:
            combo = self.table.cellWidget(row, 0)
            item = self.table.item(row, 2)
            if combo is not None and item is not None and combo.currentIndex() == 0:
                try:
                    item.setData(Qt.ItemDataRole.UserRole, float(item.text()))
                except ValueError:
                    pass
        self._update_total_duration()

    def _set_row_intensity(self, r: int, value: float) -> None:
        item = self.table.item(r, 2)
        if item is not None:
            item.setData(Qt.ItemDataRole.UserRole, float(value))
            self._refresh_intensity_cell(r)

    def _remove_row(self) -> None:
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._update_total_duration()

    def _move_row(self, delta: int) -> None:
        r = self.table.currentRow()
        n = self.table.rowCount()
        if r < 0 or not (0 <= r + delta < n):
            return
        self._swap_rows(r, r + delta)
        self.table.setCurrentCell(r + delta, 0)

    def _swap_rows(self, a: int, b: int) -> None:
        # Read both rows, then rewrite swapped (cell widgets can't be moved).
        ra = self._read_row(a)
        rb = self._read_row(b)
        self._write_row(a, rb)
        self._write_row(b, ra)

    def _read_row(self, r: int) -> tuple[bool, str, float]:
        combo = self.table.cellWidget(r, 0)
        state = combo.currentIndex() == 0
        dur = self.table.item(r, 1).text() if self.table.item(r, 1) else "0"
        item = self.table.item(r, 2)
        val = item.data(Qt.ItemDataRole.UserRole) if item else None
        intensity = float(val) if val is not None else 0.0
        return state, dur, intensity

    def _write_row(self, r: int, data: tuple[bool, str, float]) -> None:
        state, dur, intensity = data
        combo = self.table.cellWidget(r, 0)
        combo.blockSignals(True)
        combo.setCurrentIndex(0 if state else 1)
        combo.blockSignals(False)
        self.table.blockSignals(True)
        self.table.item(r, 1).setText(dur)
        self.table.item(r, 2).setData(Qt.ItemDataRole.UserRole, float(intensity))
        self.table.blockSignals(False)
        self._refresh_intensity_cell(r)

    def _update_total_duration(self, *args: object) -> None:
        try:
            tl = self._build_timeline()
            self.total_duration_label.setText(
                f"Total duration: {tl.total_duration_ms()} ms "
                f"({tl.total_duration_ms() / 1000.0:.2f} s)"
            )
        except (StimulusError, ValueError):
            self.total_duration_label.setText("Total duration: (invalid rows)")

    # -- Sync panel ----------------------------------------------------------

    def _build_sync_panel(self) -> QWidget:
        box = QGroupBox("Sync & DAQ")
        form = QFormLayout(box)

        self.sync_group = QButtonGroup(self)
        self.sync_software = QRadioButton("Software-sync (USB3 only)")
        self.sync_software.setChecked(True)
        self.sync_hardware = QRadioButton("Hardware-sync (GPIO strobe)")
        self.sync_hardware.setEnabled(False)
        self.sync_hardware.setToolTip(
            "Disabled until the camera GPIO breakout cable is wired to a DAQ "
            "digital line. Software-sync works today."
        )
        self.sync_group.addButton(self.sync_software)
        self.sync_group.addButton(self.sync_hardware)

        self.device = QLineEdit("Dev2")
        self.ao_channel = QLineEdit("ao0")

        form.addRow(self.sync_software)
        form.addRow(self.sync_hardware)
        form.addRow("DAQ device:", self.device)
        form.addRow("AO channel:", self.ao_channel)

        # --- Manual LED test (live brightness tuning) ---
        form.addRow(QLabel("<b>Manual LED test</b> (LEDD1B in MOD mode)"))

        self.led_slider = QSlider(Qt.Orientation.Horizontal)
        self.led_slider.setRange(0, MAX_LED_MA)
        self.led_slider.setValue(500)
        self.led_slider.valueChanged.connect(self._on_led_value_changed)

        self.led_spin = QSpinBox()
        self.led_spin.setRange(0, MAX_LED_MA)
        self.led_spin.setSuffix(" mA")
        self.led_spin.setValue(500)
        self.led_spin.valueChanged.connect(self._on_led_value_changed)

        form.addRow("Brightness:", self.led_slider)
        form.addRow("", self.led_spin)

        self.led_readout = QLabel()
        form.addRow(self.led_readout)

        self.led_test_btn = QPushButton("LED On (test)")
        self.led_test_btn.setCheckable(True)
        self.led_test_btn.clicked.connect(self._on_led_test_toggled)
        form.addRow(self.led_test_btn)

        self.apply_led_btn = QPushButton("Apply brightness to timeline ON rows")
        self.apply_led_btn.setToolTip(
            "Fill the current brightness (mA) into every ON row of the stimulus "
            "timeline, so the level you tuned here is what gets recorded."
        )
        self.apply_led_btn.clicked.connect(self._on_apply_led_to_timeline)
        form.addRow(self.apply_led_btn)

        self._on_led_value_changed(500)  # initialize readout
        return box

    def _on_apply_led_to_timeline(self) -> None:
        value = self.led_spin.value()
        applied = 0
        for r in range(self.table.rowCount()):
            state, _dur, _inten = self._read_row(r)
            if state:  # ON row
                self._set_row_intensity(r, value)
                applied += 1
        self._update_total_duration()
        self.statusBar().showMessage(
            f"Applied {value} mA to {applied} ON row(s) in the timeline."
        )

    # -- Manual LED control --------------------------------------------------

    def _on_led_value_changed(self, value: int) -> None:
        # Keep slider and spinbox in sync without recursing.
        for w in (self.led_slider, self.led_spin):
            if w.value() != value:
                w.blockSignals(True)
                w.setValue(value)
                w.blockSignals(False)
        volts = ma_to_voltage(value)
        pct = 100.0 * value / MAX_LED_MA
        self.led_readout.setText(
            f"→ {volts:.3f} V  (~{pct:.0f}% of the ~1 A limit)"
        )
        # If the LED is on for testing, apply the new level live.
        if self._manual_led is not None:
            try:
                self._manual_led.set_ma(float(value))
            except Exception:  # noqa: BLE001
                pass

    def _on_led_test_toggled(self) -> None:
        if self.led_test_btn.isChecked():
            try:
                from led_daq import LedController

                led = LedController(
                    device=self.device.text().strip(),
                    ao_channel=self.ao_channel.text().strip(),
                )
                led.open()
                led.set_ma(float(self.led_spin.value()))
                self._manual_led = led
            except ImportError as exc:
                self.led_test_btn.setChecked(False)
                self._error(
                    "DAQ SDK not importable in this interpreter.\n"
                    "Run from the Python 3.10 venv with nidaqmx installed.\n\n"
                    f"{exc}"
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.led_test_btn.setChecked(False)
                self._error(f"Could not turn on LED:\n{exc}")
                return
            self.led_test_btn.setText("LED Off")
        else:
            self._manual_led_off()

    def _manual_led_off(self) -> None:
        """Turn the manual test LED off and release its AO task."""
        if self._manual_led is not None:
            try:
                self._manual_led.close()  # forces AO to 0 V then releases task
            except Exception:  # noqa: BLE001
                pass
            self._manual_led = None
        self.led_test_btn.setChecked(False)
        self.led_test_btn.setText("LED On (test)")

    # -- actions -------------------------------------------------------------

    def _default_session_name(self) -> str:
        return "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def _on_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self.output_dir.text()
        )
        if path:
            self.output_dir.setText(path)

    def _on_connect(self) -> None:
        # If the shared camera is already open (e.g. from preview), just report
        # it rather than enumerating again (avoids touching the live stream).
        if self.camera is not None:
            self.camera_status.setText(f"connected: {self.camera.model_name()}")
            return
        try:
            from camera import Camera

            names = Camera.list_cameras()
        except ImportError as exc:
            self._error(
                "Camera SDK not importable in this interpreter.\n"
                "Run from the Python 3.10 venv with PySpin installed.\n\n"
                f"{exc}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._error(f"Camera detection failed:\n{exc}")
            return
        if names:
            self.camera_status.setText(f"found: {', '.join(names)}")
        else:
            self.camera_status.setText("no camera found")

    def _build_timeline(self) -> StimulusTimeline:
        intervals: list[Interval] = []
        for r in range(self.table.rowCount()):
            state, dur, inten = self._read_row(r)
            intervals.append(
                Interval(
                    state=state,
                    duration_ms=int(float(dur)),
                    intensity_ma=float(inten),
                )
            )
        return StimulusTimeline(intervals=intervals, repeat_count=self.repeat_count.value())

    def _on_start(self) -> None:
        try:
            timeline = self._build_timeline()
            timeline.validate()
        except (StimulusError, ValueError) as exc:
            self._error(f"Invalid stimulus timeline:\n{exc}")
            return

        session = self.session_name.text().strip() or self._default_session_name()

        # Pause live preview (one acquisition stream at a time) and remember to
        # resume it once recording finishes.
        self._resume_preview_after_record = self.preview_btn.isChecked()
        if self.camera is not None and self.camera.is_previewing():
            self._stop_preview()

        # Release the manual test LED so the recording's own AO task is free.
        self._manual_led_off()

        try:
            camera = self._ensure_camera()  # opens + configures to frame rate
            self.controller.start(
                camera=camera,
                output_dir=self.output_dir.text().strip(),
                session_name=session,
                frame_rate=self.frame_rate.value(),
                n_frames=self.frame_count.value(),
                timeline=timeline,
                device=self.device.text().strip(),
                ao_channel=self.ao_channel.text().strip(),
                sync_mode="hardware" if self.sync_hardware.isChecked() else "software",
            )
        except ImportError as exc:
            self._error(
                "Hardware SDK not importable in this interpreter.\n"
                "Run from the Python 3.10 venv with PySpin + nidaqmx installed.\n\n"
                f"{exc}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            self._error(f"Failed to start recording:\n{exc}")
            self.controller.close()
            return

        self._start_time = time.perf_counter()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_config_enabled(False)
        self.statusBar().showMessage(f"Recording '{session}'...")

    def _on_stop(self) -> None:
        self.controller.stop()

    def _refresh_status(self) -> None:
        st = self.controller.status()
        self.frames_label.setText(
            f"frames: {st['frames']}  (incomplete: {st['incomplete']})"
        )
        self.led_label.setText(
            f"LED: {'ON' if st['led_state'] else 'OFF'}  ({st['led_ma']:.0f} mA)"
        )
        if st["running"]:
            self.elapsed_label.setText(
                f"elapsed: {time.perf_counter() - self._start_time:.1f} s"
            )
        elif self.stop_btn.isEnabled():
            # Recording just ended; reset UI and report result.
            self._recording_ended(st)

    def _recording_ended(self, st: dict) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_config_enabled(True)
        self.controller.close()  # releases the LED task; camera stays open
        if st.get("error"):
            self._error(f"Recording error:\n{st['error']}")
            self.statusBar().showMessage("Recording failed.")
        elif st.get("summary"):
            self.statusBar().showMessage(st["summary"])
            self.session_name.setText(self._default_session_name())
        # Resume live preview if it was running before this recording.
        if self._resume_preview_after_record and self.camera is not None:
            self._resume_preview_after_record = False
            self.preview_btn.setChecked(True)
            self._start_preview()

    def _set_config_enabled(self, enabled: bool) -> None:
        for w in (
            self.frame_rate,
            self.frame_count,
            self.session_name,
            self.output_dir,
            self.table,
            self.repeat_count,
            self.device,
            self.ao_channel,
            self.connect_btn,
            self.preview_btn,
            self.led_slider,
            self.led_spin,
            self.led_test_btn,
            self.apply_led_btn,
            self.exp_auto,
            self.gain_auto,
            self.gamma_spin,
            self.binning_chk,
            self.roi_w,
            self.roi_h,
            self.roi_x,
            self.roi_y,
            self.full_frame_btn,
        ):
            w.setEnabled(enabled)
        # Exposure/gain manual widgets follow their auto checkbox when enabling.
        for w in (self.exp_slider, self.exp_spin):
            w.setEnabled(enabled and not self.exp_auto.isChecked())
        for w in (self.gain_slider, self.gain_spin):
            w.setEnabled(enabled and not self.gain_auto.isChecked())

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "Opto-Camera Control", message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        try:
            self._preview_timer.stop()
            self._manual_led_off()
            self.controller.stop()
            self.controller.close()
            if self.camera is not None:
                self.camera.close()  # stops preview + recording, releases handles
                self.camera = None
        finally:
            super().closeEvent(event)
