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
from PySide6.QtWidgets import (
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

from stimulus import Interval, StimulusError, StimulusTimeline

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
        from camera import Camera
        from led_daq import LedController
        from session_logger import SessionLogger

        self.error = None
        self.result_summary = None
        self._finalized = False

        led = LedController(device=device, ao_channel=ao_channel)
        led.open()

        camera = Camera()
        model = camera.open()
        camera.configure(frame_rate=frame_rate)

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
        try:
            if self._camera is not None:
                self._camera.close()
        finally:
            if self._led is not None:
                self._led.close()

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

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.addWidget(self._build_camera_panel())
        layout.addWidget(self._build_stimulus_panel())
        layout.addWidget(self._build_sync_panel())

        self.setStatusBar(QStatusBar())
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)  # 5 Hz
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

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
        return box

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
        self.table.cellChanged.connect(self._update_total_duration)
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
        self.table.insertRow(r)

        combo = QComboBox()
        combo.addItems(["ON", "OFF"])
        combo.setCurrentIndex(0 if state else 1)
        self.table.setCellWidget(r, 0, combo)

        dur = QTableWidgetItem(str(duration_ms))
        self.table.setItem(r, 1, dur)
        inten = QTableWidgetItem(f"{intensity_ma:g}")
        self.table.setItem(r, 2, inten)
        self._update_total_duration()

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

    def _read_row(self, r: int) -> tuple[bool, str, str]:
        combo = self.table.cellWidget(r, 0)
        state = combo.currentIndex() == 0
        dur = self.table.item(r, 1).text() if self.table.item(r, 1) else "0"
        inten = self.table.item(r, 2).text() if self.table.item(r, 2) else "0"
        return state, dur, inten

    def _write_row(self, r: int, data: tuple[bool, str, str]) -> None:
        state, dur, inten = data
        self.table.cellWidget(r, 0).setCurrentIndex(0 if state else 1)
        self.table.item(r, 1).setText(dur)
        self.table.item(r, 2).setText(inten)

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

        self.device = QLineEdit("Dev1")
        self.ao_channel = QLineEdit("ao0")

        form.addRow(self.sync_software)
        form.addRow(self.sync_hardware)
        form.addRow("DAQ device:", self.device)
        form.addRow("AO channel:", self.ao_channel)
        return box

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
        try:
            self.controller.start(
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
        self.controller.close()
        if st.get("error"):
            self._error(f"Recording error:\n{st['error']}")
            self.statusBar().showMessage("Recording failed.")
        elif st.get("summary"):
            self.statusBar().showMessage(st["summary"])
            self.session_name.setText(self._default_session_name())

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
        ):
            w.setEnabled(enabled)

    def _error(self, message: str) -> None:
        QMessageBox.critical(self, "Opto-Camera Control", message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        try:
            self.controller.stop()
            self.controller.close()
        finally:
            super().closeEvent(event)
