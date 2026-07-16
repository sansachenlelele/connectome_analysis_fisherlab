"""Per-frame logging for an opto-camera recording session.

Produces three files next to the video, all sharing the session name:

* ``<session>_timestamps.csv`` -- ONE ROW PER FRAME, keyed on ``frame_index``.
  The LED ON/OFF state and mA are resolved *per frame* (read from the live
  LedController snapshot at grab time), not merely at transitions. This is what
  makes downstream alignment with SLEAP a trivial merge on frame index:
  ``merge(sleap_df, ts_df, left_on="frame_idx", right_on="frame_index")``.
* ``<session>_transitions.csv`` -- the raw LED transition log (one row per
  voltage change), for provenance / debugging.
* ``<session>_meta.json`` -- recording settings, timeline, device, sync mode,
  clock references, and final frame/drop counts.

``host_time_s`` in both CSVs is a ``time.perf_counter()`` value (monotonic
seconds); frame rows and transition rows are directly comparable because both
clocks are the same. ``meta.json`` records the wall-clock and perf_counter
origin so those can be converted to absolute time if ever needed.
"""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Any, Optional


class SessionLogger:
    """Owns the CSV/JSON outputs for one recording session.

    Args:
        output_dir: Folder to write into (created if missing). Chosen by the
            user via the GUI Browse button; typically OUTSIDE the git repo.
        session_name: Base name shared by all output files.
        current_state_fn: Zero-arg callable returning ``(state, mA)`` for the
            LED right now -- normally ``LedController.current_state``.
        metadata: Extra key/values to fold into ``<session>_meta.json``
            (frame rate, frame count, timeline dict, device, sync mode, ...).
    """

    TIMESTAMP_FIELDS = [
        "frame_index",
        "host_time_s",
        "camera_timestamp_ns",
        "led_state",
        "led_ma",
        "incomplete",
        "strobe_sample",  # reserved for hardware-sync mode
    ]
    TRANSITION_FIELDS = ["host_time_s", "led_state", "led_ma", "voltage"]

    def __init__(
        self,
        output_dir: str,
        session_name: str,
        current_state_fn,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.session_name = session_name
        self._current_state_fn = current_state_fn
        self._metadata: dict[str, Any] = dict(metadata or {})

        self._lock = threading.Lock()
        self._ts_file = None
        self._ts_writer: Optional[csv.DictWriter] = None
        self._tr_file = None
        self._tr_writer: Optional[csv.DictWriter] = None

        self._frames_logged = 0
        self._frames_incomplete = 0
        self._t0_perf = 0.0
        self._t0_wall = 0.0

    # -- paths ---------------------------------------------------------------

    @property
    def video_basepath(self) -> str:
        """Path WITHOUT extension to hand to camera.record (SpinVideo adds .avi)."""
        return str(self.output_dir / self.session_name)

    def _path(self, suffix: str) -> Path:
        return self.output_dir / f"{self.session_name}{suffix}"

    # -- lifecycle -----------------------------------------------------------

    def open(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._t0_perf = time.perf_counter()
        self._t0_wall = time.time()

        self._ts_file = self._path("_timestamps.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._ts_writer = csv.DictWriter(self._ts_file, self.TIMESTAMP_FIELDS)
        self._ts_writer.writeheader()

        self._tr_file = self._path("_transitions.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._tr_writer = csv.DictWriter(self._tr_file, self.TRANSITION_FIELDS)
        self._tr_writer.writeheader()

    def __enter__(self) -> "SessionLogger":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.finalize()

    # -- callbacks (called from worker threads) ------------------------------

    def log_frame(
        self,
        frame_index: int,
        camera_timestamp_ns: int,
        host_time_s: float,
        incomplete: bool,
    ) -> None:
        """Write one per-frame row, tagging it with the live LED state."""
        state, ma = self._current_state_fn()
        with self._lock:
            if self._ts_writer is None:
                return
            self._ts_writer.writerow(
                {
                    "frame_index": frame_index,
                    "host_time_s": f"{host_time_s:.6f}",
                    "camera_timestamp_ns": camera_timestamp_ns,
                    "led_state": int(bool(state)),
                    "led_ma": f"{ma:.3f}",
                    "incomplete": int(bool(incomplete)),
                    "strobe_sample": "",
                }
            )
            self._frames_logged += 1
            if incomplete:
                self._frames_incomplete += 1

    def log_transition(
        self, host_time_s: float, state: bool, ma: float, voltage: float
    ) -> None:
        """Write one LED-transition row."""
        with self._lock:
            if self._tr_writer is None:
                return
            self._tr_writer.writerow(
                {
                    "host_time_s": f"{host_time_s:.6f}",
                    "led_state": int(bool(state)),
                    "led_ma": f"{ma:.3f}",
                    "voltage": f"{voltage:.4f}",
                }
            )

    # -- finish --------------------------------------------------------------

    def finalize(self, extra_meta: Optional[dict[str, Any]] = None) -> Path:
        """Flush and close the CSVs and write ``<session>_meta.json``."""
        with self._lock:
            if self._ts_file is not None:
                self._ts_file.close()
                self._ts_file = None
                self._ts_writer = None
            if self._tr_file is not None:
                self._tr_file.close()
                self._tr_file = None
                self._tr_writer = None

            meta = dict(self._metadata)
            if extra_meta:
                meta.update(extra_meta)
            meta.update(
                {
                    "session_name": self.session_name,
                    "frames_logged": self._frames_logged,
                    "frames_incomplete": self._frames_incomplete,
                    "clock": {
                        "t0_perf_counter_s": self._t0_perf,
                        "t0_wall_epoch_s": self._t0_wall,
                        "note": "host_time_s columns are perf_counter seconds; "
                        "wall_time = t0_wall + (host_time_s - t0_perf)",
                    },
                }
            )
            meta_path = self._path("_meta.json")
            with meta_path.open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=2)
            return meta_path

    @property
    def frames_logged(self) -> int:
        return self._frames_logged

    @property
    def frames_incomplete(self) -> int:
        return self._frames_incomplete
