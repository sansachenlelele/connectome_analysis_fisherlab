"""LED / DAQ control for the opto-camera GUI.

Wraps NI-DAQmx (via the ``nidaqmx`` package) to drive the Thorlabs LEDD1B
through the NI PCIe-6351 analog output ``AO0``. The LEDD1B must be in
modulation mode with its current-limit dial physically set to 1.0 A; see
:mod:`stimulus` for the mA->volt conversion.

The stimulus timeline runs on a background thread so the GUI stays responsive.
On stop, close, or any error the analog output is forced back to 0 V so the LED
is never left on.

Importing this module requires ``nidaqmx`` (Windows + NI-DAQmx runtime). The
:mod:`stimulus` model is deliberately kept import-clean of hardware so it can be
tested without the DAQ present.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import nidaqmx  # type: ignore[import-not-found]

from stimulus import StimulusTimeline, ma_to_voltage

#: Callback signature invoked at each timeline transition:
#: ``(host_time_s, state, commanded_ma, voltage)``.
TransitionCallback = Callable[[float, bool, float, float], None]


class LedController:
    """Drives LED current on a single analog-output channel.

    Args:
        device: NI device name as it appears in NI MAX (e.g. ``"Dev1"``).
        ao_channel: Analog-output channel on that device (default ``"ao0"``).
        di_channel: Digital-input line for the camera strobe (hardware-sync
            mode). Unused until the GPIO cable arrives; see the strobe stubs.
    """

    def __init__(
        self,
        device: str = "Dev1",
        ao_channel: str = "ao0",
        di_channel: Optional[str] = None,
    ) -> None:
        self.device = device
        self.ao_channel = ao_channel
        self.di_channel = di_channel

        self._ao_task: Optional[nidaqmx.Task] = None
        self._state_lock = threading.Lock()
        # Snapshot of the currently commanded LED state, read per-frame by the
        # session logger so every frame gets an ON/OFF + mA label.
        self._current_state: tuple[bool, float] = (False, 0.0)

        self._timeline_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # -- lifecycle -----------------------------------------------------------

    @property
    def ao_physical_channel(self) -> str:
        return f"{self.device}/{self.ao_channel}"

    def open(self) -> None:
        """Create the AO task and drive the LED to 0 mA (off)."""
        if self._ao_task is not None:
            return
        task = nidaqmx.Task()
        task.ao_channels.add_ao_voltage_chan(
            self.ao_physical_channel, min_val=0.0, max_val=5.0
        )
        self._ao_task = task
        self._write_voltage(0.0)
        with self._state_lock:
            self._current_state = (False, 0.0)

    def close(self) -> None:
        """Stop any running timeline, force LED off, and release the task."""
        self.stop()
        if self._ao_task is not None:
            try:
                self._write_voltage(0.0)
            finally:
                self._ao_task.close()
                self._ao_task = None

    def __enter__(self) -> "LedController":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low-level output ----------------------------------------------------

    def _write_voltage(self, volts: float) -> None:
        if self._ao_task is None:
            raise RuntimeError("LedController is not open()")
        self._ao_task.write(volts)

    def set_ma(self, ma: float, state: Optional[bool] = None) -> float:
        """Command an LED current in mA and update the shared snapshot.

        Args:
            ma: Desired current in mA (clamped to the 0-1000 mA safe range by
                :func:`ma_to_voltage`).
            state: Explicit ON/OFF label for the snapshot. Defaults to
                ``ma > 0``.

        Returns:
            The host ``perf_counter`` timestamp (seconds) at which the write
            was issued.
        """
        volts = ma_to_voltage(ma)
        self._write_voltage(volts)
        ts = time.perf_counter()
        on = (ma > 0.0) if state is None else state
        with self._state_lock:
            self._current_state = (on, ma if on else 0.0)
        return ts

    def current_state(self) -> tuple[bool, float]:
        """Return the latest commanded ``(state, mA)`` snapshot (thread-safe)."""
        with self._state_lock:
            return self._current_state

    # -- timeline execution --------------------------------------------------

    def run_timeline(
        self,
        timeline: StimulusTimeline,
        on_transition: Optional[TransitionCallback] = None,
        on_finished: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Start running *timeline* on a background thread.

        Transitions are scheduled against a monotonic clock so cumulative drift
        does not accumulate across a long sequence. ``on_transition`` (if given)
        is called at every voltage change; ``on_finished`` is called once with
        ``completed=True`` if the whole timeline ran, or ``False`` if stopped
        early.
        """
        timeline.validate()
        if self._timeline_thread and self._timeline_thread.is_alive():
            raise RuntimeError("a timeline is already running")
        self._stop_event.clear()
        self._timeline_thread = threading.Thread(
            target=self._run_timeline_worker,
            args=(timeline, on_transition, on_finished),
            name="led-timeline",
            daemon=True,
        )
        self._timeline_thread.start()

    def _run_timeline_worker(
        self,
        timeline: StimulusTimeline,
        on_transition: Optional[TransitionCallback],
        on_finished: Optional[Callable[[bool], None]],
    ) -> None:
        completed = False
        start = time.perf_counter()
        try:
            for start_ms, state, ma in timeline.transitions():
                # Sleep until this transition's scheduled offset, waking early
                # if a stop is requested.
                target = start + start_ms / 1000.0
                while True:
                    remaining = target - time.perf_counter()
                    if remaining <= 0:
                        break
                    if self._stop_event.wait(min(remaining, 0.005)):
                        return  # stopped; finally-block forces LED off
                ts = self.set_ma(ma, state=state)
                if on_transition is not None:
                    volts = ma_to_voltage(ma)
                    on_transition(ts, state, ma, volts)
                if self._stop_event.is_set():
                    return
            # Hold the final interval for its duration before finishing.
            hold_until = start + timeline.total_duration_ms() / 1000.0
            while time.perf_counter() < hold_until:
                if self._stop_event.wait(0.005):
                    return
            completed = True
        finally:
            self.set_ma(0.0, state=False)
            if on_finished is not None:
                on_finished(completed)

    def stop(self) -> None:
        """Signal the timeline thread to stop and wait for it to exit."""
        self._stop_event.set()
        thread = self._timeline_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        # Best-effort: ensure LED is off even if no timeline was running.
        if self._ao_task is not None:
            try:
                self.set_ma(0.0, state=False)
            except Exception:
                pass

    def is_running(self) -> bool:
        return bool(self._timeline_thread and self._timeline_thread.is_alive())

    # -- hardware-sync strobe capture (STUB until GPIO cable arrives) --------

    def start_strobe_capture(self, sample_rate_hz: float) -> None:
        """Begin clocked capture of the camera exposure strobe on a DI line.

        Hardware-sync mode wires the Grasshopper3 GPIO strobe output into a DAQ
        digital input so frame exposures and AO0 writes share one clock. This
        is stubbed until the GPIO Hirose breakout cable is available and the
        physical DI line is chosen.
        """
        raise NotImplementedError(
            "Hardware-sync strobe capture pending: no GPIO breakout cable / "
            "DI channel configured yet. Use software-sync mode."
        )

    def read_strobe(self) -> list[float]:
        """Read buffered strobe samples aligned to the AO clock (STUB)."""
        raise NotImplementedError(
            "Hardware-sync strobe read pending: see start_strobe_capture()."
        )


def list_devices() -> list[str]:
    """Return NI-DAQmx device names visible on this system (e.g. ['Dev1'])."""
    system = nidaqmx.system.System.local()
    return [dev.name for dev in system.devices]
