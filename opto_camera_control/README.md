# Opto-Camera Control

A Windows GUI to record a single freely-walking fly from a FLIR Grasshopper3
camera while driving a Thorlabs LED through a custom ON/OFF stimulus timeline,
logging the LED state **per frame** so the video can be aligned with SLEAP
pose-tracking output afterward.

See [`DESIGN_NOTES.md`](DESIGN_NOTES.md) for the original hardware/design
rationale. This README covers setup and use.

## Hardware

- **Camera**: FLIR Grasshopper3 `GS3-U3-41C6NIR-C` over USB3.
- **DAQ**: NI PCIe-6351, analog output `Dev1/ao0` (confirm the device name in
  NI MAX) wired via the BNC-2090A `AO 0` BNC to the LEDD1B `MOD IN`.
- **LED driver**: Thorlabs LEDD1B in **Modulation mode**.
  - ⚠️ **Set the front-panel current-limit dial to 1.0 A before use.** Software
    cannot set this. At 1.0 A, `V_ao0 = clamp(mA/1000 * 5, 0, 5)`.
- **LED**: Thorlabs M590L3 (590 nm, 1000 mA max).

## Setup (once)

The camera SDK (PySpin, from Spinnaker 4.3) only supports Python ≤ 3.12; the
rig's system Python is 3.13, so this app runs in a **dedicated Python 3.10
venv**. This does not affect any other Python on the machine (including SLEAP's
own environment).

```powershell
# 1. Install Python 3.10 (64-bit) if not already present.
# 2. Create the venv (kept out of git):
py -3.10 -m venv opto_camera_control/.venv310

# 3. Install the pip dependencies:
opto_camera_control/.venv310/Scripts/python -m pip install -r opto_camera_control/requirements.txt

# 4. Install the matching PySpin wheel (NOT on PyPI). Download the Spinnaker 4.3
#    "Python" package from Teledyne and install the cp310 win_amd64 wheel:
opto_camera_control/.venv310/Scripts/python -m pip install spinnaker_python-4.3.0.190-cp310-cp310-win_amd64.whl

# 5. Smoke-test the SDKs are importable and see the hardware:
opto_camera_control/.venv310/Scripts/python -c "import PySpin, nidaqmx; print('ok')"
```

## Run

```powershell
opto_camera_control/.venv310/Scripts/python opto_camera_control/main.py
```

The window opens with any PySide6-capable interpreter for layout inspection, but
**Connect** and **Start** require the 3.10 venv with PySpin + nidaqmx.

Set frame rate, frame count (`0` = record until Stop), an output folder (Browse;
default `~/Documents/OptoRecordings`, deliberately outside this git repo), build
the stimulus timeline (rows of ON/OFF + duration + mA, plus a repeat count), then
**Start**.

## Camera image settings

The Camera panel's "Image settings" group exposes (each flexible — auto or
fixed):

- **Exposure** — auto toggle + live slider/µs box. Shorter exposure = less
  motion blur on a walking fly (needs more light).
- **Gain** — auto toggle + live slider/dB box. Prefer more LED light over gain
  (gain adds noise).
- **Gamma** — live; 1.0 = linear.
- **ROI (Width/Height/Offset)** — crop the readout to the arena for smaller
  files / higher frame rate. "Full frame" button resets it.
- **2×2 binning** — half resolution, full field of view, brighter/faster.

Exposure/gain/gamma update the **live preview** in real time; ROI/binning are
applied when a stream starts (changing them restarts the preview).

> ⚠️ **For opto experiments, use FIXED exposure and gain.** With auto enabled,
> the camera re-adjusts brightness every time the stimulus LED turns on/off,
> making the video inconsistent. Untick "Auto exposure"/"Auto gain" and set
> fixed values so the LED state is what changes, not the camera.

## Output files

All written to the chosen output folder, sharing the session name:

| File | Contents |
| --- | --- |
| `<session>.avi` | M-JPEG video (quality 80) |
| `<session>_timestamps.csv` | **one row per frame**: `frame_index, host_time_s, camera_timestamp_ns, led_state, led_ma, incomplete, strobe_sample` |
| `<session>_transitions.csv` | raw LED transition log |
| `<session>_meta.json` | frame rate, frame count, timeline, device, sync mode, clock origin, final frame/drop counts |

`host_time_s` is a monotonic `perf_counter` value; frame rows and transition
rows share that clock. `meta.json` records the wall-clock origin to convert if
needed.

## Aligning with SLEAP

After running SLEAP inference and exporting the analysis CSV (which has a
`frame_idx` column), the light state per frame is a plain merge:

```python
import pandas as pd
ts = pd.read_csv("session_timestamps.csv")
pose = pd.read_csv("session.analysis.csv")   # SLEAP export
merged = pose.merge(ts, left_on="frame_idx", right_on="frame_index")
```

Because the timestamps CSV is one row per frame (single fly → one instance per
frame), no timestamp interpolation is needed. The one caveat is **dropped
frames**: if the camera drops frames the two indexings could diverge, so the app
counts incomplete frames (`incomplete` column + `frames_incomplete` in
`meta.json`) — check that count is 0, or account for it, before merging.

## Sync modes

- **Software-sync** (default, works today): host timestamps on each frame grab
  and each LED transition.
- **Hardware-sync** (stubbed): wire the camera GPIO strobe (Hirose HR10) into a
  DAQ digital line so frames and AO writes share one clock. Disabled in the GUI
  until the breakout cable arrives; see the stubs in
  [`led_daq.py`](led_daq.py) (`start_strobe_capture` / `read_strobe`).

## Modules

| File | Role |
| --- | --- |
| `stimulus.py` | timeline data model + mA→V formula (hardware-free, unit-tested) |
| `led_daq.py` | nidaqmx LED control + timeline thread + hardware-sync stubs |
| `camera.py` | PySpin connect/configure/record + per-frame callback + drop tracking |
| `session_logger.py` | writes the per-frame CSV, transitions CSV, and meta.json |
| `gui.py` | PySide6 three-panel window + recording controller |
| `main.py` | entry point |
| `tests/test_stimulus.py` | unit tests for the hardware-free model |
