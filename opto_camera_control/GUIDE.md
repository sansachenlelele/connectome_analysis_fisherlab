# Opto-Camera Control — Complete Guide

This is the "understand the whole thing" document. If you just need to run it,
the [README](README.md) has quick setup and a reference table. This guide
explains **what the system does, how the pieces talk to each other, why it's
built the way it is, and how to operate it** — enough to maintain or extend it,
not just click buttons.

---

## 1. What this application is for

It runs **optogenetics behavior experiments on a single fly walking freely in an
arena**. In one action it must:

1. **Record video** of the fly from a FLIR camera, and
2. **Drive a Thorlabs LED** through a custom ON/OFF light schedule (the
   "stimulus timeline"), while
3. **logging, for every single video frame, whether the light was ON or OFF (and
   at what intensity)** — so that afterwards you can line up the fly's tracked
   behavior with the light stimulus.

The last point is the whole reason this tool exists. After recording, you track
the fly with **SLEAP** (a separate pose-estimation program). SLEAP gives you,
per frame, where the fly's body parts are. This tool gives you, per frame, what
the light was doing. Because both are indexed by **frame number**, you can merge
them and ask "what did the fly do when the light turned on?"

---

## 2. The core idea, in one picture

```
   GUI (you)                         Hardware                      Files out
   ─────────                         ────────                      ─────────
  frame rate ─┐                                                   session.avi
  timeline   ─┼─► Camera ──USB3──► FLIR Grasshopper3 ──► frames ─► session_timestamps.csv
  intensity  ─┘        │                                          session_transitions.csv
                       │                                          session_skipped.csv
             LED timeline ──► NI DAQ AO0 ──► BNC-2090A ──► LEDD1B ──► LED   session_meta.json
                                (voltage)      (BNC)      (MOD mode)  (light)
```

Two independent hardware paths run at the same time:
- **Video** comes in over the **USB3** cable from the camera.
- **Light** is controlled by a **voltage** the computer sends out of the DAQ's
  analog output, which the LED driver converts to LED current.

Software records the frames, runs the light schedule, and writes one CSV row per
frame tying them together.

---

## 3. The hardware

### Components and how they connect

```
[PC] ──USB3──────────────► [FLIR Grasshopper3 GS3-U3-41C6NIR]  (camera, monochrome)

[PC] ──PCIe (internal)──► [NI PCIe-6351 DAQ card]  device name "Dev2"
                               │ 68-pin SHC68-68-EPM cable
                               ▼
                          [BNC-2090A breakout]  "AO 0" BNC = Dev2/ao0
                               │ BNC coax
                               ▼
                          [Thorlabs LEDD1B driver]  MOD IN input, MOD mode
                               │ LED cable
                               ▼
                          [Thorlabs M590L3 LED]  590 nm amber
```

### The camera — FLIR Grasshopper3 `GS3-U3-41C6NIR`
- **Monochrome** (grayscale) sensor, **2048 × 2048** pixels (~4.2 MP). Despite a
  "-C" in some labels, it only offers Mono pixel formats — perfect for SLEAP.
- Connects over **USB3** only. (It also has a separate GPIO connector for a
  hardware sync strobe — see §9 — but that's not the video path and isn't wired
  yet.)
- We record in **Mono8** (8-bit grayscale) M-JPEG AVI.

### The DAQ — NI PCIe-6351
- A multifunction data-acquisition card inside the PC. We use exactly **one**
  feature: **analog output channel 0 (`Dev2/ao0`)**, which puts out a
  programmable 0–5 V.
- Its device name in software is **`Dev2`** on this rig (look it up in NI MAX;
  it could differ on another machine).

### The breakout — NI BNC-2090A
- A passive terminal block that turns the DAQ's 68-pin connector into labeled
  BNC jacks. The **`AO 0`** BNC is `Dev2/ao0`. Bus-powered from the card (the
  "+5V" LED confirms it's live). No configuration needed for the AO0 path.

### The LED driver — Thorlabs LEDD1B
- Converts the DAQ voltage into LED current. It has a **mode switch** and a
  **current-limit dial**, both physical (no computer connection, no readback).
- **Mode must be MOD (Modulation).** In MOD mode, LED current is proportional to
  the `MOD IN` voltage: `0–5 V` maps to `0 → the dial's current`. This is why we
  can set brightness from software. (CW = knob sets brightness, ignores voltage;
  TRIG = TTL-gated. Neither is what we want.)
- The **current-limit dial** sets full-scale current. It's set to **~0.9 A**
  here — deliberately below the LED's 1.0 A max for safety margin.

### The LED — Thorlabs M590L3
- 590 nm amber, **maximum forward current 1000 mA**, Vf ≈ 2.2 V. Must be on a
  heat sink. Because the dial is at ~0.9 A, we physically cannot exceed the
  1.0 A rating.

---

## 4. How the software talks to the hardware

There is no hand-rolled "send bytes to the camera." Everything goes through
layered libraries.

### Camera side (a stack)
```
our camera.py  →  PySpin (Python wheel)  →  Spinnaker C++ SDK  →  USB3 Vision driver  →  camera
```
- **USB3 Vision** is the transport protocol (how control messages and image data
  travel over USB3).
- **GenICam** is the camera's *feature model*: the camera advertises a big list
  of named settings ("nodes") — `AcquisitionFrameRate`, `ExposureTime`, `Width`,
  `PixelFormat`, etc.

There are two kinds of operation:
- **Control** — reading/writing a GenICam node. In code, `nodemap.GetNode("X")`
  then `.SetValue()/.GetValue()`; each is effectively a register write/read to
  the camera firmware. All the `_set_enum/_set_float/_set_int` helpers in
  `camera.py` do exactly this.
- **Streaming** — image data. `BeginAcquisition()` starts the stream, the camera
  fills buffers over USB3, and `GetNextImage()` hands us each frame.

The connect handshake (`Camera.open()`): `System.GetInstance()` (start the
library) → `GetCameras()` (enumerate) → `Init()` (open a session) →
`GetNodeMap()` (get the settings handle).

### LED side
```
our led_daq.py  →  nidaqmx (Python)  →  NI-DAQmx driver  →  DAQ card  →  AO0 voltage
```
We open one analog-output task and write a voltage; the DAQ holds that voltage
until we write a new one.

---

## 5. Why it's built this way (the decisions)

- **PySpin / Spinnaker, not PyCapture2.** The original plan used FLIR's older
  PyCapture2 SDK, but that only supports Python ≤ 3.6 (long dead). PySpin
  (Spinnaker 4.3) supports modern Python and its examples cover everything we
  need. The camera works with either; PySpin is the future-proof choice.
- **A dedicated Python 3.10 virtual environment (`.venv310`).** The camera
  bindings don't support the rig's system Python (3.13), and PySpin 4.3 on
  Windows officially targets 3.10. So the app runs in its own isolated venv;
  nothing else on the machine is affected (including SLEAP's own environment).
- **`numpy < 2`.** PySpin 4.3 is compiled against NumPy 1.x and crashes on
  import under NumPy 2. The venv pins `numpy<2`.
- **PySide6 for the GUI.** Already present, modern Qt6, permissive license.
- **One row per frame, keyed on `frame_index`.** The CSV is designed so it
  merges with SLEAP's `frame_idx` export with a trivial join — no timestamp
  interpolation. This is the single most important design constraint (see §8/§10).

---

## 6. Software architecture

All files are in `opto_camera_control/`.

| Module | Responsibility |
| --- | --- |
| `stimulus.py` | Pure-Python data model of the light schedule (`Interval`, `StimulusTimeline`) and the mA→voltage formula. No hardware imports, so it's unit-tested on any Python (`tests/test_stimulus.py`). |
| `led_daq.py` | `LedController`: opens the DAQ AO task, converts mA→volts, writes AO0, and runs the stimulus timeline on a background thread. Forces the LED to 0 V on stop/error. Holds stubs for the future hardware-sync digital input. |
| `camera.py` | `Camera`: connect/configure/record via PySpin; live preview; exposure/gain/gamma/ROI/binning setters; per-frame + skipped-frame callbacks; dropped-frame detection. |
| `session_logger.py` | `SessionLogger`: writes the per-frame CSV, the transitions CSV, the skipped-frames CSV, and the meta JSON. |
| `gui.py` | PySide6 window (Preview, Camera, Stimulus, Sync & DAQ panels) + `RecordingController` that wires camera + LED + logger together. |
| `main.py` | Entry point (`QApplication`, show window). |
| `Launch Opto-Camera Control.bat` | Double-click launcher that runs `main.py` in the venv. |

### Data flow during a recording
1. You press **Start**. The window opens/configures the shared camera and creates
   an `LedController` + `SessionLogger`.
2. `Camera.record()` starts a **camera thread**: grab frame → (if complete)
   append to AVI and call `logger.log_frame(frame_index, camera_ts, host_ts)`.
3. `LedController.run_timeline()` starts an **LED thread**: at each scheduled
   transition, write the AO voltage and call `logger.log_transition(...)`.
4. On each frame, the logger reads the LED controller's *current commanded state*
   snapshot, so every frame row is tagged ON/OFF + mA.
5. Camera thread finishes at the frame count (or Stop) → LED is driven to 0 →
   logger writes `meta.json`.

Threads never touch Qt widgets directly; the GUI polls counters on a timer.

---

## 7. LED intensity model (mA, volts, and the current limit)

In MOD mode: **LED current = (AO_voltage / 5 V) × current_limit**, where
`current_limit` is the dial setting.

To command a desired current, the software inverts that:

```
V_ao0 = clamp(desired_mA / current_limit_mA × 5,  0,  5)
```

- The **"LED current limit (dial)"** field in the GUI must be set to match the
  physical dial (**~900 mA** here). When it does, **the mA you type equals the
  actual LED current** (0 up to the limit), and the CSV's `led_ma` is true
  current.
- If you left it at 1000 while the dial is at 900, the numbers would overstate
  the real current by ~11% (harmless for relative brightness, wrong for absolute).
- The dial is an imprecise analog pot, so 900 is an estimate. For an exact figure,
  measure the real current (ammeter in series / across a sense resistor) and
  enter the measured value.

Safety: because the dial caps full-scale at ~0.9 A, you can never exceed the
LED's 1.0 A rating from software.

---

## 8. Frames: rate, timing, and integrity

### Frame rate
Set by writing the camera's `AcquisitionFrameRate` GenICam node (after disabling
its auto mode). **The camera's own hardware clock then paces the frames** — our
loop just blocks on `GetNextImage()` until each hardware-timed frame arrives, so
timing is precise. The requested rate is clamped to what's physically possible;
we read back and store both `requested_frame_rate` and `actual_frame_rate` in
`meta.json`. Achievable max depends on exposure and ROI (≈83 fps at full frame,
short exposure).

### Within an ON interval, the light is steady
An ON interval writes **one** DC voltage and holds it — the LED shines
**continuously, flicker-free**, at the set brightness (this is true analog
constant-current dimming, **not** PWM). Only the timeline switches between ON and
OFF segments.

### The frame ↔ light alignment guarantee
`frame_index` in `session_timestamps.csv` is the frame's **true position in the
AVI**. Frames that are corrupt ("incomplete") or that the camera drops (detected
as gaps in the camera's FrameID counter) are kept **out of both the video and the
timestamps CSV**, and recorded in `session_skipped.csv` instead. So:

> **timestamps CSV row N always corresponds to AVI frame N** — nothing can
> silently shift the alignment.

Check `frames_incomplete` and `frames_dropped` in `meta.json` (both 0 = clean).

---

## 9. Synchronization: software vs hardware

**Software-sync (default, works today).** The camera free-runs and the LED
timeline runs on its own software clock; each frame is labeled with the LED state
read when the frame is grabbed. This is accurate to **about one frame (~33 ms at
30 fps)** at each ON↔OFF boundary, because of:
- grab time lagging the actual exposure time over USB3 (with jitter),
- OS scheduling jitter on the LED thread, and
- a transition possibly happening *mid-exposure* (that one frame is physically
  part-lit but gets a binary label).

Only the single frame at each transition is uncertain; all interior frames are
exact. For opto intervals of hundreds of ms to seconds this is negligible (and
you can discard boundary frames if you need strictness).

**Hardware-sync (stubbed, for later).** The camera's GPIO connector can emit a
TTL strobe pulse during each exposure. Wiring that into a DAQ digital input would
let a single DAQ task sample the strobe on the **same clock** as the AO writes →
frame-exact, sub-millisecond alignment. The architecture has the toggle and
`led_daq.py` has stub methods (`start_strobe_capture` / `read_strobe`); it's
disabled in the GUI until the GPIO breakout cable is available and wired.

---

## 10. Output files and SLEAP alignment

Everything lands in the folder you chose (kept **outside** the git repo), sharing
the session name:

| File | What it is |
| --- | --- |
| `<session>.avi` | The M-JPEG video (quality 80). |
| `<session>_timestamps.csv` | **One row per video frame**, index-aligned to the AVI: `frame_index, host_time_s, camera_timestamp_ns, led_state, led_ma, strobe_sample`. |
| `<session>_transitions.csv` | Raw LED transitions: `host_time_s, led_state, led_ma, voltage`. |
| `<session>_skipped.csv` | Frames excluded from the video: `kind` (incomplete/dropped), `camera_frame_id`, `camera_timestamp_ns`, `host_time_s`, `count`, `note`. |
| `<session>_meta.json` | All settings + `frames_logged / frames_incomplete / frames_dropped` + clock origin. |

`host_time_s` is a monotonic `perf_counter` value shared by the frame and
transition logs; `meta.json` stores the wall-clock origin to convert if needed.

**Aligning with SLEAP.** After SLEAP inference, export the analysis CSV (it has a
`frame_idx` column) and merge on frame number:

```python
import pandas as pd
ts   = pd.read_csv("session_timestamps.csv")
pose = pd.read_csv("session.analysis.csv")     # SLEAP export
merged = pose.merge(ts, left_on="frame_idx", right_on="frame_index")
# now every tracked frame has led_state / led_ma next to it
```
Because the timestamps CSV is one row per video frame and index-aligned, no
interpolation is needed and skipped frames can't desync the join.

---

## 11. Operating the application (typical session)

1. **Hardware check:** camera on USB3; LEDD1B **in MOD mode**, current-limit dial
   at ~0.9 A; LED on its heat sink.
2. **Launch:** double-click `Launch Opto-Camera Control.bat`.
3. **Connect / Detect** (Camera panel) → should show the Grasshopper3.
4. **Frame the arena:** click **Start Preview** and aim the camera so the whole
   arena is in view. Use **ROI**/**Full frame** if you want to crop.
5. **Set image settings (Camera panel → Image settings):** for opto, **turn OFF
   auto exposure and auto gain** and set fixed values (see the warning below).
   Adjust exposure/gain/gamma live against the preview.
6. **Tune brightness (Sync & DAQ → Manual LED test):** set the **LED current
   limit (dial)** to match your dial (e.g. 900). Click **LED On (test)** and drag
   the slider until the scene looks right, then **Apply brightness to timeline ON
   rows**. Turn **LED Off**.
7. **Build the stimulus (Stimulus panel):** add rows of ON/OFF + duration (ms) +
   intensity (mA); set a repeat count. OFF rows have no intensity (greyed).
8. **Set frame rate, frame count** (`0` = until Stop), **session name**, and an
   **output folder** (Browse — outside the repo).
9. **Start.** Watch the frame counter climb and the LED toggle; the preview
   pauses during recording and resumes after.
10. **Verify:** confirm `frames_incomplete`/`frames_dropped` are 0 in
    `meta.json`; the four output files are in your folder.

> ⚠️ **Use FIXED exposure and gain for opto.** With auto enabled, the camera
> re-adjusts brightness every time the LED turns on/off, so the video brightness
> changes with your stimulus — exactly what you don't want. Fix them so the LED
> state is the only thing changing.

---

## 12. Installation / setup (one time)

See the [README](README.md) for the exact commands. In short:
1. Install **Python 3.10 (64-bit)**.
2. Create the venv: `py -3.10 -m venv opto_camera_control/.venv310`.
3. `pip install -r opto_camera_control/requirements.txt` (PySide6, nidaqmx,
   numpy<2).
4. Install the **PySpin wheel** matching Spinnaker 4.3 + cp310 + win_amd64
   (downloaded from Teledyne — not on PyPI).
5. Set the LEDD1B dial and MOD mode; confirm the DAQ device name in NI MAX.

The `.venv310` folder, videos, and CSVs are git-ignored so they never get
committed.

---

## 13. Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| GUI opens but **Connect/Start** errors about imports | You ran it outside the venv, or PySpin/nidaqmx aren't installed in it. Use the launcher / the venv Python. |
| `import PySpin` crashes about NumPy | NumPy 2 is installed; `pip install "numpy<2"` in the venv. |
| No camera found | Check USB3 cable; confirm the camera appears in SpinView/NI-independent; only one program can hold the camera at a time. |
| DAQ / AO errors | Wrong device name — check NI MAX (we default to `Dev2`); make sure no other program holds `ao0`. |
| LED doesn't respond to brightness | LEDD1B not in **MOD** mode, or MOD IN not wired to BNC-2090A AO 0. |
| Video brightness drifts with the stimulus | Auto exposure/gain is on — switch to fixed. |
| `frames_dropped` > 0 | USB3/disk/CPU couldn't keep up: lower frame rate or resolution (ROI/binning), or record to a faster disk. The alignment is still correct; the moments are just missing. |

---

## 14. Extending it (ideas / open items)

- **Hardware-sync mode:** wire the camera GPIO strobe to a DAQ digital line and
  fill in the `led_daq.py` stubs for frame-exact alignment.
- **Pulse-train helper:** a shortcut to generate many short ON/OFF rows for
  flicker stimuli (currently you'd add rows manually).
- **Exact current calibration:** measure real LED current and enter it as the
  current limit for absolute-intensity accuracy.
- **Live actual-frame-rate readout:** show the camera's accepted rate next to the
  frame-rate box (it can be below the request when exposure is long).

---

## 15. Glossary

- **ROI** — Region of Interest: a sub-rectangle of the sensor you read out
  (crops the field of view; smaller = smaller files, higher max fps).
- **Binning** — combining 2×2 pixels into one (half resolution, same field of
  view, brighter/faster).
- **MOD mode** — the LEDD1B modulation mode where LED current follows the input
  voltage.
- **GenICam node** — a named camera setting exposed by the SDK.
- **Incomplete frame** — a frame that arrived corrupt/partial.
- **Dropped frame** — a frame the camera produced but never delivered (detected
  as a gap in the FrameID sequence).
- **SLEAP** — the pose-tracking software used downstream to find the fly's body
  parts per frame.
