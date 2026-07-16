# Opto-Camera Control GUI — Design Notes

Handoff doc from the initial planning conversation (Mac / Claude Code session,
2026-07-16). Bring this into the new session on the Windows rig so we can
pick up implementation without re-deriving these decisions.

## Goal

A GUI to simultaneously (1) start/stop recording from a FLIR camera and
(2) drive a Thorlabs LED through a custom on/off stimulus timeline, logging
LED state alongside recorded frames.

## Hardware in the loop

- **Camera**: FLIR Grasshopper3 `GS3-U3-41C6NIR-C`, USB3 (not PCIe/DAQ
  controlled). 2048x2048, Raw8 pixel format (per current FlyCap2 settings).
  Currently operated via the FlyCap2 2.13.3.61 GUI app.
- **DAQ**: NI PCIe-6351 X-Series (16 AI, 24 DIO, 2 AO, 1.25 MS/s
  single-channel). Used only to drive the LED, not the camera.
- **Breakout**: NI BNC-2090A. Its `AO 0` BNC connector is wired 1:1 to the
  DAQ's `AO0` physical channel (e.g. `Dev1/ao0` in NI-DAQmx terms) — no
  jumpers/signal conditioning involved for this path.
- **LED driver**: Thorlabs LEDD1B, connected to `AO 0` via the MOD IN BNC
  input, set to **Modulation mode**. In this mode, 0-5V input maps linearly
  to 0-(current limit setting).
  - **Current-limit dial must be physically set to the 1.0 A position**
    (front-panel potentiometer — software can't touch this). 1.2A would
    exceed the LED's rating; 1.0A is the closest safe setting at/under the
    LED's max.
- **LED**: Thorlabs M590L3, 590nm, max continuous current 1000mA, Vf = 2.2V.
- **Voltage formula** (with current limit at 1.0A):
  `V_ao0 = clamp(desired_mA / 1000.0 * 5.0, 0.0, 5.0)`
- **Platform**: Windows only — NI-DAQmx and the FlyCapture2 SDK are
  Windows-native. This whole project must be developed/run on the Windows
  rig, not the Mac used for initial planning.

## Software decisions made so far

1. **Camera SDK**: `PyCapture2` (FlyCapture2 SDK's Python bindings) — matches
   the FlyCap2 app version already installed/in use, rather than switching to
   the newer Spinnaker/PySpin SDK.
2. **DAQ control**: `nidaqmx` Python package to write `AO0` voltage.
3. **Light pattern**: **custom stimulus timeline** — an ordered, editable list
   of intervals, each `{state: ON/OFF, duration_ms, intensity_mA (for ON)}`,
   with an optional repeat count for the whole sequence. This subsumes
   simpler cases (a single on/off, or a fixed pulse train is just a 2-row
   sequence repeated N times).
4. **Timestamp sync — support BOTH modes, selectable in the GUI**:
   - **Software-only (default for now)**: log system/camera timestamp each
     time a frame is grabbed, and system timestamp each time the AO0 voltage
     is changed (i.e., at each timeline transition). Written to a
     `<session>_timestamps.csv` alongside the video. This works today with no
     extra hardware, accuracy roughly single-digit ms (subject to OS/USB
     jitter).
   - **Hardware-synced (once available)**: the Grasshopper3's GPIO connector
     (6-pin Hirose HR10, separate from the USB3 cable) can be configured to
     output a strobe pulse (TTL high during exposure). Wiring that into a
     DAQ digital input (via a BNC-2090A spring terminal, or the PFI0 BNC) lets
     a single nidaqmx task sample the strobe on the *same clock* used to
     schedule the AO0 writes — giving frame-exact alignment.
   - User does not yet have the GPIO breakout cable for the camera, so
     hardware mode should exist in the architecture (config toggle + a DAQ
     digital channel setting) but can be a stub/TODO until the cable arrives
     and the physical wiring is decided.
   - **Important**: video codec/compression (e.g., M-JPEG @ quality 80, as
     currently used in FlyCap2) does NOT interfere with either timestamp
     approach — timestamps live in a separate CSV log, not embedded in
     compressed frame data. No conflict between wanting compressed AVI output
     and wanting synced timestamps.
5. **Recording settings mirror the existing FlyCap2 workflow** (see attached
   screenshot from original conversation): frame count (500 in the example),
   frame rate (should be GUI-adjustable, not fixed at 15fps), M-JPEG AVI
   output. Output folder should NOT be inside a git repo (videos are large,
   stored elsewhere per lab convention) — GUI should have a Browse button
   (like FlyCap2's) with a sensible default (e.g.
   `C:\Users\<you>\Documents\OptoRecordings\`), overridable per session.

## Proposed architecture (not yet implemented)

- `camera.py` — wraps PyCapture2: connect/enumerate, configure frame rate,
  start/stop capture + AVI recording, per-frame timestamp capture (camera
  embedded timestamp + host system time).
- `led_daq.py` — wraps nidaqmx: mA→V conversion helper, writes AO0, runs a
  stimulus timeline on a background thread/task with scheduled transitions,
  optional hardware-sync digital input task (stub until GPIO cable exists).
- `stimulus.py` — data model + validation for the custom timeline (ordered
  intervals + repeat count).
- `gui.py` — main window with three panels: Camera (connect, frame
  rate/count, output folder, start/stop, status), Stimulus (timeline table
  editor: add/remove/reorder rows, ON intensity in mA, repeat count), Sync
  settings (software vs. hardware-synced radio/toggle, hardware option
  disabled until a DAQ digital channel is configured).
- `main.py` — entry point; wires "Start" to launch camera recording and the
  stimulus timeline together, and writes the timestamps CSV on completion.
- GUI toolkit: not yet chosen — leaning PyQt6/PySide6 for a nicer layout
  closer to FlyCap2's, vs. Tkinter for zero extra dependencies. Open call to
  make with the user once implementation starts.

## Open items to settle on the Windows machine

- Confirm `PyCapture2` and `nidaqmx` are installed/importable in whatever
  Python environment will be used (get Python version, virtualenv/conda
  details).
- Get the exact NI MAX device name assigned to the PCIe-6351 (e.g. `Dev1`).
- Physically set the LEDD1B current-limit dial to 1.0A before first use.
- Confirm/adjust the default recording output folder path.
- Decide whether this project should live in its own new repo (recommended,
  since it's unrelated to the connectome-analysis codebase) vs. a subfolder
  here — this file was placed at
  `connectome_analysis_fisherlab/opto_camera_control/DESIGN_NOTES.md` only as
  a convenient place to land it during planning, not a final decision.
- Pick the GUI toolkit.
- When the camera GPIO breakout cable arrives: wire the strobe output into a
  chosen DAQ digital line, report the physical channel, and hardware-sync
  mode can be completed.
