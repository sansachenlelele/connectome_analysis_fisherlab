"""Unit tests for the hardware-free stimulus model.

Runnable on any interpreter (no PySpin/nidaqmx):

    python -m pytest opto_camera_control/tests/test_stimulus.py
    # or without pytest:
    python opto_camera_control/tests/test_stimulus.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from stimulus import (
    AO_MAX_VOLTS,
    CURRENT_LIMIT_MA,
    Interval,
    StimulusError,
    StimulusTimeline,
    ma_to_voltage,
)


# -- ma_to_voltage -----------------------------------------------------------

def test_voltage_zero():
    assert ma_to_voltage(0) == 0.0


def test_voltage_full_scale():
    assert ma_to_voltage(CURRENT_LIMIT_MA) == AO_MAX_VOLTS


def test_voltage_midpoint():
    assert ma_to_voltage(500) == pytest.approx(2.5)


def test_voltage_clamps_high():
    # Above the 1.0 A limit is pinned to full scale, protecting the LED.
    assert ma_to_voltage(5000) == AO_MAX_VOLTS


def test_voltage_clamps_negative():
    assert ma_to_voltage(-100) == 0.0


# -- Interval validation -----------------------------------------------------

def test_interval_rejects_zero_duration():
    with pytest.raises(StimulusError):
        Interval(state=True, duration_ms=0, intensity_ma=100).validate()


def test_interval_rejects_over_limit_current():
    with pytest.raises(StimulusError):
        Interval(state=True, duration_ms=10, intensity_ma=2000).validate()


def test_off_interval_commands_zero_ma_even_if_intensity_set():
    iv = Interval(state=False, duration_ms=10, intensity_ma=500)
    assert iv.commanded_ma == 0.0


def test_on_interval_commands_its_intensity():
    iv = Interval(state=True, duration_ms=10, intensity_ma=500)
    assert iv.commanded_ma == 500


# -- Timeline validation & durations -----------------------------------------

def test_empty_timeline_rejected():
    with pytest.raises(StimulusError):
        StimulusTimeline(intervals=[]).validate()


def test_repeat_count_below_one_rejected():
    tl = StimulusTimeline(
        intervals=[Interval(True, 10, 100)], repeat_count=0
    )
    with pytest.raises(StimulusError):
        tl.validate()


def test_total_duration_includes_repeats():
    tl = StimulusTimeline(
        intervals=[Interval(True, 100, 500), Interval(False, 200, 0)],
        repeat_count=3,
    )
    assert tl.single_pass_duration_ms() == 300
    assert tl.total_duration_ms() == 900


# -- transitions -------------------------------------------------------------

def test_transitions_offsets_and_repeats():
    tl = StimulusTimeline(
        intervals=[Interval(True, 100, 500), Interval(False, 200, 0)],
        repeat_count=2,
    )
    got = list(tl.transitions())
    assert got == [
        (0, True, 500.0),
        (100, False, 0.0),
        (300, True, 500.0),
        (400, False, 0.0),
    ]


def test_transitions_validates_first():
    tl = StimulusTimeline(intervals=[Interval(True, -5, 100)])
    with pytest.raises(StimulusError):
        list(tl.transitions())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
