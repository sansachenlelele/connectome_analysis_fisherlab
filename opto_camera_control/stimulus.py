"""Stimulus timeline data model for the opto-camera control GUI.

This module is pure Python with no hardware dependencies, so it can be imported
and unit-tested on any interpreter (including the rig's Python 3.13) without
PySpin / nidaqmx installed.

The LED is driven through a Thorlabs LEDD1B in *modulation mode* with the
front-panel current-limit dial physically set to 1.0 A. In that configuration a
0-5 V analog input maps linearly to 0-1000 mA of LED current, giving the
conversion in :func:`ma_to_voltage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

# --- Hardware constants (see opto_camera_control/DESIGN_NOTES.md) ------------

#: LEDD1B analog-input full-scale voltage (modulation mode).
AO_MAX_VOLTS: float = 5.0
#: LED current (mA) corresponding to AO_MAX_VOLTS, i.e. the 1.0 A dial setting.
CURRENT_LIMIT_MA: float = 1000.0


def clamp(value: float, low: float, high: float) -> float:
    """Return *value* constrained to the closed interval [*low*, *high*]."""
    return max(low, min(high, value))


def ma_to_voltage(ma: float, current_limit_ma: float = CURRENT_LIMIT_MA) -> float:
    """Convert a desired LED current in mA to the AO0 command voltage.

    In LEDD1B modulation mode the LED current is ``(V/5) * current_limit``, where
    ``current_limit`` is set by the front-panel dial. So to command ``ma`` of
    actual current: ``V = clamp(ma / current_limit * 5, 0, 5)``.

    ``current_limit_ma`` should match the physical dial (e.g. ~900 mA). It
    defaults to 1000 mA (full scale) for backwards compatibility. The clamp
    protects the LED: requests above the dial limit pin at 5 V (== the dial
    current) and negatives at 0 V.
    """
    limit = current_limit_ma if current_limit_ma and current_limit_ma > 0 else CURRENT_LIMIT_MA
    return clamp(ma / limit * AO_MAX_VOLTS, 0.0, AO_MAX_VOLTS)


class StimulusError(ValueError):
    """Raised when a timeline or interval fails validation."""


@dataclass
class Interval:
    """A single segment of the stimulus timeline.

    Attributes:
        state: ``True`` for LED ON, ``False`` for LED OFF.
        duration_ms: How long this segment lasts, in milliseconds (> 0).
        intensity_ma: LED current while ON, in mA. Ignored when ``state`` is
            False (an OFF segment always commands 0 mA).
    """

    state: bool
    duration_ms: int
    intensity_ma: float = 0.0

    def validate(self) -> None:
        if self.duration_ms <= 0:
            raise StimulusError(
                f"duration_ms must be > 0, got {self.duration_ms!r}"
            )
        if self.state:
            if not (0.0 <= self.intensity_ma <= CURRENT_LIMIT_MA):
                raise StimulusError(
                    f"intensity_ma must be within [0, {CURRENT_LIMIT_MA}] mA "
                    f"for an ON interval, got {self.intensity_ma!r}"
                )

    @property
    def commanded_ma(self) -> float:
        """The current this interval actually commands (0 mA when OFF)."""
        return self.intensity_ma if self.state else 0.0


@dataclass
class StimulusTimeline:
    """An ordered list of intervals, optionally repeated as a whole.

    A single ON/OFF pulse, a fixed pulse train, and an arbitrary custom
    sequence are all expressible: e.g. a pulse train is a 2-row
    ``[ON, OFF]`` timeline with ``repeat_count`` > 1.
    """

    intervals: list[Interval] = field(default_factory=list)
    repeat_count: int = 1

    def validate(self) -> None:
        if not self.intervals:
            raise StimulusError("timeline must contain at least one interval")
        if self.repeat_count < 1:
            raise StimulusError(
                f"repeat_count must be >= 1, got {self.repeat_count!r}"
            )
        for i, interval in enumerate(self.intervals):
            try:
                interval.validate()
            except StimulusError as exc:
                raise StimulusError(f"interval {i}: {exc}") from exc

    def single_pass_duration_ms(self) -> int:
        """Total duration of one pass through ``intervals`` (no repeats)."""
        return sum(iv.duration_ms for iv in self.intervals)

    def total_duration_ms(self) -> int:
        """Total duration of the whole timeline including repeats."""
        return self.single_pass_duration_ms() * self.repeat_count

    def transitions(self) -> Iterator[tuple[int, bool, float]]:
        """Yield ``(start_ms, state, commanded_ma)`` for every interval.

        ``start_ms`` is the offset from timeline start at which the interval
        begins (and therefore when the AO voltage should change). Repeats are
        expanded, so a timeline repeated N times yields N * len(intervals)
        entries with monotonically increasing ``start_ms``.
        """
        self.validate()
        t = 0
        for _ in range(self.repeat_count):
            for iv in self.intervals:
                yield t, iv.state, iv.commanded_ma
                t += iv.duration_ms
