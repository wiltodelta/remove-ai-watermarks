"""Rational, half-open audio clocks for full-track resynthesis.

The model grid covers the source duration; the inverse is restricted to the
original native grid. Discarding inverse samples at/after the endpoint is not
length repair. Zero-phase filtering does not preserve edge amplitudes exactly.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from fractions import Fraction
from importlib import import_module
from math import gcd
from typing import Any


def is_available() -> bool:
    """Return whether the optional numerical dependencies can be imported."""
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy"))


@dataclass(frozen=True)
class RationalClock:
    """Exact native duration and the model grid covering that duration."""

    native_frames: int
    native_rate: int
    model_rate: int = 44100
    hop: int = 512
    minimum_model_frames: int = 1024

    def __post_init__(self) -> None:
        for name in ("native_frames", "native_rate", "model_rate", "hop", "minimum_model_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def duration(self) -> Fraction:
        return Fraction(self.native_frames, self.native_rate)

    @property
    def ratio(self) -> tuple[int, int]:
        divisor = gcd(self.native_rate, self.model_rate)
        return self.model_rate // divisor, self.native_rate // divisor

    @property
    def model_frames(self) -> int:
        up, down = self.ratio
        return (self.native_frames * up + down - 1) // down

    @property
    def inverse_frames(self) -> int:
        up, down = self.ratio
        return (self.model_frames * down + up - 1) // up

    @property
    def padded_frames(self) -> int:
        minimum = max(self.model_frames, self.minimum_model_frames)
        return ((minimum + self.hop - 1) // self.hop) * self.hop

    def record(self) -> dict[str, Any]:
        """Expose every boundary decision without rounding the source clock."""
        return {
            "native_frames": self.native_frames,
            "native_rate": self.native_rate,
            "duration": [self.duration.numerator, self.duration.denominator],
            "model_rate": self.model_rate,
            "forward_ratio": list(self.ratio),
            "model_frames": self.model_frames,
            "model_hop": self.hop,
            "model_right_reflection": self.padded_frames - self.model_frames,
            "inverse_frames": self.inverse_frames,
            "excluded_inverse_indices": [self.native_frames, self.inverse_frames],
            "boundary_policy": "constant-zero FIR exterior; right model reflection; inverse native half-open grid",
        }


def _array(samples: Any, frames: int) -> Any:
    if not is_available():
        raise RuntimeError("Audio resampling requires numpy and scipy")
    np = import_module("numpy")
    value = np.asarray(samples)
    if value.dtype.kind != "f" or value.ndim not in (1, 2) or value.shape[0] != frames:
        raise ValueError(f"Expected a floating audio array with {frames} frames")
    if value.ndim == 2 and value.shape[1] == 0:
        raise ValueError("Audio must contain at least one channel")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Audio samples must be finite in float32")
    return value


def forward(samples: Any, clock: RationalClock) -> Any:
    """Resample all native input samples, retaining ceil(T * model_rate)."""
    value = _array(samples, clock.native_frames)
    signal = import_module("scipy.signal")
    up, down = clock.ratio
    result = signal.resample_poly(value, up, down, axis=0, padtype="constant")
    return _array(result, clock.model_frames)


def pad_model(samples: Any, clock: RationalClock) -> Any:
    """Right-reflect to a hop boundary, with enough context for the mel FFT.

    The fixed 2048 FFT reflects 768 samples on each side. At least 1024
    model samples are therefore needed. A singleton repeats under numpy's
    reflect definition; no zero samples are appended to repair a clock.
    """
    value = _array(samples, clock.model_frames)
    np = import_module("numpy")
    widths = [(0, clock.padded_frames - clock.model_frames)] + [(0, 0)] * (value.ndim - 1)
    return np.pad(value, widths, mode="reflect")


def unpad_model(samples: Any, clock: RationalClock) -> Any:
    """Reject wrong model geometry before removing declared right context."""
    return _array(samples, clock.padded_frames)[: clock.model_frames]


def inverse(samples: Any, clock: RationalClock) -> Any:
    """Retain exactly the native sample instants k/native_rate < T."""
    value = _array(samples, clock.model_frames)
    signal = import_module("scipy.signal")
    up, down = clock.ratio
    result = signal.resample_poly(value, down, up, axis=0, padtype="constant")
    checked = _array(result, clock.inverse_frames)
    return _array(checked[: clock.native_frames], clock.native_frames)
