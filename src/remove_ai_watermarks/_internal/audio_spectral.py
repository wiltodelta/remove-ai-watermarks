"""Stream the calibrated research band-stop with context at block boundaries."""

from __future__ import annotations

import importlib.util
import math
from importlib import import_module
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks._internal.audio_chunks import transform_chunks

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

BAND_HZ = (1000.0, 1500.0)
CONTEXT_SECONDS = 0.1


def is_available() -> bool:
    """Whether the optional CPU audio stack is installed."""
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy"))


def filter_audio_chunks(chunks: Iterable[NDArray[Any]], sample_rate: int) -> Iterator[NDArray[Any]]:
    """Filter independent channels, retaining each block's exact sample count.

    Nonfinal input blocks must contain at least CONTEXT_SECONDS of samples.
    Lookahead and retained input context keep block boundaries away from filter
    padding. Whole-file endpoints use SciPy's reflected padding, shortened only
    for an input too short for the normal pad length.
    """
    if not is_available():
        raise ImportError("Spectral audio processing requires remove-ai-watermarks[audio]")
    import numpy as np

    # SciPy does not ship typing stubs; keep its optional boundary local.
    signal = import_module("scipy.signal")

    if sample_rate <= 2 * BAND_HZ[1]:
        raise ValueError("Audio sample rate is too low for the research band-stop profile")
    sos = np.asarray(signal.butter(4, BAND_HZ, btype="bandstop", fs=sample_rate, output="sos"), dtype=np.float64)
    context = max(1, math.ceil(sample_rate * CONTEXT_SECONDS))
    default_pad = 3 * (2 * len(sos) + 1)

    def filter_block(padded: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(signal.sosfiltfilt(sos, padded, axis=0, padlen=min(default_pad, len(padded) - 1)))

    yield from transform_chunks(chunks, filter_block, context_frames=context)
