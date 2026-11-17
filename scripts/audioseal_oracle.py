#!/usr/bin/env python3
"""Development facade for the shared pinned AudioSeal model implementation."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks._internal.audioseal import (  # noqa: E402
    DETECTION_THRESHOLD,
    DETECTOR_CARD,
    DETECTOR_SHA256,
    GENERATOR_CARD,
    GENERATOR_SHA256,
    HF_REPO,
    HF_REVISION,
    MESSAGE_THRESHOLD,
    N_BITS,
    SAMPLE_RATE,
    AudioSealReading,
    available,
    detect_sample_array,
    embed,
    import_failure,
    load_pinned_models,
    read,
    read_intervals,
    window_bounds,
)

__all__ = [
    "DETECTION_THRESHOLD",
    "DETECTOR_CARD",
    "DETECTOR_SHA256",
    "GENERATOR_CARD",
    "GENERATOR_SHA256",
    "HF_REPO",
    "HF_REVISION",
    "MESSAGE_THRESHOLD",
    "N_BITS",
    "SAMPLE_RATE",
    "AudioSealReading",
    "available",
    "detect_sample_array",
    "embed",
    "import_failure",
    "load_pinned_models",
    "read",
    "read_intervals",
    "window_bounds",
]
