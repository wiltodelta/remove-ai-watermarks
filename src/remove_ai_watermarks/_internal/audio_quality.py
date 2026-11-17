"""Delivered-audio fidelity and content hashes shared by audio and video paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import zip_longest
from math import isfinite, log10, sqrt
from typing import TYPE_CHECKING

from remove_ai_watermarks._internal import audio_io
from remove_ai_watermarks._internal.constants import AUDIO_LOSSY_OUTPUT_EXTENSIONS

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class AudioFidelity:
    """Delivered-file sample differences against the decoded marked source."""

    frames: int
    rms_error: float
    max_abs_error: float
    snr_db: float | None
    peak_amplitude: float
    samples_above_full_scale: int
    source_rms: float = 0.0
    output_rms: float = 0.0
    source_peak: float = 0.0
    source_samples_above_full_scale: int = 0


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audio_quality_passed(fidelity: AudioFidelity, *, lossy: bool = False) -> bool:
    """Evaluate model-free delivered audio integrity against source energy and full scale."""
    if fidelity.frames <= 0:
        return False
    if not all(
        isfinite(value)
        for value in (
            fidelity.rms_error,
            fidelity.max_abs_error,
            fidelity.peak_amplitude,
            fidelity.source_rms,
            fidelity.output_rms,
            fidelity.source_peak,
        )
    ):
        return False
    if fidelity.source_rms >= 1e-3 and fidelity.output_rms < max(1e-5, fidelity.source_rms * 0.005):
        return False
    if fidelity.source_rms >= 1e-4 and fidelity.output_rms > fidelity.source_rms * 4.0:
        return False
    if lossy:
        allowed_above = max(fidelity.source_samples_above_full_scale, max(16, fidelity.frames // 500))
        if fidelity.samples_above_full_scale > allowed_above:
            return False
        if fidelity.peak_amplitude > max(1.05, fidelity.source_peak * 1.05):
            return False
    else:
        if fidelity.samples_above_full_scale > fidelity.source_samples_above_full_scale:
            return False
        if fidelity.source_peak <= 1.0 and fidelity.peak_amplitude > 1.0:
            return False
    return True


def measure_fidelity(source: Path, info: audio_io.AudioStreamInfo, output: Path, *, stream: int = 0) -> AudioFidelity:
    import numpy as np

    suffix = output.suffix.lower()
    opus_clock = suffix in {".ogg", ".opus"}
    delivered = audio_io.probe_audio(output, stream=stream)
    expected_sr = 48000 if opus_clock else info.sample_rate
    if (delivered.channels, delivered.sample_rate) != (info.channels, expected_sr):
        raise RuntimeError("Audio output changed the source sample rate or channel count")
    if (
        info.channel_layout
        and suffix not in AUDIO_LOSSY_OUTPUT_EXTENSIONS
        and delivered.channel_layout != info.channel_layout
    ):
        raise RuntimeError("Audio output changed the source channel layout")
    decode_info = replace(delivered, sample_rate=info.sample_rate) if opus_clock else delivered
    left, right = audio_io.read_audio(source, info), audio_io.read_audio(output, decode_info)
    power = output_power = residual = max_error = peak = source_peak = 0.0
    count = above = source_above = 0
    tail_slack = max(2, info.sample_rate // 100) if opus_clock else 0
    total_before = total_after = 0
    try:
        for before, after in zip_longest(left, right):
            if before is not None:
                total_before += len(before)
            if after is not None:
                total_after += len(after)
            if not opus_clock:
                if before is None or after is None or before.shape != after.shape:
                    raise RuntimeError("Delivered audio changed the decoded sample-frame count")
                pair_len = len(before)
            else:
                if before is None or after is None:
                    break
                if before.shape[1:] != after.shape[1:]:
                    raise RuntimeError("Delivered audio changed the channel count")
                pair_len = min(len(before), len(after))
                before, after = before[:pair_len], after[:pair_len]
            if not np.isfinite(before).all() or not np.isfinite(after).all():
                raise RuntimeError("Delivered audio contains nonfinite samples")
            a, b = before.astype(np.float64), after.astype(np.float64)
            delta = b - a
            abs_a, abs_b = np.abs(a), np.abs(b)
            power += float(np.sum(a * a))
            output_power += float(np.sum(b * b))
            residual += float(np.sum(delta * delta))
            max_error = max(max_error, float(np.max(np.abs(delta))))
            source_peak = max(source_peak, float(np.max(abs_a)))
            peak = max(peak, float(np.max(abs_b)))
            source_above += int(np.count_nonzero(abs_a > 1))
            above += int(np.count_nonzero(abs_b > 1))
            count += pair_len
        if opus_clock:
            for extra_b in left:
                total_before += len(extra_b)
            for extra_a in right:
                total_after += len(extra_a)
            if abs(total_after - total_before) > tail_slack or count == 0:
                raise RuntimeError("Delivered audio changed the decoded sample-frame count")
    finally:
        for reader in (left, right):
            close = getattr(reader, "close", None)
            if callable(close):
                close()
    total_values = count * info.channels
    snr = 10 * log10(power / residual) if power > 0 and residual > 0 else None
    return AudioFidelity(
        count,
        sqrt(residual / total_values),
        max_error,
        snr,
        peak,
        above,
        source_rms=sqrt(power / total_values),
        output_rms=sqrt(output_power / total_values),
        source_peak=source_peak,
        source_samples_above_full_scale=source_above,
    )
