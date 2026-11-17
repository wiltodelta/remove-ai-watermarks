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
    worst_window_rms_ratio: float = 1.0
    max_side_mid_drift_db: float | None = None


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
            fidelity.worst_window_rms_ratio,
        )
    ):
        return False
    if fidelity.max_side_mid_drift_db is not None and not isfinite(fidelity.max_side_mid_drift_db):
        return False
    if fidelity.source_rms >= 1e-3 and fidelity.output_rms < max(1e-5, fidelity.source_rms * 0.005):
        return False
    if fidelity.source_rms >= 1e-4 and fidelity.output_rms > fidelity.source_rms * 4.0:
        return False
    if fidelity.source_rms >= 1e-3 and (
        fidelity.worst_window_rms_ratio < 0.005 or fidelity.worst_window_rms_ratio > 4.0
    ):
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
    win_target = max(1, info.sample_rate)
    win_min_tail = max(1, win_target // 4)
    win_frames = 0
    win_src_power = win_out_power = 0.0
    win_mid_a = win_side_a = win_mid_b = win_side_b = 0.0
    min_win_ratio: float | None = None
    max_win_ratio: float | None = None
    max_sm_drift_db: float | None = None

    def flush_window() -> None:
        nonlocal win_frames, win_src_power, win_out_power, win_mid_a, win_side_a, win_mid_b, win_side_b
        nonlocal min_win_ratio, max_win_ratio, max_sm_drift_db
        if win_frames <= 0:
            return
        denom = win_frames * info.channels
        src_w = sqrt(win_src_power / denom)
        out_w = sqrt(win_out_power / denom)
        if src_w >= 1e-3:
            ratio = out_w / src_w
            min_win_ratio = ratio if min_win_ratio is None else min(min_win_ratio, ratio)
            max_win_ratio = ratio if max_win_ratio is None else max(max_win_ratio, ratio)
        if info.channels == 2:
            ma = sqrt(win_mid_a / win_frames)
            sa = sqrt(win_side_a / win_frames)
            mb = sqrt(win_mid_b / win_frames)
            sb = sqrt(win_side_b / win_frames)
            if ma >= 1e-3 and mb >= 1e-3 and (sa >= 1e-4 or sb >= 1e-4):
                sm_a = max(sa, 1e-6) / ma
                sm_b = max(sb, 1e-6) / mb
                drift = abs(20.0 * log10(sm_b / sm_a))
                max_sm_drift_db = drift if max_sm_drift_db is None else max(max_sm_drift_db, drift)
        win_frames = 0
        win_src_power = win_out_power = 0.0
        win_mid_a = win_side_a = win_mid_b = win_side_b = 0.0

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
            residual += float(np.sum(delta * delta))
            max_error = max(max_error, float(np.max(np.abs(delta))))
            source_peak = max(source_peak, float(np.max(abs_a)))
            peak = max(peak, float(np.max(abs_b)))
            source_above += int(np.count_nonzero(abs_a > 1))
            above += int(np.count_nonzero(abs_b > 1))
            count += pair_len

            pos = 0
            while pos < pair_len:
                take = min(win_target - win_frames, pair_len - pos)
                seg_a = a[pos : pos + take]
                seg_b = b[pos : pos + take]
                if info.channels == 2:
                    la, ra = seg_a[:, 0], seg_a[:, 1]
                    lb, rb = seg_b[:, 0], seg_b[:, 1]
                    pa = float(la @ la + ra @ ra)
                    pb = float(lb @ lb + rb @ rb)
                    ca = float(la @ ra)
                    cb = float(lb @ rb)
                    win_mid_a += 0.25 * (pa + 2.0 * ca)
                    win_side_a += 0.25 * (pa - 2.0 * ca)
                    win_mid_b += 0.25 * (pb + 2.0 * cb)
                    win_side_b += 0.25 * (pb - 2.0 * cb)
                else:
                    pa = float(np.sum(seg_a * seg_a))
                    pb = float(np.sum(seg_b * seg_b))
                power += pa
                output_power += pb
                win_src_power += pa
                win_out_power += pb
                win_frames += take
                pos += take
                if win_frames >= win_target:
                    flush_window()
        if win_frames >= win_min_tail or (min_win_ratio is None and win_frames > 0):
            flush_window()
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
    worst_ratio = 1.0
    if min_win_ratio is not None and max_win_ratio is not None:
        worst_ratio = max_win_ratio if max_win_ratio > 4.0 else min_win_ratio
    if info.channels == 2 and max_sm_drift_db is None:
        max_sm_drift_db = 0.0
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
        worst_window_rms_ratio=worst_ratio,
        max_side_mid_drift_db=max_sm_drift_db,
    )
