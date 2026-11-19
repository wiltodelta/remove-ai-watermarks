"""Delivered-audio fidelity and content hashes shared by audio and video paths."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import zip_longest
from math import isfinite, log10, sqrt
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks._internal import audio_io
from remove_ai_watermarks._internal.constants import AUDIO_LOSSY_OUTPUT_EXTENSIONS

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray


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


@dataclass(frozen=True)
class AudioPerceptualMetrics:
    """Phase-invariant multi-resolution STFT and Mel spectral distances."""

    mr_stft_loss: float
    mr_stft_sc: float
    mr_stft_log_mag: float
    mel_distance_db: float


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
    if fidelity.max_side_mid_drift_db is not None and (
        not isfinite(fidelity.max_side_mid_drift_db) or abs(fidelity.max_side_mid_drift_db) > (9.0 if lossy else 6.0)
    ):
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


def _opus_aligned_decode_info(
    info: audio_io.AudioStreamInfo,
    output: Path,
    *,
    stream: int = 0,
    delivered_info: audio_io.AudioStreamInfo | None = None,
    strict_geometry: bool = False,
) -> tuple[audio_io.AudioStreamInfo, bool]:
    suffix = output.suffix.lower()
    opus_clock = suffix in {".ogg", ".opus"}
    delivered = delivered_info if delivered_info is not None else audio_io.probe_audio(output, stream=stream)
    if strict_geometry:
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
    return decode_info, opus_clock


@contextmanager
def _paired_audio_readers(
    source: Path,
    source_info: audio_io.AudioStreamInfo,
    output: Path,
    decode_info: audio_io.AudioStreamInfo,
    *,
    chunk_frames: int | None = None,
) -> Generator[tuple[Iterator[NDArray[Any]], Iterator[NDArray[Any]]]]:
    left = (
        audio_io.read_audio(source, source_info)
        if chunk_frames is None
        else audio_io.read_audio(source, source_info, chunk_frames=chunk_frames)
    )
    right = (
        audio_io.read_audio(output, decode_info)
        if chunk_frames is None
        else audio_io.read_audio(output, decode_info, chunk_frames=chunk_frames)
    )
    try:
        yield left, right
    finally:
        for reader in (left, right):
            close = getattr(reader, "close", None)
            if callable(close):
                close()


def measure_fidelity(source: Path, info: audio_io.AudioStreamInfo, output: Path, *, stream: int = 0) -> AudioFidelity:
    import numpy as np

    decode_info, opus_clock = _opus_aligned_decode_info(info, output, stream=stream, strict_geometry=True)
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

    with _paired_audio_readers(source, info, output, decode_info) as (left, right):
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


_MR_STFT_RESOLUTIONS: tuple[tuple[int, int], ...] = ((512, 128), (1024, 256), (2048, 512))
_MEL_BANDS: int = 64
_MEL_N_FFT, _MEL_HOP = _MR_STFT_RESOLUTIONS[-1]
_PERCEPTUAL_BLOCK_FRAMES: int = 163840


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> NDArray[np.float64]:
    import numpy as np

    f_min = 20.0
    f_max = min(20000.0, 0.49 * float(sample_rate))
    if f_max <= f_min:
        f_min = 0.0
        f_max = 0.49 * float(sample_rate)

    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * log10(1.0 + hz / 700.0)

    mels = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2, dtype=np.float64)
    hz_points = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate))
    fb = np.zeros((n_mels, len(fft_freqs)), dtype=np.float64)
    for m in range(n_mels):
        left_hz, center_hz, right_hz = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        up = (fft_freqs - left_hz) / max(center_hz - left_hz, 1e-9)
        down = (right_hz - fft_freqs) / max(right_hz - center_hz, 1e-9)
        w = np.maximum(0.0, np.minimum(up, down))
        norm = max(right_hz - left_hz, 1e-9)
        fb[m] = w * (2.0 / norm)
    return fb


def _stft_magnitudes(
    channel_wave: NDArray[np.float64], n_fft: int, hop: int, window: NDArray[np.float64]
) -> NDArray[np.float64]:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    arr = channel_wave
    if len(arr) < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: len(arr)] = arr
        arr = padded
    frames = sliding_window_view(arr, n_fft)[::hop]
    spec = np.fft.rfft(frames * window, axis=-1)
    return np.abs(spec)


def compute_perceptual_metrics(
    source: Path,
    info: audio_io.AudioStreamInfo,
    output: Path,
    *,
    stream: int = 0,
    delivered_info: audio_io.AudioStreamInfo | None = None,
) -> AudioPerceptualMetrics:
    """Compute bounded-memory multi-resolution STFT and 64-band log-Mel distances."""
    import numpy as np

    decode_info, _opus = _opus_aligned_decode_info(info, output, stream=stream, delivered_info=delivered_info)
    windows = {
        n_fft: 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft, dtype=np.float64) / n_fft)
        for n_fft, _hop in _MR_STFT_RESOLUTIONS
    }
    mel_fb = _mel_filterbank(info.sample_rate, _MEL_N_FFT, _MEL_BANDS)

    diff_sq = [0.0 for _ in _MR_STFT_RESOLUTIONS]
    src_sq = [0.0 for _ in _MR_STFT_RESOLUTIONS]
    log_l1 = [0.0 for _ in _MR_STFT_RESOLUTIONS]
    stft_bins = [0 for _ in _MR_STFT_RESOLUTIONS]
    mel_sq_db = 0.0
    mel_bins = 0

    buf_a: list[NDArray[Any]] = []
    buf_b: list[NDArray[Any]] = []
    buffered = 0

    def _process_block(block_a: NDArray[Any], block_b: NDArray[Any]) -> None:
        nonlocal mel_sq_db, mel_bins
        a_arr = np.asarray(block_a, dtype=np.float64)
        b_arr = np.asarray(block_b, dtype=np.float64)
        if len(a_arr) == 0:
            return
        for ch in range(a_arr.shape[1]):
            ca = a_arr[:, ch]
            cb = b_arr[:, ch]
            for idx, (n_fft, hop) in enumerate(_MR_STFT_RESOLUTIONS):
                mag_a = _stft_magnitudes(ca, n_fft, hop, windows[n_fft])
                mag_b = _stft_magnitudes(cb, n_fft, hop, windows[n_fft])
                delta_m = mag_b - mag_a
                diff_sq[idx] += float(np.sum(delta_m * delta_m))
                src_sq[idx] += float(np.sum(mag_a * mag_a))
                log_a = np.log(np.maximum(mag_a, 1e-5))
                log_b = np.log(np.maximum(mag_b, 1e-5))
                log_l1[idx] += float(np.sum(np.abs(log_b - log_a)))
                stft_bins[idx] += int(mag_a.size)
                if n_fft == _MEL_N_FFT and hop == _MEL_HOP:
                    mel_a = 10.0 * np.log10(np.maximum((mag_a * mag_a) @ mel_fb.T, 1e-10))
                    mel_b = 10.0 * np.log10(np.maximum((mag_b * mag_b) @ mel_fb.T, 1e-10))
                    diff_mel = mel_b - mel_a
                    mel_sq_db += float(np.sum(diff_mel * diff_mel))
                    mel_bins += int(diff_mel.size)

    with _paired_audio_readers(source, info, output, decode_info, chunk_frames=_PERCEPTUAL_BLOCK_FRAMES) as (
        left,
        right,
    ):
        for before, after in zip(left, right, strict=False):
            pair_len = min(len(before), len(after))
            if pair_len <= 0:
                continue
            buf_a.append(before[:pair_len])
            buf_b.append(after[:pair_len])
            buffered += pair_len
            while buffered >= _PERCEPTUAL_BLOCK_FRAMES:
                cat_a = buf_a[0] if len(buf_a) == 1 else np.concatenate(buf_a, axis=0)
                cat_b = buf_b[0] if len(buf_b) == 1 else np.concatenate(buf_b, axis=0)
                _process_block(cat_a[:_PERCEPTUAL_BLOCK_FRAMES], cat_b[:_PERCEPTUAL_BLOCK_FRAMES])
                rem_a = cat_a[_PERCEPTUAL_BLOCK_FRAMES:].copy()
                rem_b = cat_b[_PERCEPTUAL_BLOCK_FRAMES:].copy()
                buf_a = [rem_a] if len(rem_a) else []
                buf_b = [rem_b] if len(rem_b) else []
                buffered = len(rem_a)
        if buffered > 0:
            final_a = buf_a[0] if len(buf_a) == 1 else np.concatenate(buf_a, axis=0)
            final_b = buf_b[0] if len(buf_b) == 1 else np.concatenate(buf_b, axis=0)
            _process_block(final_a, final_b)

    scs = [sqrt(d) / max(sqrt(s), 1e-7) if s > 1e-12 else 0.0 for d, s in zip(diff_sq, src_sq, strict=True)]
    lms = [ll / max(nb, 1) for ll, nb in zip(log_l1, stft_bins, strict=True)]
    mean_sc = sum(scs) / len(scs)
    mean_lm = sum(lms) / len(lms)
    mel_db = sqrt(mel_sq_db / max(mel_bins, 1))
    return AudioPerceptualMetrics(
        mr_stft_loss=mean_sc + mean_lm,
        mr_stft_sc=mean_sc,
        mr_stft_log_mag=mean_lm,
        mel_distance_db=mel_db,
    )
