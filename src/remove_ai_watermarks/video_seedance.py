"""Experimental fixed Seedance video reconstruction, without a runtime oracle.

This module implements the label-free forest recipe recorded in
docs/seedance-video-study.md. Visible overlays must be handled before this stage.
"""

from __future__ import annotations

# Reuse the existing video engines' package-private codec and latent helpers.
# pyright: reportPrivateUsage=false
import json
import logging
import math
import subprocess
from collections import deque
from contextlib import suppress
from dataclasses import replace
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from remove_ai_watermarks._internal.constants import ISOBMFF_VIDEO_EXTENSIONS
from remove_ai_watermarks.video_decode import open_video_capture, open_video_capture_16bit
from remove_ai_watermarks.video_encoding import (
    HIGH_BIT_DEPTH_PIXEL_FORMATS,
    _profile_args,
    _run_ffprobe,
    _video_codec_args,
    abort_raw_video_encoder,
    atomic_video_output,
    finish_raw_video_encoder,
    mux_encoded_video,
    probe_video_encode_profile,
    probe_video_timestamps,
    raw_video_command,
    staged_video_output,
    start_raw_video_encoder,
    video_end_repair,
)
from remove_ai_watermarks.video_invisible import (
    RegenerationMetrics,
    _decode_frame_latents,
    _encode_frame_latents,
    _fit_size,
    _stream_batches,
    is_available,
    load_video_vae_runtime,
    probe_video_geometry,
)
from remove_ai_watermarks.video_temporal import (
    _backward_map,
    _motion_residual,
    anchor_low_freq_color,
    iter_stabilized_regenerated_frames,
)
from remove_ai_watermarks.video_tools import require_video_tool
from remove_ai_watermarks.video_visible import _HDR_TRANSFERS, _timestamp_time_base, _TimestampedNutWriter

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

    from remove_ai_watermarks.video_invisible import VideoVaeRuntime

log = logging.getLogger(__name__)

SEEDANCE_EXPERIMENTAL_PROFILE = "seedance-texture-v1"
SeedanceQuality = Literal["delivery", "master"]


def _frames(path: Path, *, high_depth: bool = False, width: int = 0, height: int = 0) -> Iterator[NDArray[Any]]:
    if high_depth:
        if width <= 0 or height <= 0:
            width, height, _ = probe_video_geometry(path)
        capture: Any = open_video_capture_16bit(path, width=width, height=height)
    else:
        capture = open_video_capture(path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not decode video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def _restore_texture(source: NDArray[Any], guide: NDArray[Any]) -> NDArray[Any]:
    """Apply the fixed sigma-1 RGB split, preserving native 8-bit or 16-bit depth."""
    height, width = source.shape[:2]
    high_depth = source.dtype == np.uint16
    original = source.astype(np.float64)
    reconstructed = cv2.resize(guide, (width, height), interpolation=cv2.INTER_LANCZOS4).astype(np.float64)
    if high_depth and guide.dtype != np.uint16:
        reconstructed *= 257.0
    low_original = cv2.GaussianBlur(original, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
    low_guide = cv2.GaussianBlur(reconstructed, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
    if high_depth:
        return np.clip(np.rint(low_guide + original - low_original), 0, 65535).astype(np.uint16)
    return np.clip(np.rint(low_guide + original - low_original), 0, 255).astype(np.uint8)


def _clock(path: Path, *, include_packets: bool = False) -> dict[str, Any]:
    entries = "stream=time_base,start_pts,duration_ts,duration"
    if include_packets:
        entries += ":packet=pts,dts,duration"
    raw = _run_ffprobe(
        path,
        (
            "-select_streams",
            "v:0",
            "-show_entries",
            entries,
        ),
        purpose="experimental Seedance video clock",
        output_format="json",
    )
    if raw is None:
        raise RuntimeError(f"Could not read video clock: {path}")
    return json.loads(raw)


def _match_video_end(encoded: Path, source: Path) -> None:
    original, candidate = _clock(source), _clock(encoded)
    original_stream, candidate_stream = original["streams"][0], candidate["streams"][0]
    target = float(original_stream["duration"])
    if abs(float(candidate_stream["duration"]) - target) <= 0.000002:
        return
    repair = video_end_repair(original, _clock(encoded, include_packets=True))
    ffmpeg = require_video_tool("ffmpeg")
    with TemporaryDirectory(prefix="seedance-clock-", dir=encoded.parent) as directory:
        suffix = encoded.suffix.lower()
        retimed = Path(directory) / f"retimed{suffix}"
        from remove_ai_watermarks.video_encoding import container_format_args

        _run_command(
            [
                ffmpeg,
                "-v",
                "error",
                "-copyts",
                "-i",
                str(encoded),
                "-map",
                "0",
                "-c",
                "copy",
                "-bsf:v",
                repair.bitstream_filter,
                "-movie_timescale",
                str(repair.movie_timescale),
                "-avoid_negative_ts",
                "disabled",
                "-movflags",
                "+faststart",
                *container_format_args(suffix),
                str(retimed),
            ]
        )
        if abs(float(_clock(retimed)["streams"][0]["duration"]) - target) > 0.000002:
            raise RuntimeError("Could not preserve terminal video duration")
        retimed.replace(encoded)


def _run_command(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    log.info(
        "Seedance command=%s status=%s stdout=%s stderr=%s", command, result.returncode, result.stdout, result.stderr
    )
    if result.returncode:
        raise RuntimeError(f"Experimental Seedance encoding failed: {result.stderr}")


def _encode_frames(
    frames: Iterable[NDArray[Any]],
    source: Path,
    output: Path,
    fps: float,
    timestamps: tuple[float, ...],
    *,
    crf: int,
) -> None:
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    high_depth = first.dtype == np.uint16
    raw_pix_fmt = "bgr48le" if high_depth else "bgr24"
    profile = probe_video_encode_profile(source)
    time_base = _timestamp_time_base(profile.time_base)
    target_pix_fmt = (
        profile.pixel_format if (high_depth and profile.pixel_format in HIGH_BIT_DEPTH_PIXEL_FORMATS) else "yuv420p"
    )
    profile = replace(profile, time_base=str(time_base), pixel_format=target_pix_fmt)
    with staged_video_output(output) as (encoded, temporary):
        encoder = start_raw_video_encoder(
            raw_video_command(
                encoded,
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                profile=profile,
                input_pix_fmt=raw_pix_fmt,
                timestamped_input=True,
                copy_input_timestamps=True,
            )
        )
        writer = None
        try:
            writer = _TimestampedNutWriter(
                encoder.stdin,
                width=width,
                height=height,
                time_base=time_base,
                start_pts=round(timestamps[0] / float(time_base)),
                pix_fmt=raw_pix_fmt,
            )
            for frame, timestamp in zip(chain((first,), iterator), timestamps, strict=True):
                writer.write(frame, timestamp)
            writer.close()
            writer = None
            finish_raw_video_encoder(encoder, encoded, operation="Seedance encode")
            is_isobmff = output.suffix.lower() in ISOBMFF_VIDEO_EXTENSIONS
            if is_isobmff:
                _match_video_end(encoded, source)
            mux_encoded_video(encoded, source, temporary, strip_metadata=True, copy_input_timestamps=True)
            if is_isobmff:
                _match_video_end(temporary, source)
        except Exception:
            if writer is not None:
                with suppress(Exception):
                    writer.close()
            abort_raw_video_encoder(encoder)
            raise
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


def _reconstruct(
    source: Path, runtime: VideoVaeRuntime, size: tuple[int, int], *, high_depth: bool = False
) -> Iterator[NDArray[Any]]:
    import torch

    resized = (
        cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4) for frame in _frames(source, high_depth=high_depth)
    )
    for batch in _stream_batches(resized, 4):
        latents = _encode_frame_latents(
            batch, vae=runtime.vae, device=runtime.resolved_device, batch_size=4, scaling_factor=runtime.scaling_factor
        )
        yield from _decode_frame_latents(
            latents,
            vae=runtime.vae,
            noise_std=0.0,
            shared_noise=torch.zeros_like(latents[0]),
            scaling_factor=runtime.scaling_factor,
            high_depth=high_depth,
        )


def _encode_delivery(master: Path, source: Path, output: Path) -> None:
    ffmpeg = require_video_tool("ffmpeg")
    profile = probe_video_encode_profile(source)
    high_depth = profile.pixel_format in HIGH_BIT_DEPTH_PIXEL_FORMATS
    time_base = _timestamp_time_base(profile.time_base)
    target_pix_fmt = profile.pixel_format if high_depth else "yuv420p"
    target_profile = replace(profile, time_base=str(time_base), pixel_format=target_pix_fmt)
    codec_args = _video_codec_args(
        output.suffix.lower(), crf=18, profile=target_profile, encoder="software", ffmpeg_path=ffmpeg
    )
    if not high_depth and output.suffix.lower() != ".webm":
        codec_args.extend(["-profile:v", "high"])
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-copyts",
        "-i",
        str(master),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        *codec_args,
        *_profile_args(target_profile),
        "-c:a",
        "copy",
        "-fps_mode",
        "passthrough",
    ]
    is_isobmff = output.suffix.lower() in ISOBMFF_VIDEO_EXTENSIONS
    if is_isobmff and time_base.numerator == 1:
        command.extend(["-video_track_timescale", str(time_base.denominator)])
    from remove_ai_watermarks.video_encoding import container_format_args

    command.extend(
        [
            "-avoid_negative_ts",
            "disabled",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            *(["-movflags", "+faststart"] if is_isobmff else []),
            *container_format_args(output.suffix),
            str(output),
        ]
    )
    _run_command(command)
    if is_isobmff:
        _match_video_end(output, source)


def _measure(source: Path, output: Path, fps: float, count: int, *, high_depth: bool = False) -> RegenerationMetrics:
    squared_error = 0.0
    pixels = 0
    baseline = 0.0
    residual = 0.0
    previous = None
    previous_output = None
    frames = 0
    width, height, _ = probe_video_geometry(source)
    src_iter = _frames(source, high_depth=high_depth, width=width, height=height)
    out_iter = _frames(output, high_depth=high_depth, width=width, height=height)
    try:
        for original, delivered in zip(src_iter, out_iter, strict=True):
            if original.shape != delivered.shape:
                raise RuntimeError("Experimental Seedance output geometry changed")
            difference = original.astype(np.float64) - delivered.astype(np.float64)
            squared_error += float(np.sum(difference * difference))
            pixels += original.size
            orig_u8 = (original >> 8).astype(np.uint8) if original.dtype == np.uint16 else original
            deliv_u8 = (delivered >> 8).astype(np.uint8) if delivered.dtype == np.uint16 else delivered
            current = cv2.cvtColor(orig_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            current_output = cv2.cvtColor(deliv_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if previous is not None and previous_output is not None:
                maps = _backward_map(current, previous)
                baseline += _motion_residual(current, previous, maps)
                residual += _motion_residual(current_output, previous_output, maps)
            previous, previous_output = current, current_output
            frames += 1
    finally:
        for it in (src_iter, out_iter):
            close = getattr(it, "close", None)
            if callable(close):
                close()
    if frames != count:
        raise RuntimeError("Experimental Seedance decoded frame count changed")
    mse = squared_error / pixels
    peak = 65535.0 if high_depth else 255.0
    return RegenerationMetrics(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        psnr_db=10 * math.log10(peak**2 / mse) if mse > 0 else math.inf,
        temporal_residual_ratio=_temporal_ratio(residual, baseline),
    )


def _temporal_ratio(residual: float, baseline: float) -> float:
    return residual / baseline if baseline > 1e-12 else (1.0 if residual <= 1e-12 else math.inf)


def validate_experimental_mp4_source(
    source: Path,
    *,
    label: str,
    encode_profile: Any = None,
    geometry: tuple[int, int, float] | None = None,
    timestamps: tuple[float, ...] | None = None,
    allow_high_depth: bool = False,
) -> tuple[int, int, float, tuple[float, ...]]:
    """Validate 8-bit SDR (or 10-bit HDR when allowed), even-dimension, timestamped video input."""
    active_profile = encode_profile if encode_profile is not None else probe_video_encode_profile(source)
    high_depth = allow_high_depth and active_profile.pixel_format in HIGH_BIT_DEPTH_PIXEL_FORMATS
    if (active_profile.component_depth != 8 or active_profile.color_transfer in _HDR_TRANSFERS) and not high_depth:
        raise ValueError(f"{label} requires SDR 8-bit video")
    width, height, fps = geometry if geometry is not None else probe_video_geometry(source)
    if width % 2 or height % 2:
        raise ValueError(f"{label} requires even video dimensions")
    active_timestamps = timestamps if timestamps is not None else probe_video_timestamps(source)
    if not active_timestamps:
        raise ValueError(f"{label} requires source frame timestamps")
    return width, height, fps, active_timestamps


def _verify_isobmff_timing(source: Path, staged: Path, timestamps: tuple[float, ...], label: str) -> None:
    """Verify exact timestamps and terminal duration for ISOBMFF containers."""
    if staged.suffix.lower() not in ISOBMFF_VIDEO_EXTENSIONS:
        return
    if probe_video_timestamps(staged) != timestamps:
        raise RuntimeError(f"{label} output timestamps changed")
    staged_dur = float(_clock(staged)["streams"][0]["duration"])
    source_dur = float(_clock(source)["streams"][0]["duration"])
    if abs(staged_dur - source_dur) > 0.000002:
        raise RuntimeError(f"{label} output duration changed")


def _finalize_experimental_output(
    frames: Iterable[NDArray[Any]],
    source: Path,
    master: Path,
    staged: Path,
    *,
    quality: SeedanceQuality,
    fps: float,
    timestamps: tuple[float, ...],
    label: str,
    high_depth: bool = False,
) -> RegenerationMetrics:
    """Encode lossless master, write delivery/master output, verify clock, and score."""
    _encode_frames(frames, source, master, fps, timestamps, crf=0)
    if quality == "delivery":
        _encode_delivery(master, source, staged)
    else:
        master.replace(staged)
    _verify_isobmff_timing(source, staged, timestamps, label)
    return _measure(source, staged, fps, len(timestamps), high_depth=high_depth)


def regenerate_seedance_experimental(
    source: Path,
    output: Path,
    *,
    device: str = "auto",
    quality: SeedanceQuality = "delivery",
    runtime: VideoVaeRuntime | None = None,
) -> RegenerationMetrics:
    """Run the Seedance texture-transfer recipe (8-bit SDR and 10-bit HEVC/HDR) and validate output."""
    if quality not in {"delivery", "master"}:
        raise ValueError("quality must be delivery or master")
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    if runtime is not None and (runtime.model != "stabilityai/sd-vae-ft-mse" or runtime.requested_device != device):
        raise ValueError("The supplied video VAE runtime does not match the Seedance model and device")
    encode_profile = probe_video_encode_profile(source)
    high_depth = encode_profile.pixel_format in HIGH_BIT_DEPTH_PIXEL_FORMATS
    width, height, fps, timestamps = validate_experimental_mp4_source(
        source,
        label="Seedance processing",
        encode_profile=encode_profile,
        allow_high_depth=True,
    )
    if not is_available():
        raise RuntimeError("Seedance processing requires 'remove-ai-watermarks[video,diffusion]'")
    log.info(
        "Applying %s (%s, %s); no per-file CapCut verification is performed",
        SEEDANCE_EXPERIMENTAL_PROFILE,
        quality,
        encode_profile.pixel_format,
    )
    with (
        atomic_video_output(output) as staged,
        TemporaryDirectory(prefix="seedance-experimental-", dir=output.parent) as directory,
    ):
        master = Path(directory) / f"master{source.suffix.lower()}"
        owned_runtime = runtime is None
        active_runtime = runtime or load_video_vae_runtime(model="stabilityai/sd-vae-ft-mse", device=device)
        guide_size = _fit_size(width, height, 512)
        guide_stream = (
            _reconstruct(source, active_runtime, guide_size, high_depth=True)
            if high_depth
            else _reconstruct(source, active_runtime, guide_size)
        )
        src_frames = _frames(source, high_depth=high_depth, width=width, height=height)
        src_window: deque[NDArray[Any]] = deque()

        def _guide_pairs() -> Iterator[tuple[NDArray[Any], NDArray[Any]]]:
            for src, raw_guide in zip(src_frames, guide_stream, strict=True):
                src_window.append(src)
                g_h, g_w = raw_guide.shape[:2]
                src_small = (
                    src if src.shape[:2] == (g_h, g_w) else cv2.resize(src, (g_w, g_h), interpolation=cv2.INTER_AREA)
                )
                if high_depth and raw_guide.dtype != np.uint16:
                    raw_guide = raw_guide.astype(np.uint16) * 257
                yield src_small, raw_guide

        def _restored_frames() -> Iterator[NDArray[Any]]:
            for _src_small, stab_guide in iter_stabilized_regenerated_frames(_guide_pairs()):
                src = src_window.popleft()
                up_guide = (
                    stab_guide
                    if stab_guide.shape[:2] == (height, width)
                    else cv2.resize(stab_guide, (width, height), interpolation=cv2.INTER_LANCZOS4)
                )
                yield _restore_texture(src, anchor_low_freq_color(src, up_guide))

        frames = _restored_frames()
        try:
            return _finalize_experimental_output(
                frames,
                source,
                master,
                staged,
                quality=quality,
                fps=fps,
                timestamps=timestamps,
                label="Seedance",
                high_depth=high_depth,
            )
        finally:
            if owned_runtime:
                del active_runtime
