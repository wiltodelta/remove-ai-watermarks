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
from contextlib import suppress
from dataclasses import replace
from itertools import chain
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from remove_ai_watermarks.video_encoding import (
    _run_ffprobe,
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
    _probe_video,
    _stream_batches,
    is_available,
    load_video_vae_runtime,
)
from remove_ai_watermarks.video_temporal import _backward_map, _motion_residual
from remove_ai_watermarks.video_tools import require_video_tool
from remove_ai_watermarks.video_visible import _timestamp_time_base, _TimestampedNutWriter

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

    from remove_ai_watermarks.video_invisible import VideoVaeRuntime

log = logging.getLogger(__name__)

SEEDANCE_EXPERIMENTAL_PROFILE = "seedance-texture-v1"
SeedanceQuality = Literal["delivery", "master"]


def _frames(path: Path) -> Iterator[NDArray[Any]]:
    capture = cv2.VideoCapture(str(path))
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
    """Apply the fixed sigma-1 RGB split, with ties-to-even quantization."""
    height, width = source.shape[:2]
    original = source.astype(np.float64)
    reconstructed = cv2.resize(guide, (width, height), interpolation=cv2.INTER_LANCZOS4).astype(np.float64)
    low_original = cv2.GaussianBlur(original, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
    low_guide = cv2.GaussianBlur(reconstructed, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT_101)
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
        retimed = Path(directory) / "retimed.mp4"
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
    profile = probe_video_encode_profile(source)
    time_base = _timestamp_time_base(profile.time_base)
    profile = replace(profile, time_base=str(time_base), pixel_format="yuv420p")
    with staged_video_output(output) as (encoded, temporary):
        encoder = start_raw_video_encoder(
            raw_video_command(
                encoded,
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                profile=profile,
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
            )
            for frame, timestamp in zip(chain((first,), iterator), timestamps, strict=True):
                writer.write(frame, timestamp)
            writer.close()
            writer = None
            finish_raw_video_encoder(encoder, encoded, operation="Experimental Seedance encode")
            _match_video_end(encoded, source)
            mux_encoded_video(encoded, source, temporary, strip_metadata=True, copy_input_timestamps=True)
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


def _reconstruct(source: Path, runtime: VideoVaeRuntime, size: tuple[int, int]) -> Iterator[NDArray[Any]]:
    import torch

    resized = (cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4) for frame in _frames(source))
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
        )


def _encode_delivery(master: Path, source: Path, output: Path) -> None:
    ffmpeg = require_video_tool("ffmpeg")
    profile = probe_video_encode_profile(source)
    time_base = _timestamp_time_base(profile.time_base)
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
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-enc_time_base:v",
        str(time_base),
        "-fps_mode",
        "passthrough",
    ]
    if time_base.numerator == 1:
        command.extend(["-video_track_timescale", str(time_base.denominator)])
    command.extend(
        [
            "-avoid_negative_ts",
            "disabled",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    _run_command(command)
    _match_video_end(output, source)


def _measure(source: Path, output: Path, fps: float, count: int) -> RegenerationMetrics:
    squared_error = 0.0
    pixels = 0
    baseline = 0.0
    residual = 0.0
    previous = None
    previous_output = None
    frames = 0
    width, height, _ = _probe_video(source)
    for original, delivered in zip(_frames(source), _frames(output), strict=True):
        if original.shape != delivered.shape:
            raise RuntimeError("Experimental Seedance output geometry changed")
        difference = original.astype(np.float64) - delivered.astype(np.float64)
        squared_error += float(np.sum(difference * difference))
        pixels += original.size
        current = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY).astype(np.float32)
        current_output = cv2.cvtColor(delivered, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None and previous_output is not None:
            maps = _backward_map(current, previous)
            baseline += _motion_residual(current, previous, maps)
            residual += _motion_residual(current_output, previous_output, maps)
        previous, previous_output = current, current_output
        frames += 1
    if frames != count:
        raise RuntimeError("Experimental Seedance decoded frame count changed")
    mse = squared_error / pixels
    return RegenerationMetrics(
        frames=frames,
        fps=fps,
        width=width,
        height=height,
        psnr_db=10 * math.log10(255**2 / mse) if mse > 0 else math.inf,
        temporal_residual_ratio=_temporal_ratio(residual, baseline),
    )


def _temporal_ratio(residual: float, baseline: float) -> float:
    return residual / baseline if baseline > 1e-12 else (1.0 if residual <= 1e-12 else math.inf)


def regenerate_seedance_experimental(
    source: Path,
    output: Path,
    *,
    device: str = "auto",
    quality: SeedanceQuality = "delivery",
    runtime: VideoVaeRuntime | None = None,
) -> RegenerationMetrics:
    """Run the fixed experimental recipe and validate its complete saved output."""
    if quality not in {"delivery", "master"}:
        raise ValueError("quality must be delivery or master")
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    if runtime is not None and (runtime.model != "stabilityai/sd-vae-ft-mse" or runtime.requested_device != device):
        raise ValueError("The supplied video VAE runtime does not match the Seedance model and device")
    profile = probe_video_encode_profile(source)
    if (profile.component_depth or 0) > 8 or profile.color_transfer in {"smpte2084", "arib-std-b67"}:
        raise ValueError("Experimental Seedance processing requires SDR 8-bit video")
    width, height, fps = _probe_video(source)
    if width % 2 or height % 2:
        raise ValueError("Experimental Seedance processing requires even video dimensions")
    timestamps = probe_video_timestamps(source)
    if not timestamps:
        raise ValueError("Experimental Seedance processing requires source frame timestamps")
    if not is_available():
        raise RuntimeError("Experimental Seedance processing requires 'remove-ai-watermarks[video,diffusion]'")
    log.warning(
        "Applying experimental %s (%s); no per-file CapCut verification is performed",
        SEEDANCE_EXPERIMENTAL_PROFILE,
        quality,
    )
    with (
        atomic_video_output(output) as staged,
        TemporaryDirectory(prefix="seedance-experimental-", dir=output.parent) as directory,
    ):
        guide, master = Path(directory) / "guide.mp4", Path(directory) / "master.mp4"
        runtime = runtime or load_video_vae_runtime(model="stabilityai/sd-vae-ft-mse", device=device)
        _encode_frames(
            _reconstruct(source, runtime, _fit_size(width, height, 512)), source, guide, fps, timestamps, crf=18
        )
        del runtime
        frames = (_restore_texture(a, b) for a, b in zip(_frames(source), _frames(guide), strict=True))
        _encode_frames(frames, source, master, fps, timestamps, crf=0)
        if quality == "delivery":
            _encode_delivery(master, source, staged)
        else:
            master.replace(staged)
        if probe_video_timestamps(staged) != timestamps:
            raise RuntimeError("Experimental Seedance output timestamps changed")
        if (
            abs(float(_clock(staged)["streams"][0]["duration"]) - float(_clock(source)["streams"][0]["duration"]))
            > 0.000002
        ):
            raise RuntimeError("Experimental Seedance output duration changed")
        return _measure(source, staged, fps, len(timestamps))
