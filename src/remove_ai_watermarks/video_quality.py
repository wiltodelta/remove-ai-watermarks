"""Delivered-video fidelity, including resize, temporal sampling, and encoding.

Measurements share the producer's sampling rule. They do not verify a watermark.
"""

from __future__ import annotations

import math
from contextlib import closing
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks._internal.image_quality import structural_similarity_gray as _ssim

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from numpy.typing import NDArray


def delivered_geometry(path: Path) -> tuple[int, int, float]:
    import cv2

    from remove_ai_watermarks.video_decode import open_video_capture

    capture = open_video_capture(path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0:
        raise ValueError(f"Video has no usable geometry or frame rate: {path}")
    return width, height, fps


def _iter_frames(path: Path) -> Generator[NDArray[Any], None, None]:
    from remove_ai_watermarks.video_decode import open_video_capture

    capture = open_video_capture(path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def measure_video_fidelity(
    source: Path,
    delivered: Path,
    *,
    source_geometry: tuple[int, int, float],
    duration: float | None,
    preserve_timestamps: bool = False,
) -> dict[str, Any]:
    import cv2
    import numpy as np

    from remove_ai_watermarks.video_invisible import iter_sampled_frames
    from remove_ai_watermarks.video_temporal import backward_motion_map, motion_residual_energy

    source_width, source_height, source_fps = source_geometry
    width, height, fps = delivered_geometry(delivered)

    # Drive the source through the engine's own sampler at the source's geometry,
    # so the frame the probe compares against is the frame the engine regenerated.
    # Importing the rule is what keeps the pairing correct: a count check cannot
    # catch a selection rule that reorders frames without changing how many.
    timing_error = None
    duration_error = None
    if preserve_timestamps:
        from remove_ai_watermarks.video_encoding import probe_video_duration, probe_video_timestamps

        if duration is not None:
            raise ValueError("Timestamp fidelity requires the complete source")
        source_pts, delivered_pts = probe_video_timestamps(source), probe_video_timestamps(delivered)
        if not source_pts or len(source_pts) != len(delivered_pts):
            raise ValueError("Source and delivered timestamps have different frame counts")
        timing_error = max(abs(first - second) for first, second in zip(source_pts, delivered_pts, strict=True))
        if timing_error > 0.000002:
            raise ValueError(f"Delivered frame timestamps changed by {timing_error:.6f} seconds")
        duration_error = abs(probe_video_duration(source) - probe_video_duration(delivered))
        if duration_error > 0.000002:
            raise ValueError(f"Delivered video duration changed by {duration_error:.6f} seconds")
        reference_frames = _iter_frames(source)
    else:
        reference_frames = iter_sampled_frames(
            source,
            source_fps=source_fps,
            duration=duration,
            effective_fps=fps,
            size=(source_width, source_height),
        )

    squared_error = 0.0
    pixel_count = 0
    ssim_scores: list[float] = []
    temporal_baseline = 0.0
    temporal_candidate = 0.0
    previous_gray: NDArray[Any] | None = None
    previous_reference_f32: NDArray[Any] | None = None
    previous_candidate_f32: NDArray[Any] | None = None
    frame_count = 0
    needs_upscale = (width, height) != (source_width, source_height)

    with closing(reference_frames) as reference_iter, closing(_iter_frames(delivered)) as delivered_iter:
        while True:
            reference = next(reference_iter, None)
            delivered_frame = next(delivered_iter, None)
            if reference is None and delivered_frame is None:
                break
            if reference is None or delivered_frame is None:
                raise ValueError(
                    f"{delivered.name} and the sampled source ran out at different points after "
                    f"{frame_count} frames. Pass --duration to match the prefix this output was "
                    f"produced from, or check that it came from {source.name}."
                )
            candidate = (
                cv2.resize(delivered_frame, (source_width, source_height), interpolation=cv2.INTER_LANCZOS4)
                if needs_upscale
                else delivered_frame
            )
            reference_f32 = reference.astype(np.float32)
            candidate_f32 = candidate.astype(np.float32)
            difference = reference_f32 - candidate_f32
            squared_error += float(np.sum(difference * difference, dtype=np.float64))
            pixel_count += reference.size
            ssim_scores.append(
                _ssim(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY))
            )

            current_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            if previous_gray is not None and previous_reference_f32 is not None and previous_candidate_f32 is not None:
                frame_maps = backward_motion_map(current_gray, previous_gray)
                temporal_baseline += motion_residual_energy(reference_f32, previous_reference_f32, frame_maps)
                temporal_candidate += motion_residual_energy(candidate_f32, previous_candidate_f32, frame_maps)
            previous_gray = current_gray
            previous_reference_f32 = reference_f32
            previous_candidate_f32 = candidate_f32
            frame_count += 1

    if frame_count < 2:
        raise ValueError(f"{delivered.name} paired fewer than two frames against {source.name}")

    mse = squared_error / pixel_count
    size_bytes = delivered.stat().st_size
    return {
        "file": delivered.name,
        "source": source.name,
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": round(source_fps, 4),
        "width": width,
        "height": height,
        "fps": round(fps, 4),
        "frames": frame_count,
        "max_timestamp_error_seconds": timing_error,
        "video_duration_error_seconds": duration_error,
        "pixel_ratio": round((width * height) / (source_width * source_height), 4),
        "source_psnr_db": math.inf if mse == 0.0 else round(20.0 * math.log10(255.0 / math.sqrt(mse)), 4),
        "source_ssim": round(float(np.mean(ssim_scores)), 4),
        "temporal_residual_ratio": round(temporal_candidate / max(temporal_baseline, 1e-6), 4),
        "size_bytes": size_bytes,
        # The mux copies the source audio track verbatim, so this is a container
        # bitrate. It still ranks candidates at fixed crf; it is not a video bitrate.
        "file_bitrate_kbps": round(size_bytes * 8.0 / (frame_count / fps) / 1000.0, 1),
    }
