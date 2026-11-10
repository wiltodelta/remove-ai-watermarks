"""End-to-end fidelity of a DELIVERED video against its source.

The engine's own ``psnr_db`` is measured against the already-resized frame and
before the encoder, so it reports the VAE round trip plus latent noise and
nothing else. The downscale, the frame decimation and the H.264 encode -- the
three steps that actually cost the user picture -- are invisible to it, and no
metric computed inside the streaming loop can see them, because the candidate
frame is captured before it is written to the encoder pipe.

This script measures what the engine cannot: it decodes the delivered file after
muxing, upscales each frame back to the source geometry, and scores it against
the untouched source frame it came from. That reference is deliberately harsh -
detail the downscale destroyed is unrecoverable, so the absolute number is
dominated by content the pipeline never had a chance to keep. Use it to RANK
configurations against a shared reference, not to attribute loss to one stage.

It streams, for the same reason the engine does: holding a 1080p clip plus its
upscaled candidate and flow maps in memory runs to gigabytes and grows with clip
length. Peak here is a handful of frames regardless of duration.

It is a research tool, not part of the shipped path, and it measures fidelity
only. Nothing here is a watermark verdict: only a provider-oracle row is.

    uv run python scripts/video_fidelity_probe.py source.mp4 out-512.mp4 out-768.mp4
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import cv2
import numpy as np

from remove_ai_watermarks.video_encoding import _run_ffprobe, probe_video_timestamps
from remove_ai_watermarks.video_invisible import _iter_sampled_frames, _probe_video
from remove_ai_watermarks.video_temporal import _backward_map, _motion_residual

sys.path.insert(0, str(Path(__file__).parent))

from invisible_quality_audit import _ssim  # reuse, do not reimplement a fourth SSIM

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

log = logging.getLogger(__name__)


def _delivered_geometry(path: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video: {path}")
    try:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or fps <= 0.0:
        raise click.ClickException(f"Video has no usable geometry or frame rate: {path}")
    return width, height, fps


def _iter_frames(path: Path) -> Iterator[NDArray[Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise click.ClickException(f"Could not open video: {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def _video_duration(path: Path) -> float:
    value = _run_ffprobe(
        path,
        ("-select_streams", "v:0", "-show_entries", "stream=duration"),
        purpose="video duration",
        output_format="csv=p=0",
    )
    if value is None:
        raise click.ClickException(f"Could not read video duration: {path}")
    duration = float(value.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise click.ClickException(f"Invalid video duration: {path}")
    return duration


def _measure(
    source: Path,
    delivered: Path,
    *,
    source_geometry: tuple[int, int, float],
    duration: float | None,
    preserve_timestamps: bool = False,
) -> dict[str, Any]:
    source_width, source_height, source_fps = source_geometry
    width, height, fps = _delivered_geometry(delivered)

    # Drive the source through the engine's own sampler at the source's geometry,
    # so the frame the probe compares against is the frame the engine regenerated.
    # Importing the rule is what keeps the pairing correct: a count check cannot
    # catch a selection rule that reorders frames without changing how many.
    timing_error = None
    duration_error = None
    if preserve_timestamps:
        if duration is not None:
            raise click.ClickException("Timestamp fidelity requires the complete source")
        source_pts, delivered_pts = probe_video_timestamps(source), probe_video_timestamps(delivered)
        if not source_pts or len(source_pts) != len(delivered_pts):
            raise click.ClickException("Source and delivered timestamps have different frame counts")
        timing_error = max(abs(first - second) for first, second in zip(source_pts, delivered_pts, strict=True))
        if timing_error > 0.000002:
            raise click.ClickException(f"Delivered frame timestamps changed by {timing_error:.6f} seconds")
        duration_error = abs(_video_duration(source) - _video_duration(delivered))
        if duration_error > 0.000002:
            raise click.ClickException(f"Delivered video duration changed by {duration_error:.6f} seconds")
        reference_frames = _iter_frames(source)
    else:
        reference_frames = _iter_sampled_frames(
            source, source_fps=source_fps, duration=duration, effective_fps=fps, size=(source_width, source_height)
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

    reference_iter = iter(reference_frames)
    delivered_iter = _iter_frames(delivered)
    while True:
        reference = next(reference_iter, None)
        delivered_frame = next(delivered_iter, None)
        if reference is None and delivered_frame is None:
            break
        if reference is None or delivered_frame is None:
            raise click.ClickException(
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
            frame_maps = _backward_map(current_gray, previous_gray)
            temporal_baseline += _motion_residual(reference_f32, previous_reference_f32, frame_maps)
            temporal_candidate += _motion_residual(candidate_f32, previous_candidate_f32, frame_maps)
        previous_gray = current_gray
        previous_reference_f32 = reference_f32
        previous_candidate_f32 = candidate_f32
        frame_count += 1

    if frame_count < 2:
        raise click.ClickException(f"{delivered.name} paired fewer than two frames against {source.name}")

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


@click.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("delivered", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--duration",
    type=click.FloatRange(min=0.1),
    default=None,
    help="Score only the first N seconds of the source, matching a trimmed sweep candidate.",
)
@click.option("--preserve-timestamps", is_flag=True, help="Require matching full-clip PTS and video duration.")
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), help="Also write the rows as JSON.")
def main(
    source: Path, delivered: tuple[Path, ...], duration: float | None, json_out: Path | None, preserve_timestamps: bool
) -> None:
    """Score each DELIVERED file against SOURCE end to end."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source_geometry = _probe_video(source)
    log.info("Source %s: %dx%d at %.4f fps", source.name, *source_geometry)

    rows = [
        _measure(
            source, path, source_geometry=source_geometry, duration=duration, preserve_timestamps=preserve_timestamps
        )
        for path in delivered
    ]
    for row in rows:
        log.info(
            "%s: %dx%d at %s fps, %.1f%% of source pixels, PSNR %s dB, SSIM %s, temporal %s, %s kbps",
            row["file"],
            row["width"],
            row["height"],
            row["fps"],
            row["pixel_ratio"] * 100.0,
            row["source_psnr_db"],
            row["source_ssim"],
            row["temporal_residual_ratio"],
            row["file_bitrate_kbps"],
        )
    if json_out is not None:
        json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        log.info("Wrote %s", json_out)


if __name__ == "__main__":
    main()
