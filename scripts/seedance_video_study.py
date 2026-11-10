"""Local, hash-bound Seedance experiments; no uploads or removal certification.

Compare a metadata-only control, matched re-encoding, motion-aligned temporal
averaging, and the existing VAE engine. See docs/seedance-video-study.md.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from dataclasses import replace
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import click  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from capcut_receipts import receipt_verdict  # noqa: E402
from video_fidelity_probe import _iter_frames, _measure  # noqa: E402
from watermark_benchmark import sha256_file as sha256  # noqa: E402

from remove_ai_watermarks.video import remove_video_metadata  # noqa: E402
from remove_ai_watermarks.video_encoding import (  # noqa: E402
    _run_ffprobe,
    abort_raw_video_encoder,
    finish_raw_video_encoder,
    mux_encoded_video,
    probe_video_encode_profile,
    probe_video_timestamps,
    raw_video_command,
    staged_video_output,
    start_raw_video_encoder,
    video_end_repair,
)
from remove_ai_watermarks.video_invisible import (  # noqa: E402
    _decode_frame_latents,
    _encode_frame_latents,
    _fit_size,
    _iter_sampled_frames,
    _probe_video,
    _shared_latent_noise,
    _stream_batches,
    load_video_vae_runtime,
    regenerate_video_candidate,
)
from remove_ai_watermarks.video_synthid import DEFAULT_VIDEO_SYNTHID_VAE  # noqa: E402
from remove_ai_watermarks.video_temporal import _backward_map  # noqa: E402
from remove_ai_watermarks.video_visible import _timestamp_time_base, _TimestampedNutWriter  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

log = logging.getLogger(__name__)
LOWPASS64_FILTER = "scale=64:36:flags=area,gblur=sigma=1,scale={width}:{height}:flags=bicubic"


def peak_process_rss_bytes() -> int | None:
    """Return the Unix process high-water mark, or None when unavailable."""
    try:
        import resource
    except ImportError:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)


def json_safe(value: Any) -> Any:
    """Represent unbounded metrics as null rather than nonstandard JSON Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def blend_neighbors(
    current: NDArray[Any],
    neighbors: Iterable[NDArray[Any]],
    *,
    strength: float = 0.5,
    flow_pairs: list[Any] | None = None,
) -> NDArray[Any]:
    """Average aligned source neighbors, rejecting cuts and inconsistent flow.

    This is a hypothesis about temporal noise, not a watermark detector. Each
    neighbor is an unprocessed source frame; no recursive feedback is used.
    """
    if not 0 <= strength <= 1:
        raise ValueError("Strength must be between zero and one")
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    grid_y, grid_x = np.mgrid[:height, :width].astype(np.float32)
    total = current.astype(np.float32)
    weights = np.ones((height, width), np.float32)
    for index, neighbor in enumerate(neighbors):
        if neighbor.shape != current.shape:
            raise ValueError("Neighbor geometry differs")
        if flow_pairs is None:
            neighbor_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
            maps = _backward_map(gray, neighbor_gray)
            reverse = _backward_map(neighbor_gray, gray)
        else:
            maps, reverse = flow_pairs[index]
        warped = cv2.remap(neighbor.astype(np.float32), *maps, cv2.INTER_LINEAR)
        residual = np.mean(np.abs(warped - current), axis=2)
        # Fail closed at scene changes; locally reject occlusion and out-of-frame pixels.
        if float(np.median(residual)) > 24:
            continue
        back_x = cv2.remap(reverse[0], *maps, cv2.INTER_LINEAR)
        back_y = cv2.remap(reverse[1], *maps, cv2.INTER_LINEAR)
        valid = (
            (maps[0] >= 0)
            & (maps[0] <= width - 1)
            & (maps[1] >= 0)
            & (maps[1] <= height - 1)
            & (np.hypot(back_x - grid_x, back_y - grid_y) <= 1.5)
            & (residual <= 24)
        )
        weight = strength * valid.astype(np.float32)
        total += warped * weight[..., None]
        weights += weight
    return np.clip(np.rint(total / weights[..., None]), 0, 255).astype(np.uint8)


def temporal_frames(frames: Iterable[NDArray[Any]]) -> Iterator[NDArray[Any]]:
    """Use a three-frame source window, bounded independently of clip length."""
    iterator = iter(frames)
    current = next(iterator, None)
    previous = None
    previous_pair = None
    while current is not None:
        following = next(iterator, None)
        neighbors = [frame for frame in (previous, following) if frame is not None]
        pairs = [] if previous_pair is None else [previous_pair[::-1]]
        following_pair = None
        if following is not None:
            current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
            following_gray = cv2.cvtColor(following, cv2.COLOR_BGR2GRAY)
            following_pair = (_backward_map(current_gray, following_gray), _backward_map(following_gray, current_gray))
            pairs.append(following_pair)
        yield blend_neighbors(current, neighbors, flow_pairs=pairs)
        previous, current, previous_pair = current, following, following_pair


def video_clock(path: Path, *, include_packets: bool = False) -> dict[str, Any]:
    entries = "stream=time_base,start_pts,duration_ts,duration"
    if include_packets:
        entries += ":packet=pts,dts,duration"
    payload = _run_ffprobe(
        path,
        [
            "-select_streams",
            "v:0",
            "-show_entries",
            entries,
        ],
        purpose="research video clock",
        output_format="json",
    )
    if payload is None:
        raise RuntimeError(f"Could not inspect the video clock of {path}")
    return json.loads(payload)


def match_video_end(encoded: Path, source: Path) -> None:
    """Restore terminal packet duration, failing if the final stream end differs.

    NUT carries presentation times, but rawvideo lacks the source's terminal
    packet duration. Preserve PTS/DTS explicitly when using the setts filter.
    """
    original, candidate = video_clock(source), video_clock(encoded)
    original_stream, candidate_stream = original["streams"][0], candidate["streams"][0]
    target_duration = float(original_stream["duration"])
    if abs(float(candidate_stream["duration"]) - target_duration) <= 0.000002:
        return
    repair = video_end_repair(original, video_clock(encoded, include_packets=True))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    with tempfile.TemporaryDirectory(prefix="timing-", dir=encoded.parent) as directory:
        retimed = Path(directory) / "retimed.mp4"
        command = [
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
            str(retimed),
        ]
        log.info("Restoring video end: %s", command)
        subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
        actual = float(video_clock(retimed)["streams"][0]["duration"])
        if abs(actual - target_duration) > 0.000002:
            raise ValueError(f"Could not preserve video end: {actual} != {target_duration}")
        retimed.replace(encoded)


def encode_frames(
    frames: Iterable[NDArray[Any]],
    source: Path,
    output: Path,
    fps: float,
    *,
    timestamps: tuple[float, ...] = (),
    crf: int = 18,
) -> None:
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    profile = probe_video_encode_profile(source)
    time_base = _timestamp_time_base(profile.time_base)
    if timestamps:
        profile = replace(profile, time_base=str(time_base))
    with staged_video_output(output) as (encoded, temporary):
        encoder = start_raw_video_encoder(
            raw_video_command(
                encoded,
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                profile=profile,
                timestamped_input=bool(timestamps),
                copy_input_timestamps=bool(timestamps),
            )
        )
        writer = None
        try:
            frames_to_write = chain((first,), iterator)
            if timestamps:
                writer = _TimestampedNutWriter(
                    encoder.stdin,
                    width=width,
                    height=height,
                    time_base=time_base,
                    start_pts=round(timestamps[0] / float(time_base)),
                )
                for frame, timestamp in zip(frames_to_write, timestamps, strict=True):
                    writer.write(frame, timestamp)
                writer.close()
                writer = None
            else:
                for frame in frames_to_write:
                    encoder.stdin.write(frame.tobytes())
            finish_raw_video_encoder(encoder, encoded, operation="Seedance study encode")
            if timestamps:
                match_video_end(encoded, source)
            mux_encoded_video(encoded, source, temporary, strip_metadata=True, copy_input_timestamps=bool(timestamps))
            if timestamps:
                match_video_end(temporary, source)
        except Exception:
            if writer is not None:
                with suppress(Exception):
                    writer.close()
            abort_raw_video_encoder(encoder)
            raise


def regenerated_frames(frames: Iterable[NDArray[Any]], runtime: Any, noise: float) -> Iterator[NDArray[Any]]:
    """Use the shipped VAE composition without its CFR sampling or encoding."""
    shared_noise = None
    for batch in _stream_batches(frames, 4):
        latents = _encode_frame_latents(
            batch, vae=runtime.vae, device=runtime.resolved_device, batch_size=4, scaling_factor=runtime.scaling_factor
        )
        if shared_noise is None:
            shared_noise = _shared_latent_noise(
                latents[0].shape[1:], seed=0, device=runtime.resolved_device, dtype=latents[0].dtype
            )
        yield from _decode_frame_latents(
            latents, vae=runtime.vae, noise_std=noise, shared_noise=shared_noise, scaling_factor=runtime.scaling_factor
        )


def blend_reconstruction_frames(source: Path, reconstruction: Path, weight: float) -> Iterator[NDArray[Any]]:
    """Stream a native-size source/reconstruction mixture at matching frame times."""
    if not 0 < weight < 1:
        raise ValueError("Reconstruction weight must be strictly between 0 and 1")
    source_pts, reconstruction_pts = probe_video_timestamps(source), probe_video_timestamps(reconstruction)
    if (
        not source_pts
        or len(source_pts) != len(reconstruction_pts)
        or any(abs(a - b) > 0.000002 for a, b in zip(source_pts, reconstruction_pts, strict=True))
    ):
        raise ValueError("Source and reconstruction frame timestamps differ")
    for original, reconstructed in zip(_iter_frames(source), _iter_frames(reconstruction), strict=True):
        resized = cv2.resize(reconstructed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        mixed = (1 - weight) * original.astype(np.float64) + weight * resized.astype(np.float64)
        yield np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def encode_lowpass64(source: Path, output: Path, *, lowpass: bool) -> None:
    """Run the fixed destructive low-pass arm or its native-size codec control."""
    width, height, _ = _probe_video(source)
    profile = probe_video_encode_profile(source)
    time_base = _timestamp_time_base(profile.time_base)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    with staged_video_output(output) as (encoded, temporary):
        command = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-copyts",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
        ]
        if lowpass:
            command.extend(["-vf", LOWPASS64_FILTER.format(width=width, height=height)])
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-fps_mode",
                "passthrough",
                "-enc_time_base:v",
                str(time_base),
                "-video_track_timescale",
                str(time_base.denominator),
                "-avoid_negative_ts",
                "disabled",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                str(encoded),
            ]
        )
        result = subprocess.run(command, capture_output=True, text=True)  # noqa: S603
        log.log(
            logging.ERROR if result.returncode else logging.INFO,
            "Native low-pass encode: command=%s status=%s stdout=%s stderr=%s",
            command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        result.check_returncode()
        match_video_end(encoded, source)
        mux_encoded_video(encoded, source, temporary, strip_metadata=True, copy_input_timestamps=True)
        match_video_end(temporary, source)


def suppression_status(
    candidate: dict[str, Any], rows: list[dict[str, Any]], observations: list[dict[str, Any]]
) -> str:
    """Require same-pass captured positive control and negative candidate receipts."""
    required = {row["id"]: row for row in rows}
    for session in {obs["session"] for obs in observations if isinstance(obs.get("session"), str) and obs["session"]}:

        def has(artifact_id: str, verdict: str, session: str = session) -> bool:
            artifact = required.get(artifact_id, {})
            matching = [
                obs
                for obs in observations
                if obs.get("session") == session
                and obs.get("surface") == "https://ai-generated-detect.capcut.com/"
                and obs.get("artifact_id") == artifact_id
            ]
            return bool(matching) and all(receipt_verdict(obs.get("receipt"), artifact) == verdict for obs in matching)

        if has(candidate["control_id"], "detected") and has(candidate["id"], "not_detected"):
            return "observed_suppression"
    return "unverified"


def verify_artifacts(report: dict[str, Any]) -> None:
    for row in report["artifacts"]:
        if sha256(Path(row["path"])) != row["sha256"]:
            raise click.ClickException(f"Artifact changed: {row['id']}")


def validate_input(source: Path, ffprobe: str, fps: float, *, preserve_timestamps: bool = False) -> None:
    probe = subprocess.run(  # noqa: S603
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=pix_fmt,color_transfer",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    if stream.get("pix_fmt") not in {"yuv420p", "yuv422p", "yuv444p"} or stream.get("color_transfer") in {
        "smpte2084",
        "arib-std-b67",
    }:
        raise click.ClickException("This study supports only 8-bit SDR YUV inputs")
    # Read one timestamp at a time; frame metadata must not grow with clip length.
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(  # noqa: S603
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "compact=p=0",
                str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
        )
        previous = None
        count = 0
        try:
            if process.stdout is None:
                raise RuntimeError("Missing ffprobe output")
            for line in process.stdout:
                if not line.strip():
                    continue
                field = line.split("|", 1)[0].strip()
                if not field.startswith("best_effort_timestamp_time="):
                    raise click.ClickException("Missing frame timing")
                timestamp = float(field.split("=", 1)[1])
                if not math.isfinite(timestamp) or (
                    previous is not None
                    and (
                        timestamp <= previous
                        or (
                            not preserve_timestamps
                            and not math.isclose(timestamp - previous, 1 / fps, abs_tol=0.0001, rel_tol=0)
                        )
                    )
                ):
                    raise click.ClickException("Variable frame timing requires a timestamp-preserving engine")
                previous = timestamp
                count += 1
            if process.wait() != 0:
                errors.seek(0)
                raise click.ClickException(f"ffprobe failed: {errors.read()}")
            if count < 2:
                raise click.ClickException("At least two timestamped frames are required")
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


@click.group()
def main() -> None:
    """Run local video experiments and attach separately captured evidence."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--origin", required=True, help="Generator version and delivery channel, supplied by the operator.")
@click.option("--long-side", multiple=True, type=click.IntRange(min=8), default=(512,))
@click.option("--fps", type=click.FloatRange(min=1), default=None, help="Default: source frame rate.")
@click.option(
    "--codec-crf", multiple=True, type=click.IntRange(0, 51), help="Add codec-only candidates at fixed CRF values."
)
@click.option("--noise", multiple=True, type=click.FloatRange(0, 1), default=(0.0, 0.15))
@click.option(
    "--preserve-timestamps", is_flag=True, help="Keep every source frame and its PTS; incompatible with --fps."
)
@click.option("--skip-vae", is_flag=True, help="Run controls and temporal experiment without model loading.")
@click.option(
    "--reconstruction-weight",
    "reconstruction_weights",
    multiple=True,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    help="Add native-size mixtures with this VAE fraction; requires --preserve-timestamps and VAE processing.",
)
@click.option(
    "--blend-crf",
    type=click.IntRange(0, 51),
    default=0,
    show_default=True,
    help="Encoding CRF for mixtures and their separate native-size control; 0 is lossless YUV coding.",
)
@click.option(
    "--include-lowpass64",
    is_flag=True,
    help="Add destructive native-size 64x36 low-pass and its codec control; requires --preserve-timestamps.",
)
@click.option("--device", type=click.Choice(["auto", "cpu", "mps", "cuda"]), default="auto")
def run(
    source: Path,
    output_dir: Path,
    origin: str,
    long_side: tuple[int, ...],
    fps: float | None,
    noise: tuple[float, ...],
    codec_crf: tuple[int, ...],
    skip_vae: bool,
    device: str,
    preserve_timestamps: bool,
    include_lowpass64: bool,
    reconstruction_weights: tuple[float, ...],
    blend_crf: int,
) -> None:
    """Process the complete SOURCE. Only 8-bit SDR inputs qualify."""
    source = source.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(REPO):
        raise click.ClickException("Keep experiment artifacts outside the repository")
    if output_dir.exists():
        raise click.ClickException("Output directory already exists; use a new run directory")
    if preserve_timestamps and fps is not None:
        raise click.ClickException("--preserve-timestamps cannot be combined with --fps")
    if include_lowpass64 and not preserve_timestamps:
        raise click.ClickException("--include-lowpass64 requires --preserve-timestamps")
    if reconstruction_weights and (skip_vae or not preserve_timestamps):
        raise click.ClickException("--reconstruction-weight requires --preserve-timestamps and cannot use --skip-vae")
    if any(not 0 < weight < 1 for weight in reconstruction_weights):
        raise click.ClickException("Reconstruction weights must be strictly between 0 and 1")
    geometry = _probe_video(source)
    effective_fps = min(fps or geometry[2], geometry[2])
    sizes = sorted({min(side, max(geometry[:2])) for side in long_side})
    # This runner deliberately refuses formats the existing BGR/CFR engine cannot preserve.
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise click.ClickException("ffprobe is required")
    validate_input(source, ffprobe, geometry[2], preserve_timestamps=preserve_timestamps)
    timestamps = probe_video_timestamps(source) if preserve_timestamps else ()
    if preserve_timestamps and not timestamps:
        raise click.ClickException("Authoritative source timestamps are required")
    output_dir.mkdir(parents=True)
    code_paths = [
        Path(__file__),
        REPO / "scripts/video_fidelity_probe.py",
        REPO / "scripts/capcut_receipts.py",
        REPO / "scripts/invisible_quality_audit.py",
    ] + [
        REPO / "src/remove_ai_watermarks" / name
        for name in (
            "video_invisible.py",
            "video_encoding.py",
            "video_temporal.py",
            "video_synthid.py",
            "video_visible.py",
            "metadata.py",
        )
    ]
    for path in code_paths:
        snapshot = output_dir / "code" / path.relative_to(REPO)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, snapshot)
    plan = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256(source),
        "origin": origin,
        "geometry": geometry,
        "fps": effective_fps,
        "preserve_timestamps": preserve_timestamps,
        "include_lowpass64": include_lowpass64,
        "lowpass64_filter": LOWPASS64_FILTER.format(width=geometry[0], height=geometry[1])
        if include_lowpass64
        else None,
        "long_sides": sizes,
        "noise": noise,
        "codec_crfs": sorted(set(codec_crf)),
        "skip_vae": skip_vae,
        "reconstruction_weights": sorted(set(reconstruction_weights)),
        "blend_crf": blend_crf,
        "device": device,
        "seed": 0,
        "model": DEFAULT_VIDEO_SYNTHID_VAE,
        "code_snapshot_root": "code",
        "code_sha256": {
            str(path.relative_to(REPO)): sha256(output_dir / "code" / path.relative_to(REPO)) for path in code_paths
        },
        "ffmpeg": subprocess.run(  # noqa: S603
            [shutil.which("ffmpeg") or "ffmpeg", "-version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()[0],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless")},
    }
    write_json(output_dir / "plan.json", plan)
    report: dict[str, Any] = {
        "schema_version": 1,
        "plan_sha256": sha256(output_dir / "plan.json"),
        "certifies_erasure": False,
        "audio_watermark_status": "unverified",
        "artifacts": [],
    }

    def record(
        artifact_id: str,
        path: Path,
        *,
        control_id: str | None = None,
        started: float | None = None,
        measure: bool = True,
        guide_id: str | None = None,
        reconstruction_weight: float | None = None,
    ) -> None:
        row = {
            "id": artifact_id,
            "path": str(path),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
            "control_id": control_id,
            "guide_id": guide_id,
            "reconstruction_weight": reconstruction_weight,
            "processing_seconds": None if started is None else round(time.monotonic() - started, 3),
            "watermark_status": "unverified",
            "capcut_signal_status": "unverified",
            "fidelity": None,
        }
        report["artifacts"].append(row)
        write_json(output_dir / "results.json", report)
        if measure:
            row["fidelity"] = _measure(
                source, path, source_geometry=geometry, duration=None, preserve_timestamps=preserve_timestamps
            )
            if sha256(path) != row["sha256"]:
                raise click.ClickException(f"Artifact changed while measuring: {artifact_id}")
        write_json(output_dir / "results.json", report)
        log.info("Recorded %s: %s", artifact_id, row["fidelity"])

    record("source", source, measure=False)
    metadata = output_dir / f"metadata{source.suffix}"
    remove_video_metadata(source, metadata, keep_standard=False)
    record("metadata", metadata)
    blend_control_id = f"reencode-blend-crf{blend_crf}"
    if reconstruction_weights:
        started = time.monotonic()
        path = output_dir / f"{blend_control_id}.mp4"
        encode_frames(_iter_frames(source), source, path, effective_fps, timestamps=timestamps, crf=blend_crf)
        record(blend_control_id, path, started=started)
    if include_lowpass64:
        for lowpass, artifact_id in [(False, "reencode-native"), (True, "lowpass64-native")]:
            started = time.monotonic()
            path = output_dir / f"{artifact_id}.mp4"
            encode_lowpass64(source, path, lowpass=lowpass)
            record(artifact_id, path, control_id="reencode-native" if lowpass else None, started=started)
    runtime = None
    for side in sizes:
        size = _fit_size(*geometry[:2], side)
        control_id = f"reencode-{side}"
        for method, crf in [
            ("reencode", 18),
            ("temporal", 18),
            *[("codec", value) for value in sorted(set(codec_crf))],
        ]:
            artifact_id = f"codec-{side}-crf{crf}" if method == "codec" else f"{method}-{side}"
            output = output_dir / f"{artifact_id}.mp4"
            started = time.monotonic()
            frames = (
                (cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4) for frame in _iter_frames(source))
                if preserve_timestamps
                else _iter_sampled_frames(
                    source, source_fps=geometry[2], duration=None, effective_fps=effective_fps, size=size
                )
            )
            encode_frames(
                temporal_frames(frames) if method == "temporal" else frames,
                source,
                output,
                effective_fps,
                timestamps=timestamps,
                crf=crf,
            )
            record(artifact_id, output, control_id=control_id if method != "reencode" else None, started=started)
        if skip_vae:
            continue
        if runtime is None:
            runtime = load_video_vae_runtime(device=device)
            digest = hashlib.sha256()
            for key, tensor in sorted(runtime.vae.state_dict().items()):
                digest.update(key.encode())
                digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            report["vae_runtime"] = {
                "weight_state_sha256": digest.hexdigest(),
                "device": runtime.resolved_device,
                "scaling_factor": runtime.scaling_factor,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "diffusers")},
            }
        for index, level in enumerate(noise):
            artifact_id = f"vae-{side}-{index}"
            output = output_dir / f"{artifact_id}.mp4"
            started = time.monotonic()
            if preserve_timestamps:
                frames = (cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4) for frame in _iter_frames(source))
                encode_frames(
                    regenerated_frames(frames, runtime, level), source, output, effective_fps, timestamps=timestamps
                )
            else:
                regenerate_video_candidate(
                    source,
                    output,
                    long_side=side,
                    fps=effective_fps,
                    noise_std=level,
                    device=device,
                    runtime=runtime,
                    seed=0,
                )
            record(artifact_id, output, control_id=control_id, started=started)
            for weight in sorted(set(reconstruction_weights)):
                mixed_id = f"{artifact_id}-mix{weight}"
                mixed_path = output_dir / f"{mixed_id}.mp4"
                started = time.monotonic()
                encode_frames(
                    blend_reconstruction_frames(source, output, weight),
                    source,
                    mixed_path,
                    effective_fps,
                    timestamps=timestamps,
                    crf=blend_crf,
                )
                record(
                    mixed_id,
                    mixed_path,
                    control_id=blend_control_id,
                    started=started,
                    guide_id=artifact_id,
                    reconstruction_weight=weight,
                )
    verify_artifacts(report)
    report["peak_process_rss_bytes"] = peak_process_rss_bytes()
    report["complete"] = True
    write_json(output_dir / "results.json", report)


@main.command()
@click.argument("results", type=click.Path(exists=True, path_type=Path))
@click.argument("observations", type=click.Path(exists=True, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def assess(results: Path, observations: Path, output: Path) -> None:
    """Attach captured CapCut supported-signal receipts, without uploading."""
    if output.exists():
        raise click.ClickException("Assessment output already exists")
    report = json.loads(results.read_text())
    if sha256(results.parent / "plan.json") != report["plan_sha256"]:
        raise click.ClickException("Plan changed")
    plan = json.loads((results.parent / "plan.json").read_text())
    if plan.get("code_snapshot_root") == "code":
        for relative, digest in plan["code_sha256"].items():
            if sha256(results.parent / "code" / relative) != digest:
                raise click.ClickException(f"Code snapshot changed: {relative}")
    verify_artifacts(report)
    evidence = json.loads(observations.read_text())
    if not isinstance(evidence, list) or any(not isinstance(item, dict) for item in evidence):
        raise click.ClickException("Observations must be an array of receipt records")
    report["certifies_erasure"] = False
    report["audio_watermark_status"] = "unverified"
    report["assessment_code_sha256"] = {
        str(path.relative_to(REPO)): sha256(path) for path in (Path(__file__), REPO / "scripts/capcut_receipts.py")
    }
    report["observations"] = evidence
    report["observations_sha256"] = sha256(observations)
    for row in report["artifacts"]:
        row.setdefault("size_bytes", Path(row["path"]).stat().st_size)
        row["watermark_status"] = "unverified"
        row["capcut_signal_status"] = "unverified"
        if row["control_id"]:
            row["capcut_signal_status"] = suppression_status(row, report["artifacts"], evidence)
    write_json(output, report)


if __name__ == "__main__":
    main()
