"""Experimental known-message readouts for public video watermark models.

The expected encoder message must be supplied independently of decoding the
input. These APIs do not classify AI origin or detect arbitrary unknown marks.
"""

from __future__ import annotations

import logging
import subprocess
from contextlib import closing
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from remove_ai_watermarks._internal.audio_quality import sha256_file
from remove_ai_watermarks._internal.constants import (
    DEFAULT_VIDEO_MESSAGE_LONG_SIDE,
    VIDEO_MESSAGE_DEVICES,
    VIDEO_MESSAGE_METHODS,
    VIDEO_MESSAGE_SCHEMES,
)
from remove_ai_watermarks._internal.video_message_models import (
    VideoMessageScheme,
    decode_logits,
    decode_presence_and_logits,
    load_detector,
)
from remove_ai_watermarks._internal.video_message_models import (
    is_available as _model_available,
)
from remove_ai_watermarks._internal.video_message_scores import (
    SCHEME_MESSAGE_BITS,
    MessageAccumulator,
    VideoMessageReading,
    reference_bits,
)
from remove_ai_watermarks.optional_deps import module_available

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from numpy.typing import NDArray

_SCOPE = (
    "Correlation to the caller's independently known 256-bit encoder message, under the pinned public model.",
    "AI origin, unknown messages, provider deployments, and complete information erasure remain unverified.",
    "Only the first video stream is inspected. Audio is outside these checks.",
    "The receiver checks every decoded frame, four float64 aggregations, and sliding 12-frame windows.",
)


def _scope_for_scheme(scheme: VideoMessageScheme) -> tuple[str, ...]:
    bits = SCHEME_MESSAGE_BITS[scheme]
    return (
        f"Correlation to the caller's independently known {bits}-bit encoder message, under the pinned public model.",
        *_SCOPE[1:],
    )


_CODEC_MIN_PSNR_DB = 25.0
_CODEC_MIN_SSIM = 0.75
_VAE_MIN_PSNR_DB = 18.0
_VAE_MIN_SSIM = 0.35


@dataclass(frozen=True)
class VideoMessageReport:
    """Hash-bound evidence for one known message, without a general origin verdict."""

    source: Path
    sha256: str
    scheme: VideoMessageScheme
    reference_hex: str
    status: Literal["matched", "not_matched", "unverified"]
    reading: VideoMessageReading | None
    caveats: tuple[str, ...] = _SCOPE


def is_available() -> bool:
    return _model_available() and module_available("av")


def _pack_normalized_batch(frames: list[NDArray[Any]]) -> NDArray[Any]:
    import numpy as np

    out = np.empty((len(frames), *frames[0].shape), dtype=np.float32)
    for idx, frame in enumerate(frames):
        out[idx] = frame
    out /= 255.0
    return out


def _video_batches(source: Path) -> Generator[NDArray[Any]]:
    """Decode every frame; bound batch memory and reject incomplete streams."""
    import numpy as np

    av = import_module("av")
    with av.open(str(source), options={"protocol_whitelist": "file,pipe"}) as container:
        if not container.streams.video:
            raise ValueError("Source has no video stream")
        stream = container.streams.video[0]
        stream.codec_context.options = {**stream.codec_context.options, "err_detect": "explode"}
        expected_frames = stream.frames
        count = 0
        batch: list[NDArray[Any]] = []
        pixels = 0
        for frame in container.decode(stream):
            rotation = frame.rotation
            if rotation % 90:
                raise ValueError("Known-message checks require a right-angle display rotation")
            rgb = frame.to_ndarray(format="rgb24")
            if rotation:
                rgb = np.rot90(rgb, rotation // 90)
            if batch and (rgb.shape != batch[0].shape or len(batch) >= 8 or pixels + rgb.size > 8_000_000):
                yield _pack_normalized_batch(batch)
                batch.clear()
                pixels = 0
            batch.append(rgb)
            pixels += rgb.size
            count += 1
        if batch:
            yield _pack_normalized_batch(batch)
        if not count or (expected_frames and count != expected_frames):
            raise RuntimeError("Video stream did not decode to its complete declared frame count")


def identify_video_message(
    source: str | Path,
    *,
    reference_hex: str,
    scheme: VideoMessageScheme = "videoseal",
    check_message: bool = True,
) -> VideoMessageReport:
    """Compare every frame to an independently known encoder message.

    Deriving the reference from this input would make a source positive
    circular. Missing optional decoders remain unverified. Model, numerical,
    demux, and completeness errors propagate and never become negatives.
    """
    if scheme not in VIDEO_MESSAGE_SCHEMES:
        raise ValueError("Video message scheme must be videoseal, pixelseal, or chunkyseal")
    bits = reference_bits(reference_hex, num_bits=SCHEME_MESSAGE_BITS[scheme])
    scope = _scope_for_scheme(scheme)
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Video source must be an existing file")
    digest = sha256_file(path)
    if not check_message:
        return VideoMessageReport(
            path, digest, scheme, reference_hex.lower(), "unverified", None, (*scope, "Message decoding was skipped.")
        )
    if not is_available():
        return VideoMessageReport(
            path,
            digest,
            scheme,
            reference_hex.lower(),
            "unverified",
            None,
            (*scope, "Local checks require remove-ai-watermarks[video-detect]."),
        )
    model = load_detector(scheme)
    accumulator = MessageAccumulator(bits)
    with closing(_video_batches(path)) as batches:
        for frames in batches:
            accumulator.update(decode_logits(model, frames))
    reading = accumulator.finish()
    if sha256_file(path) != digest:
        raise RuntimeError("Video source changed during message inspection")
    status = "matched" if reading.target_matched else "not_matched"
    return VideoMessageReport(path, digest, scheme, reference_hex.lower(), status, reading, scope)


@dataclass(frozen=True)
class VideoWatermarkPresenceReport:
    """Blind public-model watermark presence and recovered payload report."""

    source: Path
    sha256: str
    scheme: VideoMessageScheme
    status: Literal["detected", "not_detected", "unverified"]
    frames: int
    mean_presence_prob: float | None
    bit_sign_consistency: float | None
    decoded_hex: str | None
    caveats: tuple[str, ...] = ()


def _bits_to_hex(bits: NDArray[Any]) -> str:
    packed = bytearray()
    for start in range(0, len(bits), 8):
        byte_val = 0
        for b in bits[start : start + 8]:
            byte_val = (byte_val << 1) | int(bool(b))
        packed.append(byte_val)
    return bytes(packed).hex()


def detect_video_watermark_presence(
    source: str | Path,
    *,
    scheme: VideoMessageScheme = "videoseal",
) -> VideoWatermarkPresenceReport:
    """Inspect a video for blind VideoSeal, PixelSeal, or ChunkySeal presence without a known hex key."""
    import numpy as np

    if scheme not in VIDEO_MESSAGE_SCHEMES:
        raise ValueError("Video message scheme must be videoseal, pixelseal, or chunkyseal")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Video source must be an existing file")
    digest = sha256_file(path)
    num_bits = SCHEME_MESSAGE_BITS[scheme]
    caveats = (
        f"Blind {scheme} presence check combines channel-0 probability and {num_bits}-bit cross-frame stability.",
        "Covers only the pinned public model family; proprietary provider keys or SynthID remain unverified.",
    )
    if not is_available():
        return VideoWatermarkPresenceReport(
            path,
            digest,
            scheme,
            "unverified",
            0,
            None,
            None,
            None,
            (*caveats, "Local checks require remove-ai-watermarks[video-detect]."),
        )
    model = load_detector(scheme)
    total_frames = 0
    prob_sum = 0.0
    logit_sum = np.zeros(num_bits, dtype=np.float64)
    sign_sum = np.zeros(num_bits, dtype=np.float64)
    with closing(_video_batches(path)) as batches:
        for frames in batches:
            pres, msg = decode_presence_and_logits(model, frames)
            pres64 = np.clip(pres.astype(np.float64), -40.0, 40.0)
            probs = 1.0 / (1.0 + np.exp(-pres64))
            prob_sum += float(np.sum(probs))
            msg64 = msg.astype(np.float64)
            logit_sum += np.sum(msg64, axis=0)
            sign_sum += np.sum(np.where(msg64 >= 0.0, 1.0, -1.0), axis=0)
            total_frames += len(frames)
    if sha256_file(path) != digest:
        raise RuntimeError("Video source changed during presence inspection")
    mean_prob = prob_sum / max(total_frames, 1)
    consistency = float(np.mean(np.abs(sign_sum / max(total_frames, 1))))
    detected = (mean_prob >= 0.70 and consistency >= 0.65) if total_frames >= 4 else (mean_prob >= 0.85)
    recovered_hex = _bits_to_hex(logit_sum > 0.0) if detected else None
    return VideoWatermarkPresenceReport(
        path,
        digest,
        scheme,
        "detected" if detected else "not_detected",
        total_frames,
        mean_prob,
        consistency,
        recovered_hex,
        caveats,
    )


@dataclass(frozen=True)
class VideoMessageRemovalResult:
    """A saved known-target result, with delivered-file quality and explicit scope."""

    source: Path
    output: Path | None
    before: VideoMessageReport
    after: VideoMessageReport | None
    verified_removed: bool | None
    skipped: bool
    method: str | None = None
    noise_std: float | None = None
    quality: dict[str, Any] | None = None
    copied_audio_streams: tuple[int, ...] = ()
    quality_passed: bool | None = None


class VideoMessageVerificationIncomplete(RuntimeError):
    """A candidate did not pass the requested known-message checks."""

    def __init__(self, before: VideoMessageReport, after: VideoMessageReport, reason: str | None = None) -> None:
        message = f"Saved video message check is {after.status}; existing output was preserved"
        super().__init__(message if reason is None else f"{message}. {reason}")
        self.before = before
        self.after = after


def _expected_sampled_frames(timestamps: Sequence[float], effective_fps: float) -> int:
    from remove_ai_watermarks.video_invisible import select_sampled_indices

    return len(select_sampled_indices(timestamps, effective_fps))


def _quality_passes(profile: str, quality: dict[str, Any]) -> bool:
    import math

    ssim = quality.get("source_ssim")
    if not isinstance(ssim, (int, float)) or not math.isfinite(ssim):
        return False
    if quality.get("lossless_pixels") is True:
        return ssim >= (_CODEC_MIN_SSIM if profile == "codec" else _VAE_MIN_SSIM)
    psnr = quality.get("source_psnr_db")
    if not isinstance(psnr, (int, float)) or not math.isfinite(psnr):
        return False
    if profile == "codec":
        return psnr >= _CODEC_MIN_PSNR_DB and ssim >= _CODEC_MIN_SSIM
    return psnr >= _VAE_MIN_PSNR_DB and ssim >= _VAE_MIN_SSIM


def _processing_source(
    source: Path,
) -> tuple[tuple[int, int, float], tuple[dict[str, Any], ...], tuple[float, ...]]:
    """Validate single-video source streams and monotonic timestamps (CFR, VFR, or non-zero start PTS)."""
    import itertools
    from fractions import Fraction

    from remove_ai_watermarks.video_audio import stream_inventory
    from remove_ai_watermarks.video_encoding import probe_video_timestamps
    from remove_ai_watermarks.video_invisible import probe_video_geometry

    streams = stream_inventory(source)
    videos = [stream for stream in streams if stream["codec_type"] == "video"]
    if len(videos) != 1 or any(stream["codec_type"] not in {"video", "audio"} for stream in streams):
        raise ValueError("Video message processing requires one video stream and optional audio streams")
    geometry = probe_video_geometry(source)
    timestamps = probe_video_timestamps(source)
    tolerance = float(Fraction(videos[0]["time_base"])) + 0.000001
    if len(timestamps) < 2 or any(b < a - tolerance for a, b in itertools.pairwise(timestamps)):
        raise ValueError("Video message processing requires at least two monotonic video timestamps")
    return geometry, tuple(stream for stream in streams if stream["codec_type"] == "audio"), timestamps


def _restore_audio_metadata(output: Path, audio_streams: tuple[dict[str, Any], ...]) -> None:
    """Restore only audio language and disposition after the metadata-stripping pixel encoder."""
    if not audio_streams:
        return
    from remove_ai_watermarks._internal.audio_io import require_tool
    from remove_ai_watermarks._internal.utils import staged_sibling

    with staged_sibling(output) as staged:
        command = [
            require_tool("ffmpeg"),
            "-v",
            "error",
            "-y",
            "-copyts",
            "-i",
            str(output),
            "-map",
            "0",
            "-c",
            "copy",
            "-map_metadata",
            "-1",
            "-map_chapters",
            "-1",
            "-avoid_negative_ts",
            "disabled",
        ]
        for ordinal, stream in enumerate(audio_streams):
            language = stream.get("tags", {}).get("language")
            if language is not None:
                command.extend([f"-metadata:s:a:{ordinal}", f"language={language}"])
            disposition = "+".join(name for name, value in stream.get("disposition", {}).items() if value)
            command.extend([f"-disposition:a:{ordinal}", disposition or "0"])
        from remove_ai_watermarks._internal.constants import ISOBMFF_VIDEO_EXTENSIONS

        if output.suffix.lower() in ISOBMFF_VIDEO_EXTENSIONS:
            command.extend(["-movflags", "+faststart"])
        command.append(str(staged))
        result = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
        log.info(
            "Video audio metadata restore: command=%s status=%s stdout=%s stderr=%s",
            command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        if result.returncode:
            raise RuntimeError(f"Could not restore copied audio metadata: {result.stderr}")


def _check_copied_audio(source: Path, output: Path, audio_streams: tuple[dict[str, Any], ...]) -> tuple[int, ...]:
    from remove_ai_watermarks._internal import audio_io
    from remove_ai_watermarks._internal.audio_quality import measure_fidelity
    from remove_ai_watermarks.video_audio import stream_inventory, verify_stream_packets

    delivered = [stream for stream in stream_inventory(output) if stream["codec_type"] == "audio"]
    if len(delivered) != len(audio_streams):
        raise RuntimeError("Video message processing changed the audio stream count")
    for ordinal, (before, after) in enumerate(zip(audio_streams, delivered, strict=True)):
        verify_stream_packets(source, output, int(before["index"]), output_index=int(after["index"]))
        info = audio_io.probe_audio(source, stream=ordinal)
        fidelity = measure_fidelity(source, info, output, stream=ordinal)
        if fidelity.max_abs_error != 0:
            raise RuntimeError("Video message processing changed decoded audio samples")
        if before.get("tags", {}).get("language") != after.get("tags", {}).get("language"):
            raise RuntimeError("Video message processing changed an audio language")
        if before.get("disposition", {}) != after.get("disposition", {}):
            raise RuntimeError("Video message processing changed an audio disposition")
    return tuple(int(stream["index"]) for stream in audio_streams)


def remove_video_message(
    source: str | Path,
    output: str | Path,
    *,
    reference_hex: str,
    scheme: VideoMessageScheme = "videoseal",
    method: Literal["auto", "codec", "vae"] = "auto",
    noise_std: float | None = None,
    long_side: int = DEFAULT_VIDEO_MESSAGE_LONG_SIDE,
    device: str = "auto",
    temporal_stabilize: bool = False,
    force: bool = False,
    verify: bool = True,
) -> VideoMessageRemovalResult:
    """Process a known public-model target behind a saved-file verification gate.

    Auto tries codec-only, VAE round trip, then fixed latent-noise profiles. The
    first complete negative is published. Force permits only the final failed
    candidate, preserving its verification status. Processing samples at up to
    12 fps, strips metadata, and copies source audio without cleaning its marks.
    Nonuniform or nonzero-based video timelines and additional stream types are
    rejected. These public-model checks do not establish removal of SynthID.
    """
    import math
    from dataclasses import replace

    from remove_ai_watermarks._internal.utils import staged_sibling
    from remove_ai_watermarks.video_invisible import (
        encode_video_frames,
        fit_video_size,
        iter_sampled_frames,
        regenerate_video_candidate,
    )
    from remove_ai_watermarks.video_quality import measure_video_fidelity

    if not verify and not force:
        raise ValueError("Disabling verification requires force=True")
    if method not in VIDEO_MESSAGE_METHODS:
        raise ValueError("Video message method must be auto, codec, or vae")
    if noise_std is not None and (method != "vae" or not math.isfinite(noise_std) or not 0 <= noise_std <= 1):
        raise ValueError("Explicit noise_std requires method='vae' and a finite value from 0 to 1")
    if long_side < 8 or device not in VIDEO_MESSAGE_DEVICES:
        raise ValueError("Video message geometry or device is invalid")
    original = Path(source).expanduser().resolve()
    target = Path(output).expanduser()
    if target.resolve() == original or (target.exists() and original.exists() and target.samefile(original)):
        raise ValueError("Video output must differ from the source")
    from remove_ai_watermarks._internal.constants import VIDEO_EXTENSIONS

    if target.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Video message outputs must be MP4, MOV, M4V, MKV, WebM, AVI, or FLV")
    before = identify_video_message(original, reference_hex=reference_hex, scheme=scheme, check_message=verify)
    if before.status != "matched" and not force:
        return VideoMessageRemovalResult(original, None, before, None, None, True)
    geometry, audio_streams, source_timestamps = _processing_source(original)
    effective_fps = min(12.0, geometry[2])
    expected_frames = _expected_sampled_frames(source_timestamps, effective_fps)
    profiles: tuple[tuple[str, float | None], ...]
    if method == "auto":
        profiles = (("codec", None), ("vae", 0.0), ("vae", 0.15), ("vae", 0.25))
    elif method == "vae":
        profiles = (("vae", 0.15 if noise_std is None else noise_std),)
    else:
        profiles = (("codec", None),)
    target.parent.mkdir(parents=True, exist_ok=True)
    for index, (profile, noise) in enumerate(profiles):
        try:
            with staged_sibling(target) as staged:
                if profile == "codec":
                    frames = iter_sampled_frames(
                        original,
                        source_fps=geometry[2],
                        duration=None,
                        effective_fps=effective_fps,
                        size=fit_video_size(*geometry[:2], long_side),
                        timestamps=source_timestamps,
                    )
                    encode_video_frames(frames, original, staged, fps=effective_fps)
                else:
                    from remove_ai_watermarks._internal.video_message_regeneration import load_runtime

                    runtime = load_runtime(device)
                    regenerate_video_candidate(
                        original,
                        staged,
                        noise_std=float(noise or 0),
                        long_side=long_side,
                        fps=12,
                        batch_size=1,
                        seed=0,
                        device=device,
                        runtime=runtime,
                        temporal_stabilize=temporal_stabilize,
                        timestamps=source_timestamps,
                    )
                _restore_audio_metadata(staged, audio_streams)
                after = identify_video_message(staged, reference_hex=reference_hex, scheme=scheme, check_message=verify)
                if verify and after.status != "not_matched" and (not force or index < len(profiles) - 1):
                    raise VideoMessageVerificationIncomplete(before, after)
                copied = _check_copied_audio(original, staged, audio_streams)
                quality = measure_video_fidelity(
                    original,
                    staged,
                    source_geometry=geometry,
                    duration=None,
                    source_timestamps=source_timestamps,
                )
                if quality.get("frames") != expected_frames or (
                    after.reading is not None and after.reading.frames != expected_frames
                ):
                    raise RuntimeError("Delivered video frame count does not match the source timeline")
                quality["file"] = target.name
                if quality["source_psnr_db"] == math.inf:
                    quality["source_psnr_db"] = None
                    quality["lossless_pixels"] = True
                quality_ok = _quality_passes(profile, quality)
                if not quality_ok and (not force or index < len(profiles) - 1):
                    raise VideoMessageVerificationIncomplete(
                        before,
                        after,
                        "Delivered video failed the minimum quality gate",
                    )
                if sha256_file(original) != before.sha256 or sha256_file(staged) != after.sha256:
                    raise RuntimeError("Video source or candidate changed during processing")
                verified = (
                    True
                    if before.status == "matched" and after.status == "not_matched"
                    else False
                    if after.status == "matched"
                    else None
                )
                return VideoMessageRemovalResult(
                    original,
                    target,
                    before,
                    replace(after, source=target),
                    verified,
                    False,
                    profile,
                    noise,
                    quality,
                    copied,
                    quality_ok,
                )
        except VideoMessageVerificationIncomplete:
            if index == len(profiles) - 1:
                raise
    raise RuntimeError("No video message processing profile ran")
