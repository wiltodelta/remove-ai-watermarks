"""Oracle-certified VAE regeneration for video SynthID removal.

Google does not publish a local video SynthID decoder. The default profile is
therefore certified against Google's matching content-verification flow and
also reports local fidelity metrics.
"""

from __future__ import annotations

# torch/diffusers/cv2 expose incomplete types at this boundary. Pure helpers
# remain annotated while third-party tensor and image calls are relaxed here.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false
import logging
import math
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks.humanizer import unsharp_mask
from remove_ai_watermarks.video_decode import open_video_capture, open_video_capture_16bit
from remove_ai_watermarks.video_encoding import (
    HIGH_BIT_DEPTH_PIXEL_FORMATS,
    abort_raw_video_encoder,
    finish_raw_video_encoder,
    mux_encoded_video,
    probe_video_encode_profile,
    raw_video_command,
    staged_video_output,
    start_raw_video_encoder,
)
from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_FPS,
    DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    DEFAULT_VIDEO_SYNTHID_VAE,
    DEFAULT_VIDEO_SYNTHID_VAE_REVISION,
    VIDEO_SYNTHID_LATENT_MULTIPLE,
    VIDEO_SYNTHID_VAE_SCALING_FACTOR,
)
from remove_ai_watermarks.video_temporal import (
    _backward_map,
    _motion_residual,
    anchor_low_freq_color,
    build_temporal_reference,
    iter_stabilized_regenerated_frames,
    temporal_residual_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

log = logging.getLogger(__name__)

__all__ = ["build_temporal_reference", "temporal_residual_ratio"]


@dataclass(frozen=True)
class RegenerationMetrics:
    """Measured properties of one regenerated video."""

    frames: int
    fps: float
    width: int
    height: int
    psnr_db: float
    temporal_residual_ratio: float


@dataclass(frozen=True)
class VideoVaeRuntime:
    """Loaded VAE state reusable across multiple video regenerations."""

    model: str
    requested_device: str
    resolved_device: str
    vae: Any
    # The gate that validates this factor and the encode/decode calls that apply it
    # must read one value, not three independent reads of the same attribute.
    scaling_factor: float


def is_available() -> bool:
    """Return whether the optional VAE runtime can be imported."""
    from remove_ai_watermarks.optional_deps import module_available

    return module_available("torch", "diffusers")


def _fit_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Fit dimensions to ``long_side`` while preserving aspect and VAE alignment."""
    if width <= 0 or height <= 0:
        raise ValueError("Video dimensions must be positive")
    if long_side < VIDEO_SYNTHID_LATENT_MULTIPLE:
        raise ValueError(f"Long side must be at least {VIDEO_SYNTHID_LATENT_MULTIPLE}")
    scale = long_side / max(width, height)
    fitted_width = max(
        VIDEO_SYNTHID_LATENT_MULTIPLE,
        round(width * scale) // VIDEO_SYNTHID_LATENT_MULTIPLE * VIDEO_SYNTHID_LATENT_MULTIPLE,
    )
    fitted_height = max(
        VIDEO_SYNTHID_LATENT_MULTIPLE,
        round(height * scale) // VIDEO_SYNTHID_LATENT_MULTIPLE * VIDEO_SYNTHID_LATENT_MULTIPLE,
    )
    return fitted_width, fitted_height


def pick_device(requested: str) -> str:
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")
    return requested


def load_video_vae_runtime(
    *,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
) -> VideoVaeRuntime:
    """Load one reusable video VAE runtime."""
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    if not is_available():
        raise RuntimeError("Video SynthID regeneration requires the diffusion extra")

    import torch
    from diffusers import AutoencoderKL

    resolved_device = pick_device(device)
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    log.info("Loading %s on %s", model, resolved_device)
    model_options: dict[str, Any] = {}
    if model == DEFAULT_VIDEO_SYNTHID_VAE:
        model_options.update(revision=DEFAULT_VIDEO_SYNTHID_VAE_REVISION, use_safetensors=True)
    vae = AutoencoderKL.from_pretrained(model, torch_dtype=dtype, **model_options).to(resolved_device)
    vae.train(False)
    vae.enable_slicing()
    scaling_factor = float(vae.config.scaling_factor)
    log.info("Latent scaling factor %.5f", scaling_factor)
    if model != DEFAULT_VIDEO_SYNTHID_VAE:
        log.warning("No oracle-certified profile exists for %s; the shipped noise_std is not calibrated for it", model)
    elif scaling_factor != VIDEO_SYNTHID_VAE_SCALING_FACTOR:
        raise RuntimeError(
            f"{model} loaded with latent scaling factor {scaling_factor}, but the certified "
            f"profile is defined against {VIDEO_SYNTHID_VAE_SCALING_FACTOR}. The perturbation "
            "would be rescaled and the output would no longer match any certified row."
        )
    return VideoVaeRuntime(
        model=model,
        requested_device=device,
        resolved_device=resolved_device,
        vae=vae,
        scaling_factor=scaling_factor,
    )


def _shared_latent_noise(
    spatial_shape: Sequence[int],
    *,
    seed: int,
    device: str,
    dtype: Any,
) -> Any:
    """Return one deterministic spatial noise field for reuse across time."""
    import torch

    if len(spatial_shape) != 3 or any(size <= 0 for size in spatial_shape):
        raise ValueError("Expected a positive CHW latent shape")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, *spatial_shape), generator=generator, dtype=torch.float32)
    return noise.to(device=device, dtype=dtype)


def paired_psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return paired PSNR over uint8 or uint16 frame stacks."""
    if reference.shape != candidate.shape:
        raise ValueError("PSNR inputs must have matching shapes")
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    if mse == 0.0:
        return math.inf
    peak = 65535.0 if reference.dtype == np.uint16 else 255.0
    return 20.0 * math.log10(peak / math.sqrt(mse))


def _probe_video(source: Path) -> tuple[int, int, float]:
    capture = open_video_capture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    try:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Video has no usable dimensions: {source}")
    if source_fps <= 0.0:
        raise ValueError(f"Video has no usable frame rate: {source}")
    return width, height, source_fps


def read_sampled_frames(
    source: Path,
    *,
    duration: float | None,
    output_fps: float,
    size: tuple[int, int],
) -> tuple[list[np.ndarray], float]:
    """Read uniformly sampled frames and resize them to the VAE geometry."""
    _width, _height, source_fps = _probe_video(source)
    effective_fps = min(output_fps, source_fps)
    frames = list(
        _iter_sampled_frames(
            source,
            source_fps=source_fps,
            duration=duration,
            effective_fps=effective_fps,
            size=size,
        )
    )
    if len(frames) < 2:
        raise ValueError("The selected clip produced fewer than two frames")
    return frames, effective_fps


def select_sampled_indices(
    timestamps: Sequence[float],
    effective_fps: float,
    *,
    duration: float | None = None,
) -> list[int]:
    """Return frame indices selected by the uniform relative-timestamp sampling schedule."""
    if not timestamps:
        return []
    sample_period = 1.0 / effective_fps
    base_ts = float(timestamps[0])
    next_sample_time = 0.0
    selected: list[int] = []
    for index, ts in enumerate(timestamps):
        rel_ts = float(ts) - base_ts
        if duration is not None and rel_ts + 1e-9 >= duration:
            break
        if rel_ts + 1e-9 >= next_sample_time:
            selected.append(index)
            next_sample_time += sample_period
    return selected


def _iter_sampled_frames(
    source: Path,
    *,
    source_fps: float,
    duration: float | None,
    effective_fps: float,
    size: tuple[int, int],
    high_depth: bool = False,
    source_size: tuple[int, int] | None = None,
    timestamps: Sequence[float] | None = None,
    _frame_stream: Iterable[NDArray[Any]] | None = None,
) -> Generator[NDArray[Any], None, None]:
    """Yield uniformly sampled frames without retaining the full video or converting dropped frames."""
    sample_period = 1.0 / effective_fps
    next_sample_time = 0.0
    base_ts = float(timestamps[0]) if timestamps else 0.0
    selected_set = (
        set(select_sampled_indices(timestamps, effective_fps, duration=duration)) if timestamps is not None else None
    )
    max_selected = max(selected_set, default=-1) if selected_set is not None else None

    if _frame_stream is not None:
        for frame_index, frame in enumerate(_frame_stream):
            if selected_set is not None:
                if max_selected is not None and frame_index > max_selected:
                    break
                if frame_index in selected_set:
                    yield (
                        frame
                        if (frame.shape[1], frame.shape[0]) == size
                        else cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                    )
                continue
            timestamp = frame_index / source_fps
            if duration is not None and timestamp + 1e-9 >= duration:
                break
            if timestamp + 1e-9 >= next_sample_time:
                yield (
                    frame
                    if (frame.shape[1], frame.shape[0]) == size
                    else cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                )
                next_sample_time += sample_period
        return

    if high_depth:
        src_w, src_h = source_size if source_size is not None else _probe_video(source)[:2]
        capture: Any = open_video_capture_16bit(source, width=src_w, height=src_h)
    else:
        capture = open_video_capture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    has_grab = callable(getattr(capture, "grab", None)) and callable(getattr(capture, "retrieve", None))
    try:
        frame_index = 0
        while True:
            if selected_set is not None:
                if max_selected is not None and frame_index > max_selected:
                    break
                want = frame_index in selected_set
            else:
                timestamp = (
                    float(timestamps[frame_index]) - base_ts
                    if timestamps is not None and frame_index < len(timestamps)
                    else frame_index / source_fps
                )
                if duration is not None and timestamp + 1e-9 >= duration:
                    break
                want = timestamp + 1e-9 >= next_sample_time
                if want:
                    next_sample_time += sample_period
            if has_grab:
                if not capture.grab():
                    break
                if want:
                    ok, frm = capture.retrieve()
                    if not ok or frm is None:
                        break
                    yield (
                        frm
                        if (frm.shape[1], frm.shape[0]) == size
                        else cv2.resize(frm, size, interpolation=cv2.INTER_LANCZOS4)
                    )
            else:
                ok, frm = capture.read()
                if not ok or frm is None:
                    break
                if want:
                    yield (
                        frm
                        if (frm.shape[1], frm.shape[0]) == size
                        else cv2.resize(frm, size, interpolation=cv2.INTER_LANCZOS4)
                    )
            frame_index += 1
    finally:
        capture.release()


# Shared sampling policy for delivered-file measurements and bounded writers.
iter_sampled_frames = _iter_sampled_frames
fit_video_size = _fit_size
probe_video_geometry = _probe_video


def _frame_batches(frames: Sequence[np.ndarray], batch_size: int) -> Iterable[Sequence[np.ndarray]]:
    for start in range(0, len(frames), batch_size):
        yield frames[start : start + batch_size]


def _stream_batches(frames: Iterable[np.ndarray], batch_size: int) -> Iterable[list[np.ndarray]]:
    batch: list[np.ndarray] = []
    for frame in frames:
        batch.append(frame)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _encode_frame_latents(
    frames: Sequence[np.ndarray],
    *,
    vae: Any,
    device: str,
    batch_size: int,
    scaling_factor: float,
) -> list[Any]:
    """Encode source frames once so every candidate can reuse identical latents."""
    import torch

    latent_batches: list[Any] = []
    with torch.inference_mode():
        for batch in _frame_batches(frames, batch_size):
            rgb = np.stack([frame[:, :, ::-1] for frame in batch])
            if rgb.dtype == np.uint16:
                rgb_f32 = np.ascontiguousarray(rgb, dtype=np.float32) / 32767.5 - 1.0
                tensor = torch.from_numpy(rgb_f32).permute(0, 3, 1, 2).to(device=device, dtype=vae.dtype)
            else:
                tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)
                tensor = tensor.to(device=device, dtype=vae.dtype) / 127.5 - 1.0
            latents = vae.encode(tensor).latent_dist.mode() * scaling_factor
            if not torch.isfinite(latents).all():
                raise RuntimeError("Video VAE encoder returned nonfinite latents")
            latent_batches.append(latents)
    return latent_batches


def _decode_frame_latents(
    latent_batches: Sequence[Any],
    *,
    vae: Any,
    noise_std: float,
    shared_noise: Any,
    scaling_factor: float,
    high_depth: bool = False,
) -> list[np.ndarray]:
    """Decode cached latents with one perturbation shared across time."""
    import torch

    output: list[np.ndarray] = []
    with torch.inference_mode():
        for latents in latent_batches:
            perturbed = latents + noise_std * shared_noise.expand(latents.shape[0], -1, -1, -1)
            decoded = vae.decode(perturbed / scaling_factor).sample
            if not torch.isfinite(decoded).all():
                raise RuntimeError("Video VAE decoder returned nonfinite pixels")
            unit = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
            if high_depth:
                quantized = (unit * 65535.0).round().to(torch.int32).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint16)
            else:
                quantized = (unit * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            output.extend(np.ascontiguousarray(frame[:, :, ::-1]) for frame in quantized)
    return output


def encode_video_frames(
    frames: Iterable[np.ndarray],
    source: Path,
    output: Path,
    *,
    fps: float,
) -> None:
    """Encode regenerated frames, copy audio, and omit source metadata."""
    iterator = iter(frames)
    with ExitStack() as lifetime:
        close = getattr(iterator, "close", None)
        if callable(close):
            lifetime.callback(close)
        first = next(iterator, None)
        if first is None:
            raise ValueError("At least one frame is required for video encoding")
        height, width = first.shape[:2]
        raw_pix_fmt = "bgr48le" if first.dtype == np.uint16 else "bgr24"
        with staged_video_output(output) as (encoded_video, temporary_output):
            process = start_raw_video_encoder(
                raw_video_command(
                    encoded_video,
                    width=width,
                    height=height,
                    fps=fps,
                    crf=18,
                    profile=probe_video_encode_profile(source),
                    input_pix_fmt=raw_pix_fmt,
                )
            )
            frame_pipe = process.stdin
            try:
                for frame in chain((first,), iterator):
                    if frame.shape[:2] != (height, width):
                        raise ValueError("Video frames must have matching dimensions")
                    frame_pipe.write(frame.tobytes())
                finish_raw_video_encoder(process, encoded_video, operation="SynthID removal encode")
                mux_encoded_video(encoded_video, source, temporary_output, strip_metadata=True)
            except Exception:
                abort_raw_video_encoder(process)
                raise


def _upscale_clean_guide(
    native_reference: NDArray[Any],
    small_reference: NDArray[Any],
    guide_small: NDArray[Any],
) -> NDArray[Any]:
    """Color-anchor at guide resolution and upscale cleanly without source HF residual leakage."""
    target_h, target_w = native_reference.shape[:2]
    anchored_small = anchor_low_freq_color(small_reference, guide_small)
    if anchored_small.shape[:2] == (target_h, target_w):
        return anchored_small
    upscaled = cv2.resize(anchored_small, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return unsharp_mask(upscaled, amount=0.35, sigma=1.2)


def regenerate_video_candidate(
    source: Path,
    output: Path,
    *,
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float | None = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    duration: float | None = None,
    runtime: VideoVaeRuntime | None = None,
    temporal_stabilize: bool = True,
    native_guide: bool = False,
    timestamps: Sequence[float] | None = None,
    _frame_stream: Iterable[NDArray[Any]] | None = None,
) -> RegenerationMetrics:
    """Regenerate video pixels and return fidelity metrics.

    The default profile is certified against the provider oracle. This function
    does not perform a per-file SynthID decode because Google exposes no local
    decoder. When ``fps=None``, the source video's native frame rate is preserved.
    When ``native_guide=True``, latent regeneration runs at ``long_side`` and
    upscales cleanly to native ``(width, height)`` with CIE Lab color anchoring.
    """
    if not 0.0 <= noise_std <= 1.0:
        raise ValueError("noise_std must be between 0 and 1")
    if fps is not None and fps < 1.0:
        raise ValueError("fps must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    width, height, source_fps = _probe_video(source)
    size = _fit_size(width, height, long_side)
    out_w, out_h = (width, height) if native_guide else size
    effective_fps = source_fps if fps is None else min(fps, source_fps)
    if out_w < width or out_h < height or effective_fps < source_fps - 0.5:
        log.warning(
            "2D SynthID VAE fallback downscaling %dx%d@%.2ffps -> %dx%d@%.2ffps; "
            "use --native-guide or CUDA Wan 2.1 (wan21-synthid) for native-resolution output",
            width,
            height,
            source_fps,
            out_w,
            out_h,
            effective_fps,
        )

    if runtime is None:
        runtime = load_video_vae_runtime(model=model, device=device)
    elif runtime.model != model or runtime.requested_device != device:
        raise ValueError("The supplied video VAE runtime does not match the requested model and device")
    resolved_device = runtime.resolved_device
    vae = runtime.vae
    if max(size) > 768 and callable(getattr(vae, "enable_tiling", None)):
        vae.enable_tiling()
    elif callable(getattr(vae, "disable_tiling", None)):
        vae.disable_tiling()

    encode_profile = probe_video_encode_profile(source)
    high_depth = getattr(encode_profile, "pixel_format", None) in HIGH_BIT_DEPTH_PIXEL_FORMATS
    comp_depth = getattr(encode_profile, "component_depth", None)
    color_trc = getattr(encode_profile, "color_transfer", None)
    if ((comp_depth or 0) > 8 or color_trc in {"smpte2084", "arib-std-b67"}) and not high_depth:
        source_format = getattr(encode_profile, "source_pixel_format", None) or "unknown pixel format"
        raise RuntimeError(
            "Invisible video regeneration cannot preserve this high-bit-depth or HDR stream without "
            f"refusing to silently reduce {source_format} / "
            f"{color_trc or 'unknown transfer'} to 8-bit SDR"
        )

    raw_pix_fmt = "bgr48le" if high_depth else "bgr24"
    with staged_video_output(output) as (encoded_video, temporary_output):
        process = start_raw_video_encoder(
            raw_video_command(
                encoded_video,
                width=out_w,
                height=out_h,
                fps=effective_fps,
                crf=18,
                profile=encode_profile,
                input_pix_fmt=raw_pix_fmt,
            )
        )
        frame_pipe = process.stdin
        frame_count = 0
        squared_error = 0.0
        pixel_count = 0
        temporal_baseline = 0.0
        temporal_candidate = 0.0
        previous_gray: np.ndarray | None = None
        previous_reference_f32: np.ndarray | None = None
        previous_candidate_f32: np.ndarray | None = None
        shared_noise: Any | None = None
        try:
            sample_kwargs: dict[str, Any] = {
                "source_fps": source_fps,
                "duration": duration,
                "effective_fps": effective_fps,
                "size": (out_w, out_h),
            }
            if high_depth:
                sample_kwargs["high_depth"] = True
                sample_kwargs["source_size"] = (width, height)
            if timestamps is not None:
                sample_kwargs["timestamps"] = timestamps
            if _frame_stream is not None:
                sample_kwargs["_frame_stream"] = _frame_stream
            sampled_frames = _iter_sampled_frames(source, **sample_kwargs)

            def _encode_decode_small(small_frames: Sequence[NDArray[Any]]) -> list[NDArray[Any]]:
                nonlocal shared_noise
                latent_batches = _encode_frame_latents(
                    small_frames,
                    vae=vae,
                    device=resolved_device,
                    batch_size=batch_size,
                    scaling_factor=runtime.scaling_factor,
                )
                latents = latent_batches[0]
                if shared_noise is None:
                    log.info(
                        "First latent batch spread %.4f against noise_std %.4f",
                        float(latents.float().std()),
                        noise_std,
                    )
                    shared_noise = _shared_latent_noise(
                        latents.shape[1:],
                        seed=seed,
                        device=resolved_device,
                        dtype=latents.dtype,
                    )
                decode_kwargs: dict[str, Any] = {
                    "vae": vae,
                    "noise_std": noise_std,
                    "shared_noise": shared_noise,
                    "scaling_factor": runtime.scaling_factor,
                }
                if high_depth:
                    decode_kwargs["high_depth"] = True
                return _decode_frame_latents(latent_batches, **decode_kwargs)

            def _final_pairs() -> Generator[tuple[NDArray[Any], NDArray[Any]], None, None]:
                frame_queue: deque[NDArray[Any]] = deque()

                def _small_pairs() -> Generator[tuple[NDArray[Any], NDArray[Any]], None, None]:
                    for out_batch in _stream_batches(sampled_frames, batch_size):
                        frame_queue.extend(out_batch)
                        small_batch = (
                            [cv2.resize(frm, size, interpolation=cv2.INTER_AREA) for frm in out_batch]
                            if size != (out_w, out_h)
                            else out_batch
                        )
                        yield from zip(small_batch, _encode_decode_small(small_batch), strict=True)

                raw_pairs: Iterable[tuple[NDArray[Any], NDArray[Any]]] = (
                    iter_stabilized_regenerated_frames(_small_pairs()) if temporal_stabilize else _small_pairs()
                )
                for small_ref, small_cand in raw_pairs:
                    out_ref = frame_queue.popleft()
                    yield out_ref, _upscale_clean_guide(out_ref, small_ref, small_cand)

            for reference, candidate in _final_pairs():
                frame_pipe.write(candidate.tobytes())
                reference_f32 = reference.astype(np.float32)
                candidate_f32 = candidate.astype(np.float32)
                difference = reference_f32 - candidate_f32
                squared_error += float(np.sum(difference * difference, dtype=np.float64))
                pixel_count += reference.size
                ref_u8 = (reference >> 8).astype(np.uint8) if reference.dtype == np.uint16 else reference
                current_gray = cv2.cvtColor(ref_u8, cv2.COLOR_BGR2GRAY)
                if (
                    previous_gray is not None
                    and previous_reference_f32 is not None
                    and previous_candidate_f32 is not None
                ):
                    frame_maps = _backward_map(current_gray, previous_gray)
                    temporal_baseline += _motion_residual(reference_f32, previous_reference_f32, frame_maps)
                    temporal_candidate += _motion_residual(candidate_f32, previous_candidate_f32, frame_maps)
                previous_gray = current_gray
                previous_reference_f32 = reference_f32
                previous_candidate_f32 = candidate_f32
                frame_count += 1
            if frame_count < 2:
                raise ValueError("The selected clip produced fewer than two frames")
            finish_raw_video_encoder(
                process,
                encoded_video,
                operation="SynthID removal encode",
            )
            mux_encoded_video(encoded_video, source, temporary_output, strip_metadata=True)
        except Exception:
            abort_raw_video_encoder(process)
            raise

        mse = squared_error / pixel_count
        peak = 65535.0 if high_depth else 255.0
        return RegenerationMetrics(
            frames=frame_count,
            fps=effective_fps,
            width=out_w,
            height=out_h,
            psnr_db=math.inf if mse == 0.0 else 20.0 * math.log10(peak / math.sqrt(mse)),
            temporal_residual_ratio=temporal_candidate / max(temporal_baseline, 1e-6),
        )
