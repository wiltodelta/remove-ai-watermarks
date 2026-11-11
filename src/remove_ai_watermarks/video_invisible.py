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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from remove_ai_watermarks.video_decode import open_video_capture
from remove_ai_watermarks.video_encoding import (
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
    VIDEO_SYNTHID_LATENT_MULTIPLE,
    VIDEO_SYNTHID_VAE_SCALING_FACTOR,
)
from remove_ai_watermarks.video_temporal import (
    _backward_map,
    _motion_residual,
    build_temporal_reference,
    temporal_residual_ratio,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

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


def _pick_device(requested: str) -> str:
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

    resolved_device = _pick_device(device)
    dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    log.info("Loading %s on %s", model, resolved_device)
    vae = AutoencoderKL.from_pretrained(model, torch_dtype=dtype).to(resolved_device)
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
    """Return paired PSNR over uint8 frame stacks."""
    if reference.shape != candidate.shape:
        raise ValueError("PSNR inputs must have matching shapes")
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


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


def _iter_sampled_frames(
    source: Path,
    *,
    source_fps: float,
    duration: float | None,
    effective_fps: float,
    size: tuple[int, int],
) -> Iterable[np.ndarray]:
    """Yield uniformly sampled frames without retaining the full video."""
    capture = open_video_capture(source)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    sample_period = 1.0 / effective_fps
    next_sample_time = 0.0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            if duration is not None and timestamp + 1e-9 >= duration:
                break
            if timestamp + 1e-9 >= next_sample_time:
                yield cv2.resize(frame, size, interpolation=cv2.INTER_LANCZOS4)
                next_sample_time += sample_period
            frame_index += 1
    finally:
        capture.release()


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
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2)
            tensor = tensor.to(device=device, dtype=vae.dtype) / 127.5 - 1.0
            latents = vae.encode(tensor).latent_dist.mode() * scaling_factor
            latent_batches.append(latents)
    return latent_batches


def _decode_frame_latents(
    latent_batches: Sequence[Any],
    *,
    vae: Any,
    noise_std: float,
    shared_noise: Any,
    scaling_factor: float,
) -> list[np.ndarray]:
    """Decode cached latents with one perturbation shared across time."""
    import torch

    output: list[np.ndarray] = []
    with torch.inference_mode():
        for latents in latent_batches:
            perturbed = latents + noise_std * shared_noise.expand(latents.shape[0], -1, -1, -1)
            decoded = vae.decode(perturbed / scaling_factor).sample
            decoded = ((decoded / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
            decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
            output.extend(np.ascontiguousarray(frame[:, :, ::-1]) for frame in decoded)
    return output


def encode_video_frames(
    frames: Sequence[np.ndarray],
    source: Path,
    output: Path,
    *,
    fps: float,
) -> None:
    """Encode regenerated frames, copy audio, and omit source metadata."""
    if not frames:
        raise ValueError("At least one frame is required for video encoding")
    height, width = frames[0].shape[:2]
    with staged_video_output(output) as (encoded_video, temporary_output):
        process = start_raw_video_encoder(
            raw_video_command(
                encoded_video,
                width=width,
                height=height,
                fps=fps,
                crf=18,
                profile=probe_video_encode_profile(source),
            )
        )
        frame_pipe = process.stdin
        try:
            for frame in frames:
                if frame.shape[:2] != (height, width):
                    raise ValueError("Video frames must have matching dimensions")
                frame_pipe.write(frame.tobytes())
            finish_raw_video_encoder(process, encoded_video, operation="SynthID removal encode")
            mux_encoded_video(encoded_video, source, temporary_output, strip_metadata=True)
        except Exception:
            abort_raw_video_encoder(process)
            raise


def regenerate_video_candidate(
    source: Path,
    output: Path,
    *,
    noise_std: float = DEFAULT_VIDEO_SYNTHID_NOISE_STD,
    long_side: int = DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
    fps: float = DEFAULT_VIDEO_SYNTHID_FPS,
    batch_size: int = 4,
    seed: int = 0,
    model: str = DEFAULT_VIDEO_SYNTHID_VAE,
    device: str = "auto",
    duration: float | None = None,
    runtime: VideoVaeRuntime | None = None,
) -> RegenerationMetrics:
    """Regenerate video pixels and return fidelity metrics.

    The default profile is certified against the provider oracle. This function
    does not perform a per-file SynthID decode because Google exposes no local
    decoder.
    """
    if not 0.0 <= noise_std <= 1.0:
        raise ValueError("noise_std must be between 0 and 1")
    if fps < 1.0:
        raise ValueError("fps must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if duration is not None and duration <= 0:
        raise ValueError("duration must be positive")
    if device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be auto, cuda, mps, or cpu")
    width, height, source_fps = _probe_video(source)
    size = _fit_size(width, height, long_side)
    effective_fps = min(fps, source_fps)

    if runtime is None:
        runtime = load_video_vae_runtime(model=model, device=device)
    elif runtime.model != model or runtime.requested_device != device:
        raise ValueError("The supplied video VAE runtime does not match the requested model and device")
    resolved_device = runtime.resolved_device
    vae = runtime.vae

    with staged_video_output(output) as (encoded_video, temporary_output):
        process = start_raw_video_encoder(
            raw_video_command(
                encoded_video,
                width=size[0],
                height=size[1],
                fps=effective_fps,
                crf=18,
                profile=probe_video_encode_profile(source),
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
            sampled_frames = _iter_sampled_frames(
                source,
                source_fps=source_fps,
                duration=duration,
                effective_fps=effective_fps,
                size=size,
            )
            for frames in _stream_batches(sampled_frames, batch_size):
                latent_batches = _encode_frame_latents(
                    frames,
                    vae=vae,
                    device=resolved_device,
                    batch_size=batch_size,
                    scaling_factor=runtime.scaling_factor,
                )
                latents = latent_batches[0]
                if shared_noise is None:
                    # Removal strength is the ratio of the perturbation to this spread,
                    # not noise_std alone: it is the only local quantity that makes two
                    # models' doses comparable.
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
                regenerated = _decode_frame_latents(
                    latent_batches,
                    vae=vae,
                    noise_std=noise_std,
                    shared_noise=shared_noise,
                    scaling_factor=runtime.scaling_factor,
                )
                for reference, candidate in zip(frames, regenerated, strict=True):
                    frame_pipe.write(candidate.tobytes())
                    reference_f32 = reference.astype(np.float32)
                    candidate_f32 = candidate.astype(np.float32)
                    difference = reference_f32 - candidate_f32
                    squared_error += float(np.sum(difference * difference, dtype=np.float64))
                    pixel_count += reference.size
                    current_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
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
        return RegenerationMetrics(
            frames=frame_count,
            fps=effective_fps,
            width=size[0],
            height=size[1],
            psnr_db=math.inf if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse)),
            temporal_residual_ratio=temporal_candidate / max(temporal_baseline, 1e-6),
        )
