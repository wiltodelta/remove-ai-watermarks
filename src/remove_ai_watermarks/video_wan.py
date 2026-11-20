"""Experimental CUDA 3D Wan 2.1 video diffusion profiles for SynthID and Seedance.

Uses pinned Apache-2.0 ``Wan-AI/Wan2.1-T2V-1.3B-Diffusers`` video-to-video
regeneration with symmetric 16-pixel reflection padding, tail-frame causal 3D
alignment, source PTS preservation, and stream-copied audio. Visible overlays
must be handled separately before this pixel stage. No runtime detector verdict
is produced.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from remove_ai_watermarks._internal.constants import WAN_SUPPORTED_EXTENSIONS
from remove_ai_watermarks._internal.two_stage_pipeline import (
    _load_prompt_payload,
    _prompt_cache_path,
    _store_prompt_payload,
)
from remove_ai_watermarks.optional_deps import module_available
from remove_ai_watermarks.video_encoding import (
    atomic_video_output,
    probe_video_encode_profile,
    probe_video_timestamps,
)
from remove_ai_watermarks.video_seedance import (
    SeedanceQuality,
    _clock,
    _encode_frames,
    _frames,
    _measure,
    validate_experimental_mp4_source,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from remove_ai_watermarks._internal.constants import WanExperimentalProfile
    from remove_ai_watermarks.video_invisible import RegenerationMetrics

log = logging.getLogger(__name__)

WAN21_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN21_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
WAN21_FLOW_SHIFT = 3.0
WAN21_INFERENCE_STEPS = 20
WAN21_GUIDANCE_SCALE = 1.0
WAN21_SYNTHID_STRENGTH = 0.38
WAN21_SEEDANCE_STRENGTH = 0.12
WAN21_SPATIAL_MULTIPLE = 16
WAN21_DEFAULT_CHUNK_FRAMES = 193
WAN21_COMPACT_CHUNK_FRAMES = 81
WAN21_DEFAULT_OVERLAP_FRAMES = 17
WAN21_AUTO_MAX_FRAME_PIXELS = 1920 * 1080
WAN21_MAX_FRAME_PIXELS = 1920 * 1080
WAN21_MAX_TOTAL_PIXEL_FRAMES = 7200 * 1920 * 1080
WAN21_MIN_CUDA_VRAM_BYTES = 8 * 1024**3
WAN21_HIGH_VRAM_BYTES = 16 * 1024**3
WAN21_DEFAULT_PROMPT = "High fidelity natural video, sharp details, accurate motion, clean cinematic lighting."
WAN21_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, compression artifacts, flicker, warped edges, text, logo, watermark"
)
_WAN21_PROMPT_OUTPUT_PARAMS = ("wan21_v2v_prompt_embeds_v1",)

_PROFILE_STRENGTHS: dict[WanExperimentalProfile, float] = {
    "wan21-synthid-experimental": WAN21_SYNTHID_STRENGTH,
    "wan21-seedance-experimental": WAN21_SEEDANCE_STRENGTH,
}


def profile_strength(profile: WanExperimentalProfile) -> float:
    """Return the calibrated flow-matching denoising strength for a Wan 2.1 profile."""
    return _PROFILE_STRENGTHS[profile]


@dataclass(frozen=True)
class VideoWanRuntime:
    """Loaded Wan 2.1 video-to-video pipeline reusable across video runs."""

    model: str
    revision: str
    resolved_device: str
    pipe: Any
    prompt_embeds: Any = None


def is_available() -> bool:
    """Return whether the optional Wan 2.1 diffusion dependencies are installed."""
    return module_available(
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "sentencepiece",
        "ftfy",
    )


def resolve_wan_device(requested: str) -> str:
    """Resolve native Ampere-or-newer NVIDIA BF16 CUDA or Apple Silicon MPS for 3D Wan 2.1."""
    if requested not in {"auto", "cuda", "mps"}:
        raise ValueError("Wan 2.1 3D video diffusion requires device='auto', 'cuda', or 'mps'")
    import torch

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Wan 2.1 3D video diffusion requested MPS, but Apple Silicon MPS is unavailable")
        return "mps"
    if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        raise RuntimeError("Wan 2.1 3D video diffusion requires an NVIDIA CUDA GPU (or explicit device='mps')")
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Wan 2.1 3D video diffusion requires a BF16-capable NVIDIA CUDA GPU (Ampere or newer)")
    return "cuda"


def _wan_prompt_cache_path() -> Path:
    """Return the atomic disk cache path for the static Wan 2.1 default prompt embedding."""
    return _prompt_cache_path(
        f"{WAN21_MODEL_ID}@{WAN21_MODEL_REVISION}",
        _WAN21_PROMPT_OUTPUT_PARAMS,
        WAN21_DEFAULT_PROMPT,
    )


def _cuda_vram_below(limit_bytes: int) -> bool:
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        dev_idx = torch.cuda.current_device()
    except Exception:
        dev_idx = 0
    try:
        props = torch.cuda.get_device_properties(dev_idx)
        return int(getattr(props, "total_memory", limit_bytes)) < limit_bytes
    except Exception:
        return False


def _effective_chunk_frames(
    chunk_frames: int,
    overlap_frames: int,
    *,
    width: int,
    height: int,
    resolved_device: str,
) -> int:
    """Step down the default 193-frame window to 81 frames on 1080p, MPS, or <16 GiB CUDA."""
    if chunk_frames != WAN21_DEFAULT_CHUNK_FRAMES or overlap_frames >= WAN21_COMPACT_CHUNK_FRAMES:
        return chunk_frames
    if width * height > 1280 * 720 or resolved_device == "mps" or _cuda_vram_below(WAN21_HIGH_VRAM_BYTES):
        return WAN21_COMPACT_CHUNK_FRAMES
    return chunk_frames


def load_video_wan_runtime(*, device: str = "auto", local_files_only: bool = False) -> VideoWanRuntime:
    """Load pinned Wan 2.1 1.3B V2V, serving cached static prompt embeddings without UMT5-XXL when present."""
    if not is_available():
        raise RuntimeError("Wan 2.1 video processing requires 'remove-ai-watermarks[video,diffusion]'")
    resolved = resolve_wan_device(device)
    import gc

    import torch
    from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.pipelines.wan.pipeline_wan_video2video import WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    dit_dtype = torch.float16 if resolved == "mps" else torch.bfloat16
    vae = AutoencoderKLWan.from_pretrained(
        WAN21_MODEL_ID,
        subfolder="vae",
        revision=WAN21_MODEL_REVISION,
        torch_dtype=torch.float32,
        local_files_only=local_files_only,
    )
    vae.enable_tiling()
    vae.enable_slicing()

    prompt_embeds = None
    cache_path = _wan_prompt_cache_path() if WAN21_GUIDANCE_SCALE <= 1.0 else None
    if cache_path is not None and cache_path.is_file():
        try:
            prompt_embeds = _load_prompt_payload(cache_path, resolved, dit_dtype)
        except Exception as exc:
            log.warning("Ignoring unreadable Wan 2.1 prompt embedding cache %s (%s)", cache_path, exc)
            prompt_embeds = None

    pipe_kwargs: dict[str, Any] = {
        "revision": WAN21_MODEL_REVISION,
        "vae": vae,
        "torch_dtype": dit_dtype,
        "local_files_only": local_files_only,
    }
    if prompt_embeds is not None:
        pipe_kwargs["text_encoder"] = None
        pipe_kwargs["tokenizer"] = None

    pipe = WanVideoToVideoPipeline.from_pretrained(WAN21_MODEL_ID, **pipe_kwargs)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=WAN21_FLOW_SHIFT,
    )
    # Cold cache miss: encode static prompt on CPU once, persist atomically, and drop UMT5-XXL before GPU/MPS move.
    if prompt_embeds is None and hasattr(pipe, "encode_prompt") and WAN21_GUIDANCE_SCALE <= 1.0:
        with torch.inference_mode():
            cpu_embeds, _ = pipe.encode_prompt(
                prompt=WAN21_DEFAULT_PROMPT,
                device=torch.device("cpu"),
                num_videos_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            if cache_path is not None:
                try:
                    _store_prompt_payload(cache_path, cpu_embeds)
                except OSError as exc:
                    log.warning("Could not persist Wan 2.1 prompt embedding cache %s (%s)", cache_path, exc)
            prompt_embeds = cpu_embeds.to(device=resolved, dtype=dit_dtype)
        if hasattr(pipe, "text_encoder"):
            pipe.text_encoder = None
        if hasattr(pipe, "tokenizer"):
            pipe.tokenizer = None
        gc.collect()
    pipe.to(resolved)
    pipe.set_progress_bar_config(disable=True)
    if resolved == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif resolved == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    log.info("Loaded pinned Wan 2.1 %s@%s on %s", WAN21_MODEL_ID, WAN21_MODEL_REVISION, resolved)
    return VideoWanRuntime(
        model=WAN21_MODEL_ID,
        revision=WAN21_MODEL_REVISION,
        resolved_device=resolved,
        pipe=pipe,
        prompt_embeds=prompt_embeds,
    )


def _validate_wan_window_args(chunk_frames: int, overlap_frames: int) -> None:
    if chunk_frames < 17 or (chunk_frames - 1) % 4 != 0:
        raise ValueError("chunk_frames must be a 4k+1 integer >= 17")
    if overlap_frames < 0 or overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must satisfy 0 <= overlap_frames < chunk_frames")


def _validate_wan_source_budget(width: int, height: int, frame_count: int) -> None:
    """Enforce conservative spatial and total-frame limits before decoding."""
    frame_pixels = width * height
    if frame_pixels > WAN21_MAX_FRAME_PIXELS:
        raise ValueError(
            f"Experimental Wan 2.1 processing requires frame area <= {WAN21_MAX_FRAME_PIXELS} pixels "
            f"(got {width}x{height})"
        )
    if frame_pixels * frame_count > WAN21_MAX_TOTAL_PIXEL_FRAMES:
        raise ValueError(
            f"Experimental Wan 2.1 host-memory budget exceeded: {frame_count} frames at {width}x{height} "
            f"exceeds {WAN21_MAX_TOTAL_PIXEL_FRAMES} pixel-frames"
        )


def can_run_wan_on_source(
    source: Path,
    *,
    device: str,
    runtime: Any = None,
    for_auto: bool = False,
) -> bool:
    """Return whether source, device, VRAM, and runtime satisfy Wan 2.1 eligibility."""
    if source.suffix.lower() not in WAN_SUPPORTED_EXTENSIONS or device not in {"auto", "cuda", "mps"}:
        return False
    if for_auto and device == "mps":
        return False
    if runtime is not None:
        if not isinstance(runtime, VideoWanRuntime):
            return False
        if runtime.model != WAN21_MODEL_ID or runtime.revision != WAN21_MODEL_REVISION:
            return False
    else:
        if not is_available():
            return False
        try:
            resolved = resolve_wan_device(device)
            if for_auto and resolved == "cuda" and _cuda_vram_below(WAN21_MIN_CUDA_VRAM_BYTES):
                return False
        except Exception:
            return False
    try:
        width, height, _fps, timestamps = validate_experimental_mp4_source(
            source,
            label="Experimental Wan 2.1 processing",
            encode_profile=probe_video_encode_profile(source),
        )
    except Exception:
        return False
    if for_auto and width * height > WAN21_AUTO_MAX_FRAME_PIXELS:
        return False
    try:
        _validate_wan_source_budget(width, height, len(timestamps))
    except ValueError as exc:
        log.warning("Wan 2.1 eligibility skipped for %s (%s); auto profile will use 2D fallback", source.name, exc)
        return False
    return True


def _run_wan_window(
    bgr_window: list[np.ndarray],
    runtime: VideoWanRuntime,
    *,
    strength: float,
    seed: int,
    orig_h: int,
    orig_w: int,
    inference_steps: int = WAN21_INFERENCE_STEPS,
) -> np.ndarray:
    """Pad spatial/temporal axes on one window, run 3D diffusion, and crop back to BGR."""
    import torch

    win_t = len(bgr_window)
    pad_h = (WAN21_SPATIAL_MULTIPLE - (orig_h % WAN21_SPATIAL_MULTIPLE)) % WAN21_SPATIAL_MULTIPLE
    pad_w = (WAN21_SPATIAL_MULTIPLE - (orig_w % WAN21_SPATIAL_MULTIPLE)) % WAN21_SPATIAL_MULTIPLE
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    target_h, target_w = orig_h + pad_h, orig_w + pad_w

    pil_frames: list[Image.Image] = []
    for frame_bgr in bgr_window:
        rgb = frame_bgr[..., ::-1]
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((top, bottom), (left, right), (0, 0)), mode="reflect")
        pil_frames.append(Image.fromarray(np.ascontiguousarray(rgb), mode="RGB"))

    rem = (len(pil_frames) - 1) % 4
    pad_t = (4 - rem) % 4
    if pad_t:
        pil_frames.extend([pil_frames[-1]] * pad_t)

    gen_device = "cpu" if runtime.resolved_device == "mps" else runtime.resolved_device
    generator = torch.Generator(device=gen_device).manual_seed(seed)
    call_kwargs: dict[str, Any] = {
        "video": pil_frames,
        "height": target_h,
        "width": target_w,
        "num_inference_steps": inference_steps,
        "guidance_scale": WAN21_GUIDANCE_SCALE,
        "strength": strength,
        "generator": generator,
        "output_type": "np",
    }
    if runtime.prompt_embeds is not None:
        call_kwargs["prompt_embeds"] = runtime.prompt_embeds
    else:
        call_kwargs["prompt"] = WAN21_DEFAULT_PROMPT
        call_kwargs["negative_prompt"] = WAN21_DEFAULT_NEGATIVE_PROMPT
    with torch.inference_mode():
        generated = runtime.pipe(**call_kwargs).frames[0]

    raw_rgb = np.asarray(generated)[:win_t]
    if raw_rgb.dtype != np.uint8:
        raw_rgb = np.clip(np.rint(raw_rgb * 255.0), 0, 255).astype(np.uint8)
    cropped_rgb = raw_rgb[:, top : top + orig_h, left : left + orig_w]
    out_bgr = cropped_rgb[..., ::-1].copy()
    if out_bgr.shape != (win_t, orig_h, orig_w, 3):
        raise RuntimeError(f"Wan 2.1 output geometry {out_bgr.shape} did not match {(win_t, orig_h, orig_w, 3)}")
    return out_bgr


def _stream_wan_windows(
    frames_bgr: Iterable[np.ndarray],
    runtime: VideoWanRuntime,
    *,
    strength: float,
    seed: int,
    chunk_frames: int = WAN21_DEFAULT_CHUNK_FRAMES,
    overlap_frames: int = WAN21_DEFAULT_OVERLAP_FRAMES,
    inference_steps: int = WAN21_INFERENCE_STEPS,
) -> Iterator[np.ndarray]:
    """Stream 3D causal windows while retaining only one active window and the overlap tail."""
    _validate_wan_window_args(chunk_frames, overlap_frames)
    from itertools import islice

    import torch

    it = iter(frames_bgr)
    step = chunk_frames - overlap_frames
    fade_in = (
        (0.5 - 0.5 * np.cos(np.pi * np.arange(1, overlap_frames + 1, dtype=np.float32) / (overlap_frames + 1)))[
            :, None, None, None
        ]
        if overlap_frames > 0
        else np.empty((0, 1, 1, 1), dtype=np.float32)
    )
    fade_out = 1.0 - fade_in

    try:
        peek: np.ndarray | None = next(it, None)
        if peek is None:
            return
        orig_h, orig_w = int(peek.shape[0]), int(peek.shape[1])
        in_tail: list[np.ndarray] = []
        out_tail: np.ndarray | None = None
        window_index = 0
        while True:
            if window_index > 0 and runtime.resolved_device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            need = chunk_frames if window_index == 0 else step
            in_buf = [*in_tail, peek, *islice(it, need - 1)]
            next_peek = next(it, None)
            curr_bgr = _run_wan_window(
                in_buf,
                runtime,
                strength=strength,
                seed=seed + window_index,
                orig_h=orig_h,
                orig_w=orig_w,
                inference_steps=inference_steps,
            )
            if out_tail is not None and overlap_frames > 0:
                blended = (
                    out_tail.astype(np.float32) * fade_out + curr_bgr[:overlap_frames].astype(np.float32) * fade_in
                )
                curr_bgr[:overlap_frames] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
            if next_peek is None:
                yield from curr_bgr
                return
            yield from curr_bgr[:step]
            in_tail = in_buf[step:] if overlap_frames > 0 else []
            out_tail = curr_bgr[step:].copy() if overlap_frames > 0 else None
            del in_buf, curr_bgr
            peek = next_peek
            window_index += 1
    finally:
        close = getattr(it, "close", None)
        if callable(close):
            close()


def pad_and_run_wan(
    frames_bgr: list[np.ndarray],
    runtime: VideoWanRuntime,
    *,
    strength: float,
    seed: int,
    chunk_frames: int = WAN21_DEFAULT_CHUNK_FRAMES,
    overlap_frames: int = WAN21_DEFAULT_OVERLAP_FRAMES,
    inference_steps: int = WAN21_INFERENCE_STEPS,
) -> np.ndarray:
    """Pad spatial/temporal axes for 3D causal diffusion and restore exact geometry."""
    return np.stack(
        list(
            _stream_wan_windows(
                frames_bgr,
                runtime,
                strength=strength,
                seed=seed,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
                inference_steps=inference_steps,
            )
        ),
        axis=0,
    )


_pad_and_run_wan = pad_and_run_wan


def _finalize_wan_output(
    frames: Iterable[np.ndarray],
    source: Path,
    staged: Path,
    *,
    quality: SeedanceQuality,
    fps: float,
    timestamps: tuple[float, ...],
) -> RegenerationMetrics:
    """Encode Wan 2.1 frames directly to delivery (CRF 18) or master (CRF 0) in a single pass."""
    target_crf = 18 if quality == "delivery" else 0
    _encode_frames(frames, source, staged, fps, timestamps, crf=target_crf)
    if probe_video_timestamps(staged) != timestamps:
        raise RuntimeError("Experimental Wan 2.1 output timestamps changed")
    staged_dur = float(_clock(staged)["streams"][0]["duration"])
    source_dur = float(_clock(source)["streams"][0]["duration"])
    if abs(staged_dur - source_dur) > 0.000002:
        raise RuntimeError("Experimental Wan 2.1 output duration changed")
    return _measure(source, staged, fps, len(timestamps))


def regenerate_wan_experimental(
    source: Path,
    output: Path,
    *,
    profile: WanExperimentalProfile = "wan21-synthid-experimental",
    device: str = "auto",
    quality: SeedanceQuality = "delivery",
    seed: int = 0,
    chunk_frames: int = WAN21_DEFAULT_CHUNK_FRAMES,
    overlap_frames: int = WAN21_DEFAULT_OVERLAP_FRAMES,
    inference_steps: int | None = None,
    runtime: VideoWanRuntime | None = None,
    _frame_stream: Iterable[np.ndarray] | None = None,
) -> RegenerationMetrics:
    """Run pinned 3D Wan 2.1 video-to-video regeneration and validate timestamps."""
    if profile not in _PROFILE_STRENGTHS:
        raise ValueError("profile must be wan21-synthid-experimental or wan21-seedance-experimental")
    if quality not in {"delivery", "master"}:
        raise ValueError("quality must be delivery or master")
    if device not in {"auto", "cuda", "mps"}:
        raise ValueError("Wan 2.1 3D video diffusion requires device='auto', 'cuda', or 'mps'")
    steps = WAN21_INFERENCE_STEPS if inference_steps is None else inference_steps
    if steps < 1:
        raise ValueError("inference_steps must be >= 1")
    _validate_wan_window_args(chunk_frames, overlap_frames)
    if runtime is not None and (runtime.model != WAN21_MODEL_ID or runtime.revision != WAN21_MODEL_REVISION):
        raise ValueError("The supplied Wan 2.1 runtime does not match the pinned model and revision")

    width, height, fps, timestamps = validate_experimental_mp4_source(
        source,
        label="Experimental Wan 2.1 processing",
        encode_profile=probe_video_encode_profile(source),
    )
    _validate_wan_source_budget(width, height, len(timestamps))
    if runtime is None and not is_available():
        raise RuntimeError("Wan 2.1 video processing requires 'remove-ai-watermarks[video,diffusion]'")
    if runtime is None:
        resolve_wan_device(device)

    strength = profile_strength(profile)
    log.warning(
        "Applying experimental %s (strength=%.2f, %s); no per-file oracle verification is performed",
        profile,
        strength,
        quality,
    )
    with atomic_video_output(output) as staged:
        active_runtime = runtime or load_video_wan_runtime(device=device)
        eff_chunk = _effective_chunk_frames(
            chunk_frames,
            overlap_frames,
            width=width,
            height=height,
            resolved_device=active_runtime.resolved_device,
        )
        decoded_count = 0

        def counted_source() -> Iterator[np.ndarray]:
            nonlocal decoded_count
            raw_iter = _frame_stream if _frame_stream is not None else _frames(source)
            for frame in raw_iter:
                decoded_count += 1
                yield frame
            if decoded_count != len(timestamps):
                raise RuntimeError("Decoded frame count did not match probed source frame timestamps")

        streamed_bgr = _stream_wan_windows(
            counted_source(),
            active_runtime,
            strength=strength,
            seed=seed,
            chunk_frames=eff_chunk,
            overlap_frames=overlap_frames,
            inference_steps=steps,
        )
        if runtime is None:
            del active_runtime
        return _finalize_wan_output(
            streamed_bgr,
            source,
            staged,
            quality=quality,
            fps=fps,
            timestamps=timestamps,
        )
