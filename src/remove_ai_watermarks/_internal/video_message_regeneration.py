"""Pinned VAE graph for the experimental public-message video profiles."""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from remove_ai_watermarks._internal.audio_quality import sha256_file
from remove_ai_watermarks.optional_deps import module_available
from remove_ai_watermarks.video_synthid import (
    DEFAULT_VIDEO_SYNTHID_VAE,
    DEFAULT_VIDEO_SYNTHID_VAE_REVISION,
    VIDEO_SYNTHID_VAE_SCALING_FACTOR,
)

if TYPE_CHECKING:
    from remove_ai_watermarks.video_invisible import VideoVaeRuntime

log = logging.getLogger(__name__)

MODEL_ID = DEFAULT_VIDEO_SYNTHID_VAE
REVISION = DEFAULT_VIDEO_SYNTHID_VAE_REVISION
FILES = {
    "config.json": "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e",
    "diffusion_pytorch_model.safetensors": "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
}


def is_available() -> bool:
    return all(module_available(name) for name in ("torch", "diffusers", "huggingface_hub", "cv2"))


@lru_cache(maxsize=4)
def _load_resolved_runtime(resolved: str) -> VideoVaeRuntime:
    from remove_ai_watermarks.video_invisible import VideoVaeRuntime

    torch = import_module("torch")
    diffusers = import_module("diffusers")
    hub = import_module("huggingface_hub")
    files: dict[str, Path] = {}
    for name, expected in FILES.items():
        log.info("Video VAE request: repo=%s revision=%s file=%s anonymous=True", MODEL_ID, REVISION, name)
        path = Path(hub.hf_hub_download(repo_id=MODEL_ID, filename=name, revision=REVISION, token=False))
        digest = sha256_file(path)
        log.info("Video VAE response saved: path=%s sha256=%s", path, digest)
        if digest != expected:
            raise RuntimeError(f"Video VAE file differs from its pinned digest: {name}")
        files[name] = path
    with torch.random.fork_rng(devices=[]):
        vae, loading = diffusers.AutoencoderKL.from_pretrained(
            str(files["config.json"].parent),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.float32,
            output_loading_info=True,
        )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError("Video VAE checkpoint did not load completely")
    vae = vae.to(resolved)
    vae.train(False)
    vae.enable_slicing()
    factor = float(vae.config.scaling_factor)
    if factor != VIDEO_SYNTHID_VAE_SCALING_FACTOR:
        raise RuntimeError("Video VAE latent scaling differs from the checked profile")
    return VideoVaeRuntime(MODEL_ID, resolved, resolved, vae, factor)


def load_runtime(device: str = "auto") -> VideoVaeRuntime:
    """Load verified safetensors at float32; leave the legacy SynthID profile unchanged."""
    if not is_available():
        raise ImportError("Video VAE processing requires remove-ai-watermarks[video,diffusion]")
    from remove_ai_watermarks.video_invisible import pick_device

    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Video VAE device must be auto, cpu, cuda, or mps")
    resolved = pick_device(device)
    runtime = _load_resolved_runtime(resolved)
    return runtime if device == resolved else replace(runtime, requested_device=device)


load_runtime.cache_clear = _load_resolved_runtime.cache_clear  # type: ignore[attr-defined]
