"""Optional pinned BigVGAN full-track FP32 resynthesis, with exact native clocks.

This is a signal transform, not a watermark detector or a removal guarantee.
Stereo uses raw mid/side; mono uses one independent model pass. Other channel
layouts require separate qualification. No normalization, additional output clipping,
alignment, streaming, or automatic device fallback is performed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from remove_ai_watermarks._internal import audio_resampling as src
from remove_ai_watermarks._internal._bigvgan_network import build_generator, mel_spectrogram

log = logging.getLogger(__name__)
MODEL_REPOSITORY = "nvidia/bigvgan_v2_44khz_128band_512x"
MODEL_REVISION = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb"
SOURCE_REVISION = "7d2b454564a6c7d014227f635b7423881f14bdac"
CHECKPOINT_FILENAME = "bigvgan_generator.pt"
CHECKPOINT_SHA256 = "d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357"
CONFIG_SHA256 = "65b7f487bfaf15256056a75f6667ef5f640214c981922a111927873705ea4ddb"


def is_available() -> bool:
    """Check optional dependencies without importing numerical runtimes."""
    return all(
        importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "torch", "librosa", "huggingface_hub")
    )


def _verify_digest(path: Path, expected: str, label: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"BigVGAN {label} digest mismatch: expected {expected}, got {actual}")


@lru_cache(maxsize=1)
def load_pinned_model(device: str = "cpu", *, local_files_only: bool = True) -> Any:
    """Load only verified config/weights, with no downloaded Python execution.

    Cached artifacts are required by default. Explicit local_files_only=False
    permits Hugging Face's pinned artifact download, never remote source code.
    The cached in-memory model is reused without moving it between devices.
    """
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("BigVGAN device must be cpu, mps, or cuda")
    if not is_available():
        raise RuntimeError("BigVGAN requires numpy, scipy, torch, librosa, and huggingface_hub")
    torch = import_module("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested BigVGAN MPS device is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested BigVGAN CUDA device is unavailable")
    hub = import_module("huggingface_hub")
    paths: dict[str, Path] = {}
    for filename, digest in (("config.json", CONFIG_SHA256), (CHECKPOINT_FILENAME, CHECKPOINT_SHA256)):
        path = Path(
            hub.hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=filename,
                revision=MODEL_REVISION,
                local_files_only=local_files_only,
            )
        )
        _verify_digest(path, digest, filename)
        paths[filename] = path
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    model = build_generator(config)
    payload = torch.load(paths[CHECKPOINT_FILENAME], map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "generator" not in payload:
        raise RuntimeError("BigVGAN checkpoint has no generator state")
    model.load_state_dict(payload["generator"], strict=True)
    model.remove_weight_norm()
    model.train(False).requires_grad_(False).float().to(device)
    log.info("Loaded pinned BigVGAN %s@%s on %s (FP32)", MODEL_REPOSITORY, MODEL_REVISION, device)
    return model


def resynthesize(
    samples: Any,
    sample_rate: int,
    *,
    device: str = "cpu",
    local_files_only: bool = True,
) -> Any:
    """Resynthesize a finite floating (frames, 1|2) matrix on its native grid.

    Full arrays are intentional: chunk boundaries require separate audio and
    watermark qualification. Memory exhaustion fails rather than switching
    device or silently chunking. FP32 MPS is numerically, not bitwise, CPU-like.
    The generator retains its trained architecture's own final [-1, 1] clamp;
    reconstructed L/R may exceed that range and is returned without clipping.
    """
    if not is_available():
        raise RuntimeError("BigVGAN requires numpy, scipy, torch, librosa, and huggingface_hub")
    np = import_module("numpy")
    torch = import_module("torch")
    value = np.asarray(samples)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] not in (1, 2) or value.dtype.kind != "f":
        raise ValueError("BigVGAN expects a nonempty floating (frames, 1 or 2 channels) matrix")
    clock = src.RationalClock(int(value.shape[0]), sample_rate)
    if not np.isfinite(value).all():
        raise ValueError("BigVGAN input must be finite")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("BigVGAN input must be finite in float32")
    if value.shape[1] == 2:
        value = np.column_stack(((value[:, 0] + value[:, 1]) / 2, (value[:, 0] - value[:, 1]) / 2))
    prepared = src.pad_model(src.forward(value, clock), clock)
    model = load_pinned_model(device, local_files_only=local_files_only)
    output_channels: list[Any] = []
    with torch.inference_mode():
        for channel in range(value.shape[1]):
            mel = mel_spectrogram(np.ascontiguousarray(prepared[:, channel]))
            output = model(mel.to(device=device, dtype=torch.float32))
            if tuple(output.shape) != (1, 1, clock.padded_frames):
                raise RuntimeError(f"BigVGAN returned unexpected shape {tuple(output.shape)}")
            decoded = output[0, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(decoded).all():
                raise RuntimeError("BigVGAN returned nonfinite samples")
            output_channels.append(src.inverse(src.unpad_model(decoded, clock), clock))
    result = np.column_stack(output_channels)
    if result.shape[1] == 2:
        mid, side = result[:, 0], result[:, 1]
        result = np.column_stack((mid + side, mid - side))
    if result.shape != (clock.native_frames, value.shape[1]) or not np.isfinite(result).all():
        raise RuntimeError("BigVGAN returned invalid native-grid output")
    return np.asarray(result, dtype=np.float32)
