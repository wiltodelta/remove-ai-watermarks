"""Optional pinned EnCodec resynthesis, preserving native audio geometry."""

from __future__ import annotations

import logging
import threading
from functools import lru_cache
from importlib import import_module
from math import gcd
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from remove_ai_watermarks._internal.audio_chunks import transform_chunks
from remove_ai_watermarks._internal.audio_quality import sha256_file
from remove_ai_watermarks._internal.constants import AUDIO_BANDWIDTHS_KBPS
from remove_ai_watermarks.optional_deps import module_available

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from numpy.typing import NDArray

log = logging.getLogger(__name__)
MODEL_REPOSITORY = "facebook/encodec_24khz"
MODEL_REVISION = "c1dbe2ae3f1de713481a3b3e7c47f357092ee040"
CHECKPOINT_FILENAME = "model.safetensors"
CHECKPOINT_SHA256 = "37a7cb100f71a29e6c1d815aca8666a1d7ea8885ebe44c306c751a5103559d57"
CONFIG_SHA256 = "cd7da7da3b3b34daee262e213e3daa24dc01b4fdc3aadd38ed760f0d6f955871"
BANDWIDTHS_KBPS = AUDIO_BANDWIDTHS_KBPS
CONTEXT_SECONDS = 1
_MODEL_LOCK = threading.RLock()


def is_available() -> bool:
    """Whether the optional codec stack is installed, without downloading weights."""
    return all(
        module_available(name) for name in ("numpy", "scipy", "torch", "transformers", "huggingface_hub", "safetensors")
    )


@lru_cache(maxsize=1)
def load_pinned_model() -> Any:
    """Load the CPU model only after validating the full official checkpoint hash."""
    if not is_available():
        raise ImportError("Neural audio processing requires remove-ai-watermarks[audio-codec]")
    hub = import_module("huggingface_hub")
    checkpoint: Path | None = None
    for filename, expected in (("config.json", CONFIG_SHA256), (CHECKPOINT_FILENAME, CHECKPOINT_SHA256)):
        path = Path(hub.hf_hub_download(repo_id=MODEL_REPOSITORY, filename=filename, revision=MODEL_REVISION))
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"Pinned EnCodec checkpoint digest mismatch for {filename}: {digest}")
        if filename == CHECKPOINT_FILENAME:
            checkpoint = path
    if checkpoint is None:
        raise RuntimeError("Pinned EnCodec checkpoint was not downloaded")
    torch = import_module("torch")
    transformers = import_module("transformers")
    log.info("Loading pinned EnCodec model: repository=%s revision=%s", MODEL_REPOSITORY, MODEL_REVISION)
    model, loading = transformers.EncodecModel.from_pretrained(
        str(checkpoint.parent),
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Pinned EnCodec checkpoint did not load completely: {loading}")
    return model.to("cpu").train(False)


def resynthesize(
    samples: NDArray[Any],
    sample_rate: int,
    bandwidth_kbps: float,
    *,
    stereo_mode: Literal["independent", "ms"] = "independent",
) -> NDArray[Any]:
    """Resynthesize channels independently or in Mid/Side stereo, trimming only padding."""
    import numpy as np

    if not is_available():
        raise ImportError("Neural audio processing requires remove-ai-watermarks[audio-codec]")
    if bandwidth_kbps not in BANDWIDTHS_KBPS or sample_rate <= 0:
        raise ValueError("EnCodec requires a positive sample rate and bandwidth 6, 12, or 24 kbps")
    if stereo_mode not in {"independent", "ms"}:
        raise ValueError("EnCodec stereo_mode must be independent or ms")
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError("EnCodec requires finite nonempty sample frames and channels")
    use_ms = stereo_mode == "ms" and array.shape[1] == 2
    if use_ms:
        mid = 0.5 * (array[:, 0] + array[:, 1])
        side = 0.5 * (array[:, 0] - array[:, 1])
        codec_in = np.column_stack((mid, side)).astype(np.float32, copy=False)
    else:
        codec_in = array
    torch = import_module("torch")
    signal = import_module("scipy.signal")

    def resample(values: NDArray[Any], source_rate: int, target_rate: int) -> NDArray[Any]:
        if source_rate == target_rate:
            return values
        factor = gcd(source_rate, target_rate)
        return np.asarray(
            signal.resample_poly(values, target_rate // factor, source_rate // factor, axis=0), dtype=np.float32
        )

    with _MODEL_LOCK, torch.inference_mode():
        model = load_pinned_model()
        native = resample(codec_in, sample_rate, 24000)
        inputs = torch.from_numpy(np.ascontiguousarray(native.T)).unsqueeze(1)
        reconstructed = model(inputs, bandwidth=bandwidth_kbps).audio_values.squeeze(1).cpu().numpy().T
        delivered = resample(reconstructed, 24000, sample_rate)
    if delivered.shape[0] < len(array) or delivered.shape[1] != array.shape[1]:
        raise RuntimeError("EnCodec returned insufficient samples or changed channels")
    trimmed = np.asarray(delivered[: len(array)], dtype=np.float32)
    if use_ms:
        result = np.column_stack((trimmed[:, 0] + trimmed[:, 1], trimmed[:, 0] - trimmed[:, 1])).astype(
            np.float32, copy=False
        )
    else:
        result = trimmed
    if not np.isfinite(result).all():
        raise RuntimeError("EnCodec returned nonfinite samples")
    return result


def resynthesize_chunks(
    chunks: Iterable[NDArray[Any]],
    sample_rate: int,
    *,
    bandwidth_kbps: float = 24.0,
    stereo_mode: Literal["independent", "ms"] = "independent",
) -> Iterator[NDArray[Any]]:
    """Use bounded neighboring context; preserve central samples and channel order."""
    if bandwidth_kbps not in BANDWIDTHS_KBPS or sample_rate <= 0:
        raise ValueError("EnCodec requires a positive sample rate and bandwidth 6, 12, or 24 kbps")
    if stereo_mode not in {"independent", "ms"}:
        raise ValueError("EnCodec stereo_mode must be independent or ms")

    def process(samples: NDArray[Any]) -> NDArray[Any]:
        return resynthesize(samples, sample_rate, bandwidth_kbps, stereo_mode=stereo_mode)

    yield from transform_chunks(chunks, process, context_frames=sample_rate * CONTEXT_SECONDS)
