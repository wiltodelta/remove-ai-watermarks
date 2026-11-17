"""Internal full-track Mimi codec; no provider-removal or voice-identity guarantee.

Weights are optional, downloaded separately under CC-BY-4.0. See licenses/mimi/.
The 24 kHz model cannot preserve the source band above 12 kHz.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

from remove_ai_watermarks._internal import audio_resampling as src
from remove_ai_watermarks._internal.audio_quality import sha256_file

log = logging.getLogger(__name__)
MODEL_REPOSITORY = "kyutai/mimi"
MODEL_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"
CHECKPOINT_FILENAME = "model.safetensors"
CHECKPOINT_SHA256 = "bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e"
CONFIG_SHA256 = "aca6f44b04f7bc2e7466b71597d2d51e463ed1cf3cd7025d8848595580546c36"
_MODEL_LOCK = threading.RLock()


def is_available() -> bool:
    """Inspect optional dependencies without importing them or fetching weights."""
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("numpy", "scipy", "torch", "transformers", "huggingface_hub", "safetensors")
    )


def _check_config(config: Any) -> None:
    expected = {
        "model_type": "mimi",
        "sampling_rate": 24000,
        "audio_channels": 1,
        "num_quantizers": 32,
        "use_cache": False,
        "normalize": False,
        "use_causal_conv": True,
        "trim_right_ratio": 1.0,
    }
    if any(getattr(config, name, None) != value for name, value in expected.items()):
        raise RuntimeError("Pinned Mimi configuration does not match the fixed inference profile")


@lru_cache(maxsize=2)
def load_pinned_model(*, local_files_only: bool = True) -> Any:
    """Load built-in Transformers Mimi on CPU after checking both artifact hashes."""
    if not is_available():
        raise ImportError("Mimi requires numpy, scipy, torch, transformers, huggingface_hub, and safetensors")
    hub = import_module("huggingface_hub")
    paths: dict[str, Path] = {}
    for filename, expected in (("config.json", CONFIG_SHA256), (CHECKPOINT_FILENAME, CHECKPOINT_SHA256)):
        path = Path(
            hub.hf_hub_download(
                repo_id=MODEL_REPOSITORY,
                filename=filename,
                revision=MODEL_REVISION,
                local_files_only=local_files_only,
            )
        )
        if sha256_file(path) != expected:
            raise RuntimeError(f"Pinned Mimi checkpoint digest mismatch for {filename}")
        paths[filename] = path
    if paths["config.json"].parent != paths[CHECKPOINT_FILENAME].parent:
        raise RuntimeError("Pinned Mimi artifacts must share one snapshot directory")
    torch = import_module("torch")
    transformers = import_module("transformers")
    config = transformers.MimiConfig.from_pretrained(str(paths["config.json"].parent), local_files_only=True)
    _check_config(config)
    model, loading = transformers.MimiModel.from_pretrained(
        str(paths[CHECKPOINT_FILENAME].parent),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Pinned Mimi checkpoint did not load completely: {loading}")
    _check_config(model.config)
    log.info("Loaded pinned Mimi %s@%s on CPU (FP32, 32 quantizers)", MODEL_REPOSITORY, MODEL_REVISION)
    return model.to("cpu").train(False)


def resynthesize(samples: Any, sample_rate: int, *, stereo_mode: str = "ms", local_files_only: bool = True) -> Any:
    """Return full native mono/stereo samples without gain, clipping, or lag repair.

    Stereo defaults to raw mid/side; independent L/R is an internal comparison.
    The official forward removes its own decoder's right excess using the input
    mask. No external padding or chunking is applied. Exact counts are not proof
    of learned phase preservation, speaker identity, or watermark removal.
    """
    if not is_available():
        raise ImportError("Mimi requires the optional numerical and Transformers codec dependencies")
    if stereo_mode not in {"ms", "lr"}:
        raise ValueError("Mimi stereo mode must be ms or lr")
    np = import_module("numpy")
    torch = import_module("torch")
    value = np.asarray(samples)
    if value.ndim != 2 or not value.shape[0] or value.shape[1] not in (1, 2) or value.dtype.kind != "f":
        raise ValueError("Mimi expects a nonempty floating (frames, 1 or 2 channels) matrix")
    clock = src.RationalClock(int(value.shape[0]), sample_rate, model_rate=24000, hop=1, minimum_model_frames=1)
    if not np.isfinite(value).all():
        raise ValueError("Mimi input must be finite")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Mimi input must be finite in float32")
    prepared = src.forward(value, clock)
    mid_side = value.shape[1] == 2 and stereo_mode == "ms"
    if mid_side:
        prepared = np.column_stack(((prepared[:, 0] + prepared[:, 1]) / 2, (prepared[:, 0] - prepared[:, 1]) / 2))
    if not np.isfinite(prepared).all():
        raise ValueError("Mimi prepared input must be finite")
    with _MODEL_LOCK, torch.inference_mode():
        model = load_pinned_model(local_files_only=local_files_only)
        inputs = torch.from_numpy(np.ascontiguousarray(prepared.T)).unsqueeze(1)
        mask = torch.ones_like(inputs, dtype=torch.bool)
        output = model(inputs, padding_mask=mask).audio_values
        if tuple(output.shape) != (value.shape[1], 1, clock.model_frames):
            raise RuntimeError("Mimi returned incorrect model sample geometry")
        reconstructed = output[:, 0, :].cpu().numpy().T
    if not np.isfinite(reconstructed).all():
        raise RuntimeError("Mimi returned nonfinite samples")
    if mid_side:
        reconstructed = np.column_stack(
            (reconstructed[:, 0] + reconstructed[:, 1], reconstructed[:, 0] - reconstructed[:, 1])
        )
    delivered = src.inverse(reconstructed, clock)
    if delivered.shape != value.shape or not np.isfinite(delivered).all():
        raise RuntimeError("Mimi returned invalid native sample geometry or values")
    return delivered
