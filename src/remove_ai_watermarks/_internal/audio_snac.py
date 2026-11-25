"""Internal full-track 44.1 kHz Multi-Scale Neural Audio Codec (SNAC) profile.

Weights are optional, downloaded separately under MIT. See licenses/snac/.
Uses fixed 3-codebook multi-scale quantization (strides 8, 4, 2; dropping the
finest stride-1 residual codebook that retains phase/magnitude watermarks) on a
44.1 kHz internal rational sample grid with raw mid/side stereo decomposition
and stem-level RMS preservation.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import threading
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from remove_ai_watermarks._internal import audio_resampling as src
from remove_ai_watermarks._internal._snac_network import build_snac
from remove_ai_watermarks._internal.audio_quality import sha256_file

log = logging.getLogger(__name__)
MODEL_REPOSITORY = "hubertsiuzdak/snac_44khz"
MODEL_REVISION = "873ebef9718b89660340c6f55a2b515e98cfa1d9"
CHECKPOINT_FILENAME = "pytorch_model.bin"
CHECKPOINT_SHA256 = "b0a676cbdc8d1cc53186f6d777bc956fb7932ceacdc657a4c3741646e9e7ead0"
CONFIG_SHA256 = "4c98973dd6b4278f6352cedbadc9922a2fa3e4b97619e1769e304e2836c099df"
EXPECTED_STATE_KEY_COUNT = 286
FIXED_QUANTIZERS = 3
HOP_LENGTH = 3072
_SILENT_STEM_RMS = 1e-5
_HEADROOM_CEILING = 0.95
_PEAK_CEILING = 0.999
_MODEL_LOCK = threading.RLock()

_EXPECTED_CONFIG: dict[str, Any] = {
    "sampling_rate": 44100,
    "encoder_dim": 64,
    "encoder_rates": [2, 3, 8, 8],
    "decoder_dim": 1536,
    "decoder_rates": [8, 8, 3, 2],
    "attn_window_size": 32,
    "codebook_size": 4096,
    "codebook_dim": 8,
    "vq_strides": [8, 4, 2, 1],
    "noise": True,
    "depthwise": True,
}


def is_available() -> bool:
    """Inspect optional dependencies without importing them or fetching weights."""
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "torch", "huggingface_hub"))


def _check_config(config: Any) -> None:
    if config != _EXPECTED_CONFIG:
        raise RuntimeError("Pinned SNAC configuration does not match the fixed inference profile")


_MODEL_CACHE: dict[str, Any] = {}


def _clear_model_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


@lru_cache(maxsize=4)
def _load_by_device(device: str, local_files_only: bool) -> Any:
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("SNAC device must be cpu, mps, or cuda")
    if not is_available():
        raise ImportError("SNAC requires numpy, scipy, torch, and huggingface_hub")
    with _MODEL_LOCK:
        if device in _MODEL_CACHE:
            return _MODEL_CACHE[device]
    torch = import_module("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested SNAC MPS device is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested SNAC CUDA device is unavailable")
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
            raise RuntimeError(f"Pinned SNAC checkpoint digest mismatch for {filename}")
        paths[filename] = path
    if paths["config.json"].parent != paths[CHECKPOINT_FILENAME].parent:
        raise RuntimeError("Pinned SNAC artifacts must share one snapshot directory")
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    _check_config(config)
    raw_state: Any = torch.load(paths[CHECKPOINT_FILENAME], map_location="cpu", weights_only=True)
    if not isinstance(raw_state, dict):
        raise RuntimeError("Pinned SNAC checkpoint state must be a dictionary")
    state_dict = cast("dict[str, Any]", raw_state)
    if len(state_dict) != EXPECTED_STATE_KEY_COUNT:
        raise RuntimeError("Pinned SNAC checkpoint state key count did not match expected schema")
    model = build_snac(config)
    model.load_state_dict(state_dict, strict=True)
    model.fuse_weight_norm()
    model.train(False).requires_grad_(False).float().to(device)
    log.info(
        "Loaded pinned SNAC %s@%s on %s (FP32, %d quantizers)",
        MODEL_REPOSITORY,
        MODEL_REVISION,
        device,
        FIXED_QUANTIZERS,
    )
    with _MODEL_LOCK:
        _MODEL_CACHE[device] = model
    return model


def load_pinned_model(*, local_files_only: bool = True, device: str = "cpu") -> Any:
    """Load the pinned 44.1 kHz SNAC checkpoint on the requested device after verifying digests."""
    with _MODEL_LOCK:
        if device in _MODEL_CACHE:
            return _MODEL_CACHE[device]
    return _load_by_device(device, local_files_only)


load_pinned_model.cache_clear = lambda: (_clear_model_cache(), _load_by_device.cache_clear())  # type: ignore[attr-defined]


def clear_model_cache() -> None:
    """Release cached SNAC models from CPU/accelerator memory."""
    load_pinned_model.cache_clear()  # type: ignore[attr-defined]


def resynthesize(
    samples: Any,
    sample_rate: int,
    *,
    stereo_mode: str = "ms",
    local_files_only: bool = True,
    device: str = "cpu",
    preserve_hf_hz: float | None = None,
) -> Any:
    """Resynthesize a finite mono or stereo signal on its exact native sample grid."""
    if not is_available():
        raise ImportError("SNAC requires numpy, scipy, torch, and huggingface_hub")
    if stereo_mode not in {"ms", "lr"}:
        raise ValueError("SNAC stereo mode must be ms or lr")
    if preserve_hf_hz is not None:
        from remove_ai_watermarks._internal.audio_dac import validate_preserve_hf_hz

        validate_preserve_hf_hz(preserve_hf_hz, sample_rate)
    np = import_module("numpy")
    torch = import_module("torch")
    value = np.asarray(samples)
    if value.ndim != 2 or not value.shape[0] or value.shape[1] not in (1, 2) or value.dtype.kind != "f":
        raise ValueError("SNAC expects a nonempty floating (frames, 1 or 2 channels) matrix")
    clock = src.RationalClock(
        int(value.shape[0]),
        sample_rate,
        model_rate=44100,
        hop=HOP_LENGTH,
        minimum_model_frames=HOP_LENGTH,
    )
    if not np.isfinite(value).all():
        raise ValueError("SNAC input must be finite")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("SNAC input must be finite in float32")
    mid_side = value.shape[1] == 2 and stereo_mode == "ms"
    stems_native = (
        np.column_stack(((value[:, 0] + value[:, 1]) / 2, (value[:, 0] - value[:, 1]) / 2)) if mid_side else value
    )
    stem_rms = [
        float(np.sqrt(np.mean(np.square(stems_native[:, ch], dtype=np.float64)))) for ch in range(stems_native.shape[1])
    ]
    active_indices = [ch for ch, rms in enumerate(stem_rms) if rms >= _SILENT_STEM_RMS]
    if not active_indices:
        return np.zeros_like(value, dtype=np.float32)

    prepared_active = src.pad_model(src.forward(stems_native[:, active_indices], clock), clock)
    if not np.isfinite(prepared_active).all():
        raise ValueError("SNAC prepared input must be finite")

    decoded_by_channel: dict[int, Any] = {}
    with _MODEL_LOCK, torch.inference_mode():
        model = load_pinned_model(local_files_only=local_files_only, device=device)
        for pos, channel in enumerate(active_indices):
            model_stem = prepared_active[:, pos]
            peak_in = float(np.max(np.abs(model_stem)))
            scale = min(32.0, 0.50 / max(peak_in, 1e-6)) if peak_in > 1e-6 else 1.0
            snac_in = (
                torch.from_numpy(np.ascontiguousarray(model_stem * scale, dtype=np.float32)).view(1, 1, -1).to(device)
            )
            decoded_tensor = model.reconstruct(snac_in, n_quantizers=FIXED_QUANTIZERS)
            decoded = decoded_tensor.reshape(-1).detach().to(device="cpu", dtype=torch.float32).numpy() / scale
            if decoded.shape[0] < clock.padded_frames:
                raise RuntimeError("SNAC returned fewer samples than expected for the padded model grid")
            decoded = decoded[: clock.padded_frames]
            if not np.isfinite(decoded).all():
                raise RuntimeError("SNAC returned nonfinite samples")
            decoded_by_channel[channel] = decoded

    output_channels: list[Any] = []
    for channel in range(stems_native.shape[1]):
        if channel not in decoded_by_channel:
            output_channels.append(np.zeros_like(stems_native[:, channel], dtype=np.float32))
            continue
        restored = src.inverse(src.unpad_model(decoded_by_channel[channel], clock), clock)
        in_rms = stem_rms[channel]
        out_rms = float(np.sqrt(np.mean(np.square(restored, dtype=np.float64))))
        if out_rms > 1e-6:
            restored = restored * (in_rms / out_rms)
        output_channels.append(np.asarray(restored, dtype=np.float32))

    delivered = np.column_stack(output_channels)
    if mid_side:
        delivered = np.column_stack((delivered[:, 0] + delivered[:, 1], delivered[:, 0] - delivered[:, 1]))
    if preserve_hf_hz is not None:
        from remove_ai_watermarks._internal.audio_dac import blend_hf_crossover

        delivered = blend_hf_crossover(delivered, value, sample_rate, preserve_hf_hz)
    peak = float(np.max(np.abs(delivered)))
    if peak > _PEAK_CEILING:
        delivered = delivered * (_PEAK_CEILING / peak)
    delivered = np.asarray(delivered, dtype=np.float32)
    if delivered.shape != value.shape or not np.isfinite(delivered).all():
        raise RuntimeError("SNAC returned invalid native sample geometry or values")
    return delivered
