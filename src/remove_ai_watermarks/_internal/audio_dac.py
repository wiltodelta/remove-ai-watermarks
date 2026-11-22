"""Internal full-track 44.1 kHz DAC codec; no provider-removal or voice guarantee.

Weights are optional, downloaded separately under MIT. See licenses/dac/.
Uses fixed 3-codebook quantization on a 44.1 kHz internal rational sample grid
with raw mid/side stereo decomposition and stem-level RMS preservation.
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
MODEL_REPOSITORY = "descript/dac_44khz"
MODEL_REVISION = "c1bc521685adf9cfe247bc39a5ca58917eda1ac4"
CHECKPOINT_FILENAME = "model.safetensors"
CHECKPOINT_SHA256 = "6128ebff483a41422b0164d079a3773b0d8d82e64c4293d775994cbf8baf913a"
CONFIG_SHA256 = "4eb55fb9af1990b8d608184ad29b70e358589719af7ea8d3c06998f7c2264a64"
FIXED_QUANTIZERS = 3
_SILENT_STEM_RMS = 1e-5
_HEADROOM_CEILING = 0.95
_PEAK_CEILING = 0.999
_MODEL_LOCK = threading.RLock()


def is_available() -> bool:
    """Inspect optional dependencies without importing them or fetching weights."""
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("numpy", "scipy", "torch", "transformers", "huggingface_hub", "safetensors")
    )


def _check_config(config: Any) -> None:
    expected = {
        "model_type": "dac",
        "sampling_rate": 44100,
        "n_codebooks": 9,
        "codebook_size": 1024,
        "hop_length": 512,
    }
    if any(getattr(config, name, None) != value for name, value in expected.items()):
        raise RuntimeError("Pinned DAC configuration does not match the fixed inference profile")


@lru_cache(maxsize=4)
def load_pinned_model(*, local_files_only: bool = True, device: str = "cpu") -> Any:
    """Load built-in Transformers DacModel on the requested device after checking artifact hashes."""
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("DAC device must be cpu, mps, or cuda")
    if not is_available():
        raise ImportError("DAC requires numpy, scipy, torch, transformers, huggingface_hub, and safetensors")
    torch = import_module("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested DAC MPS device is unavailable")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested DAC CUDA device is unavailable")
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
            raise RuntimeError(f"Pinned DAC checkpoint digest mismatch for {filename}")
        paths[filename] = path
    if paths["config.json"].parent != paths[CHECKPOINT_FILENAME].parent:
        raise RuntimeError("Pinned DAC artifacts must share one snapshot directory")
    torch = import_module("torch")
    transformers = import_module("transformers")
    config = transformers.DacConfig.from_pretrained(str(paths["config.json"].parent), local_files_only=True)
    _check_config(config)
    model, loading = transformers.DacModel.from_pretrained(
        str(paths[CHECKPOINT_FILENAME].parent),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        trust_remote_code=False,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Pinned DAC checkpoint did not load completely: {loading}")
    _check_config(model.config)
    log.info(
        "Loaded pinned DAC %s@%s on %s (FP32, %d quantizers)",
        MODEL_REPOSITORY,
        MODEL_REVISION,
        device,
        FIXED_QUANTIZERS,
    )
    return model.to(device).train(False)


def validate_preserve_hf_hz(preserve_hf_hz: float, sample_rate: int) -> None:
    """Validate an explicit high-frequency crossover cutoff for ``sample_rate``."""
    import math

    if not math.isfinite(preserve_hf_hz) or preserve_hf_hz < 8000.0 or preserve_hf_hz > sample_rate * 0.45:
        raise ValueError(f"preserve_hf_hz must be between 8000 and {sample_rate * 0.45:.0f} Hz for {sample_rate} Hz")


@lru_cache(maxsize=16)
def _hf_crossover_sos(preserve_hf_hz: float, sample_rate: int) -> tuple[Any, Any]:
    sig = import_module("scipy.signal")
    return (
        sig.butter(8, preserve_hf_hz, btype="low", fs=sample_rate, output="sos"),
        sig.butter(8, preserve_hf_hz, btype="high", fs=sample_rate, output="sos"),
    )


def blend_hf_crossover(
    processed: Any,
    original: Any,
    sample_rate: int,
    preserve_hf_hz: float,
) -> Any:
    """Preserve original high-frequency transients above ``preserve_hf_hz`` via zero-phase Butterworth crossover."""
    validate_preserve_hf_hz(preserve_hf_hz, sample_rate)
    if processed.shape[0] < 64:
        return processed
    np = import_module("numpy")
    sig = import_module("scipy.signal")
    sos_lp, sos_hp = _hf_crossover_sos(float(preserve_hf_hz), int(sample_rate))
    lp_proc = sig.sosfiltfilt(sos_lp, processed, axis=0)
    lp_orig = sig.sosfiltfilt(sos_lp, original, axis=0)
    in_lp_rms = float(np.sqrt(np.mean(np.square(lp_orig, dtype=np.float64))))
    out_lp_rms = float(np.sqrt(np.mean(np.square(lp_proc, dtype=np.float64))))
    if in_lp_rms >= 1e-6 and out_lp_rms >= 1e-6:
        lp_proc = lp_proc * (in_lp_rms / out_lp_rms)
    out = lp_proc + sig.sosfiltfilt(sos_hp, original, axis=0)
    return np.asarray(out, dtype=np.float32)


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
        raise ImportError("DAC requires the optional numerical and Transformers codec dependencies")
    if stereo_mode not in {"ms", "lr"}:
        raise ValueError("DAC stereo mode must be ms or lr")
    if preserve_hf_hz is not None:
        validate_preserve_hf_hz(preserve_hf_hz, sample_rate)
    np = import_module("numpy")
    torch = import_module("torch")
    value = np.asarray(samples)
    if value.ndim != 2 or not value.shape[0] or value.shape[1] not in (1, 2) or value.dtype.kind != "f":
        raise ValueError("DAC expects a nonempty floating (frames, 1 or 2 channels) matrix")
    clock = src.RationalClock(int(value.shape[0]), sample_rate, model_rate=44100, hop=512, minimum_model_frames=512)
    if not np.isfinite(value).all():
        raise ValueError("DAC input must be finite")
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("DAC input must be finite in float32")
    mid_side = value.shape[1] == 2 and stereo_mode == "ms"
    stems_native = (
        np.column_stack(((value[:, 0] + value[:, 1]) / 2, (value[:, 0] - value[:, 1]) / 2)) if mid_side else value
    )
    prepared = src.pad_model(src.forward(stems_native, clock), clock)
    if not np.isfinite(prepared).all():
        raise ValueError("DAC prepared input must be finite")

    output_channels: list[Any] = []
    with _MODEL_LOCK, torch.inference_mode():
        model = (
            load_pinned_model(local_files_only=local_files_only)
            if device == "cpu"
            else load_pinned_model(local_files_only=local_files_only, device=device)
        )
        for channel in range(prepared.shape[1]):
            native_stem = stems_native[:, channel]
            in_rms = float(np.sqrt(np.mean(np.square(native_stem, dtype=np.float64))))
            if in_rms < _SILENT_STEM_RMS:
                output_channels.append(np.zeros_like(native_stem, dtype=np.float32))
                continue
            model_stem = prepared[:, channel]
            peak_in = float(np.max(np.abs(model_stem)))
            scale = min(32.0, 0.50 / max(peak_in, 1e-6)) if peak_in > 1e-6 else 1.0
            dac_in = (
                torch.from_numpy(np.ascontiguousarray(model_stem * scale, dtype=np.float32)).view(1, 1, -1).to(device)
            )
            encoded = model.encode(dac_in, n_quantizers=FIXED_QUANTIZERS)
            decoded_tensor = model.decode(audio_codes=encoded.audio_codes).audio_values
            decoded = decoded_tensor.reshape(-1).detach().to(device="cpu", dtype=torch.float32).numpy() / scale
            if decoded.shape[0] < clock.padded_frames:
                raise RuntimeError("DAC returned fewer samples than expected for the padded model grid")
            decoded = decoded[: clock.padded_frames]
            if not np.isfinite(decoded).all():
                raise RuntimeError("DAC returned nonfinite samples")
            restored = src.inverse(src.unpad_model(decoded, clock), clock)
            out_rms = float(np.sqrt(np.mean(np.square(restored, dtype=np.float64))))
            if out_rms > 1e-6:
                restored = restored * (in_rms / out_rms)
            output_channels.append(np.asarray(restored, dtype=np.float32))

    delivered = np.column_stack(output_channels)
    if mid_side:
        delivered = np.column_stack((delivered[:, 0] + delivered[:, 1], delivered[:, 0] - delivered[:, 1]))
    if preserve_hf_hz is not None:
        delivered = blend_hf_crossover(delivered, value, sample_rate, preserve_hf_hz)
    peak = float(np.max(np.abs(delivered)))
    if peak > _PEAK_CEILING:
        delivered = delivered * (_PEAK_CEILING / peak)
    delivered = np.asarray(delivered, dtype=np.float32)
    if delivered.shape != value.shape or not np.isfinite(delivered).all():
        raise RuntimeError("DAC returned invalid native sample geometry or values")
    return delivered
