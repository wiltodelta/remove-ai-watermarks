"""Internal full-track FocalCodec continuous-feature resynthesis on CPU FP32.

Weights are optional, downloaded separately under Apache-2.0. See
licenses/focalcodec/. The 16 kHz model cannot preserve the source band above
8 kHz, and Vocos STFT synthesis does not preserve carrier waveform phase or
guarantee provider-watermark removal on arbitrary inputs.
"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from remove_ai_watermarks._internal import audio_resampling as src
from remove_ai_watermarks._internal import audio_spectral
from remove_ai_watermarks._internal._focalcodec_network import build_focalcodec
from remove_ai_watermarks._internal.audio_quality import sha256_file
from remove_ai_watermarks.optional_deps import module_available

log = logging.getLogger(__name__)
MODEL_REPOSITORY = "lucadellalib/focalcodec_50hz"
MODEL_REVISION = "d6d6d9524e52155c85193c2c3b8da1cf8842f019"
SOURCE_REVISION = "cad8bf81d70875f756770449c34909d8b0951642"
CHECKPOINT_FILENAME = "model.safetensors"
CHECKPOINT_SHA256 = "2700a916ca8d1c11a899995ef8d451ee53a481486650c4b78cd96feff9ac77f0"
CONFIG_SHA256 = "fc5d9ec56ce795ffeb889e4209b10f8b40a95b9e821662b09d39bf31a4429bd0"
EXPECTED_STATE_KEY_COUNT = 355
WAVLM_CONTEXT_EACH = 40
_MODEL_LOCK = threading.RLock()

_EXPECTED_CONFIG: dict[str, Any] = {
    "encoder_name": "WavLM",
    "encoder_config": {
        "hidden_dims": [512, 512, 512, 512, 512, 512, 512],
        "kernel_sizes": [10, 3, 3, 3, 3, 2, 2],
        "strides": [5, 2, 2, 2, 2, 2, 2],
        "num_layers": 6,
        "dim": 1024,
        "ffn_dim": 4096,
        "num_heads": 16,
        "num_buckets": 320,
        "max_distance": 800,
        "dropout": 0.0,
        "conv_pos": 128,
        "conv_pos_groups": 16,
    },
    "compressor_name": "FocalEncoder",
    "compressor_config": {
        "input_dim": 1024,
        "output_dim": 13,
        "hidden_dims": [1024, 512, 256],
        "downscale_factors": [1, 1, 1],
        "focal_window": 7,
        "focal_level": 2,
        "focal_factor": 2,
        "dropout": 0.0,
        "use_post_norm": False,
        "use_layerscale": False,
        "layerscale_init": 0.0001,
        "normalize_modulator": False,
    },
    "quantizer_name": "BinarySphericalQuantizer",
    "quantizer_config": {"codebook_size": 8192},
    "decompressor_name": "FocalDecoder",
    "decompressor_config": {
        "input_dim": 13,
        "output_dim": 1024,
        "hidden_dims": [256, 512, 1024],
        "upscale_factors": [1, 1, 1],
        "focal_window": 7,
        "focal_level": 2,
        "focal_factor": 2,
        "dropout": 0.0,
        "use_post_norm": False,
        "use_layerscale": False,
        "layerscale_init": 0.0001,
        "normalize_modulator": False,
    },
    "decoder_name": "Vocos",
    "decoder_config": {
        "input_channels": 1024,
        "num_layers": 8,
        "dim": 512,
        "ffn_dim": 1536,
        "kernel_size": 7,
        "padding": 3,
        "layerscale_init": None,
        "n_fft": 1024,
        "hop_length": 320,
    },
}


def is_available() -> bool:
    """Inspect optional dependencies without importing them or fetching weights."""
    return all(module_available(name) for name in ("numpy", "scipy", "torch", "huggingface_hub", "safetensors"))


def _check_config(config: Any) -> None:
    if config != _EXPECTED_CONFIG:
        raise RuntimeError("Pinned FocalCodec configuration does not match the fixed inference profile")


@lru_cache(maxsize=2)
def load_pinned_model(*, local_files_only: bool = True) -> Any:
    """Load the pinned FocalCodec 50 Hz checkpoint on CPU FP32 after hash checks."""
    if not is_available():
        raise ImportError("FocalCodec requires numpy, scipy, torch, huggingface_hub, and safetensors")
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
            raise RuntimeError(f"Pinned FocalCodec checkpoint digest mismatch for {filename}")
        paths[filename] = path
    if paths["config.json"].parent != paths[CHECKPOINT_FILENAME].parent:
        raise RuntimeError("Pinned FocalCodec artifacts must share one snapshot directory")
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    _check_config(config)
    safetensors_torch = import_module("safetensors.torch")
    raw_state = safetensors_torch.load_file(str(paths[CHECKPOINT_FILENAME]), device="cpu")
    if not isinstance(raw_state, dict):
        raise RuntimeError("Pinned FocalCodec checkpoint did not contain a state dictionary")
    state = cast("dict[str, Any]", raw_state)
    if len(state) != EXPECTED_STATE_KEY_COUNT:
        raise RuntimeError("Pinned FocalCodec checkpoint did not contain the expected 355 state keys")
    model = build_focalcodec(config)
    incompatible = model.load_state_dict(state, strict=True)
    if getattr(incompatible, "missing_keys", ()) or getattr(incompatible, "unexpected_keys", ()):
        raise RuntimeError(f"Pinned FocalCodec checkpoint did not load completely: {incompatible}")
    model = model.to("cpu").float().train(False).requires_grad_(False)
    log.info("Loaded pinned FocalCodec %s@%s on CPU (FP32, continuous features)", MODEL_REPOSITORY, MODEL_REVISION)
    return model


def _prepare_input(samples: Any, sample_rate: int) -> tuple[Any, src.RationalClock]:
    np = import_module("numpy")
    value = np.asarray(samples)
    if value.ndim != 2 or not value.shape[0] or value.shape[1] not in (1, 2) or value.dtype.kind != "f":
        raise ValueError("FocalCodec expects a nonempty floating (frames, 1 or 2 channels) matrix")
    clock = src.RationalClock(int(value.shape[0]), sample_rate, model_rate=16000, hop=320, minimum_model_frames=320)
    if not np.isfinite(value).all():
        raise ValueError("FocalCodec input must be finite")
    value_f32 = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value_f32).all():
        raise ValueError("FocalCodec input must be finite in float32")
    return value_f32, clock


def resynthesize(samples: Any, sample_rate: int, *, stereo_mode: str = "ms", local_files_only: bool = True) -> Any:
    """Return full native mono/stereo samples without gain, clipping, or lag repair.

    Uses the 16 kHz model grid (hop 320, minimum 320 samples) with right
    reflection to a complete 50 Hz frame boundary when needed, plus symmetric
    (40, 40) receptive-field context before WavLM feature extraction so every
    320-sample hop frame produces one feature vector without dropping tails.
    Stereo defaults to raw mid/side; each channel runs in its own single-item
    batch so mono and stereo share identical numerical behavior.
    """
    if not is_available():
        raise ImportError("FocalCodec requires numpy, scipy, torch, huggingface_hub, and safetensors")
    if stereo_mode not in {"ms", "lr"}:
        raise ValueError("FocalCodec stereo mode must be ms or lr")
    np = import_module("numpy")
    torch = import_module("torch")
    functional = torch.nn.functional
    value, clock = _prepare_input(samples, sample_rate)
    prepared = src.forward(value, clock)
    mid_side = value.shape[1] == 2 and stereo_mode == "ms"
    if mid_side:
        prepared = np.column_stack(((prepared[:, 0] + prepared[:, 1]) / 2, (prepared[:, 0] - prepared[:, 1]) / 2))
    if not np.isfinite(prepared).all():
        raise ValueError("FocalCodec prepared input must be finite")
    padded = src.pad_model(prepared, clock)
    reconstructed_channels: list[Any] = []
    with _MODEL_LOCK, torch.inference_mode():
        model = load_pinned_model(local_files_only=local_files_only)
        for channel in range(value.shape[1]):
            wave = torch.from_numpy(np.ascontiguousarray(padded[:, channel]))[None]
            context_wave = functional.pad(wave, (WAVLM_CONTEXT_EACH, WAVLM_CONTEXT_EACH))
            output = model.resynthesize_features(context_wave)
            if tuple(output.shape) != (1, clock.padded_frames):
                raise RuntimeError("FocalCodec returned incorrect model sample geometry")
            decoded = output[0].detach().to(device="cpu", dtype=torch.float32).numpy()
            if not np.isfinite(decoded).all():
                raise RuntimeError("FocalCodec returned nonfinite samples")
            reconstructed_channels.append(src.unpad_model(decoded, clock))
    reconstructed = np.column_stack(reconstructed_channels)
    if mid_side:
        reconstructed = np.column_stack(
            (reconstructed[:, 0] + reconstructed[:, 1], reconstructed[:, 0] - reconstructed[:, 1])
        )
    delivered = src.inverse(reconstructed, clock)
    if delivered.shape != value.shape or not np.isfinite(delivered).all():
        raise RuntimeError("FocalCodec returned invalid native sample geometry or values")
    return np.asarray(delivered, dtype=np.float32)


_ENVELOPE_CUTS_HZ: tuple[float, ...] = (200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0)
_ENVELOPE_LOWPASS_HZ: float = 35.0
_ENVELOPE_MAX_GAIN: float = 2.5


def resynthesize_conditioned(
    samples: Any,
    sample_rate: int,
    *,
    stereo_mode: str = "ms",
    local_files_only: bool = True,
) -> Any:
    """Precondition the mid-band mark region, resynthesize, and restore sub-band envelopes.

    1. Applies the calibrated 1.0-1.5 kHz zero-phase band-stop when ``sample_rate``
       supports it so the WavLM encoder does not encode mid-band watermark
       modulation into its continuous 50 Hz features.
    2. Runs full-track FocalCodec continuous-feature resynthesis.
    3. Restores complementary zero-phase Butterworth sub-band 35 Hz envelopes from
       the original waveform (clamped to ``[0.0, 2.5]``), suppressing idle vocoder
       pause hiss while preserving speech formant balance and 10 ms sub-window
       envelope timing.
    """
    if not is_available():
        raise ImportError("FocalCodec requires numpy, scipy, torch, huggingface_hub, and safetensors")
    np = import_module("numpy")
    signal = import_module("scipy.signal")

    source_f32, _ = _prepare_input(samples, sample_rate)

    if sample_rate > int(2 * audio_spectral.BAND_HZ[1]) and source_f32.shape[0] > 1:
        preconditioned = np.concatenate(
            list(audio_spectral.filter_audio_chunks((source_f32,), sample_rate)),
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        preconditioned = source_f32

    import inspect

    resynth_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if "stereo_mode" in inspect.signature(resynthesize).parameters:
        resynth_kwargs["stereo_mode"] = stereo_mode
    raw = np.asarray(resynthesize(preconditioned, sample_rate, **resynth_kwargs), dtype=np.float32)
    if raw.shape != source_f32.shape or not np.isfinite(raw).all():
        raise RuntimeError("FocalCodec returned invalid native sample geometry or values")
    if source_f32.shape[0] <= 2 or sample_rate <= int(2 * _ENVELOPE_LOWPASS_HZ):
        return raw

    x64 = source_f32.astype(np.float64)
    y64 = raw.astype(np.float64)
    valid_cuts = [cut for cut in _ENVELOPE_CUTS_HZ if cut < 0.47 * sample_rate]
    env_sos = np.asarray(
        signal.butter(2, _ENVELOPE_LOWPASS_HZ, btype="lowpass", fs=sample_rate, output="sos"),
        dtype=np.float64,
    )
    pad_env = min(3 * (2 * len(env_sos) + 1), len(x64) - 1)
    x_prev = np.zeros_like(x64)
    y_prev = np.zeros_like(y64)
    out = np.zeros_like(y64)

    for cut in [*valid_cuts, None]:
        if cut is not None:
            sos = np.asarray(signal.butter(4, cut, btype="lowpass", fs=sample_rate, output="sos"), dtype=np.float64)
            pad_lp = min(3 * (2 * len(sos) + 1), len(x64) - 1)
            x_lp = np.asarray(signal.sosfiltfilt(sos, x64, axis=0, padlen=pad_lp), dtype=np.float64)
            y_lp = np.asarray(signal.sosfiltfilt(sos, y64, axis=0, padlen=pad_lp), dtype=np.float64)
            xb = x_lp - x_prev
            yb = y_lp - y_prev
            x_prev = x_lp
            y_prev = y_lp
        else:
            xb = x64 - x_prev
            yb = y64 - y_prev
        xe = np.sqrt(np.maximum(signal.sosfiltfilt(env_sos, xb * xb, axis=0, padlen=pad_env), 0.0) + 1e-14)
        ye = np.sqrt(np.maximum(signal.sosfiltfilt(env_sos, yb * yb, axis=0, padlen=pad_env), 0.0) + 1e-14)
        gain = np.clip(xe / ye, 0.0, _ENVELOPE_MAX_GAIN)
        out += gain * yb

    src_peak = max(abs(float(x64.min())), abs(float(x64.max())))
    out_peak = max(abs(float(out.min())), abs(float(out.max())))
    ceiling = min(max(src_peak, 0.25), 0.96) if src_peak <= 1.0 else src_peak
    if out_peak > ceiling > 0.0:
        knee = 0.85 * ceiling
        headroom = ceiling - knee
        over = (out > knee) | (out < -knee)
        vals = out[over]
        out[over] = np.sign(vals) * (knee + headroom * np.tanh((np.abs(vals) - knee) / headroom))

    delivered_cond = np.asarray(out, dtype=np.float32)
    if delivered_cond.shape != source_f32.shape or not np.isfinite(delivered_cond).all():
        raise RuntimeError("FocalCodec envelope conditioning produced nonfinite samples")
    return delivered_cond
