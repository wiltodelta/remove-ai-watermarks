"""Experimental pinned SilentCipher-44k receiver with exhaustive phase checks.

The receiver profile uses three complete message repetitions and at least 0.9
symbol agreement. These are project receiver criteria, not SDK defaults.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)
MODEL_REPOSITORY = "sony/silentcipher"
MODEL_REVISION = "a1c4d021905e0dc5b24be5f68db5fc4dba410ee1"
MODEL_FILES = {
    "44_1_khz/73999_iteration/dec_m_0.ckpt": "829540c058270d29788f05294894d45bf436e44add0e1099422242ebb94b7088",
    "44_1_khz/73999_iteration/hparams.yaml": "27735fd4db0c00c29fef7af872ad1e5efe854e48d530a86c75beac2b87f09ce3",
}
SAMPLE_RATE = 44_100
WINDOW_SECONDS = 3
MIN_READ_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
PERIOD = 21
MIN_CYCLES = 3
THRESHOLD = 0.9
HOP_LENGTH = 2048
PHASE_STEP = 10
MODEL_LOCK = RLock()


@dataclass(frozen=True)
class SilentCipherReading:
    """Accepted payload or a complete negative under this receiver profile.

    Symbol consistency is agreement, not a watermark probability. On a negative
    it reports the largest agreement, including invalid framing hypotheses.
    """

    payload: tuple[int, ...] | None
    symbol_consistency: float
    phase_offset: int | None
    phases_checked: int
    device: str

    @property
    def detected(self) -> bool:
        return self.payload is not None


class ModelCheckError(RuntimeError):
    """Invalid model numerics or framing must remain unverified."""


@dataclass(frozen=True)
class DecoderModel:
    decoder: Any
    window: Any
    device: str


def is_available() -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in ("numpy", "scipy", "torch", "librosa", "huggingface_hub")
    )


def _download_files() -> dict[str, Path]:
    return download_pinned_files(MODEL_FILES)


def download_pinned_files(files: dict[str, str]) -> dict[str, Path]:
    hub = import_module("huggingface_hub")
    result: dict[str, Path] = {}
    for name, expected in files.items():
        path = Path(hub.hf_hub_download(repo_id=MODEL_REPOSITORY, filename=name, revision=MODEL_REVISION, token=False))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"SilentCipher model file differs from its pinned digest: {name}")
        result[name] = path
    return result


@lru_cache(maxsize=3)
def load_model(device: str | None = None) -> DecoderModel:
    if not is_available():
        raise ImportError("SilentCipher requires remove-ai-watermarks[audio-silentcipher]")
    return load_decoder(_download_files(), "44_1_khz/73999_iteration/dec_m_0.ckpt", device)


def clear_model_cache() -> None:
    """Release cached SilentCipher decoders."""
    with MODEL_LOCK:
        load_model.cache_clear()


def load_decoder(
    paths: dict[str, Path],
    checkpoint: str,
    device: str | None,
    *,
    message_dim: int = 5,
    message_band_size: int = 1024,
    n_fft: int = 4096,
) -> DecoderModel:
    from remove_ai_watermarks._internal._silentcipher_network import build_decoder

    torch = import_module("torch")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if device not in {"cpu", "mps", "cuda"}:
        raise ValueError("Unsupported SilentCipher inference device")
    with MODEL_LOCK:
        raw_state: object = torch.load(paths[checkpoint], map_location="cpu", weights_only=True)
        if not isinstance(raw_state, dict):
            raise RuntimeError("SilentCipher checkpoint is not a parameter dictionary")
        entries = cast("dict[object, object]", raw_state)
        state: dict[str, Any] = {}
        for key, value in entries.items():
            if not isinstance(key, str):
                raise RuntimeError("SilentCipher checkpoint parameter name is invalid")
            normalized = key.removeprefix("module.")
            if normalized in state:
                raise RuntimeError("SilentCipher checkpoint contains duplicate parameter names")
            state[normalized] = value
        with torch.random.fork_rng(devices=[]):
            decoder = build_decoder(message_dim=message_dim, message_band_size=message_band_size)
        decoder.load_state_dict(state, strict=True)
        # The released receiver uses training-mode batch normalization. Keep its
        # per-input statistics, serialize calls, and disable gradients locally.
        decoder = decoder.to(device)
        window = torch.hann_window(n_fft, dtype=torch.float32, device="cpu")
    log.info(
        "Loaded pinned SilentCipher decoder: revision=%s checkpoint=%s device=%s", MODEL_REVISION, checkpoint, device
    )
    return DecoderModel(decoder, window, device)


def _decode_sequence(raw: NDArray[Any]) -> tuple[tuple[int, ...] | None, float]:
    import numpy as np

    stats = import_module("scipy.stats")

    scores = np.asarray(raw)
    if scores.ndim != 2 or scores.shape[0] != 5 or not np.isfinite(scores).all():
        raise ModelCheckError("SilentCipher returned invalid raw neural values")
    length = scores.shape[1] // PERIOD * PERIOD
    if length < PERIOD * MIN_CYCLES:
        raise ModelCheckError("SilentCipher returned too few complete message repetitions")
    predicted = np.argmax(scores, axis=0)[:length].reshape(-1, PERIOD)
    symbols = np.asarray(stats.mode(predicted, axis=0, keepdims=False).mode, dtype=np.int64)
    consistency = float(np.asarray(predicted == symbols[None], dtype=np.float64).mean())
    zeros = np.flatnonzero(symbols == 0)
    if len(zeros) != 1:
        return None, consistency
    end = int(zeros[0])
    letters = np.concatenate([symbols[end + 1 :], symbols[:end]]) - 1
    if len(letters) != 20 or not np.isin(letters, [0, 1, 2, 3]).all():
        return None, consistency
    bits = "".join(format(int(n), "02b") for n in letters)
    return tuple(int(bits[i : i + 8], 2) for i in range(0, 40, 8)), consistency


def _read_native(model: DecoderModel, samples: NDArray[Any]) -> SilentCipherReading:
    import numpy as np

    from remove_ai_watermarks._internal._silentcipher_network import magnitude

    torch = import_module("torch")
    power = np.mean(samples**2)
    if not np.isfinite(power) or power <= 0:
        raise ModelCheckError("SilentCipher cannot normalize silent or invalid input power")
    values = samples * np.sqrt(0.002837200844477648 / power)
    signal = torch.from_numpy(values.copy()).unsqueeze(0).to(dtype=torch.float32, device=model.device)
    best = 0.0
    checked = 0
    with MODEL_LOCK, torch.inference_mode():
        for shift in range(0, HOP_LENGTH, PHASE_STEP):
            carrier = magnitude(signal[:, shift:], model.window)
            raw = model.decoder(carrier[:, None])
            if tuple(raw.shape[:3]) != (1, 1, 5):
                raise ModelCheckError("SilentCipher returned an invalid neural output shape")
            payload, consistency = _decode_sequence(raw[0, 0].detach().cpu().numpy())
            checked += 1
            best = max(best, consistency)
            if payload is not None and consistency >= THRESHOLD:
                return SilentCipherReading(payload, consistency, shift, checked, model.device)
    return SilentCipherReading(None, best, None, checked, model.device)


def read(model: DecoderModel, samples: object, *, sample_rate: int = SAMPLE_RATE) -> SilentCipherReading:
    import numpy as np

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or sample_rate <= 0 or len(array) < WINDOW_SECONDS * sample_rate or not np.isfinite(array).all():
        raise ValueError("SilentCipher requires at least three seconds of finite mono audio")
    if sample_rate != SAMPLE_RATE:
        librosa = import_module("librosa")
        array = np.asarray(librosa.resample(array, orig_sr=sample_rate, target_sr=SAMPLE_RATE), dtype=np.float32)
    if len(array) < MIN_READ_SAMPLES or not np.isfinite(array).all():
        raise RuntimeError("SilentCipher resampling returned invalid samples")
    try:
        return _read_native(model, array)
    except ModelCheckError:
        raise
    except RuntimeError as exc:
        if model.device == "cpu":
            raise
        log.warning("SilentCipher %s check failed; retrying on CPU: %s", model.device, exc)
        return _read_native(load_model("cpu"), array)
