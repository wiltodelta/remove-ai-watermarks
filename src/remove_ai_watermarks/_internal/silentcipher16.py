"""Experimental SilentCipher-16k receiver for a terminated eight-symbol pattern.

This profile accepts seven ternary payload symbols and one zero terminator.
It is not the published SDK byte API or the paper's general 32-bit profile.
The public API and CLI select this profile explicitly as ``silentcipher16``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal

from remove_ai_watermarks._internal import silentcipher as base

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)
MODEL_FILES = {
    "16_khz/97561_iteration/dec_m_0.ckpt": "bf443c7298f6994284082c0039bf824c70560cd437869bf6f2643b1482fcba71",
    "16_khz/97561_iteration/hparams.yaml": "e4c25258e2789aa30f7e31486df8d9e491eed78c3a2eeed45a59f2a2cb3560f9",
}
SAMPLE_RATE = 16000
WINDOW_SECONDS = 3
MIN_READ_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS
PERIOD = 8
MIN_CYCLES = 3
THRESHOLD = 0.9
HOP_LENGTH = 1024
PHASE_STEP = 10


@dataclass(frozen=True)
class SilentCipher16Reading:
    """Recovered ternary symbols, or a negative for this experimental pattern.

    These symbols are not bytes. Agreement is not a watermark probability;
    the diagnostic maximum includes invalid framing hypotheses on a negative.
    """

    payload_symbols: tuple[int, ...] | None
    symbol_consistency: float
    phase_offset: int | None
    phases_checked: int
    device: str
    framing: Literal["terminated8"] = "terminated8"

    @property
    def detected(self) -> bool:
        return self.payload_symbols is not None


def is_available() -> bool:
    return base.is_available()


@lru_cache(maxsize=3)
def load_model(device: str | None = None) -> base.DecoderModel:
    if not is_available():
        raise ImportError("SilentCipher requires remove-ai-watermarks[audio-silentcipher]")
    return base.load_decoder(
        base.download_pinned_files(MODEL_FILES),
        "16_khz/97561_iteration/dec_m_0.ckpt",
        device,
        message_dim=4,
        message_band_size=512,
        n_fft=2048,
    )


def _decode_sequence(raw: NDArray[Any]) -> tuple[tuple[int, ...] | None, float]:
    import numpy as np

    stats = import_module("scipy.stats")
    scores = np.asarray(raw)
    if scores.ndim != 2 or scores.shape[0] != 4 or not np.isfinite(scores).all():
        raise base.ModelCheckError("SilentCipher-16k returned invalid raw neural values")
    length = scores.shape[1] // PERIOD * PERIOD
    if length < PERIOD * MIN_CYCLES:
        raise base.ModelCheckError("SilentCipher-16k returned too few complete message repetitions")
    predicted = np.argmax(scores, axis=0)[:length].reshape(-1, PERIOD)
    symbols = np.asarray(stats.mode(predicted, axis=0, keepdims=False).mode, dtype=np.int64)
    consistency = float(np.asarray(predicted == symbols[None], dtype=np.float64).mean())
    zeros = np.flatnonzero(symbols == 0)
    if len(zeros) != 1:
        return None, consistency
    end = int(zeros[0])
    letters = np.concatenate([symbols[end + 1 :], symbols[:end]]) - 1
    return tuple(int(value) for value in letters), consistency


def _read_native(model: base.DecoderModel, samples: NDArray[Any]) -> SilentCipher16Reading:
    import numpy as np

    from remove_ai_watermarks._internal._silentcipher_network import magnitude

    torch = import_module("torch")
    power = np.mean(samples**2)
    if not np.isfinite(power) or power <= 0:
        raise base.ModelCheckError("SilentCipher-16k cannot normalize silent or invalid input power")
    values = samples * np.sqrt(0.002837200844477648 / power)
    signal = torch.from_numpy(values.copy()).unsqueeze(0).to(dtype=torch.float32, device=model.device)
    best, checked = 0.0, 0
    with base.MODEL_LOCK, torch.inference_mode():
        for shift in range(0, HOP_LENGTH, PHASE_STEP):
            carrier = magnitude(signal[:, shift:], model.window, n_fft=2048, hop_length=HOP_LENGTH)
            raw = model.decoder(carrier[:, None])
            if tuple(raw.shape[:3]) != (1, 1, 4):
                raise base.ModelCheckError("SilentCipher-16k returned an invalid neural output shape")
            payload, consistency = _decode_sequence(raw[0, 0].detach().cpu().numpy())
            checked += 1
            best = max(best, consistency)
            if payload is not None and consistency >= THRESHOLD:
                return SilentCipher16Reading(payload, consistency, shift, checked, model.device)
    return SilentCipher16Reading(None, best, None, checked, model.device)


def read(model: base.DecoderModel, samples: object, *, sample_rate: int = SAMPLE_RATE) -> SilentCipher16Reading:
    import numpy as np

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or sample_rate <= 0 or len(array) < WINDOW_SECONDS * sample_rate or not np.isfinite(array).all():
        raise ValueError("SilentCipher-16k requires at least three seconds of finite mono audio")
    if sample_rate != SAMPLE_RATE:
        librosa = import_module("librosa")
        array = np.asarray(librosa.resample(array, orig_sr=sample_rate, target_sr=SAMPLE_RATE), dtype=np.float32)
    if len(array) < MIN_READ_SAMPLES or not np.isfinite(array).all():
        raise RuntimeError("SilentCipher-16k resampling returned invalid samples")
    try:
        return _read_native(model, array)
    except base.ModelCheckError:
        raise
    except RuntimeError as exc:
        if model.device == "cpu":
            raise
        log.warning("SilentCipher-16k %s check failed; retrying on CPU: %s", model.device, exc)
        return _read_native(load_model("cpu"), array)
