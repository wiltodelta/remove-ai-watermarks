"""Pinned public Perth v1 decoder, separate from production Perth deployments."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000
MIN_READ_SAMPLES = 16_000
PACKAGE_NAME = "resemble-perth"
PACKAGE_VERSION = "1.0.1"
CHECKPOINT_RELATIVE_PATH = Path("perth_net/pretrained/implicit/perth_net_250000.pth.tar")
CHECKPOINT_SHA256 = "a15bce457ebc53ce5e6c9c3f11df78cf7ee2bf9cdab0a798902135b4c4027670"
SOURCE_TREE_SHA256 = "8b1f4c8b6b0f4e0193b8ffa235435b44277c60957f82a1e8a7a91f15a16c444b"
_MODEL_FILES = {
    "hparams.yaml": "c66e8f8ce5b09d100e719843a064d37a97c5aec13510824343bf4e096b3184db",
    "id.txt": "f4129d0cce1fcd76a01c778dd46aeecc84130e38d83c98402abf2e1b9c49770d",
    "perth_net_250000.pth.tar": CHECKPOINT_SHA256,
}
_MODEL_LOCK = RLock()


@dataclass(frozen=True)
class PerthReading:
    """Implicit watermark presence, without a user-selected payload."""

    confidence: float

    @property
    def detected(self) -> bool:
        return self.confidence > 0.5


def _validated_package() -> Path:
    spec = importlib.util.find_spec("perth")
    if spec is None or spec.origin is None:
        raise ImportError("Perth requires remove-ai-watermarks[audio-perth]")
    if importlib.metadata.version("resemble-perth") != "1.0.1":
        raise RuntimeError("Perth requires resemble-perth==1.0.1")
    package = Path(spec.origin).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    if digest.hexdigest() != SOURCE_TREE_SHA256:
        raise RuntimeError("Installed Perth source differs from the pinned public model")
    directory = package / "perth_net/pretrained/implicit"
    if {p.name for p in directory.glob("*.pth.tar")} != {"perth_net_250000.pth.tar"}:
        raise RuntimeError("Perth checkpoint selection differs from the pinned model")
    for name, expected in _MODEL_FILES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Perth model file differs from its pinned digest: {name}")
    return package


def import_failure() -> Exception | None:
    """Verify package code, config, and checkpoint before importing neural code."""
    try:
        _validated_package()
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        return exc
    return None


def is_available() -> bool:
    return import_failure() is None


@lru_cache(maxsize=1)
def load_model() -> object:
    failure = import_failure()
    if failure is not None:
        raise RuntimeError("Perth requires its pinned audio-perth installation") from failure
    diagnostics = io.StringIO()
    with _MODEL_LOCK, contextlib.redirect_stdout(diagnostics):
        package = _validated_package()
        name = "remove_ai_watermarks._internal._perth_v1"
        namespace = ModuleType(name)
        namespace.__path__ = [str(package / "perth_net/perth_net_implicit")]
        namespace.__package__ = name
        namespace.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        sys.modules[name] = namespace
        implementation = import_module(name + ".model.perth_net")
        model = implementation.PerthNet.load("implicit", str(package / "perth_net/pretrained")).to("cpu")
    log.info("Perth model initialization: %s", diagnostics.getvalue().strip())
    return model


def clear_model_cache() -> None:
    """Release the cached Perth model."""
    with _MODEL_LOCK:
        load_model.cache_clear()


def read(model: Any, samples: object) -> PerthReading:
    import numpy as np

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or len(array) < MIN_READ_SAMPLES or not np.isfinite(array).all():
        raise ValueError("Perth requires at least one second of finite mono samples at 16 kHz")
    torch = import_module("torch")
    with _MODEL_LOCK, torch.inference_mode():
        if model.hp.sample_rate != SAMPLE_RATE:
            librosa = import_module("librosa")
            array = librosa.resample(array, orig_sr=SAMPLE_RATE, target_sr=model.hp.sample_rate, res_type="polyphase")
        signal = torch.from_numpy(array.copy()).to(dtype=torch.float32, device=model.device)
        magnitudes, _phase = model.ap.signal_to_magphase(signal)
        prediction = model.decoder(magnitudes[None])[0]
        raw = np.asarray(prediction.detach().cpu().numpy(), dtype=np.float32)
    if raw.size != 1 or not np.isfinite(raw).all():
        raise RuntimeError("Perth returned an invalid presence confidence")
    return PerthReading(float(np.clip(raw.reshape(-1)[0], 0.0, 1.0)))


def embed(model: Any, samples: object) -> Any:
    """Research fixture generation; upstream resampling can shorten its tail."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or len(array) < MIN_READ_SAMPLES or not np.isfinite(array).all():
        raise ValueError("Perth embedding requires finite mono samples at 16 kHz")
    torch = import_module("torch")
    librosa = import_module("librosa")
    with _MODEL_LOCK, torch.inference_mode():
        native = librosa.resample(array, orig_sr=SAMPLE_RATE, target_sr=model.hp.sample_rate)
        signal = torch.from_numpy(native.copy()).to(dtype=torch.float32, device=model.device)
        magnitudes, phase = model.ap.signal_to_magphase(signal)
        marked, _mask = model.encoder(magnitudes[None])
        reconstructed = model.ap.magphase_to_signal(marked[0], phase).detach().cpu().numpy()
        delivered = np.asarray(
            librosa.resample(reconstructed, orig_sr=model.hp.sample_rate, target_sr=SAMPLE_RATE), dtype=np.float32
        )
    if delivered.ndim != 1 or not delivered.size or not np.isfinite(delivered).all():
        raise RuntimeError("Perth embedding returned invalid samples")
    return delivered


def checkpoint_path() -> Path:
    return _validated_package() / CHECKPOINT_RELATIVE_PATH


def available() -> bool:
    return is_available()
