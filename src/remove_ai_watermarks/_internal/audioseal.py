"""Pinned open AudioSeal model helpers for optional runtime verification.

The shared reader exposes compatible-model watermark presence and payload bits,
not a proprietary Meta Content Seal or general AI-origin verdict. Generator
helpers remain internal and support controlled development fixtures.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

_MODEL_LOCK = RLock()

HF_REPO = "facebook/audioseal"
HF_REVISION = "3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba"
GENERATOR_SHA256 = "7a845b5fbe9364a63a3909d8ab3fe064d13a76ae4c2e983573e08c69b7b51748"
DETECTOR_SHA256 = "8a78e8a83584113523e161fc599fcab10fd0e94c04d2eb9d2fa1e9ec91ab69d9"
SAMPLE_RATE = 16_000
N_BITS = 16
DETECTION_THRESHOLD = 0.5
MESSAGE_THRESHOLD = 0.5
GENERATOR_CARD = "audioseal_wm_16bits"
DETECTOR_CARD = "audioseal_detector_16bits"


@dataclass(frozen=True)
class AudioSealReading:
    """Detector readings on one decoded mono sample array."""

    mean_detect_prob: float
    frac_above_threshold: float
    decoded_bits: tuple[int, ...]

    @property
    def detected(self) -> bool:
        """The detector's own document decision rule at the default threshold."""
        return self.frac_above_threshold >= DETECTION_THRESHOLD

    @property
    def label(self) -> str:
        return "".join(str(bit) for bit in self.decoded_bits)


def import_failure() -> ImportError | None:
    """Return the missing-dependency error, or None when the stack is present."""
    try:
        for name in ("audioseal", "huggingface_hub", "omegaconf", "torch"):
            import_module(name)
    except ImportError as exc:
        return exc
    return None


def available() -> bool:
    """Whether the audioseal stack is importable in this environment."""
    return import_failure() is None


def _load_pinned_model(kind: Literal["generator", "detector"]) -> Any:
    """Load generator and detector from the pinned, digest-verified revision."""
    failure = import_failure()
    if failure is not None:
        raise RuntimeError("AudioSeal verification requires remove-ai-watermarks[audio-detect]") from failure

    import hashlib

    audioseal_pkg = import_module("audioseal")
    AudioSeal = audioseal_pkg.AudioSeal
    builder = import_module("audioseal.builder")
    loader = import_module("audioseal.loader")
    AudioSealDetectorConfig = loader.AudioSealDetectorConfig
    AudioSealWMConfig = loader.AudioSealWMConfig
    create_detector, create_generator = builder.create_detector, builder.create_generator
    convert_state_dict_for_scriptable_model = loader.convert_state_dict_for_scriptable_model
    load_model_checkpoint, load_state_dict = loader.load_model_checkpoint, loader.load_state_dict
    hf_hub_download = import_module("huggingface_hub").hf_hub_download
    OmegaConf = import_module("omegaconf").OmegaConf

    module_file = audioseal_pkg.__file__
    if module_file is None:
        raise RuntimeError("AudioSeal package has no model-card location")
    cards_dir = Path(module_file).parent / "cards"

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load(card_name: str, filename: str, cfg_type: Any, create: Any) -> Any:
        local = Path(hf_hub_download(HF_REPO, filename, revision=HF_REVISION))
        digest = sha256_file(local)
        expected = GENERATOR_SHA256 if "generator" in filename else DETECTOR_SHA256
        if digest != expected:
            raise RuntimeError(f"pinned checkpoint digest mismatch for {filename}: {digest} != {expected}")
        raw_config = OmegaConf.to_container(OmegaConf.load(cards_dir / f"{card_name}.yaml"))
        if not isinstance(raw_config, dict):
            raise RuntimeError(f"audioseal card {card_name} did not parse into a mapping")
        config_dict = cast("dict[str, Any]", raw_config)
        config_dict.pop("checkpoint")
        checkpoint = load_model_checkpoint(str(local))
        merged: dict[str, Any] = {**checkpoint.get("xp.cfg", {}), **config_dict}
        config = AudioSeal.parse_config(merged, cfg_type)
        raw = checkpoint.get("best_state", checkpoint.get("model", checkpoint))
        raw = convert_state_dict_for_scriptable_model(raw)
        model = create(config, device="cpu")
        load_state_dict(model, state_dict=raw)
        model.train(False)
        return model

    if kind == "generator":
        return load(GENERATOR_CARD, "generator_base.pth", AudioSealWMConfig, create_generator)
    return load(DETECTOR_CARD, "detector_base.pth", AudioSealDetectorConfig, create_detector)


def load_pinned_models() -> tuple[Any, Any]:
    """Load the controlled-fixture generator and its compatible detector."""
    return _load_pinned_model("generator"), _load_pinned_model("detector")


@lru_cache(maxsize=1)
def load_pinned_detector() -> Any:
    """Load only the detector needed by the runtime path."""
    return _load_pinned_model("detector")


@contextmanager
def _eager() -> Generator[None, None, None]:
    """Use AudioSeal's local eager context without changing process settings."""
    with _MODEL_LOCK:
        no_compile = import_module("audioseal.libs.moshi.utils.compile").no_compile
        with no_compile():
            yield


def is_available() -> bool:
    """Whether the optional compatible AudioSeal stack imports successfully."""
    return available()


def embed(generator: Any, samples: Any, message: Sequence[int]) -> Any:
    """Embed one fixed bit message into a float32 mono sample array."""
    torch = import_module("torch")

    wav = torch.from_numpy(samples).view(1, 1, -1)
    msg = torch.tensor([list(message)], dtype=torch.float32)
    with _eager(), torch.no_grad():
        marked = generator(wav, sample_rate=SAMPLE_RATE, alpha=1.0, message=msg)
    return marked.view(-1).numpy()


def read(detector: Any, samples: Any) -> AudioSealReading:
    """Read detection probabilities and the decoded message from samples."""
    import numpy as np

    torch = import_module("torch")

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("AudioSeal samples must be nonempty finite mono audio")
    wav = torch.from_numpy(np.ascontiguousarray(array)).view(1, 1, -1)
    with _eager(), torch.no_grad():
        result, message = detector(wav, sample_rate=SAMPLE_RATE)
    probs = result[0, 1, :].numpy()
    message_values = message.view(-1).numpy()
    if not np.isfinite(probs).all() or not np.isfinite(message_values).all():
        raise RuntimeError("AudioSeal returned nonfinite detector values")
    bits = tuple(int(bit) for bit in (message_values > MESSAGE_THRESHOLD).astype(int))
    return AudioSealReading(
        mean_detect_prob=float(probs.mean()),
        frac_above_threshold=float((probs > DETECTION_THRESHOLD).mean()),
        decoded_bits=bits,
    )


def window_bounds(total_samples: int, window_s: float, sample_rate: int) -> list[tuple[int, int]]:
    """Half-open sample bounds of fixed seconds-long windows over a signal.

    A trailing remainder shorter than ``window_s`` is its own final window, so
    interval reporting never silently drops a partial second.
    """
    if total_samples < 0:
        raise ValueError("total_samples must be non-negative")
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    size = int(window_s * sample_rate)
    if size <= 0:
        raise ValueError("window_s is shorter than one sample")
    return [(start, min(start + size, total_samples)) for start in range(0, total_samples, size)]


def read_intervals(detector: Any, samples: Any, *, window_s: float = 1.0) -> list[AudioSealReading]:
    """Read one detection per fixed time window, oldest window first."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    return [read(detector, array[start:end]) for start, end in window_bounds(array.shape[0], window_s, SAMPLE_RATE)]


def detect_sample_array(detector: Any, samples: Any) -> str | None:
    """Return the decoded message label when the detector fires, else None."""
    reading = read(detector, samples)
    return reading.label if reading.detected else None
