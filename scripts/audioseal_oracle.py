#!/usr/bin/env python3
"""Revision-pinned local AudioSeal oracle for development-only benchmarks.

One shared seam for the benchmark kernel adapter, the audio cohort builder,
and the audio-provenance experiment. Weights are pinned to an exact Hugging
Face revision of ``facebook/audioseal`` and verified against pinned SHA-256
digests before loading; the loader mirrors ``AudioSeal.parse_model`` with the
checkpoint bytes pinned, so card values override the checkpoint's embedded
``xp.cfg`` exactly like the package loader does for card loads.

The first load may download the pinned checkpoints into the local Hugging
Face cache when it is incomplete, like the optional TrustMark weights. Prepare
the dependency before an offline run. Nothing here contacts a provenance
oracle or provider API, and none of this ships in the installed CLI or the
public Python API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

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
        import audioseal  # noqa: F401
        import huggingface_hub  # noqa: F401
        import omegaconf  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        return exc
    return None


def available() -> bool:
    """Whether the audioseal stack is importable in this environment."""
    return import_failure() is None


def load_pinned_models() -> tuple[object, object]:
    """Load generator and detector from the pinned, digest-verified revision."""
    failure = import_failure()
    if failure is not None:
        raise RuntimeError(
            "audioseal oracle requires the dev extra: uv sync --extra dev (audioseal, torch, huggingface-hub)"
        ) from failure

    import hashlib

    import audioseal as audioseal_pkg
    import torch
    from audioseal import AudioSeal
    from audioseal.builder import create_detector, create_generator
    from audioseal.loader import (
        AudioSealDetectorConfig,
        AudioSealWMConfig,
        convert_state_dict_for_scriptable_model,
        load_model_checkpoint,
        load_state_dict,
    )
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf

    cards_dir = Path(audioseal_pkg.__file__).parent / "cards"

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def load(card_name: str, filename: str, cfg_type: object, create: object) -> object:
        local = Path(hf_hub_download(HF_REPO, filename, revision=HF_REVISION))
        digest = sha256_file(local)
        expected = GENERATOR_SHA256 if "generator" in filename else DETECTOR_SHA256
        if digest != expected:
            raise RuntimeError(f"pinned checkpoint digest mismatch for {filename}: {digest} != {expected}")
        config_dict = OmegaConf.to_container(OmegaConf.load(cards_dir / f"{card_name}.yaml"))
        if not isinstance(config_dict, dict):
            raise RuntimeError(f"audioseal card {card_name} did not parse into a mapping")
        config_dict.pop("checkpoint")
        checkpoint = load_model_checkpoint(str(local))
        merged = {**checkpoint.get("xp.cfg", {}), **config_dict}
        config = AudioSeal.parse_config(merged, cfg_type)  # type: ignore[arg-type]
        raw = checkpoint.get("best_state", checkpoint.get("model", checkpoint))
        raw = convert_state_dict_for_scriptable_model(raw)
        model = create(config, device="cpu")  # type: ignore[call-arg]
        load_state_dict(model, state_dict=raw)
        model.eval()
        return model

    generator = load(GENERATOR_CARD, "generator_base.pth", AudioSealWMConfig, create_generator)
    detector = load(DETECTOR_CARD, "detector_base.pth", AudioSealDetectorConfig, create_detector)
    if torch is None:  # pragma: no cover - defensive, import kept for symmetry
        raise RuntimeError("torch import disappeared")
    return generator, detector


def embed(generator: object, samples: object, message: Sequence[int]) -> object:
    """Embed one fixed bit message into a float32 mono sample array."""
    import torch

    wav = torch.from_numpy(samples).view(1, 1, -1)
    msg = torch.tensor([list(message)], dtype=torch.float32)
    with torch.no_grad():
        marked = generator(wav, sample_rate=SAMPLE_RATE, alpha=1.0, message=msg)
    return marked.view(-1).numpy()


def read(detector: object, samples: object) -> AudioSealReading:
    """Read detection probabilities and the decoded message from samples."""
    import torch

    wav = torch.from_numpy(samples).view(1, 1, -1)
    with torch.no_grad():
        result, message = detector(wav, sample_rate=SAMPLE_RATE)
    probs = result[0, 1, :].numpy()
    bits = tuple(int(bit) for bit in (message.view(-1).numpy() > MESSAGE_THRESHOLD).astype(int))
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


def read_intervals(detector: object, samples: object, *, window_s: float = 1.0) -> list[AudioSealReading]:
    """Read one detection per fixed time window, oldest window first."""
    import numpy as np

    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    return [read(detector, array[start:end]) for start, end in window_bounds(array.shape[0], window_s, SAMPLE_RATE)]


def detect_sample_array(detector: object, samples: object) -> str | None:
    """Return the decoded message label when the detector fires, else None."""
    reading = read(detector, samples)
    return reading.label if reading.detected else None
