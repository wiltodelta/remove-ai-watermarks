#!/usr/bin/env python3
"""Hash-pinned public WavMark model; production-provider compatibility is unverified."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import logging
from dataclasses import dataclass, replace
from functools import lru_cache
from importlib import import_module
from math import comb
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)
SOURCE_COMMIT = "6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37"
MODEL_REPOSITORY = "M4869/WavMark"
MODEL_REVISION = "0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef"
MODEL_FILENAME = "step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl"
MODEL_SHA256 = "7a3d873d115c2c644685fb4cf109bdb299718296b88c871d3cf1c8e4cbc13144"
SAMPLE_RATE = 16_000
MESSAGE = (1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1)
SOURCE_TREE_SHA256 = "aecef6b16fe5ad11247960c264dc789f62005705fe41a4737f07384343d6cecb"
MIN_READ_SAMPLES = 16_000
MIN_EMBED_SAMPLES = 17_600
OPENVOICE_START_SAMPLES = (0, 32_000)
OPENVOICE_PREFIX_SAMPLES = 48_000
OPENVOICE_REFERENCE = b"default "
SIGNATURE_TAIL_THRESHOLD = 1e-6


@dataclass(frozen=True)
class WavMarkReading:
    """A payload accepted by the upstream exact synchronization-pattern check."""

    payload: tuple[int, ...] | None
    start_samples: tuple[int, ...]

    @property
    def detected(self) -> bool:
        return self.payload is not None


@dataclass(frozen=True)
class WavMarkSignatureReading:
    """Known-reference correlation; binomial tails are not population FPRs."""

    start_samples: tuple[int, ...]
    decoded_bits: tuple[int, ...]
    reference_hex: str
    matching_bits: int
    adjusted_binomial_tail: float
    framing: str = "openvoice-default"

    @property
    def detected(self) -> bool:
        """Apply the fixed, multiple-comparison-corrected signature criterion."""
        return self.adjusted_binomial_tail < SIGNATURE_TAIL_THRESHOLD


def import_failure() -> Exception | None:
    """Validate the public package source before importing its neural stack."""
    try:
        spec = importlib.util.find_spec("wavmark")
        if spec is None or spec.origin is None:
            raise ImportError("WavMark requires remove-ai-watermarks[audio-wavmark]")
        if importlib.metadata.version("wavmark") != "0.0.3":
            raise RuntimeError("WavMark requires wavmark==0.0.3")
        package = Path(spec.origin).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(package.rglob("*.py")):
            digest.update(path.relative_to(package).as_posix().encode() + b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        if digest.hexdigest() != SOURCE_TREE_SHA256:
            raise RuntimeError("Installed WavMark source differs from the pinned public model")
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError, ValueError, OSError) as exc:
        return exc
    return None


def available() -> bool:
    return import_failure() is None


def is_available() -> bool:
    return available()


def checkpoint_path() -> Path:
    """Download the revision-pinned checkpoint and verify bytes before loading."""
    hub = import_module("huggingface_hub")
    path = Path(hub.hf_hub_download(repo_id=MODEL_REPOSITORY, filename=MODEL_FILENAME, revision=MODEL_REVISION))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Pinned WavMark checkpoint digest mismatch: {digest}")
    return path


@lru_cache(maxsize=1)
def load_model() -> object:
    failure = import_failure()
    if failure is not None:
        raise RuntimeError("WavMark requires its pinned public package and audio-wavmark dependencies") from failure
    wavmark = import_module("wavmark")

    return wavmark.load_model(str(checkpoint_path()))


def clear_model_cache() -> None:
    """Release the cached WavMark model."""
    load_model.cache_clear()


def _samples(samples: object, minimum: int) -> NDArray[Any]:
    import numpy as np

    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or len(array) < minimum or not np.isfinite(array).all():
        raise ValueError(f"WavMark requires at least {minimum} finite mono samples at 16 kHz")
    return array


def embed(model: object, samples: object, *, message: tuple[int, ...] = MESSAGE) -> object:
    """Apply upstream framing and SNR policy; keep skipped sections in the evidence."""
    import numpy as np

    wavmark = import_module("wavmark")

    array = _samples(samples, MIN_EMBED_SAMPLES)
    if len(message) != 16 or any(bit not in (0, 1) for bit in message):
        raise ValueError("WavMark payload must contain 16 binary bits")
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics):
        marked, info = wavmark.encode_watermark(model, array, np.asarray(message), show_progress=False)
    log.info("WavMark embedding: info=%s diagnostics=%s", info, diagnostics.getvalue())
    result = np.asarray(marked, dtype=np.float32)
    if result.shape != array.shape or not np.isfinite(result).all():
        raise RuntimeError("WavMark returned invalid embedded samples")
    return result


def read(model: Any, samples: object) -> WavMarkReading:
    """Check the public synchronization rule, including the final valid window.

    The upstream sliding wrapper omits its final positions and performs no
    checks for an exactly one-second input. Preserve its 50 ms stride, exact
    synchronization rule, and payload vote, while covering those endpoints and
    rejecting nonfinite raw model outputs before converting them into bits.
    """
    import numpy as np

    torch = import_module("torch")
    wavmark = import_module("wavmark")
    array = _samples(samples, MIN_READ_SAMPLES)
    step = SAMPLE_RATE // 20
    end = len(array) - SAMPLE_RATE
    starts = list(range(0, end + 1, step))
    if starts[-1] != end:
        starts.append(end)
    pattern = np.asarray(wavmark.wm_add_util.fix_pattern[:16])
    device = next(model.parameters()).device
    locations: list[int] = []
    messages: list[NDArray[Any]] = []
    for offset in range(0, len(starts), 64):
        points = starts[offset : offset + 64]
        batch = np.stack([array[start : start + SAMPLE_RATE] for start in points])
        with torch.inference_mode():
            raw = model.decode(torch.from_numpy(batch).to(device)).detach().cpu().numpy()
        if raw.shape != (len(points), 32) or not np.isfinite(raw).all():
            raise RuntimeError("WavMark returned invalid raw message values")
        bits = raw >= 0.5
        for start, message in zip(points, bits, strict=True):
            if np.array_equal(message[:16], pattern):
                locations.append(start)
                messages.append(message[16:])
    if not messages:
        return WavMarkReading(None, ())
    payload = np.mean(np.stack(messages), axis=0) >= 0.5
    return WavMarkReading(tuple(int(bit) for bit in payload), tuple(locations))


def read_openvoice_default(
    model: Any, samples: object
) -> tuple[WavMarkSignatureReading, tuple[WavMarkSignatureReading, ...]]:
    """Check OpenVoice's fixed default chunks on the native sample clock.

    The published helper writes raw 32-bit blocks at samples 0 and 32000,
    without WavMark's 16-bit synchronization wrapper or sample-rate conversion.
    Custom messages and shifted/resampled framing are outside this receiver.
    """
    import numpy as np

    torch = import_module("torch")
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1 or len(array) < MIN_READ_SAMPLES or not np.isfinite(array).all():
        raise ValueError("OpenVoice default framing requires at least 16000 finite native mono samples")
    starts = tuple(start for start in OPENVOICE_START_SAMPLES if start + MIN_READ_SAMPLES <= len(array))
    batch = np.stack([array[start : start + MIN_READ_SAMPLES] for start in starts])
    device = next(model.parameters()).device
    with torch.inference_mode():
        raw = model.decode(torch.from_numpy(batch).to(device)).detach().cpu().numpy()
    if raw.shape != (len(starts), 32) or not np.isfinite(raw).all():
        raise RuntimeError("WavMark returned invalid raw signature values")
    decoded = raw >= 0.5
    comparisons = 3 if len(starts) == 2 else 1

    def score(values: NDArray[Any], reference: bytes, positions: tuple[int, ...]) -> WavMarkSignatureReading:
        expected = np.unpackbits(np.frombuffer(reference, dtype=np.uint8))
        matches = int(np.count_nonzero(values == expected))
        size = len(expected)
        tail = sum(comb(size, count) for count in range(matches, size + 1)) / (1 << size)
        return WavMarkSignatureReading(
            positions,
            tuple(int(bit) for bit in values),
            reference.hex(),
            matches,
            min(1.0, comparisons * tail),
        )

    fragments = tuple(
        score(decoded[index], OPENVOICE_REFERENCE[index * 4 : (index + 1) * 4], (start,))
        for index, start in enumerate(starts)
    )
    whole = score(decoded.reshape(-1), OPENVOICE_REFERENCE[: len(starts) * 4], starts)
    whole = replace(
        whole,
        adjusted_binomial_tail=min(whole.adjusted_binomial_tail, *(item.adjusted_binomial_tail for item in fragments)),
    )
    return whole, fragments
