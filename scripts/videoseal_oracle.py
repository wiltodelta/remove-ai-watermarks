#!/usr/bin/env python3
"""Revision-pinned local VideoSeal oracle for development-only benchmarks.

Loads Meta's standalone TorchScript release of VideoSeal (MIT license, 256-bit
message) with nothing but PyTorch: no ``videoseal`` package, whose wheel pulls
a research dependency chain (decord without macOS arm64 wheels, timm,
pycocotools) and whose loader resolves model cards and attenuation configs
relative to the working directory. The TorchScript build published in the
project's ``docs/torchscript.md`` guide exposes the same embed/detect surface
as one self-contained file.

The checkpoint is pinned by URL, source commit, and SHA-256 digest, downloaded
once into a user-cache directory outside any repository, and verified before
every load. The detector's own "detection bit" column is unused by the
upstream video evaluation (it prepends a constant); the practical detection
signal is the aggregated message decode, so this oracle's matched decision
rule is bit accuracy of the 256-bit aggregated decode against the oracle's
fixed message. The 0.9 threshold is this adapter's explicit rule, not an
upstream constant; upstream reports accuracy curves rather than a verdict.

Nothing here contacts a provenance oracle or provider API, and none of this
ships in the installed CLI or the public Python API.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

JIT_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit"
JIT_SHA256 = "5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b"
SOURCE_COMMIT = "870ca7fb33578b90f14c602016b6c2788096226e"
N_BITS = 256
MESSAGE_SEED = 7
DETECTION_BIT_ACCURACY_THRESHOLD = 0.9
AGGREGATION = "avg"
AGGREGATIONS: tuple[str, ...] = ("avg", "squared_avg", "l1norm_avg", "l2norm_avg")
CACHE_DIR = Path(
    os.environ.get(
        "VIDEOSEAL_CACHE_DIR",
        str(Path.home() / ".cache" / "remove-ai-watermarks-dev" / "videoseal"),
    )
)


@dataclass(frozen=True)
class VideoSealImageReading:
    """Decoded bits and matched-message accuracy for one image."""

    bit_accuracy: float
    decoded_bits: tuple[int, ...]

    @property
    def detected(self) -> bool:
        """This adapter's explicit matched-detection rule."""
        return self.bit_accuracy >= DETECTION_BIT_ACCURACY_THRESHOLD

    @property
    def label(self) -> str:
        return "".join(f"{byte:02x}" for byte in _bytes(self.decoded_bits))


@dataclass(frozen=True)
class VideoSealReading(VideoSealImageReading):
    """Aggregated decode of one video against the oracle's fixed message."""

    per_frame_bit_accuracy: tuple[float, ...]


def _bytes(bits: Sequence[int]) -> bytes:
    return bytes(sum(bit << (7 - index) for index, bit in enumerate(bits[i : i + 8])) for i in range(0, len(bits), 8))


def message_bits(seed: int = MESSAGE_SEED) -> tuple[int, ...]:
    """The oracle's fixed 256-bit message, deterministic across processes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return tuple(int(bit) for bit in rng.integers(0, 2, N_BITS))


def import_failure() -> ImportError | None:
    """Return the missing-dependency error, or None when torch is present."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        return exc
    return None


def available() -> bool:
    """Whether the TorchScript stack is importable in this environment."""
    return import_failure() is None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path() -> Path:
    """Return the pinned local checkpoint, downloading and verifying it once."""
    target = CACHE_DIR / "y_256b_img.jit"
    if target.is_file() and _sha256_file(target) == JIT_SHA256:
        return target
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_DIR / "y_256b_img.jit.part"
    print(f"downloading pinned VideoSeal TorchScript checkpoint from {JIT_URL}")
    urllib.request.urlretrieve(JIT_URL, temporary)
    digest = _sha256_file(temporary)
    if digest != JIT_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"pinned VideoSeal checkpoint digest mismatch: {digest} != {JIT_SHA256}")
    temporary.replace(target)
    return target


def load_model() -> object:
    """Load the pinned TorchScript marker/detector on the CPU."""
    failure = import_failure()
    if failure is not None:
        raise RuntimeError("videoseal oracle requires the dev extra: uv sync --extra dev (torch)") from failure
    import torch

    model = torch.jit.load(str(checkpoint_path()), map_location="cpu")
    model.train(False)
    return model


def embed(model: object, frames: object, message: Sequence[int]) -> object:
    """Embed the fixed message into float RGB frames in [0, 1] (T, H, W, 3)."""
    import numpy as np
    import torch

    array = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))
    tensor = array.permute(0, 3, 1, 2)
    msg = torch.tensor([list(message)], dtype=torch.float32)
    with torch.no_grad():
        marked = model.embed(tensor, msg, is_video=True)
    return marked.permute(0, 2, 3, 1).numpy()


def _image_pixels_f32(pixels: object) -> object:
    import numpy as np

    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"VideoSeal image mode expects HxWx3 RGB pixels, got {array.shape}")
    values = array.astype(np.float32) / 255.0 if array.dtype == np.uint8 else np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all() or float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("VideoSeal image RGB pixels must be finite and in [0, 1]")
    return values


def _image_tensor(pixels: object) -> object:
    import numpy as np
    import torch

    values = _image_pixels_f32(pixels)
    array = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))
    return array.permute(2, 0, 1).unsqueeze(0)


def embed_image(model: object, pixels: object, message: Sequence[int]) -> object:
    """Embed a message into one RGB image in [0, 1] (H, W, 3)."""
    import torch

    tensor = _image_tensor(pixels)
    msg = torch.tensor([list(message)], dtype=torch.float32)
    with torch.no_grad():
        marked = model.embed(tensor, msg, is_video=False)
    return marked[0].permute(1, 2, 0).numpy()


def decode_image(model: object, pixels: object) -> tuple[int, ...]:
    """Decode the message bits from one RGB image."""
    import torch

    tensor = _image_tensor(pixels)
    with torch.no_grad():
        predictions = model.detect(tensor, is_video=False)
    if isinstance(predictions, dict):
        predictions = predictions["preds"]
    return tuple(int(bit) for bit in (predictions[:, 1:] > 0).to(torch.int64).view(-1).tolist())


def read_image(
    model: object,
    pixels: object,
    *,
    message: Sequence[int] | None = None,
) -> VideoSealImageReading:
    """Decode one image and compare its bits with the oracle's fixed message."""
    expected = message_bits() if message is None else tuple(message)
    decoded = decode_image(model, pixels)
    if len(decoded) != len(expected):
        raise ValueError(f"decoded bit count {len(decoded)} does not match expected {len(expected)}")
    accuracy = sum(int(bit == want) for bit, want in zip(decoded, expected, strict=True)) / len(expected)
    return VideoSealImageReading(bit_accuracy=accuracy, decoded_bits=decoded)


def _aggregate_bit_preds(bit_preds: object, aggregation: str) -> object:
    """Aggregate per-frame bit logits exactly as upstream ``extract_message``.

    The TorchScript build ignores the aggregation argument of
    ``detect_video_and_aggregate`` (identical bool output for every choice),
    so the aggregation is computed here from the raw per-frame logits using
    the upstream formulas.
    """
    import torch

    if aggregation == "avg":
        return bit_preds.mean(dim=0)
    if aggregation == "squared_avg":
        return (bit_preds * bit_preds.abs()).mean(dim=0)
    if aggregation == "l1norm_avg":
        weights = torch.norm(bit_preds, p=1, dim=1).unsqueeze(1)
        return (bit_preds * weights).mean(dim=0)
    if aggregation == "l2norm_avg":
        weights = torch.norm(bit_preds, p=2, dim=1).unsqueeze(1)
        return (bit_preds * weights).mean(dim=0)
    raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")


def _raw_preds(model: object, tensor: object) -> object:
    """One detector pass; the TorchScript build returns the tensor directly."""
    preds = model.detect(tensor, is_video=True)
    if isinstance(preds, dict):
        preds = preds["preds"]
    return preds


def read(
    model: object,
    frames: object,
    *,
    message: Sequence[int] | None = None,
    aggregation: str = AGGREGATION,
) -> VideoSealReading:
    """Decode one video: aggregated bits plus per-frame bit accuracy."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    import numpy as np
    import torch

    expected = message_bits() if message is None else tuple(message)
    array = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))
    tensor = array.permute(0, 3, 1, 2)
    with torch.no_grad():
        preds = _raw_preds(model, tensor)
    bit_preds = preds[:, 1:]
    aggregated = _aggregate_bit_preds(bit_preds, aggregation)
    decoded = tuple(int(bit) for bit in (aggregated.view(-1) > 0).to(torch.int64).tolist())
    accuracy = sum(int(bit == want) for bit, want in zip(decoded, expected, strict=True)) / N_BITS
    frame_bits = (bit_preds > 0).to(torch.int64)
    want = torch.tensor(expected, dtype=torch.int64).view(1, -1)
    per_frame = (frame_bits == want).float().mean(dim=1).tolist()
    return VideoSealReading(
        bit_accuracy=accuracy,
        decoded_bits=decoded,
        per_frame_bit_accuracy=tuple(float(value) for value in per_frame),
    )


def read_aggregation_matrix(model: object, frames: object) -> dict[str, float]:
    """Bit accuracy under every supported frame aggregation, one detect pass."""
    import numpy as np
    import torch

    expected = message_bits()
    array = torch.from_numpy(np.ascontiguousarray(frames, dtype=np.float32))
    tensor = array.permute(0, 3, 1, 2)
    with torch.no_grad():
        bit_preds = _raw_preds(model, tensor)[:, 1:]
    matrix: dict[str, float] = {}
    for aggregation in AGGREGATIONS:
        aggregated = _aggregate_bit_preds(bit_preds, aggregation)
        decoded = (aggregated.view(-1) > 0).to(torch.int64).tolist()
        matrix[aggregation] = sum(int(b == w) for b, w in zip(decoded, expected, strict=True)) / N_BITS
    return matrix


def detect_sample_array(model: object, frames: object) -> str | None:
    """Benchmark-kernel seam: the decoded message label, or None when absent."""
    reading = read(model, frames)
    return reading.label if reading.detected else None
