#!/usr/bin/env python3
"""Revision-pinned local PixelSeal oracle for development-only benchmarks.

Meta publishes PixelSeal in the VideoSeal source tree, but not as a standalone
TorchScript artifact.  This loader uses the exact upstream source revision and
official checkpoint, verifies both the model card and checkpoint digests, and
builds only the image inference graph.  It avoids the upstream convenience
loader because that loader imports training datasets and writes downloads
relative to the current working directory.

Nothing here ships in the installed CLI or public Python API.
"""

from __future__ import annotations

# ruff: noqa: E402 - Bootstrap the source layout before sharing model assets.
import hashlib
import importlib.metadata
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from remove_ai_watermarks._internal.video_message_models import ASSETS
from remove_ai_watermarks._internal.video_message_models import SOURCE_COMMIT as SOURCE_COMMIT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from videoseal_oracle import VideoSealReading

log = logging.getLogger(__name__)

PACKAGE_NAME = "videoseal"
PACKAGE_VERSION = "1.0"
CHECKPOINT_URL, CHECKPOINT_SHA256 = ASSETS["pixelseal"]
MODEL_CARD_SHA256 = "96c4a448f7d81810780e5bde5a51b039db544b7ec2baf65a605302c1b17362c9"
N_BITS = 256
MESSAGE_SEED = 7
DETECTION_BIT_ACCURACY_THRESHOLD = 0.9
CACHE_DIR = Path(
    os.environ.get(
        "PIXELSEAL_CACHE_DIR",
        str(Path.home() / ".cache" / "remove-ai-watermarks-dev" / "pixelseal"),
    )
)


@dataclass(frozen=True)
class PixelSealReading:
    """Decoded bits and matched-message accuracy for one image."""

    bit_accuracy: float
    decoded_bits: tuple[int, ...]

    @property
    def detected(self) -> bool:
        """Apply this adapter's explicit matched-message rule."""
        return self.bit_accuracy >= DETECTION_BIT_ACCURACY_THRESHOLD

    @property
    def label(self) -> str:
        return "".join(f"{byte:02x}" for byte in _bytes(self.decoded_bits))


def _bytes(bits: Sequence[int]) -> bytes:
    return bytes(sum(bit << (7 - index) for index, bit in enumerate(bits[i : i + 8])) for i in range(0, len(bits), 8))


def message_bits(seed: int = MESSAGE_SEED) -> tuple[int, ...]:
    """The oracle's fixed 256-bit message, deterministic across processes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return tuple(int(bit) for bit in rng.integers(0, 2, N_BITS))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_failure() -> Exception | None:
    """Return the missing or incompatible dependency error, if any."""
    try:
        import omegaconf  # noqa: F401
        import timm  # noqa: F401
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        import videoseal  # noqa: F401

        if importlib.metadata.version(PACKAGE_NAME) != PACKAGE_VERSION:
            raise RuntimeError(
                f"{PACKAGE_NAME} must be installed from source commit {SOURCE_COMMIT} "
                f"(package version {PACKAGE_VERSION})"
            )
    except (ImportError, importlib.metadata.PackageNotFoundError, RuntimeError) as exc:
        return exc
    return None


def available() -> bool:
    """Whether the pinned PixelSeal source stack is importable."""
    return import_failure() is None


def checkpoint_path() -> Path:
    """Return the pinned local checkpoint, downloading and verifying it once."""
    target = CACHE_DIR / "checkpoint.pth"
    if target.is_file() and _sha256_file(target) == CHECKPOINT_SHA256:
        return target
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CACHE_DIR / "checkpoint.pth.part"
    log.info("Downloading pinned PixelSeal checkpoint from %s", CHECKPOINT_URL)
    urllib.request.urlretrieve(CHECKPOINT_URL, temporary)
    digest = _sha256_file(temporary)
    if digest != CHECKPOINT_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"pinned PixelSeal checkpoint digest mismatch: {digest} != {CHECKPOINT_SHA256}")
    temporary.replace(target)
    return target


def load_model() -> object:
    """Build PixelSeal from the pinned official source and weights on CPU."""
    failure = import_failure()
    if failure is not None:
        raise RuntimeError(
            "PixelSeal oracle requires facebookresearch/videoseal at source commit "
            f"{SOURCE_COMMIT}, plus torch, torchvision, timm, and omegaconf"
        ) from failure

    import torch
    import videoseal
    from omegaconf import OmegaConf
    from videoseal.augmentation.augmenter import get_dummy_augmenter
    from videoseal.models import Videoseal, build_embedder, build_extractor
    from videoseal.modules.jnd import JND

    package_root = Path(videoseal.__file__).resolve().parent
    card_path = package_root / "cards" / "pixelseal.yaml"
    if not card_path.is_file() or _sha256_file(card_path) != MODEL_CARD_SHA256:
        raise RuntimeError("installed videoseal source does not carry the pinned PixelSeal model card")
    config = OmegaConf.load(card_path)
    args = config.args
    embedder = build_embedder(
        config.embedder.model,
        config.embedder.params,
        args.nbits,
        args.hidden_size_multiplier,
    )
    extractor = build_extractor(
        config.extractor.model,
        config.extractor.params,
        args.img_size_proc,
        args.nbits,
    )
    model = Videoseal(
        embedder,
        extractor,
        get_dummy_augmenter(),
        attenuation=JND(in_channels=1, out_channels=1),
        scaling_w=args.scaling_w,
        scaling_i=args.scaling_i,
        img_size=args.img_size_proc,
        chunk_size=args.videowam_chunk_size,
        step_size=args.videowam_step_size,
    )
    checkpoint = torch.load(checkpoint_path(), map_location="cpu", weights_only=True)
    loaded = model.load_state_dict(checkpoint["model"], strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(
            f"PixelSeal checkpoint/source mismatch: missing={loaded.missing_keys}, unexpected={loaded.unexpected_keys}"
        )
    model.train(False)
    return model


def _image_tensor(image: object) -> object:
    import numpy as np
    import torch

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"PixelSeal expects HxWx3 RGB pixels, got {array.shape}")
    values = array.astype(np.float32) / 255.0 if array.dtype == np.uint8 else np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all() or float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("PixelSeal RGB pixels must be finite and in [0, 1]")
    return torch.from_numpy(np.ascontiguousarray(values)).permute(2, 0, 1).unsqueeze(0)


def embed(model: object, image: object, message: Sequence[int]) -> object:
    """Embed one message into an RGB image, returning float RGB in [0, 1]."""
    import torch

    tensor = _image_tensor(image)
    msg = torch.tensor([list(message)], dtype=torch.float32)
    with torch.inference_mode():
        marked = model.embed(tensor, msgs=msg, is_video=False)["imgs_w"]
    return marked[0].permute(1, 2, 0).numpy()


def read(model: object, image: object, *, message: Sequence[int] | None = None) -> PixelSealReading:
    """Decode one image and compare its bits with the oracle's fixed message."""
    import torch

    expected = message_bits() if message is None else tuple(message)
    tensor = _image_tensor(image)
    with torch.inference_mode():
        predictions = model.detect(tensor)["preds"]
    decoded = tuple(int(bit) for bit in (predictions[0, 1:] > 0).to(torch.int64).tolist())
    accuracy = sum(int(bit == want) for bit, want in zip(decoded, expected, strict=True)) / N_BITS
    return PixelSealReading(bit_accuracy=accuracy, decoded_bits=decoded)


def _video_frames_f32(frames: object) -> object:
    """Validate and normalize a nonempty RGB video without changing its geometry."""
    import numpy as np

    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[3] != 3 or any(size == 0 for size in array.shape[:3]):
        raise ValueError(f"PixelSeal video expects nonempty TxHxWx3 RGB pixels, got {array.shape}")
    values = array.astype(np.float32) / 255.0 if array.dtype == np.uint8 else np.asarray(array, dtype=np.float32)
    if not np.isfinite(values).all() or float(values.min()) < 0.0 or float(values.max()) > 1.0:
        raise ValueError("PixelSeal video RGB pixels must be finite and in [0, 1]")
    return np.ascontiguousarray(values)


def embed_video(model: object, frames: object, message: Sequence[int]) -> object:
    """Use the pinned model's temporal propagation, returning float RGB frames."""
    import torch

    tensor = torch.from_numpy(_video_frames_f32(frames)).permute(0, 3, 1, 2)
    msg = torch.tensor([list(message)], dtype=torch.float32)
    with torch.inference_mode():
        marked = model.embed(tensor, msgs=msg, is_video=True)["imgs_w"]
    return marked.permute(0, 2, 3, 1).numpy()


def read_video(model: object, frames: object, *, message: Sequence[int] | None = None) -> VideoSealReading:
    """Decode PixelSeal video through the same pinned-family logit aggregation."""
    from videoseal_oracle import read as read_video_message

    expected = message_bits() if message is None else message
    return read_video_message(model, _video_frames_f32(frames), message=expected)
