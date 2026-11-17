"""Pinned CPU decoders for public VideoSeal and PixelSeal, without training SDKs.

PixelSeal replaces only the detector weights in the verified compatible JIT
inference graph. Its embedding network is deliberately not exposed here.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal, cast

from remove_ai_watermarks._internal.audio_quality import sha256_file as _digest
from remove_ai_watermarks._internal.utils import staged_sibling

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)
VideoMessageScheme = Literal["videoseal", "pixelseal"]
SOURCE_COMMIT = "870ca7fb33578b90f14c602016b6c2788096226e"
ASSETS = {
    "videoseal": (
        "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit",
        "5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b",
    ),
    "pixelseal": (
        "https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth",
        "0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd",
    ),
}
CACHE_DIR = Path.home() / ".cache" / "remove-ai-watermarks" / "video-models"
_MODEL_LOCK = RLock()


@dataclass(frozen=True)
class VideoMessageDecoder:
    """Decoder-only handle; the underlying generator is not a PixelSeal API."""

    scheme: VideoMessageScheme
    _network: Any

    def predict(self, tensor: Any) -> Any:
        """Read only the detector, validating every frame before thresholding."""
        torch = import_module("torch")
        with _MODEL_LOCK, torch.inference_mode():
            raw: Any = self._network.detect(tensor, is_video=True)
            if isinstance(raw, dict):
                raw = cast("dict[str, Any]", raw)["preds"]
            return validate_logits(raw, len(tensor))


def is_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))


def _asset(scheme: VideoMessageScheme) -> Path:
    url, expected = ASSETS[scheme]
    suffix = ".jit" if scheme == "videoseal" else ".pth"
    target = CACHE_DIR / f"{scheme}-{expected}{suffix}"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with staged_sibling(target) as staged:
            log.info("Public video model request: GET %s anonymous=True", url)
            with urllib.request.urlopen(url, timeout=60) as response, staged.open("wb") as output:  # noqa: S310
                shutil.copyfileobj(response, output)
            digest = _digest(staged)
            log.info("Public video model response saved: path=%s sha256=%s", staged, digest)
            if digest != expected:
                raise RuntimeError(f"{scheme} model download differs from its pinned digest")
    if _digest(target) != expected:
        raise RuntimeError(f"{scheme} cached model differs from its pinned digest")
    return target


@lru_cache(maxsize=2)
def load_detector(scheme: VideoMessageScheme) -> VideoMessageDecoder:
    """Verify every asset before loading the public CPU inference graph."""
    if scheme not in ASSETS:
        raise ValueError("Video message scheme must be videoseal or pixelseal")
    if not is_available():
        raise ImportError("Public video message decoding requires remove-ai-watermarks[video-detect]")
    torch = import_module("torch")
    with _MODEL_LOCK:
        graph = _asset("videoseal")
        checkpoint_path = _asset("pixelseal") if scheme == "pixelseal" else None
        model = torch.jit.load(str(graph), map_location="cpu")
        if checkpoint_path is not None:
            checkpoint: object = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise RuntimeError("PixelSeal checkpoint is not a parameter dictionary")
            entries = cast("dict[object, object]", checkpoint)
            raw_state = entries.get("model")
            if not isinstance(raw_state, dict):
                raise RuntimeError("PixelSeal checkpoint has no model parameters")
            state_entries = cast("dict[object, object]", raw_state)
            state: dict[str, Any] = {}
            for key, value in state_entries.items():
                if not isinstance(key, str):
                    raise RuntimeError("PixelSeal parameter name is invalid")
                if key.startswith("detector."):
                    state[key.removeprefix("detector.")] = value
            if not state or not model.detector.state_dict():
                raise RuntimeError("PixelSeal requires a nonempty compatible detector graph")
            model.detector.load_state_dict(state, strict=True)
        model.train(False)
    log.info("Loaded pinned public video decoder: scheme=%s source=%s device=cpu", scheme, SOURCE_COMMIT)
    return VideoMessageDecoder(scheme, model)


def validate_logits(predictions: Any, frames: int) -> Any:
    """Return the same valid tensor, rejecting incomplete or invalid NN output."""
    torch = import_module("torch")
    if (
        not isinstance(predictions, torch.Tensor)
        or tuple(predictions.shape) != (frames, 257)
        or frames < 1
        or not torch.isfinite(predictions).all()
    ):
        raise RuntimeError("Video watermark model returned invalid video logits")
    return predictions


def decode_logits(model: VideoMessageDecoder, frames: object) -> NDArray[Any]:
    """Read finite message logits from normalized RGB frames, without a verdict."""
    import numpy as np

    values = np.asarray(frames, dtype=np.float32)
    if (
        values.ndim != 4
        or values.shape[-1] != 3
        or not values.size
        or not np.isfinite(values).all()
        or values.min() < 0
        or values.max() > 1
    ):
        raise ValueError("Video decoding requires nonempty finite TxHxWx3 RGB frames in [0, 1]")
    torch = import_module("torch")
    tensor = torch.from_numpy(np.ascontiguousarray(values)).permute(0, 3, 1, 2)
    raw = model.predict(tensor)
    return np.asarray(raw[:, 1:].detach().cpu().numpy(), dtype=np.float32)
