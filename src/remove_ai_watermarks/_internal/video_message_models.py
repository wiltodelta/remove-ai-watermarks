"""Pinned CPU decoders for public VideoSeal, PixelSeal, and ChunkySeal models.

PixelSeal replaces only the detector weights in the verified compatible JIT
inference graph. ChunkySeal instantiates the pinned 1024-bit ConvNeXtV2 +
PixelDecoder detector graph directly in PyTorch without training SDKs or
deserializing unused optimizer states. Embedding networks are not exposed here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import shutil
import struct
import time
import urllib.request
import zlib
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, cast

from remove_ai_watermarks._internal.audio_quality import sha256_file as _digest
from remove_ai_watermarks._internal.utils import staged_sibling
from remove_ai_watermarks._internal.video_message_scores import SCHEME_MESSAGE_BITS

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)
VideoMessageScheme = Literal["videoseal", "pixelseal", "chunkyseal"]
SOURCE_COMMIT = "870ca7fb33578b90f14c602016b6c2788096226e"
ASSETS: dict[VideoMessageScheme, tuple[str, str]] = {
    "videoseal": (
        "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit",
        "5c7a4581c36fc6090aafdcfb3999123bae5172a4847f22e2da4e7fd1a39d1e1b",
    ),
    "pixelseal": (
        "https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth",
        "0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd",
    ),
    "chunkyseal": (
        "https://dl.fbaipublicfiles.com/videoseal/chunkyseal/checkpoint.pth",
        "a546154b6709abb31ca8c6d70ff1c88c9bd7ce973b95a4d2748ebf5746418ea7",
    ),
}
_CHUNKYSEAL_CD_RANGE = (13_376_956_224, 13_377_060_066)
_CHUNKYSEAL_CD_SHA256 = "7352b84d4c1937f3f99ca62eead28efcdc1be8002780f7756e00e1bd38eed205"
_CHUNKYSEAL_SLICE_START = 4_091_571_676
_CHUNKYSEAL_SLICE_END_EXCL = 7_186_475_484
_CHUNKYSEAL_RANGE_CHUNK = 8 * 1024 * 1024
_CHUNKYSEAL_FIRST_STORAGE_KEY = 228
CACHE_DIR = Path.home() / ".cache" / "remove-ai-watermarks" / "video-models"
_MODEL_LOCK = RLock()


@dataclass(frozen=True)
class VideoMessageDecoder:
    """Decoder-only handle; underlying generators are not exposed here."""

    scheme: VideoMessageScheme
    _network: Any

    def predict(self, tensor: Any) -> Any:
        """Read only the detector, validating every frame before thresholding."""
        torch = import_module("torch")
        num_bits = SCHEME_MESSAGE_BITS[self.scheme]
        with _MODEL_LOCK, torch.inference_mode():
            raw: Any = self._network.detect(tensor, is_video=True)
            if isinstance(raw, dict):
                raw = cast("dict[str, Any]", raw)["preds"]
            return validate_logits(raw, len(tensor), num_bits=num_bits)


def is_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("numpy", "torch"))


def _fetch_range_bytes(url: str, start: int, end_inclusive: int) -> bytes:
    expected_len = end_inclusive - start + 1
    for attempt in range(4):
        try:
            if not url.startswith("https://"):
                raise ValueError(f"Unsupported video model URL scheme: {url}")
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})  # noqa: S310
            with urllib.request.urlopen(req, timeout=60) as response:  # noqa: S310
                data = response.read()
            if len(data) != expected_len:
                raise RuntimeError(f"Short HTTP Range read for {url}: got {len(data)}, expected {expected_len}")
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.0 * (attempt + 1))
    raise AssertionError


def _download_chunkyseal_detector_slice(url: str, staged: Path) -> None:
    """Verify the archive directory and stream only the contiguous 3.10 GB detector slice."""
    cd_start, cd_end = _CHUNKYSEAL_CD_RANGE
    log.info("Public video model request: GET %s Range=bytes=%d-%d anonymous=True", url, cd_start, cd_end)
    cd_bytes = _fetch_range_bytes(url, cd_start, cd_end)
    cd_digest = hashlib.sha256(cd_bytes).hexdigest()
    log.info("Public video model response verified: component=central_directory sha256=%s", cd_digest)
    if cd_digest != _CHUNKYSEAL_CD_SHA256:
        raise RuntimeError("chunkyseal archive central directory differs from its pinned digest")

    log.info(
        "Public video model request: GET %s Range=bytes=%d-%d anonymous=True",
        url,
        _CHUNKYSEAL_SLICE_START,
        _CHUNKYSEAL_SLICE_END_EXCL - 1,
    )
    with staged.open("wb") as output:
        cursor = _CHUNKYSEAL_SLICE_START
        while cursor < _CHUNKYSEAL_SLICE_END_EXCL:
            end_incl = min(cursor + _CHUNKYSEAL_RANGE_CHUNK, _CHUNKYSEAL_SLICE_END_EXCL) - 1
            output.write(_fetch_range_bytes(url, cursor, end_incl))
            cursor = end_incl + 1


def _asset(scheme: VideoMessageScheme) -> Path:
    url, expected = ASSETS[scheme]
    suffix = ".jit" if scheme == "videoseal" else (".slice.pth" if scheme == "chunkyseal" else ".pth")
    target = CACHE_DIR / f"{scheme}-{expected}{suffix}"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with staged_sibling(target) as staged:
            if scheme == "chunkyseal":
                _download_chunkyseal_detector_slice(url, staged)
            else:
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


def _build_chunkyseal_network() -> Any:
    """Construct the pinned 1024-bit ConvNeXtV2 + PixelDecoder detector graph."""
    torch = import_module("torch")
    nn = torch.nn
    functional = torch.nn.functional

    class _LayerNorm(nn.Module):
        def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_first") -> None:
            nn.Module.__init__(self)
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.data_format = data_format
            self.normalized_shape = (normalized_shape,)

        def forward(self, x: Any) -> Any:
            if self.data_format == "channels_last":
                return functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            normed = (x - mean) / torch.sqrt(var + self.eps)
            return self.weight[:, None, None] * normed + self.bias[:, None, None]

    class _GRN(nn.Module):
        def __init__(self, dim: int) -> None:
            nn.Module.__init__(self)
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

        def forward(self, x: Any) -> Any:
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
            return self.gamma * (x * nx) + self.beta + x

    class _ConvNeXtBlock(nn.Module):
        def __init__(self, dim: int) -> None:
            nn.Module.__init__(self)
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
            self.norm = _LayerNorm(dim, eps=1e-6, data_format="channels_last")
            self.pwconv1 = nn.Linear(dim, 4 * dim)
            self.act = nn.GELU()
            self.grn = _GRN(4 * dim)
            self.pwconv2 = nn.Linear(4 * dim, dim)

        def forward(self, x: Any) -> Any:
            residual = x
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.grn(x)
            x = self.pwconv2(x)
            x = x.permute(0, 3, 1, 2).contiguous()
            return residual + x

    class _ConvNeXtV2(nn.Module):
        def __init__(self, depths: tuple[int, ...], dims: tuple[int, ...], stem_stride: int = 2) -> None:
            nn.Module.__init__(self)
            self.downsample_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(3, dims[0], kernel_size=4, stride=stem_stride),
                        _LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
                    ),
                    *(
                        nn.Sequential(
                            _LayerNorm(dims[idx], eps=1e-6, data_format="channels_first"),
                            nn.Conv2d(dims[idx], dims[idx + 1], kernel_size=2, stride=2),
                        )
                        for idx in range(3)
                    ),
                ]
            )
            self.stages = nn.ModuleList(
                [nn.Sequential(*(_ConvNeXtBlock(dims[idx]) for _ in range(depths[idx]))) for idx in range(4)]
            )

        def forward(self, x: Any) -> Any:
            for down, stage in zip(self.downsample_layers, self.stages, strict=True):
                x = stage(down(x))
            return x

    class _Upsample(nn.Module):
        def __init__(self, dim: int) -> None:
            nn.Module.__init__(self)
            self.upsample_block = nn.Sequential(
                nn.Upsample(scale_factor=1, mode="bilinear", align_corners=False),
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=False),
                _LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                nn.GELU(),
            )

        def forward(self, x: Any) -> Any:
            return self.upsample_block(x)

    class _PixelDecoder(nn.Module):
        def __init__(self, embed_dim: int, nbits: int) -> None:
            nn.Module.__init__(self)
            self.output_upscaling = nn.Sequential(_Upsample(embed_dim))
            self.linear = nn.Linear(embed_dim, nbits + 1)

        def forward(self, embeddings: Any) -> Any:
            pooled = self.output_upscaling(embeddings).mean(dim=[-2, -1])
            return self.linear(pooled)

    class _ConvnextExtractor(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            # proportional_dim=True on nbits=1024 scales [128, 256, 512, 1024] by sqrt(1024 / 128).
            self.convnext = _ConvNeXtV2(depths=(3, 3, 27, 3), dims=(362, 724, 1448, 2896), stem_stride=2)
            self.pixel_decoder = _PixelDecoder(embed_dim=2896, nbits=1024)

        def forward(self, imgs: Any) -> Any:
            return self.pixel_decoder(self.convnext(imgs * 2.0 - 1.0))

    class _ChunkySealDetectorWrapper(nn.Module):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.detector = _ConvnextExtractor()
            self.img_size = 256

        def detect(self, imgs: Any, is_video: bool = True) -> dict[str, Any]:
            del is_video
            if tuple(imgs.shape[-2:]) != (self.img_size, self.img_size):
                imgs = functional.interpolate(
                    imgs,
                    size=(self.img_size, self.img_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            return {"preds": self.detector(imgs)}

    return _ChunkySealDetectorWrapper()


def _peek_zip_entry_name(path: Path) -> str | None:
    with path.open("rb") as stream:
        header = stream.read(30)
        if len(header) < 30 or header[:4] != b"PK\x03\x04":
            return None
        fname_len = struct.unpack("<H", header[26:28])[0]
        raw_name = stream.read(fname_len)
        if len(raw_name) != fname_len:
            return None
        return raw_name.decode("utf-8", errors="replace")


def _read_exact(stream: BinaryIO, length: int, label: str) -> bytes:
    buf = stream.read(length)
    if len(buf) != length:
        raise RuntimeError(f"Truncated ChunkySeal detector slice while reading {label}")
    return buf


def _load_chunkyseal_slice(slice_path: Path, detector: Any) -> None:
    """Load the verified raw ZIP64 detector slice without executing pickle."""
    import numpy as np

    torch = import_module("torch")
    template = detector.state_dict()
    loaded: dict[str, Any] = {}
    with slice_path.open("rb") as stream:
        for offset_idx, (param_name, ref_tensor) in enumerate(template.items()):
            expected_entry = f"checkpoint/data/{_CHUNKYSEAL_FIRST_STORAGE_KEY + offset_idx}"
            header = _read_exact(stream, 30, expected_entry)
            if header[:4] != b"PK\x03\x04":
                raise RuntimeError(f"Invalid ZIP entry signature for {expected_entry}")
            gp_flags, comp_method = struct.unpack("<HH", header[6:10])
            crc_hdr, comp_hdr, uncomp_hdr, fname_len, extra_len = struct.unpack("<IIIHH", header[14:30])
            if comp_method != 0:
                raise RuntimeError(f"Unexpected compression method {comp_method} in {expected_entry}")
            entry_name = _read_exact(stream, fname_len, f"{expected_entry} name").decode("utf-8", errors="replace")
            if entry_name != expected_entry:
                raise RuntimeError(f"ChunkySeal storage order mismatch: expected {expected_entry}, got {entry_name}")
            if extra_len:
                _read_exact(stream, extra_len, f"{expected_entry} extra")
            nbytes = int(ref_tensor.numel()) * 4
            raw_bytes = _read_exact(stream, nbytes, f"{expected_entry} payload")
            actual_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
            if gp_flags & 0x0008:
                desc = _read_exact(stream, 16, f"{expected_entry} descriptor")
                if desc[:4] != b"PK\x07\x08":
                    raise RuntimeError(f"Missing ZIP data descriptor after {expected_entry}")
                crc_desc, comp_desc, uncomp_desc = struct.unpack("<III", desc[4:16])
                if crc_desc != actual_crc or comp_desc != nbytes or uncomp_desc != nbytes:
                    raise RuntimeError(f"Corrupted ZIP data descriptor for {expected_entry}")
            elif crc_hdr != actual_crc or comp_hdr != nbytes or uncomp_hdr != nbytes:
                raise RuntimeError(f"Corrupted ZIP header sizes/CRC for {expected_entry}")
            arr = np.frombuffer(raw_bytes, dtype="<f4").copy().reshape(tuple(ref_tensor.shape))
            loaded[param_name] = torch.from_numpy(arr)
    detector.load_state_dict(loaded, strict=True)


def _extract_detector_state_dict(checkpoint_path: Path, label: str) -> dict[str, Any]:
    torch = import_module("torch")
    checkpoint: object = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{label} checkpoint is not a parameter dictionary")
    entries = cast("dict[object, object]", checkpoint)
    raw_state = entries.get("model")
    if not isinstance(raw_state, dict):
        raise RuntimeError(f"{label} checkpoint has no model parameters")
    state_entries = cast("dict[object, object]", raw_state)
    state: dict[str, Any] = {}
    for key, value in state_entries.items():
        if not isinstance(key, str):
            raise RuntimeError(f"{label} parameter name is invalid")
        if key.startswith("detector."):
            state[key.removeprefix("detector.")] = value
    if not state:
        raise RuntimeError(f"{label} requires a nonempty compatible detector graph")
    return state


@lru_cache(maxsize=3)
def load_detector(scheme: VideoMessageScheme) -> VideoMessageDecoder:
    """Verify every asset before loading the public CPU inference graph."""
    if scheme not in ASSETS:
        raise ValueError("Video message scheme must be videoseal, pixelseal, or chunkyseal")
    if not is_available():
        raise ImportError("Public video message decoding requires remove-ai-watermarks[video-detect]")
    torch = import_module("torch")
    with _MODEL_LOCK:
        if scheme == "chunkyseal":
            checkpoint_path = _asset("chunkyseal")
            model = _build_chunkyseal_network()
            if _peek_zip_entry_name(checkpoint_path) == f"checkpoint/data/{_CHUNKYSEAL_FIRST_STORAGE_KEY}":
                _load_chunkyseal_slice(checkpoint_path, model.detector)
            else:
                state = _extract_detector_state_dict(checkpoint_path, "ChunkySeal")
                model.detector.load_state_dict(state, strict=True)
        else:
            graph = _asset("videoseal")
            checkpoint_path = _asset("pixelseal") if scheme == "pixelseal" else None
            model = torch.jit.load(str(graph), map_location="cpu")
            if checkpoint_path is not None:
                state = _extract_detector_state_dict(checkpoint_path, "PixelSeal")
                if not model.detector.state_dict():
                    raise RuntimeError("PixelSeal requires a nonempty compatible detector graph")
                model.detector.load_state_dict(state, strict=True)
        model.train(False)
    log.info("Loaded pinned public video decoder: scheme=%s source=%s device=cpu", scheme, SOURCE_COMMIT)
    return VideoMessageDecoder(scheme, model)


def clear_model_cache() -> None:
    """Release cached VideoSeal/PixelSeal/ChunkySeal decoders."""
    with _MODEL_LOCK:
        load_detector.cache_clear()


def validate_logits(predictions: Any, frames: int, *, num_bits: int = 256) -> Any:
    """Return the same valid tensor, rejecting incomplete or invalid NN output."""
    torch = import_module("torch")
    if (
        not isinstance(predictions, torch.Tensor)
        or tuple(predictions.shape) != (frames, num_bits + 1)
        or frames < 1
        or not torch.isfinite(predictions).all()
    ):
        raise RuntimeError("Video watermark model returned invalid video logits")
    return predictions


def decode_presence_and_logits(model: VideoMessageDecoder, frames: object) -> tuple[NDArray[Any], NDArray[Any]]:
    """Read finite presence logits (channel 0) and message logits (channels 1..) from normalized RGB frames."""
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
    raw = model.predict(tensor).detach().cpu().numpy()
    return np.asarray(raw[:, 0], dtype=np.float32), np.asarray(raw[:, 1:], dtype=np.float32)


def decode_logits(model: VideoMessageDecoder, frames: object) -> NDArray[Any]:
    """Read finite message logits from normalized RGB frames, without a verdict."""
    _presence, message = decode_presence_and_logits(model, frames)
    return message
