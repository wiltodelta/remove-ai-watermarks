#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#     "remove-ai-watermarks[source-classify]>=0.64.0",
#     "huggingface-hub>=0.26.0",
#     "onnxruntime>=1.24.0",
#     "peft>=0.14.0",
#     "pyarrow>=17.0.0",
#     "safetensors>=0.4.5",
#     "sentence-transformers>=6.1.0",
#     "timm>=1.0.9",
#     "torch>=2.5.0",
#     "transformers>=5.20.0",
# ]
# ///
"""Reproducible open benchmark: `wiltodelta/raiw-source-classify` vs 2025-2026 frontier and legacy HF AI detectors.

Streams pinned, un-gated datasets directly from Hugging Face Hub at immutable commit SHAs
(no private files, no manual dataset setup) across three evaluation regimes:
1. Non-AI false-positive traps (camera photos, UI screenshots, science diagrams,
   scanned documents).
2. Untouched native AI generator exports across FLUX.1-dev, OpenAI GPT-Image-1,
   Google Gemini Nano Banana Pro, Stable Diffusion XL, and Midjourney v6.1.
3. Transcoded crowd-arena AI outputs across multi-provider preference shards re-encoded
   into square JPEG/WebP/PNG arena containers.

Evaluates open-weights configurations across artifact size, license, runtime stack,
warm single-image CPU latency, non-AI FPR by slice, binary recall/precision, and exact
generator-family attribution accuracy.

Run directly with `uv`:

    uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/benchmark.py
    uv run benchmark.py --tier frontier --json-out results.json
    uv run benchmark.py --raiw-only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import itertools
import json
import logging
import math
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from collections.abc import Callable

    from huggingface_hub import HfApi, HfFileSystem

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if (_REPO_SRC / "remove_ai_watermarks").is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

log = logging.getLogger(__name__)

Regime = Literal["non_ai", "native", "arena"]

MIN_SIDE_PX = 256
MAX_AREA_PX = 9_000_000
LATENCY_SAMPLE_COUNT = 30
SMAUG_CPU_LATENCY_COUNT = 10
EXIF_ORIENTATION_TAG = 274
TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class SampleItem:
    """One deduplicated benchmark image in original encoded bytes."""

    sha256: str
    category: str
    expected_provider: str | None
    container_format: str
    width: int
    height: int
    raw_bytes: bytes
    regime: Regime = "arena"

    def __post_init__(self) -> None:
        if (self.regime == "non_ai") != (self.expected_provider is None):
            raise ValueError(
                f"Inconsistent SampleItem regime={self.regime!r} for expected_provider={self.expected_provider!r}"
            )

    @property
    def is_ai(self) -> bool:
        return self.expected_provider is not None


@dataclass(frozen=True)
class ParquetSliceSpec:
    """Pinned Hugging Face single-image Parquet slice."""

    category: str
    dataset_id: str
    revision: str
    parquet_path: str
    limit: int = 50
    expected_provider: str | None = None
    regime: Regime = "non_ai"


@dataclass(frozen=True)
class RawRepoSliceSpec:
    """Pinned Hugging Face dataset tree slice containing native generator export files."""

    category: str
    dataset_id: str
    revision: str
    path_in_repo: str | None
    limit: int
    expected_provider: str


@dataclass(frozen=True)
class PreferenceSliceSpec:
    """Pinned Hugging Face pairwise-preference Parquet slice."""

    dataset_id: str
    revision: str
    parquet_path: str
    per_provider_cap: int
    provider_map: dict[str, str]
    max_row_groups: int | None = None
    substring_match: bool = False


@dataclass(frozen=True)
class OpenDetectorSpec:
    """Pinned Hugging Face `transformers` binary AI-image detector checkpoint."""

    repo_id: str
    revision: str
    weights_mb: float
    license_id: str
    ai_labels: frozenset[str] = field(default_factory=frozenset)
    tier: str = "legacy"
    use_sigmoid: bool = False


@dataclass(frozen=True)
class TimmViTSpec:
    """Pinned `vit_small_patch16_384` checkpoint evaluated at native CommFor preprocessing."""

    repo_id: str
    revision: str
    weights_mb: float
    license_id: str
    logit_threshold: float


RAPIDATA_4O_PROVIDER_MAP: dict[str, str] = {
    "4o-26-3-25": "openai",
    "dalle-3": "openai",
    "imagen-3": "google",
    "nano-banana": "google",
    "flux-1-pro": "flux",
    "flux-1.1-pro": "flux",
    "midjourney-5.2": "midjourney",
    "stable-diffusion-3": "stable_diffusion",
    "ideogram": "ideogram",
    "aurora-20-1-25": "xai",
    "seedream-3": "bytedance",
}

SEEDREAM_PROVIDER_MAP: dict[str, str] = {
    "seedream": "bytedance",
}

NON_AI_SLICES: tuple[ParquetSliceSpec, ...] = (
    ParquetSliceSpec(
        category="camera_photo",
        dataset_id="visual-layer/imagenet-1k-vl-enriched",
        revision="ac6afcdeb3be31c5ff6a7ff579874b3d372b7074",
        parquet_path="data/validation-00000-of-00008.parquet",
        limit=50,
    ),
    ParquetSliceSpec(
        category="ui_screenshot",
        dataset_id="rootsautomation/ScreenSpot",
        revision="0be08781e2e188582f6131625ae1598d443b4d5d",
        parquet_path="data/test-00000-of-00003.parquet",
        limit=50,
    ),
    ParquetSliceSpec(
        category="science_diagram",
        dataset_id="lmms-lab/ai2d",
        revision="c83a9b9692933aff8349157c88a413df9d02c4e5",
        parquet_path="data/test-00000-of-00002.parquet",
        limit=50,
    ),
    ParquetSliceSpec(
        category="scanned_document",
        dataset_id="lmms-lab/DocVQA",
        revision="539088ef8a8ada01ac8e2e6d4e372586748a265e",
        parquet_path="DocVQA/test-00000-of-00006.parquet",
        limit=50,
    ),
)

NATIVE_RAW_AI_SLICES: tuple[RawRepoSliceSpec, ...] = (
    RawRepoSliceSpec(
        category="native_flux",
        dataset_id="LukasT9/Flux-1-Dev-Images-1k",
        revision="ade8147d5f10bac016d691fd2b564e877e6315f7",
        path_in_repo="images",
        limit=25,
        expected_provider="flux",
    ),
    RawRepoSliceSpec(
        category="native_openai",
        dataset_id="Tungtom2004/Output_dataset_gptImage1",
        revision="ef327a0e5f089b34e27f25d2a7700c5fba811bf7",
        path_in_repo="10000-20000",
        limit=25,
        expected_provider="openai",
    ),
    RawRepoSliceSpec(
        category="native_stable_diffusion",
        dataset_id="aziksh/animals_with_objects_sdxl",
        revision="478f6a267926671d01d468e9211b5f5132d90aa3",
        path_in_repo="cat_blue boat",
        limit=25,
        expected_provider="stable_diffusion",
    ),
    RawRepoSliceSpec(
        category="native_midjourney",
        dataset_id="saq1b/midjourney-v6.1",
        revision="97a3ff2aea504384145af73d77df7f124613e9c6",
        path_in_repo=None,
        limit=25,
        expected_provider="midjourney",
    ),
)

NATIVE_GOOGLE_PARQUET_SPEC = ParquetSliceSpec(
    category="native_google",
    dataset_id="kaupane/nano-banana-pro-gen",
    revision="5c61390a0307e6c38788ab85a5280fbc9893415b",
    parquet_path="data/train-00000-of-00005.parquet",
    limit=25,
    expected_provider="google",
    regime="native",
)

RAPIDATA_4O_SPEC = PreferenceSliceSpec(
    dataset_id="Rapidata/OpenAI-4o_t2i_human_preference",
    revision="ca6c084c0fa0021d66bbdd44d6e61fbeb8e1198a",
    parquet_path="data/train-00000-of-00022.parquet",
    per_provider_cap=35,
    provider_map=RAPIDATA_4O_PROVIDER_MAP,
)

NANO_BANANA_ARENA_SPEC = ParquetSliceSpec(
    category="ai_google",
    dataset_id="bitmind/nano-banana",
    revision="9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4",
    parquet_path="data/train-00000-of-00031.parquet",
    limit=25,
    expected_provider="google",
    regime="arena",
)

SEEDREAM_ARENA_SPEC = PreferenceSliceSpec(
    dataset_id="Rapidata/Seedream-3_t2i_human_preference",
    revision="0867c34e5583d63376e3289b938e6a2cb6f43751",
    parquet_path="data/train-00000-of-00101.parquet",
    per_provider_cap=25,
    provider_map=SEEDREAM_PROVIDER_MAP,
    max_row_groups=3,
    substring_match=True,
)

FRONTIER_VIT_DETECTORS: tuple[TimmViTSpec, ...] = (
    TimmViTSpec(
        repo_id="TensorFold/detectra-v1",
        revision="2bbc31161daa46932e05ee6f96e7016ee817bbde",
        weights_mb=41.7,
        license_id="mit",
        logit_threshold=0.6190392,  # logit(0.65)
    ),
    TimmViTSpec(
        repo_id="Thermostatic/community-forensics-frontier-detector-2026-08",
        revision="16db135220b318d811b207db576d90368980b595",
        weights_mb=41.8,
        license_id="mit",
        logit_threshold=1.359375,
    ),
)

COMMFOR_DEEPFAKEDET_SPEC = OpenDetectorSpec(
    repo_id="buildborderless/CommunityForensics-DeepfakeDet-ViT",
    revision="ac6ee457bea904a373065754107451793b56db00",
    weights_mb=83.2,
    license_id="mit",
    tier="frontier",
    use_sigmoid=True,
)

SMAUG_REPO = "abacusai/Smaug-AI-Image-Detector"
SMAUG_REVISION = "c7c95951e04a87b884a4ad479401b1128c3903bd"

DETECTRA_OPENFAKE_REPO = "s-gehring/detectra-openfake"
DETECTRA_OPENFAKE_REVISION = "c84565e5bf8969d95908e12844f551e77ae87105"

OPENFAKE_BASE_REPO = "bendavidsteel/openfake-swinv2-base"
OPENFAKE_BASE_REVISION = "90837e19c45b9a49af435d35071114ba1448c239"
OPENFAKE_ATTR_REPO = "bendavidsteel/openfake-swinv2-base-attribution-head"
OPENFAKE_ATTR_REVISION = "000944122ba497638fc88eea671dbe998fe635da"
OPENFAKE_LORA_REPO = "bendavidsteel/openfake-swinv2-base-textgraphics-lora"
OPENFAKE_LORA_REVISION = "b319ab4689286a620c3ef4a678709f4d04f25793"

LEGACY_BINARY_DETECTORS: tuple[OpenDetectorSpec, ...] = (
    OpenDetectorSpec(
        repo_id="haywoodsloan/ai-image-detector-deploy",
        revision="4ae1822603cb1ec980c217cfde1ef1b8e0477a16",
        ai_labels=frozenset({"artificial"}),
        weights_mb=744.7,
        license_id="apache-2.0",
    ),
    OpenDetectorSpec(
        repo_id="Organika/sdxl-detector",
        revision="b37fede8562cb72b89ec201c0987f96ba21b518a",
        ai_labels=frozenset({"artificial"}),
        weights_mb=331.4,
        license_id="cc-by-nc-3.0",
    ),
    OpenDetectorSpec(
        repo_id="dima806/ai_vs_human_generated_image_detection",
        revision="341594f2b003d185db789becf48a2286c177b926",
        ai_labels=frozenset({"ai-generated", "artificial", "fake"}),
        weights_mb=327.3,
        license_id="apache-2.0",
    ),
    OpenDetectorSpec(
        repo_id="umm-maybe/AI-image-detector",
        revision="c7e223baf11bc40528af364ba7bdea030ef42f9e",
        ai_labels=frozenset({"artificial"}),
        weights_mb=331.5,
        license_id="cc-by-4.0",
    ),
    OpenDetectorSpec(
        repo_id="Ateeqq/ai-vs-human-image-detector",
        revision="60e82406916921b823616bee33397baab38af3f0",
        ai_labels=frozenset({"ai"}),
        weights_mb=354.4,
        license_id="apache-2.0",
    ),
)

OPENFAKE_FAMILY_MAP: dict[str, str] = {
    "DALLE3": "openai",
    "GPT-Image": "openai",
    "Imagen": "google",
    "FLUX1": "flux",
    "Midjourney": "midjourney",
    "SD1": "stable_diffusion",
    "SD2": "stable_diffusion",
    "SD3": "stable_diffusion",
    "SDXL": "stable_diffusion",
    "Ideogram": "ideogram",
    "Grok": "xai",
}


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        return


def _inspect_image_bytes(raw_bytes: bytes) -> tuple[str, int, int] | None:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            w, h = im.size
            if min(w, h) < MIN_SIDE_PX or (w * h) > MAX_AREA_PX:
                return None
            fmt = str(im.format or "UNKNOWN")
            orientation = im.getexif().get(EXIF_ORIENTATION_TAG, 1)
            im.load()
            if orientation in TRANSPOSED_ORIENTATIONS:
                w, h = h, w
            return fmt, w, h
    except Exception:
        return None


def _decode_rgb(raw_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(raw_bytes)) as im:
        ImageOps.exif_transpose(im, in_place=True)
        return im.convert("RGB") if im.mode != "RGB" else im.copy()


def _preprocess_commfor_chw(rgb: Image.Image, short_edge: int = 440, crop: int = 384) -> np.ndarray:
    w, h = rgb.size
    scale = short_edge / min(w, h)
    nw, nh = round(w * scale), round(h * scale)
    resized = rgb.resize((nw, nh), Image.Resampling.BILINEAR)
    left = (nw - crop) // 2
    top = (nh - crop) // 2
    cropped = resized.crop((left, top, left + crop, top + crop))
    arr = np.asarray(cropped, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    return (chw - IMAGENET_MEAN) / IMAGENET_STD


def _smaug_views(rgb: Image.Image) -> list[Image.Image]:
    crop_px, side_px = 528, 576
    w, h = rgb.size
    if min(w, h) > side_px:
        scale = side_px / min(w, h)
        rgb = rgb.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    elif min(w, h) < crop_px:
        scale = crop_px / min(w, h)
        rgb = rgb.resize((math.ceil(w * scale), math.ceil(h * scale)), Image.Resampling.BICUBIC)
    w, h = rgb.size
    cx, cy = (w - crop_px) // 2, (h - crop_px) // 2
    if w >= h:
        return [rgb.crop((x, cy, x + crop_px, cy + crop_px)) for x in (0, cx, w - crop_px)]
    return [rgb.crop((cx, y, cx + crop_px, y + crop_px)) for y in (0, cy, h - crop_px)]


def _try_build_sample(
    cell: Any,
    local_seen: set[str],
    *,
    category: str,
    expected_provider: str | None,
    regime: Regime,
) -> SampleItem | None:
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    if not isinstance(raw, (bytes, bytearray)):
        return None
    raw_bytes = bytes(raw)
    sha = hashlib.sha256(raw_bytes).hexdigest()
    if sha in local_seen:
        return None
    meta = _inspect_image_bytes(raw_bytes)
    if meta is None:
        return None
    fmt, w, h = meta
    local_seen.add(sha)
    return SampleItem(
        sha256=sha,
        category=category,
        expected_provider=expected_provider,
        container_format=fmt,
        width=w,
        height=h,
        raw_bytes=raw_bytes,
        regime=regime,
    )


def _latency_indices(total: int, count: int = LATENCY_SAMPLE_COUNT) -> list[int]:
    """Deterministic stratified indices across the full sample list for CPU timing."""
    if total <= count:
        return list(range(total))
    step = max(1, total // count)
    return [i * step for i in range(count)]


def _read_single_image_slice(fs: HfFileSystem, spec: ParquetSliceSpec) -> list[SampleItem]:
    import pyarrow.parquet as pq

    items: list[SampleItem] = []
    local_seen: set[str] = set()
    uri = f"datasets/{spec.dataset_id}/{spec.parquet_path}"
    with fs.open(uri, "rb", revision=spec.revision) as handle:
        pf = pq.ParquetFile(handle)
        for batch in pf.iter_batches(batch_size=spec.limit + 16, columns=["image"]):
            col = batch["image"]
            for row_idx in range(len(col)):
                if len(items) >= spec.limit:
                    break
                sample = _try_build_sample(
                    col[row_idx].as_py(),
                    local_seen,
                    category=spec.category,
                    expected_provider=spec.expected_provider,
                    regime=spec.regime,
                )
                if sample is not None:
                    items.append(sample)
            if len(items) >= spec.limit:
                break
    log.info("Loaded %s images from %s (%s)", len(items), spec.dataset_id, spec.category)
    return items


def _read_raw_repo_slice(api: HfApi, spec: RawRepoSliceSpec) -> list[SampleItem]:
    from huggingface_hub import hf_hub_download

    tree_iter = api.list_repo_tree(
        spec.dataset_id,
        path_in_repo=spec.path_in_repo,
        repo_type="dataset",
        revision=spec.revision,
    )
    candidate_paths = list(
        itertools.islice(
            (
                str(entry.path)
                for entry in tree_iter
                if str(getattr(entry, "path", "")).lower().endswith(IMAGE_SUFFIXES)
            ),
            spec.limit + 8,
        )
    )

    def _download_bytes(rel_path: str) -> bytes:
        local_path = Path(
            hf_hub_download(
                spec.dataset_id,
                rel_path,
                repo_type="dataset",
                revision=spec.revision,
            )
        )
        return local_path.read_bytes()

    items: list[SampleItem] = []
    local_seen: set[str] = set()
    offset = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        while len(items) < spec.limit and offset < len(candidate_paths):
            needed = spec.limit - len(items)
            batch_paths = candidate_paths[offset : offset + needed]
            offset += len(batch_paths)
            for blob in pool.map(_download_bytes, batch_paths):
                sample = _try_build_sample(
                    blob,
                    local_seen,
                    category=spec.category,
                    expected_provider=spec.expected_provider,
                    regime="native",
                )
                if sample is not None:
                    items.append(sample)
                    if len(items) >= spec.limit:
                        break
    log.info("Loaded %s native images from %s (%s)", len(items), spec.dataset_id, spec.category)
    return items


def _read_preference_slice(fs: HfFileSystem, spec: PreferenceSliceSpec) -> list[SampleItem]:
    import pyarrow.parquet as pq

    items: list[SampleItem] = []
    local_seen: set[str] = set()
    counts: dict[str, int] = {}
    norm_map = {k.lower(): v for k, v in spec.provider_map.items()}
    target_providers = set(norm_map.values())
    uri = f"datasets/{spec.dataset_id}/{spec.parquet_path}"

    def _resolve_provider(raw_tag: Any) -> str | None:
        tag_lc = str(raw_tag).lower()
        if spec.substring_match:
            return next((prov for needle, prov in norm_map.items() if needle in tag_lc), None)
        return norm_map.get(tag_lc)

    with fs.open(uri, "rb", revision=spec.revision) as handle:
        pf = pq.ParquetFile(handle)
        rg_limit = pf.num_row_groups if spec.max_row_groups is None else min(spec.max_row_groups, pf.num_row_groups)
        for rg_idx in range(rg_limit):
            if all(counts.get(p, 0) >= spec.per_provider_cap for p in target_providers):
                break
            meta_rg = pf.read_row_group(rg_idx, columns=["model1", "model2"])
            p1_list = [_resolve_provider(t) for t in meta_rg["model1"].to_pylist()]
            p2_list = [_resolve_provider(t) for t in meta_rg["model2"].to_pylist()]
            need_col1 = any(p is not None and counts.get(p, 0) < spec.per_provider_cap for p in p1_list)
            need_col2 = any(p is not None and counts.get(p, 0) < spec.per_provider_cap for p in p2_list)
            img_cols = [c for c, flag in (("image1", need_col1), ("image2", need_col2)) if flag]
            if not img_cols:
                continue
            img_rg = pf.read_row_group(rg_idx, columns=img_cols)
            for row_idx, (p1, p2) in enumerate(zip(p1_list, p2_list, strict=True)):
                for prov, icol in ((p1, "image1"), (p2, "image2")):
                    if prov is None or counts.get(prov, 0) >= spec.per_provider_cap or icol not in img_cols:
                        continue
                    sample = _try_build_sample(
                        img_rg[icol][row_idx].as_py(),
                        local_seen,
                        category=f"ai_{prov}",
                        expected_provider=prov,
                        regime="arena",
                    )
                    if sample is not None:
                        counts[prov] = counts.get(prov, 0) + 1
                        items.append(sample)
    log.info("Loaded %s preference AI images from %s", len(items), spec.dataset_id)
    return items


def _dedup_ordered(batches: list[list[SampleItem]], seen_shas: set[str]) -> list[SampleItem]:
    merged: list[SampleItem] = []
    for batch in batches:
        for item in batch:
            if item.sha256 not in seen_shas:
                seen_shas.add(item.sha256)
                merged.append(item)
    return merged


def load_all_samples(fs: HfFileSystem, api: HfApi) -> list[SampleItem]:
    """Stream all pinned non-AI, native-export AI, and crowd-arena AI slices in deterministic order."""
    with ThreadPoolExecutor(max_workers=6) as pool:
        non_ai_futures = [pool.submit(_read_single_image_slice, fs, spec) for spec in NON_AI_SLICES]
        native_raw_futures = [pool.submit(_read_raw_repo_slice, api, spec) for spec in NATIVE_RAW_AI_SLICES]
        fut_native_google = pool.submit(_read_single_image_slice, fs, NATIVE_GOOGLE_PARQUET_SPEC)
        fut_4o = pool.submit(_read_preference_slice, fs, RAPIDATA_4O_SPEC)
        fut_nb = pool.submit(_read_single_image_slice, fs, NANO_BANANA_ARENA_SPEC)
        fut_sd = pool.submit(_read_preference_slice, fs, SEEDREAM_ARENA_SPEC)

        seen_shas: set[str] = set()
        return _dedup_ordered(
            [
                *[f.result() for f in non_ai_futures],
                *[f.result() for f in native_raw_futures],
                fut_native_google.result(),
                fut_4o.result(),
                fut_nb.result(),
                fut_sd.result(),
            ],
            seen_shas,
        )


def _select_torch_accelerator() -> Any:
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate_rgb_detector(
    samples: list[SampleItem],
    rgb_cache: list[Image.Image] | None,
    *,
    score_cpu_batch: Callable[[list[Image.Image]], list[dict[str, Any]]],
    make_batch_scorer: Callable[[], Callable[[list[Image.Image]], list[dict[str, Any]]]] | None = None,
    batch_size: int = 16,
    latency_count: int = LATENCY_SAMPLE_COUNT,
    model_id: str,
    revision: str,
    runtime: str,
    weights_mb: float,
    license_id: str,
    tier: str = "frontier",
) -> dict[str, Any]:
    """Shared harness: warm up on CPU, measure single-image CPU latency from raw bytes, then batch untimed items."""
    rgb0 = rgb_cache[0] if rgb_cache is not None else _decode_rgb(samples[0].raw_bytes)
    _ = score_cpu_batch([rgb0])

    lat_idxs = _latency_indices(len(samples), latency_count)
    lat_idx_set = set(lat_idxs)
    latencies_ms: list[float] = []
    preds_by_idx: dict[int, dict[str, Any]] = {}

    for idx in lat_idxs:
        t0 = time.perf_counter()
        rgb = _decode_rgb(samples[idx].raw_bytes)
        pred = score_cpu_batch([rgb])[0]
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        preds_by_idx[idx] = pred

    batch_scorer = make_batch_scorer() if make_batch_scorer is not None else score_cpu_batch
    untimed_indices = [idx for idx in range(len(samples)) if idx not in lat_idx_set]
    for start in range(0, len(untimed_indices), batch_size):
        chunk_idxs = untimed_indices[start : start + batch_size]
        rgbs = [rgb_cache[idx] if rgb_cache is not None else _decode_rgb(samples[idx].raw_bytes) for idx in chunk_idxs]
        for sample_idx, pred in zip(chunk_idxs, batch_scorer(rgbs), strict=True):
            preds_by_idx[sample_idx] = pred

    return _summarize_predictions(
        model_id=model_id,
        revision=revision,
        runtime=runtime,
        weights_mb=weights_mb,
        license_id=license_id,
        median_cpu_ms=round(statistics.median(latencies_ms), 1),
        samples=samples,
        preds=[preds_by_idx[i] for i in range(len(samples))],
        tier=tier,
    )


def evaluate_raiw(
    samples: list[SampleItem],
    *,
    weights_path: Path | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Evaluate `wiltodelta/raiw-source-classify` on CPU."""
    from remove_ai_watermarks import source_classify as sc

    weights_file = sc.resolve_weights_path(weights_path)
    weights_mb = round(weights_file.stat().st_size / (1024 * 1024), 2)

    def _classify_one(raw_bytes: bytes) -> dict[str, Any]:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            res = sc.classify_source(im, weights_path=weights_file)
        return {
            "pred_ai": bool(getattr(res, "synthetic_detected", res.label != "unknown")),
            "pred_provider": None if res.label == "unknown" else res.label,
        }

    _classify_one(samples[0].raw_bytes)

    lat_idxs = _latency_indices(len(samples))
    lat_idx_set = set(lat_idxs)
    latencies_ms: list[float] = []
    preds_by_idx: dict[int, dict[str, Any]] = {}

    for idx in lat_idxs:
        t0 = time.perf_counter()
        preds_by_idx[idx] = _classify_one(samples[idx].raw_bytes)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    for idx in range(len(samples)):
        if idx not in lat_idx_set:
            preds_by_idx[idx] = _classify_one(samples[idx].raw_bytes)

    return _summarize_predictions(
        model_id=sc.WEIGHTS_REPO,
        revision=revision or sc.WEIGHTS_REVISION,
        runtime="NumPy + ONNX Runtime (CPU)",
        weights_mb=weights_mb,
        license_id="apache-2.0",
        median_cpu_ms=round(statistics.median(latencies_ms), 1),
        samples=samples,
        preds=[preds_by_idx[i] for i in range(len(samples))],
        tier="raiw",
    )


def evaluate_hf_binary_detector(
    spec: OpenDetectorSpec,
    samples: list[SampleItem],
    rgb_cache: list[Image.Image] | None = None,
) -> dict[str, Any]:
    """Evaluate an open Hugging Face `transformers` binary AI-image detector."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification, ViTForImageClassification

    processor = AutoImageProcessor.from_pretrained(spec.repo_id, revision=spec.revision)
    ai_indices: list[int] = []
    if spec.use_sigmoid:
        model = ViTForImageClassification.from_pretrained(spec.repo_id, revision=spec.revision)
    else:
        model = AutoModelForImageClassification.from_pretrained(spec.repo_id, revision=spec.revision)
        id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        ai_indices = [idx for idx, lbl in id2label.items() if lbl in spec.ai_labels]
        if not ai_indices:
            raise RuntimeError(f"Could not map AI label for {spec.repo_id}: {id2label}")

    def logits_to_probs(logits: Any) -> Any:
        if spec.use_sigmoid:
            return torch.sigmoid(logits.reshape(-1))
        return torch.softmax(logits, dim=-1)[:, ai_indices].sum(dim=-1)

    model.train(False)
    model.to("cpu")

    def _score_on(dev: Any) -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        def _run(rgbs: list[Image.Image]) -> list[dict[str, Any]]:
            inputs = {k: v.to(dev) for k, v in processor(images=rgbs, return_tensors="pt").items()}
            with torch.inference_mode():
                probs = logits_to_probs(model(**inputs).logits).cpu().tolist()
            return [{"pred_ai": float(p_val) >= 0.5, "pred_provider": None} for p_val in probs]

        return _run

    def _move_to_acc() -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        dev = _select_torch_accelerator()
        model.to(dev)
        return _score_on(dev)

    return _evaluate_rgb_detector(
        samples,
        rgb_cache,
        score_cpu_batch=_score_on(torch.device("cpu")),
        make_batch_scorer=_move_to_acc,
        model_id=spec.repo_id,
        revision=spec.revision,
        runtime="PyTorch + Transformers",
        weights_mb=spec.weights_mb,
        license_id=spec.license_id,
        tier=spec.tier,
    )


def evaluate_timm_vit_detector(
    spec: TimmViTSpec,
    samples: list[SampleItem],
    rgb_cache: list[Image.Image] | None = None,
) -> dict[str, Any]:
    """Evaluate a `vit_small_patch16_384` frontier detector at official 440->384 preprocessing."""
    import timm
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    raw_state = load_file(hf_hub_download(spec.repo_id, "model.safetensors", revision=spec.revision))
    clean_state = {(k[4:] if k.startswith("vit.") else k): v for k, v in raw_state.items()}
    del raw_state
    model = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    model.load_state_dict(clean_state, strict=True)
    del clean_state
    model.train(False)
    model.to("cpu")

    def _score_on(dev: Any) -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        def _run(rgbs: list[Image.Image]) -> list[dict[str, Any]]:
            batch_np = np.stack([_preprocess_commfor_chw(rgb) for rgb in rgbs])
            with torch.inference_mode():
                logits = model(torch.from_numpy(batch_np).to(dev)).reshape(-1).cpu().tolist()
            return [{"pred_ai": float(val) >= spec.logit_threshold, "pred_provider": None} for val in logits]

        return _run

    def _move_to_acc() -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        dev = _select_torch_accelerator()
        model.to(dev)
        return _score_on(dev)

    return _evaluate_rgb_detector(
        samples,
        rgb_cache,
        score_cpu_batch=_score_on(torch.device("cpu")),
        make_batch_scorer=_move_to_acc,
        model_id=spec.repo_id,
        revision=spec.revision,
        runtime="PyTorch + timm / ONNX",
        weights_mb=spec.weights_mb,
        license_id=spec.license_id,
        tier="frontier",
    )


def evaluate_detectra_openfake(
    samples: list[SampleItem],
    rgb_cache: list[Image.Image] | None = None,
) -> dict[str, Any]:
    """Evaluate `s-gehring/detectra-openfake` INT8 ONNX with its documented two-pass Platt rule."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    onnx_path = Path(hf_hub_download(DETECTRA_OPENFAKE_REPO, "model.onnx", revision=DETECTRA_OPENFAKE_REVISION))
    weights_mb = round(onnx_path.stat().st_size / (1024 * 1024), 2)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def _platt(logit_val: float) -> float:
        return _sigmoid(0.552 * logit_val + 0.1933)

    def _score_batch(rgbs: list[Image.Image]) -> list[dict[str, Any]]:
        x_batch = np.stack([_preprocess_commfor_chw(rgb, short_edge=440, crop=384) for rgb in rgbs])
        logits = np.asarray(sess.run(None, {in_name: x_batch})[0]).reshape(-1)
        probs = [_platt(float(val)) for val in logits]

        borderline = [idx for idx, p_val in enumerate(probs) if 0.5 <= p_val < 0.95]
        if borderline:
            hi_crops = np.stack(
                [_preprocess_commfor_chw(rgbs[idx], short_edge=se, crop=384) for idx in borderline for se in (576, 768)]
            )
            hi_logits = np.asarray(sess.run(None, {in_name: hi_crops})[0]).reshape(len(borderline), 2)
            for b_pos, idx in enumerate(borderline):
                probs[idx] = max(probs[idx], _platt(float(hi_logits[b_pos, 0])), _platt(float(hi_logits[b_pos, 1])))

        return [{"pred_ai": p_val >= 0.95, "pred_provider": None} for p_val in probs]

    return _evaluate_rgb_detector(
        samples,
        rgb_cache,
        score_cpu_batch=_score_batch,
        model_id=DETECTRA_OPENFAKE_REPO,
        revision=DETECTRA_OPENFAKE_REVISION,
        runtime="ONNX Runtime INT8 (CPU)",
        weights_mb=weights_mb,
        license_id="cc-by-nc-4.0",
        tier="frontier",
    )


def evaluate_openfake_attribution(
    samples: list[SampleItem],
    rgb_cache: list[Image.Image] | None = None,
) -> dict[str, Any]:
    """Evaluate OpenFake SwinV2-base paired with its 15-family sub-center attribution head."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as f_nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    base_st = Path(hf_hub_download(OPENFAKE_BASE_REPO, "model.safetensors", revision=OPENFAKE_BASE_REVISION))
    attr_cfg_path = Path(hf_hub_download(OPENFAKE_ATTR_REPO, "config.json", revision=OPENFAKE_ATTR_REVISION))
    attr_st_path = Path(hf_hub_download(OPENFAKE_ATTR_REPO, "model.safetensors", revision=OPENFAKE_ATTR_REVISION))
    attr_mb = round((base_st.stat().st_size + attr_st_path.stat().st_size) / (1024 * 1024), 2)

    attr_cfg = json.loads(attr_cfg_path.read_text(encoding="utf-8"))
    attr_weights = load_file(str(attr_st_path))
    families: list[str] = [str(x) for x in attr_cfg["families"]]

    processor = AutoImageProcessor.from_pretrained(OPENFAKE_BASE_REPO, revision=OPENFAKE_BASE_REVISION)
    detector = AutoModelForImageClassification.from_pretrained(OPENFAKE_BASE_REPO, revision=OPENFAKE_BASE_REVISION)
    detector.train(False)
    detector.to("cpu")

    head = nn.Sequential(
        nn.Linear(int(attr_cfg["in_dim"]), int(attr_cfg["emb_dim"])),
        nn.BatchNorm1d(int(attr_cfg["emb_dim"])),
    )
    head.load_state_dict({k[len("head.") :]: v for k, v in attr_weights.items() if k.startswith("head.")})
    head.train(False)
    head.to("cpu")
    family_protos_cpu = attr_weights["family_prototypes"].float()
    del attr_weights

    def _score_on(dev: Any, protos: Any) -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        def _run(rgbs: list[Image.Image]) -> list[dict[str, Any]]:
            px = processor(images=rgbs, return_tensors="pt")["pixel_values"].to(dev)
            with torch.inference_mode():
                pooled = detector.swinv2(pixel_values=px).pooler_output
                p_fake = (detector.classifier(pooled) / 2.0).softmax(-1)[:, 1]
                emb = f_nn.normalize(head(f_nn.normalize(pooled)))
                fam_idx = (emb @ protos.T).argmax(dim=-1)
            return [
                {
                    "pred_ai": float(p_val) >= 0.5,
                    "pred_provider": (
                        OPENFAKE_FAMILY_MAP.get(families[int(f_i)], f"other:{families[int(f_i)]}")
                        if float(p_val) >= 0.5
                        else None
                    ),
                }
                for p_val, f_i in zip(p_fake.cpu().tolist(), fam_idx.cpu().tolist(), strict=True)
            ]

        return _run

    def _move_to_acc() -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        dev = _select_torch_accelerator()
        detector.to(dev)
        head.to(dev)
        return _score_on(dev, family_protos_cpu.to(dev))

    return _evaluate_rgb_detector(
        samples,
        rgb_cache,
        score_cpu_batch=_score_on(torch.device("cpu"), family_protos_cpu),
        make_batch_scorer=_move_to_acc,
        model_id="bendavidsteel/openfake-swinv2-base + attribution-head",
        revision=f"{OPENFAKE_BASE_REVISION[:10]}+{OPENFAKE_ATTR_REVISION[:10]}",
        runtime="PyTorch + Transformers",
        weights_mb=attr_mb,
        license_id="cc-by-nc-4.0",
        tier="frontier",
    )


def evaluate_openfake_textgraphics_lora(
    samples: list[SampleItem],
    rgb_cache: list[Image.Image] | None = None,
) -> dict[str, Any]:
    """Evaluate OpenFake SwinV2-base with the McGill text-graphics false-positive LoRA adapter."""
    import torch
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    base_st = Path(hf_hub_download(OPENFAKE_BASE_REPO, "model.safetensors", revision=OPENFAKE_BASE_REVISION))
    lora_st = Path(hf_hub_download(OPENFAKE_LORA_REPO, "adapter_model.safetensors", revision=OPENFAKE_LORA_REVISION))
    lora_mb = round((base_st.stat().st_size + lora_st.stat().st_size) / (1024 * 1024), 2)

    processor = AutoImageProcessor.from_pretrained(OPENFAKE_BASE_REPO, revision=OPENFAKE_BASE_REVISION)
    base_model = AutoModelForImageClassification.from_pretrained(OPENFAKE_BASE_REPO, revision=OPENFAKE_BASE_REVISION)
    lora_model = PeftModel.from_pretrained(base_model, OPENFAKE_LORA_REPO, revision=OPENFAKE_LORA_REVISION)
    lora_model.train(False)
    lora_model.to("cpu")

    def _score_on(dev: Any) -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        def _run(rgbs: list[Image.Image]) -> list[dict[str, Any]]:
            inputs = {k: v.to(dev) for k, v in processor(images=rgbs, return_tensors="pt").items()}
            with torch.inference_mode():
                probs = (lora_model(**inputs).logits / 2.0).softmax(-1)[:, 1].cpu().tolist()
            return [{"pred_ai": float(p_val) >= 0.5, "pred_provider": None} for p_val in probs]

        return _run

    def _move_to_acc() -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        dev = _select_torch_accelerator()
        lora_model.to(dev)
        return _score_on(dev)

    return _evaluate_rgb_detector(
        samples,
        rgb_cache,
        score_cpu_batch=_score_on(torch.device("cpu")),
        make_batch_scorer=_move_to_acc,
        model_id="bendavidsteel/openfake-swinv2-base + textgraphics-lora",
        revision=f"{OPENFAKE_BASE_REVISION[:10]}+{OPENFAKE_LORA_REVISION[:10]}",
        runtime="PyTorch + PEFT",
        weights_mb=lora_mb,
        license_id="cc-by-nc-4.0",
        tier="frontier",
    )


def evaluate_smaug(
    samples: list[SampleItem],
    rgb_cache: list[Image.Image] | None = None,
) -> dict[str, Any]:
    """Evaluate `abacusai/Smaug-AI-Image-Detector` (EmbeddingGemma-2 439M, 3x 528px crops)."""
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    snap = Path(snapshot_download(SMAUG_REPO, revision=SMAUG_REVISION))
    weights_bytes = sum(p.stat().st_size for p in snap.rglob("*.safetensors") if p.is_file())
    weights_mb = round(weights_bytes / (1024 * 1024), 2)

    st = SentenceTransformer(
        str(snap),
        device="cpu",
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    ip = st[0].processor.image_processor
    ip.max_soft_tokens = ip.image_seq_length = 140

    def _score_batch(rgbs: list[Image.Image]) -> list[dict[str, Any]]:
        flat = [{"image": crop} for rgb in rgbs for crop in _smaug_views(rgb)]
        with torch.inference_mode():
            logits = st.encode(flat, batch_size=16, convert_to_tensor=True, show_progress_bar=False).float()
        probs = torch.sigmoid(logits.view(len(rgbs), 3).mean(dim=1)).cpu().tolist()
        return [{"pred_ai": float(p_val) >= 0.5, "pred_provider": None} for p_val in probs]

    def _move_to_acc() -> Callable[[list[Image.Image]], list[dict[str, Any]]]:
        dev = _select_torch_accelerator()
        if dev.type == "cuda":
            st.to(device=dev, dtype=torch.bfloat16)
        elif dev.type != "cpu":
            st.to(device=dev)
        return _score_batch

    return _evaluate_rgb_detector(
        samples,
        rgb_cache,
        score_cpu_batch=_score_batch,
        make_batch_scorer=_move_to_acc,
        batch_size=8,
        latency_count=SMAUG_CPU_LATENCY_COUNT,
        model_id=SMAUG_REPO,
        revision=SMAUG_REVISION,
        runtime="PyTorch + SentenceTransformers",
        weights_mb=weights_mb,
        license_id="apache-2.0",
        tier="frontier",
    )


def _summarize_ai_subset(
    ai_pairs: list[tuple[SampleItem, dict[str, Any]]],
    *,
    non_ai_fp: int,
) -> dict[str, Any]:
    per_prov: dict[str, dict[str, int]] = {}
    tp = 0
    exact_provider_hits = 0
    total = len(ai_pairs)

    for item, pred in ai_pairs:
        flagged = bool(pred["pred_ai"])
        prov = item.expected_provider or "unknown"
        bucket = per_prov.setdefault(prov, {"count": 0, "detected_ai": 0, "exact_provider": 0})
        bucket["count"] += 1
        if flagged:
            tp += 1
            bucket["detected_ai"] += 1
        if pred["pred_provider"] is not None and pred["pred_provider"] == item.expected_provider:
            exact_provider_hits += 1
            bucket["exact_provider"] += 1

    precision = (tp / (tp + non_ai_fp) * 100.0) if (tp + non_ai_fp) else 0.0
    recall = (tp / total * 100.0) if total else 0.0
    attr_acc = (exact_provider_hits / total * 100.0) if total else 0.0
    attr_when_classified = (exact_provider_hits / tp * 100.0) if tp else 0.0
    per_provider_summary = {
        prov: {
            "count": b["count"],
            "detected_ai": b["detected_ai"],
            "exact_provider": b["exact_provider"],
            "recall_pct": round(b["detected_ai"] / b["count"] * 100.0, 1),
            "attribution_pct": round(b["exact_provider"] / b["count"] * 100.0, 1),
        }
        for prov, b in sorted(per_prov.items())
    }
    return {
        "total": total,
        "tp": tp,
        "recall_pct": round(recall, 1),
        "precision_pct": round(precision, 1),
        "provider_attribution_pct": round(attr_acc, 1),
        "provider_attribution_when_classified_pct": round(attr_when_classified, 1),
        "per_provider": per_provider_summary,
    }


def _summarize_predictions(
    *,
    model_id: str,
    revision: str,
    runtime: str,
    weights_mb: float,
    license_id: str,
    median_cpu_ms: float,
    samples: list[SampleItem],
    preds: list[dict[str, Any]],
    tier: str = "frontier",
) -> dict[str, Any]:
    non_ai_by_cat: dict[str, list[bool]] = {}
    native_ai_pairs: list[tuple[SampleItem, dict[str, Any]]] = []
    arena_ai_pairs: list[tuple[SampleItem, dict[str, Any]]] = []

    for item, pred in zip(samples, preds, strict=True):
        if item.regime == "non_ai":
            non_ai_by_cat.setdefault(item.category, []).append(bool(pred["pred_ai"]))
        elif item.regime == "native":
            native_ai_pairs.append((item, pred))
        elif item.regime == "arena":
            arena_ai_pairs.append((item, pred))
        else:
            raise ValueError(f"Unsupported sample regime: {item.regime!r}")

    non_ai_total = sum(len(flags) for flags in non_ai_by_cat.values())
    fp = sum(sum(flags) for flags in non_ai_by_cat.values())
    fpr = (fp / non_ai_total * 100.0) if non_ai_total else 0.0
    slice_fprs = {cat: round(sum(flags) / len(flags) * 100.0, 1) for cat, flags in non_ai_by_cat.items()}

    all_ai = _summarize_ai_subset(native_ai_pairs + arena_ai_pairs, non_ai_fp=fp)
    native_ai = _summarize_ai_subset(native_ai_pairs, non_ai_fp=fp)
    arena_ai = _summarize_ai_subset(arena_ai_pairs, non_ai_fp=fp)

    return {
        "model_id": model_id,
        "tier": tier,
        "revision": revision,
        "license": license_id,
        "weights_mb": weights_mb,
        "runtime": runtime,
        "median_cpu_ms": median_cpu_ms,
        "non_ai_total": non_ai_total,
        "non_ai_fp": fp,
        "non_ai_fpr_pct": round(fpr, 2),
        "slice_fpr_pct": slice_fprs,
        "ai_total": all_ai["total"],
        "ai_tp": all_ai["tp"],
        "ai_recall_pct": all_ai["recall_pct"],
        "precision_pct": all_ai["precision_pct"],
        "provider_attribution_pct": all_ai["provider_attribution_pct"],
        "provider_attribution_when_classified_pct": all_ai["provider_attribution_when_classified_pct"],
        "per_provider": all_ai["per_provider"],
        "native_ai": native_ai,
        "arena_ai": arena_ai,
    }


def format_markdown_table(results: list[dict[str, Any]]) -> str:
    """Format head-to-head results across non-AI traps, native exports, and transcoded arena images."""
    n_non_ai = results[0]["non_ai_total"] if results else 0
    n_native = results[0]["native_ai"]["total"] if results else 0
    n_arena = results[0]["arena_ai"]["total"] if results else 0

    header = (
        f"| Model | License | Size | CPU ms | Non-AI FPR (`{n_non_ai}`) | Photo / UI / Diag / Doc FPR "
        f"| Native AI Recall (`{n_native}`) | Native Attribution (`{n_native}`) "
        f"| Arena AI Recall (`{n_arena}`) | Arena Attribution (`{n_arena}`) | Precision |"
    )
    sep = "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in results:
        sf = r["slice_fpr_pct"]
        nat = r["native_ai"]
        arn = r["arena_ai"]
        slice_str = (
            f"{sf.get('camera_photo', 0.0):.0f}% / {sf.get('ui_screenshot', 0.0):.0f}% / "
            f"{sf.get('science_diagram', 0.0):.0f}% / {sf.get('scanned_document', 0.0):.0f}%"
        )
        lines.append(
            f"| `{r['model_id']}` | `{r['license']}` | {r['weights_mb']:.1f} MB | {r['median_cpu_ms']:.0f} ms "
            f"| {r['non_ai_fpr_pct']:.1f}% ({r['non_ai_fp']}/{r['non_ai_total']}) "
            f"| {slice_str} "
            f"| {nat['recall_pct']:.1f}% "
            f"| {nat['provider_attribution_pct']:.1f}% "
            f"| {arn['recall_pct']:.1f}% "
            f"| {arn['provider_attribution_pct']:.1f}% "
            f"| {r['precision_pct']:.1f}% |"
        )
    return "\n".join(lines)


def _format_buckets(items: list[SampleItem]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for item in items:
        prov = item.expected_provider or item.category
        fmap = out.setdefault(prov, {})
        key = f"{item.container_format}:{item.width}x{item.height}"
        fmap[key] = fmap.get(key, 0) + 1
    return out


def main() -> int:
    from huggingface_hub import HfApi, HfFileSystem

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="Optional path to write full JSON benchmark report.")
    parser.add_argument("--raiw-only", action="store_true", help="Run only wiltodelta/raiw-source-classify.")
    parser.add_argument(
        "--tier",
        choices=("all", "frontier", "legacy"),
        default="all",
        help="Baseline tier to run alongside wiltodelta/raiw-source-classify (default: all).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fs = HfFileSystem()
    api = HfApi()

    all_samples = load_all_samples(fs, api)
    non_ai = [s for s in all_samples if s.regime == "non_ai"]
    native_ai = [s for s in all_samples if s.regime == "native"]
    arena_ai = [s for s in all_samples if s.regime == "arena"]
    log.info(
        "Evaluating %s total images (%s non-AI + %s native AI + %s transcoded arena AI)",
        len(all_samples),
        len(non_ai),
        len(native_ai),
        len(arena_ai),
    )

    results: list[dict[str, Any]] = [evaluate_raiw(all_samples)]
    _release_memory()

    if not args.raiw_only:
        with ThreadPoolExecutor(max_workers=8) as pool:
            rgb_cache = list(pool.map(lambda s: _decode_rgb(s.raw_bytes), all_samples))

        evaluators: list[tuple[str, Callable[[], dict[str, Any]]]] = []
        if args.tier in ("all", "frontier"):
            evaluators.extend(
                [
                    (SMAUG_REPO, lambda: evaluate_smaug(all_samples, rgb_cache)),
                    (
                        COMMFOR_DEEPFAKEDET_SPEC.repo_id,
                        lambda: evaluate_hf_binary_detector(COMMFOR_DEEPFAKEDET_SPEC, all_samples, rgb_cache),
                    ),
                    *[
                        (
                            vit_spec.repo_id,
                            lambda vs=vit_spec: evaluate_timm_vit_detector(vs, all_samples, rgb_cache),
                        )
                        for vit_spec in FRONTIER_VIT_DETECTORS
                    ],
                    (DETECTRA_OPENFAKE_REPO, lambda: evaluate_detectra_openfake(all_samples, rgb_cache)),
                    (
                        "bendavidsteel/openfake-swinv2-base + attribution-head",
                        lambda: evaluate_openfake_attribution(all_samples, rgb_cache),
                    ),
                    (
                        "bendavidsteel/openfake-swinv2-base + textgraphics-lora",
                        lambda: evaluate_openfake_textgraphics_lora(all_samples, rgb_cache),
                    ),
                ]
            )
        if args.tier in ("all", "legacy"):
            evaluators.extend(
                [
                    (spec.repo_id, lambda sp=spec: evaluate_hf_binary_detector(sp, all_samples, rgb_cache))
                    for spec in LEGACY_BINARY_DETECTORS
                ]
            )

        for label, run_eval in evaluators:
            log.info("Evaluating %s ...", label)
            results.append(run_eval())
            _release_memory()
        del rgb_cache
        _release_memory()

    md_table = format_markdown_table(results)
    sys.stdout.write("\n" + md_table + "\n")

    if args.json_out is not None:
        native_formats = _format_buckets(native_ai)
        arena_formats = _format_buckets(arena_ai)
        payload = {
            "non_ai_count": len(non_ai),
            "native_ai_count": len(native_ai),
            "arena_ai_count": len(arena_ai),
            "native_ai_providers": {p: sum(m.values()) for p, m in sorted(native_formats.items())},
            "native_ai_formats": native_formats,
            "arena_ai_providers": {p: sum(m.values()) for p, m in sorted(arena_formats.items())},
            "arena_ai_formats": arena_formats,
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.info("Wrote JSON report to %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
