#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#     "remove-ai-watermarks[source-classify]>=0.62.1",
#     "huggingface-hub>=0.26.0",
#     "pyarrow>=17.0.0",
#     "torch>=2.2.0",
#     "transformers>=4.46.0",
# ]
# ///
"""Reproducible open benchmark: `wiltodelta/raiw-source-classify` vs top Hugging Face binary AI detectors.

Streams pinned, un-gated Parquet row groups directly from Hugging Face Hub over
HTTP byte-ranges (no private files, no manual dataset setup) and evaluates:
1. Non-AI false-positive traps (camera photos, UI screenshots, science diagrams,
   scanned receipts/invoices).
2. Modern AI generator outputs across multiple provider families (OpenAI, Google,
   Black Forest Labs FLUX, ByteDance Seedream, Midjourney, Stable Diffusion 3,
   Ideogram, xAI Grok).
3. Model artifact size, runtime stack, single-image warm CPU latency, binary
   precision/recall/FPR, and exact generator attribution accuracy.

Run directly with `uv`:

    uv run https://huggingface.co/wiltodelta/raiw-source-classify/resolve/main/benchmark.py
    uv run benchmark.py --json-out results.json
    uv run benchmark.py --raiw-only
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageOps

if TYPE_CHECKING:
    from huggingface_hub import HfFileSystem

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if (_REPO_SRC / "remove_ai_watermarks").is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

log = logging.getLogger(__name__)

MIN_SIDE_PX = 256
MAX_AREA_PX = 9_000_000
LATENCY_SAMPLE_COUNT = 40
EXIF_ORIENTATION_TAG = 274
TRANSPOSED_ORIENTATIONS = frozenset({5, 6, 7, 8})


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


@dataclass(frozen=True)
class PreferenceSliceSpec:
    """Pinned Hugging Face pairwise-preference Parquet slice."""

    dataset_id: str
    revision: str
    parquet_path: str
    per_provider_cap: int
    max_row_groups: int | None = None


@dataclass(frozen=True)
class OpenDetectorSpec:
    """Pinned Hugging Face binary AI-image detector checkpoint."""

    repo_id: str
    revision: str
    ai_labels: frozenset[str]
    weights_mb: float
    license_id: str


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
        category="receipt_invoice",
        dataset_id="mychen76/invoices-and-receipts_ocr_v1",
        revision="83835c87346de32ac9223bdce5264e69ef3366ad",
        parquet_path="data/test-00000-of-00001-af2d92d1cee28514.parquet",
        limit=50,
    ),
)

RAPIDATA_4O_SPEC = PreferenceSliceSpec(
    dataset_id="Rapidata/OpenAI-4o_t2i_human_preference",
    revision="ca6c084c0fa0021d66bbdd44d6e61fbeb8e1198a",
    parquet_path="data/train-00000-of-00022.parquet",
    per_provider_cap=35,
)

NANO_BANANA_SPEC = ParquetSliceSpec(
    category="ai_google",
    dataset_id="bitmind/nano-banana",
    revision="9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4",
    parquet_path="data/train-00000-of-00031.parquet",
    limit=25,
    expected_provider="google",
)

SEEDREAM_SPEC = PreferenceSliceSpec(
    dataset_id="Rapidata/Seedream-3_t2i_human_preference",
    revision="0867c34e5583d63376e3289b938e6a2cb6f43751",
    parquet_path="data/train-00000-of-00101.parquet",
    per_provider_cap=25,
    max_row_groups=3,
)

OPEN_BINARY_DETECTORS: tuple[OpenDetectorSpec, ...] = (
    OpenDetectorSpec(
        repo_id="umm-maybe/AI-image-detector",
        revision="c7e223baf11bc40528af364ba7bdea030ef42f9e",
        ai_labels=frozenset({"artificial"}),
        weights_mb=331.5,
        license_id="cc-by-4.0",
    ),
    OpenDetectorSpec(
        repo_id="Organika/sdxl-detector",
        revision="b37fede8562cb72b89ec201c0987f96ba21b518a",
        ai_labels=frozenset({"artificial"}),
        weights_mb=331.4,
        license_id="cc-by-nc-3.0",
    ),
    OpenDetectorSpec(
        repo_id="Ateeqq/ai-vs-human-image-detector",
        revision="60e82406916921b823616bee33397baab38af3f0",
        ai_labels=frozenset({"ai"}),
        weights_mb=354.4,
        license_id="apache-2.0",
    ),
    OpenDetectorSpec(
        repo_id="haywoodsloan/ai-image-detector-deploy",
        revision="4ae1822603cb1ec980c217cfde1ef1b8e0477a16",
        ai_labels=frozenset({"artificial"}),
        weights_mb=744.7,
        license_id="apache-2.0",
    ),
    OpenDetectorSpec(
        repo_id="dima806/ai_vs_human_generated_image_detection",
        revision="341594f2b003d185db789becf48a2286c177b926",
        ai_labels=frozenset({"ai-generated", "artificial", "fake"}),
        weights_mb=327.3,
        license_id="apache-2.0",
    ),
)

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
    "Seedream-3": "bytedance",
}


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


def _try_build_sample(
    cell: Any,
    local_seen: set[str],
    *,
    category: str,
    expected_provider: str | None,
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
    )


def _latency_indices(total: int) -> list[int]:
    """Deterministic stratified indices across the full sample list for CPU timing."""
    if total <= LATENCY_SAMPLE_COUNT:
        return list(range(total))
    step = max(1, total // LATENCY_SAMPLE_COUNT)
    return [i * step for i in range(LATENCY_SAMPLE_COUNT)]


def _read_single_image_slice(fs: HfFileSystem, spec: ParquetSliceSpec) -> list[SampleItem]:
    import pyarrow.parquet as pq

    items: list[SampleItem] = []
    local_seen: set[str] = set()
    uri = f"datasets/{spec.dataset_id}/{spec.parquet_path}"
    with fs.open(uri, "rb", revision=spec.revision) as handle:
        pf = pq.ParquetFile(handle)
        for rg_idx in range(pf.num_row_groups):
            if len(items) >= spec.limit:
                break
            col = pf.read_row_group(rg_idx, columns=["image"])["image"]
            for row_idx in range(col.length()):
                if len(items) >= spec.limit:
                    break
                sample = _try_build_sample(
                    col[row_idx].as_py(),
                    local_seen,
                    category=spec.category,
                    expected_provider=spec.expected_provider,
                )
                if sample is not None:
                    items.append(sample)
    log.info("Loaded %s images from %s (%s)", len(items), spec.dataset_id, spec.category)
    return items


def _read_preference_slice(
    fs: HfFileSystem,
    spec: PreferenceSliceSpec,
    *,
    seedream_only: bool = False,
) -> list[SampleItem]:
    import pyarrow.parquet as pq

    items: list[SampleItem] = []
    local_seen: set[str] = set()
    counts: dict[str, int] = {}
    target_providers = {"bytedance"} if seedream_only else set(RAPIDATA_4O_PROVIDER_MAP.values())
    uri = f"datasets/{spec.dataset_id}/{spec.parquet_path}"

    def _map_tag(tag: str) -> str | None:
        if seedream_only:
            return "bytedance" if "seedream" in tag.lower() else None
        return RAPIDATA_4O_PROVIDER_MAP.get(tag)

    with fs.open(uri, "rb", revision=spec.revision) as handle:
        pf = pq.ParquetFile(handle)
        rg_limit = pf.num_row_groups if spec.max_row_groups is None else min(spec.max_row_groups, pf.num_row_groups)
        for rg_idx in range(rg_limit):
            if all(counts.get(p, 0) >= spec.per_provider_cap for p in target_providers):
                break
            meta_rg = pf.read_row_group(rg_idx, columns=["model1", "model2"])
            m1_list = meta_rg["model1"].to_pylist()
            m2_list = meta_rg["model2"].to_pylist()
            need_col1 = any(
                (p := _map_tag(str(t))) is not None and counts.get(p, 0) < spec.per_provider_cap for t in m1_list
            )
            need_col2 = any(
                (p := _map_tag(str(t))) is not None and counts.get(p, 0) < spec.per_provider_cap for t in m2_list
            )
            img_cols = [c for c, flag in (("image1", need_col1), ("image2", need_col2)) if flag]
            if not img_cols:
                continue
            img_rg = pf.read_row_group(rg_idx, columns=img_cols)
            for row_idx, (t1, t2) in enumerate(zip(m1_list, m2_list, strict=True)):
                for tag, icol in ((str(t1), "image1"), (str(t2), "image2")):
                    prov = _map_tag(tag)
                    if prov is None or counts.get(prov, 0) >= spec.per_provider_cap or icol not in img_cols:
                        continue
                    sample = _try_build_sample(
                        img_rg[icol][row_idx].as_py(),
                        local_seen,
                        category=f"ai_{prov}",
                        expected_provider=prov,
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


def load_all_samples(fs: HfFileSystem) -> tuple[list[SampleItem], list[SampleItem]]:
    """Stream all 7 pinned non-AI and modern AI Parquet shards concurrently in deterministic order."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        non_ai_futures = [pool.submit(_read_single_image_slice, fs, spec) for spec in NON_AI_SLICES]
        fut_4o = pool.submit(_read_preference_slice, fs, RAPIDATA_4O_SPEC, seedream_only=False)
        fut_nb = pool.submit(_read_single_image_slice, fs, NANO_BANANA_SPEC)
        fut_sd = pool.submit(_read_preference_slice, fs, SEEDREAM_SPEC, seedream_only=True)

        seen_shas: set[str] = set()
        non_ai = _dedup_ordered([f.result() for f in non_ai_futures], seen_shas)
        ai_samples = _dedup_ordered([fut_4o.result(), fut_nb.result(), fut_sd.result()], seen_shas)
    return non_ai, ai_samples


def evaluate_raiw(samples: list[SampleItem]) -> dict[str, Any]:
    """Evaluate `wiltodelta/raiw-source-classify` in pure NumPy on CPU."""
    from remove_ai_watermarks import source_classify as sc

    weights_file = sc.resolve_weights_path()
    weights_mb = round(weights_file.stat().st_size / (1024 * 1024), 2)

    def _classify_one(raw_bytes: bytes) -> dict[str, Any]:
        with Image.open(io.BytesIO(raw_bytes)) as im:
            res = sc.classify_source(im, weights_path=weights_file)
        return {
            "pred_ai": res.label != "unknown",
            "pred_provider": None if res.label == "unknown" else res.label,
        }

    _classify_one(samples[0].raw_bytes)

    lat_idx_set = set(_latency_indices(len(samples)))
    latencies_ms: list[float] = []
    preds_by_idx: dict[int, dict[str, Any]] = {}

    for idx in sorted(lat_idx_set):
        t0 = time.perf_counter()
        preds_by_idx[idx] = _classify_one(samples[idx].raw_bytes)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    untimed_indices = [idx for idx in range(len(samples)) if idx not in lat_idx_set]
    with ThreadPoolExecutor(max_workers=4) as pool:
        untimed_preds = list(pool.map(lambda i: _classify_one(samples[i].raw_bytes), untimed_indices))
    for idx, pred in zip(untimed_indices, untimed_preds, strict=True):
        preds_by_idx[idx] = pred

    preds = [preds_by_idx[i] for i in range(len(samples))]
    return _summarize_predictions(
        model_id=sc.WEIGHTS_REPO,
        revision=sc.WEIGHTS_REVISION,
        runtime="NumPy + OpenCV (CPU)",
        weights_mb=weights_mb,
        license_id="apache-2.0",
        median_cpu_ms=round(statistics.median(latencies_ms), 1),
        samples=samples,
        preds=preds,
    )


def evaluate_hf_binary_detector(spec: OpenDetectorSpec, samples: list[SampleItem]) -> dict[str, Any]:
    """Evaluate an open Hugging Face `transformers` binary AI-image detector."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(spec.repo_id, revision=spec.revision)
    model = AutoModelForImageClassification.from_pretrained(spec.repo_id, revision=spec.revision)
    model.eval()
    model.to("cpu")

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    ai_indices = [idx for idx, lbl in id2label.items() if lbl in spec.ai_labels]
    if not ai_indices:
        raise RuntimeError(f"Could not map AI label for {spec.repo_id}: {id2label}")

    rgb0 = _decode_rgb(samples[0].raw_bytes)
    with torch.inference_mode():
        _ = model(**processor(images=rgb0, return_tensors="pt"))

    lat_idxs = _latency_indices(len(samples))
    lat_idx_set = set(lat_idxs)
    latencies_ms: list[float] = []
    preds_by_idx: dict[int, dict[str, Any]] = {}

    for idx in lat_idxs:
        t0 = time.perf_counter()
        rgb = _decode_rgb(samples[idx].raw_bytes)
        inputs = processor(images=rgb, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        probs_1 = torch.softmax(logits, dim=-1)[0]
        p_ai = float(sum(probs_1[ai_idx].item() for ai_idx in ai_indices))
        preds_by_idx[idx] = {"pred_ai": p_ai >= 0.5, "pred_provider": None}

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    untimed_indices = [idx for idx in range(len(samples)) if idx not in lat_idx_set]
    batch_size = 16
    for start in range(0, len(untimed_indices), batch_size):
        chunk_idxs = untimed_indices[start : start + batch_size]
        rgbs = [_decode_rgb(samples[idx].raw_bytes) for idx in chunk_idxs]
        inputs = {k: v.to(device) for k, v in processor(images=rgbs, return_tensors="pt").items()}
        with torch.inference_mode():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu()
        for row_idx, sample_idx in enumerate(chunk_idxs):
            p_ai = float(sum(probs[row_idx, ai_idx].item() for ai_idx in ai_indices))
            preds_by_idx[sample_idx] = {"pred_ai": p_ai >= 0.5, "pred_provider": None}

    preds = [preds_by_idx[i] for i in range(len(samples))]
    return _summarize_predictions(
        model_id=spec.repo_id,
        revision=spec.revision,
        runtime="PyTorch + Transformers",
        weights_mb=spec.weights_mb,
        license_id=spec.license_id,
        median_cpu_ms=round(statistics.median(latencies_ms), 1),
        samples=samples,
        preds=preds,
    )


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
) -> dict[str, Any]:
    non_ai_by_cat: dict[str, list[bool]] = {}
    per_prov: dict[str, dict[str, int]] = {}
    tp = fp = tn = 0
    exact_provider_hits = 0
    ai_total = 0

    for item, pred in zip(samples, preds, strict=True):
        flagged = bool(pred["pred_ai"])
        if item.is_ai:
            ai_total += 1
            prov = item.expected_provider or "unknown"
            bucket = per_prov.setdefault(prov, {"count": 0, "detected_ai": 0, "exact_provider": 0})
            bucket["count"] += 1
            if flagged:
                tp += 1
                bucket["detected_ai"] += 1
            if pred["pred_provider"] is not None and pred["pred_provider"] == item.expected_provider:
                exact_provider_hits += 1
                bucket["exact_provider"] += 1
        else:
            non_ai_by_cat.setdefault(item.category, []).append(flagged)
            if flagged:
                fp += 1
            else:
                tn += 1

    non_ai_total = fp + tn
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) else 0.0
    recall = (tp / ai_total * 100.0) if ai_total else 0.0
    fpr = (fp / non_ai_total * 100.0) if non_ai_total else 0.0
    attr_acc = (exact_provider_hits / ai_total * 100.0) if ai_total else 0.0
    attr_when_classified = (exact_provider_hits / tp * 100.0) if tp else 0.0

    slice_fprs = {cat: round(sum(flags) / len(flags) * 100.0, 1) for cat, flags in non_ai_by_cat.items()}
    per_provider_summary = {
        prov: {
            "count": b["count"],
            "detected_ai": b["detected_ai"],
            "exact_provider": b["exact_provider"],
            "recall_pct": round(b["detected_ai"] / b["count"] * 100.0, 1) if b["count"] else 0.0,
            "attribution_pct": round(b["exact_provider"] / b["count"] * 100.0, 1) if b["count"] else 0.0,
        }
        for prov, b in sorted(per_prov.items())
    }
    return {
        "model_id": model_id,
        "revision": revision,
        "license": license_id,
        "weights_mb": weights_mb,
        "runtime": runtime,
        "median_cpu_ms": median_cpu_ms,
        "non_ai_total": non_ai_total,
        "non_ai_fp": fp,
        "non_ai_fpr_pct": round(fpr, 2),
        "slice_fpr_pct": slice_fprs,
        "ai_total": ai_total,
        "ai_tp": tp,
        "ai_recall_pct": round(recall, 1),
        "precision_pct": round(precision, 1),
        "provider_attribution_pct": round(attr_acc, 1),
        "provider_attribution_when_classified_pct": round(attr_when_classified, 1),
        "per_provider": per_provider_summary,
    }


def format_markdown_table(results: list[dict[str, Any]]) -> str:
    """Format head-to-head results as a GitHub-flavored Markdown table."""
    header = (
        "| Model | Size | Runtime | CPU ms/img | Non-AI FPR (all) | Photo FPR "
        "| UI Screenshot FPR | Diagram FPR | Receipt FPR | AI Recall | Precision | Generator Attribution |"
    )
    sep = "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in results:
        sf = r["slice_fpr_pct"]
        lines.append(
            f"| `{r['model_id']}` | {r['weights_mb']:.1f} MB | {r['runtime']} | {r['median_cpu_ms']:.1f} ms "
            f"| {r['non_ai_fpr_pct']:.1f}% ({r['non_ai_fp']}/{r['non_ai_total']}) "
            f"| {sf.get('camera_photo', 0.0):.1f}% "
            f"| {sf.get('ui_screenshot', 0.0):.1f}% "
            f"| {sf.get('science_diagram', 0.0):.1f}% "
            f"| {sf.get('receipt_invoice', 0.0):.1f}% "
            f"| {r['ai_recall_pct']:.1f}% "
            f"| {r['precision_pct']:.1f}% "
            f"| {r['provider_attribution_pct']:.1f}% |"
        )
    return "\n".join(lines)


def main() -> int:
    from huggingface_hub import HfFileSystem

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="Optional path to write full JSON benchmark report.")
    parser.add_argument("--raiw-only", action="store_true", help="Run only wiltodelta/raiw-source-classify.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fs = HfFileSystem()

    non_ai, ai_samples = load_all_samples(fs)
    all_samples = non_ai + ai_samples
    log.info("Evaluating %s total images (%s non-AI + %s AI)", len(all_samples), len(non_ai), len(ai_samples))

    results: list[dict[str, Any]] = [evaluate_raiw(all_samples)]
    if not args.raiw_only:
        for spec in OPEN_BINARY_DETECTORS:
            log.info("Evaluating %s ...", spec.repo_id)
            results.append(evaluate_hf_binary_detector(spec, all_samples))

    md_table = format_markdown_table(results)
    sys.stdout.write("\n" + md_table + "\n")

    if args.json_out is not None:
        format_counts: dict[str, dict[str, int]] = {}
        for x in ai_samples:
            prov = x.expected_provider or "unknown"
            fmap = format_counts.setdefault(prov, {})
            key = f"{x.container_format}:{x.width}x{x.height}"
            fmap[key] = fmap.get(key, 0) + 1
        payload = {
            "non_ai_count": len(non_ai),
            "ai_count": len(ai_samples),
            "ai_providers": {prov: sum(fmap.values()) for prov, fmap in sorted(format_counts.items())},
            "ai_provider_formats": format_counts,
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        log.info("Wrote JSON report to %s", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
