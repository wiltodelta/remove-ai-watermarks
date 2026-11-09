"""Abstaining 10-provider pixel source-pipeline classifier with Stage-2 version attribution.

This classifier identifies the likely AI generation and export pixel pipeline across
10 provider families (``openai``, ``google``, ``bytedance``, ``flux``,
``stable_diffusion``, ``meta``, ``midjourney``, ``ideogram``, ``firefly``, ``xai``),
returning ``unknown`` when no provider's damped margin clears its calibrated
threshold. It is not a provenance verifier and does not detect or verify SynthID or
any invisible watermark.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import math
import os
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from remove_ai_watermarks.optional_deps import module_available

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PIL import Image

    FloatArray: TypeAlias = NDArray[Any]
else:
    FloatArray = Any

log = logging.getLogger(__name__)

ProviderClass = Literal[
    "openai",
    "google",
    "bytedance",
    "flux",
    "stable_diffusion",
    "meta",
    "midjourney",
    "ideogram",
    "firefly",
    "xai",
]
SourceLabel = Literal[
    "openai",
    "google",
    "bytedance",
    "flux",
    "stable_diffusion",
    "meta",
    "midjourney",
    "ideogram",
    "firefly",
    "xai",
    "unknown",
]
SourceReason = Literal["classified", "abstained", "feature_unavailable"]

ALL_PROVIDER_CLASSES: tuple[ProviderClass, ...] = (
    "openai",
    "google",
    "bytedance",
    "flux",
    "stable_diffusion",
    "meta",
    "midjourney",
    "ideogram",
    "firefly",
    "xai",
)
CLASSES: tuple[SourceLabel, ...] = (*ALL_PROVIDER_CLASSES, "unknown")

WEIGHTS_ENV = "RAIW_SOURCE_CLASSIFY_WEIGHTS"
WEIGHTS_REPO = "wiltodelta/raiw-source-classify"
WEIGHTS_REVISION = "32a52ed694f6287d5de35beb0c030e0b1b57f90c"
MODEL_FILE = "multi-provider-source-v3.npz"
MODEL_SHA256 = "f65dac7cd7a6d8025e0b849bba9117635d6280628c226f47b599332898580f3c"
MODEL_SCHEMA = "multi-provider-source-v3"
SOURCE_CLASSIFY_EXTRA = "'remove-ai-watermarks[source-classify]'"

MIN_SIDE = 256
FORENSIC_DIM = 124
SPECTRAL_GRID = 16
SPECTRAL_DIM = 3 * SPECTRAL_GRID * SPECTRAL_GRID
BASE_FEATURE_DIM = FORENSIC_DIM + SPECTRAL_DIM
EXTRA_FEATURE_DIM = 64
FEATURE_DIM = BASE_FEATURE_DIM + EXTRA_FEATURE_DIM

CLAIM = (
    "Identifies the likely AI generation/export pixel pipeline across 10 provider "
    "families with positive-part sub-unit pooling, damped cross-provider margin "
    "suppression, and stacked Stage-2 version attribution. This is not provenance "
    "verification and does not detect or verify SynthID or any invisible watermark."
)


@dataclass(frozen=True)
class SourceClassification:
    """Result of abstaining 10-provider source-pipeline classification."""

    label: SourceLabel
    version_subfamily: str | None
    fine_subfamily: str | None
    reason: SourceReason
    confidence: float
    pooled_scores: dict[str, float]
    margins: dict[str, float]
    thresholds: dict[str, float]
    watermark_truth: Literal["unknown"] = "unknown"
    claim: str = CLAIM
    model_revision: str = MODEL_SCHEMA

    @property
    def scores(self) -> dict[str, float]:
        """Return provider pooled scores."""
        return self.pooled_scores

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the result."""
        return {
            "label": self.label,
            "version_subfamily": self.version_subfamily,
            "fine_subfamily": self.fine_subfamily,
            "reason": self.reason,
            "confidence": self.confidence,
            "pooled_scores": dict(self.pooled_scores),
            "margins": dict(self.margins),
            "thresholds": dict(self.thresholds),
            "watermark_truth": self.watermark_truth,
            "claim": self.claim,
            "model_revision": self.model_revision,
        }


@dataclass(frozen=True)
class _MLPBlock:
    """Three-layer GELU MLP weight block."""

    w0: FloatArray
    b0: FloatArray
    w1: FloatArray
    b1: FloatArray
    w2: FloatArray
    b2: FloatArray


@dataclass(frozen=True)
class _PackedBinaryHGB:
    """Self-looping fixed-depth binary HistGradientBoosting tree ensemble."""

    baseline: float
    max_depth: int
    feat_idx: FloatArray
    thresholds: FloatArray
    left: FloatArray
    right: FloatArray
    values: FloatArray


@dataclass(frozen=True)
class _Stage1Unit:
    """One binary specialist or subfamily sub-specialist unit."""

    blend_weights: tuple[float, float, float]
    base_offset: float
    pool_alpha: float
    mlp_soup: tuple[_MLPBlock, ...]
    et_mlp: _MLPBlock
    hgb: _PackedBinaryHGB


@dataclass(frozen=True)
class _ProviderSpecialist:
    """Provider specialist combining one global unit and zero or more subfamily units."""

    global_unit: _Stage1Unit
    sub_units: dict[str, _Stage1Unit]
    pool_recenter: float


@dataclass(frozen=True)
class _Stage2VersionHead:
    """Stacked Stage-2 version classifier over 956-d features + 47 Stage-1 unit logits."""

    subfamilies: tuple[str, ...]
    baseline: FloatArray
    max_depth: int
    feat_idx: FloatArray
    thresholds: FloatArray
    left: FloatArray
    right: FloatArray
    values: FloatArray


@dataclass(frozen=True)
class _SourceModel:
    """Validated pickle-free multi-provider v3 model."""

    mean: FloatArray
    scale: FloatArray
    damping_gamma: float
    router_thresholds: dict[str, float]
    specialists: dict[str, _ProviderSpecialist]
    unit_descriptors: tuple[tuple[str, str], ...]
    version_heads: dict[str, _Stage2VersionHead]
    router_meta_tau: FloatArray | None = None
    router_meta_w: FloatArray | None = None


def is_available() -> bool:
    """Return True if optional source-classifier dependencies are installed."""
    return module_available("numpy", "cv2", "huggingface_hub")


def _require_dependencies() -> None:
    if not is_available():
        raise ImportError(
            "Source-pipeline classification requires opencv-python-headless and "
            f"huggingface-hub. Install with: pip install {SOURCE_CLASSIFY_EXTRA}"
        )


def _verify_sha256(path: Path, expected_sha256: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Source classifier weights hash mismatch for {path}: expected {expected_sha256}, got {actual}"
        )


def _resolve_weights_path(weights_path: Path | None = None) -> Path:
    if weights_path is not None:
        path = Path(weights_path).expanduser()
        return path / MODEL_FILE if path.is_dir() else path
    override = os.environ.get(WEIGHTS_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        candidate = path / MODEL_FILE if path.is_dir() else path
        if not candidate.is_file():
            raise RuntimeError(f"{WEIGHTS_ENV} does not contain {MODEL_FILE}: {path}")
        return candidate
    _require_dependencies()
    hub: Any = import_module("huggingface_hub")
    downloaded = Path(
        str(
            hub.hf_hub_download(
                repo_id=WEIGHTS_REPO,
                filename=MODEL_FILE,
                revision=WEIGHTS_REVISION,
            )
        )
    )
    _verify_sha256(downloaded, MODEL_SHA256)
    return downloaded


@functools.lru_cache(maxsize=1)
def _spectral_grid_masks() -> tuple[
    tuple[FloatArray, FloatArray, FloatArray, FloatArray],
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    import numpy as np

    yy, xx = np.meshgrid(np.arange(SPECTRAL_GRID), np.arange(SPECTRAL_GRID), indexing="ij")
    rr = np.sqrt(yy**2 + xx**2)
    b0 = rr < 4.0
    b1 = (rr >= 4.0) & (rr < 8.0)
    b2 = (rr >= 8.0) & (rr < 12.0)
    b3 = rr >= 12.0
    axis_mask = (yy <= 1) | (xx <= 1)
    diag_mask = (np.abs(yy - xx) <= 2) & (rr >= 4.0)
    b_mid = b1 | b2
    return (b0, b1, b2, b3), axis_mask & b_mid, diag_mask & b_mid, axis_mask & b3, diag_mask & b3


def augment_892d_to_956d(x_892: FloatArray) -> FloatArray:
    """Deterministically expand 892-d (124-d forensic + 768-d spectral) to 956-d."""
    import numpy as np

    arr = np.asarray(x_892, dtype=np.float64)
    single = arr.ndim == 1
    mat = arr[None, :] if single else arr
    if mat.ndim != 2 or mat.shape[1] != BASE_FEATURE_DIM:
        raise ValueError(f"Expected 892-d feature input, got shape {arr.shape}")

    n = mat.shape[0]
    fore = mat[:, :FORENSIC_DIM]
    spec = mat[:, FORENSIC_DIM:BASE_FEATURE_DIM].reshape(n, 3, SPECTRAL_GRID, SPECTRAL_GRID)

    bands, ax_mid_mask, dg_mid_mask, ax_hi_mask, dg_hi_mask = _spectral_grid_masks()
    b0, b1, b2, b3 = bands

    extra: list[FloatArray] = []

    # 1. Cross-channel spectral differences across 4 radial bands (24 features)
    for c1, c2 in ((0, 1), (0, 2), (1, 2)):
        diff_c = spec[:, c1, :, :] - spec[:, c2, :, :]
        for b_mask in bands:
            vals = diff_c[:, b_mask]
            extra.append(np.mean(vals, axis=1))
            extra.append(np.std(vals, axis=1))

    # 2. Per-channel radial slope, curvature & Nyquist 8x8 boundary jumps (18 features)
    for c in range(3):
        sc = spec[:, c, :, :]
        m0 = np.mean(sc[:, b0], axis=1)
        m1 = np.mean(sc[:, b1], axis=1)
        m2 = np.mean(sc[:, b2], axis=1)
        m3 = np.mean(sc[:, b3], axis=1)
        extra.append(m3 - m2)
        extra.append(m3 - m1)
        extra.append(m2 - m0)
        extra.append(m3 - 2.0 * m2 + m1)
        nyq_y = np.mean(sc[:, 14:16, :], axis=(1, 2)) - np.mean(sc[:, 11:13, :], axis=(1, 2))
        nyq_x = np.mean(sc[:, :, 14:16], axis=(1, 2)) - np.mean(sc[:, :, 11:13], axis=(1, 2))
        extra.append(nyq_y)
        extra.append(nyq_x)

    # 3. Azimuthal anisotropy across 3 channels (12 features)
    for c in range(3):
        sc = spec[:, c, :, :]
        ax_mid = np.mean(sc[:, ax_mid_mask], axis=1)
        dg_mid = np.mean(sc[:, dg_mid_mask], axis=1)
        ax_hi = np.mean(sc[:, ax_hi_mask], axis=1)
        dg_hi = np.mean(sc[:, dg_hi_mask], axis=1)
        horiz_hi = np.mean(sc[:, :3, 8:], axis=(1, 2))
        vert_hi = np.mean(sc[:, 8:, :3], axis=(1, 2))
        extra.append(ax_mid - dg_mid)
        extra.append(ax_hi - dg_hi)
        extra.append(horiz_hi - vert_hi)
        extra.append(np.abs(horiz_hi - vert_hi))

    # 4. Forensic-to-spectral interaction terms (10 features)
    spec_l_hi = np.mean(spec[:, 0, b3], axis=1)
    spec_rg_hi = np.mean(spec[:, 1, b3], axis=1)
    spec_yb_hi = np.mean(spec[:, 2, b3], axis=1)
    spec_l_lo = np.mean(spec[:, 0, b0], axis=1)
    extra.append(spec_l_hi - 0.5 * (spec_rg_hi + spec_yb_hi))
    extra.append(spec_rg_hi - spec_yb_hi)
    extra.append((spec_l_hi - spec_l_lo) / (np.std(spec[:, 0, :, :], axis=(1, 2)) + 1e-5))
    for f_idx in range(7):
        extra.append(fore[:, f_idx] * (spec_l_hi - spec_rg_hi))

    extra_mat = np.stack(extra, axis=1)
    combined = np.concatenate([mat, extra_mat], axis=1)
    return combined[0] if single else combined


def extract_956d_features(image: Image.Image) -> FloatArray | None:
    """Compute the 956-d forensic + spectral + chroma/grid vector, or None if < 256x256."""
    import numpy as np

    from remove_ai_watermarks._internal.forensic_124d import image_features
    from remove_ai_watermarks._internal.source_spectral import feature_from_image, make_geometry

    if min(image.size) < MIN_SIDE:
        return None
    rgb_image = image if image.mode == "RGB" else image.convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[0], rgb.shape[1]) < MIN_SIDE:
        return None
    forensic = image_features(rgb)
    if forensic is None:
        return None
    spectral = feature_from_image(rgb_image, make_geometry(grid=SPECTRAL_GRID))
    vec_892 = np.concatenate([forensic.astype(np.float64), spectral.astype(np.float64)])
    if vec_892.shape != (BASE_FEATURE_DIM,) or not np.all(np.isfinite(vec_892)):
        return None
    vec_956 = augment_892d_to_956d(vec_892)
    if vec_956.shape != (FEATURE_DIM,) or not np.all(np.isfinite(vec_956)):
        return None
    return vec_956


def _gelu_np(x: FloatArray) -> FloatArray:
    import numpy as np

    return (0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))).astype(np.float32)


def _logit_scalar(prob: float) -> float:
    clipped = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid_scalar(val: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(val)))))


def _eval_mlp_block(block: _MLPBlock, x_std_1d: FloatArray) -> float:
    h0 = _gelu_np(x_std_1d @ block.w0.T + block.b0)
    h1 = _gelu_np(h0 @ block.w1.T + block.b1)
    out = h1 @ block.w2.T + block.b2
    return max(-14.0, min(14.0, float(out.squeeze())))


def _eval_packed_binary_hgb(hgb: _PackedBinaryHGB, x_1d: FloatArray) -> float:
    import numpy as np

    n_trees = hgb.feat_idx.shape[0]
    t_arange = np.arange(n_trees, dtype=np.int32)
    node_idx = np.zeros(n_trees, dtype=np.int32)
    for _ in range(hgb.max_depth):
        f = hgb.feat_idx[t_arange, node_idx]
        thr = hgb.thresholds[t_arange, node_idx]
        node_idx = np.where(
            x_1d[f] <= thr,
            hgb.left[t_arange, node_idx],
            hgb.right[t_arange, node_idx],
        )
    raw = hgb.baseline + float(np.sum(hgb.values[t_arange, node_idx]))
    prob = _sigmoid_scalar(raw)
    return _logit_scalar(prob)


def _score_stage1_unit(unit: _Stage1Unit, x_1d: FloatArray, x_std_1d: FloatArray) -> float:
    w_m, w_h, w_e = unit.blend_weights
    mlp_l = (
        sum(_eval_mlp_block(b, x_std_1d) for b in unit.mlp_soup) / len(unit.mlp_soup)
        if unit.mlp_soup and w_m > 0.0
        else 0.0
    )
    hgb_l = _eval_packed_binary_hgb(unit.hgb, x_1d) if w_h > 0.0 else 0.0
    et_l = _eval_mlp_block(unit.et_mlp, x_std_1d) if w_e > 0.0 else 0.0
    return w_m * mlp_l + w_h * hgb_l + w_e * et_l


def _eval_stage2_version_head(head: _Stage2VersionHead, x_stacked_1d: FloatArray) -> str:
    import numpy as np

    n_iter, c_tree, _ = head.feat_idx.shape
    i_arange = np.arange(n_iter, dtype=np.int32)[:, None]
    c_arange = np.arange(c_tree, dtype=np.int32)[None, :]
    node_idx = np.zeros((n_iter, c_tree), dtype=np.int32)
    for _ in range(head.max_depth):
        f = head.feat_idx[i_arange, c_arange, node_idx]
        thr = head.thresholds[i_arange, c_arange, node_idx]
        node_idx = np.where(
            x_stacked_1d[f] <= thr,
            head.left[i_arange, c_arange, node_idx],
            head.right[i_arange, c_arange, node_idx],
        )
    raw = head.baseline + np.sum(head.values[i_arange, c_arange, node_idx], axis=0)
    if c_tree == 1:
        p1 = _sigmoid_scalar(float(raw[0]))
        pred_idx = 1 if p1 >= 0.5 else 0
    else:
        pred_idx = int(np.argmax(raw))
    return head.subfamilies[pred_idx]


def _load_mlp_block(data: Any, prefix: str, in_dim: int) -> _MLPBlock:
    import numpy as np

    w0 = np.asarray(data[f"{prefix}_w0"], dtype=np.float32)
    b0 = np.asarray(data[f"{prefix}_b0"], dtype=np.float32)
    w1 = np.asarray(data[f"{prefix}_w1"], dtype=np.float32)
    b1 = np.asarray(data[f"{prefix}_b1"], dtype=np.float32)
    w2 = np.asarray(data[f"{prefix}_w2"], dtype=np.float32)
    b2 = np.asarray(data[f"{prefix}_b2"], dtype=np.float32)
    if (
        w0.ndim != 2
        or w0.shape[1] != in_dim
        or b0.shape != (w0.shape[0],)
        or w1.ndim != 2
        or w1.shape[1] != w0.shape[0]
        or b1.shape != (w1.shape[0],)
        or w2.ndim != 2
        or w2.shape != (1, w1.shape[0])
        or b2.shape != (1,)
    ):
        raise ValueError(f"Invalid MLP weight shapes for {prefix}")
    for arr in (w0, b0, w1, b1, w2, b2):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite MLP weights in {prefix}")
    return _MLPBlock(w0=w0, b0=b0, w1=w1, b1=b1, w2=w2, b2=b2)


def _load_binary_hgb(data: Any, prefix: str, in_dim: int) -> _PackedBinaryHGB:
    import numpy as np

    baseline = float(np.asarray(data[f"{prefix}_baseline"], dtype=np.float64).squeeze())
    max_depth = int(np.asarray(data[f"{prefix}_max_depth"], dtype=np.int64).squeeze())
    feat_idx = np.asarray(data[f"{prefix}_feat_idx"], dtype=np.int16)
    thresholds = np.asarray(data[f"{prefix}_thresholds"], dtype=np.float64)
    left = np.asarray(data[f"{prefix}_left"], dtype=np.uint8)
    right = np.asarray(data[f"{prefix}_right"], dtype=np.uint8)
    values = np.asarray(data[f"{prefix}_values"], dtype=np.float64)

    if not math.isfinite(baseline) or max_depth < 1 or max_depth > 16:
        raise ValueError(f"Invalid HGB metadata for {prefix}")
    if (
        feat_idx.ndim != 2
        or thresholds.shape != feat_idx.shape
        or left.shape != feat_idx.shape
        or right.shape != feat_idx.shape
        or values.shape != feat_idx.shape
    ):
        raise ValueError(f"Mismatched HGB tree array shapes for {prefix}")
    n_nodes = feat_idx.shape[1]
    if (
        np.any(feat_idx < 0)
        or np.any(feat_idx >= in_dim)
        or np.any(left >= n_nodes)
        or np.any(right >= n_nodes)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(f"Out-of-bounds or non-finite HGB tree parameters for {prefix}")
    return _PackedBinaryHGB(
        baseline=baseline,
        max_depth=max_depth,
        feat_idx=feat_idx,
        thresholds=thresholds,
        left=left,
        right=right,
        values=values,
    )


def _load_stage1_unit(data: Any, prefix: str, *, is_sub: bool) -> _Stage1Unit:
    import numpy as np

    bw = np.asarray(data[f"{prefix}_blend_weights"], dtype=np.float64)
    if bw.shape != (3,) or not np.all(np.isfinite(bw)) or np.any(bw < 0.0):
        raise ValueError(f"Invalid blend_weights for {prefix}")
    base_offset = float(np.asarray(data[f"{prefix}_base_offset"], dtype=np.float64).squeeze())
    pool_alpha = float(np.asarray(data[f"{prefix}_pool_alpha"], dtype=np.float64).squeeze()) if is_sub else 1.0
    if not math.isfinite(base_offset) or not math.isfinite(pool_alpha) or pool_alpha < 0.0:
        raise ValueError(f"Invalid offset or pool_alpha for {prefix}")
    soup_count = int(np.asarray(data[f"{prefix}_soup_count"], dtype=np.int64).squeeze())
    if soup_count < 1 or soup_count > 16:
        raise ValueError(f"Invalid soup_count for {prefix}: {soup_count}")
    soup = tuple(_load_mlp_block(data, f"{prefix}_s{i}", FEATURE_DIM) for i in range(soup_count))
    et_mlp = _load_mlp_block(data, f"{prefix}_et_mlp", FEATURE_DIM)
    hgb = _load_binary_hgb(data, f"{prefix}_hgb", FEATURE_DIM)
    return _Stage1Unit(
        blend_weights=(float(bw[0]), float(bw[1]), float(bw[2])),
        base_offset=base_offset,
        pool_alpha=pool_alpha,
        mlp_soup=soup,
        et_mlp=et_mlp,
        hgb=hgb,
    )


@functools.lru_cache(maxsize=4)
def _load_model_from_path(resolved: Path) -> _SourceModel:
    import numpy as np

    with np.load(resolved, allow_pickle=False) as data:
        if "schema" not in data:
            raise ValueError(f"Missing schema key in {resolved}")
        schema = str(np.asarray(data["schema"]).item())
        if schema != MODEL_SCHEMA:
            raise ValueError(f"Unsupported source-classifier model schema: {schema!r}")

        feat_dim = int(np.asarray(data["feature_dim"], dtype=np.int64).squeeze())
        if feat_dim != FEATURE_DIM:
            raise ValueError(f"Expected feature_dim={FEATURE_DIM}, got {feat_dim}")

        cls_key = "provider_classes" if "provider_classes" in data else "all_ai_classes"
        all_cls = tuple(str(x) for x in np.asarray(data[cls_key]).tolist())
        if all_cls != ALL_PROVIDER_CLASSES:
            raise ValueError(f"Unexpected provider_classes: {all_cls}")

        damping_gamma = float(np.asarray(data["damping_gamma"], dtype=np.float64).squeeze())
        if not math.isfinite(damping_gamma) or not (0.0 <= damping_gamma <= 1.0):
            raise ValueError(f"Invalid damping_gamma: {damping_gamma}")

        mean = np.asarray(data["mean"], dtype=np.float64)
        scale = np.asarray(data["scale"], dtype=np.float64)
        if (
            mean.shape != (FEATURE_DIM,)
            or scale.shape != (FEATURE_DIM,)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0.0)
        ):
            raise ValueError("Invalid standardization arrays mean/scale")

        thr_key = "router_thresholds" if "router_thresholds" in data else "router_all11_thresholds"
        thr_arr = np.asarray(data[thr_key], dtype=np.float64)
        if thr_arr.shape != (len(ALL_PROVIDER_CLASSES),) or not np.all(np.isfinite(thr_arr)):
            raise ValueError("Invalid router threshold array")

        router_thresholds = {c: float(t) for c, t in zip(ALL_PROVIDER_CLASSES, thr_arr.tolist(), strict=True)}

        specialists: dict[str, _ProviderSpecialist] = {}
        for cls_name in ALL_PROVIDER_CLASSES:
            g_unit = _load_stage1_unit(data, f"s1_{cls_name}_global", is_sub=False)
            pool_rec = float(np.asarray(data[f"s1_{cls_name}_pool_recenter"], dtype=np.float64).squeeze())
            if not math.isfinite(pool_rec):
                raise ValueError(f"Non-finite pool_recenter for {cls_name}")
            sub_names = [str(x) for x in np.asarray(data[f"s1_{cls_name}_sub_names"]).tolist()]
            sub_units: dict[str, _Stage1Unit] = {}
            for s_idx, s_name in enumerate(sub_names):
                sub_units[s_name] = _load_stage1_unit(data, f"s1_{cls_name}_sub{s_idx}", is_sub=True)
            specialists[cls_name] = _ProviderSpecialist(
                global_unit=g_unit,
                sub_units=sub_units,
                pool_recenter=pool_rec,
            )

        desc_cls = [str(x) for x in np.asarray(data["unit_descriptor_classes"]).tolist()]
        desc_names = [str(x) for x in np.asarray(data["unit_descriptor_names"]).tolist()]
        if len(desc_cls) != len(desc_names):
            raise ValueError("Mismatched unit_descriptor arrays")
        unit_descriptors = tuple(zip(desc_cls, desc_names, strict=True))
        for d_c, d_n in unit_descriptors:
            if d_c not in specialists:
                raise ValueError(f"Unknown descriptor class {d_c!r}")
            if d_n != "__global__" and d_n not in specialists[d_c].sub_units:
                raise ValueError(f"Unknown descriptor sub-unit {d_n!r} for {d_c!r}")

        stacked_dim = FEATURE_DIM + len(unit_descriptors)
        v_classes = [str(x) for x in np.asarray(data["stage2_version_classes"]).tolist()]
        version_heads: dict[str, _Stage2VersionHead] = {}
        for p_cls in v_classes:
            subs = tuple(str(x) for x in np.asarray(data[f"s2_{p_cls}_subfamilies"]).tolist())
            if len(subs) < 2:
                raise ValueError(f"Stage-2 head for {p_cls} must have >= 2 subfamilies")
            prefix = f"s2_{p_cls}_hgb"
            baseline = np.asarray(data[f"{prefix}_baseline"], dtype=np.float64).reshape(-1)
            max_depth = int(np.asarray(data[f"{prefix}_max_depth"], dtype=np.int64).squeeze())
            feat_idx = np.asarray(data[f"{prefix}_feat_idx"], dtype=np.int16)
            thresholds = np.asarray(data[f"{prefix}_thresholds"], dtype=np.float64)
            left = np.asarray(data[f"{prefix}_left"], dtype=np.uint8)
            right = np.asarray(data[f"{prefix}_right"], dtype=np.uint8)
            values = np.asarray(data[f"{prefix}_values"], dtype=np.float64)
            if (
                feat_idx.ndim != 3
                or thresholds.shape != feat_idx.shape
                or left.shape != feat_idx.shape
                or right.shape != feat_idx.shape
                or values.shape != feat_idx.shape
                or baseline.shape != (feat_idx.shape[1],)
                or np.any(feat_idx < 0)
                or np.any(feat_idx >= stacked_dim)
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(f"Invalid Stage-2 HGB arrays for {p_cls}")
            version_heads[p_cls] = _Stage2VersionHead(
                subfamilies=subs,
                baseline=baseline,
                max_depth=max_depth,
                feat_idx=feat_idx,
                thresholds=thresholds,
                left=left,
                right=right,
                values=values,
            )

        router_meta_tau: FloatArray | None = None
        router_meta_w: FloatArray | None = None
        if "router_meta_tau" in data or "router_meta_w" in data:
            if "router_meta_tau" not in data or "router_meta_w" not in data:
                raise ValueError("Both router_meta_tau and router_meta_w must be present together")
            n_units = len(unit_descriptors)
            n_cls = len(ALL_PROVIDER_CLASSES)
            tau_arr = np.asarray(data["router_meta_tau"], dtype=np.float64)
            w_arr = np.asarray(data["router_meta_w"], dtype=np.float64)
            if (
                tau_arr.shape != (n_units,)
                or w_arr.shape != (n_units + n_cls, n_cls)
                or not np.all(np.isfinite(tau_arr))
                or not np.all(np.isfinite(w_arr))
            ):
                raise ValueError("Invalid router_meta_tau or router_meta_w shape or values")
            router_meta_tau = tau_arr
            router_meta_w = w_arr

    return _SourceModel(
        mean=mean,
        scale=scale,
        damping_gamma=damping_gamma,
        router_thresholds=router_thresholds,
        specialists=specialists,
        unit_descriptors=unit_descriptors,
        version_heads=version_heads,
        router_meta_tau=router_meta_tau,
        router_meta_w=router_meta_w,
    )


def _load_model(weights_path: Path | None = None) -> _SourceModel:
    return _load_model_from_path(_resolve_weights_path(weights_path))


def _unavailable_result() -> SourceClassification:
    return SourceClassification(
        label="unknown",
        version_subfamily=None,
        fine_subfamily=None,
        reason="feature_unavailable",
        confidence=0.0,
        pooled_scores={},
        margins={},
        thresholds={},
    )


def classify_source_features(
    features_956d: FloatArray,
    *,
    weights_path: Path | None = None,
) -> SourceClassification:
    """Classify a precomputed 956-d feature vector into one of 10 providers or unknown."""
    import numpy as np

    x_1d = np.asarray(features_956d, dtype=np.float64)
    if x_1d.shape != (FEATURE_DIM,) or not np.all(np.isfinite(x_1d)):
        return _unavailable_result()

    model = _load_model(weights_path)
    x_std_1d = np.clip((x_1d - model.mean) / model.scale, -12.0, 12.0).astype(np.float32)

    thresholds = model.router_thresholds

    raw_unit_map: dict[tuple[str, str], float] = {}
    pooled_z: dict[str, float] = {}
    best_fine_sub_by_cls: dict[str, str | None] = {}

    for cls_name in ALL_PROVIDER_CLASSES:
        spec = model.specialists[cls_name]
        g_raw = _score_stage1_unit(spec.global_unit, x_1d, x_std_1d)
        raw_unit_map[(cls_name, "__global__")] = g_raw
        max_z = g_raw - spec.global_unit.base_offset
        best_sub: str | None = None
        best_sub_centered = -1e9

        for sub_name, su in spec.sub_units.items():
            s_raw = _score_stage1_unit(su, x_1d, x_std_1d)
            raw_unit_map[(cls_name, sub_name)] = s_raw
            s_centered = s_raw - su.base_offset
            if s_centered > best_sub_centered:
                best_sub_centered = s_centered
                best_sub = sub_name
            if su.pool_alpha > 0.0 and s_centered > 0.0:
                boost = su.pool_alpha * s_centered
                if boost > max_z:
                    max_z = boost

        pooled_z[cls_name] = max_z - spec.pool_recenter
        best_fine_sub_by_cls[cls_name] = best_sub

    u_vec: FloatArray | None = None
    if model.router_meta_tau is not None and model.router_meta_w is not None:
        u_vec = np.asarray(
            [raw_unit_map[desc] for desc in model.unit_descriptors],
            dtype=np.float64,
        )
        h_u = np.maximum(0.0, u_vec - model.router_meta_tau)
        h_z = np.asarray(
            [max(0.0, pooled_z[c]) for c in ALL_PROVIDER_CLASSES],
            dtype=np.float64,
        )
        meta_delta = np.concatenate([h_u, h_z]) @ model.router_meta_w
        for idx_c, cls_name in enumerate(ALL_PROVIDER_CLASSES):
            pooled_z[cls_name] += float(meta_delta[idx_c])

    margins: dict[str, float] = {}
    excess: dict[str, float] = {}
    for cls_name in ALL_PROVIDER_CLASSES:
        other_pos = [max(0.0, pooled_z[other]) for other in ALL_PROVIDER_CLASSES if other != cls_name]
        max_other = max(other_pos) if other_pos else 0.0
        m_val = pooled_z[cls_name] - model.damping_gamma * max_other
        margins[cls_name] = m_val
        excess[cls_name] = m_val - thresholds[cls_name]

    best_cls = max(ALL_PROVIDER_CLASSES, key=lambda c: excess[c])
    confidence = _sigmoid_scalar(excess[best_cls])
    if excess[best_cls] >= 0.0:
        label: SourceLabel = best_cls
        reason: SourceReason = "classified"
        fine_subfamily = best_fine_sub_by_cls.get(best_cls)
        version_subfamily = fine_subfamily
        if best_cls in model.version_heads:
            if u_vec is None:
                u_vec = np.asarray(
                    [raw_unit_map[desc] for desc in model.unit_descriptors],
                    dtype=np.float64,
                )
            x_stacked = np.concatenate([x_1d, u_vec])
            version_subfamily = _eval_stage2_version_head(model.version_heads[best_cls], x_stacked)
    else:
        label = "unknown"
        reason = "abstained"
        fine_subfamily = None
        version_subfamily = None

    return SourceClassification(
        label=label,
        version_subfamily=version_subfamily,
        fine_subfamily=fine_subfamily,
        reason=reason,
        confidence=confidence,
        pooled_scores=pooled_z,
        margins=margins,
        thresholds=dict(thresholds),
    )


def classify_source(
    source: str | os.PathLike[str] | Image.Image,
    *,
    weights_path: Path | None = None,
) -> SourceClassification:
    """Classify an image into one of 10 AI provider pipelines or ``unknown``.

    Parameters
    ----------
    source
        Path to an image file or an already-opened PIL Image.
    weights_path
        Optional local ``.npz`` override. Defaults to ``RAIW_SOURCE_CLASSIFY_WEIGHTS``
        or the pinned Hugging Face Hub artifact verified by SHA-256.
    """
    if not module_available("numpy", "cv2"):
        _require_dependencies()
    from PIL import Image as PilImage

    from remove_ai_watermarks import image_io

    image_io._register_heif()  # pyright: ignore[reportPrivateUsage]
    if isinstance(source, (str, os.PathLike)):
        with PilImage.open(source) as opened:
            features = extract_956d_features(opened)
    else:
        features = extract_956d_features(source)

    if features is None:
        return _unavailable_result()

    return classify_source_features(
        features,
        weights_path=weights_path,
    )
