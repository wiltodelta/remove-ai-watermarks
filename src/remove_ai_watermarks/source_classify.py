"""Abstaining 10-provider pixel source-pipeline classifier with Stage-2 version attribution.

This classifier identifies the likely AI generation and export pixel pipeline across
10 provider families (``openai``, ``google``, ``bytedance``, ``flux``,
``stable_diffusion``, ``meta``, ``midjourney``, ``ideogram``, ``firefly``, ``xai``),
returning ``unknown`` when the image does not clear its calibrated acceptance
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
WEIGHTS_REVISION = "8260c14098d768e2d9f7304a4f5905bf00c7d66f"
MODEL_FILE = "multi-provider-source-v6.npz"
MODEL_SHA256 = "87868eea08b8b811999f675186b82b067c4de655981cb7f25114aac00b1dd284"
MODEL_SCHEMA_V3 = "multi-provider-source-v3"
MODEL_SCHEMA_V4 = "multi-provider-source-v4"
MODEL_SCHEMA_V5 = "multi-provider-source-v5"
MODEL_SCHEMA = MODEL_SCHEMA_V5
SOURCE_CLASSIFY_EXTRA = "'remove-ai-watermarks[source-classify]'"

MIN_SIDE = 256
FORENSIC_DIM = 124
SPECTRAL_GRID = 16
SPECTRAL_DIM = 3 * SPECTRAL_GRID * SPECTRAL_GRID
BASE_FEATURE_DIM = FORENSIC_DIM + SPECTRAL_DIM
EXTRA_FEATURE_DIM = 64
FEATURE_DIM = BASE_FEATURE_DIM + EXTRA_FEATURE_DIM
FEATURE_DIM_V4 = 512
REGIME_START_V4 = 480
REGIME_DIM_V4 = 32

CLAIM = (
    "Identifies the likely AI generation/export pixel pipeline across 10 provider "
    "families with decoupled regime-aware acceptance, threshold-invariant attribution, "
    "and stacked Stage-2 version attribution. This is not provenance "
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
class _Stage2LinearBlend:
    """Optional standardized L2 linear head blended with Stage-2 HGB logits."""

    mean: FloatArray
    scale: FloatArray
    w: FloatArray
    b: FloatArray
    hgb_weight: float


@dataclass(frozen=True)
class _Stage2VersionHead:
    """Stacked Stage-2 version classifier over input features + Stage-1 unit logits."""

    subfamilies: tuple[str, ...]
    baseline: FloatArray
    max_depth: int
    feat_idx: FloatArray
    thresholds: FloatArray
    left: FloatArray
    right: FloatArray
    values: FloatArray
    linear: _Stage2LinearBlend | None = None


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
    schema: str = MODEL_SCHEMA_V3


@dataclass(frozen=True)
class _ResidualMLPTrunk:
    """Multi-task Regime-FiLM Residual-MLP trunk (provider/unknown[/accept] + M_sub subfamily heads)."""

    w_in: FloatArray
    b_in: FloatArray
    ln0_g: FloatArray
    ln0_b: FloatArray
    w_film: FloatArray
    b_film: FloatArray
    w_r1a: FloatArray
    b_r1a: FloatArray
    ln1_g: FloatArray
    ln1_b: FloatArray
    w_r1b: FloatArray
    b_r1b: FloatArray
    w_mid: FloatArray
    b_mid: FloatArray
    ln2_g: FloatArray
    ln2_b: FloatArray
    w_r2a: FloatArray
    b_r2a: FloatArray
    ln3_g: FloatArray
    ln3_b: FloatArray
    w_r2b: FloatArray
    b_r2b: FloatArray
    w_prov: FloatArray
    b_prov: FloatArray
    w_sub: FloatArray
    b_sub: FloatArray


@dataclass(frozen=True)
class _TwoTowerBackbone:
    """Shared 512-d two-tower trunk + packed binary HGB backbone for v4 and v5 models."""

    mean: FloatArray
    scale: FloatArray
    trunks: tuple[_ResidualMLPTrunk, ...]
    n_subs: int
    sub_by_provider: dict[str, tuple[tuple[int, str], ...]]
    hgb_units: tuple[_PackedBinaryHGB, ...]


@dataclass(frozen=True)
class _RouterGates2D:
    """Shared positive/negative hinge thresholds and 2D multiplicative interaction gates."""

    router_tau_pos: FloatArray
    router_tau_neg: FloatArray
    gate_idx_a: FloatArray
    gate_idx_b: FloatArray
    gate_sign_a: FloatArray
    gate_sign_b: FloatArray
    gate_tau_a: FloatArray
    gate_tau_b: FloatArray


@dataclass(frozen=True)
class _SourceModelV4:
    """Validated pickle-free multi-provider v4 Two-Tower + Regime-Conditioned Router model."""

    backbone: _TwoTowerBackbone
    gates: _RouterGates2D
    router_bias: FloatArray
    router_w: FloatArray
    null_mu_bias: FloatArray
    null_mu_w: FloatArray
    null_sig_bias: FloatArray
    null_sig_w: FloatArray
    damping_gammas: FloatArray
    router_thresholds: dict[str, float]
    conf_scale: float
    version_heads: dict[str, _Stage2VersionHead]


@dataclass(frozen=True)
class _SourceModelV5:
    """Validated pickle-free v5 model with decoupled regime-aware acceptance and attribution."""

    backbone: _TwoTowerBackbone
    gates: _RouterGates2D
    u_med: FloatArray
    u_scale: FloatArray
    det_bias: FloatArray
    det_w: FloatArray
    null_h_w: FloatArray
    null_h_b: FloatArray
    null_lin_mu_w: FloatArray
    null_mu_bias: FloatArray
    null_mu_w: FloatArray
    null_lin_sig_w: FloatArray
    null_sig_bias: FloatArray
    null_sig_w: FloatArray
    det_thresholds: FloatArray
    attr_w1: FloatArray
    attr_b1: FloatArray
    attr_ln_g: FloatArray
    attr_ln_b: FloatArray
    attr_w2: FloatArray
    attr_b2: FloatArray
    attr_skip_w: FloatArray
    attr_temp: float
    conf_scale: float
    version_heads: dict[str, _Stage2VersionHead]


_LoadedSourceModel: TypeAlias = _SourceModel | _SourceModelV4 | _SourceModelV5


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


def extract_512d_v4_features(image: Image.Image) -> FloatArray | None:
    """Compute the 512-d physics-grounded v4 feature vector, or None if < 256x256."""
    from remove_ai_watermarks._internal.source_features_v4 import extract_v4_features

    return extract_v4_features(image)


def _gelu_np(x: FloatArray) -> FloatArray:
    import numpy as np

    return (0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))))).astype(np.float32)


def _silu_np(x: FloatArray) -> FloatArray:
    import numpy as np

    clipped = np.clip(x, -30.0, 30.0)
    return (x / (1.0 + np.exp(-clipped))).astype(x.dtype, copy=False)


def _layernorm_1d(x: FloatArray, gamma: FloatArray, beta: FloatArray) -> FloatArray:
    mu = float(x.mean())
    diff = x - mu
    var = float((diff * diff).mean())
    return (diff * (1.0 / math.sqrt(var + 1e-5)) * gamma + beta).astype(x.dtype, copy=False)


def _eval_residual_trunk(trunk: _ResidualMLPTrunk, x_std_1d: FloatArray) -> tuple[FloatArray, FloatArray]:
    import numpy as np

    r_std = x_std_1d[REGIME_START_V4 : REGIME_START_V4 + REGIME_DIM_V4]
    h0 = _silu_np(_layernorm_1d(x_std_1d @ trunk.w_in.T + trunk.b_in, trunk.ln0_g, trunk.ln0_b))
    film = r_std @ trunk.w_film.T + trunk.b_film
    h1_dim = h0.shape[0]
    gamma_f = film[:h1_dim]
    beta_f = film[h1_dim:]
    h0 = (h0 * (1.0 + 0.25 * np.tanh(gamma_f)) + 0.25 * beta_f).astype(np.float32)

    u1 = _silu_np(_layernorm_1d(h0 @ trunk.w_r1a.T + trunk.b_r1a, trunk.ln1_g, trunk.ln1_b))
    h1 = (h0 + (u1 @ trunk.w_r1b.T + trunk.b_r1b)).astype(np.float32)

    h2 = _silu_np(_layernorm_1d(h1 @ trunk.w_mid.T + trunk.b_mid, trunk.ln2_g, trunk.ln2_b))
    u2 = _silu_np(_layernorm_1d(h2 @ trunk.w_r2a.T + trunk.b_r2a, trunk.ln3_g, trunk.ln3_b))
    h3 = (h2 + (u2 @ trunk.w_r2b.T + trunk.b_r2b)).astype(np.float32)

    prov = np.clip((h3 @ trunk.w_prov.T + trunk.b_prov).astype(np.float64), -18.0, 18.0)
    sub = np.clip((h3 @ trunk.w_sub.T + trunk.b_sub).astype(np.float64), -18.0, 18.0)
    return prov, sub


def _eval_packed_binary_hgb_raw(hgb: _PackedBinaryHGB, x_1d: FloatArray) -> float:
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
    return max(-18.0, min(18.0, raw))


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
    lin = head.linear
    if lin is not None and lin.hgb_weight <= 0.0:
        x_lin = np.clip((x_stacked_1d - lin.mean) / lin.scale, -10.0, 10.0)
        raw = x_lin @ lin.w + lin.b
    else:
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
        if lin is not None and lin.hgb_weight < 1.0:
            x_lin = np.clip((x_stacked_1d - lin.mean) / lin.scale, -10.0, 10.0)
            lin_raw = x_lin @ lin.w + lin.b
            raw = lin.hgb_weight * raw + (1.0 - lin.hgb_weight) * lin_raw

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


def _load_1d_standardizer(
    data: Any,
    center_key: str,
    scale_key: str,
    dim: int,
    *,
    err_msg: str,
) -> tuple[FloatArray, FloatArray]:
    import numpy as np

    center = np.asarray(data[center_key], dtype=np.float64)
    scale = np.asarray(data[scale_key], dtype=np.float64)
    if (
        center.shape != (dim,)
        or scale.shape != (dim,)
        or np.any(scale <= 0.0)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(scale))
    ):
        raise ValueError(err_msg)
    return center, scale


def _load_stage2_heads(data: Any, stacked_dim: int) -> dict[str, _Stage2VersionHead]:
    import numpy as np

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
        lin_prefix = f"s2_{p_cls}_lin"
        lin_keys = tuple(f"{lin_prefix}_{s}" for s in ("mean", "scale", "w", "b", "hgb_weight"))
        present_lin = [k in data for k in lin_keys]
        linear_blend: _Stage2LinearBlend | None = None
        if any(present_lin):
            if not all(present_lin):
                raise ValueError(f"Incomplete Stage-2 linear blend keys for {p_cls}")
            c_tree = feat_idx.shape[1]
            lin_err = f"Invalid Stage-2 linear blend arrays for {p_cls}"
            lin_mean, lin_scale = _load_1d_standardizer(
                data, f"{lin_prefix}_mean", f"{lin_prefix}_scale", stacked_dim, err_msg=lin_err
            )
            lin_w = np.asarray(data[f"{lin_prefix}_w"], dtype=np.float64)
            lin_b = np.asarray(data[f"{lin_prefix}_b"], dtype=np.float64).reshape(-1)
            hgb_weight = float(np.asarray(data[f"{lin_prefix}_hgb_weight"], dtype=np.float64).squeeze())
            if (
                lin_w.shape != (stacked_dim, c_tree)
                or lin_b.shape != (c_tree,)
                or not math.isfinite(hgb_weight)
                or not (0.0 <= hgb_weight <= 1.0)
                or not np.all(np.isfinite(lin_w))
                or not np.all(np.isfinite(lin_b))
            ):
                raise ValueError(lin_err)
            linear_blend = _Stage2LinearBlend(
                mean=lin_mean,
                scale=lin_scale,
                w=lin_w,
                b=lin_b,
                hgb_weight=hgb_weight,
            )
        version_heads[p_cls] = _Stage2VersionHead(
            subfamilies=subs,
            baseline=baseline,
            max_depth=max_depth,
            feat_idx=feat_idx,
            thresholds=thresholds,
            left=left,
            right=right,
            values=values,
            linear=linear_blend,
        )
    return version_heads


def _load_residual_trunk(
    data: Any,
    prefix: str,
    in_dim: int,
    n_subs: int,
    n_prov_out: int,
) -> _ResidualMLPTrunk:
    import numpy as np

    keys = (
        "w_in",
        "b_in",
        "ln0_g",
        "ln0_b",
        "w_film",
        "b_film",
        "w_r1a",
        "b_r1a",
        "ln1_g",
        "ln1_b",
        "w_r1b",
        "b_r1b",
        "w_mid",
        "b_mid",
        "ln2_g",
        "ln2_b",
        "w_r2a",
        "b_r2a",
        "ln3_g",
        "ln3_b",
        "w_r2b",
        "b_r2b",
        "w_prov",
        "b_prov",
        "w_sub",
        "b_sub",
    )
    arrays = {k: np.asarray(data[f"{prefix}_{k}"], dtype=np.float32) for k in keys}
    for k, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite trunk parameter {prefix}_{k}")

    h1 = arrays["w_in"].shape[0] if arrays["w_in"].ndim == 2 else -1
    h2 = arrays["w_mid"].shape[0] if arrays["w_mid"].ndim == 2 else -1
    if (
        h1 < 4
        or h2 < 4
        or arrays["w_in"].shape != (h1, in_dim)
        or arrays["b_in"].shape != (h1,)
        or arrays["ln0_g"].shape != (h1,)
        or arrays["ln0_b"].shape != (h1,)
        or arrays["w_film"].shape != (2 * h1, REGIME_DIM_V4)
        or arrays["b_film"].shape != (2 * h1,)
        or arrays["w_r1a"].shape != (h1, h1)
        or arrays["b_r1a"].shape != (h1,)
        or arrays["ln1_g"].shape != (h1,)
        or arrays["ln1_b"].shape != (h1,)
        or arrays["w_r1b"].shape != (h1, h1)
        or arrays["b_r1b"].shape != (h1,)
        or arrays["w_mid"].shape != (h2, h1)
        or arrays["b_mid"].shape != (h2,)
        or arrays["ln2_g"].shape != (h2,)
        or arrays["ln2_b"].shape != (h2,)
        or arrays["w_r2a"].shape != (h2, h2)
        or arrays["b_r2a"].shape != (h2,)
        or arrays["ln3_g"].shape != (h2,)
        or arrays["ln3_b"].shape != (h2,)
        or arrays["w_r2b"].shape != (h2, h2)
        or arrays["b_r2b"].shape != (h2,)
        or arrays["w_prov"].shape != (n_prov_out, h2)
        or arrays["b_prov"].shape != (n_prov_out,)
        or arrays["w_sub"].shape != (n_subs, h2)
        or arrays["b_sub"].shape != (n_subs,)
    ):
        raise ValueError(f"Invalid v4/v5 trunk shapes for {prefix}")

    return _ResidualMLPTrunk(**arrays)


def _load_two_tower_backbone(data: Any, *, n_prov_out: int, tag: str) -> _TwoTowerBackbone:
    import numpy as np

    feat_dim = int(np.asarray(data["feature_dim"], dtype=np.int64).squeeze())
    if feat_dim != FEATURE_DIM_V4:
        raise ValueError(f"Expected feature_dim={FEATURE_DIM_V4}, got {feat_dim}")

    all_cls = tuple(str(x) for x in np.asarray(data["provider_classes"]).tolist())
    if all_cls != ALL_PROVIDER_CLASSES:
        raise ValueError(f"Unexpected provider_classes: {all_cls}")

    mean, scale = _load_1d_standardizer(
        data, "mean", "scale", FEATURE_DIM_V4, err_msg=f"Invalid {tag} standardization arrays mean/scale"
    )

    sub_names = tuple(str(x) for x in np.asarray(data["sub_names"]).tolist())
    sub_providers = tuple(str(x) for x in np.asarray(data["sub_providers"]).tolist())
    n_subs = len(sub_names)
    if n_subs < len(ALL_PROVIDER_CLASSES) or len(sub_providers) != n_subs:
        raise ValueError(f"Invalid sub_names / sub_providers in {tag} bundle")
    by_prov: dict[str, list[tuple[int, str]]] = {c: [] for c in ALL_PROVIDER_CLASSES}
    for s_idx, (s_name, sp) in enumerate(zip(sub_names, sub_providers, strict=True)):
        if sp not in by_prov:
            raise ValueError(f"Unknown subfamily provider {sp!r} in {tag} bundle")
        by_prov[sp].append((s_idx, s_name))
    sub_by_provider = {c: tuple(items) for c, items in by_prov.items()}

    trunk_count = int(np.asarray(data["trunk_count"], dtype=np.int64).squeeze())
    if trunk_count < 1 or trunk_count > 12:
        raise ValueError(f"Invalid trunk_count: {trunk_count}")
    trunks = tuple(
        _load_residual_trunk(data, f"t1_s{i}", FEATURE_DIM_V4, n_subs, n_prov_out) for i in range(trunk_count)
    )

    n_hgb = int(np.asarray(data["hgb_names"]).shape[0])
    hgb_units = tuple(_load_binary_hgb(data, f"t2_h{i}", FEATURE_DIM_V4) for i in range(n_hgb))

    return _TwoTowerBackbone(
        mean=mean,
        scale=scale,
        trunks=trunks,
        n_subs=n_subs,
        sub_by_provider=sub_by_provider,
        hgb_units=hgb_units,
    )


def _load_router_gates(data: Any, d_u: int, *, tag: str) -> _RouterGates2D:
    import numpy as np

    router_tau_pos = np.asarray(data["router_tau_pos"], dtype=np.float64)
    router_tau_neg = np.asarray(data["router_tau_neg"], dtype=np.float64)
    gate_idx_a = np.asarray(data["gate_idx_a"], dtype=np.int64)
    gate_idx_b = np.asarray(data["gate_idx_b"], dtype=np.int64)
    gate_sign_a = np.asarray(data["gate_sign_a"], dtype=np.float64)
    gate_sign_b = np.asarray(data["gate_sign_b"], dtype=np.float64)
    gate_tau_a = np.asarray(data["gate_tau_a"], dtype=np.float64)
    gate_tau_b = np.asarray(data["gate_tau_b"], dtype=np.float64)
    n_gates = gate_idx_a.shape[0] if gate_idx_a.ndim == 1 else -1
    if (
        n_gates < 0
        or router_tau_pos.shape != (d_u,)
        or router_tau_neg.shape != (d_u,)
        or gate_idx_b.shape != (n_gates,)
        or gate_sign_a.shape != (n_gates,)
        or gate_sign_b.shape != (n_gates,)
        or gate_tau_a.shape != (n_gates,)
        or gate_tau_b.shape != (n_gates,)
        or (n_gates > 0 and (np.any(gate_idx_a < 0) or np.any(gate_idx_a >= d_u)))
        or (n_gates > 0 and (np.any(gate_idx_b < 0) or np.any(gate_idx_b >= d_u)))
        or not all(
            np.all(np.isfinite(arr))
            for arr in (router_tau_pos, router_tau_neg, gate_sign_a, gate_sign_b, gate_tau_a, gate_tau_b)
        )
    ):
        raise ValueError(f"Invalid {tag} router or 2D gate arrays")
    return _RouterGates2D(
        router_tau_pos=router_tau_pos,
        router_tau_neg=router_tau_neg,
        gate_idx_a=gate_idx_a,
        gate_idx_b=gate_idx_b,
        gate_sign_a=gate_sign_a,
        gate_sign_b=gate_sign_b,
        gate_tau_a=gate_tau_a,
        gate_tau_b=gate_tau_b,
    )


def _load_model_v4(data: Any) -> _SourceModelV4:
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    backbone = _load_two_tower_backbone(data, n_prov_out=n_cls + 1, tag="v4")
    d_u = n_cls + 1 + backbone.n_subs + len(backbone.hgb_units) + REGIME_DIM_V4
    gates = _load_router_gates(data, d_u, tag="v4")
    d_phi = 3 * d_u + gates.gate_idx_a.shape[0]

    router_bias = np.asarray(data["router_bias"], dtype=np.float64)
    router_w = np.asarray(data["router_w"], dtype=np.float64)
    if (
        router_bias.shape != (n_cls,)
        or router_w.shape != (d_phi, n_cls)
        or not np.all(np.isfinite(router_bias))
        or not np.all(np.isfinite(router_w))
    ):
        raise ValueError("Invalid v4 router or 2D gate arrays")

    null_mu_bias = np.asarray(data["null_mu_bias"], dtype=np.float64)
    null_mu_w = np.asarray(data["null_mu_w"], dtype=np.float64)
    null_sig_bias = np.asarray(data["null_sig_bias"], dtype=np.float64)
    null_sig_w = np.asarray(data["null_sig_w"], dtype=np.float64)
    if (
        null_mu_bias.shape != (n_cls,)
        or null_mu_w.shape != (REGIME_DIM_V4, n_cls)
        or null_sig_bias.shape != (n_cls,)
        or null_sig_w.shape != (REGIME_DIM_V4, n_cls)
        or not all(np.all(np.isfinite(arr)) for arr in (null_mu_bias, null_mu_w, null_sig_bias, null_sig_w))
    ):
        raise ValueError("Invalid v4 regime null standardization arrays")

    damping_gammas = np.asarray(data["damping_gammas"], dtype=np.float64)
    if (
        damping_gammas.shape != (n_cls,)
        or not np.all(np.isfinite(damping_gammas))
        or np.any(damping_gammas < 0.0)
        or np.any(damping_gammas > 1.0)
    ):
        raise ValueError("Invalid v4 damping_gammas")

    thr_arr = np.asarray(data["router_thresholds"], dtype=np.float64)
    if thr_arr.shape != (n_cls,) or not np.all(np.isfinite(thr_arr)):
        raise ValueError("Invalid v4 router_thresholds")
    router_thresholds = {c: float(t) for c, t in zip(ALL_PROVIDER_CLASSES, thr_arr.tolist(), strict=True)}

    conf_scale = float(np.asarray(data["conf_scale"], dtype=np.float64).squeeze())
    if not math.isfinite(conf_scale) or conf_scale <= 0.0:
        raise ValueError(f"Invalid v4 conf_scale: {conf_scale}")

    version_heads = _load_stage2_heads(data, FEATURE_DIM_V4 + d_u)
    return _SourceModelV4(
        backbone=backbone,
        gates=gates,
        router_bias=router_bias,
        router_w=router_w,
        null_mu_bias=null_mu_bias,
        null_mu_w=null_mu_w,
        null_sig_bias=null_sig_bias,
        null_sig_w=null_sig_w,
        damping_gammas=damping_gammas,
        router_thresholds=router_thresholds,
        conf_scale=conf_scale,
        version_heads=version_heads,
    )


def _load_model_v5(data: Any) -> _SourceModelV5:
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    n_det = n_cls + 1
    backbone = _load_two_tower_backbone(data, n_prov_out=n_cls + 2, tag="v5")
    d_u = n_cls + 2 + backbone.n_subs + len(backbone.hgb_units) + REGIME_DIM_V4
    gates = _load_router_gates(data, d_u, tag="v5")
    d_phi = 3 * d_u + gates.gate_idx_a.shape[0]

    u_med, u_scale = _load_1d_standardizer(
        data, "u_med", "u_scale", d_u, err_msg="Invalid v5 channel normalization arrays"
    )

    det_bias = np.asarray(data["det_bias"], dtype=np.float64)
    det_w = np.asarray(data["det_w"], dtype=np.float64)
    null_h_w = np.asarray(data["null_h_w"], dtype=np.float64)
    null_h_b = np.asarray(data["null_h_b"], dtype=np.float64)
    null_lin_mu_w = np.asarray(data["null_lin_mu_w"], dtype=np.float64)
    null_mu_bias = np.asarray(data["null_mu_bias"], dtype=np.float64)
    null_mu_w = np.asarray(data["null_mu_w"], dtype=np.float64)
    null_lin_sig_w = np.asarray(data["null_lin_sig_w"], dtype=np.float64)
    null_sig_bias = np.asarray(data["null_sig_bias"], dtype=np.float64)
    null_sig_w = np.asarray(data["null_sig_w"], dtype=np.float64)
    det_thresholds = np.asarray(data["det_thresholds"], dtype=np.float64)

    h_null_dim = null_h_b.shape[0] if null_h_b.ndim == 1 else -1
    if (
        h_null_dim < 2
        or det_bias.shape != (n_det,)
        or det_w.shape != (d_phi, n_det)
        or null_h_w.shape != (REGIME_DIM_V4, h_null_dim)
        or null_lin_mu_w.shape != (REGIME_DIM_V4, n_det)
        or null_mu_bias.shape != (n_det,)
        or null_mu_w.shape != (h_null_dim, n_det)
        or null_lin_sig_w.shape != (REGIME_DIM_V4, n_det)
        or null_sig_bias.shape != (n_det,)
        or null_sig_w.shape != (h_null_dim, n_det)
        or det_thresholds.shape != (n_det,)
        or not all(
            np.all(np.isfinite(arr))
            for arr in (
                det_bias,
                det_w,
                null_h_w,
                null_h_b,
                null_lin_mu_w,
                null_mu_bias,
                null_mu_w,
                null_lin_sig_w,
                null_sig_bias,
                null_sig_w,
                det_thresholds,
            )
        )
    ):
        raise ValueError("Invalid v5 acceptance or regime-null arrays")

    attr_w1 = np.asarray(data["attr_w1"], dtype=np.float64)
    attr_b1 = np.asarray(data["attr_b1"], dtype=np.float64)
    attr_ln_g = np.asarray(data["attr_ln_g"], dtype=np.float64)
    attr_ln_b = np.asarray(data["attr_ln_b"], dtype=np.float64)
    attr_w2 = np.asarray(data["attr_w2"], dtype=np.float64)
    attr_b2 = np.asarray(data["attr_b2"], dtype=np.float64)
    attr_skip_w = np.asarray(data["attr_skip_w"], dtype=np.float64)
    attr_temp = float(np.asarray(data["attr_temp"], dtype=np.float64).squeeze())
    conf_scale = float(np.asarray(data["conf_scale"], dtype=np.float64).squeeze())

    h_attr_dim = attr_b1.shape[0] if attr_b1.ndim == 1 else -1
    if (
        h_attr_dim < 4
        or attr_w1.shape != (d_phi, h_attr_dim)
        or attr_ln_g.shape != (h_attr_dim,)
        or attr_ln_b.shape != (h_attr_dim,)
        or attr_w2.shape != (h_attr_dim, n_cls)
        or attr_b2.shape != (n_cls,)
        or attr_skip_w.shape != (d_phi, n_cls)
        or not math.isfinite(attr_temp)
        or attr_temp <= 0.0
        or not math.isfinite(conf_scale)
        or conf_scale <= 0.0
        or not all(
            np.all(np.isfinite(arr)) for arr in (attr_w1, attr_b1, attr_ln_g, attr_ln_b, attr_w2, attr_b2, attr_skip_w)
        )
    ):
        raise ValueError("Invalid v5 attribution head arrays")

    version_heads = _load_stage2_heads(data, FEATURE_DIM_V4 + d_u)
    return _SourceModelV5(
        backbone=backbone,
        gates=gates,
        u_med=u_med,
        u_scale=u_scale,
        det_bias=det_bias,
        det_w=det_w,
        null_h_w=null_h_w,
        null_h_b=null_h_b,
        null_lin_mu_w=null_lin_mu_w,
        null_mu_bias=null_mu_bias,
        null_mu_w=null_mu_w,
        null_lin_sig_w=null_lin_sig_w,
        null_sig_bias=null_sig_bias,
        null_sig_w=null_sig_w,
        det_thresholds=det_thresholds,
        attr_w1=attr_w1,
        attr_b1=attr_b1,
        attr_ln_g=attr_ln_g,
        attr_ln_b=attr_ln_b,
        attr_w2=attr_w2,
        attr_b2=attr_b2,
        attr_skip_w=attr_skip_w,
        attr_temp=attr_temp,
        conf_scale=conf_scale,
        version_heads=version_heads,
    )


@functools.lru_cache(maxsize=4)
def _load_model_from_path(resolved: Path) -> _LoadedSourceModel:
    import numpy as np

    with np.load(resolved, allow_pickle=False) as data:
        if "schema" not in data:
            raise ValueError(f"Missing schema key in {resolved}")
        schema = str(np.asarray(data["schema"]).item())
        if schema == MODEL_SCHEMA_V5:
            return _load_model_v5(data)
        if schema == MODEL_SCHEMA_V4:
            return _load_model_v4(data)
        if schema != MODEL_SCHEMA_V3:
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

        mean, scale = _load_1d_standardizer(
            data, "mean", "scale", FEATURE_DIM, err_msg="Invalid standardization arrays mean/scale"
        )

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
        version_heads = _load_stage2_heads(data, stacked_dim)

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


def _load_model(weights_path: Path | None = None) -> _LoadedSourceModel:
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


def _eval_two_tower_backbone(
    x_1d: FloatArray,
    backbone: _TwoTowerBackbone,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float, float]:
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    x_std_1d = np.clip((x_1d - backbone.mean) / backbone.scale, -12.0, 12.0).astype(np.float32)
    r_std = x_std_1d[REGIME_START_V4 : REGIME_START_V4 + REGIME_DIM_V4].astype(np.float64)

    n_trunks = len(backbone.trunks)
    prov_sum = np.zeros(backbone.trunks[0].b_prov.shape[0], dtype=np.float64)
    sub_sum = np.zeros(backbone.n_subs, dtype=np.float64)
    for trunk in backbone.trunks:
        p_log, s_log = _eval_residual_trunk(trunk, x_std_1d)
        prov_sum += p_log
        sub_sum += s_log
    mean_prov = prov_sum / float(n_trunks)
    mean_sub = sub_sum / float(n_trunks)

    unk_logit = float(mean_prov[n_cls])
    max_ai = float(np.max(mean_prov[:n_cls]))
    lse_ai = max_ai + float(np.log(np.sum(np.exp(mean_prov[:n_cls] - max_ai))))
    hgb_vec = np.asarray(
        [_eval_packed_binary_hgb_raw(h, x_1d) for h in backbone.hgb_units],
        dtype=np.float64,
    )
    return mean_prov, mean_sub, hgb_vec, r_std, unk_logit, lse_ai


def _eval_2d_gates(ch: FloatArray, gates: _RouterGates2D, *, cap: float) -> FloatArray:
    import numpy as np

    ga = np.maximum(0.0, gates.gate_sign_a * (ch[gates.gate_idx_a] - gates.gate_tau_a))
    gb = np.maximum(0.0, gates.gate_sign_b * (ch[gates.gate_idx_b] - gates.gate_tau_b))
    return np.minimum(ga * gb, cap)


def _resolve_two_tower_subfamilies(
    best_cls: ProviderClass,
    mean_sub: FloatArray,
    x_1d: FloatArray,
    u_vec: FloatArray,
    backbone: _TwoTowerBackbone,
    version_heads: dict[str, _Stage2VersionHead],
) -> tuple[str | None, str | None]:
    import numpy as np

    best_sub: str | None = None
    best_sub_val = -1e18
    for s_idx, s_name in backbone.sub_by_provider[best_cls]:
        val = float(mean_sub[s_idx])
        if val > best_sub_val:
            best_sub_val = val
            best_sub = s_name
    version_subfamily = best_sub
    if best_cls in version_heads:
        version_subfamily = _eval_stage2_version_head(version_heads[best_cls], np.concatenate([x_1d, u_vec]))
    return best_sub, version_subfamily


def _classify_v4_features(x_1d: FloatArray, model: _SourceModelV4) -> SourceClassification:
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    mean_prov, mean_sub, hgb_vec, r_std, unk_logit, lse_ai = _eval_two_tower_backbone(x_1d, model.backbone)
    prov_log_odds = mean_prov[:n_cls] - unk_logit
    ai_vs_unk = np.asarray([lse_ai - unk_logit], dtype=np.float64)
    sub_log_odds = mean_sub - unk_logit

    u_vec = np.concatenate([prov_log_odds, ai_vs_unk, sub_log_odds, hgb_vec, r_std])
    h_pos = np.maximum(0.0, u_vec - model.gates.router_tau_pos)
    h_neg = np.maximum(0.0, model.gates.router_tau_neg - u_vec)
    g_2d = _eval_2d_gates(u_vec, model.gates, cap=64.0)

    phi = np.concatenate([u_vec, h_pos, h_neg, g_2d])
    z_raw = model.router_bias + phi @ model.router_w

    mu_r = model.null_mu_bias + r_std @ model.null_mu_w
    sig_lin = np.clip(model.null_sig_bias + r_std @ model.null_sig_w, -20.0, 20.0)
    sigma_r = np.log1p(np.exp(sig_lin)) + 0.25
    z_std = (z_raw - mu_r) / sigma_r

    pooled_z: dict[str, float] = {cls_name: float(z_std[idx]) for idx, cls_name in enumerate(ALL_PROVIDER_CLASSES)}

    thresholds = model.router_thresholds
    margins: dict[str, float] = {}
    excess: dict[str, float] = {}
    for idx_c, cls_name in enumerate(ALL_PROVIDER_CLASSES):
        other_pos = [max(0.0, pooled_z[other]) for other in ALL_PROVIDER_CLASSES if other != cls_name]
        max_other = max(other_pos) if other_pos else 0.0
        m_val = pooled_z[cls_name] - float(model.damping_gammas[idx_c]) * max_other
        margins[cls_name] = m_val
        excess[cls_name] = m_val - thresholds[cls_name]

    best_cls = max(ALL_PROVIDER_CLASSES, key=lambda c: excess[c])
    confidence = _sigmoid_scalar(model.conf_scale * excess[best_cls])
    if excess[best_cls] >= 0.0:
        label: SourceLabel = best_cls
        reason: SourceReason = "classified"
        fine_subfamily, version_subfamily = _resolve_two_tower_subfamilies(
            best_cls, mean_sub, x_1d, u_vec, model.backbone, model.version_heads
        )
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
        model_revision=MODEL_SCHEMA_V4,
    )


def _classify_v5_features(x_1d: FloatArray, model: _SourceModelV5) -> SourceClassification:
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    mean_prov, mean_sub, hgb_vec, r_std, unk_logit, lse_ai = _eval_two_tower_backbone(x_1d, model.backbone)
    prov_log_odds = mean_prov[:n_cls] - unk_logit
    ai_accept_pair = np.asarray([lse_ai - unk_logit, float(mean_prov[n_cls + 1])], dtype=np.float64)

    u_vec = np.concatenate([prov_log_odds, ai_accept_pair, mean_sub, hgb_vec, r_std])
    u_z = (u_vec - model.u_med) / model.u_scale
    u_sat = 8.0 * np.tanh(u_z / 8.0)
    h_pos = np.clip(u_z - model.gates.router_tau_pos, 0.0, 12.0)
    h_neg = np.clip(model.gates.router_tau_neg - u_z, 0.0, 12.0)
    g_2d = _eval_2d_gates(u_z, model.gates, cap=36.0)
    phi = np.concatenate([u_sat, h_pos, h_neg, g_2d])

    # 1. Decoupled Regime-Aware Supported-AI Acceptance Gate (10 provider channels + 1 global AI channel)
    z_det_raw = model.det_bias + phi @ model.det_w
    h_null = _silu_np(r_std @ model.null_h_w + model.null_h_b)
    mu_r = model.null_mu_bias + r_std @ model.null_lin_mu_w + h_null @ model.null_mu_w
    sig_lin = np.clip(model.null_sig_bias + r_std @ model.null_lin_sig_w + h_null @ model.null_sig_w, -8.0, 8.0)
    sigma_r = np.log1p(np.exp(sig_lin)) + 0.25
    z_det_std = (z_det_raw - mu_r) / sigma_r
    det_excess = z_det_std - model.det_thresholds
    accept_excess = float(np.max(det_excess))

    # 2. Decoupled Prior-Calibrated Conditional Multi-Class Attribution Head (threshold-invariant)
    h_attr = _silu_np(_layernorm_1d(phi @ model.attr_w1 + model.attr_b1, model.attr_ln_g, model.attr_ln_b))
    attr_logits = (phi @ model.attr_skip_w + h_attr @ model.attr_w2 + model.attr_b2) / model.attr_temp

    win_idx = int(np.argmax(attr_logits))
    best_cls: ProviderClass = ALL_PROVIDER_CLASSES[win_idx]
    prov_thr = model.det_thresholds[:n_cls]
    # Synthesize per-provider margins so argmax(margins - thresholds) == best_cls and
    # margins[best_cls] >= thresholds[best_cls] <=> accept_excess >= 0 across decoupled heads.
    prov_margins = prov_thr + accept_excess - (float(attr_logits[win_idx]) - attr_logits)
    pooled_z: dict[str, float] = dict(zip(ALL_PROVIDER_CLASSES, z_det_std[:n_cls].tolist(), strict=True))
    thresholds: dict[str, float] = dict(zip(ALL_PROVIDER_CLASSES, prov_thr.tolist(), strict=True))
    margins: dict[str, float] = dict(zip(ALL_PROVIDER_CLASSES, prov_margins.tolist(), strict=True))

    confidence = _sigmoid_scalar(model.conf_scale * accept_excess)
    if accept_excess >= 0.0:
        label: SourceLabel = best_cls
        reason: SourceReason = "classified"
        fine_subfamily, version_subfamily = _resolve_two_tower_subfamilies(
            best_cls, mean_sub, x_1d, u_vec, model.backbone, model.version_heads
        )
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
        thresholds=thresholds,
        model_revision=MODEL_SCHEMA_V5,
    )


def _classify_loaded_model(
    features: FloatArray,
    model: _LoadedSourceModel,
) -> SourceClassification:
    import numpy as np

    x_1d = np.asarray(features, dtype=np.float64)
    if x_1d.ndim != 1 or not np.all(np.isfinite(x_1d)):
        return _unavailable_result()

    if isinstance(model, (_SourceModelV4, _SourceModelV5)):
        if x_1d.shape != (FEATURE_DIM_V4,):
            return _unavailable_result()
        if isinstance(model, _SourceModelV5):
            return _classify_v5_features(x_1d, model)
        return _classify_v4_features(x_1d, model)

    if x_1d.shape != (FEATURE_DIM,):
        return _unavailable_result()

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
        model_revision=MODEL_SCHEMA_V3,
    )


def classify_source_features(
    features: FloatArray,
    *,
    weights_path: Path | None = None,
) -> SourceClassification:
    """Classify a precomputed 956-d (v3) or 512-d (v4/v5) feature vector into 10 providers or unknown."""
    import numpy as np

    x_1d = np.asarray(features, dtype=np.float64)
    if x_1d.ndim != 1 or x_1d.shape[0] not in (FEATURE_DIM, FEATURE_DIM_V4) or not np.all(np.isfinite(x_1d)):
        return _unavailable_result()

    return _classify_loaded_model(x_1d, _load_model(weights_path))


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

    def _extract_from_loaded(img: Image.Image) -> tuple[FloatArray, _LoadedSourceModel] | None:
        if min(img.size) < MIN_SIDE:
            return None
        loaded_model = _load_model(weights_path)
        extracted = (
            extract_512d_v4_features(img)
            if isinstance(loaded_model, (_SourceModelV4, _SourceModelV5))
            else extract_956d_features(img)
        )
        if extracted is None:
            return None
        return extracted, loaded_model

    if isinstance(source, (str, os.PathLike)):
        with PilImage.open(source) as opened:
            pair = _extract_from_loaded(opened)
    else:
        pair = _extract_from_loaded(source)

    if pair is None:
        return _unavailable_result()
    features, model = pair
    return _classify_loaded_model(features, model)
