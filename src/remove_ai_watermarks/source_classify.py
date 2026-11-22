"""Abstaining 10-provider pixel source-pipeline classifier with Stage-2 version attribution.

This classifier identifies the likely AI generation and export pixel pipeline across
10 provider families (``openai``, ``google``, ``bytedance``, ``flux``,
``stable_diffusion``, ``meta``, ``midjourney``, ``ideogram``, ``firefly``, ``xai``),
returning ``unknown`` when the image does not clear its calibrated acceptance
threshold. It is not a provenance verifier and does not detect or verify SynthID or
any invisible watermark.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import math
import os
import threading
import urllib.request
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from remove_ai_watermarks._internal.utils import model_cache_dir, staged_sibling
from remove_ai_watermarks.optional_deps import module_available

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

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
WEIGHTS_REVISION = "ebc2dcf0cface677535fe7c1903dd68793ee1f30"
GITHUB_WEIGHTS_REPO = "wiltodelta/raiw-source-classify"
GITHUB_WEIGHTS_TAG = "v8"
CONFIG_FILE = "config.json"
MODEL_FILE = "multi-provider-source-v8.npz"
MODEL_SHA256 = "7e7a9ee708bc2d64b45b87829b2c45e487aed3936220fb09843267662c99e0da"
MODEL_SCHEMA_V3 = "multi-provider-source-v3"
MODEL_SCHEMA_V4 = "multi-provider-source-v4"
MODEL_SCHEMA_V5 = "multi-provider-source-v5"
MODEL_SCHEMA_V7 = "multi-provider-source-v7"
MODEL_SCHEMA_V8 = "multi-provider-source-v8"
MODEL_SCHEMA = MODEL_SCHEMA_V8
SOURCE_CLASSIFY_EXTRA = "'remove-ai-watermarks[source-classify]'"

VIT_SHORT_EDGE_V7 = 440
VIT_DETAIL_SHORT_EDGE_V7 = 672
VIT_CROP_SIZE_V7 = 384
VIT_OUT_DIM_V7 = 28
SYN_IN_DIM_V7 = 67
MM_PROV_IN_DIM_V7 = 82

VIT_OUT_DIM_V8 = 59
SYN_IN_DIM_V8 = 106
MM_PROV_IN_DIM_V8 = 195

_IMAGENET_MEAN_RGB: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD_RGB: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Zero-leakage 26-d codec- & geometry-invariant macro-scene Born-Digital Graphics Veto
# over `x_512[464:480]` (Block D4 saturation/achromatic/clipping/palette) + `x_512[502:512]` (Block E scene).
_GFX_MEAN_26: tuple[float, ...] = (
    -0.00866062,
    -0.00369257,
    -0.03726238,
    -0.0227263,
    0.02860176,
    0.32059201,
    0.19851867,
    0.66889976,
    0.1629835,
    0.01311125,
    0.05012025,
    0.40246035,
    0.27661094,
    0.84704429,
    0.46758068,
    0.14369827,
    0.14918268,
    0.40246035,
    0.0852835,
    0.46758068,
    0.32059201,
    0.66889976,
    0.01311125,
    0.05012025,
    0.48955523,
    0.45048192,
)
_GFX_SCALE_26: tuple[float, ...] = (
    0.1038899,
    0.09136275,
    0.08558824,
    0.07392866,
    0.13310937,
    0.18244784,
    0.07912959,
    0.25684499,
    0.24843859,
    0.06171596,
    0.16283455,
    0.2023694,
    0.25040855,
    0.17989174,
    0.23715294,
    0.10365412,
    0.21141834,
    0.2023694,
    0.19076729,
    0.23715294,
    0.18244784,
    0.25684499,
    0.06171596,
    0.16283455,
    0.19453504,
    0.12261727,
)
_GFX_COEF_26: tuple[float, ...] = (
    0.28593333,
    -0.01833078,
    -0.54306846,
    0.23017136,
    -0.33946035,
    -0.20786306,
    0.41448862,
    -0.10714731,
    -0.19169794,
    0.09411158,
    0.19919772,
    0.22211316,
    -0.4822873,
    -0.57568771,
    0.56877772,
    -0.08537862,
    -0.67524892,
    0.22211316,
    -0.11560082,
    0.56877772,
    -0.20786306,
    -0.10714731,
    0.09411158,
    0.19919772,
    0.42927782,
    -0.20077164,
)
_GFX_INTERCEPT = -4.46995339

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
    synthetic_detected: bool = False
    synthetic_probability: float = 0.0

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
            "synthetic_detected": self.synthetic_detected,
            "synthetic_probability": self.synthetic_probability,
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


class _V7OnnxHolder:
    """Lazy, thread-safe ONNX Runtime session holder that drops raw graph bytes after compilation."""

    def __init__(self, npz_path: Path) -> None:
        self.npz_path = npz_path
        self._lock = threading.Lock()
        self._session: Any = None

    def get_session(self) -> Any:
        if self._session is not None:
            return self._session
        _require_dependencies(need_hub=False, need_onnx=True)
        with self._lock:
            if self._session is not None:
                return self._session
            import numpy as np

            with np.load(self.npz_path, allow_pickle=False) as data:
                raw_bytes = np.asarray(data["vit_onnx_bytes"], dtype=np.uint8).tobytes()
            ort: Any = import_module("onnxruntime")
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            omp_threads = os.environ.get("OMP_NUM_THREADS", "").strip()
            if omp_threads.isdigit() and int(omp_threads) > 0:
                sess_opts.intra_op_num_threads = int(omp_threads)
                sess_opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                raw_bytes,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            return self._session


@dataclass(frozen=True)
class _MMProvHeadV7:
    """Open-set 12-way (`10` providers + `foreign_ai` + `real`) multimodal attribution head."""

    in_mean: FloatArray
    in_scale: FloatArray
    w1: FloatArray
    b1: FloatArray
    ln_g: FloatArray
    ln_b: FloatArray
    w2: FloatArray
    b2: FloatArray
    skip_w: FloatArray
    prov_thresholds: FloatArray
    min_gap: float
    conf_scale: float
    phys_attr_beta: float = 0.0


_MMProvHeadV8 = _MMProvHeadV7


@dataclass(frozen=True)
class _MultiScaleGrayConfigV7:
    """Conditional second-scale (`672px` 2-crop) configuration for borderline widescreen/high-res frames."""

    short_edge: int
    min_short_px: int
    min_long_px: int
    gray_lo: float
    gfx_ceil: float
    detail_alpha: float
    thr_offset: float


@dataclass(frozen=True)
class _SourceModelV7:
    """Unified pickle-free v7 model combining 512-d physics/codec ensembles and embedded ViT ONNX."""

    base_v5: _SourceModelV5
    onnx_holder: _V7OnnxHolder
    vit_attr_alpha: float
    phys_veto_cap: float
    phys_veto_syn_floor: float
    syn_in_mean: FloatArray
    syn_in_scale: FloatArray
    syn_w1: FloatArray
    syn_b1: FloatArray
    syn_ln_g: FloatArray
    syn_ln_b: FloatArray
    syn_w2: FloatArray
    syn_b2: float
    syn_skip_w: FloatArray
    syn_threshold: float
    syn_conf_scale: float
    mm_prov: _MMProvHeadV7 | None = None
    ms_cfg: _MultiScaleGrayConfigV7 | None = None


@dataclass(frozen=True)
class _NativeTileConfigV8:
    """Dual-trigger 1:1 modulo-16 native-tile configuration for gray-zone & attribution recovery."""

    min_short_px: int
    min_long_px: int
    gray_lo: float
    gray_hi: float
    gfx_ceil: float
    min_photo_frac: float
    detail_alpha: float
    thr_offset: float
    attr_enable: bool
    attr_min_short_px: int
    attr_uncertain_gap: float


@dataclass(frozen=True)
class _MMStage2HeadV8:
    """Optional multimodal Stage-2 subfamily head with calibrated low-margin abstention."""

    subfamilies: tuple[str, ...]
    in_mean: FloatArray
    in_scale: FloatArray
    w1: FloatArray
    b1: FloatArray
    w2: FloatArray
    b2: FloatArray
    skip_w: FloatArray
    blend_alpha: float
    min_prob_margin: float


@dataclass(frozen=True)
class _SourceModelV8:
    """Unified pickle-free v8 model with 512-d physics, SVD+RINE+ArcFace ViT-59d ONNX, and 1:1 native tiles."""

    base_v5: _SourceModelV5
    onnx_holder: _V7OnnxHolder
    vit_attr_alpha: float
    phys_veto_cap: float
    phys_veto_syn_floor: float
    foreign_phys_cap: float
    foreign_mm_floor: float
    syn_in_mean: FloatArray
    syn_in_scale: FloatArray
    syn_w1: FloatArray
    syn_b1: FloatArray
    syn_ln_g: FloatArray
    syn_ln_b: FloatArray
    syn_w2: FloatArray
    syn_b2: float
    syn_skip_w: FloatArray
    syn_threshold: float
    syn_conf_scale: float
    mm_prov: _MMProvHeadV8 | None = None
    nt_cfg: _NativeTileConfigV8 | None = None
    mm_version_heads: dict[str, _MMStage2HeadV8] | None = None


_LoadedSourceModel: TypeAlias = _SourceModel | _SourceModelV4 | _SourceModelV5 | _SourceModelV7 | _SourceModelV8


def is_available() -> bool:
    """Return True if optional source-classifier dependencies are installed."""
    return module_available("numpy", "cv2", "huggingface_hub", "onnxruntime")


def _require_dependencies(*, need_hub: bool = True, need_onnx: bool = False) -> None:
    missing_core = not module_available("numpy", "cv2")
    missing_hub = need_hub and not module_available("huggingface_hub")
    missing_ort = need_onnx and not module_available("onnxruntime")
    if missing_core or missing_hub or missing_ort:
        raise ImportError(
            "Source-pipeline classification requires opencv-python-headless, "
            f"huggingface-hub, and onnxruntime. Install with: pip install {SOURCE_CLASSIFY_EXTRA}"
        )


def _verify_sha256(path: Path, expected_sha256: str) -> None:
    """Verify ``path`` matches ``expected_sha256``, raising :class:`RuntimeError` on mismatch."""
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"SHA-256 hash mismatch for {path.name}: expected {expected_sha256}, got {actual}. "
            "Clear the cached artifact or point to a verified freeze directory."
        )


def _github_weights_cache_path(github_repo: str, github_tag: str, model_file: str) -> Path:
    return model_cache_dir() / "source-classify" / github_repo.replace("/", "__") / github_tag / model_file


def _probe_github_weights_cache(
    github_repo: str,
    github_tag: str,
    model_file: str,
    expected_sha256: str,
) -> Path | None:
    dest = _github_weights_cache_path(github_repo, github_tag, model_file)
    if not dest.is_file():
        return None
    try:
        _verify_sha256(dest, expected_sha256)
        return dest
    except RuntimeError:
        log.warning("Cached GitHub weights failed SHA-256 verification at %s; re-downloading", dest)
        dest.unlink(missing_ok=True)
        return None


def _download_github_release_weights(
    github_repo: str,
    github_tag: str,
    model_file: str,
    expected_sha256: str,
) -> Path:
    cached = _probe_github_weights_cache(github_repo, github_tag, model_file, expected_sha256)
    if cached is not None:
        return cached

    dest = _github_weights_cache_path(github_repo, github_tag, model_file)
    url = f"https://github.com/{github_repo}/releases/download/{github_tag}/{model_file}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("GitHub Release download request: GET %s", url)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "remove-ai-watermarks/source-classify"},
    )
    try:
        with staged_sibling(dest) as staged_path:
            digest = hashlib.sha256()
            with (
                urllib.request.urlopen(req, timeout=60) as resp,  # noqa: S310 - fixed HTTPS source
                staged_path.open("wb") as out_handle,
            ):
                status = getattr(resp, "status", None)
                content_length = resp.headers.get("Content-Length") if hasattr(resp, "headers") else None
                log.info(
                    "GitHub Release download response: url=%s status=%s content_length=%s",
                    url,
                    status,
                    content_length,
                )
                while chunk := resp.read(1024 * 1024):
                    digest.update(chunk)
                    out_handle.write(chunk)
            actual = digest.hexdigest()
            if actual.lower() != expected_sha256.lower():
                raise RuntimeError(
                    f"SHA-256 hash mismatch for {model_file}: expected {expected_sha256}, got {actual}. "
                    "Clear the cached artifact or point to a verified freeze directory."
                )
    except RuntimeError:
        raise
    except Exception as exc:
        log.error("GitHub Release download failed for %s: %s", url, exc)
        raise RuntimeError(f"Failed to download verified weights from Hugging Face or {url}: {exc}") from exc
    return dest


@functools.lru_cache(maxsize=4)
def _resolve_hub_weights(
    repo_id: str,
    revision: str,
    model_file: str,
    config_file: str,
    expected_sha256: str,
    github_repo: str,
    github_tag: str,
) -> Path:
    _require_dependencies(need_hub=False, need_onnx=False)
    has_hub = module_available("huggingface_hub")
    hub: Any = import_module("huggingface_hub") if has_hub else None
    if hub is not None:
        try_cache = getattr(hub, "try_to_load_from_cache", None)
        cached = try_cache(repo_id=repo_id, filename=model_file, revision=revision) if callable(try_cache) else None
        if isinstance(cached, str):
            downloaded = Path(cached)
            _verify_sha256(downloaded, expected_sha256)
            return downloaded

    gh_cached = _probe_github_weights_cache(github_repo, github_tag, model_file, expected_sha256)
    if gh_cached is not None:
        return gh_cached

    if hub is None:
        log.info(
            "huggingface_hub is not installed; downloading weights from GitHub Release %s@%s",
            github_repo,
            github_tag,
        )
        return _download_github_release_weights(github_repo, github_tag, model_file, expected_sha256)

    hub_errors = getattr(hub, "errors", None)
    not_found_exc = getattr(hub_errors, "EntryNotFoundError", None)
    ignorable_config_exc: tuple[type[BaseException], ...] = (
        (not_found_exc,) if isinstance(not_found_exc, type) and issubclass(not_found_exc, BaseException) else ()
    )
    try:
        with contextlib.suppress(*ignorable_config_exc):
            hub.hf_hub_download(
                repo_id=repo_id,
                filename=config_file,
                revision=revision,
            )
        downloaded = Path(
            str(
                hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=model_file,
                    revision=revision,
                )
            )
        )
    except Exception as exc:
        log.warning(
            "Hugging Face Hub download failed for %s@%s/%s (%s); falling back to GitHub Release %s@%s",
            repo_id,
            revision,
            model_file,
            exc,
            github_repo,
            github_tag,
        )
        return _download_github_release_weights(github_repo, github_tag, model_file, expected_sha256)

    _verify_sha256(downloaded, expected_sha256)
    return downloaded


def resolve_weights_path(weights_path: Path | None = None) -> Path:
    """Resolve the verified local, Hugging Face Hub, or GitHub Release `.npz` weights path."""
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
    hub_args = (
        WEIGHTS_REPO,
        WEIGHTS_REVISION,
        MODEL_FILE,
        CONFIG_FILE,
        MODEL_SHA256,
        GITHUB_WEIGHTS_REPO,
        GITHUB_WEIGHTS_TAG,
    )
    resolved = _resolve_hub_weights(*hub_args)
    if not resolved.is_file():
        _resolve_hub_weights.cache_clear()
        resolved = _resolve_hub_weights(*hub_args)
    return resolved


_resolve_weights_path = resolve_weights_path


def _get_v7_onnx_session(model: _SourceModelV7 | _SourceModelV8) -> Any:
    return model.onnx_holder.get_session()


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


def _eval_stage2_version_head_raw(head: _Stage2VersionHead, x_stacked_1d: FloatArray) -> FloatArray:
    import numpy as np

    n_iter, c_tree, _ = head.feat_idx.shape
    lin = head.linear
    if lin is not None and lin.hgb_weight <= 0.0:
        x_lin = np.clip((x_stacked_1d - lin.mean) / lin.scale, -10.0, 10.0)
        return np.asarray(x_lin @ lin.w + lin.b, dtype=np.float64)
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
    return np.asarray(raw, dtype=np.float64)


def _eval_stage2_version_head(head: _Stage2VersionHead, x_stacked_1d: FloatArray) -> str:
    import numpy as np

    raw = _eval_stage2_version_head_raw(head, x_stacked_1d)
    if head.feat_idx.shape[1] == 1:
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


def _inspect_npz_1d_uint8_length(data: Any, key: str, *, bundle_tag: str = "v7") -> int:
    import numpy as np

    npfmt: Any = import_module("numpy.lib.format")
    member = f"{key}.npy"
    zf = getattr(data, "zip", None)
    if zf is not None and member in zf.namelist():
        with zf.open(member) as handle:
            version: tuple[int, int] = npfmt.read_magic(handle)
            header_fn: Any = npfmt.read_array_header_1_0 if version == (1, 0) else npfmt.read_array_header_2_0
            hdr: tuple[tuple[int, ...], bool, Any] = header_fn(handle)
            shape, _fortran, dtype = hdr
        if dtype != np.dtype(np.uint8) or len(shape) != 1:
            raise ValueError(f"Invalid {key} array in {bundle_tag} model bundle")
        return int(shape[0])
    raw = np.asarray(data[key])
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError(f"Invalid {key} array in {bundle_tag} model bundle")
    return int(raw.size)


def _load_syn_fusion_arrays(
    data: Any,
    *,
    in_dim: int,
    bundle_tag: str,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    FloatArray,
    float,
    float,
    float,
    float,
    float,
]:
    import numpy as np

    if "vit_onnx_bytes" not in data:
        raise ValueError(f"Missing vit_onnx_bytes in {bundle_tag} model bundle")
    if _inspect_npz_1d_uint8_length(data, "vit_onnx_bytes", bundle_tag=bundle_tag) < 32:
        raise ValueError(f"Invalid vit_onnx_bytes array in {bundle_tag} model bundle")

    syn_in_mean, syn_in_scale = _load_1d_standardizer(
        data,
        "syn_in_mean",
        "syn_in_scale",
        in_dim,
        err_msg=f"Invalid {bundle_tag} syn_in_mean/syn_in_scale standardization arrays",
    )
    syn_w1 = np.asarray(data["syn_w1"], dtype=np.float64)
    syn_b1 = np.asarray(data["syn_b1"], dtype=np.float64)
    syn_ln_g = np.asarray(data["syn_ln_g"], dtype=np.float64)
    syn_ln_b = np.asarray(data["syn_ln_b"], dtype=np.float64)
    syn_w2 = np.asarray(data["syn_w2"], dtype=np.float64)
    syn_b2 = float(np.asarray(data["syn_b2"], dtype=np.float64).squeeze())
    syn_skip_w = np.asarray(data["syn_skip_w"], dtype=np.float64)
    syn_threshold = float(np.asarray(data["syn_threshold"], dtype=np.float64).squeeze())
    syn_conf_scale = float(np.asarray(data["syn_conf_scale"], dtype=np.float64).squeeze())
    vit_attr_alpha = float(np.asarray(data["vit_attr_alpha"], dtype=np.float64).squeeze())
    phys_veto_cap = float(np.asarray(data["phys_veto_cap"], dtype=np.float64).squeeze())
    phys_veto_syn_floor = float(np.asarray(data["phys_veto_syn_floor"], dtype=np.float64).squeeze())

    h_syn_dim = syn_b1.shape[0] if syn_b1.ndim == 1 else -1
    if (
        h_syn_dim < 4
        or syn_w1.shape != (in_dim, h_syn_dim)
        or syn_ln_g.shape != (h_syn_dim,)
        or syn_ln_b.shape != (h_syn_dim,)
        or syn_w2.shape != (h_syn_dim,)
        or syn_skip_w.shape != (in_dim,)
        or not all(
            math.isfinite(val)
            for val in (syn_b2, syn_threshold, syn_conf_scale, vit_attr_alpha, phys_veto_cap, phys_veto_syn_floor)
        )
        or syn_conf_scale <= 0.0
        or vit_attr_alpha < 0.0
        or not all(np.all(np.isfinite(arr)) for arr in (syn_w1, syn_b1, syn_ln_g, syn_ln_b, syn_w2, syn_skip_w))
    ):
        raise ValueError(f"Invalid {bundle_tag} synthetic fusion or provider-blend arrays")
    return (
        syn_in_mean,
        syn_in_scale,
        syn_w1,
        syn_b1,
        syn_ln_g,
        syn_ln_b,
        syn_w2,
        syn_b2,
        syn_skip_w,
        syn_threshold,
        syn_conf_scale,
        vit_attr_alpha,
        phys_veto_cap,
        phys_veto_syn_floor,
    )


def _load_mm_prov_head(
    data: Any,
    *,
    in_dim: int,
    bundle_tag: str,
    allow_phys_beta: bool = False,
) -> _MMProvHeadV7 | None:
    import numpy as np

    if "mm_prov_w1" not in data:
        return None
    n_cls = len(ALL_PROVIDER_CLASSES)
    n_out = n_cls + 2
    mm_in_mean, mm_in_scale = _load_1d_standardizer(
        data,
        "mm_prov_in_mean",
        "mm_prov_in_scale",
        in_dim,
        err_msg=f"Invalid {bundle_tag} mm_prov standardization arrays",
    )
    mm_w1 = np.asarray(data["mm_prov_w1"], dtype=np.float64)
    mm_b1 = np.asarray(data["mm_prov_b1"], dtype=np.float64)
    mm_ln_g = np.asarray(data["mm_prov_ln_g"], dtype=np.float64)
    mm_ln_b = np.asarray(data["mm_prov_ln_b"], dtype=np.float64)
    mm_w2 = np.asarray(data["mm_prov_w2"], dtype=np.float64)
    mm_b2 = np.asarray(data["mm_prov_b2"], dtype=np.float64)
    mm_skip_w = np.asarray(data["mm_prov_skip_w"], dtype=np.float64)
    mm_thr = np.asarray(data["mm_prov_thresholds"], dtype=np.float64)
    mm_gap = float(np.asarray(data["mm_prov_min_gap"], dtype=np.float64).squeeze())
    mm_cscale = float(np.asarray(data["mm_prov_conf_scale"], dtype=np.float64).squeeze())
    mm_beta = (
        float(np.asarray(data["mm_prov_phys_beta"], dtype=np.float64).squeeze())
        if (allow_phys_beta and "mm_prov_phys_beta" in data)
        else 0.0
    )
    h_mm = mm_b1.shape[0] if mm_b1.ndim == 1 else -1
    if (
        h_mm < 4
        or mm_w1.shape != (in_dim, h_mm)
        or mm_ln_g.shape != (h_mm,)
        or mm_ln_b.shape != (h_mm,)
        or mm_w2.shape != (h_mm, n_out)
        or mm_b2.shape != (n_out,)
        or mm_skip_w.shape != (in_dim, n_out)
        or mm_thr.shape != (n_cls,)
        or not math.isfinite(mm_gap)
        or mm_gap < 0.0
        or not math.isfinite(mm_cscale)
        or mm_cscale <= 0.0
        or not math.isfinite(mm_beta)
        or mm_beta < 0.0
        or not all(
            np.all(np.isfinite(arr)) for arr in (mm_w1, mm_b1, mm_ln_g, mm_ln_b, mm_w2, mm_b2, mm_skip_w, mm_thr)
        )
    ):
        raise ValueError(f"Invalid {bundle_tag} multimodal open-set provider head arrays")
    return _MMProvHeadV7(
        in_mean=mm_in_mean,
        in_scale=mm_in_scale,
        w1=mm_w1,
        b1=mm_b1,
        ln_g=mm_ln_g,
        ln_b=mm_ln_b,
        w2=mm_w2,
        b2=mm_b2,
        skip_w=mm_skip_w,
        prov_thresholds=mm_thr,
        min_gap=mm_gap,
        conf_scale=mm_cscale,
        phys_attr_beta=mm_beta,
    )


def _load_model_v7(data: Any, resolved: Path) -> _SourceModelV7:
    import numpy as np

    base_v5 = _load_model_v5(data)
    (
        syn_in_mean,
        syn_in_scale,
        syn_w1,
        syn_b1,
        syn_ln_g,
        syn_ln_b,
        syn_w2,
        syn_b2,
        syn_skip_w,
        syn_threshold,
        syn_conf_scale,
        vit_attr_alpha,
        phys_veto_cap,
        phys_veto_syn_floor,
    ) = _load_syn_fusion_arrays(data, in_dim=SYN_IN_DIM_V7, bundle_tag="v7")
    mm_prov = _load_mm_prov_head(data, in_dim=MM_PROV_IN_DIM_V7, bundle_tag="v7")

    ms_cfg: _MultiScaleGrayConfigV7 | None = None
    if "ms_detail_alpha" in data:
        ms_se = int(np.asarray(data["ms_short_edge"], dtype=np.int64).squeeze())
        ms_min_s = int(np.asarray(data["ms_min_short_px"], dtype=np.int64).squeeze())
        ms_min_l = int(np.asarray(data["ms_min_long_px"], dtype=np.int64).squeeze())
        ms_gray_lo = float(np.asarray(data["ms_gray_lo"], dtype=np.float64).squeeze())
        ms_gfx_ceil = float(np.asarray(data["ms_gfx_ceil"], dtype=np.float64).squeeze())
        ms_alpha = float(np.asarray(data["ms_detail_alpha"], dtype=np.float64).squeeze())
        ms_off = float(np.asarray(data["ms_thr_offset"], dtype=np.float64).squeeze())
        if (
            ms_se < VIT_CROP_SIZE_V7
            or ms_min_s < MIN_SIDE
            or ms_min_l < ms_min_s
            or not all(math.isfinite(val) for val in (ms_gray_lo, ms_gfx_ceil, ms_alpha, ms_off))
            or not (0.0 <= ms_alpha <= 1.0)
        ):
            raise ValueError("Invalid v7 conditional multiscale gray-zone configuration")
        ms_cfg = _MultiScaleGrayConfigV7(
            short_edge=ms_se,
            min_short_px=ms_min_s,
            min_long_px=ms_min_l,
            gray_lo=ms_gray_lo,
            gfx_ceil=ms_gfx_ceil,
            detail_alpha=ms_alpha,
            thr_offset=ms_off,
        )

    return _SourceModelV7(
        base_v5=base_v5,
        onnx_holder=_V7OnnxHolder(resolved),
        vit_attr_alpha=vit_attr_alpha,
        phys_veto_cap=phys_veto_cap,
        phys_veto_syn_floor=phys_veto_syn_floor,
        syn_in_mean=syn_in_mean,
        syn_in_scale=syn_in_scale,
        syn_w1=syn_w1,
        syn_b1=syn_b1,
        syn_ln_g=syn_ln_g,
        syn_ln_b=syn_ln_b,
        syn_w2=syn_w2,
        syn_b2=syn_b2,
        syn_skip_w=syn_skip_w,
        syn_threshold=syn_threshold,
        syn_conf_scale=syn_conf_scale,
        mm_prov=mm_prov,
        ms_cfg=ms_cfg,
    )


def _load_model_v8(data: Any, resolved: Path) -> _SourceModelV8:
    import numpy as np

    base_v5 = _load_model_v5(data)
    (
        syn_in_mean,
        syn_in_scale,
        syn_w1,
        syn_b1,
        syn_ln_g,
        syn_ln_b,
        syn_w2,
        syn_b2,
        syn_skip_w,
        syn_threshold,
        syn_conf_scale,
        vit_attr_alpha,
        phys_veto_cap,
        phys_veto_syn_floor,
    ) = _load_syn_fusion_arrays(data, in_dim=SYN_IN_DIM_V8, bundle_tag="v8")
    foreign_phys_cap = (
        float(np.asarray(data["foreign_phys_cap"], dtype=np.float64).squeeze()) if "foreign_phys_cap" in data else 2.20
    )
    foreign_mm_floor = (
        float(np.asarray(data["foreign_mm_floor"], dtype=np.float64).squeeze()) if "foreign_mm_floor" in data else -0.80
    )
    if not (math.isfinite(foreign_phys_cap) and math.isfinite(foreign_mm_floor)):
        raise ValueError("Invalid v8 synthetic fusion or provider-blend arrays")

    mm_prov = _load_mm_prov_head(data, in_dim=MM_PROV_IN_DIM_V8, bundle_tag="v8", allow_phys_beta=True)

    nt_cfg: _NativeTileConfigV8 | None = None
    if "nt_detail_alpha" in data or "nt_attr_enable" in data:
        nt_min_s = int(np.asarray(data["nt_min_short_px"], dtype=np.int64).squeeze())
        nt_min_l = int(np.asarray(data["nt_min_long_px"], dtype=np.int64).squeeze())
        nt_gray_lo = float(np.asarray(data["nt_gray_lo"], dtype=np.float64).squeeze())
        nt_gray_hi = float(np.asarray(data["nt_gray_hi"], dtype=np.float64).squeeze())
        nt_gfx_ceil = float(np.asarray(data["nt_gfx_ceil"], dtype=np.float64).squeeze())
        nt_min_pf = float(np.asarray(data["nt_min_photo_frac"], dtype=np.float64).squeeze())
        nt_alpha = float(np.asarray(data["nt_detail_alpha"], dtype=np.float64).squeeze())
        nt_off = float(np.asarray(data["nt_thr_offset"], dtype=np.float64).squeeze())
        nt_attr_en = bool(int(np.asarray(data["nt_attr_enable"], dtype=np.int64).squeeze()))
        nt_attr_min_s = int(np.asarray(data["nt_attr_min_short_px"], dtype=np.int64).squeeze())
        nt_attr_gap = float(np.asarray(data["nt_attr_uncertain_gap"], dtype=np.float64).squeeze())
        if (
            nt_min_s < MIN_SIDE
            or nt_min_l < nt_min_s
            or nt_attr_min_s < MIN_SIDE
            or not all(
                math.isfinite(val)
                for val in (nt_gray_lo, nt_gray_hi, nt_gfx_ceil, nt_min_pf, nt_alpha, nt_off, nt_attr_gap)
            )
            or not (0.0 <= nt_alpha <= 1.0)
            or not (0.0 <= nt_min_pf <= 1.0)
            or nt_attr_gap < 0.0
        ):
            raise ValueError("Invalid v8 native-tile configuration")
        nt_cfg = _NativeTileConfigV8(
            min_short_px=nt_min_s,
            min_long_px=nt_min_l,
            gray_lo=nt_gray_lo,
            gray_hi=nt_gray_hi,
            gfx_ceil=nt_gfx_ceil,
            min_photo_frac=nt_min_pf,
            detail_alpha=nt_alpha,
            thr_offset=nt_off,
            attr_enable=nt_attr_en,
            attr_min_short_px=nt_attr_min_s,
            attr_uncertain_gap=nt_attr_gap,
        )

    mm_version_heads: dict[str, _MMStage2HeadV8] | None = None
    if "mm_s2_classes" in data:
        mm_version_heads = {}
        mm_s2_cls = [str(x) for x in np.asarray(data["mm_s2_classes"]).tolist()]
        for c_name in mm_s2_cls:
            if c_name not in ALL_PROVIDER_CLASSES:
                raise ValueError(f"Unknown provider in mm_s2_classes: {c_name!r}")
            subs = tuple(str(x) for x in np.asarray(data[f"mm_s2_{c_name}_subfamilies"]).tolist())
            n_k = len(subs)
            n_s1_subs = len(base_v5.backbone.sub_by_provider.get(c_name, ()))
            d_s2 = n_k + n_s1_subs + 102
            s2_mean, s2_scale = _load_1d_standardizer(
                data,
                f"mm_s2_{c_name}_in_mean",
                f"mm_s2_{c_name}_in_scale",
                d_s2,
                err_msg=f"Invalid mm_s2_{c_name} standardization arrays",
            )
            s2_w1 = np.asarray(data[f"mm_s2_{c_name}_w1"], dtype=np.float64)
            s2_b1 = np.asarray(data[f"mm_s2_{c_name}_b1"], dtype=np.float64)
            s2_w2 = np.asarray(data[f"mm_s2_{c_name}_w2"], dtype=np.float64)
            s2_b2 = np.asarray(data[f"mm_s2_{c_name}_b2"], dtype=np.float64)
            s2_sw = np.asarray(data[f"mm_s2_{c_name}_skip_w"], dtype=np.float64)
            s2_alpha = float(np.asarray(data[f"mm_s2_{c_name}_blend_alpha"], dtype=np.float64).squeeze())
            s2_min_m = float(np.asarray(data[f"mm_s2_{c_name}_min_prob_margin"], dtype=np.float64).squeeze())
            h_s2 = s2_b1.shape[0] if s2_b1.ndim == 1 else -1
            if (
                n_k < 2
                or h_s2 < 4
                or s2_w1.shape != (d_s2, h_s2)
                or s2_w2.shape != (h_s2, n_k)
                or s2_b2.shape != (n_k,)
                or s2_sw.shape != (d_s2, n_k)
                or not (math.isfinite(s2_alpha) and 0.0 <= s2_alpha <= 1.0)
                or not (math.isfinite(s2_min_m) and 0.0 <= s2_min_m < 1.0)
                or not all(np.all(np.isfinite(a)) for a in (s2_w1, s2_b1, s2_w2, s2_b2, s2_sw))
            ):
                raise ValueError(f"Invalid v8 mm_s2 head arrays for {c_name}")
            mm_version_heads[c_name] = _MMStage2HeadV8(
                subfamilies=subs,
                in_mean=s2_mean,
                in_scale=s2_scale,
                w1=s2_w1,
                b1=s2_b1,
                w2=s2_w2,
                b2=s2_b2,
                skip_w=s2_sw,
                blend_alpha=s2_alpha,
                min_prob_margin=s2_min_m,
            )

    return _SourceModelV8(
        base_v5=base_v5,
        onnx_holder=_V7OnnxHolder(resolved),
        vit_attr_alpha=vit_attr_alpha,
        phys_veto_cap=phys_veto_cap,
        phys_veto_syn_floor=phys_veto_syn_floor,
        foreign_phys_cap=foreign_phys_cap,
        foreign_mm_floor=foreign_mm_floor,
        syn_in_mean=syn_in_mean,
        syn_in_scale=syn_in_scale,
        syn_w1=syn_w1,
        syn_b1=syn_b1,
        syn_ln_g=syn_ln_g,
        syn_ln_b=syn_ln_b,
        syn_w2=syn_w2,
        syn_b2=syn_b2,
        syn_skip_w=syn_skip_w,
        syn_threshold=syn_threshold,
        syn_conf_scale=syn_conf_scale,
        mm_prov=mm_prov,
        nt_cfg=nt_cfg,
        mm_version_heads=mm_version_heads,
    )


@functools.lru_cache(maxsize=4)
def _load_model_from_path(resolved: Path) -> _LoadedSourceModel:
    import numpy as np

    with np.load(resolved, allow_pickle=False) as data:
        if "schema" not in data:
            raise ValueError(f"Missing schema key in {resolved}")
        schema = str(np.asarray(data["schema"]).item())
        if schema == MODEL_SCHEMA_V8:
            return _load_model_v8(data, resolved)
        if schema == MODEL_SCHEMA_V7:
            return _load_model_v7(data, resolved)
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
        synthetic_detected=False,
        synthetic_probability=0.0,
    )


@functools.lru_cache(maxsize=1)
def _gfx_veto_arrays() -> tuple[FloatArray, FloatArray, FloatArray]:
    import numpy as np

    return (
        np.asarray(_GFX_MEAN_26, dtype=np.float64),
        np.asarray(_GFX_SCALE_26, dtype=np.float64),
        np.asarray(_GFX_COEF_26, dtype=np.float64),
    )


@functools.lru_cache(maxsize=1)
def _imagenet_norm_arrays() -> tuple[FloatArray, FloatArray]:
    import numpy as np

    return (
        np.asarray(_IMAGENET_MEAN_RGB, dtype=np.float32).reshape(1, 3, 1, 1),
        np.asarray(_IMAGENET_STD_RGB, dtype=np.float32).reshape(1, 3, 1, 1),
    )


def eval_born_digital_graphics_score(features_512d: FloatArray) -> float:
    """Score the 26-d codec-invariant macro-scene Born-Digital Graphics logit on a 512-d v4-v7 vector."""
    import numpy as np

    x_1d = np.asarray(features_512d, dtype=np.float64)
    if x_1d.shape != (FEATURE_DIM_V4,) or not np.all(np.isfinite(x_1d)):
        raise ValueError(f"Expected finite ({FEATURE_DIM_V4},) feature vector, got {x_1d.shape}")
    scene_26 = np.concatenate([x_1d[464:480], x_1d[502:512]])
    mean_26, scale_26, coef_26 = _gfx_veto_arrays()
    z_26 = np.clip((scene_26 - mean_26) / scale_26, -12.0, 12.0)
    return float(z_26 @ coef_26 + _GFX_INTERCEPT)


def _orient_rgb_image(image: Image.Image) -> Image.Image:
    from PIL import ImageOps

    orientation = image.getexif().get(274, 1)
    oriented = ImageOps.exif_transpose(image) if orientation not in (1, None) else image
    return oriented if oriented.mode == "RGB" else oriented.convert("RGB")


def _resize_short_edge_bilinear(rgb: Image.Image, short_edge: int) -> Image.Image:
    from PIL import Image as PilImage

    w, h = rgb.size
    if w <= h:
        new_w = short_edge
        new_h = max(VIT_CROP_SIZE_V7, round(h * short_edge / w))
    else:
        new_h = short_edge
        new_w = max(VIT_CROP_SIZE_V7, round(w * short_edge / h))
    return rgb.resize((new_w, new_h), PilImage.Resampling.BILINEAR)


def _crops_to_imagenet_nchw(src: Image.Image, boxes: Sequence[tuple[int, int, int, int]]) -> FloatArray:
    import numpy as np

    buf = np.empty((len(boxes), 3, VIT_CROP_SIZE_V7, VIT_CROP_SIZE_V7), dtype=np.float32)
    for idx_b, box in enumerate(boxes):
        buf[idx_b] = np.transpose(np.asarray(src.crop(box), dtype=np.uint8), (2, 0, 1))
    mean_rgb, std_rgb = _imagenet_norm_arrays()
    buf *= 1.0 / 255.0
    buf -= mean_rgb
    buf /= std_rgb
    return buf


def preprocess_vit_384_chw(image: Image.Image) -> FloatArray:
    """Preprocess a PIL image into a C-contiguous 1x3x384x384 float32 normalized ViT tensor."""
    resized = _resize_short_edge_bilinear(_orient_rgb_image(image), VIT_SHORT_EDGE_V7)
    new_w, new_h = resized.size
    left = (new_w - VIT_CROP_SIZE_V7) // 2
    top = (new_h - VIT_CROP_SIZE_V7) // 2
    return _crops_to_imagenet_nchw(resized, ((left, top, left + VIT_CROP_SIZE_V7, top + VIT_CROP_SIZE_V7),))


def preprocess_vit_384_detail_2chw(
    image: Image.Image,
    *,
    short_edge: int = VIT_DETAIL_SHORT_EDGE_V7,
) -> FloatArray:
    """Extract two complementary 384x384 higher-scale crops (`2x3x384x384`) along the long axis."""
    rgb = _orient_rgb_image(image)
    target_se = max(VIT_CROP_SIZE_V7, min(int(short_edge), min(rgb.size)))
    resized = _resize_short_edge_bilinear(rgb, target_se)
    new_w, new_h = resized.size

    if new_w >= new_h:
        span_x = new_w - VIT_CROP_SIZE_V7
        mid_y = (new_h - VIT_CROP_SIZE_V7) // 2
        boxes = (
            (span_x // 4, mid_y, span_x // 4 + VIT_CROP_SIZE_V7, mid_y + VIT_CROP_SIZE_V7),
            ((3 * span_x) // 4, mid_y, (3 * span_x) // 4 + VIT_CROP_SIZE_V7, mid_y + VIT_CROP_SIZE_V7),
        )
    else:
        span_y = new_h - VIT_CROP_SIZE_V7
        mid_x = (new_w - VIT_CROP_SIZE_V7) // 2
        boxes = (
            (mid_x, span_y // 4, mid_x + VIT_CROP_SIZE_V7, span_y // 4 + VIT_CROP_SIZE_V7),
            (mid_x, (3 * span_y) // 4, mid_x + VIT_CROP_SIZE_V7, (3 * span_y) // 4 + VIT_CROP_SIZE_V7),
        )
    return _crops_to_imagenet_nchw(resized, boxes)


def _snap_crop_16(pos: int, limit: int) -> int:
    """Snap a crop offset to a multiple of 16 pixels within `[0, limit]`."""
    return min(max(0, (pos + 8) & ~15), limit & ~15)


def _snap_long_pair_16(limit: int) -> tuple[int, int]:
    c0 = _snap_crop_16(limit // 4, limit)
    c1 = _snap_crop_16((3 * limit) // 4, limit)
    if c0 == c1 and limit >= 16:
        return 0, limit & ~15
    return c0, c1


def _snap_short_pair_16(limit: int) -> tuple[int, int]:
    if limit >= 64:
        return _snap_crop_16(limit // 3, limit), _snap_crop_16((2 * limit) // 3, limit)
    mid = _snap_crop_16(limit // 2, limit)
    return mid, mid


def preprocess_vit_384_native_2chw(image: Image.Image) -> FloatArray:
    """Extract two 1:1 unresampled modulo-16-aligned 384x384 crops (`2x3x384x384`) for v8."""
    rgb = _orient_rgb_image(image)
    crop_sz = VIT_CROP_SIZE_V7
    src = _resize_short_edge_bilinear(rgb, crop_sz) if min(rgb.size) < crop_sz else rgb
    new_w, new_h = src.size
    limit_x, limit_y = new_w - crop_sz, new_h - crop_sz
    if new_w >= new_h:
        (x0, x1), (y0, y1) = _snap_long_pair_16(limit_x), _snap_short_pair_16(limit_y)
    else:
        (x0, x1), (y0, y1) = _snap_short_pair_16(limit_x), _snap_long_pair_16(limit_y)
    return _crops_to_imagenet_nchw(
        src,
        ((x0, y0, x0 + crop_sz, y0 + crop_sz), (x1, y1, x1 + crop_sz, y1 + crop_sz)),
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


def _classify_v4_features(
    x_1d: FloatArray,
    model: _SourceModelV4,
) -> SourceClassification:
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
        synthetic_detected=label != "unknown",
        synthetic_probability=confidence,
    )


def _eval_v5_core_state(
    x_1d: FloatArray,
    model: _SourceModelV5,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float]:
    """Return `(z_det_std, attr_logits, r_std, mean_sub, u_vec, accept_excess)` for v5/v7 models."""
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

    z_det_raw = model.det_bias + phi @ model.det_w
    h_null = _silu_np(r_std @ model.null_h_w + model.null_h_b)
    mu_r = model.null_mu_bias + r_std @ model.null_lin_mu_w + h_null @ model.null_mu_w
    sig_lin = np.clip(model.null_sig_bias + r_std @ model.null_lin_sig_w + h_null @ model.null_sig_w, -8.0, 8.0)
    sigma_r = np.log1p(np.exp(sig_lin)) + 0.25
    z_det_std = (z_det_raw - mu_r) / sigma_r
    accept_excess = float(np.max(z_det_std - model.det_thresholds))

    h_attr = _silu_np(_layernorm_1d(phi @ model.attr_w1 + model.attr_b1, model.attr_ln_g, model.attr_ln_b))
    attr_logits = (phi @ model.attr_skip_w + h_attr @ model.attr_w2 + model.attr_b2) / model.attr_temp
    return z_det_std, attr_logits, r_std, mean_sub, u_vec, accept_excess


def _top2_gap(arr: FloatArray) -> float:
    import numpy as np

    part = np.partition(arr, -2)
    return float(part[-1] - part[-2])


def _build_v5_like_result(
    *,
    x_1d: FloatArray,
    base_model: _SourceModelV5,
    z_det_std: FloatArray,
    attr_logits: FloatArray,
    mean_sub: FloatArray,
    u_vec: FloatArray,
    accept_excess: float,
    synthetic_detected: bool,
    synthetic_probability: float,
    model_revision: str,
    version_resolver: Callable[[ProviderClass], tuple[bool, str | None]] | None = None,
) -> SourceClassification:
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    win_idx = int(np.argmax(attr_logits))
    best_cls: ProviderClass = ALL_PROVIDER_CLASSES[win_idx]
    prov_thr = base_model.det_thresholds[:n_cls]
    prov_margins = prov_thr + accept_excess - (float(attr_logits[win_idx]) - attr_logits)
    pooled_z: dict[str, float] = dict(zip(ALL_PROVIDER_CLASSES, z_det_std[:n_cls].tolist(), strict=True))
    thresholds: dict[str, float] = dict(zip(ALL_PROVIDER_CLASSES, prov_thr.tolist(), strict=True))
    margins: dict[str, float] = dict(zip(ALL_PROVIDER_CLASSES, prov_margins.tolist(), strict=True))

    if accept_excess >= 0.0:
        attr_gap = max(0.0, _top2_gap(attr_logits))
        confidence = _sigmoid_scalar(min(base_model.conf_scale * accept_excess, attr_gap))
        label: SourceLabel = best_cls
        reason: SourceReason = "classified"
        override_pair = version_resolver(best_cls) if version_resolver is not None else (False, None)
        if override_pair[0]:
            fine_subfamily, _ = _resolve_two_tower_subfamilies(best_cls, mean_sub, x_1d, u_vec, base_model.backbone, {})
            version_subfamily = override_pair[1]
        else:
            fine_subfamily, version_subfamily = _resolve_two_tower_subfamilies(
                best_cls, mean_sub, x_1d, u_vec, base_model.backbone, base_model.version_heads
            )
    else:
        confidence = _sigmoid_scalar(base_model.conf_scale * accept_excess)
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
        model_revision=model_revision,
        synthetic_detected=synthetic_detected,
        synthetic_probability=synthetic_probability,
    )


def _classify_v5_features(
    x_1d: FloatArray,
    model: _SourceModelV5,
    *,
    model_revision: str = MODEL_SCHEMA_V5,
) -> SourceClassification:
    z_det_std, attr_logits, _r_std, mean_sub, u_vec, accept_excess = _eval_v5_core_state(x_1d, model)
    return _build_v5_like_result(
        x_1d=x_1d,
        base_model=model,
        z_det_std=z_det_std,
        attr_logits=attr_logits,
        mean_sub=mean_sub,
        u_vec=u_vec,
        accept_excess=accept_excess,
        synthetic_detected=accept_excess >= 0.0,
        synthetic_probability=_sigmoid_scalar(model.conf_scale * accept_excess),
        model_revision=model_revision,
    )


def _logsumexp_pair(a: float, b: float) -> float:
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))


def build_v7_syn_input_1d(
    x_1d: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_out_1d: FloatArray,
) -> FloatArray:
    """Assemble the 67-d open-world multimodal synthetic-fusion input vector for schema v7.

    `vit_out_1d` layout (`28-d`):
    `[orig_syn(0), prov_10(1:11), foreign_ai(11), real_non_ai(12), gfx_veto(13), dom_syn(14), emb(15:28)]`.
    """
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    vit_syn = float(vit_out_1d[0])
    vit_all_ai = vit_out_1d[1 : 2 + n_cls]
    vit_for = float(vit_out_1d[1 + n_cls])
    vit_real = float(vit_out_1d[2 + n_cls])
    vit_max_ai = float(np.max(vit_all_ai))
    vit_lse_ai = float((vit_max_ai + math.log(float(np.sum(np.exp(vit_all_ai - vit_max_ai))))) - vit_real)
    vit_max_lo = vit_max_ai - vit_real
    vit_for_lo = vit_for - vit_real
    vit_tail = vit_out_1d[3 + n_cls : VIT_OUT_DIM_V7]

    z_sat = 8.0 * np.tanh(z_det_std / 8.0)
    exc_sat = 8.0 * math.tanh(accept_excess / 8.0)
    ch10 = float(z_det_std[n_cls])
    gfx_score = eval_born_digital_graphics_score(x_1d)
    scalars_head = np.asarray([vit_syn, vit_lse_ai, vit_max_lo, vit_for_lo], dtype=np.float64)
    scalars_tail = np.asarray(
        [exc_sat, max(0.0, ch10), min(0.0, ch10), gfx_score, max(0.0, gfx_score)],
        dtype=np.float64,
    )
    return np.concatenate([scalars_head, vit_tail, z_sat, scalars_tail, r_std])


def build_v7_mm_prov_input_1d(
    attr_logits: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_out_1d: FloatArray,
) -> FloatArray:
    """Assemble the 82-d multimodal open-set 10-provider + `unknown` attribution input vector."""
    import numpy as np

    attr_sat = 8.0 * np.tanh(attr_logits / 8.0)
    z_sat = 8.0 * np.tanh(z_det_std / 8.0)
    exc_sat = np.asarray([8.0 * math.tanh(accept_excess / 8.0)], dtype=np.float64)
    return np.concatenate([attr_sat, z_sat, exc_sat, vit_out_1d.astype(np.float64), r_std])


def _eval_syn_head_scalar(s_in: FloatArray, model: _SourceModelV7 | _SourceModelV8) -> float:
    import numpy as np

    s_std = np.clip((s_in - model.syn_in_mean) / model.syn_in_scale, -12.0, 12.0)
    h_syn = _silu_np(_layernorm_1d(s_std @ model.syn_w1 + model.syn_b1, model.syn_ln_g, model.syn_ln_b))
    return float(s_std @ model.syn_skip_w + h_syn @ model.syn_w2 + model.syn_b2)


def _eval_mm_prov_head_log_odds(p_in: FloatArray, mm_head: _MMProvHeadV7) -> FloatArray:
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    p_std = np.clip((p_in - mm_head.in_mean) / mm_head.in_scale, -12.0, 12.0)
    h_p = _silu_np(_layernorm_1d(p_std @ mm_head.w1 + mm_head.b1, mm_head.ln_g, mm_head.ln_b))
    logits_12 = p_std @ mm_head.skip_w + h_p @ mm_head.w2 + mm_head.b2
    open_null = _logsumexp_pair(float(logits_12[n_cls]), float(logits_12[n_cls + 1]))
    return logits_12[:n_cls] - open_null


def _apply_phys_veto(accept_excess: float, syn_margin: float, model: _SourceModelV7 | _SourceModelV8) -> float:
    if 0.0 <= accept_excess < model.phys_veto_cap and syn_margin < model.phys_veto_syn_floor:
        return -1e-4
    return accept_excess


def _fuse_mm_decision(
    *,
    attr_logits: FloatArray,
    mm_log_odds: FloatArray,
    eff_phys_excess: float,
    syn_margin: float,
    vit_attr_alpha: float,
    syn_conf_scale: float,
    base_conf_scale: float,
    mm_prov: _MMProvHeadV7 | None,
    foreign_phys_cap: float = 2.20,
    foreign_mm_floor: float = -0.80,
) -> tuple[FloatArray, float, bool, float]:
    import numpy as np

    joint_attr_logits = attr_logits + vit_attr_alpha * mm_log_odds
    final_excess = eff_phys_excess
    if eff_phys_excess >= 0.0:
        synthetic_detected = True
        scaled_joint = max(0.0, syn_conf_scale * syn_margin, base_conf_scale * eff_phys_excess)
        synthetic_probability = _sigmoid_scalar(scaled_joint)
        if mm_prov is not None and eff_phys_excess < foreign_phys_cap and float(np.max(mm_log_odds)) < foreign_mm_floor:
            final_excess = -1e-4
    else:
        synthetic_detected = syn_margin >= 0.0
        synthetic_probability = _sigmoid_scalar(syn_conf_scale * syn_margin)
        if synthetic_detected and mm_prov is not None:
            win_mm = int(np.argmax(mm_log_odds))
            mm_exc = float(mm_log_odds[win_mm] - mm_prov.prov_thresholds[win_mm])
            if mm_exc >= 0.0 and _top2_gap(mm_log_odds) >= mm_prov.min_gap:
                joint_attr_logits = mm_log_odds + mm_prov.phys_attr_beta * attr_logits
                final_excess = max(1e-4, mm_exc * (mm_prov.conf_scale / max(base_conf_scale, 1e-4)))
    return joint_attr_logits, final_excess, synthetic_detected, synthetic_probability


def _run_vit_onnx(
    model: _SourceModelV7 | _SourceModelV8,
    pixel_values: FloatArray,
    expected_shape: tuple[int, ...],
    *,
    label: str,
) -> FloatArray:
    import numpy as np

    out = _get_v7_onnx_session(model).run(["vit_out"], {"pixel_values": pixel_values})[0]
    arr = np.asarray(out, dtype=np.float64)
    if len(expected_shape) == 1:
        arr = arr.reshape(-1)
    if arr.shape != expected_shape or not np.all(np.isfinite(arr)):
        raise RuntimeError(f"Invalid {label} ONNX output shape {arr.shape}")
    return arr


def _eval_v7_syn_raw(
    x_1d: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_out_1d: FloatArray,
    model: _SourceModelV7,
) -> float:
    return _eval_syn_head_scalar(build_v7_syn_input_1d(x_1d, z_det_std, accept_excess, r_std, vit_out_1d), model)


def _eval_v7_mm_prov_log_odds(
    attr_logits: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_out_1d: FloatArray,
    mm_head: _MMProvHeadV7,
) -> FloatArray:
    return _eval_mm_prov_head_log_odds(
        build_v7_mm_prov_input_1d(attr_logits, z_det_std, accept_excess, r_std, vit_out_1d), mm_head
    )


def _should_run_v7_detail_pass(
    x_1d: FloatArray,
    vit_primary_1d: FloatArray,
    eff_phys_excess: float,
    syn_margin_primary: float,
    image_size: tuple[int, int] | None,
    ms_cfg: _MultiScaleGrayConfigV7 | None,
) -> bool:
    if ms_cfg is None or ms_cfg.detail_alpha <= 0.0:
        return False
    if eff_phys_excess >= 0.0 or not (ms_cfg.gray_lo <= syn_margin_primary < 0.0):
        return False
    if image_size is not None:
        w, h = image_size
        if min(w, h) < ms_cfg.min_short_px or max(w, h) < ms_cfg.min_long_px:
            return False
    n_cls = len(ALL_PROVIDER_CLASSES)
    if float(vit_primary_1d[3 + n_cls]) >= ms_cfg.gfx_ceil:
        return False
    return eval_born_digital_graphics_score(x_1d) < 1.20


def _classify_v7_with_vit(
    x_1d: FloatArray,
    vit_out_1d: FloatArray,
    model: _SourceModelV7,
    *,
    vit_detail_2d: FloatArray | None = None,
    vit_detail_loader: Callable[[], FloatArray] | None = None,
    image_size: tuple[int, int] | None = None,
    precomputed_v5_state: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float] | None = None,
) -> SourceClassification:
    n_cls = len(ALL_PROVIDER_CLASSES)
    z_det_std, attr_logits, r_std, mean_sub, u_vec, accept_excess = (
        precomputed_v5_state if precomputed_v5_state is not None else _eval_v5_core_state(x_1d, model.base_v5)
    )

    syn_raw = _eval_v7_syn_raw(x_1d, z_det_std, accept_excess, r_std, vit_out_1d, model)
    syn_margin = syn_raw - model.syn_threshold
    eff_phys_excess = _apply_phys_veto(accept_excess, syn_margin, model)

    vit_for_prov = vit_out_1d
    ms_cfg = model.ms_cfg
    want_detail = ms_cfg is not None and _should_run_v7_detail_pass(
        x_1d, vit_out_1d, eff_phys_excess, syn_margin, image_size, ms_cfg
    )
    if want_detail and vit_detail_2d is None and vit_detail_loader is not None:
        vit_detail_2d = vit_detail_loader()
    if (
        want_detail
        and vit_detail_2d is not None
        and ms_cfg is not None
        and float(vit_detail_2d[0, 3 + n_cls]) < ms_cfg.gfx_ceil
        and float(vit_detail_2d[1, 3 + n_cls]) < ms_cfg.gfx_ceil
    ):
        syn_d0 = _eval_v7_syn_raw(x_1d, z_det_std, accept_excess, r_std, vit_detail_2d[0], model)
        syn_d1 = _eval_v7_syn_raw(x_1d, z_det_std, accept_excess, r_std, vit_detail_2d[1], model)
        syn_detail_pool = 0.75 * max(syn_d0, syn_d1) + 0.25 * min(syn_d0, syn_d1)
        syn_raw = (1.0 - ms_cfg.detail_alpha) * syn_raw + ms_cfg.detail_alpha * syn_detail_pool
        syn_margin = syn_raw - (model.syn_threshold + ms_cfg.thr_offset)
        vit_for_prov = 0.50 * vit_out_1d + 0.25 * (vit_detail_2d[0] + vit_detail_2d[1])

    if model.mm_prov is not None:
        mm_log_odds = _eval_v7_mm_prov_log_odds(
            attr_logits, z_det_std, accept_excess, r_std, vit_for_prov, model.mm_prov
        )
    else:
        open_null_v = _logsumexp_pair(float(vit_for_prov[1 + n_cls]), float(vit_for_prov[2 + n_cls]))
        mm_log_odds = vit_for_prov[1 : 1 + n_cls] - open_null_v

    joint_attr_logits, final_excess, synthetic_detected, synthetic_probability = _fuse_mm_decision(
        attr_logits=attr_logits,
        mm_log_odds=mm_log_odds,
        eff_phys_excess=eff_phys_excess,
        syn_margin=syn_margin,
        vit_attr_alpha=model.vit_attr_alpha,
        syn_conf_scale=model.syn_conf_scale,
        base_conf_scale=model.base_v5.conf_scale,
        mm_prov=model.mm_prov,
    )
    return _build_v5_like_result(
        x_1d=x_1d,
        base_model=model.base_v5,
        z_det_std=z_det_std,
        attr_logits=joint_attr_logits,
        mean_sub=mean_sub,
        u_vec=u_vec,
        accept_excess=final_excess,
        synthetic_detected=synthetic_detected,
        synthetic_probability=synthetic_probability,
        model_revision=MODEL_SCHEMA_V7,
    )


def _eval_v7_vit_onnx(image: Image.Image, model: _SourceModelV7) -> FloatArray:
    return _run_vit_onnx(model, preprocess_vit_384_chw(image), (VIT_OUT_DIM_V7,), label="v7 ViT")


def _eval_v7_vit_detail_onnx(image: Image.Image, model: _SourceModelV7, *, short_edge: int) -> FloatArray:
    return _run_vit_onnx(
        model,
        preprocess_vit_384_detail_2chw(image, short_edge=short_edge),
        (2, VIT_OUT_DIM_V7),
        label="v7 ViT detail",
    )


def build_v8_syn_input_1d(
    x_1d: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_out_1d: FloatArray,
) -> FloatArray:
    """Assemble the 106-d spatial + ArcFace + RINE multimodal synthetic-fusion input vector for v8.

    `vit_out_1d` layout (`59-d`):
    `[orig_syn(0), prov_10(1:11), foreign_ai(11), real_non_ai(12), gfx_veto(13), dom_syn(14),`
    ` emb_24(15:39), proto_sim_10(39:49), sp_top12(49), sp_top48(50), sp_top144(51), sp_lse(52),`
    ` sp_contrast(53), sp_tv_edge(54), non_gfx_frac(55), photo_region_syn(56),`
    ` sp_smooth_syn(57), sp_tex_syn(58)]`.
    """
    import numpy as np

    n_cls = len(ALL_PROVIDER_CLASSES)
    vit_syn = float(vit_out_1d[0])
    vit_all_ai = vit_out_1d[1 : 2 + n_cls]
    vit_for = float(vit_out_1d[1 + n_cls])
    vit_real = float(vit_out_1d[2 + n_cls])
    vit_gfx = float(vit_out_1d[3 + n_cls])
    dom_syn = float(vit_out_1d[4 + n_cls])

    vit_max_ai = float(np.max(vit_all_ai))
    vit_lse_ai = float((vit_max_ai + math.log(float(np.sum(np.exp(vit_all_ai - vit_max_ai))))) - vit_real)
    vit_max_lo = vit_max_ai - vit_real
    vit_for_lo = vit_for - vit_real
    gfx_softplus = math.log1p(math.exp(max(-15.0, min(15.0, vit_gfx))))
    dom_syn_clean = dom_syn - 2.0 * gfx_softplus

    max_proto_sim = float(np.max(vit_out_1d[39:49]))
    sp_top12 = float(vit_out_1d[49])
    sp_top48 = float(vit_out_1d[50])
    sp_contrast = float(vit_out_1d[53])
    non_gfx_frac = max(0.0, min(1.0, float(vit_out_1d[55])))
    photo_syn = float(vit_out_1d[56])
    sp_smooth = float(vit_out_1d[57])
    sp_tex = float(vit_out_1d[58])

    scalars_head = np.asarray(
        [
            vit_syn,
            vit_lse_ai,
            vit_max_lo,
            vit_for_lo,
            dom_syn_clean,
            photo_syn,
            max_proto_sim,
            sp_tex - sp_smooth,
        ],
        dtype=np.float64,
    )
    vit_tail = vit_out_1d[3 + n_cls : VIT_OUT_DIM_V8].astype(np.float64)
    z_sat = 8.0 * np.tanh(z_det_std / 8.0)
    exc_sat = 8.0 * math.tanh(accept_excess / 8.0)
    ch10 = float(z_det_std[n_cls])
    phys_scalars = np.asarray([exc_sat, max(0.0, ch10), min(0.0, ch10)], dtype=np.float64)

    gfx_score = eval_born_digital_graphics_score(x_1d)
    pure_gfx_gate = max(0.0, vit_gfx) * max(0.0, 0.35 - non_gfx_frac)
    social_photo_gate = sp_top48 * non_gfx_frac
    inpaint_gate = sp_top12 * max(0.0, sp_contrast)
    phys_vit_agree = exc_sat * max(-6.0, min(6.0, dom_syn))
    interactions = np.asarray(
        [gfx_score, max(0.0, gfx_score), pure_gfx_gate, social_photo_gate, inpaint_gate, phys_vit_agree],
        dtype=np.float64,
    )
    return np.concatenate([scalars_head, vit_tail, z_sat, phys_scalars, interactions, r_std])


def build_v8_mm_prov_input_1d(
    attr_logits: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_global_1d: FloatArray,
    vit_detail_2d: FloatArray | None = None,
) -> FloatArray:
    """Assemble the 195-d dual-view + ArcFace-prototype open-set provider attribution input vector."""
    import numpy as np

    attr_sat = 8.0 * np.tanh(attr_logits / 8.0)
    z_sat = 8.0 * np.tanh(z_det_std / 8.0)
    exc_sat = np.asarray([8.0 * math.tanh(accept_excess / 8.0)], dtype=np.float64)
    vg = vit_global_1d.astype(np.float64)
    if vit_detail_2d is not None:
        vd = vit_detail_2d.astype(np.float64)
        v_nat_mean = 0.5 * (vd[0] + vd[1])
        d_prov12 = np.maximum(vd[0, 1:13], vd[1, 1:13]) - vg[1:13]
        d_proto10 = np.maximum(vd[0, 39:49], vd[1, 39:49]) - vg[39:49]
        v_nat_delta = np.concatenate([d_prov12, d_proto10])
        has_nat = np.asarray([1.0], dtype=np.float64)
    else:
        v_nat_mean = vg
        v_nat_delta = np.zeros(22, dtype=np.float64)
        has_nat = np.asarray([0.0], dtype=np.float64)
    return np.concatenate([attr_sat, z_sat, exc_sat, vg, v_nat_mean, v_nat_delta, has_nat, r_std])


def _eval_v8_syn_raw(
    x_1d: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_out_1d: FloatArray,
    model: _SourceModelV8,
) -> float:
    return _eval_syn_head_scalar(build_v8_syn_input_1d(x_1d, z_det_std, accept_excess, r_std, vit_out_1d), model)


def _eval_v8_mm_prov_log_odds(
    attr_logits: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_global_1d: FloatArray,
    mm_head: _MMProvHeadV8,
    *,
    vit_detail_2d: FloatArray | None = None,
) -> FloatArray:
    return _eval_mm_prov_head_log_odds(
        build_v8_mm_prov_input_1d(
            attr_logits, z_det_std, accept_excess, r_std, vit_global_1d, vit_detail_2d=vit_detail_2d
        ),
        mm_head,
    )


def _is_v8_crop_pure_gfx(vit_1d: FloatArray, nt_cfg: _NativeTileConfigV8) -> bool:
    n_cls = len(ALL_PROVIDER_CLASSES)
    return float(vit_1d[3 + n_cls]) >= nt_cfg.gfx_ceil and float(vit_1d[55]) < nt_cfg.min_photo_frac


def _should_run_v8_gray_native_tiles(
    x_1d: FloatArray,
    vit_primary_1d: FloatArray,
    eff_phys_excess: float,
    syn_margin_primary: float,
    image_size: tuple[int, int] | None,
    nt_cfg: _NativeTileConfigV8 | None,
) -> bool:
    if nt_cfg is None or nt_cfg.detail_alpha <= 0.0:
        return False
    if eff_phys_excess >= 0.0 or not (nt_cfg.gray_lo <= syn_margin_primary < nt_cfg.gray_hi):
        return False
    if image_size is not None:
        w, h = image_size
        if min(w, h) < nt_cfg.min_short_px or max(w, h) < nt_cfg.min_long_px:
            return False
    if _is_v8_crop_pure_gfx(vit_primary_1d, nt_cfg):
        return False
    photo_frac = float(vit_primary_1d[55])
    return not (eval_born_digital_graphics_score(x_1d) >= 1.50 and photo_frac < nt_cfg.min_photo_frac)


def _check_v8_attr_native_tiles(
    eff_phys_excess: float,
    syn_margin: float,
    attr_logits: FloatArray,
    z_det_std: FloatArray,
    accept_excess: float,
    r_std: FloatArray,
    vit_primary_1d: FloatArray,
    image_size: tuple[int, int] | None,
    model: _SourceModelV8,
) -> tuple[bool, FloatArray | None]:
    import numpy as np

    nt_cfg = model.nt_cfg
    if nt_cfg is None or not nt_cfg.attr_enable or model.mm_prov is None:
        return False, None
    if eff_phys_excess < 0.0 and syn_margin < 0.0:
        return False, None
    if image_size is not None and min(image_size) < nt_cfg.attr_min_short_px:
        return False, None
    if 0.0 <= eff_phys_excess < model.foreign_phys_cap:
        return True, None
    mm_lo_0 = _eval_v8_mm_prov_log_odds(
        attr_logits, z_det_std, accept_excess, r_std, vit_primary_1d, model.mm_prov, vit_detail_2d=None
    )
    if eff_phys_excess < 0.0:
        win_0 = int(np.argmax(mm_lo_0))
        exc_0 = float(mm_lo_0[win_0] - model.mm_prov.prov_thresholds[win_0])
        need = exc_0 < nt_cfg.attr_uncertain_gap or _top2_gap(mm_lo_0) < (
            model.mm_prov.min_gap + nt_cfg.attr_uncertain_gap
        )
        return need, mm_lo_0
    joint_0 = attr_logits + model.vit_attr_alpha * mm_lo_0
    need = _top2_gap(joint_0) < nt_cfg.attr_uncertain_gap or int(np.argmax(attr_logits)) != int(np.argmax(mm_lo_0))
    return need, mm_lo_0


def _eval_v8_mm_stage2_version(
    best_cls: ProviderClass,
    mm_s2: _MMStage2HeadV8,
    x_1d: FloatArray,
    u_vec: FloatArray,
    mean_sub: FloatArray,
    r_std: FloatArray,
    vit_out_1d: FloatArray,
    vit_detail_2d: FloatArray | None,
    model: _SourceModelV8,
) -> str | None:
    import numpy as np

    n_k = len(mm_s2.subfamilies)
    if best_cls in model.base_v5.version_heads:
        raw_hgb = _eval_stage2_version_head_raw(
            model.base_v5.version_heads[best_cls],
            np.concatenate([x_1d, u_vec]),
        )
        if raw_hgb.shape == (1,) and n_k == 2:
            s2_logits = np.asarray([-0.5 * float(raw_hgb[0]), 0.5 * float(raw_hgb[0])], dtype=np.float64)
        elif raw_hgb.shape == (n_k,):
            s2_logits = raw_hgb
        else:
            s2_logits = np.zeros(n_k, dtype=np.float64)
    else:
        s2_logits = np.zeros(n_k, dtype=np.float64)

    s1_subs = np.asarray(
        [float(mean_sub[s_idx]) for s_idx, _ in model.base_v5.backbone.sub_by_provider.get(best_cls, ())],
        dtype=np.float64,
    )
    vit_emb_24 = vit_out_1d[15:39].astype(np.float64)
    vit_nat_emb_24 = (
        (0.5 * (vit_detail_2d[0, 15:39] + vit_detail_2d[1, 15:39])).astype(np.float64)
        if vit_detail_2d is not None
        else vit_emb_24
    )
    z_in = np.concatenate(
        [
            np.clip(s2_logits, -12.0, 12.0),
            np.clip(s1_subs, -12.0, 12.0),
            vit_out_1d[1:13].astype(np.float64),
            vit_out_1d[39:49].astype(np.float64),
            vit_emb_24,
            vit_nat_emb_24,
            r_std,
        ]
    )
    z_std = np.clip((z_in - mm_s2.in_mean) / mm_s2.in_scale, -12.0, 12.0)
    h_s2 = _silu_np(z_std @ mm_s2.w1 + mm_s2.b1)
    mm_logits = z_std @ mm_s2.skip_w + h_s2 @ mm_s2.w2 + mm_s2.b2
    comb = (1.0 - mm_s2.blend_alpha) * s2_logits + mm_s2.blend_alpha * mm_logits
    c_shift = comb - float(np.max(comb))
    exp_c = np.exp(c_shift)
    probs = exp_c / float(np.sum(exp_c))
    if _top2_gap(probs) < mm_s2.min_prob_margin:
        return None
    return mm_s2.subfamilies[int(np.argmax(comb))]


def _classify_v8_with_vit(
    x_1d: FloatArray,
    vit_out_1d: FloatArray,
    model: _SourceModelV8,
    *,
    vit_detail_2d: FloatArray | None = None,
    vit_detail_loader: Callable[[], FloatArray] | None = None,
    image_size: tuple[int, int] | None = None,
    precomputed_v5_state: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float] | None = None,
) -> SourceClassification:
    n_cls = len(ALL_PROVIDER_CLASSES)
    z_det_std, attr_logits, r_std, mean_sub, u_vec, accept_excess = (
        precomputed_v5_state if precomputed_v5_state is not None else _eval_v5_core_state(x_1d, model.base_v5)
    )

    syn_raw = _eval_v8_syn_raw(x_1d, z_det_std, accept_excess, r_std, vit_out_1d, model)
    syn_margin = syn_raw - model.syn_threshold
    eff_phys_excess = _apply_phys_veto(accept_excess, syn_margin, model)

    nt_cfg = model.nt_cfg
    need_gray = nt_cfg is not None and _should_run_v8_gray_native_tiles(
        x_1d, vit_out_1d, eff_phys_excess, syn_margin, image_size, nt_cfg
    )
    cached_mm_lo_1v: FloatArray | None = None
    if vit_detail_2d is None and vit_detail_loader is not None and nt_cfg is not None:
        if need_gray:
            vit_detail_2d = vit_detail_loader()
        else:
            need_attr, cached_mm_lo_1v = _check_v8_attr_native_tiles(
                eff_phys_excess,
                syn_margin,
                attr_logits,
                z_det_std,
                accept_excess,
                r_std,
                vit_out_1d,
                image_size,
                model,
            )
            if need_attr:
                vit_detail_2d = vit_detail_loader()

    if need_gray and vit_detail_2d is not None and nt_cfg is not None:
        valid_d = [
            _eval_v8_syn_raw(x_1d, z_det_std, accept_excess, r_std, vit_detail_2d[k_d], model)
            for k_d in (0, 1)
            if not _is_v8_crop_pure_gfx(vit_detail_2d[k_d], nt_cfg)
        ]
        if valid_d:
            syn_detail_pool = 0.75 * max(valid_d) + 0.25 * min(valid_d) if len(valid_d) == 2 else valid_d[0]
            syn_raw = (1.0 - nt_cfg.detail_alpha) * syn_raw + nt_cfg.detail_alpha * syn_detail_pool
            syn_margin = syn_raw - (model.syn_threshold + nt_cfg.thr_offset)

    if model.mm_prov is not None:
        mm_log_odds = (
            cached_mm_lo_1v
            if (vit_detail_2d is None and cached_mm_lo_1v is not None)
            else _eval_v8_mm_prov_log_odds(
                attr_logits, z_det_std, accept_excess, r_std, vit_out_1d, model.mm_prov, vit_detail_2d=vit_detail_2d
            )
        )
    else:
        open_null_v = _logsumexp_pair(float(vit_out_1d[1 + n_cls]), float(vit_out_1d[2 + n_cls]))
        mm_log_odds = vit_out_1d[1 : 1 + n_cls] - open_null_v

    joint_attr_logits, final_excess, synthetic_detected, synthetic_probability = _fuse_mm_decision(
        attr_logits=attr_logits,
        mm_log_odds=mm_log_odds,
        eff_phys_excess=eff_phys_excess,
        syn_margin=syn_margin,
        vit_attr_alpha=model.vit_attr_alpha,
        syn_conf_scale=model.syn_conf_scale,
        base_conf_scale=model.base_v5.conf_scale,
        mm_prov=model.mm_prov,
        foreign_phys_cap=model.foreign_phys_cap,
        foreign_mm_floor=model.foreign_mm_floor,
    )

    mm_heads = model.mm_version_heads

    def _resolve_v8_version(best_prov: ProviderClass) -> tuple[bool, str | None]:
        if mm_heads is None or best_prov not in mm_heads:
            return False, None
        return True, _eval_v8_mm_stage2_version(
            best_prov, mm_heads[best_prov], x_1d, u_vec, mean_sub, r_std, vit_out_1d, vit_detail_2d, model
        )

    return _build_v5_like_result(
        x_1d=x_1d,
        base_model=model.base_v5,
        z_det_std=z_det_std,
        attr_logits=joint_attr_logits,
        mean_sub=mean_sub,
        u_vec=u_vec,
        accept_excess=final_excess,
        synthetic_detected=synthetic_detected,
        synthetic_probability=synthetic_probability,
        model_revision=MODEL_SCHEMA_V8,
        version_resolver=_resolve_v8_version if mm_heads else None,
    )


def _eval_v8_vit_onnx(image: Image.Image, model: _SourceModelV8) -> FloatArray:
    return _run_vit_onnx(model, preprocess_vit_384_chw(image), (VIT_OUT_DIM_V8,), label="v8 ViT")


def _eval_v8_vit_native_onnx(image: Image.Image, model: _SourceModelV8) -> FloatArray:
    return _run_vit_onnx(
        model,
        preprocess_vit_384_native_2chw(image),
        (2, VIT_OUT_DIM_V8),
        label="v8 ViT native-tile",
    )


def _classify_loaded_model(
    features: FloatArray,
    model: _LoadedSourceModel,
    *,
    vit_features: FloatArray | None = None,
    vit_detail_features: FloatArray | None = None,
    vit_detail_loader: Callable[[], FloatArray] | None = None,
    image_size: tuple[int, int] | None = None,
    precomputed_v5_state: tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, float] | None = None,
) -> SourceClassification:
    import numpy as np

    x_1d = np.asarray(features, dtype=np.float64)
    if x_1d.ndim != 1 or not np.all(np.isfinite(x_1d)):
        return _unavailable_result()

    if isinstance(model, (_SourceModelV4, _SourceModelV5, _SourceModelV7, _SourceModelV8)):
        if x_1d.shape != (FEATURE_DIM_V4,):
            return _unavailable_result()
        if isinstance(model, (_SourceModelV7, _SourceModelV8)):
            is_v8 = isinstance(model, _SourceModelV8)
            vit_dim = VIT_OUT_DIM_V8 if is_v8 else VIT_OUT_DIM_V7
            schema = MODEL_SCHEMA_V8 if is_v8 else MODEL_SCHEMA_V7
            if vit_features is None:
                return _classify_v5_features(x_1d, model.base_v5, model_revision=schema)
            v_1d = np.asarray(vit_features, dtype=np.float64)
            if v_1d.shape != (vit_dim,) or not np.all(np.isfinite(v_1d)):
                return _unavailable_result()
            v_det: FloatArray | None = None
            if vit_detail_features is not None:
                v_det = np.asarray(vit_detail_features, dtype=np.float64)
                if v_det.shape != (2, vit_dim) or not np.all(np.isfinite(v_det)):
                    return _unavailable_result()
            if isinstance(model, _SourceModelV8):
                return _classify_v8_with_vit(
                    x_1d,
                    v_1d,
                    model,
                    vit_detail_2d=v_det,
                    vit_detail_loader=vit_detail_loader,
                    image_size=image_size,
                    precomputed_v5_state=precomputed_v5_state,
                )
            return _classify_v7_with_vit(
                x_1d,
                v_1d,
                model,
                vit_detail_2d=v_det,
                vit_detail_loader=vit_detail_loader,
                image_size=image_size,
                precomputed_v5_state=precomputed_v5_state,
            )
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
        synthetic_detected=label != "unknown",
        synthetic_probability=confidence,
    )


def classify_source_features(
    features: FloatArray,
    *,
    weights_path: Path | None = None,
    vit_features: FloatArray | None = None,
    vit_detail_features: FloatArray | None = None,
) -> SourceClassification:
    """Classify a precomputed 956-d (v3) or 512-d (v4-v8) feature vector into 10 providers or unknown."""
    import numpy as np

    x_1d = np.asarray(features, dtype=np.float64)
    if x_1d.ndim != 1 or x_1d.shape[0] not in (FEATURE_DIM, FEATURE_DIM_V4) or not np.all(np.isfinite(x_1d)):
        return _unavailable_result()

    return _classify_loaded_model(
        x_1d,
        _load_model(weights_path),
        vit_features=vit_features,
        vit_detail_features=vit_detail_features,
    )


def classify_source(
    source: str | os.PathLike[str] | Image.Image,
    *,
    weights_path: Path | None = None,
) -> SourceClassification:
    """Classify an image into 10 AI provider pipelines or ``unknown`` plus synthetic detection.

    Parameters
    ----------
    source
        Path to an image file or an already-opened PIL Image.
    weights_path
        Optional local ``.npz`` override. Defaults to ``RAIW_SOURCE_CLASSIFY_WEIGHTS``
        or the pinned Hugging Face Hub artifact verified by SHA-256.
    """
    _require_dependencies(need_hub=False, need_onnx=False)
    from PIL import Image as PilImage

    from remove_ai_watermarks import image_io

    image_io._register_heif()  # pyright: ignore[reportPrivateUsage]

    def _run_on_image(img: Image.Image) -> SourceClassification:
        if min(img.size) < MIN_SIDE:
            return _unavailable_result()
        loaded_model = _load_model(weights_path)
        if isinstance(loaded_model, (_SourceModelV7, _SourceModelV8)):
            _ = _get_v7_onnx_session(loaded_model)
        extracted = (
            extract_512d_v4_features(img)
            if isinstance(loaded_model, (_SourceModelV4, _SourceModelV5, _SourceModelV7, _SourceModelV8))
            else extract_956d_features(img)
        )
        if extracted is None:
            return _unavailable_result()
        if not isinstance(loaded_model, (_SourceModelV7, _SourceModelV8)):
            return _classify_loaded_model(extracted, loaded_model)

        rgb_img = _orient_rgb_image(img)
        if isinstance(loaded_model, _SourceModelV8):
            vit_vec = _eval_v8_vit_onnx(rgb_img, loaded_model)
            return _classify_loaded_model(
                extracted,
                loaded_model,
                vit_features=vit_vec,
                vit_detail_loader=(
                    (lambda: _eval_v8_vit_native_onnx(rgb_img, loaded_model))
                    if loaded_model.nt_cfg is not None
                    else None
                ),
                image_size=rgb_img.size,
            )

        vit_vec = _eval_v7_vit_onnx(rgb_img, loaded_model)
        ms_cfg = loaded_model.ms_cfg
        return _classify_loaded_model(
            extracted,
            loaded_model,
            vit_features=vit_vec,
            vit_detail_loader=(
                (lambda: _eval_v7_vit_detail_onnx(rgb_img, loaded_model, short_edge=ms_cfg.short_edge))
                if ms_cfg is not None
                else None
            ),
            image_size=rgb_img.size,
        )

    if isinstance(source, (str, os.PathLike)):
        with PilImage.open(source) as opened:
            return _run_on_image(opened)
    return _run_on_image(source)
