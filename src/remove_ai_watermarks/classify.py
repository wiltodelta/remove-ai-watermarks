# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Metadata-free photo classifier: Model 1 then gated Model 2.

This is not provenance and is not a cleanup command. ``identify``,
``has_invisible_target``, and ``all`` must not import this module.
Call :func:`classify_pixels` explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

CLASSIFY_EXTRA = "'remove-ai-watermarks[classify]'"
WEIGHTS_ENV = "RAIW_CLASSIFY_WEIGHTS"
WEIGHTS_REPO = "wiltodelta/raiw-photo-classify"
WEIGHTS_REVISION = "ffc46db5135ee3a83f51538f2c8f7483b9b8b40c"
CLIP_FILE = "clip-l-ft.pt"
PROBE_FILE = "probe-weights-clip-l-ft.npz"
DETECTOR_FILE = "detector.pt"
PROVIDER_FILE = "provider.pt"
OPERATING_POINT_FILE = "operating-point.json"
_WEIGHT_FILES = (CLIP_FILE, PROBE_FILE, DETECTOR_FILE, PROVIDER_FILE)

# 2026-09-02 receipt-document gate: linear head on the same CLIP-L-ft
# vector, trained on CORD-v2 train (800, CC BY 4.0) plus 200 synthetic
# capture-style receipts against 2,400 ai_train negatives. Threshold is the
# min of the 1st-percentile scores on CORD validation and a synthetic
# holdout; no field-receipt pixels were used for training or the threshold.
# The head versions with the model (Hub snapshot or RAIW_CLASSIFY_WEIGHTS
# directory); the package asset is the fallback for pre-gate freezes.
# 2026-09-07: the gate is MODEL-SIDE. The weights directory carries the
# artifact under the STABLE name ``receipt-gate.npz`` with the threshold
# inside; the dated name below stays readable as the legacy spelling, and
# the bundled package asset is the last-resort fallback. Head and
# threshold always come from ONE artifact, so a model-side gate update
# needs no lib release; ``RECEIPT_GATE_THRESHOLD`` is only the legacy
# pinned value kept for the package-asset fallback and its pinned tests.
RECEIPT_GATE_STABLE_FILE = "receipt-gate.npz"
RECEIPT_GATE_FILE = "receipt-gate-2026-09-02.npz"
RECEIPT_GATE_THRESHOLD = 2.0432573877459697

# The complete file set the runtime's snapshot_download may request. Deploy-time
# pre-caches in offline environments (HF_HUB_OFFLINE=1) must install exactly
# this list: a subset leaves the runtime resolution unsatisfiable and every
# classify call fails on weights that are four-fifths present. Exported so a
# pre-cache imports one name instead of re-listing the files and silently
# drifting the next time a release adds one.
WEIGHTS_ALLOW_PATTERNS: tuple[str, ...] = (
    *_WEIGHT_FILES,
    OPERATING_POINT_FILE,
    RECEIPT_GATE_STABLE_FILE,
    RECEIPT_GATE_FILE,
)

# 2026-08-31 freeze operating point. Ridge threshold is also in the probe file;
# the runtime prefers the file when weights load.
MLP_THRESHOLD = 5.9586493237495395
RIDGE_THRESHOLD = 0.3056212276800537
PROVIDER_MARGIN = 0.30
CLIP_WIDTH = 768
# Checkpoint keys in provider.pt, the SUPERSET across snapshot generations.
# openai/google are provider classes; meta_muse_image is Muse Image;
# bytedance is the shared ByteDance generator lineage (Doubao and Jimeng
# render with the same model family -- measured 2026-09-07: separate heads
# cross-fire 32/22 ways even at 3.5x train mass, the union holds 83.9%
# on the frozen test cell). tc260 is NOT one class of anything: it covers
# the REST of China's generator ecosystem (Qwen, Yuanbao, Kling, ...,
# producers that are peers of openai/google/meta), whose heads do not
# exist yet at honest mass, so its argmax win abstains (provider=None).
# The loader tolerates older snapshots that predate bytedance: a head
# absent from provider.pt is skipped, and those rows fall to the veto.
# Measured on the shipped weights: deleting the veto from the argmax
# would falsely name 207/379 China test rows openai/google/muse-image
# (153 of them muse-image); the veto leaves every other cell
# byte-identical (openai 345/380, google 339/373, meta v3 177/198).
PROVIDER_LABELS = ("openai", "google", "bytedance", "tc260", "meta_muse_image", "no_ai")

DetectorLevel = Literal["definitely", "possibly", "likely_human"]
PixelLabel = Literal["ai", "human", "unknown"]
PixelDomain = Literal["photo"]
PixelProvider = Literal["openai", "google", "muse-image", "bytedance"]
PUBLIC_PROVIDER: dict[str, PixelProvider | None] = {
    "openai": "openai",
    "google": "google",
    "bytedance": "bytedance",
    "tc260": None,
    "meta_muse_image": "muse-image",
}

_runtime: _Runtime | None = None
_runtime_lock = threading.Lock()
_receipt_gate_cache: dict[str, dict[str, Any]] = {}


def _load_receipt_gate(folder: Path | None = None) -> dict[str, Any]:
    """Load the receipt-gate head, preferring the weights directory.

    Model-side first: the stable ``receipt-gate.npz`` in the Hub snapshot
    or ``RAIW_CLASSIFY_WEIGHTS`` directory wins, then the legacy dated
    spelling, with the bundled package asset as the fallback for weights
    directories frozen before the gate existed. The head is fitted on the
    freeze CLIP-L-ft embedding space, which is why it versions with the
    model, not with the code; the threshold travels inside the artifact.
    """
    import numpy as np

    key = str(folder) if folder is not None else "<package>"
    if key in _receipt_gate_cache:
        return _receipt_gate_cache[key]
    source: Path | None = None
    if folder is not None:
        source = _find_weight(folder, RECEIPT_GATE_STABLE_FILE) or _find_weight(folder, RECEIPT_GATE_FILE)
    if source is None:
        source = Path(__file__).parent / "assets" / RECEIPT_GATE_FILE
    payload = np.load(source)
    gate = {
        "w": np.asarray(payload["w"], dtype=np.float64),
        "b": float(payload["b"]),
        "mu": np.asarray(payload["mu"], dtype=np.float64),
        "sd": np.asarray(payload["sd"], dtype=np.float64),
        "threshold": float(payload["threshold"]),
    }
    _receipt_gate_cache[key] = gate
    return gate


def receipt_gate_score(clip_vector: Any, gate: dict[str, Any] | None = None) -> float:
    """Receipt-document score of a CLIP-L-ft vector; higher is more receipt-like.

    ``gate`` must be the SAME artifact whose threshold decides the hit
    (``classify_pixels`` passes the weights-directory gate, so head and
    threshold can never come from different files); the default loads the
    package-fallback artifact for standalone probing.
    """
    import numpy as np

    g = _load_receipt_gate() if gate is None else gate
    v = np.asarray(clip_vector, dtype=np.float64).ravel()
    v = v / (np.linalg.norm(v) + 1e-12)
    return float((((v - g["mu"]) / g["sd"]) @ g["w"]) + g["b"])


@dataclass(frozen=True)
class PixelClassification:
    """Combined Model 1 and Model 2 result for one image."""

    label: PixelLabel
    domain: PixelDomain
    detector: DetectorLevel
    provider: PixelProvider | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "domain": self.domain,
            "detector": self.detector,
            "provider": self.provider,
        }


def is_available() -> bool:
    """True when the ``classify`` extra can import torch, transformers, numpy, and cv2."""
    from remove_ai_watermarks.optional_deps import module_available

    return module_available("torch", "transformers", "numpy", "cv2")


def detector_level(
    ridge_score: float,
    mlp_score: float,
    *,
    ridge_threshold: float = RIDGE_THRESHOLD,
    mlp_threshold: float = MLP_THRESHOLD,
) -> DetectorLevel:
    """Map the freeze AND/OR rule onto definitely / possibly / likely_human."""
    ridge = ridge_score > ridge_threshold
    mlp = mlp_score > mlp_threshold
    if ridge and mlp:
        return "definitely"
    if ridge or mlp:
        return "possibly"
    return "likely_human"


def label_for(level: DetectorLevel) -> PixelLabel:
    """Public label: only DEFINITELY is ``ai``. POSSIBLY abstains."""
    if level == "definitely":
        return "ai"
    if level == "likely_human":
        return "human"
    return "unknown"


def provider_from_scores(scores: dict[str, float], *, margin: float = PROVIDER_MARGIN) -> PixelProvider | None:
    """Argmax among the named classes that beat ``no_ai`` by ``margin``.

    Candidates are the heads the LOADED snapshot actually carries, so
    older provider.pt files without a generation's newest class keep
    working (their rows fall to the tc260 veto and abstain). The tc260
    head joins the argmax only as a group veto: it covers the rest of
    China's generator ecosystem, producers that are peers of
    openai/google/meta, so until per-producer classes exist its win
    abstains and ``None`` is published instead of a false company claim.
    """
    no_ai = scores["no_ai"]
    ai_names = [name for name in scores if name != "no_ai"]
    passed = [name for name in ai_names if scores[name] > no_ai + margin]
    if not passed:
        return None
    best = max(passed, key=lambda name: scores[name])
    return PUBLIC_PROVIDER[best]


def classify_from_scores(
    ridge_score: float,
    mlp_score: float,
    provider_scores: dict[str, float] | None,
    *,
    ridge_threshold: float = RIDGE_THRESHOLD,
    mlp_threshold: float = MLP_THRESHOLD,
    provider_margin: float = PROVIDER_MARGIN,
    receipt_score: float | None = None,
    receipt_threshold: float | None = None,
) -> PixelClassification:
    """Pure gate: detector AND provider, no file I/O.

    A DEFINITELY verdict whose ``receipt_score`` meets ``receipt_threshold``
    downgrades the public label to ``unknown`` (document-domain abstain):
    the raw detector level stays ``definitely`` and no provider is read.
    ``receipt_threshold=None`` (the default) resolves from the loaded gate
    artifact, so the operating point versions with the model.
    """
    if receipt_threshold is None:
        receipt_threshold = float(_load_receipt_gate()["threshold"])
    level = detector_level(
        ridge_score,
        mlp_score,
        ridge_threshold=ridge_threshold,
        mlp_threshold=mlp_threshold,
    )
    label = label_for(level)
    if level == "definitely" and receipt_score is not None and receipt_score >= receipt_threshold:
        return PixelClassification(label="unknown", domain="photo", detector=level, provider=None)
    provider: PixelProvider | None = None
    if label == "ai" and provider_scores is not None:
        provider = provider_from_scores(provider_scores, margin=provider_margin)
    return PixelClassification(label=label, domain="photo", detector=level, provider=provider)


def classify_pixels(path: Path, *, device: str | None = None) -> PixelClassification:
    """Run Model 1 then, on DEFINITELY, the receipt gate and Model 2.

    Never called by ``identify``. A DEFINITELY verdict that the receipt gate
    accepts is published as ``unknown`` before the 124-d provider pass.
    """
    if not is_available():
        raise RuntimeError(f"pixel classification requires the classify extra. Install: pip install {CLASSIFY_EXTRA}")
    runtime = _get_runtime(device)
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        clip_vector = _embed(runtime, rgb)
        ridge = float(((clip_vector - runtime.ridge_mean) / runtime.ridge_scale) @ runtime.ridge_weights)
        mlp = _mlp_score(runtime, clip_vector)
        level = detector_level(
            ridge,
            mlp,
            ridge_threshold=runtime.ridge_threshold,
            mlp_threshold=runtime.mlp_threshold,
        )
        gate = _load_receipt_gate(runtime.weights_dir)
        r_score = receipt_gate_score(clip_vector, gate) if level == "definitely" else None
        provider_scores = None
        if r_score is not None and r_score < gate["threshold"]:
            forensic = _forensic(rgb)
            if forensic is not None:
                provider_scores = _provider_scores(runtime, forensic)
            else:
                log.info("124-d features unavailable for %s, provider abstains", path)
    result = classify_from_scores(
        ridge,
        mlp,
        provider_scores,
        ridge_threshold=runtime.ridge_threshold,
        mlp_threshold=runtime.mlp_threshold,
        provider_margin=runtime.provider_margin,
        receipt_score=r_score,
        receipt_threshold=gate["threshold"],
    )
    if result.label == "unknown" and r_score is not None:
        log.info("receipt-document gate abstained on %s (score %.2f)", path, r_score)
    return result


@dataclass
class _Runtime:
    device: Any
    clip_model: Any
    processor: Any
    detector: Any
    provider_heads: dict[str, Any]
    ridge_mean: Any
    ridge_scale: Any
    ridge_weights: Any
    ridge_threshold: float
    mlp_threshold: float
    provider_margin: float
    weights_dir: Path | None


def _resolve_device(device: str | None) -> Any:
    import torch

    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("cuda was requested but is not available")
        return torch.device("cuda")
    raise ValueError(f"unsupported classify device {device!r}")


def _weights_dir() -> Path:
    override = os.environ.get(WEIGHTS_ENV)
    if override:
        path = Path(override)
        if not path.is_dir():
            raise RuntimeError(f"{WEIGHTS_ENV} is not a directory: {path}")
        return path
    from huggingface_hub import snapshot_download

    try:
        return Path(
            snapshot_download(
                repo_id=WEIGHTS_REPO,
                revision=WEIGHTS_REVISION,
                allow_patterns=list(WEIGHTS_ALLOW_PATTERNS),
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "pixel classification weights are not installed. Set "
            f"{WEIGHTS_ENV} to a directory with {', '.join(_WEIGHT_FILES)}, "
            f"or install {CLASSIFY_EXTRA} and download {WEIGHTS_REPO}@{WEIGHTS_REVISION}."
        ) from exc


def _find_weight(folder: Path, name: str) -> Path | None:
    direct = folder / name
    nested = folder / "run1" / name
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    return None


def _weight_path(folder: Path, name: str) -> Path:
    found = _find_weight(folder, name)
    if found is None:
        raise RuntimeError(f"pixel classification weights are missing ({name}) in {folder}")
    return found


def _require_files(folder: Path) -> None:
    missing = [name for name in _WEIGHT_FILES if _find_weight(folder, name) is None]
    if missing:
        raise RuntimeError(
            "pixel classification weights are missing "
            f"({', '.join(missing)}). Set {WEIGHTS_ENV} to a directory with the "
            f"2026-08-31 freeze files, or install {CLASSIFY_EXTRA} and allow the "
            f"Hugging Face download from {WEIGHTS_REPO}."
        )


def _operating_point(folder: Path) -> tuple[float, float]:
    path = folder / OPERATING_POINT_FILE
    if not path.is_file():
        return MLP_THRESHOLD, PROVIDER_MARGIN
    payload = json.loads(path.read_text())
    return float(payload["model1"]["mlp_threshold"]), float(payload["model2"]["margin"])


def _build_detector() -> Any:
    from torch import nn

    return nn.Sequential(
        nn.Linear(CLIP_WIDTH, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 1),
    )


def _build_provider_head() -> Any:
    from torch import nn

    return nn.Sequential(
        nn.Linear(124, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    )


def _load_runtime(device: Any) -> _Runtime:
    import numpy as np
    import torch

    from remove_ai_watermarks._internal.clip_l_ft import load_headed_clip, load_processor

    folder = _weights_dir()
    _require_files(folder)
    probe = np.load(_weight_path(folder, PROBE_FILE))
    detector = _build_detector()
    detector.load_state_dict(torch.load(_weight_path(folder, DETECTOR_FILE), map_location="cpu", weights_only=True))
    detector.to(device).eval()
    packed = torch.load(_weight_path(folder, PROVIDER_FILE), map_location="cpu", weights_only=True)
    heads = {}
    for name in PROVIDER_LABELS:
        if name not in packed:
            log.info("provider head %s absent from this snapshot; rows it would name abstain", name)
            continue
        head = _build_provider_head()
        head.load_state_dict(packed[name])
        heads[name] = head.to(device).eval()
    ridge_threshold = float(probe["thr_oi_1pct"])
    mlp_threshold, provider_margin = _operating_point(folder)
    log.info("loaded photo-classify freeze from %s", folder)
    return _Runtime(
        device=device,
        clip_model=load_headed_clip(_weight_path(folder, CLIP_FILE), device),
        processor=load_processor(),
        detector=detector,
        provider_heads=heads,
        ridge_mean=np.asarray(probe["mean"], dtype=np.float64),
        ridge_scale=np.asarray(probe["scale"], dtype=np.float64),
        ridge_weights=np.asarray(probe["weights"], dtype=np.float64),
        ridge_threshold=ridge_threshold,
        mlp_threshold=mlp_threshold,
        provider_margin=provider_margin,
        weights_dir=folder,
    )


def _get_runtime(device: str | None) -> _Runtime:
    global _runtime
    resolved = _resolve_device(device)
    with _runtime_lock:
        if _runtime is None or _runtime.device != resolved:
            _runtime = _load_runtime(resolved)
        return _runtime


def _embed(runtime: _Runtime, image: Any) -> Any:
    from remove_ai_watermarks._internal.clip_l_ft import embed_image

    return embed_image(runtime.clip_model, runtime.processor, image, runtime.device)


def _forensic(image: Any) -> Any:
    import numpy as np

    from remove_ai_watermarks._internal.forensic_124d import image_features

    pixels = np.asarray(image, dtype=np.uint8)
    return image_features(pixels)


def _mlp_score(runtime: _Runtime, clip_vector: Any) -> float:
    import torch

    tensor = torch.from_numpy(clip_vector.astype("float32")).unsqueeze(0).to(runtime.device)
    with torch.inference_mode():
        return float(runtime.detector(tensor).squeeze().cpu())


def _provider_scores(runtime: _Runtime, forensic: Any) -> dict[str, float]:
    import torch

    tensor = torch.from_numpy(forensic.astype("float32")).unsqueeze(0).to(runtime.device)
    with torch.inference_mode():
        return {name: float(head(tensor).squeeze().cpu()) for name, head in runtime.provider_heads.items()}
