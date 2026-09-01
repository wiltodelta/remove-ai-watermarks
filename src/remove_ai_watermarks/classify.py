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
WEIGHTS_REPO = "wiltodelta/raiw-models"
WEIGHTS_REVISION = "c0ac82b6f1ae9fc0b92c467562282e9422da6da6"
CLIP_FILE = "clip-l-ft.pt"
PROBE_FILE = "probe-weights-clip-l-ft.npz"
DETECTOR_FILE = "detector.pt"
PROVIDER_FILE = "provider.pt"
OPERATING_POINT_FILE = "operating-point.json"
_WEIGHT_FILES = (CLIP_FILE, PROBE_FILE, DETECTOR_FILE, PROVIDER_FILE)

# 2026-08-31 freeze operating point. Ridge threshold is also in the probe file;
# the runtime prefers the file when weights load.
MLP_THRESHOLD = 5.9586493237495395
RIDGE_THRESHOLD = 0.3056212276800537
PROVIDER_MARGIN = 0.30
CLIP_WIDTH = 768
# Checkpoint keys in provider.pt. openai/google/tc260 are provider classes;
# meta_muse_image is Muse Image output. Public names follow that split.
PROVIDER_LABELS = ("openai", "google", "tc260", "meta_muse_image", "no_ai")

DetectorLevel = Literal["definitely", "possibly", "likely_human"]
PixelLabel = Literal["ai", "human", "unknown"]
PixelDomain = Literal["photo"]
PixelProvider = Literal["openai", "google", "muse-image", "tc260"]
PUBLIC_PROVIDER: dict[str, PixelProvider] = {
    "openai": "openai",
    "google": "google",
    "tc260": "tc260",
    "meta_muse_image": "muse-image",
}

_runtime: _Runtime | None = None
_runtime_lock = threading.Lock()


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
    """Argmax among openai/google/tc260/muse-image that beat ``no_ai`` by ``margin``."""
    no_ai = scores["no_ai"]
    ai_names = [name for name in PROVIDER_LABELS if name != "no_ai"]
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
) -> PixelClassification:
    """Pure gate: detector AND provider, no file I/O."""
    level = detector_level(
        ridge_score,
        mlp_score,
        ridge_threshold=ridge_threshold,
        mlp_threshold=mlp_threshold,
    )
    label = label_for(level)
    provider: PixelProvider | None = None
    if label == "ai" and provider_scores is not None:
        provider = provider_from_scores(provider_scores, margin=provider_margin)
    return PixelClassification(label=label, domain="photo", detector=level, provider=provider)


def classify_pixels(path: Path, *, device: str | None = None) -> PixelClassification:
    """Run Model 1 then, on DEFINITELY, Model 2. Never called by ``identify``."""
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
        provider_scores = None
        if level == "definitely":
            forensic = _forensic(rgb)
            if forensic is not None:
                provider_scores = _provider_scores(runtime, forensic)
            else:
                log.info("124-d features unavailable for %s, provider abstains", path)
    return classify_from_scores(
        ridge,
        mlp,
        provider_scores,
        ridge_threshold=runtime.ridge_threshold,
        mlp_threshold=runtime.mlp_threshold,
        provider_margin=runtime.provider_margin,
    )


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
                allow_patterns=[*_WEIGHT_FILES, OPERATING_POINT_FILE],
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
