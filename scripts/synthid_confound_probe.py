"""Measure how well non-watermark confounds predict provider SynthID labels.

The experiment deliberately trains three provider-specific baselines:

``container``
    File size, decoded geometry, aspect ratio, and format.
``thumbnail``
    Container features plus a small RGB thumbnail that can learn generator and
    content style.
``canonical``
    Decoded, orientation-normalized RGB at a fixed geometry, with no container,
    original-resolution, metadata, filename, or path features.

These are challenge baselines, not SynthID detectors. A candidate detector must
beat the canonical baseline on same-provider hard negatives and a temporal
holdout before its result can be attributed to a watermark-specific signal.

Usage:
    uv run --extra pixels python scripts/synthid_confound_probe.py \
        .local-eval/synthid/manifest.csv --target-provider google \
        --report-out .local-eval/synthid/google-d1-confounds.json
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
from PIL import Image, ImageOps
from synthid_research_manifest import artifact_sha256, audit_manifest, resolve_artifact_path

log = logging.getLogger(__name__)

FEATURE_FAMILIES = ("container", "thumbnail", "canonical")
FINAL_SPLITS = ("train", "validation", "test", "temporal")
THUMBNAIL_SIZE = 8
FORMAT_NAMES = ("png", "jpeg", "webp")
LOGISTIC_ITERATIONS = 2_000
LOGISTIC_LEARNING_RATE = 0.2
LOGISTIC_L2 = 0.01


@dataclass(frozen=True)
class Example:
    """One ordinary, provider-targeted manifest row eligible for D1."""

    artifact_path: Path
    artifact_sha256: str
    group_id: str
    split: str
    label: int
    negative_cohort: str | None


@dataclass(frozen=True)
class LogisticModel:
    """Standardization and regularized logistic-regression parameters."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float


@dataclass(frozen=True)
class Metrics:
    """Binary metrics at one validation-frozen threshold."""

    count: int
    positives: int
    negatives: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    tpr: float | None
    fpr: float | None
    auc: float | None


def _safe_artifact_path(root: Path, value: str) -> Path:
    """Resolve a manifest-relative path without accepting traversal."""
    candidate = resolve_artifact_path(root, value)
    if candidate is None:
        raise ValueError(f"unsafe artifact_path {value!r}")
    return candidate


def _read_manifest(path: Path) -> list[dict[str, str]]:
    """Read manifest rows after the caller has run the canonical auditor."""
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_examples(manifest: Path, target_provider: str) -> list[Example]:
    """Load ordinary final-label examples for one provider target.

    Candidate, sham, and source-control rows are deliberately excluded. A
    remover-generated negative must not certify the detector that created it,
    and repeated controls must not receive extra sample weight.
    """
    root = manifest.parent
    examples: list[Example] = []
    for row in _read_manifest(manifest):
        if row.get("target_provider") != target_provider:
            continue
        if row.get("split") not in FINAL_SPLITS or row.get("oracle_role") != "ordinary":
            continue
        outcome = row.get("synthid_outcome")
        if outcome not in {"detected", "not_detected"}:
            continue
        source_provider = row.get("source_provider", "")
        if outcome == "detected" and source_provider != target_provider:
            raise ValueError(
                f"positive group {row.get('group_id')!r} targets {target_provider!r} "
                f"but declares source_provider {source_provider!r}"
            )
        negative_cohort: str | None = None
        if outcome == "not_detected":
            if source_provider == target_provider:
                negative_cohort = "same_provider"
            elif source_provider in {"openai", "google", "other_ai"}:
                negative_cohort = "other_ai"
            else:
                negative_cohort = "external"
        examples.append(
            Example(
                artifact_path=_safe_artifact_path(root, row.get("artifact_path", "")),
                artifact_sha256=row.get("artifact_sha256", ""),
                group_id=row.get("group_id", ""),
                split=row.get("split", ""),
                label=1 if outcome == "detected" else 0,
                negative_cohort=negative_cohort,
            )
        )
    if not examples:
        raise ValueError(f"manifest has no eligible ordinary rows for target_provider={target_provider!r}")
    return examples


def _decoded_rgb(path: Path) -> Image.Image:
    """Return rendered RGB pixels with EXIF orientation applied once."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _thumbnail_features(image: Image.Image) -> np.ndarray:
    """Return a fixed-size RGB content fingerprint with no source geometry."""
    thumbnail = image.resize((THUMBNAIL_SIZE, THUMBNAIL_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(thumbnail, dtype=np.float64).reshape(-1) / 255.0


def extract_feature_families(example: Example) -> dict[str, np.ndarray]:
    """Extract every confounded feature family with one image decode."""
    with _decoded_rgb(example.artifact_path) as image:
        width, height = image.size
        thumbnail = _thumbnail_features(image)
    file_size = example.artifact_path.stat().st_size
    pixels = width * height
    image_format = example.artifact_path.suffix.lower().lstrip(".")
    if image_format == "jpg":
        image_format = "jpeg"
    container = np.asarray(
        [
            np.log1p(width),
            np.log1p(height),
            np.log1p(pixels),
            np.log1p(file_size),
            width / height,
            file_size / pixels,
            *(1.0 if image_format == name else 0.0 for name in FORMAT_NAMES),
        ],
        dtype=np.float64,
    )
    return {
        "container": container,
        "thumbnail": np.concatenate((container, thumbnail)),
        "canonical": thumbnail,
    }


def extract_features(example: Example, family: str) -> np.ndarray:
    """Extract one deliberately confounded feature family."""
    if family not in FEATURE_FAMILIES:
        raise ValueError(f"unsupported feature family {family!r}")
    return extract_feature_families(example)[family]


def feature_matrix(examples: list[Example], family: str) -> np.ndarray:
    """Extract a dense matrix in manifest order."""
    return np.stack([extract_features(example, family) for example in examples])


def feature_matrices(examples: list[Example]) -> dict[str, np.ndarray]:
    """Extract all dense feature matrices while decoding each artifact once."""
    rows = [extract_feature_families(example) for example in examples]
    return {family: np.stack([row[family] for row in rows]) for family in FEATURE_FAMILIES}


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Give each class equal total weight regardless of corpus imbalance."""
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("training requires at least one positive and one negative")
    return np.where(labels == 1, 0.5 / positives, 0.5 / negatives)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Evaluate a numerically stable logistic sigmoid."""
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = LOGISTIC_ITERATIONS,
    learning_rate: float = LOGISTIC_LEARNING_RATE,
    l2: float = LOGISTIC_L2,
) -> LogisticModel:
    """Fit deterministic class-balanced L2 logistic regression."""
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("feature and label shapes are inconsistent")
    if iterations < 1 or learning_rate <= 0.0 or l2 < 0.0:
        raise ValueError("iterations and learning_rate must be positive; l2 must be nonnegative")
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    sample_weights = _balanced_sample_weights(labels)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(iterations):
        probabilities = _sigmoid(standardized @ weights + bias)
        error = (probabilities - labels) * sample_weights
        gradient = standardized.T @ error + l2 * weights
        weights -= learning_rate * gradient
        bias -= learning_rate * float(np.sum(error))
    return LogisticModel(mean=mean, scale=scale, weights=weights, bias=bias)


def predict_scores(model: LogisticModel, features: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities for FEATURES."""
    standardized = (features - model.mean) / model.scale
    return _sigmoid(standardized @ model.weights + model.bias)


def select_threshold(labels: np.ndarray, scores: np.ndarray, *, max_fpr: float) -> float:
    """Choose the validation threshold with maximum TPR under MAX_FPR."""
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("validation labels and scores must be one-dimensional and aligned")
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be between zero and one")
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("threshold selection requires positive and negative validation examples")
    candidates = [float(np.nextafter(np.max(scores), np.inf)), *sorted(set(map(float, scores)), reverse=True)]
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        predicted = scores >= threshold
        tpr = float(np.mean(predicted[labels == 1]))
        fpr = float(np.mean(predicted[labels == 0]))
        if fpr > max_fpr:
            continue
        candidate = (tpr, -fpr, threshold)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("threshold search found no feasible operating point")
    return best[2]


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Return tie-aware ROC AUC, or None when one class is absent."""
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Metrics:
    """Calculate confusion counts, rates, and AUC at THRESHOLD."""
    predicted = scores >= threshold
    positives = labels == 1
    negatives = ~positives
    true_positives = int(np.sum(predicted & positives))
    false_positives = int(np.sum(predicted & negatives))
    true_negatives = int(np.sum(~predicted & negatives))
    false_negatives = int(np.sum(~predicted & positives))
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    return Metrics(
        count=len(labels),
        positives=positive_count,
        negatives=negative_count,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        tpr=true_positives / positive_count if positive_count else None,
        fpr=false_positives / negative_count if negative_count else None,
        auc=_auc(labels, scores),
    )


def _split_indices(examples: list[Example], split: str) -> np.ndarray:
    """Return integer indices for SPLIT."""
    return np.asarray([index for index, example in enumerate(examples) if example.split == split], dtype=np.int64)


def _cohort_metrics(
    examples: list[Example], scores: np.ndarray, threshold: float, split: str
) -> dict[str, dict[str, int | float | None]]:
    """Report negative-only false-positive rates by provenance cohort."""
    result: dict[str, dict[str, int | float | None]] = {}
    for cohort in ("same_provider", "other_ai", "external"):
        indices = np.asarray(
            [
                index
                for index, example in enumerate(examples)
                if example.split == split and example.negative_cohort == cohort
            ],
            dtype=np.int64,
        )
        if not len(indices):
            result[cohort] = {"count": 0, "false_positives": 0, "fpr": None}
            continue
        cohort_scores = scores[indices]
        false_positives = int(np.sum(cohort_scores >= threshold))
        result[cohort] = {
            "count": len(indices),
            "false_positives": false_positives,
            "fpr": false_positives / len(indices),
        }
    return result


def _require_split_classes(examples: list[Example], split: str) -> None:
    """Require both labels in the train, validation, and locked test splits."""
    labels = {example.label for example in examples if example.split == split}
    if labels != {0, 1}:
        raise ValueError(f"split {split!r} must contain at least one ordinary positive and negative")


def run_experiment(
    manifest: Path,
    target_provider: str,
    *,
    max_fpr: float = 0.001,
    verify_files: bool = True,
) -> dict[str, object]:
    """Audit MANIFEST, train all confound baselines, and return a JSON-safe report."""
    errors = audit_manifest(manifest, verify_files=verify_files)
    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"manifest audit failed with {len(errors)} error(s): {preview}")
    examples = load_examples(manifest, target_provider)
    for split in ("train", "validation", "test"):
        _require_split_classes(examples, split)

    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    split_counts = Counter(example.split for example in examples)
    negative_counts = Counter(example.negative_cohort for example in examples if example.negative_cohort is not None)
    train_indices = _split_indices(examples, "train")
    validation_indices = _split_indices(examples, "validation")
    report_families: dict[str, object] = {}
    matrices = feature_matrices(examples)

    for family in FEATURE_FAMILIES:
        features = matrices[family]
        model = fit_logistic(features[train_indices], labels[train_indices])
        scores = predict_scores(model, features)
        threshold = select_threshold(labels[validation_indices], scores[validation_indices], max_fpr=max_fpr)
        split_metrics: dict[str, object] = {}
        cohort_metrics: dict[str, object] = {}
        for split in FINAL_SPLITS:
            indices = _split_indices(examples, split)
            if not len(indices):
                split_metrics[split] = None
                cohort_metrics[split] = {
                    cohort: {"count": 0, "false_positives": 0, "fpr": None}
                    for cohort in ("same_provider", "other_ai", "external")
                }
                continue
            split_metrics[split] = asdict(calculate_metrics(labels[indices], scores[indices], threshold))
            cohort_metrics[split] = _cohort_metrics(examples, scores, threshold, split)
        report_families[family] = {
            "feature_count": features.shape[1],
            "threshold": threshold,
            "metrics": split_metrics,
            "negative_cohorts": cohort_metrics,
        }

    manifest_digest = artifact_sha256(manifest)
    has_same_provider_test = any(
        example.split == "test" and example.negative_cohort == "same_provider" for example in examples
    )
    has_temporal_both_classes = {example.label for example in examples if example.split == "temporal"} == {0, 1}
    return {
        "schema_version": 1,
        "experiment": "synthid-d1-confounds",
        "target_provider": target_provider,
        "manifest_sha256": manifest_digest,
        "max_validation_fpr": max_fpr,
        "eligible_examples": len(examples),
        "split_counts": dict(sorted(split_counts.items())),
        "negative_cohort_counts": dict(sorted(negative_counts.items())),
        "evidence_ready": has_same_provider_test and has_temporal_both_classes,
        "evidence_missing": [
            reason
            for missing, reason in (
                (not has_same_provider_test, "locked test has no same-provider hard negative"),
                (not has_temporal_both_classes, "temporal split does not contain both labels"),
            )
            if missing
        ],
        "feature_schema": {
            "families": list(FEATURE_FAMILIES),
            "thumbnail_size": THUMBNAIL_SIZE,
            "format_names": list(FORMAT_NAMES),
        },
        "logistic_regression": {
            "iterations": LOGISTIC_ITERATIONS,
            "learning_rate": LOGISTIC_LEARNING_RATE,
            "l2": LOGISTIC_L2,
            "class_balancing": "equal total weight per class",
        },
        "families": report_families,
    }


@click.command()
@click.argument("manifest", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target-provider", required=True, type=click.Choice(["openai", "google"]))
@click.option("--report-out", required=True, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--max-fpr", type=click.FloatRange(0.0, 1.0), default=0.001, show_default=True)
@click.option("--verify-files/--no-verify-files", default=True, show_default=True)
def main(manifest: Path, target_provider: str, report_out: Path, max_fpr: float, verify_files: bool) -> None:
    """Run D1 confound baselines from a provider-specific research MANIFEST."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = run_experiment(manifest, target_provider, max_fpr=max_fpr, verify_files=verify_files)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("Wrote D1 confound report: %s", report_out)
    if report["evidence_ready"]:
        log.info("D1 report contains same-provider locked negatives and a two-class temporal holdout")
    else:
        log.warning("D1 report lacks required evidence: %s", "; ".join(report["evidence_missing"]))


if __name__ == "__main__":
    main()
