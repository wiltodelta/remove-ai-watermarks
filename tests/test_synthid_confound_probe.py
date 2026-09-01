"""Tests for the manifest-driven SynthID D1 confound challenge."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_confound_probe as probe
import synthid_research_manifest as manifest


def _write_image(path: Path, color: tuple[int, int, int], *, note: str | None = None) -> tuple[str, str]:
    image = Image.new("RGB", (12, 10), color)
    pnginfo = None
    if note is not None:
        pnginfo = PngInfo()
        pnginfo.add_text("note", note)
    image.save(path, pnginfo=pnginfo)
    return hashlib.sha256(path.read_bytes()).hexdigest(), hashlib.sha256(image.tobytes()).hexdigest()


def _row(
    artifact_sha: str,
    pixel_sha: str,
    artifact_path: str,
    *,
    group_id: str,
    split: str,
    outcome: str,
    source_provider: str,
    oracle_role: str = "ordinary",
) -> dict[str, str]:
    matching_oracle = source_provider == "openai"
    return {
        "artifact_sha256": artifact_sha,
        "pixel_sha256": pixel_sha,
        "artifact_path": artifact_path,
        "parent_sha256": "",
        "group_id": group_id,
        "target_provider": "openai",
        "source_provider": source_provider,
        "surface": "synthetic-test",
        "model_epoch": "test-epoch",
        "generation_session": f"session-{group_id}",
        "content_stratum": "flat-graphic",
        "width": "12",
        "height": "10",
        "format": "png",
        "transform": "original",
        "split": split,
        "c2pa_outcome": "detected" if outcome == "detected" else "not_present",
        "synthid_outcome": outcome,
        "verified_via": "openai-api" if matching_oracle else "source-evidence",
        "evidence_reference": "" if matching_oracle else f"https://example.test/{group_id}",
        "oracle_session": f"oracle-{group_id}" if matching_oracle else "",
        "oracle_role": oracle_role,
        "captured_at": "2026-08-09T10:00:00Z",
        "oracle_checked_at": "2026-08-09T10:05:00Z",
        "notes": "synthetic confound fixture",
    }


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    path = root / "manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest.FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _corpus(root: Path, *, include_temporal: bool = True) -> Path:
    rows: list[dict[str, str]] = []
    index = 0
    split_sizes = {"train": 4, "validation": 2, "test": 2}
    if include_temporal:
        split_sizes["temporal"] = 2
    for split, per_class in split_sizes.items():
        for label in (1, 0):
            for offset in range(per_class):
                index += 1
                color = (230, 10 + index, 20 + offset) if label else (20 + offset, 10 + index, 230)
                path = root / f"image-{index}.png"
                artifact_sha, pixel_sha = _write_image(path, color)
                source_provider = "openai" if label or offset == 0 else "camera"
                rows.append(
                    _row(
                        artifact_sha,
                        pixel_sha,
                        path.name,
                        group_id=f"group-{index}",
                        split=split,
                        outcome="detected" if label else "not_detected",
                        source_provider=source_provider,
                    )
                )
    return _write_manifest(root, rows)


def test_canonical_features_ignore_container_metadata(tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first_sha, _ = _write_image(first, (10, 20, 30), note="short")
    second_sha, _ = _write_image(second, (10, 20, 30), note="a much longer metadata value")
    first_example = probe.Example(first, first_sha, "one", "train", 1, None)
    second_example = probe.Example(second, second_sha, "two", "train", 1, None)

    assert not np.array_equal(
        probe.extract_features(first_example, "container"),
        probe.extract_features(second_example, "container"),
    )
    np.testing.assert_array_equal(
        probe.extract_features(first_example, "canonical"),
        probe.extract_features(second_example, "canonical"),
    )


def test_feature_matrices_decode_each_artifact_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first_sha, _ = _write_image(first, (10, 20, 30))
    second_sha, _ = _write_image(second, (30, 20, 10))
    examples = [
        probe.Example(first, first_sha, "one", "train", 1, None),
        probe.Example(second, second_sha, "two", "train", 0, "external"),
    ]
    original = probe._decoded_rgb
    decode_count = 0

    def counting_decode(path: Path):
        nonlocal decode_count
        decode_count += 1
        return original(path)

    monkeypatch.setattr(probe, "_decoded_rgb", counting_decode)

    matrices = probe.feature_matrices(examples)

    assert decode_count == len(examples)
    assert set(matrices) == set(probe.FEATURE_FAMILIES)
    assert all(matrix.shape[0] == len(examples) for matrix in matrices.values())


def test_logistic_model_separates_simple_content_confound():
    features = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

    model = probe.fit_logistic(features, labels)
    scores = probe.predict_scores(model, features)

    assert max(scores[:2]) < min(scores[2:])


def test_threshold_respects_validation_false_positive_limit():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    scores = np.asarray([0.9, 0.8, 0.7, 0.1], dtype=np.float64)

    threshold = probe.select_threshold(labels, scores, max_fpr=0.0)
    metrics = probe.calculate_metrics(labels, scores, threshold)

    assert threshold == pytest.approx(0.8)
    assert metrics.tpr == 1.0
    assert metrics.fpr == 0.0


def test_run_experiment_reports_same_provider_and_temporal_gate(tmp_path: Path):
    path = _corpus(tmp_path)

    report = probe.run_experiment(path, "openai")

    assert report["evidence_ready"] is True
    assert report["evidence_missing"] == []
    canonical = report["families"]["canonical"]
    assert canonical["metrics"]["test"]["auc"] == 1.0
    assert canonical["negative_cohorts"]["test"]["same_provider"]["count"] == 1
    assert canonical["metrics"]["temporal"]["positives"] == 2
    assert canonical["metrics"]["temporal"]["negatives"] == 2
    assert str(tmp_path) not in json.dumps(report)


def test_report_stays_explicit_when_temporal_holdout_is_missing(tmp_path: Path):
    path = _corpus(tmp_path, include_temporal=False)

    report = probe.run_experiment(path, "openai")

    assert report["evidence_ready"] is False
    assert report["evidence_missing"] == ["temporal split does not contain both labels"]
    assert report["families"]["container"]["metrics"]["temporal"] is None


def test_candidate_rows_are_not_detector_examples(tmp_path: Path):
    path = _corpus(tmp_path)
    rows = probe._read_manifest(path)
    candidate_path = tmp_path / "candidate.png"
    artifact_sha, pixel_sha = _write_image(candidate_path, (90, 90, 90))
    rows.append(
        _row(
            artifact_sha,
            pixel_sha,
            candidate_path.name,
            group_id="candidate-group",
            split="train",
            outcome="detected",
            source_provider="openai",
            oracle_role="candidate",
        )
    )
    _write_manifest(tmp_path, rows)

    examples = probe.load_examples(path, "openai")

    assert all(example.artifact_path.name != candidate_path.name for example in examples)


def test_manifest_audit_blocks_group_leakage_before_training(tmp_path: Path):
    path = _corpus(tmp_path)
    rows = probe._read_manifest(path)
    rows[-1]["group_id"] = rows[0]["group_id"]
    _write_manifest(tmp_path, rows)

    with pytest.raises(ValueError, match="manifest audit failed"):
        probe.run_experiment(path, "openai")


def test_positive_source_provider_must_match_target(tmp_path: Path):
    path = _corpus(tmp_path)
    rows = probe._read_manifest(path)
    positive = next(row for row in rows if row["synthid_outcome"] == "detected")
    positive["source_provider"] = "google"
    _write_manifest(tmp_path, rows)

    with pytest.raises(ValueError, match="declares source_provider"):
        probe.load_examples(path, "openai")
