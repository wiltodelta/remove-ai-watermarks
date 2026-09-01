"""Tests for the private SynthID research-manifest auditor."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_research_manifest as manifest


def _write_image(path: Path, color: tuple[int, int, int]) -> tuple[str, str]:
    image = Image.new("RGB", (8, 6), color)
    image.save(path)
    artifact_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
    return artifact_sha, pixel_sha


def _row(artifact_sha: str, pixel_sha: str, *, path: str = "image.png", **updates: str) -> dict[str, str]:
    row = {
        "artifact_sha256": artifact_sha,
        "pixel_sha256": pixel_sha,
        "artifact_path": path,
        "parent_sha256": "",
        "group_id": "group-1",
        "target_provider": "openai",
        "source_provider": "openai",
        "surface": "api",
        "model_epoch": "gpt-image-2026-08",
        "generation_session": "session-1",
        "content_stratum": "flat-graphic",
        "width": "8",
        "height": "6",
        "format": "png",
        "transform": "original",
        "split": "train",
        "c2pa_outcome": "detected",
        "synthid_outcome": "detected",
        "verified_via": "openai-api",
        "evidence_reference": "",
        "oracle_session": "oracle-1",
        "oracle_role": "ordinary",
        "captured_at": "2026-08-08T12:00:00Z",
        "oracle_checked_at": "2026-08-08T12:05:00Z",
        "notes": "synthetic test row",
    }
    row.update(updates)
    return row


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest.FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_valid_manifest_and_files_pass(tmp_path: Path):
    artifact = tmp_path / "image.png"
    artifact_sha, pixel_sha = _write_image(artifact, (10, 20, 30))
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [_row(artifact_sha, pixel_sha)])

    assert manifest.audit_manifest(path, verify_files=True) == []


def test_rejects_same_provider_negative_without_matching_oracle(tmp_path: Path):
    artifact_sha = "a" * 64
    pixel_sha = "b" * 64
    path = tmp_path / "manifest.csv"
    row = _row(
        artifact_sha,
        pixel_sha,
        synthid_outcome="not_detected",
        verified_via="source-evidence",
        oracle_session="",
    )
    _write_manifest(path, [row])

    errors = manifest.audit_manifest(path)

    assert any("same-provider negative" in error for error in errors)


def test_rejects_external_source_evidence_without_reference(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    row = _row(
        "a" * 64,
        "b" * 64,
        source_provider="camera",
        synthid_outcome="not_detected",
        verified_via="source-evidence",
        oracle_session="",
        c2pa_outcome="not_present",
    )
    _write_manifest(path, [row])

    errors = manifest.audit_manifest(path)

    assert any("source-evidence requires evidence_reference" in error for error in errors)


def test_accepts_external_source_evidence_with_reference(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    row = _row(
        "a" * 64,
        "b" * 64,
        source_provider="camera",
        synthid_outcome="not_detected",
        verified_via="source-evidence",
        evidence_reference="https://example.test/original-record",
        oracle_session="",
        c2pa_outcome="not_present",
    )
    _write_manifest(path, [row])

    assert manifest.audit_manifest(path) == []


def test_rejects_indeterminate_training_label(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    row = _row(
        "a" * 64,
        "b" * 64,
        synthid_outcome="indeterminate",
        verified_via="openai-api",
        oracle_checked_at="",
    )
    _write_manifest(path, [row])

    errors = manifest.audit_manifest(path)

    assert any("requires a detected or not_detected" in error for error in errors)


def test_rejects_group_leakage_across_splits(tmp_path: Path):
    first = _row("a" * 64, "b" * 64)
    second = _row(
        "c" * 64,
        "d" * 64,
        artifact_path="other.png",
        split="test",
        generation_session="session-2",
    )
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [first, second])

    errors = manifest.audit_manifest(path)

    assert any("leaks across splits" in error for error in errors)


def test_rejects_derivative_with_missing_parent(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    row = _row("a" * 64, "b" * 64, transform="jpeg-q90", parent_sha256="c" * 64)
    _write_manifest(path, [row])

    errors = manifest.audit_manifest(path)

    assert any("parent_sha256 is not present" in error for error in errors)


def test_rejects_identical_pixels_in_different_groups(tmp_path: Path):
    first = _row("a" * 64, "b" * 64)
    second = _row(
        "c" * 64,
        "b" * 64,
        artifact_path="other.png",
        group_id="group-2",
        generation_session="session-2",
    )
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [first, second])

    errors = manifest.audit_manifest(path)

    assert any("appears in multiple groups" in error for error in errors)


def test_verify_files_detects_changed_artifact(tmp_path: Path):
    artifact = tmp_path / "image.png"
    artifact_sha, pixel_sha = _write_image(artifact, (10, 20, 30))
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [_row(artifact_sha, pixel_sha)])
    _write_image(artifact, (11, 21, 31))

    errors = manifest.audit_manifest(path, verify_files=True)

    assert any("artifact_sha256 does not match" in error for error in errors)
    assert any("pixel_sha256 does not match" in error for error in errors)


def test_rejects_path_traversal_during_file_verification(tmp_path: Path):
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [_row("a" * 64, "b" * 64, path="../outside.png")])

    errors = manifest.audit_manifest(path, verify_files=True)

    assert any("safe manifest-relative path" in error for error in errors)


def test_accepts_candidate_negative_with_detected_session_control(tmp_path: Path):
    control = _row("a" * 64, "b" * 64, oracle_role="source_control")
    candidate = _row(
        "c" * 64,
        "d" * 64,
        artifact_path="candidate.png",
        parent_sha256="a" * 64,
        transform="carrier-subtract",
        synthid_outcome="not_detected",
        oracle_role="candidate",
    )
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [control, candidate])

    assert manifest.audit_manifest(path) == []


def test_rejects_candidate_negative_without_detected_session_control(tmp_path: Path):
    candidate = _row(
        "a" * 64,
        "b" * 64,
        synthid_outcome="not_detected",
        oracle_role="candidate",
    )
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [candidate])

    errors = manifest.audit_manifest(path)

    assert any("requires a detected source_control" in error for error in errors)


def test_rejects_failed_source_control(tmp_path: Path):
    control = _row(
        "a" * 64,
        "b" * 64,
        synthid_outcome="not_detected",
        oracle_role="source_control",
    )
    path = tmp_path / "manifest.csv"
    _write_manifest(path, [control])

    errors = manifest.audit_manifest(path)

    assert any("source_control must have a detected" in error for error in errors)
