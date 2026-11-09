"""Contracts for the publication-cleared real-image cohort builder."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "watermark_benchmark_real_cohort.py"
SPEC = importlib.util.spec_from_file_location("watermark_benchmark_real_cohort", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

FIELDS = (
    "pair_id",
    "provider",
    "model",
    "file",
    "width",
    "height",
    "content_stratum",
    "sha256",
    "source_payload_sha256",
    "reuse_basis",
    "prompt",
)


def _source_row(root: Path, provider: str, color: tuple[int, int, int]) -> dict[str, str]:
    path = root / f"{provider}.png"
    Image.new("RGB", (320, 256), color).save(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "pair_id": "00",
        "provider": provider,
        "model": f"{provider}-model",
        "file": path.name,
        "width": "320",
        "height": "256",
        "content_stratum": "landscape",
        "sha256": digest,
        "source_payload_sha256": digest,
        "reuse_basis": MODULE.REUSE_BASIS,
        "prompt": "a test landscape",
    }


def _write_source_manifest(root: Path, rows: list[dict[str, str]]) -> Path:
    manifest = root / "sources.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_real_cohort_records_source_rights_strata_and_standardization(tmp_path: Path) -> None:
    source_manifest = _write_source_manifest(
        tmp_path,
        [
            _source_row(tmp_path, "alpha", (20, 80, 140)),
            _source_row(tmp_path, "beta", (160, 90, 30)),
        ],
    )

    manifest = MODULE.build_real_cohort(
        tmp_path / "cohort",
        source_manifest=source_manifest,
        adapters=("dwt-dct",),
        dwt_schemes=("sdxl",),
        attacks=(),
        size=256,
    )
    rows = MODULE.load_manifest(manifest)

    assert len(rows) == 4
    assert {(row.arm, row.state) for row in rows} == {
        ("matched_negative", "clean"),
        ("positive", "marked"),
    }
    assert {row.transform.parameters["source"]["provider"] for row in rows} == {"alpha", "beta"}
    assert {row.transform.parameters["source"]["content_stratum"] for row in rows} == {"landscape"}
    assert {row.transform.parameters["source"]["reuse_basis"] for row in rows} == {MODULE.REUSE_BASIS}
    assert all(len(row.transform.parameters["source"]["cohort_helper_sha256"]) == 64 for row in rows)
    assert all(len(row.transform.parameters["source"]["manifest_helper_sha256"]) == 64 for row in rows)
    assert all(row.source_revision.startswith("engine-selection-content@sha256:") for row in rows)
    for row in rows:
        with Image.open(row.path) as image:
            assert image.size == (256, 256)


def test_real_cohort_rejects_duplicate_pixels_before_creating_output(tmp_path: Path) -> None:
    first = _source_row(tmp_path, "alpha", (20, 80, 140))
    second = _source_row(tmp_path, "beta", (20, 80, 140))
    source_manifest = _write_source_manifest(tmp_path, [first, second])
    output = tmp_path / "cohort"

    with pytest.raises(ValueError, match="duplicate source sha256"):
        MODULE.build_real_cohort(output, source_manifest=source_manifest, adapters=("dwt-dct",), size=256)

    assert not output.exists()


def test_real_cohort_requires_explicit_public_reuse_basis(tmp_path: Path) -> None:
    rows = [
        _source_row(tmp_path, "alpha", (20, 80, 140)),
        _source_row(tmp_path, "beta", (160, 90, 30)),
    ]
    rows[1]["reuse_basis"] = "unspecified"
    source_manifest = _write_source_manifest(tmp_path, rows)

    with pytest.raises(ValueError, match="unsupported reuse_basis"):
        MODULE.load_real_carriers(source_manifest)


def test_real_cohort_requires_provider_balance(tmp_path: Path) -> None:
    source_manifest = _write_source_manifest(
        tmp_path,
        [_source_row(tmp_path, "alpha", (20, 80, 140))],
    )

    with pytest.raises(ValueError, match="at least two providers"):
        MODULE.load_real_carriers(source_manifest)


def test_real_cohort_supports_videoseal_image_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import videoseal_oracle

    monkeypatch.setattr(videoseal_oracle, "available", lambda: True)
    monkeypatch.setattr(videoseal_oracle, "load_model", lambda: object())
    monkeypatch.setattr(
        videoseal_oracle,
        "embed_image",
        lambda _model, pixels, _message: (pixels + 0.01).clip(0.0, 1.0),
    )
    source_manifest = _write_source_manifest(
        tmp_path,
        [
            _source_row(tmp_path, "alpha", (20, 80, 140)),
            _source_row(tmp_path, "beta", (160, 90, 30)),
        ],
    )

    manifest = MODULE.build_real_cohort(
        tmp_path / "cohort",
        source_manifest=source_manifest,
        adapters=("videoseal-image",),
        attacks=("jpeg-q90",),
        size=256,
    )
    rows = MODULE.load_manifest(manifest)

    assert len(rows) == 8
    assert {row.adapter for row in rows} == {"videoseal-image"}
    assert {(row.arm, row.state) for row in rows} == {
        ("matched_negative", "clean"),
        ("positive", "marked"),
        ("positive", "attacked"),
        ("wrong_key", "forged"),
    }
