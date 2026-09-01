from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_oracle_batch as batch
from synthid_periodic_tile import fold_residual_template, unit_tile
from synthid_periodic_tile_probe import PeriodicTileModel
from synthid_periodic_tile_probe import save_model as save_tile_model
from synthid_phase_carrier import PhaseCarrierModel
from synthid_phase_carrier import save_model as save_phase_model


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(23)
    raw_tile = rng.normal(size=(8, 8, 3))
    raw_tile -= np.mean(raw_tile, axis=(0, 1), keepdims=True)
    raw_tile, _ = unit_tile(raw_tile)
    source = np.clip(np.rint(128.0 + 24.0 * np.tile(raw_tile, (8, 8, 1))), 0, 255).astype(np.uint8)
    source_path = tmp_path / "source.png"
    Image.fromarray(source, mode="RGB").save(source_path)

    folded = fold_residual_template(
        source,
        tile_height=8,
        tile_width=8,
        denoise_sigma=1.0,
    )
    template, expected_norm = unit_tile(folded)
    tile_model = PeriodicTileModel(64, 64, 8, 8, 1.0, template, expected_norm)
    tile_model_path = tmp_path / "tile-model.npz"
    save_tile_model(tile_model_path, tile_model)

    spectra = np.stack([np.fft.rfft2(source[:, :, channel]) for channel in range(3)], axis=2)
    magnitude = np.abs(spectra)
    magnitude[0, 0, :] = 0.0
    row, column, channel = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
    phase_model = PhaseCarrierModel(
        64,
        64,
        np.asarray([row], dtype=np.int32),
        np.asarray([column], dtype=np.int32),
        np.asarray([channel], dtype=np.int8),
        np.asarray([np.angle(spectra[row, column, channel])]),
        np.asarray([1.0]),
        np.asarray([magnitude[row, column, channel]]),
    )
    phase_model_path = tmp_path / "phase-model.npz"
    save_phase_model(phase_model_path, phase_model)
    return source_path, tile_model_path, phase_model_path


def _build(tmp_path: Path) -> tuple[Path, Path]:
    source, tile_model, phase_model = _fixture(tmp_path)
    output_dir = tmp_path / "oracle-batch"
    manifest_path = batch.build_batch(
        [source],
        output_dir=output_dir,
        tile_model_path=tile_model,
        phase_model_path=phase_model,
        tile_threshold=0.5,
        phase_threshold=0.5,
        active_threshold=0.0,
        strength=2.0,
        seed=20260810,
        provider="google",
        repository_root=Path(__file__).resolve().parent.parent,
    )
    return output_dir, manifest_path


def test_build_preregisters_fixed_request_order_without_copying_source(tmp_path: Path) -> None:
    output_dir, manifest_path = _build(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["status"] == "preregistered_unsubmitted"
    assert manifest["request_count"] == 5
    assert [row["role"] for row in manifest["rows"]] == list(batch.ROLE_ORDER)
    assert manifest["rows"][0]["in_batch"] is False
    assert not (output_dir / "source.png").exists()
    assert manifest["provider"] == "google"
    assert "shifted, and orthogonal_random are detected" in manifest["decision_rule"]
    assert all(row["synthid_result"] is None for row in manifest["rows"])
    assert all(row["c2pa_result"] is None for row in manifest["rows"])
    template = json.loads((output_dir / "results-template.json").read_text(encoding="utf-8"))
    assert template["manifest_sha256"] == batch.artifact_sha256(manifest_path)
    assert [row["artifact_sha256"] for row in template["rows"]] == [row["artifact_sha256"] for row in manifest["rows"]]
    assert (
        batch.verify_batch(
            manifest_path,
            repository_root=Path(__file__).resolve().parent.parent,
        )["source_count"]
        == 1
    )


def test_verify_rejects_derivative_mutation(tmp_path: Path) -> None:
    output_dir, manifest_path = _build(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aligned_row = next(row for row in manifest["rows"] if row["role"] == "aligned")
    aligned_path = output_dir / aligned_row["path"]
    with Image.open(aligned_path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    pixels[0, 0, 0] ^= 1
    Image.fromarray(pixels, mode="RGB").save(aligned_path)
    assert batch.artifact_sha256(aligned_path) != aligned_row["artifact_sha256"]

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        batch.verify_batch(
            manifest_path,
            repository_root=Path(__file__).resolve().parent.parent,
        )


def test_build_rejects_output_inside_repository(tmp_path: Path) -> None:
    source, tile_model, phase_model = _fixture(tmp_path)
    repository_root = Path(__file__).resolve().parent.parent

    with pytest.raises(ValueError, match="outside the repository"):
        batch.build_batch(
            [source],
            output_dir=repository_root / ".local-eval/oracle-batch-test",
            tile_model_path=tile_model,
            phase_model_path=phase_model,
            tile_threshold=0.5,
            phase_threshold=0.5,
            active_threshold=0.0,
            strength=2.0,
            seed=20260810,
            provider="google",
            repository_root=repository_root,
        )


def test_build_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    source, tile_model, phase_model = _fixture(tmp_path)
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "marker.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(ValueError, match="must not already contain"):
        batch.build_batch(
            [source],
            output_dir=output_dir,
            tile_model_path=tile_model,
            phase_model_path=phase_model,
            tile_threshold=0.5,
            phase_threshold=0.5,
            active_threshold=0.0,
            strength=2.0,
            seed=20260810,
            provider="google",
            repository_root=Path(__file__).resolve().parent.parent,
        )


def test_evaluate_requires_controls_and_aligned_outcome(tmp_path: Path) -> None:
    output_dir, manifest_path = _build(tmp_path)
    results = json.loads((output_dir / "results-template.json").read_text(encoding="utf-8"))
    for row in results["rows"]:
        row["synthid_result"] = "not_detected" if row["role"] == "aligned" else "detected"
        row["c2pa_result"] = "unavailable"
        row["raw_response"] = f"verbatim {row['role']} result"
        row["submitted_at"] = "2026-08-10T20:00:00Z"
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    report = batch.evaluate_results(
        manifest_path,
        results_path,
        repository_root=Path(__file__).resolve().parent.parent,
    )

    assert report["counts"] == {
        "causal_success": 1,
        "aligned_still_detected": 0,
        "control_failed": 0,
        "indeterminate": 0,
    }

    shifted = next(row for row in results["rows"] if row["role"] == "shifted")
    shifted["synthid_result"] = "not_detected"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    control_report = batch.evaluate_results(
        manifest_path,
        results_path,
        repository_root=Path(__file__).resolve().parent.parent,
    )
    assert control_report["counts"]["control_failed"] == 1


def test_evaluate_rejects_incomplete_result(tmp_path: Path) -> None:
    output_dir, manifest_path = _build(tmp_path)
    results = json.loads((output_dir / "results-template.json").read_text(encoding="utf-8"))
    results["rows"].pop()
    results_path = output_dir / "incomplete-results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cover every"):
        batch.evaluate_results(
            manifest_path,
            results_path,
            repository_root=Path(__file__).resolve().parent.parent,
        )
