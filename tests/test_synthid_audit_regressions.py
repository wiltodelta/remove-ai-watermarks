"""Regression coverage for research evidence contracts."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import synthid_affine_lattice_probe as lattice
import synthid_periodic_tile_probe as periodic
import synthid_pixel_probe as pixel
import synthid_runtime_expert_scores as experts
from click.testing import CliRunner
from PIL import Image


def test_archive_imports():
    root = Path(__file__).resolve().parents[1]
    assert Path(lattice.__file__).is_relative_to(root)
    assert Path(experts.__file__).is_relative_to(root)


@pytest.mark.parametrize(
    "detector_name",
    [
        experts.synthid_detector.OPPONENT_REGISTERED_DETECTOR_ID,
        experts.synthid_detector.FINE_OPPONENT_REGISTERED_DETECTOR_ID,
    ],
)
def test_registered_fallback_preserves_identity(monkeypatch, detector_name):
    def detect(_path, *, image, register_scale):
        return experts.synthid_detector.SynthIDDetection(
            status="detected",
            width=768,
            height=768,
            score=1.02,
            threshold=1.05,
            detector=detector_name if register_scale else experts.FIXED_EXPERT_NAME,
        )

    monkeypatch.setattr(experts.synthid_detector, "detect_synthid", detect)
    observations = experts.score_pixels(np.zeros((768, 768, 3), dtype=np.uint8))
    routed = next(item for item in observations if item["name"] == detector_name)
    assert routed["score"] == 1.02
    assert not next(item for item in observations if item["name"] == experts.REGISTERED_EXPERT_NAME)["supported"]


def test_period_selection_does_not_read_confirmation(monkeypatch):
    pixels = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    template = np.random.default_rng(43).normal(size=(8, 8, 3))
    monkeypatch.setattr(lattice, "_period_candidate_indices", lambda *_args: np.array([0, 1]))
    monkeypatch.setattr(lattice, "_period_alias_candidate_indices", lambda *_args: np.array([0, 1]))
    confirmation = {16.0: 0.1, 17.0: 0.9}

    def whitened(*_args, period, **_kwargs):
        return (0.9 if period == 16.0 else 0.8, confirmation[period])

    def amplitude(*_args, period, **_kwargs):
        return lattice._AmplitudeScore(period, 0.9 if period == 16.0 else 0.8, confirmation[period], 0, 0, 0.0, 0.0)

    monkeypatch.setattr(lattice, "_content_whitened_score", whitened)
    monkeypatch.setattr(lattice, "_amplitude_score", amplitude)

    def score():
        return lattice.score_lattice(
            pixels,
            template,
            periods=np.array([16.0, 17.0]),
            rotations_degrees=np.array([0.0]),
            patch_size=32,
            grid_size=4,
            harmonic_count=4,
        )

    first = score()
    confirmation.update({16.0: 0.9, 17.0: 0.1})
    second = score()
    assert first.selected_period == second.selected_period == 16.0
    assert first.confirmation_whitened_match != second.confirmation_whitened_match


@pytest.mark.parametrize("size", [64, 65])
def test_periodic_discover_save_load_registered_json(tmp_path, size):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (size, size, 3), dtype=np.uint8)).save(path)
        paths.append(path)
    model = periodic.discover_model(paths, tile_height=8, tile_width=8)
    target = tmp_path / "model.npz"
    periodic.save_model(target, model)
    restored = periodic.load_model(target)
    result = periodic.score_image(paths[0], restored, register=True)
    assert type(result.row_shift) is int
    assert type(result.column_shift) is int
    assert json.loads(json.dumps(asdict(result)))["score"] == result.score
    report = tmp_path / "scores.json"
    command = CliRunner().invoke(
        periodic.main, ["score", str(target), str(paths[0]), "--register", "--report-out", str(report)]
    )
    assert command.exit_code == 0, command.exception
    assert json.loads(report.read_text())["scores"]


@pytest.mark.parametrize("clean_shape", [(32, 32), (32, 128)])
def test_removal_rejects_incomparable_geometry(tmp_path, clean_shape):
    positive = tmp_path / "positive.png"
    clean = tmp_path / "clean.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(positive)
    Image.fromarray(np.random.default_rng(2).integers(0, 256, (*clean_shape, 3), dtype=np.uint8)).save(clean)
    result = CliRunner().invoke(pixel.cli, ["removal", "--pos", str(positive), "--cleaned", str(clean)])
    assert result.exit_code != 0
    assert "geometry" in result.output.lower()
    assert "attenuated" not in result.output


@pytest.mark.parametrize(("score", "verdict"), [(1.02, "abstain"), (1.06, "detected")])
def test_fine_fallback_router_uses_its_own_threshold(monkeypatch, score, verdict):
    from synthid_conformal_cascade import ExpertObservation
    from synthid_routed_expert_bank import classify_routed

    def detect(_path, *, image, register_scale):
        return experts.synthid_detector.SynthIDDetection(
            status="indeterminate" if score < 1.05 else "detected",
            width=768,
            height=768,
            score=score,
            threshold=1.05,
            detector=experts.synthid_detector.FINE_OPPONENT_REGISTERED_DETECTOR_ID
            if register_scale
            else experts.FIXED_EXPERT_NAME,
        )

    monkeypatch.setattr(experts.synthid_detector, "detect_synthid", detect)
    observations = tuple(
        ExpertObservation(**item) for item in experts.score_pixels(np.zeros((768, 768, 3), dtype=np.uint8))
    )
    result = classify_routed(observations)
    assert result.verdict == verdict
    assert result.fine_opponent_score == score
    assert result.registered_score is None
    if verdict == "detected":
        assert result.selected_expert == experts.synthid_detector.FINE_OPPONENT_REGISTERED_DETECTOR_ID


def test_legacy_mislabeled_runtime_manifest_requires_rescoring(tmp_path):
    from synthid_conformal_cascade import load_observation_records

    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "id": "old",
                        "observations": [{"name": experts.REGISTERED_EXPERT_NAME, "supported": True, "score": 1.1}],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="rescoring"):
        load_observation_records(path)
