from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_routed_expert_bank as bank
from synthid_conformal_cascade import ExpertObservation


def _observations(
    fixed_score: float | None,
    registered_score: float | None,
    *,
    fixed_supported: bool = True,
    registered_supported: bool = True,
    large_score: float | None = None,
    large_supported: bool = False,
) -> tuple[ExpertObservation, ...]:
    return (
        ExpertObservation(bank.synthid_detector.DETECTOR_ID, fixed_supported, fixed_score),
        ExpertObservation(
            bank.synthid_detector.REGISTERED_DETECTOR_ID,
            registered_supported,
            registered_score,
        ),
        ExpertObservation(
            bank.synthid_detector.LARGE_DETECTOR_ID,
            large_supported,
            large_score,
        ),
    )


def test_registered_crossing_is_the_only_positive_route() -> None:
    result = bank.classify_routed(_observations(-1.0, 1.1))

    assert result.verdict == "detected"
    assert result.reason == "registered_threshold_crossed"
    assert result.selected_expert == bank.synthid_detector.REGISTERED_DETECTOR_ID


def test_large_crossing_is_a_separate_positive_route() -> None:
    result = bank.classify_routed(
        _observations(
            None,
            None,
            fixed_supported=False,
            registered_supported=False,
            large_score=1.1,
            large_supported=True,
        )
    )

    assert result.verdict == "detected"
    assert result.reason == "large_threshold_crossed"
    assert result.selected_expert == bank.synthid_detector.LARGE_DETECTOR_ID


def test_fixed_crossing_in_overlapping_geometry_abstains() -> None:
    result = bank.classify_routed(_observations(0.5, 0.0))

    assert result.verdict == "abstain"
    assert result.reason == "fixed_only_ambiguous"


def test_fixed_crossing_outside_registered_geometry_abstains() -> None:
    result = bank.classify_routed(
        _observations(0.5, None, registered_supported=False),
    )

    assert result.verdict == "abstain"
    assert result.reason == "fixed_only_geometry_uncalibrated"


def test_unsupported_bank_abstains() -> None:
    result = bank.classify_routed(
        _observations(None, None, fixed_supported=False, registered_supported=False),
    )

    assert result.verdict == "abstain"
    assert result.reason == "unsupported"


def test_observations_must_cover_the_exact_routed_bank() -> None:
    with pytest.raises(ValueError, match=r"missing=.*synthid-periodic-tile-large-v1"):
        bank.classify_routed((ExpertObservation(bank.synthid_detector.DETECTOR_ID, True, 0.5),))


def test_cli_writes_hash_pinned_report(tmp_path: Path) -> None:
    observations_path = tmp_path / "observations.json"
    report_path = tmp_path / "report.json"
    observations_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "id": "candidate-1",
                        "observations": [
                            {
                                "name": bank.synthid_detector.DETECTOR_ID,
                                "supported": True,
                                "score": 0.5,
                            },
                            {
                                "name": bank.synthid_detector.REGISTERED_DETECTOR_ID,
                                "supported": True,
                                "score": 0.0,
                            },
                            {
                                "name": bank.synthid_detector.LARGE_DETECTOR_ID,
                                "supported": False,
                                "score": None,
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        bank.main,
        [str(observations_path), "--report-out", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["observation_sha256"]) == 64
    assert report["counts"] == {"detected": 0, "abstain": 1}
    assert report["records"][0]["result"]["reason"] == "fixed_only_ambiguous"
