from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import synthid_conformal_cascade as cascade


def _scores(value: float, count: int = 1999) -> tuple[float, ...]:
    return (value,) * count


def _expert(name: str, *, higher_is_positive: bool = True) -> cascade.ExpertCalibration:
    return cascade.ExpertCalibration(
        name=name,
        positive_scores=_scores(1.0),
        negative_scores=_scores(0.0),
        higher_is_positive=higher_is_positive,
    )


def _config(*experts: cascade.ExpertCalibration, coverage_complete: bool = False) -> cascade.CascadeConfig:
    return cascade.CascadeConfig(
        experts=experts,
        positive_alpha=0.001,
        negative_alpha=0.001,
        coverage_complete=coverage_complete,
        scope="synthetic test bank",
    )


def _observation(name: str, score: float | None, *, supported: bool = True) -> cascade.ExpertObservation:
    return cascade.ExpertObservation(name=name, supported=supported, score=score)


def test_empirical_tail_p_values_include_ties_and_smoothing() -> None:
    scores = (0.1, 0.2, 0.3)

    assert cascade._upper_tail_p_value(scores, 0.3) == 0.5
    assert cascade._upper_tail_p_value(scores, 0.31) == 0.25
    assert cascade._lower_tail_p_value(scores, 0.1) == 0.5
    assert cascade._lower_tail_p_value(scores, 0.09) == 0.25


def test_any_expert_can_detect_with_familywise_correction() -> None:
    config = _config(_expert("fixed"), _expert("registered"))
    observations = (_observation("fixed", 2.0), _observation("registered", 0.5))

    result = cascade.classify_observations(config, observations)

    assert result.verdict == "detected"
    assert result.reason == "watermarked_hypothesis_supported"
    assert result.clean_null_p_value == 0.001
    assert result.watermarked_p_value is not None
    assert result.watermarked_p_value > config.negative_alpha


def test_familywise_correction_blocks_bank_wide_false_alarm() -> None:
    config = _config(_expert("fixed"), _expert("registered"), _expert("version-3"))

    result = cascade.classify_observations(
        config,
        (
            _observation("fixed", 2.0),
            _observation("registered", 0.5),
            _observation("version-3", 0.5),
        ),
    )

    assert result.verdict == "abstain"
    assert result.reason == "insufficient_evidence"
    assert result.clean_null_p_value == 0.0015


def test_incomplete_version_coverage_never_claims_absence() -> None:
    config = _config(_expert("fixed"), coverage_complete=False)

    result = cascade.classify_observations(config, (_observation("fixed", -1.0),))

    assert result.verdict == "abstain"
    assert result.reason == "incomplete_coverage"
    assert result.watermarked_p_value == 0.0005


def test_complete_bank_can_reject_every_watermarked_expert() -> None:
    config = _config(_expert("fixed"), _expert("registered"), coverage_complete=True)

    result = cascade.classify_observations(
        config,
        (_observation("fixed", -1.0), _observation("registered", -1.0)),
    )

    assert result.verdict == "not_detected"
    assert result.reason == "unwatermarked_hypothesis_supported"
    assert result.watermarked_p_value == 0.0005


def test_watermarked_union_survives_when_one_version_remains_plausible() -> None:
    ambiguous = cascade.ExpertCalibration(
        name="registered",
        positive_scores=_scores(0.0),
        negative_scores=_scores(0.0),
    )
    config = _config(_expert("fixed"), ambiguous, coverage_complete=True)

    result = cascade.classify_observations(
        config,
        (_observation("fixed", -1.0), _observation("registered", 0.0)),
    )

    assert result.verdict == "abstain"
    assert result.reason == "insufficient_evidence"
    assert result.watermarked_p_value == 1.0


def test_missing_geometry_support_prevents_negative_verdict() -> None:
    config = _config(_expert("fixed"), _expert("registered"), coverage_complete=True)

    result = cascade.classify_observations(
        config,
        (_observation("fixed", -1.0), _observation("registered", None, supported=False)),
    )

    assert result.verdict == "abstain"
    assert result.reason == "incomplete_support"


def test_out_of_distribution_gap_abstains_on_conflicting_evidence() -> None:
    config = _config(_expert("fixed"), coverage_complete=True)

    result = cascade.classify_observations(config, (_observation("fixed", 0.5),))

    assert result.verdict == "abstain"
    assert result.reason == "conflicting_evidence"
    assert result.clean_null_p_value == 0.0005
    assert result.watermarked_p_value == 0.0005


def test_lower_scores_can_be_oriented_as_positive() -> None:
    expert = cascade.ExpertCalibration(
        name="inverse",
        positive_scores=_scores(-1.0),
        negative_scores=_scores(0.0),
        higher_is_positive=False,
    )

    result = cascade.classify_observations(_config(expert), (_observation("inverse", -2.0),))

    assert result.verdict == "detected"


def test_observations_must_explicitly_cover_the_expert_bank() -> None:
    config = _config(_expert("fixed"), _expert("registered"))

    with pytest.raises(ValueError, match=r"missing=\['registered'\]"):
        cascade.classify_observations(config, (_observation("fixed", 2.0),))


def test_cli_writes_hash_pinned_tri_state_report(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    observation_path = tmp_path / "observations.json"
    report_path = tmp_path / "report.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "synthetic CLI test",
                "positive_alpha": 0.001,
                "negative_alpha": 0.001,
                "coverage_complete": False,
                "experts": [
                    {
                        "name": "fixed",
                        "higher_is_positive": True,
                        "positive_scores": list(_scores(1.0)),
                        "negative_scores": list(_scores(0.0)),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    observation_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    {
                        "id": "candidate-1",
                        "observations": [{"name": "fixed", "supported": True, "score": 2.0}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cascade.main,
        [str(calibration_path), str(observation_path), "--report-out", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["scope"] == "synthetic CLI test"
    assert report["counts"] == {"detected": 1, "not_detected": 0, "abstain": 0}
    assert len(report["calibration_sha256"]) == 64
    assert report["records"][0]["result"]["verdict"] == "detected"


def test_loader_rejects_string_boolean_for_complete_coverage(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "invalid test",
                "positive_alpha": 0.001,
                "negative_alpha": 0.001,
                "coverage_complete": "false",
                "experts": [
                    {
                        "name": "fixed",
                        "positive_scores": [1.0],
                        "negative_scores": [0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coverage_complete must be a boolean"):
        cascade.load_config(calibration_path)
