"""Calibrate a versioned SynthID expert bank without forcing binary verdicts.

This research utility combines already-computed pixel-only expert scores. It
does not inspect provenance, metadata, filenames, or provider labels at
inference. Expert support must be determined from predeclared geometry or model
scope, never from the observed score.

The clean null is a union test: any supported expert may provide positive
evidence, so its smallest empirical upper-tail p-value receives a Bonferroni
correction. The watermarked hypothesis is itself a union over possible encoder
states and can be rejected only when every configured expert has complete
coverage and gives a small empirical lower-tail p-value.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import click
from synthid_research_manifest import artifact_sha256

log = logging.getLogger(__name__)

CascadeVerdict = Literal["detected", "not_detected", "abstain"]


@dataclass(frozen=True)
class ExpertCalibration:
    """Frozen positive and negative score distributions for one expert."""

    name: str
    positive_scores: tuple[float, ...]
    negative_scores: tuple[float, ...]
    higher_is_positive: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("expert name must not be empty")
        if not self.positive_scores or not self.negative_scores:
            raise ValueError(f"expert {self.name!r} needs positive and negative calibration scores")
        if not all(math.isfinite(score) for score in (*self.positive_scores, *self.negative_scores)):
            raise ValueError(f"expert {self.name!r} contains a non-finite calibration score")
        direction = 1.0 if self.higher_is_positive else -1.0
        object.__setattr__(self, "positive_scores", tuple(sorted(direction * score for score in self.positive_scores)))
        object.__setattr__(self, "negative_scores", tuple(sorted(direction * score for score in self.negative_scores)))

    def orient(self, score: float) -> float:
        """Return SCORE in the common higher-means-more-positive direction."""
        return score if self.higher_is_positive else -score


@dataclass(frozen=True)
class CascadeConfig:
    """Calibration distributions and two-sided decision levels."""

    experts: tuple[ExpertCalibration, ...]
    positive_alpha: float
    negative_alpha: float
    coverage_complete: bool
    scope: str

    def __post_init__(self) -> None:
        if not self.experts:
            raise ValueError("at least one expert is required")
        names = [expert.name for expert in self.experts]
        if len(set(names)) != len(names):
            raise ValueError("expert names must be unique")
        for label, value in (("positive_alpha", self.positive_alpha), ("negative_alpha", self.negative_alpha)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{label} must be in (0, 1]")
        if not self.scope:
            raise ValueError("detector scope must not be empty")


@dataclass(frozen=True)
class ExpertObservation:
    """One expert score, or an explicit unsupported result."""

    name: str
    supported: bool
    score: float | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observation expert name must not be empty")
        if self.supported:
            if self.score is None or not math.isfinite(self.score):
                raise ValueError(f"supported expert {self.name!r} needs a finite score")
        elif self.score is not None:
            raise ValueError(f"unsupported expert {self.name!r} must not provide a score")


@dataclass(frozen=True)
class ExpertEvidence:
    """Two empirical p-values for one supported expert."""

    name: str
    score: float
    clean_null_p_value: float
    watermarked_p_value: float
    positive_calibration_count: int
    negative_calibration_count: int


@dataclass(frozen=True)
class CascadeResult:
    """Auditable tri-state verdict for one observation record."""

    verdict: CascadeVerdict
    reason: str
    clean_null_p_value: float | None
    watermarked_p_value: float | None
    supported_expert_count: int
    configured_expert_count: int
    coverage_complete: bool
    evidence: tuple[ExpertEvidence, ...]


def _upper_tail_p_value(sorted_scores: tuple[float, ...], score: float) -> float:
    """Smoothed empirical probability of a calibration score at least SCORE."""
    tail_count = len(sorted_scores) - bisect.bisect_left(sorted_scores, score)
    return (tail_count + 1.0) / (len(sorted_scores) + 1.0)


def _lower_tail_p_value(sorted_scores: tuple[float, ...], score: float) -> float:
    """Smoothed empirical probability of a calibration score at most SCORE."""
    tail_count = bisect.bisect_right(sorted_scores, score)
    return (tail_count + 1.0) / (len(sorted_scores) + 1.0)


def classify_observations(config: CascadeConfig, observations: tuple[ExpertObservation, ...]) -> CascadeResult:
    """Combine one explicit observation from every configured expert."""
    calibration_by_name = {expert.name: expert for expert in config.experts}
    observation_by_name = {observation.name: observation for observation in observations}
    if len(observation_by_name) != len(observations):
        raise ValueError("observation expert names must be unique")
    if observation_by_name.keys() != calibration_by_name.keys():
        missing = sorted(calibration_by_name.keys() - observation_by_name.keys())
        unknown = sorted(observation_by_name.keys() - calibration_by_name.keys())
        raise ValueError(f"observations must cover the configured bank; missing={missing}, unknown={unknown}")

    evidence: list[ExpertEvidence] = []
    for calibration in config.experts:
        observation = observation_by_name[calibration.name]
        if not observation.supported:
            continue
        if observation.score is None:
            raise RuntimeError("validated supported observation lost its score")
        oriented_score = calibration.orient(observation.score)
        evidence.append(
            ExpertEvidence(
                name=calibration.name,
                score=observation.score,
                clean_null_p_value=_upper_tail_p_value(calibration.negative_scores, oriented_score),
                watermarked_p_value=_lower_tail_p_value(calibration.positive_scores, oriented_score),
                positive_calibration_count=len(calibration.positive_scores),
                negative_calibration_count=len(calibration.negative_scores),
            )
        )

    if not evidence:
        return CascadeResult(
            verdict="abstain",
            reason="unsupported",
            clean_null_p_value=None,
            watermarked_p_value=None,
            supported_expert_count=0,
            configured_expert_count=len(config.experts),
            coverage_complete=config.coverage_complete,
            evidence=(),
        )

    supported_count = len(evidence)
    clean_null_p_value = min(1.0, supported_count * min(item.clean_null_p_value for item in evidence))
    watermarked_p_value = max(item.watermarked_p_value for item in evidence)
    rejects_clean_null = clean_null_p_value <= config.positive_alpha
    full_support = supported_count == len(config.experts)
    rejects_watermarked = config.coverage_complete and full_support and watermarked_p_value <= config.negative_alpha

    if rejects_clean_null and rejects_watermarked:
        verdict: CascadeVerdict = "abstain"
        reason = "conflicting_evidence"
    elif rejects_clean_null:
        verdict = "detected"
        reason = "watermarked_hypothesis_supported"
    elif rejects_watermarked:
        verdict = "not_detected"
        reason = "unwatermarked_hypothesis_supported"
    elif config.coverage_complete and not full_support:
        verdict = "abstain"
        reason = "incomplete_support"
    elif not config.coverage_complete and watermarked_p_value <= config.negative_alpha:
        verdict = "abstain"
        reason = "incomplete_coverage"
    else:
        verdict = "abstain"
        reason = "insufficient_evidence"

    return CascadeResult(
        verdict=verdict,
        reason=reason,
        clean_null_p_value=clean_null_p_value,
        watermarked_p_value=watermarked_p_value,
        supported_expert_count=supported_count,
        configured_expert_count=len(config.experts),
        coverage_complete=config.coverage_complete,
        evidence=tuple(evidence),
    )


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast("dict[str, object]", value)


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return cast("list[object]", value)


def _scores(value: object, label: str) -> tuple[float, ...]:
    scores: list[float] = []
    for index, score in enumerate(_sequence(value, label)):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{label}[{index}] must be a number")
        scores.append(float(score))
    return tuple(scores)


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def load_config(path: Path) -> CascadeConfig:
    """Load a schema-versioned calibration manifest."""
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "calibration manifest")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported calibration manifest schema")
    experts: list[ExpertCalibration] = []
    for index, raw_expert in enumerate(_sequence(payload.get("experts"), "experts")):
        expert = _mapping(raw_expert, f"experts[{index}]")
        experts.append(
            ExpertCalibration(
                name=_string(expert.get("name"), f"experts[{index}].name"),
                positive_scores=_scores(expert.get("positive_scores"), f"experts[{index}].positive_scores"),
                negative_scores=_scores(expert.get("negative_scores"), f"experts[{index}].negative_scores"),
                higher_is_positive=_boolean(
                    expert.get("higher_is_positive", True),
                    f"experts[{index}].higher_is_positive",
                ),
            )
        )
    return CascadeConfig(
        experts=tuple(experts),
        positive_alpha=_number(payload.get("positive_alpha"), "positive_alpha"),
        negative_alpha=_number(payload.get("negative_alpha"), "negative_alpha"),
        coverage_complete=_boolean(payload.get("coverage_complete", False), "coverage_complete"),
        scope=_string(payload.get("scope"), "scope"),
    )


def load_observation_records(path: Path) -> list[tuple[str, tuple[ExpertObservation, ...]]]:
    """Load named score records with explicit support for every expert."""
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "observation manifest")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported observation manifest schema")
    records: list[tuple[str, tuple[ExpertObservation, ...]]] = []
    for record_index, raw_record in enumerate(_sequence(payload.get("records"), "records")):
        record = _mapping(raw_record, f"records[{record_index}]")
        record_id = _string(record.get("id"), f"records[{record_index}].id")
        observations: list[ExpertObservation] = []
        for observation_index, raw_observation in enumerate(
            _sequence(record.get("observations"), f"records[{record_index}].observations")
        ):
            observation = _mapping(raw_observation, f"records[{record_index}].observations[{observation_index}]")
            raw_score = observation.get("score")
            observations.append(
                ExpertObservation(
                    name=_string(
                        observation.get("name"),
                        f"records[{record_index}].observations[{observation_index}].name",
                    ),
                    supported=_boolean(
                        observation.get("supported", False),
                        f"records[{record_index}].observations[{observation_index}].supported",
                    ),
                    score=None
                    if raw_score is None
                    else _number(raw_score, f"records[{record_index}].observations[{observation_index}].score"),
                )
            )
        records.append((record_id, tuple(observations)))
    return records


@click.command()
@click.argument("calibration_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("observation_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(calibration_path: Path, observation_path: Path, report_out: Path) -> None:
    """Classify precomputed expert scores using CALIBRATION_PATH."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_config(calibration_path)
    rows: list[dict[str, object]] = []
    verdict_counts: dict[CascadeVerdict, int] = {"detected": 0, "not_detected": 0, "abstain": 0}
    for record_id, observations in load_observation_records(observation_path):
        result = classify_observations(config, observations)
        verdict_counts[result.verdict] += 1
        rows.append({"id": record_id, "result": asdict(result)})
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": config.scope,
                "calibration_sha256": artifact_sha256(calibration_path),
                "observation_sha256": artifact_sha256(observation_path),
                "counts": verdict_counts,
                "records": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d conformal cascade verdicts: %s", len(rows), report_out)


if __name__ == "__main__":
    main()
