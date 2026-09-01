"""Route SynthID pixel experts without an unsafe union of overlapping positives.

The registered expert owns its measured scale-search range and the large expert
owns its separately challenged native large-image range. A fixed-only crossing
remains auditable evidence but cannot produce a bank-level detection. The bank
never claims absence because encoder-version coverage is incomplete.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import click

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from synthid_conformal_cascade import (  # noqa: E402
    ExpertObservation,
    load_observation_records,
)
from synthid_research_manifest import artifact_sha256  # noqa: E402
from synthid_runtime import synthid_detector  # noqa: E402

log = logging.getLogger(__name__)

RoutedVerdict = Literal["detected", "abstain"]


@dataclass(frozen=True)
class RoutedBankResult:
    """One conservative bank-level decision with every expert score retained."""

    verdict: RoutedVerdict
    reason: str
    selected_expert: str | None
    fixed_supported: bool
    fixed_score: float | None
    registered_supported: bool
    registered_score: float | None
    large_supported: bool
    large_score: float | None


def classify_routed(observations: tuple[ExpertObservation, ...]) -> RoutedBankResult:
    """Route explicit fixed, registered, and large observations without an OR rule."""
    by_name = {observation.name: observation for observation in observations}
    if len(by_name) != len(observations):
        raise ValueError("observation expert names must be unique")
    expected = {
        synthid_detector.DETECTOR_ID,
        synthid_detector.REGISTERED_DETECTOR_ID,
        synthid_detector.LARGE_DETECTOR_ID,
    }
    if by_name.keys() != expected:
        missing = sorted(expected - by_name.keys())
        unknown = sorted(by_name.keys() - expected)
        raise ValueError(f"observations must cover the routed bank; missing={missing}, unknown={unknown}")

    fixed = by_name[synthid_detector.DETECTOR_ID]
    registered = by_name[synthid_detector.REGISTERED_DETECTOR_ID]
    large = by_name[synthid_detector.LARGE_DETECTOR_ID]
    if large.supported:
        if large.score is None:
            raise RuntimeError("validated large observation lost its score")
        if large.score >= synthid_detector.LARGE_THRESHOLD:
            verdict: RoutedVerdict = "detected"
            reason = "large_threshold_crossed"
            selected_expert: str | None = large.name
        else:
            verdict = "abstain"
            reason = "large_below_threshold"
            selected_expert = None
    elif registered.supported:
        if registered.score is None:
            raise RuntimeError("validated registered observation lost its score")
        if registered.score >= synthid_detector.REGISTERED_THRESHOLD:
            verdict = "detected"
            reason = "registered_threshold_crossed"
            selected_expert = registered.name
        else:
            verdict = "abstain"
            reason = (
                "fixed_only_ambiguous"
                if fixed.supported and fixed.score is not None and fixed.score >= synthid_detector.TILE_THRESHOLD
                else "registered_below_threshold"
            )
            selected_expert = None
    elif fixed.supported:
        verdict = "abstain"
        reason = (
            "fixed_only_geometry_uncalibrated"
            if fixed.score is not None and fixed.score >= synthid_detector.TILE_THRESHOLD
            else "registered_unsupported"
        )
        selected_expert = None
    else:
        verdict = "abstain"
        reason = "unsupported"
        selected_expert = None

    return RoutedBankResult(
        verdict=verdict,
        reason=reason,
        selected_expert=selected_expert,
        fixed_supported=fixed.supported,
        fixed_score=fixed.score,
        registered_supported=registered.supported,
        registered_score=registered.score,
        large_supported=large.supported,
        large_score=large.score,
    )


@click.command()
@click.argument("observation_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(observation_path: Path, report_out: Path) -> None:
    """Route a three-expert pixel score manifest from OBSERVATION_PATH."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    counts: dict[RoutedVerdict, int] = {"detected": 0, "abstain": 0}
    rows: list[dict[str, object]] = []
    for record_id, observations in load_observation_records(observation_path):
        result = classify_routed(observations)
        counts[result.verdict] += 1
        rows.append({"id": record_id, "result": asdict(result)})
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "observation_sha256": artifact_sha256(observation_path),
                "counts": counts,
                "records": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d routed expert-bank verdicts: %s", len(rows), report_out)


if __name__ == "__main__":
    main()
