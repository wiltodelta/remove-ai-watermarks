"""Score images and apply the conservative SynthID expert-bank router."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TypedDict

import click
from synthid_conformal_cascade import ExpertObservation
from synthid_routed_expert_bank import classify_routed
from synthid_runtime_expert_scores import ExpertScore, score_path

log = logging.getLogger(__name__)


class RoutedImage(TypedDict):
    """One scored image and its conservative routed result."""

    id: str
    path: str
    width: int
    height: int
    observations: list[ExpertScore]
    result: dict[str, object]


def detect_path(path: Path) -> RoutedImage:
    """Score and conservatively route one image PATH."""
    scored = score_path(path)
    observations = tuple(
        ExpertObservation(
            name=observation["name"],
            supported=observation["supported"],
            score=observation["score"],
        )
        for observation in scored["observations"]
    )
    return {**scored, "result": asdict(classify_routed(observations))}


@click.command()
@click.argument("images", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(images: tuple[Path, ...], report_out: Path) -> None:
    """Score and route IMAGES through the conservative pixel expert bank."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = [detect_path(path) for path in images]
    detected = sum(record["result"].get("verdict") == "detected" for record in records)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "counts": {"detected": detected, "abstain": len(records) - detected},
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("Wrote %d routed image verdicts: %s", len(records), report_out)


if __name__ == "__main__":
    main()
