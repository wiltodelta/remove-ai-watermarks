#!/usr/bin/env python3
"""Summarize paired fidelity results from the engine-selection study.

Run from the repository root after the ``engine_selection_study.py`` harness has
written ``out/engine-selection-study/summary.json``:

    uv run scripts/analyze_engine_selection_study.py

The report uses per-image paired differences and exact two-sided sign tests.
It does not turn the discovery set into a production routing rule.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)

_METRICS: dict[str, tuple[Callable[[float], float], str]] = {
    "lpips_512": (lambda value: value, "lower"),
    "ssim": (lambda value: -value, "higher"),
    "psnr_db": (lambda value: -value, "higher"),
    "mae": (lambda value: value, "lower"),
    "edge_f1": (lambda value: -value, "higher"),
    "laplacian_ratio": (lambda value: abs(math.log(max(value, 1e-12))), "closer_to_one"),
}


def _sign_test_two_sided(wins: int, losses: int) -> float:
    """Return the exact two-sided sign-test p-value, excluding ties."""
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def _index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["provider"], row["pair_id"]): row for row in rows}


def analyze(report: dict[str, Any]) -> dict[str, Any]:
    qwen = _index(report["qwen-zimage"]["rows"])
    chroma = _index(report["chroma-zimage"]["rows"])
    if qwen.keys() != chroma.keys():
        raise ValueError("Qwen and Chroma result keys differ")

    by_provider: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key in sorted(qwen):
        by_provider[key[0]].append((qwen[key], chroma[key]))

    providers: dict[str, Any] = {}
    for provider, pairs in sorted(by_provider.items()):
        metrics: dict[str, Any] = {}
        for metric, (loss_transform, direction) in _METRICS.items():
            deltas: list[float] = []
            chroma_wins = 0
            qwen_wins = 0
            ties = 0
            for qwen_row, chroma_row in pairs:
                qwen_value = float(qwen_row["fidelity"][metric])
                chroma_value = float(chroma_row["fidelity"][metric])
                delta = loss_transform(chroma_value) - loss_transform(qwen_value)
                deltas.append(delta)
                if math.isclose(delta, 0.0, abs_tol=1e-12):
                    ties += 1
                elif delta < 0:
                    chroma_wins += 1
                else:
                    qwen_wins += 1
            metrics[metric] = {
                "preferred_direction": direction,
                "delta_definition": "chroma_loss_minus_qwen_loss; negative favors Chroma",
                "chroma_wins": chroma_wins,
                "qwen_wins": qwen_wins,
                "ties": ties,
                "mean_delta": statistics.fmean(deltas),
                "median_delta": statistics.median(deltas),
                "sign_test_two_sided_p": _sign_test_two_sided(chroma_wins, qwen_wins),
            }
        providers[provider] = {"pairs": len(pairs), "metrics": metrics}

    return {
        "method": "paired per-image differences with exact two-sided sign tests; ties excluded",
        "warning": "Discovery-set evidence only; challenge any proposed rule on unused generations.",
        "providers": providers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("out/engine-selection-study/summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/engine-selection-study/analysis.json"),
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
