#!/usr/bin/env python3
"""Aggregate one or more watermark benchmark JSONL result files.

The report keeps detector observations, post-removal observations, fidelity,
and adapter timing separate. Repeated result files are independent runs.
Repeated pixels are deduplicated by artifact SHA-256 before case-level counts,
while raw observation counts remain visible. The first measured adapter call in
each input file is classified as cold; later calls to that adapter are warm.

This is a development-only reader. It calls no detector, model, oracle, or
provider API and refuses to overwrite an existing report.

    uv run python scripts/watermark_benchmark_report.py results/*.jsonl \
      --output .local-eval/watermark-benchmark-report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from watermark_benchmark import (  # noqa: E402
    DETECTOR_STATUSES,
    SCHEMA_VERSION,
    parse_strict_json,
    require_nonempty_string,
    require_sha256,
    sha256_file,
)

log = logging.getLogger(__name__)

TimingPhase = Literal["cold", "warm"]
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "case_id",
    "pair_id",
    "media_type",
    "adapter",
    "arm",
    "state",
    "artifact",
    "reference",
    "transform",
    "run",
    "detection",
    "removal",
    "fidelity",
}


@dataclass(frozen=True)
class ResultRecord:
    """One minimally validated result row plus its run identity."""

    source: Path
    line_number: int
    data: dict[str, Any]


def _object(value: object, *, field: str, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: {field} must be an object")
    return cast("dict[str, Any]", value)


def _validate_result(raw: object, *, location: str) -> dict[str, Any]:
    values = _object(raw, field="result", location=location)
    if set(values) != _TOP_LEVEL_FIELDS:
        raise ValueError(f"{location}: result fields do not match benchmark schema v{SCHEMA_VERSION}")
    if values["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{location}: unsupported benchmark schema {values['schema_version']!r}")
    for field in ("case_id", "pair_id", "adapter", "arm", "state"):
        require_nonempty_string(values[field], field=field, location=location)
    if values["media_type"] not in ("image", "audio", "video"):
        raise ValueError(f"{location}: media_type must be image, audio, or video")

    artifact = _object(values["artifact"], field="artifact", location=location)
    require_sha256(artifact.get("sha256"), field="artifact.sha256", location=location)
    require_nonempty_string(artifact.get("source_revision"), field="artifact.source_revision", location=location)
    reference = values["reference"]
    if reference is not None:
        require_sha256(
            _object(reference, field="reference", location=location).get("sha256"),
            field="reference.sha256",
            location=location,
        )

    transform = _object(values["transform"], field="transform", location=location)
    require_nonempty_string(transform.get("name"), field="transform.name", location=location)
    require_nonempty_string(transform.get("revision"), field="transform.revision", location=location)
    _object(transform.get("parameters"), field="transform.parameters", location=location)

    run = _object(values["run"], field="run", location=location)
    _object(run.get("repository"), field="run.repository", location=location)
    require_sha256(run.get("kernel_source_sha256"), field="run.kernel_source_sha256", location=location)
    require_sha256(run.get("adapter_source_sha256"), field="run.adapter_source_sha256", location=location)

    detection = _object(values["detection"], field="detection", location=location)
    status = detection.get("status")
    if status not in DETECTOR_STATUSES:
        raise ValueError(f"{location}: unknown detection status {status!r}")
    elapsed = detection.get("adapter_elapsed_ms")
    invalid_elapsed = (
        not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0
    )
    if elapsed is not None and invalid_elapsed:
        raise ValueError(f"{location}: detection.adapter_elapsed_ms must be a finite non-negative number or null")
    _object(values["removal"], field="removal", location=location)
    _object(values["fidelity"], field="fidelity", location=location)
    return values


def load_results(paths: Sequence[Path]) -> list[ResultRecord]:
    """Read benchmark results in input and line order."""
    if not paths:
        raise ValueError("at least one result file is required")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate result file")

    records: list[ResultRecord] = []
    for path in resolved:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    raw = parse_strict_json(line)
                except (json.JSONDecodeError, ValueError) as exc:
                    detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
                    raise ValueError(f"{location}: invalid JSON: {detail}") from exc
                records.append(
                    ResultRecord(
                        source=path,
                        line_number=line_number,
                        data=_validate_result(raw, location=location),
                    )
                )
    if not records:
        raise ValueError("result files contain no records")
    return records


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _case_identity(record: ResultRecord) -> tuple[object, ...]:
    row = record.data
    artifact = cast("dict[str, Any]", row["artifact"])
    reference = cast("dict[str, Any] | None", row["reference"])
    detection = cast("dict[str, Any]", row["detection"])
    return (
        row["pair_id"],
        row["media_type"],
        row["adapter"],
        row["arm"],
        row["state"],
        artifact["sha256"],
        artifact["source_revision"],
        reference["sha256"] if reference is not None else None,
        _canonical(row["transform"]),
        detection.get("expected"),
    )


def nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    """Return a nearest-rank percentile from values sorted in ascending order."""
    if not ordered:
        raise ValueError("cannot calculate a percentile without measurements")
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _timing_stats(adapter: str, phase: TimingPhase, values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "adapter": adapter,
        "phase": phase,
        "n": len(ordered),
        "p50_ms": nearest_rank(ordered, 0.50),
        "p90_ms": nearest_rank(ordered, 0.90),
        "max_ms": ordered[-1],
    }


def _artifact_status_summary(group: Sequence[ResultRecord]) -> dict[str, int]:
    artifacts: dict[str, set[str]] = defaultdict(set)
    for record in group:
        row = record.data
        artifact = cast("dict[str, Any]", row["artifact"])
        detection = cast("dict[str, Any]", row["detection"])
        artifacts[str(artifact["sha256"])].add(str(detection["status"]))
    counts: dict[str, int] = dict.fromkeys(DETECTOR_STATUSES, 0)
    unstable = 0
    for statuses in artifacts.values():
        if len(statuses) != 1:
            unstable += 1
            continue
        counts[next(iter(statuses))] += 1
    return {
        "unique_artifacts": len(artifacts),
        "observations": len(group),
        **counts,
        "unstable": unstable,
    }


def _detection_groups(records: Sequence[ResultRecord], *, include_transform: bool) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[ResultRecord]] = defaultdict(list)
    for record in records:
        row = record.data
        identity = (str(row["adapter"]), str(row["state"]))
        if include_transform:
            transform = cast("dict[str, Any]", row["transform"])
            identity += (str(transform["name"]),)
        grouped[identity].append(record)

    summaries: list[dict[str, Any]] = []
    for identity, group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "adapter": identity[0],
            "state": identity[1],
        }
        if include_transform:
            summary["transform"] = identity[2]
        summary.update(_artifact_status_summary(group))
        summaries.append(summary)
    return summaries


def _detection_groups_by_source_field(
    records: Sequence[ResultRecord],
    *,
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[ResultRecord]] = defaultdict(list)
    for record in records:
        row = record.data
        transform = cast("dict[str, Any]", row["transform"])
        parameters = cast("dict[str, Any]", transform["parameters"])
        source = parameters.get("source")
        if not isinstance(source, dict):
            continue
        value = cast("dict[str, object]", source).get(field)
        if not isinstance(value, str) or not value:
            continue
        grouped[(str(row["adapter"]), str(row["state"]), value)].append(record)

    summaries: list[dict[str, Any]] = []
    for (adapter, state, value), group in sorted(grouped.items()):
        summaries.append(
            {
                "adapter": adapter,
                "state": state,
                field: value,
                **_artifact_status_summary(group),
            }
        )
    return summaries


def _watermark_profile(parameters: Mapping[str, Any]) -> str | None:
    scheme = parameters.get("scheme")
    if isinstance(scheme, str) and scheme:
        return scheme
    variant = parameters.get("variant")
    schema = parameters.get("schema")
    if isinstance(variant, str) and variant and isinstance(schema, int) and not isinstance(schema, bool):
        return f"{variant}/schema-{schema}"
    return None


def _detection_groups_by_profile(records: Sequence[ResultRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[ResultRecord]] = defaultdict(list)
    for record in records:
        row = record.data
        transform = cast("dict[str, Any]", row["transform"])
        parameters = cast("dict[str, Any]", transform["parameters"])
        profile = _watermark_profile(parameters)
        if profile is not None:
            grouped[(str(row["adapter"]), str(row["state"]), profile)].append(record)
    return [
        {
            "adapter": adapter,
            "state": state,
            "profile": profile,
            **_artifact_status_summary(group),
        }
        for (adapter, state, profile), group in sorted(grouped.items())
    ]


def _attack_transition_groups(records: Sequence[ResultRecord]) -> list[dict[str, Any]]:
    marked: dict[tuple[str, str], set[str]] = defaultdict(set)
    attacked: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for record in records:
        row = record.data
        state = str(row["state"])
        if state not in ("marked", "attacked"):
            continue
        adapter = str(row["adapter"])
        pair_id = str(row["pair_id"])
        detection = cast("dict[str, Any]", row["detection"])
        status = str(detection["status"])
        if state == "marked":
            marked[(adapter, pair_id)].add(status)
        else:
            transform = cast("dict[str, Any]", row["transform"])
            attacked[(adapter, pair_id, str(transform["name"]))].add(status)

    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"retained": 0, "lost": 0, "gained": 0, "still_not_detected": 0, "unmeasured": 0}
    )
    for (adapter, pair_id, transform), attacked_statuses in attacked.items():
        counts = grouped[(adapter, transform)]
        marked_statuses = marked.get((adapter, pair_id), set())
        if len(marked_statuses) != 1 or len(attacked_statuses) != 1:
            counts["unmeasured"] += 1
            continue
        before = next(iter(marked_statuses))
        after = next(iter(attacked_statuses))
        transition = {
            ("detected", "detected"): "retained",
            ("detected", "not_detected"): "lost",
            ("not_detected", "detected"): "gained",
            ("not_detected", "not_detected"): "still_not_detected",
        }.get((before, after))
        if transition is None:
            counts["unmeasured"] += 1
        else:
            counts[transition] += 1
    return [
        {"adapter": adapter, "transform": transform, "pairs": sum(counts.values()), **counts}
        for (adapter, transform), counts in sorted(grouped.items())
    ]


def _removal_groups(records: Sequence[ResultRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ResultRecord]] = defaultdict(list)
    for record in records:
        if record.data["state"] == "removed":
            grouped[str(record.data["adapter"])].append(record)

    summaries: list[dict[str, Any]] = []
    for adapter, group in sorted(grouped.items()):
        artifacts: dict[str, set[str]] = defaultdict(set)
        for record in group:
            row = record.data
            removal = cast("dict[str, Any]", row["removal"])
            artifact = cast("dict[str, Any]", row["artifact"])
            artifacts[str(artifact["sha256"])].add(str(removal.get("status")))
        statuses = (
            "signal_detected_after_removal",
            "no_recognized_signal_after_removal",
            "unmeasured",
        )
        counts: dict[str, int] = dict.fromkeys(statuses, 0)
        unstable = 0
        for observed in artifacts.values():
            if len(observed) != 1 or next(iter(observed)) not in counts:
                unstable += 1
            else:
                counts[next(iter(observed))] += 1
        summaries.append(
            {
                "adapter": adapter,
                "unique_artifacts": len(artifacts),
                "observations": len(group),
                **counts,
                "unstable": unstable,
            }
        )
    return summaries


def _fidelity_groups(records: Sequence[ResultRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[ResultRecord]] = defaultdict(list)
    for record in records:
        grouped[(str(record.data["adapter"]), str(record.data["state"]))].append(record)

    summaries: list[dict[str, Any]] = []
    for (adapter, state), group in sorted(grouped.items()):
        pairs: dict[tuple[str, str | None], tuple[str, dict[str, Any]]] = {}
        inconsistent_pairs: set[tuple[str, str | None]] = set()
        for record in group:
            row = record.data
            artifact = cast("dict[str, Any]", row["artifact"])
            reference = cast("dict[str, Any] | None", row["reference"])
            key = (str(artifact["sha256"]), str(reference["sha256"]) if reference is not None else None)
            fidelity = cast("dict[str, Any]", row["fidelity"])
            canonical = _canonical(fidelity)
            previous = pairs.setdefault(key, (canonical, fidelity))
            if previous[0] != canonical:
                inconsistent_pairs.add(key)

        counts = {"measured": 0, "incomparable": 0, "not_measured": 0, "inconsistent": 0}
        identical = 0
        psnr_values: list[float] = []
        mean_psnr_values: list[float] = []
        snr_values: list[float] = []
        for key, (_, value) in pairs.items():
            if key in inconsistent_pairs:
                counts["inconsistent"] += 1
                continue
            status = str(value.get("status"))
            if status not in ("measured", "incomparable", "not_measured"):
                counts["inconsistent"] += 1
                continue
            counts[status] += 1
            if status == "measured":
                identical += int(value.get("identical") is True)
                psnr = value.get("psnr_db")
                if isinstance(psnr, (int, float)) and not isinstance(psnr, bool) and math.isfinite(psnr):
                    psnr_values.append(float(psnr))
                mean_psnr = value.get("mean_psnr_db")
                if isinstance(mean_psnr, (int, float)) and not isinstance(mean_psnr, bool) and math.isfinite(mean_psnr):
                    mean_psnr_values.append(float(mean_psnr))
                snr = value.get("snr_db")
                if isinstance(snr, (int, float)) and not isinstance(snr, bool) and math.isfinite(snr):
                    snr_values.append(float(snr))
        summaries.append(
            {
                "adapter": adapter,
                "state": state,
                "unique_pairs": len(pairs),
                **counts,
                "identical": identical,
                "finite_psnr": len(psnr_values),
                "p50_psnr_db": nearest_rank(sorted(psnr_values), 0.50) if psnr_values else None,
                "finite_mean_psnr": len(mean_psnr_values),
                "p50_mean_psnr_db": nearest_rank(sorted(mean_psnr_values), 0.50) if mean_psnr_values else None,
                "finite_snr": len(snr_values),
                "p50_snr_db": nearest_rank(sorted(snr_values), 0.50) if snr_values else None,
            }
        )
    return summaries


def summarize(records: Sequence[ResultRecord]) -> dict[str, Any]:
    """Build a deterministic, artifact-aware summary of benchmark records."""
    if not records:
        raise ValueError("cannot summarize an empty result set")

    identities: dict[str, tuple[object, ...]] = {}
    case_statuses: dict[tuple[str, str], set[str]] = defaultdict(set)
    mismatches: set[tuple[str, str, str]] = set()
    non_results: set[tuple[str, str, str, str | None]] = set()
    timing_values: dict[tuple[str, TimingPhase], list[float]] = defaultdict(list)
    timed_adapters: set[tuple[Path, str]] = set()

    for record in records:
        row = record.data
        removal = cast("dict[str, Any]", row["removal"])
        if removal.get("certifies_erasure") is not False:
            raise ValueError(f"{record.source}:{record.line_number}: removal.certifies_erasure must be false")
        case_id = str(row["case_id"])
        identity = _case_identity(record)
        previous = identities.setdefault(case_id, identity)
        if previous != identity:
            raise ValueError(f"case identity drift for {case_id!r} at {record.source}:{record.line_number}")

        adapter = str(row["adapter"])
        detection = cast("dict[str, Any]", row["detection"])
        status = str(detection["status"])
        case_statuses[(case_id, adapter)].add(status)
        if detection.get("matches_expected") is False:
            mismatches.add((case_id, adapter, status))
        if status in ("error", "unavailable"):
            error = detection.get("error")
            non_results.add((case_id, adapter, status, str(error) if error is not None else None))

        elapsed = detection.get("adapter_elapsed_ms")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            timing_key = (record.source, adapter)
            phase: TimingPhase = "warm" if timing_key in timed_adapters else "cold"
            timed_adapters.add(timing_key)
            timing_values[(adapter, phase)].append(float(elapsed))

    unstable_cases = [
        {"case_id": case_id, "adapter": adapter, "statuses": sorted(statuses)}
        for (case_id, adapter), statuses in sorted(case_statuses.items())
        if len(statuses) > 1
    ]
    timing = [_timing_stats(adapter, phase, values) for (adapter, phase), values in sorted(timing_values.items())]
    source_counts = Counter(record.source for record in records)
    inputs = [
        {
            "path": str(source),
            "sha256": sha256_file(source),
            "observations": source_counts[source],
        }
        for source in sorted(source_counts)
    ]

    return {
        "runs": len(source_counts),
        "observations": len(records),
        "unique_cases": len(identities),
        "inputs": inputs,
        "detection": _detection_groups(records, include_transform=False),
        "detection_by_transform": _detection_groups(records, include_transform=True),
        "detection_by_profile": _detection_groups_by_profile(records),
        "attack_transitions": _attack_transition_groups(records),
        "detection_by_source_provider": _detection_groups_by_source_field(records, field="provider"),
        "detection_by_content_stratum": _detection_groups_by_source_field(records, field="content_stratum"),
        "removal": _removal_groups(records),
        "fidelity": _fidelity_groups(records),
        "timing": timing,
        "unstable_cases": unstable_cases,
        "expected_mismatches": [
            {"case_id": case_id, "adapter": adapter, "status": status}
            for case_id, adapter, status in sorted(mismatches)
        ],
        "non_results": [
            {"case_id": case_id, "adapter": adapter, "status": status, "error": error}
            for case_id, adapter, status, error in sorted(non_results)
        ],
    }


def markdown_cell(value: object, *, missing: str = "-") -> str:
    """Render one escaped Markdown table cell."""
    if value is None:
        return missing
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(
    columns: Sequence[tuple[str, str]],
    rows: Sequence[Mapping[str, Any]],
    *,
    missing: str = "-",
) -> list[str]:
    """Render a Markdown table from named columns and mapping rows."""
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(markdown_cell(row.get(key), missing=missing) for key, _ in columns) + " |" for row in rows
    ]
    return [header, separator, *body]


def render_markdown(summary: Mapping[str, Any], *, title: str = "Watermark benchmark report") -> str:
    """Render a standalone report without merging distinct evidence questions."""
    lines = [
        f"# {title}",
        "",
        f"Runs: {summary['runs']}. Observations: {summary['observations']}. Unique cases: {summary['unique_cases']}.",
        "",
        (
            "> `not_detected` means only that the selected adapter did not recognize its signal. "
            "It is not evidence that a watermark is absent. Post-removal rows never certify erasure."
        ),
        "",
        "## Detection by state",
        "",
        *markdown_table(
            (
                ("adapter", "Adapter"),
                ("state", "State"),
                ("unique_artifacts", "Unique artifacts"),
                ("observations", "Observations"),
                ("detected", "Detected"),
                ("not_detected", "Not detected"),
                ("unavailable", "Unavailable"),
                ("error", "Error"),
                ("unstable", "Unstable"),
            ),
            cast("list[dict[str, Any]]", summary["detection"]),
        ),
        "",
        "## Detection by transform",
        "",
        *markdown_table(
            (
                ("adapter", "Adapter"),
                ("state", "State"),
                ("transform", "Transform"),
                ("unique_artifacts", "Unique artifacts"),
                ("detected", "Detected"),
                ("not_detected", "Not detected"),
                ("unavailable", "Unavailable"),
                ("error", "Error"),
                ("unstable", "Unstable"),
            ),
            cast("list[dict[str, Any]]", summary["detection_by_transform"]),
        ),
        "",
    ]
    by_profile = cast("list[dict[str, Any]]", summary["detection_by_profile"])
    if by_profile:
        lines.extend(
            [
                "## Detection by watermark profile",
                "",
                *markdown_table(
                    (
                        ("adapter", "Adapter"),
                        ("state", "State"),
                        ("profile", "Profile"),
                        ("unique_artifacts", "Unique artifacts"),
                        ("detected", "Detected"),
                        ("not_detected", "Not detected"),
                        ("unavailable", "Unavailable"),
                        ("error", "Error"),
                        ("unstable", "Unstable"),
                    ),
                    by_profile,
                ),
                "",
            ]
        )
    attack_transitions = cast("list[dict[str, Any]]", summary["attack_transitions"])
    if attack_transitions:
        lines.extend(
            [
                "## Attack transitions from the matched marked case",
                "",
                *markdown_table(
                    (
                        ("adapter", "Adapter"),
                        ("transform", "Attack transform"),
                        ("pairs", "Pairs"),
                        ("retained", "Detected -> detected"),
                        ("lost", "Detected -> not detected"),
                        ("gained", "Not detected -> detected"),
                        ("still_not_detected", "Not detected -> not detected"),
                        ("unmeasured", "Unmeasured"),
                    ),
                    attack_transitions,
                ),
                "",
                (
                    "A gained detection is a threshold-crossing observation after the transform, not evidence "
                    "that the attack added a watermark."
                ),
                "",
            ]
        )
    source_group_columns = (
        ("adapter", "Adapter"),
        ("state", "State"),
        ("unique_artifacts", "Unique artifacts"),
        ("detected", "Detected"),
        ("not_detected", "Not detected"),
        ("unavailable", "Unavailable"),
        ("error", "Error"),
        ("unstable", "Unstable"),
    )
    by_provider = cast("list[dict[str, Any]]", summary["detection_by_source_provider"])
    if by_provider:
        lines.extend(
            [
                "## Detection by source provider",
                "",
                *markdown_table(
                    (*source_group_columns[:2], ("provider", "Provider"), *source_group_columns[2:]), by_provider
                ),
                "",
            ]
        )
    by_stratum = cast("list[dict[str, Any]]", summary["detection_by_content_stratum"])
    if by_stratum:
        lines.extend(
            [
                "## Detection by content stratum",
                "",
                *markdown_table(
                    (*source_group_columns[:2], ("content_stratum", "Content stratum"), *source_group_columns[2:]),
                    by_stratum,
                ),
                "",
            ]
        )
    lines.extend(["## Post-removal observation", ""])
    removal = cast("list[dict[str, Any]]", summary["removal"])
    if removal:
        lines.extend(
            markdown_table(
                (
                    ("adapter", "Adapter"),
                    ("unique_artifacts", "Unique artifacts"),
                    ("signal_detected_after_removal", "Signal remains"),
                    ("no_recognized_signal_after_removal", "No recognized signal"),
                    ("unmeasured", "Unmeasured"),
                    ("unstable", "Unstable"),
                ),
                removal,
            )
        )
    else:
        lines.append("No removed-state artifacts were present.")

    lines.extend(["", "## Adapter timing", ""])
    timing = cast("list[dict[str, Any]]", summary["timing"])
    if timing:
        lines.extend(
            markdown_table(
                (
                    ("adapter", "Adapter"),
                    ("phase", "Phase"),
                    ("n", "n"),
                    ("p50_ms", "p50 ms"),
                    ("p90_ms", "p90 ms"),
                    ("max_ms", "max ms"),
                ),
                timing,
            )
        )
    else:
        lines.append("No adapter calls carried timing measurements.")

    lines.extend(
        [
            "",
            (
                "The first measured call for each adapter in each input file is cold; subsequent measured calls are "
                "warm. Image decode, fidelity calculation, and process startup are outside the interval."
            ),
            "",
            "## Fidelity",
            "",
            *markdown_table(
                (
                    ("adapter", "Adapter"),
                    ("state", "State"),
                    ("unique_pairs", "Unique pairs"),
                    ("measured", "Measured"),
                    ("identical", "Identical"),
                    ("incomparable", "Incomparable"),
                    ("not_measured", "Not measured"),
                    ("inconsistent", "Inconsistent"),
                    ("finite_psnr", "Finite PSNR"),
                    ("p50_psnr_db", "p50 PSNR dB"),
                    ("finite_mean_psnr", "Finite video mean PSNR"),
                    ("p50_mean_psnr_db", "p50 video mean PSNR dB"),
                    ("finite_snr", "Finite SNR"),
                    ("p50_snr_db", "p50 SNR dB"),
                ),
                cast("list[dict[str, Any]]", summary["fidelity"]),
            ),
            "",
            "## Stability and non-results",
            "",
        ]
    )
    unstable = cast("list[dict[str, Any]]", summary["unstable_cases"])
    mismatches = cast("list[dict[str, Any]]", summary["expected_mismatches"])
    non_results = cast("list[dict[str, Any]]", summary["non_results"])
    lines.append(
        f"Unstable cases: {len(unstable)}. Expected-result mismatches: {len(mismatches)}. "
        f"Errors or unavailable observations: {len(non_results)}."
    )
    if unstable:
        lines.extend(
            [
                "",
                *markdown_table(
                    (("case_id", "Unstable case"), ("adapter", "Adapter"), ("statuses", "Statuses")),
                    [row | {"statuses": ", ".join(row["statuses"])} for row in unstable],
                ),
            ]
        )
    if mismatches:
        lines.extend(
            [
                "",
                *markdown_table(
                    (("case_id", "Expected-result mismatch"), ("adapter", "Adapter"), ("status", "Observed")),
                    mismatches,
                ),
            ]
        )
    if non_results:
        lines.extend(
            [
                "",
                *markdown_table(
                    (
                        ("case_id", "Non-result case"),
                        ("adapter", "Adapter"),
                        ("status", "Status"),
                        ("error", "Error"),
                    ),
                    non_results,
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Inputs",
            "",
            *markdown_table(
                (("path", "Result file"), ("sha256", "SHA-256"), ("observations", "Observations")),
                cast("list[dict[str, Any]]", summary["inputs"]),
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path, report: str) -> None:
    """Write a complete report once, refusing an existing destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(report)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing report: {path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="one or more benchmark result JSONL files")
    parser.add_argument("--output", type=Path, required=True, help="new Markdown report path; never overwritten")
    parser.add_argument("--title", default="Watermark benchmark report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        records = load_results(args.results)
        summary = summarize(records)
        write_report(args.output, render_markdown(summary, title=args.title))
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    log.info("Wrote benchmark report for %s observations to %s", summary["observations"], args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
