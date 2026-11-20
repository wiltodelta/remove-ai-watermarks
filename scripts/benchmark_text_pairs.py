"""Bind blind fidelity reviews and exact detector receipts for a paired text comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)
Side = Literal["marked", "clean"]
SIDES: tuple[Side, ...] = ("marked", "clean")


@dataclass(frozen=True)
class Trial:
    pair: str
    detector: str
    side: Side
    source: str
    outputs: dict[str, str | None]
    retained_prose: dict[str, list[str]]


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_file_digests(root: Path, records: object, required: set[str]) -> None:
    if not isinstance(records, dict) or not required <= records.keys():
        raise ValueError(f"A hash is required for every pinned file in {sorted(required)}.")
    for name, expected in records.items():
        path = Path(name)
        if not path.parts or path.is_absolute() or ".." in path.parts:
            raise ValueError("Pinned paths must stay within their supplied directory.")
        if _file_digest(root / path) != expected:
            raise ValueError(f"Pinned file digest differs for {name}.")


def _distinct_fields(fields: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in fields:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def _read(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_distinct_fields), hashlib.sha256(raw).hexdigest()


def _write(path: Path, value: object) -> str:
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _manifest(path: Path, *, minimum_arms: int = 2) -> tuple[list[str], dict[str, int], list[Trial], str]:
    data, digest = _read(path)
    if not isinstance(data, dict):
        raise ValueError("Expected a manifest object.")
    arms, detectors, pairs = (data.get(field) for field in ("arms", "detectors", "pairs"))
    if (
        not isinstance(arms, list)
        or len(arms) < minimum_arms
        or not all(_nonempty_string(arm) for arm in arms)
        or len(set(arms)) != len(arms)
    ):
        raise ValueError(f"A manifest requires at least {minimum_arms} uniquely named arms.")
    if (
        not isinstance(detectors, dict)
        or not detectors
        or any(not _nonempty_string(name) or type(count) is not int or count < 1 for name, count in detectors.items())
    ):
        raise ValueError("Each detector must specify a positive minimum number of scored positions.")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("A comparison requires marked and clean source pairs.")

    ids: set[str] = set()
    sources: set[tuple[str, str]] = set()
    trials = []
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Each pair must be an object.")
        ident, detector = pair.get("id"), pair.get("detector")
        if not _nonempty_string(ident) or ident in ids:
            raise ValueError("Each pair requires a unique nonempty ID.")
        if not _nonempty_string(detector) or detector not in detectors:
            raise ValueError(f"Pair {ident} requires a declared detector.")
        ids.add(ident)
        for side in SIDES:
            trial = pair.get(side)
            if not isinstance(trial, dict) or not _nonempty_string(trial.get("source")):
                raise ValueError(f"Pair {ident} requires a nonempty {side} source.")
            source = trial["source"]
            source_key = (detector, text_digest(source))
            if source_key in sources:
                raise ValueError(f"Duplicate source for detector {detector} in pair {ident}.")
            sources.add(source_key)
            outputs = trial.get("outputs")
            if not isinstance(outputs, dict) or set(outputs) != set(arms):
                raise ValueError(f"Pair {ident} {side} requires one output or refusal for every arm.")
            if any(value is not None and not _nonempty_string(value) for value in outputs.values()):
                raise ValueError(f"Pair {ident} {side} outputs must be nonempty strings or null refusals.")
            retained = trial.get("retained_prose", {})
            if not isinstance(retained, dict) or retained.keys() - outputs.keys():
                raise ValueError(f"Pair {ident} {side} retained prose must name declared arms.")
            for arm, fragments in retained.items():
                candidate = outputs[arm]
                if (
                    candidate is None
                    or candidate == source
                    or not isinstance(fragments, list)
                    or not fragments
                    or any(
                        not _nonempty_string(item) or item not in source or item not in candidate for item in fragments
                    )
                ):
                    raise ValueError(f"Pair {ident} {side} {arm} requires exact retained prose in both texts.")
            trials.append(Trial(ident, detector, side, source, outputs, retained))
    return arms, detectors, trials, digest


def _review_id(source: str, candidate: str) -> str:
    return text_digest(json.dumps([source, candidate], ensure_ascii=False))


def _blind_pairs(trials: list[Trial]) -> list[dict[str, str]]:
    pairs = {
        _review_id(trial.source, candidate): {"source": trial.source, "candidate": candidate}
        for trial in trials
        if trial.side == "marked"
        for candidate in trial.outputs.values()
        if candidate is not None and candidate != trial.source
    }
    return [{"id": ident, **pair} for ident, pair in sorted(pairs.items())]


def bind(manifest: Path, run: Path) -> None:
    """Write a new blind packet without method or detector information."""
    _, _, trials, digest = _manifest(manifest)
    run.mkdir()
    packet = _blind_pairs(trials)
    packet_digest = _write(run / "blind-pairs.json", packet)
    _write(run / "input-lock.json", {"manifest_sha256": digest, "packet_sha256": packet_digest})
    log.info("Bound %s distinct marked source/candidate pairs.", len(packet))


def _check_reviews(run: Path, reviews: Path) -> tuple[dict[str, Any], str, str]:
    binding, _ = _read(run / "input-lock.json")
    packet, digest = _read(run / "blind-pairs.json")
    if digest != binding["packet_sha256"]:
        raise ValueError("Blind packet has changed since binding.")
    labels, review_digest = _read(reviews)
    if not isinstance(labels, dict) or labels.keys() != {pair["id"] for pair in packet}:
        raise ValueError("Reviews must label each blind pair exactly once.")
    if any(
        not isinstance(label, dict)
        or type(label.get("faithful")) is not bool
        or not _nonempty_string(label.get("reason"))
        for label in labels.values()
    ):
        raise ValueError("Each review requires a boolean faithful verdict and a nonempty reason.")
    return labels, review_digest, digest


def seal(run: Path, reviews: Path) -> None:
    """Freeze the reviewed packet before opening detector receipts."""
    _, review_digest, packet_digest = _check_reviews(run, reviews)
    _write(run / "review-lock.json", {"review_sha256": review_digest, "packet_sha256": packet_digest})


def _trial_values(trial: Trial) -> list[str]:
    return [
        value
        for value in (
            trial.source,
            *trial.outputs.values(),
            *(fragment for fragments in trial.retained_prose.values() for fragment in fragments),
        )
        if value is not None
    ]


def _observations(path: Path, trials: list[Trial]) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    receipts, digest = _read(path)
    if not isinstance(receipts, list):
        raise ValueError("Expected a list of detector receipts.")
    expected = {(trial.detector, text_digest(value)) for trial in trials for value in _trial_values(trial)}
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for row in receipts:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("detector"), str)
            or not isinstance(row.get("sha256"), str)
            or type(row.get("detected")) is not bool
            or type(row.get("positions")) is not int
            or row["positions"] < 0
        ):
            raise ValueError("Each detector receipt requires detector, sha256, detected and positions.")
        key = (row["detector"], row["sha256"])
        if key not in expected or key in observations:
            raise ValueError("Unexpected or duplicate detector receipt.")
        observations[key] = row
    if observations.keys() != expected:
        raise ValueError(f"Missing {len(expected - observations.keys())} detector receipts.")
    return observations, digest


def _paired(
    trials: list[Trial], arms: list[str], baseline: str, joint: dict[tuple[str, str], bool]
) -> list[dict[str, Any]]:
    by_detector: dict[str, list[str]] = defaultdict(list)
    for trial in trials:
        if trial.side == "marked":
            by_detector[trial.detector].append(trial.pair)
    result = []
    for detector, pairs in sorted(by_detector.items()):
        for arm in arms:
            if arm == baseline:
                continue
            outcomes = [(joint[pair, arm], joint[pair, baseline]) for pair in pairs]
            wins = outcomes.count((True, False))
            losses = outcomes.count((False, True))
            discordant = wins + losses
            lower_tail = sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
            result.append(
                {
                    "detector": detector,
                    "arm": arm,
                    "baseline": baseline,
                    "n": len(pairs),
                    "wins": wins,
                    "losses": losses,
                    "p_value": min(1.0, 2 * lower_tail / 2**discordant),
                }
            )
    return result


def _sealed_reviews(
    run: Path, reviews: Path, trials: list[Trial], manifest_digest: str
) -> tuple[dict[str, Any], str, str]:
    binding, _ = _read(run / "input-lock.json")
    if binding["manifest_sha256"] != manifest_digest:
        raise ValueError("Manifest has changed since binding.")
    labels, review_digest, packet_digest = _check_reviews(run, reviews)
    lock, _ = _read(run / "review-lock.json")
    if lock != {"review_sha256": review_digest, "packet_sha256": packet_digest}:
        raise ValueError("Reviews have changed since sealing.")
    packet, _ = _read(run / "blind-pairs.json")
    if packet != _blind_pairs(trials):
        raise ValueError("Blind packet does not match the manifest.")
    return labels, review_digest, packet_digest


def score(manifest: Path, run: Path, reviews: Path, detections: Path, *, baseline: str) -> dict[str, Any]:
    """Count all attempts, including unchanged results and evaluator refusals."""
    arms, minimum, trials, manifest_digest = _manifest(manifest)
    if baseline not in arms:
        raise ValueError("Baseline must name a declared arm.")
    labels, review_digest, packet_digest = _sealed_reviews(run, reviews, trials, manifest_digest)
    observations, detector_digest = _observations(detections, trials)

    for trial in trials:
        control = observations[trial.detector, text_digest(trial.source)]
        if control["positions"] < minimum[trial.detector] or control["detected"] != (trial.side == "marked"):
            raise ValueError(f"Pair {trial.pair} has an unqualified {trial.side} source.")

    group_fields = ("n", "refused", "unchanged", "returned", "insufficient_positions", "detector_positive", "clear")
    groups: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    joint: dict[tuple[str, str], bool] = {}
    results = []
    for trial in trials:
        for arm, candidate in trial.outputs.items():
            group = groups[trial.detector, trial.side, arm]
            group["n"] += 1
            result: dict[str, Any] = {
                "pair": trial.pair,
                "detector": trial.detector,
                "side": trial.side,
                "arm": arm,
                "source_sha256": text_digest(trial.source),
            }
            if candidate is None:
                status = "refused"
            elif candidate == trial.source:
                status = "unchanged"
            else:
                status = "returned"
                observation = observations[trial.detector, text_digest(candidate)]
                enough = observation["positions"] >= minimum[trial.detector]
                clear = enough and not observation["detected"]
                group["clear"] += clear
                group["detector_positive"] += enough and observation["detected"]
                group["insufficient_positions"] += not enough
                result.update(candidate=observation, clear=clear)
                fragments = trial.retained_prose.get(arm, [])
                retained = [observations[trial.detector, text_digest(fragment)] for fragment in fragments]
                partial_unverified = any(
                    row["detected"] or row["positions"] < minimum[trial.detector] for row in retained
                )
                if retained:
                    group["partial"] += 1
                    group["partial_unverified"] += partial_unverified
                    result.update(retained_prose=retained, partial_unverified=partial_unverified)
                if trial.side == "marked":
                    ident = _review_id(trial.source, candidate)
                    faithful = labels[ident]["faithful"]
                    group["faithful"] += faithful
                    result.update(
                        review_id=ident, faithful=faithful, joint=faithful and clear and not partial_unverified
                    )
            group[status] += 1
            result["status"] = status
            if trial.side == "marked":
                joint[trial.pair, arm] = result.setdefault("joint", False)
                group["joint"] += joint[trial.pair, arm]
            results.append(result)
    return {
        "input_sha256": manifest_digest,
        "packet_sha256": packet_digest,
        "review_sha256": review_digest,
        "detector_receipts_sha256": detector_digest,
        "groups": [
            {
                "detector": detector,
                "side": side,
                "arm": arm,
                **{name: values[name] for name in group_fields},
                **({name: values[name] for name in ("partial", "partial_unverified")} if values["partial"] else {}),
                **({"faithful": values["faithful"], "joint": values["joint"]} if side == "marked" else {}),
            }
            for (detector, side, arm), values in sorted(groups.items())
        ],
        "paired": _paired(trials, arms, baseline, joint),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("bind", "seal", "score"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        if name != "seal":
            command.add_argument("--input", type=Path, required=True)
        if name != "bind":
            command.add_argument("--reviews", type=Path, required=True)
        if name == "score":
            command.add_argument("--detections", type=Path, required=True)
            command.add_argument("--baseline", required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "bind":
            bind(args.input, args.run)
        elif args.command == "seal":
            seal(args.run, args.reviews)
        else:
            _write(args.output, score(args.input, args.run, args.reviews, args.detections, baseline=args.baseline))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
