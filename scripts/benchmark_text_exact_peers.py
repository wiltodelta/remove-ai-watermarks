"""Compare exact text recovery against pinned peer sanitizers on public controls."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_text_exact as exact  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

log = logging.getLogger(__name__)
REVISIONS = {
    "c2pa-text": "7e818755818281748c41aef8766037b30ce9fb04",
    "dewatermark": "fd620a93b9d96c7a4730d80be2d7aa90d5017575",
    "no-watermark": "627e77889e051ebbbde1cad050f88cef41fff996",
    "wm-hook": "90f4e28c8ebd37427297b5f5c47ed820a96f0861",
}


def _require_checkout(root: Path, revision: str) -> None:
    """Reject revisions and source changes outside the pinned comparison."""
    git = ["git", "-C", str(root)]
    actual = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()  # noqa: S603
    status = subprocess.check_output([*git, "status", "--porcelain"], text=True)  # noqa: S603
    if actual != revision or status:
        raise ValueError(f"Expected a clean {root.name} checkout at {revision}")


def _unicode_cases() -> list[dict[str, str]]:
    bodies = dict(exact.UNICODE_CONTROLS)
    bodies.update(
        {
            "cjk-ivs": "The cited glyph is 葛\U000e0100, and the reading is recorded.",
            "persian-multiple-joiners": "لطفاً می\u200cروم و دوباره برمی\u200cگردم.",
            "markdown-inline": "Preserve `cafe\u0301 --help` and [the reference](https://example.invalid).",
        }
    )
    mark = exact.WINDOWS_AI_MARK
    cases = []
    for name, body in bodies.items():
        middle = len(body) // 2
        for position, marked in (
            ("start", mark + body),
            ("middle", body[:middle] + mark + body[middle:]),
            ("end", body + mark),
            ("repeated", mark + body + mark),
        ):
            cases.append({"case": f"{name}-{position}", "family": "unicode", "clean": body, "marked": marked})
    return cases


def _peer_cleaners(root: Path) -> dict[str, Callable[[str], str]]:
    load = exact.c2pa.load_source
    dewatermark = load("dewatermark", root / "dewatermark/src")
    nowatermark = load("nowatermark.remover", root / "no-watermark/src")
    wm_hook = load("wm_hook.cli", root / "wm-hook/src")
    return {
        "dewatermark-safe": dewatermark.sanitize,
        "dewatermark-aggressive": lambda value: dewatermark.sanitize(value, profile="aggressive"),
        "no-watermark-default": lambda value: nowatermark.clean(value)[0],
        "wm-hook-default": lambda value: wm_hook.clean_text(value)[0],
    }


def _peer_result(case: dict[str, Any], cleaner: Callable[[str], str]) -> dict[str, Any]:
    recovered = cleaner(case["marked"])
    clean_output = cleaner(case["clean"])
    detector = case.get("residue")
    return {
        "recovered": recovered == case["clean"],
        "clean_preserved": clean_output == case["clean"],
        "recovered_sha256": exact._digest(recovered.encode()),
        "clean_output_sha256": exact._digest(clean_output.encode()),
        "idempotent": cleaner(recovered) == recovered,
        "reference_residue": detector(recovered) if detector is not None else None,
    }


def _summary(rows: list[dict[str, Any]], peers: dict[str, Callable[[str], str]]) -> dict[str, Any]:
    library = [row["library"] for row in rows]
    summary = {
        "library": {
            "recovered": sum(
                row["outputs"]["recovery"]["api_exact"] and row["outputs"]["recovery"]["cli_exact"] for row in library
            ),
            "clean_preserved": sum(
                row["outputs"]["clean_control"]["api_exact"] and row["outputs"]["clean_control"]["cli_exact"]
                for row in library
            ),
            "idempotent": sum(row["idempotent"] for row in library),
            "source_preserved": sum(
                all(output["source_preserved"] for output in row["outputs"].values()) for row in library
            ),
            "reference_residue": sum(row["reference_residue"] is True for row in library),
        }
    }
    for peer in peers:
        summary[peer] = {
            field: sum(row["peers"][peer][field] is True for row in rows)
            for field in ("recovered", "clean_preserved", "idempotent", "reference_residue")
        }
    return {"cases": len(rows), "methods": summary}


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkouts", type=Path, required=True, help="Directory containing the four pinned Git checkouts"
    )
    parser.add_argument("--output", type=Path, required=True, help="New JSON report")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output report already exists")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from remove_ai_watermarks import cli, text

    root = args.checkouts.resolve()
    for name, revision in REVISIONS.items():
        _require_checkout(root / name, revision)
    reference = exact.c2pa.load_source("c2pa_text", root / "c2pa-text/python/src")
    vectors = (root / "c2pa-text/golden/vectors.json").read_bytes()
    groups = {
        "unicode": _unicode_cases(),
        "serialized": exact.c2pa.make_cases(reference, json.loads(vectors)),
    }
    peers = _peer_cleaners(root)
    results = {}
    with TemporaryDirectory(prefix="raiw-exact-text-peers-") as scratch:
        for name, cases in groups.items():
            results[name] = [
                {
                    "library": exact._measure(case, Path(scratch) / f"{name}-{index}"),
                    "peers": {peer: _peer_result(case, cleaner) for peer, cleaner in peers.items()},
                }
                for index, case in enumerate(cases)
            ]
    summary = {group: _summary(rows, peers) for group, rows in results.items()}
    report = {
        "provenance": {
            "revisions": REVISIONS,
            "c2pa_vector_sha256": exact._digest(vectors),
            "python": sys.version.split()[0],
            "library_sha256": {
                module.__name__: exact._digest(Path(module.__file__).read_bytes()) for module in (cli, text)
            },
            "benchmark_sha256": {
                path.name: exact._digest(path.read_bytes())
                for path in map(Path, (__file__, exact.__file__, exact.c2pa.__file__))
            },
        },
        "summary": summary,
        "results": results,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    log.info("Exact peer comparison\n%s", json.dumps(summary, indent=2))
    exact_fields = ("recovered", "clean_preserved", "idempotent", "source_preserved")
    for group in summary.values():
        library = group["methods"]["library"]
        if library["reference_residue"] or any(library[field] != group["cases"] for field in exact_fields):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
