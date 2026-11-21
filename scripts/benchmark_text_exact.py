"""Benchmark exact UTF-8 recovery and clean-control preservation through API and CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import benchmark_c2pa_text as c2pa  # noqa: E402

log = logging.getLogger(__name__)
WINDOWS_AI_MARK = "\u200b\u034f\u2060\u200b"
UNICODE_CONTROLS = {
    "emoji-zwj": "A tester uses 👨🏽\u200d🍳 and ❤️.",
    "emoji-keycap": "The room number is 4\ufe0f\u20e3.",
    "decomposed-latin": "The café sign reads cafe\u0301.",
    "thai-word-boundary": "พบกัน\u200bวันพรุ่งนี้",
    "persian-joiner": "گزارش می\u200cرسد.",
    "indic-joiner": "क्\u200dषेत्र की रिपोर्ट",
    "hebrew-isolate": "Label: \u2067של\u05d5ם\u2069.",
    "arabic-isolate": "Label: \u2067مرحب\u0627\u2069.",
    "metric-spacing": "The measured length was 12\u202fmm.",
    "nonbreaking-space": "Keep these two\u00a0words together.",
    "math-joiner": "Write 8\u2060\u00d7\u20605 on the first line.",
    "soft-hyphen": "We will re\u00adcheck the measurement.",
    "technical-and-crlf": 'URL: https://example.invalid/a?count=1.\r\n\r\n```txt\r\nname="cafe\u0301"\r\n```\r\n',
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unicode_cases() -> list[dict[str, str]]:
    return [
        {
            "case": name,
            "family": "unicode",
            "clean": "Test entry. " + body,
            "marked": "Test " + WINDOWS_AI_MARK + "entry. " + body,
        }
        for name, body in UNICODE_CONTROLS.items()
    ]


def _measure(case: dict[str, Any], root: Path) -> dict[str, Any]:
    from click.testing import CliRunner

    from remove_ai_watermarks import text
    from remove_ai_watermarks.cli import main

    root.mkdir()
    suffix = ".html" if case["family"].startswith("html") else ".md"
    runner = CliRunner()
    results = {}
    outputs = {}
    clean = case["clean"].encode("utf-8")
    for arm, value in (("recovery", case["marked"]), ("clean_control", case["clean"])):
        source, destination = root / ("source-" + arm + suffix), root / ("result-" + arm + suffix)
        original = value.encode("utf-8")
        source.write_bytes(original)
        api = text.remove_text_all(value).cleaned_text.encode("utf-8")
        cli = runner.invoke(main, ["all", str(source), "-o", str(destination)])
        written = destination.read_bytes() if cli.exit_code == 0 else None
        outputs[arm] = api
        results[arm] = {
            "input_sha256": _digest(original),
            "api_output_sha256": _digest(api),
            "cli_output_sha256": _digest(written) if written is not None else None,
            "api_exact": api == clean,
            "cli_exact": written == clean,
            "cli_exit": cli.exit_code,
            "source_preserved": source.read_bytes() == original,
        }
    restored = outputs["recovery"].decode("utf-8")
    detector = case.get("residue")
    return {
        "case": case["case"],
        "family": case["family"],
        "expected_sha256": _digest(clean),
        "outputs": results,
        "idempotent": text.remove_text_all(restored).cleaned_text == restored,
        "reference_residue": detector(restored) if detector is not None else None,
    }


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, help="Optional checkout of encypherai/c2pa-text")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.output.exists():
        parser.error("output report already exists")
    from remove_ai_watermarks import cli, text

    groups: dict[str, list[dict[str, Any]]] = {"unicode": _unicode_cases()}
    provenance: dict[str, Any] = {
        "benchmark_sha256": _digest(Path(__file__).read_bytes()),
        "library_sha256": {m.__name__: _digest(Path(m.__file__).read_bytes()) for m in (text, cli)},
    }
    if args.reference_root is not None:
        reference = c2pa.load_source("c2pa_text", args.reference_root / "python/src")
        vectors = (args.reference_root / "golden/vectors.json").read_bytes()
        groups["serialized"] = c2pa.make_cases(reference, json.loads(vectors))
        provenance.update(vector_sha256=_digest(vectors), reference_source_sha256=c2pa.source_hashes(reference))

    with TemporaryDirectory(prefix="raiw-exact-text-") as temp:
        outcomes = {
            name: [_measure(case, Path(temp) / f"{name}-{index}") for index, case in enumerate(cases)]
            for name, cases in groups.items()
        }
    summary = {}
    failed = False
    for name, rows in outcomes.items():
        counts = {
            arm: sum(
                row["outputs"][arm]["api_exact"]
                and row["outputs"][arm]["cli_exact"]
                and row["outputs"][arm]["source_preserved"]
                for row in rows
            )
            for arm in ("recovery", "clean_control")
        }
        residue = sum(row["reference_residue"] is True for row in rows)
        idempotent = sum(row["idempotent"] for row in rows)
        failed |= counts != dict.fromkeys(counts, len(rows)) or residue > 0 or idempotent != len(rows)
        summary[name] = {"cases": len(rows), **counts, "idempotent": idempotent, "reference_residue": residue}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump({"provenance": provenance, "summary": summary, "results": outcomes}, stream, indent=2)
        stream.write("\n")
    log.info("Exact recovery and control results\n%s", json.dumps(summary, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
