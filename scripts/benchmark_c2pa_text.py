"""Replay published C2PA text vectors and check exact restoration with an independent decoder."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)


def load_source(name: str, source: Path) -> ModuleType:
    path = source.resolve(strict=True)
    sys.path.insert(0, str(path))
    module = importlib.import_module(name)
    if not Path(module.__file__).resolve().is_relative_to(path):
        raise RuntimeError(f"Loaded {name} outside the specified source tree")
    return module


def source_hashes(module: ModuleType) -> dict[str, str]:
    source = Path(module.__file__).resolve().parent
    return {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source.rglob("*.py"))
    }


def structured_residue(reference: ModuleType, text: str) -> bool:
    try:
        reference.extract_structured(text)
    except reference.StructuredError as exc:
        if exc.code != "manifest.structuredText.noManifest":
            raise
        return False
    return True


def make_cases(reference: ModuleType, vectors: dict[str, Any]) -> list[dict[str, Any]]:
    families: dict[str, tuple[str, str, Callable[[str], bool]]] = {
        "unstructured_embed": (
            "text",
            "expected_embed_hex",
            lambda text: reference.extract_manifest(text)[0] is not None,
        ),
        "structured_embed": ("text", "expected_text_hex", lambda text: structured_residue(reference, text)),
        "html_inline": ("html", "expected_html_hex", lambda text: reference.extract_html(text) is not None),
        "html_reference": ("html", "expected_html_hex", lambda text: reference.extract_html(text) is not None),
    }
    cases = []
    for family, (body, encoded, detector) in families.items():
        for row in vectors[family]:
            clean = row[body]
            marked = bytes.fromhex(row[encoded]).decode()
            if marked == clean or not detector(marked) or detector(clean):
                raise ValueError(f"Invalid published control: {family}/{row['name']}")
            cases.append({"case": row["name"], "family": family, "clean": clean, "marked": marked, "residue": detector})

    begin, end = "-----BEGIN C2PA MANIFEST-----", "-----END C2PA MANIFEST-----"
    for label, newline in (("lf", "\n"), ("crlf", "\r\n")):
        example = f"<!-- {begin} https://example.com/quoted.c2pa {end} -->"
        clean = newline.join(("C2PA example", "", "~~~markdown", example, "~~~", ""))
        marked = f"# {begin} https://example.com/inserted.c2pa {end}" + newline + clean
        cases.append({"case": f"fenced-example-{label}", "family": "literal_example", "clean": clean, "marked": marked})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True, help="Checkout of encypherai/c2pa-text")
    parser.add_argument("--dewatermark-root", type=Path, help="Optional checkout of cyzanfar/text-watermark-remover")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report file")
    args = parser.parse_args()

    reference = load_source("c2pa_text", args.reference_root / "python/src")
    vector_bytes = (args.reference_root / "golden/vectors.json").read_bytes()
    sys.path.insert(0, str(ROOT / "src"))
    from remove_ai_watermarks import text as library

    cleaners: dict[str, Callable[[str], str]] = {
        "library": lambda text: library.remove_text_metadata(text).cleaned_text,
    }
    competitor = None
    if args.dewatermark_root:
        competitor = load_source("dewatermark", args.dewatermark_root / "src")
        cleaners["dewatermark_safe"] = competitor.sanitize
        cleaners["dewatermark_aggressive"] = lambda text: competitor.sanitize(text, profile="aggressive")

    results = []
    for case in make_cases(reference, json.loads(vector_bytes)):
        clean, marked = case["clean"], case["marked"]
        detector = case.get("residue")
        for method, transform in cleaners.items():
            result = transform(marked)
            results.append(
                {
                    "case": case["case"],
                    "family": case["family"],
                    "method": method,
                    "marked_sha256": hashlib.sha256(marked.encode()).hexdigest(),
                    "expected_sha256": hashlib.sha256(clean.encode()).hexdigest(),
                    "result_sha256": hashlib.sha256(result.encode()).hexdigest(),
                    "exact_recovery": result == clean,
                    "clean_preserved": transform(clean) == clean,
                    "idempotent": transform(result) == result,
                    "reference_residue": detector(result) if detector is not None else None,
                }
            )

    report = {
        "vector_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_source_sha256": source_hashes(reference),
        "dewatermark_source_sha256": source_hashes(competitor) if competitor is not None else None,
        "library_sha256": hashlib.sha256(Path(library.__file__).read_bytes()).hexdigest(),
        "dewatermark_version": competitor.__version__ if competitor is not None else None,
        "results": results,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    log.info("Wrote %d method-case results to %s", len(results), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
