"""Replay frozen text with the DIPPER paper's published NumPy KGW detector.

This is a specific public-code variant, not MarkLLM, a private-provider detector,
DIPPER inference, or a claim of meaning preservation. No model weights are loaded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import math
import sys
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
REFERENCE_REVISION = "95f3e2cb5e239929a1fc4bed26bf93f2c368da31"
REFERENCE_SHA256 = "cb4d8f52b60ea977df30a2900c8cb7882ae3061f0f13f227e967d9dc7bdef11a"
VOCAB_SHA256 = "e35d8b86ebd35ebd260d040aa455e09759f7e675f4dbb7f3d727516f27eca190"
TOKENIZATION_SHA256 = "8b744ed78270f4b70552713cda283db61f17065505f4fc4d994f520cd5a5bd8a"
FUNCTION_NAMES = {"watermark_detect", "hash_fn", "get_z", "get_green_list"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_contract(manifest: dict[str, Any], context: bytes) -> None:
    """Bind the prefix and detector settings to the frozen input provenance."""
    generation = manifest["generation"]
    expected = {
        "detector_variant": "dipper_paper_numpy",
        "reference_revision": REFERENCE_REVISION,
        "reference_utils_sha256": REFERENCE_SHA256,
        "vocab_sha256": VOCAB_SHA256,
        "tokenization_sha256": TOKENIZATION_SHA256,
        "watermark_fraction": 0.5,
        "vocab_size": 50257,
        "context_sha256": hashlib.sha256(context).hexdigest(),
    }
    if not context.decode("utf-8").strip():
        raise ValueError("Detection context must be nonempty UTF-8 text")
    for key, value in expected.items():
        if type(generation.get(key)) is not type(value) or generation[key] != value:
            raise ValueError(f"DIPPER reference contract mismatch: {key}")


def load_reference(path: Path) -> dict[str, Any]:
    """Extract unchanged pinned numerical functions without running API imports."""
    if sys.byteorder != "little":
        raise ValueError("DIPPER reference replay requires the validated little-endian seed encoding")
    if sha(path) != REFERENCE_SHA256:
        raise ValueError("DIPPER reference utils hash mismatch")
    import numpy as np

    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES]
    if len(functions) != len(FUNCTION_NAMES) or {node.name for node in functions} != FUNCTION_NAMES:
        raise ValueError("DIPPER reference function inventory mismatch")
    with tempfile.TemporaryDirectory(prefix="dipper-reference-") as directory:
        extracted = Path(directory) / "numerical_reference.py"
        extracted.write_text(ast.unparse(ast.Module(body=list[ast.stmt](functions), type_ignores=[])), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("dipper_numerical_reference", extracted)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load extracted DIPPER numerical functions")
        module = importlib.util.module_from_spec(spec)
        module.__dict__.update(np=np, hashlib=hashlib)
        spec.loader.exec_module(module)
    return vars(module)


def validate_tokenizer(tokenizer: Any) -> None:
    """Reject same-size vocabularies with different token IDs."""
    digest = hashlib.sha256(
        json.dumps(tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if tokenizer.vocab_size != 50257 or digest != VOCAB_SHA256:
        raise ValueError("DIPPER reference requires the pinned GPT-2 token mapping")
    structure = json.loads(tokenizer.backend_tokenizer.to_str())
    structure_hash = hashlib.sha256(json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if structure_hash != TOKENIZATION_SHA256:
        raise ValueError("DIPPER reference tokenization structure mismatch")


def score_texts(texts: dict[str, str], context: str, tokenizer: Any, reference: dict[str, Any]) -> dict[str, Any]:
    """Score every full completion with the original prefix's last token."""
    context_ids = tokenizer(context, add_special_tokens=True)["input_ids"]
    if not context_ids:
        raise ValueError("DIPPER context tokenization is empty")
    scores: dict[str, Any] = {}
    for name, text in texts.items():
        token_ids = [context_ids[-1], *tokenizer(text, add_special_tokens=False)["input_ids"]]
        if len(token_ids) < 2:
            raise ValueError(f"DIPPER {name} contains no completion tokens")
        if any(type(value) is not int or not 0 <= value < 50257 for value in token_ids):
            raise ValueError(f"DIPPER {name} contains invalid token IDs")
        raw, _ = reference["watermark_detect"](token_ids, {}, 0.5, 50257)
        if isinstance(raw, bool) or not isinstance(raw, Real) or not math.isfinite(raw):
            raise ValueError(f"DIPPER {name} returned an invalid score")
        scores[name] = {"z": float(raw), "scored_transitions": len(token_ids) - 1}
    return scores


def assess(scores: dict[str, Any], retained: int) -> str:
    """Use the author's strict z>4 diagnostic while keeping controls separate."""
    if scores["source"]["z"] <= 4:
        return "invalid_positive_control"
    if scores["baseline"]["z"] > 4:
        return "invalid_negative_control"
    if "after" not in scores:
        return "no_after_output"
    if scores["after"]["z"] > 4:
        return "retained"
    return "partial_unverified" if retained else "not_detected_at_reference_threshold"


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.benchmark_alimark_reference import load_inputs

    texts, manifest = load_inputs(args.input_dir)
    context = (args.input_dir / "context.txt").read_bytes()
    validate_contract(manifest, context)
    reference = load_reference(args.reference_utils)
    import transformers

    tokenizer_class: Any = transformers.AutoTokenizer
    tokenizer: Any = tokenizer_class.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    validate_tokenizer(tokenizer)
    scores = score_texts(texts, context.decode("utf-8"), tokenizer, reference)
    report = {
        "scheme": "DIPPER-paper KGW-family NumPy-shuffle variant",
        "scope": "Full completions, not the paper's matched-length minimum-200-token ROC experiment.",
        "limitation": "Diagnostic research result, no calibrated production FPR or semantic certificate.",
        "manifest": manifest,
        "scores": scores,
        "threshold": 4.0,
        "status": assess(scores, manifest.get("retained_prose_spans", 0)),
        "reference_revision": REFERENCE_REVISION,
        "reference_utils_sha256": sha(args.reference_utils),
        "adapter_sha256": sha(Path(__file__)),
        "versions": {name: __import__(name).__version__ for name in ("numpy", "transformers")},
        "byteorder": sys.byteorder,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-utils", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    report = run(args)
    log.info("DIPPER reference replay: %s", report["status"])


if __name__ == "__main__":
    main()
