"""Replay frozen synthetic text with the external author's AliMark detector.

The detector's score is not a provider verdict. A caller-declared threshold does
not establish a calibrated false-positive rate or semantic quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger(__name__)


def assess_scores(
    before: float,
    baseline: float,
    after: float | None,
    *,
    threshold: float | None,
    retained_prose_spans: int = 0,
) -> str:
    """Keep numerical replay, controls and partial-output evidence distinct."""
    values = [before, baseline, *([] if after is None else [after]), *([] if threshold is None else [threshold])]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("AliMark scores and threshold must be finite")
    if retained_prose_spans < 0:
        raise ValueError("retained_prose_spans must be nonnegative")
    if threshold is None:
        return "scored_uncalibrated"
    if before < threshold:
        return "invalid_positive_control"
    if baseline >= threshold:
        return "invalid_negative_control"
    if after is None:
        return "no_after_output"
    if after >= threshold:
        return "retained"
    return "partial_unverified" if retained_prose_spans else "below_declared_threshold"


def load_inputs(directory: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """Check exact UTF-8 bytes against a required frozen input manifest."""
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_bytes())
    hashes: dict[str, str] = manifest["inputs_sha256"]
    if set(hashes) not in ({"source", "baseline"}, {"source", "baseline", "after"}):
        raise ValueError("Manifest must declare source, baseline and optional after hashes")
    retained = manifest.get("retained_prose_spans", 0)
    if type(retained) is not int or retained < 0:
        raise ValueError("retained_prose_spans must be a nonnegative integer")
    if not isinstance(manifest.get("generation"), dict) or not manifest["generation"]:
        raise ValueError("Manifest must declare generation provenance")
    texts: dict[str, str] = {}
    for name, expected_hash in hashes.items():
        raw = (directory / f"{name}.txt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Frozen {name} hash mismatch")
        text = raw.decode("utf-8")
        if not text.strip():
            raise ValueError(f"Frozen {name} text is empty")
        texts[name] = text
    if (directory / "after.txt").exists() and "after" not in hashes:
        raise ValueError("An after file must be included in the frozen hash manifest")
    return texts, manifest


def _unavailable_vllm(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError("This replay supports author detection only, not vLLM generation")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run author detection without replacing the encoder or alignment logic."""
    texts, manifest = load_inputs(args.input_dir)
    embedder = Path(args.embedder).resolve()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    if args.dependency_dir is not None:
        sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference_root.resolve()))
    os.environ["NLTK_DATA"] = str(args.nltk_data.resolve())
    nltk: Any = importlib.import_module("nltk")
    nltk.data.path.insert(0, str(args.nltk_data.resolve()))
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab/english"):
        nltk.data.find(resource)
    original_download = nltk.download

    def cached_download(resource: Any, *_args: Any, **_kwargs: Any) -> bool:
        # Author import requests punkt; the resources were checked above offline.
        if resource != "punkt" and resource != ["punkt"]:
            raise RuntimeError("Unexpected author NLTK download request")
        return True

    # Author imports vLLM even for load_llm=False. The sentinel fails if invoked.
    original_vllm = sys.modules.get("vllm")
    sentinel: Any = types.ModuleType("vllm")
    sentinel.LLM = _unavailable_vllm
    sentinel.SamplingParams = _unavailable_vllm
    if original_vllm is None:
        sys.modules["vllm"] = sentinel
    nltk.download = cached_download
    previous_cwd = Path.cwd()
    reference_root = args.reference_root.resolve()
    try:
        os.chdir(reference_root)
        author: Any = importlib.import_module("watermark.alimark")
        config = argparse.Namespace(
            watermark_model="not_loaded_for_replay",
            watermark_embedder=str(embedder),
            watermark_num_next_sentence_candidates=0,
            min_new_sentences=0,
            device=args.device,
            watermark_rs_dropout=0.0,
            seed=42,
            watermark_block_size=args.block_size,
            watermark_embedding_dim=768,
        )
        detector: Any = author.AliMark(config, load_llm=False)
        scores = {name: detector.detect_watermark(text) for name, text in texts.items()}
    finally:
        os.chdir(previous_cwd)
        nltk.download = original_download
        if original_vllm is None:
            sys.modules.pop("vllm", None)
    status = assess_scores(
        float(scores["source"]["score"]),
        float(scores["baseline"]["score"]),
        None if "after" not in scores else float(scores["after"]["score"]),
        threshold=args.threshold,
        retained_prose_spans=manifest.get("retained_prose_spans", 0),
    )
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git is required to record the reference revision")
    revision = subprocess.check_output(  # noqa: S603 - fixed git subcommand, no shell
        [git, "-C", str(reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    reference_hashes = {
        str(path.relative_to(reference_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (reference_root / "watermark").rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".pkl"}
    }
    report = {
        "scheme": "AliMark author detector replay",
        "reference_revision": revision,
        "reference_hashes": reference_hashes,
        "generation_manifest": manifest,
        "block_size": args.block_size,
        "embedder_path": str(embedder),
        "embedder_files_sha256": {
            str(path.relative_to(embedder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in embedder.rglob("*")
            if path.is_file() and path.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"}
        },
        "encoder_device": args.device,
        "threshold": args.threshold,
        "status": status,
        "scores": scores,
        "notice": (
            "Numerical author-reference evidence only. No calibrated FPR, semantic certification "
            "or native-provider compatibility claim."
        ),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    log.info("AliMark replay status: %s", status)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedder", type=Path, required=True, help="Local author-compatible 768-dimensional encoder")
    parser.add_argument("--nltk-data", type=Path, required=True, help="Prepared punkt and English punkt_tab resources")
    parser.add_argument("--block-size", type=int, choices=(1, 2, 4, 8, 16), default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threshold", type=float, help="Explicit experimental threshold; no default calibrated cutoff")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
