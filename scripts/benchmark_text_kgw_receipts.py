"""Score sealed text trials with a pinned local KGW reference and tokenizer."""

from __future__ import annotations

import argparse
import importlib
import logging
import math
import sys
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_text_pairs as paired  # noqa: E402

if TYPE_CHECKING:
    from types import ModuleType

log = logging.getLogger(__name__)
REFERENCE_FILES = {"normalizers.py", "watermark_processor.py"}
TOKENIZER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}


def _settings(config: dict[str, Any]) -> tuple[str, int, float, float, float]:
    detector, key, gamma, delta, threshold = (
        config.get(name) for name in ("detector", "key", "gamma", "delta", "z_threshold")
    )
    if not paired._nonempty_string(detector) or type(key) is not int or key < 0:
        raise ValueError("A detector ID and a nonnegative integer key are required.")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (gamma, delta, threshold)):
        raise ValueError("Gamma, delta and the z threshold must be finite.")
    if not 0 < gamma < 1 or delta <= 0:
        raise ValueError("Gamma must lie between zero and one, and delta must be positive.")
    return detector, key, gamma, delta, threshold


def _reference_module(reference_root: Path) -> ModuleType:
    if any(Path(name).stem in sys.modules for name in REFERENCE_FILES):
        raise RuntimeError("The reference module namespace is already loaded.")
    sys.path.insert(0, str(reference_root.resolve()))
    module = importlib.import_module("watermark_processor")
    for name in REFERENCE_FILES:
        resolved = Path(sys.modules[Path(name).stem].__file__).resolve()
        if resolved != (reference_root / name).resolve():
            raise RuntimeError(f"Reference module loaded outside the pinned directory: {name}.")
    return module


def generate(
    manifest: Path,
    run: Path,
    reviews: Path,
    config_path: Path,
    reference_root: Path,
    tokenizer_dir: Path,
    output: Path,
    dependency_dir: Path | None = None,
    *,
    second_reviews: Path | None = None,
) -> None:
    """Produce one receipt per exact text with a separate raw-score record."""
    _, minimum, trials, manifest_digest = paired._manifest(manifest)
    _, review_digest, packet_digest = paired._sealed_reviews(
        run, reviews, trials, manifest_digest, second_reviews=second_reviews
    )
    config, config_digest = paired._read(config_path)
    if not isinstance(config, dict):
        raise ValueError("Expected a KGW configuration object.")
    detector_id, key, gamma, delta, threshold = _settings(config)
    trials = [trial for trial in trials if trial.detector == detector_id]
    if not trials:
        raise ValueError("The configured detector must match at least one trial in the manifest.")
    paired._verify_file_digests(reference_root, config.get("reference_sha256"), REFERENCE_FILES)
    paired._verify_file_digests(tokenizer_dir, config.get("tokenizer_sha256"), TOKENIZER_FILES)
    tokenizer_config, _ = paired._read(tokenizer_dir / "config.json")
    vocab = tokenizer_config.get("vocab_size") if isinstance(tokenizer_config, dict) else None
    if type(vocab) is not int or vocab < 2:
        raise ValueError("The pinned model config must provide a valid vocabulary size.")
    if output.exists():
        raise FileExistsError(output)

    if dependency_dir is not None:
        sys.path.append(str(dependency_dir.resolve()))
    import torch
    from transformers import AutoTokenizer

    module = _reference_module(reference_root)

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    detector = module.WatermarkDetector(
        vocab=list(range(vocab)),
        gamma=gamma,
        delta=delta,
        seeding_scheme="simple_1",
        select_green_tokens=True,
        device=torch.device("cpu"),
        tokenizer=tokenizer,
        z_threshold=threshold,
        normalizers=[],
        ignore_repeated_bigrams=True,
        hash_key=key,
    )
    scores: dict[str, dict[str, Any]] = {}
    for trial in trials:
        for value in paired._trial_values(trial):
            digest = paired.text_digest(value)
            if digest in scores:
                continue
            ids = tokenizer.encode(value, add_special_tokens=False)
            scored = ids[1:] if ids and ids[0] == tokenizer.bos_token_id else ids
            positions = len(set(pairwise(scored)))
            z = None
            positive = False
            if positions:
                report = detector.detect(tokenized_text=torch.tensor(ids))
                z = float(report["z_score"])
                positive = bool(report["prediction"])
                if int(report["num_tokens_scored"]) != positions or not math.isfinite(z) or positive != (z > threshold):
                    raise RuntimeError("The reference detector returned inconsistent scoring metadata.")
            scores[digest] = {"positions": positions, "z": z, "detected": positive}
    receipts = [
        {"detector": detector_id, "sha256": digest, "detected": row["detected"], "positions": row["positions"]}
        for digest, row in scores.items()
    ]
    output.mkdir()
    receipts_digest = paired._write(output / "detections.json", receipts)
    scores_digest = paired._write(output / "reference-scores.json", scores)
    paired._write(
        output / "settings.json",
        {
            "config": config,
            "config_sha256": config_digest,
            "manifest_sha256": manifest_digest,
            "packet_sha256": packet_digest,
            "reviews_sha256": review_digest,
            "receipts_sha256": receipts_digest,
            "scores_sha256": scores_digest,
            "worker_sha256": paired._file_digest(Path(__file__)),
            "scorer_sha256": paired._file_digest(Path(paired.__file__)),
            "minimum_positions": minimum[detector_id],
            "profile": "cpu-simple_1-distinct-bigrams",
            "runtime": {name: version(name) for name in ("nltk", "scipy", "tokenizers", "torch", "transformers")},
        },
    )
    log.info("Scored %s distinct texts with %s.", len(scores), detector_id)


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("input", "run", "reviews", "config", "reference-root", "tokenizer-dir", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path)
    parser.add_argument("--second-reviews", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        generate(
            args.input,
            args.run,
            args.reviews,
            args.config,
            args.reference_root,
            args.tokenizer_dir,
            args.output_dir,
            args.dependency_dir,
            second_reviews=args.second_reviews,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        log.error("%s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
