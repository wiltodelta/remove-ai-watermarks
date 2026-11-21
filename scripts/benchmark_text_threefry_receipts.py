"""Score sealed text trials with the pinned Threefry reference and the library."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from remove_ai_watermarks import text  # noqa: E402
from scripts import benchmark_text_pairs as paired  # noqa: E402

log = logging.getLogger(__name__)
REFERENCE_FILES = {
    "synthid_text/detector_frequentist.py",
    "synthid_text/hashing_function.py",
    "synthid_text/logits_processing.py",
}
TOKENIZER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}


def _settings(config: dict[str, Any]) -> tuple[str, dict[str, int], tuple[int, ...], float]:
    integers = {"key": 0, "ngram_len": 2, "depth": 1, "context_history_size": 1}
    scheme = config.get("scheme")
    if not isinstance(scheme, dict):
        raise ValueError("Expected a Threefry scheme object.")
    if any(type(scheme.get(name)) is not int or scheme[name] < low for name, low in integers.items()):
        raise ValueError("Invalid Threefry key, ngram length, depth or context history.")
    if scheme["key"] >= (1 << 63) or scheme.get("num_leaves") != 2:
        raise ValueError("The parity profile requires a 63-bit key and exactly two tournament leaves.")
    eos = config.get("eos_token_ids")
    if not isinstance(eos, list) or not eos or any(type(x) is not int or not 0 <= x < (1 << 32) for x in eos):
        raise ValueError("Expected nonempty EOS token IDs from source generation.")
    alpha, detector = config.get("alpha"), config.get("detector")
    if type(alpha) not in (int, float) or not 0 < alpha < 1 or not paired._nonempty_string(detector):
        raise ValueError("A detector ID and an alpha strictly between zero and one are required.")
    return detector, {name: scheme[name] for name in integers}, tuple(eos), alpha


def _verify_reference(reference_root: Path, digests: object) -> None:
    required = set(REFERENCE_FILES)
    if (reference_root / "synthid_text/__init__.py").is_file():
        required.add("synthid_text/__init__.py")
    paired._verify_file_digests(reference_root, digests, required)


def _reference_types(reference_root: Path) -> tuple[type, type]:
    if "synthid_text" in sys.modules:
        raise RuntimeError("The Threefry reference package is already loaded.")
    sys.path.insert(0, str(reference_root.resolve()))
    from synthid_text.detector_frequentist import FrequentistDetector
    from synthid_text.logits_processing import SynthIDLogitsProcessor

    for name in REFERENCE_FILES:
        module = sys.modules[f"synthid_text.{Path(name).stem}"]
        if Path(module.__file__).resolve() != (reference_root / name).resolve():
            raise RuntimeError(f"Reference module loaded outside the pinned directory: {name}.")

    return FrequentistDetector, SynthIDLogitsProcessor


def _score_token_ids(
    ids: list[int],
    scheme: dict[str, int],
    eos: tuple[int, ...],
    alpha: float,
    processor: Any,
    reference: Any,
    digest: str,
) -> dict[str, Any]:
    import torch

    ours = text.score_synthid_threefry_text(ids, **scheme, eos_token_ids=eos, alpha=alpha)
    p_value, positions, ones = 1.0, 0, 0
    if len(ids) >= scheme["ngram_len"]:
        tokens = torch.tensor([ids], dtype=torch.long)
        p_value = float(reference.p_value(tokens)[0])
        hits, mask = processor.compute_g_values_and_mask(tokens, eos)
        positions, ones = int(mask.sum()), int(hits[mask].sum())
    if (
        (ours.num_tokens, ours.num_ones) != (positions, ones)
        or not math.isclose(ours.p_value, p_value, rel_tol=0.0, abs_tol=2e-12)
        or ours.detected != (p_value <= alpha)
    ):
        raise RuntimeError(f"Threefry library and reference scores differ for text SHA-256 {digest}.")
    return {"positions": positions, "ones": ones, "p": p_value, "library_p": ours.p_value, "detected": ours.detected}


def generate(
    manifest: Path,
    run: Path,
    reviews: Path,
    config_path: Path,
    reference_root: Path,
    tokenizer_dir: Path,
    output: Path,
    *,
    second_reviews: Path | None = None,
) -> None:
    """Emit exact-text detector receipts after independently checking Threefry parity."""
    _, minimum, trials, manifest_digest = paired._manifest(manifest)
    _, review_digest, packet_digest = paired._sealed_reviews(
        run, reviews, trials, manifest_digest, second_reviews=second_reviews
    )
    config, config_digest = paired._read(config_path)
    if not isinstance(config, dict):
        raise ValueError("Expected a Threefry configuration object.")
    detector_id, scheme, eos, alpha = _settings(config)
    trials = [trial for trial in trials if trial.detector == detector_id]
    if not trials:
        raise ValueError("The configured detector must match at least one trial in the manifest.")
    _verify_reference(reference_root, config.get("reference_sha256"))
    paired._verify_file_digests(tokenizer_dir, config.get("tokenizer_sha256"), TOKENIZER_FILES)
    if output.exists():
        raise FileExistsError(output)

    import torch
    from transformers import AutoTokenizer

    FrequentistDetector, SynthIDLogitsProcessor = _reference_types(reference_root)

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    processor = SynthIDLogitsProcessor(device=torch.device("cpu"), num_leaves=2, **scheme)
    reference = FrequentistDetector(processor, eos)
    scores: dict[str, dict[str, Any]] = {}
    for trial in trials:
        for value in paired._trial_values(trial):
            digest = paired.text_digest(value)
            if digest in scores:
                continue
            ids = tokenizer.encode(value, add_special_tokens=False)
            scores[digest] = _score_token_ids(ids, scheme, eos, alpha, processor, reference, digest)
    receipts = [
        {"detector": detector_id, "sha256": digest, "detected": row["detected"], "positions": row["positions"]}
        for digest, row in scores.items()
    ]
    runtime = {name: version(name) for name in ("numpy", "scipy", "tokenizers", "torch", "transformers")}
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
            "library_sha256": paired._file_digest(Path(text.__file__)),
            "minimum_positions": minimum[detector_id],
            "profile": "cpu-single-key-threefry",
            "runtime": runtime,
        },
    )
    log.info("Scored %s distinct texts with %s and verified library parity.", len(scores), detector_id)


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("input", "run", "reviews", "config", "reference-root", "tokenizer-dir", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
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
            second_reviews=args.second_reviews,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        log.error("%s", error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
