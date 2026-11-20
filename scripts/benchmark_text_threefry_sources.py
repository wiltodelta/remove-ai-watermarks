"""Generate matched clean and marked Threefry texts with a pinned local model."""

from __future__ import annotations

import argparse
import logging
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_text_local_repair as repair  # noqa: E402
from scripts import benchmark_text_pairs as paired  # noqa: E402
from scripts import benchmark_text_source_sampling as sampling  # noqa: E402
from scripts import benchmark_text_threefry_receipts as threefry  # noqa: E402

log = logging.getLogger(__name__)


def generate(config_path: Path, reference_root: Path, model_dir: Path, output: Path, *, device: str) -> None:
    """Generate and qualify paired continuations without replaying any detector results."""
    config, config_digest = paired._read(config_path)
    if not isinstance(config, dict):
        raise ValueError("Expected a Threefry source plan object.")
    detector, scheme, expected_eos, alpha = threefry._settings(config)
    sampling.check_plan(config)
    wrong_key = config.get("wrong_key")
    if type(wrong_key) is not int or not 0 <= wrong_key < (1 << 63) or wrong_key == scheme["key"]:
        raise ValueError("The wrong-key control must use a different nonnegative 63-bit key.")
    if config["top_k"] == 0:
        raise ValueError("The Threefry source profile needs a positive top-k sampling width.")
    sampling.verify_checkpoint(model_dir, config.get("model_sha256"))
    threefry._verify_reference(reference_root, config.get("reference_sha256"))
    if output.exists():
        raise FileExistsError(output)

    import torch
    from transformers import LogitsProcessor

    FrequentistDetector, SynthIDLogitsProcessor = threefry._reference_types(reference_root)
    tokenizer, model, eos = sampling.load_checkpoint(model_dir, device)
    if eos != expected_eos:
        raise ValueError("The configured EOS token IDs must match the checkpoint in sorted order.")
    if config["top_k"] > model.config.vocab_size:
        raise ValueError("The top-k sampling width exceeds the model vocabulary.")

    cpu = torch.device("cpu")

    class SampleCandidates(LogitsProcessor):
        def __init__(self, marked: bool) -> None:
            self.watermark = SynthIDLogitsProcessor(device=cpu, num_leaves=2, **scheme) if marked else None

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            scaled = scores / config["temperature"]
            values, indices = torch.topk(scaled, config["top_k"], dim=-1)
            if self.watermark is not None:
                values = self.watermark.watermark_candidates(input_ids.cpu(), values.float().cpu(), indices.cpu()).to(
                    device=scores.device, dtype=scores.dtype
                )
            return torch.full_like(scaled, float("-inf")).scatter(1, indices, values)

    profiles = {"matched": scheme, "wrong_key": scheme | {"key": wrong_key}}
    scoring = {}
    for name, profile in profiles.items():
        processor = SynthIDLogitsProcessor(device=cpu, num_leaves=2, **profile)
        scoring[name] = (processor, FrequentistDetector(processor, eos))

    generated = {}
    for side in ("clean", "marked"):
        observations = []
        for batch in sampling.batches(
            tokenizer,
            model,
            config,
            eos,
            processor=lambda marked=side == "marked": [SampleCandidates(marked)],
            top_k=0,
            temperature=1.0,
        ):
            for raw_ids in batch:
                row, ids = sampling.continuation(tokenizer, raw_ids, eos)
                row["score"] = {
                    name: threefry._score_token_ids(ids, profiles[name], eos, alpha, *engine, row["sha256"])
                    for name, engine in scoring.items()
                }
                row["checks"]["scored"] = row["score"]["matched"]["positions"] >= config["minimum_positions"]
                observations.append(row)
            log.info("Scored %s %s sources.", len(observations), side)
        generated[side] = observations

    rows, trials = [], []
    accepted: set[str] = set()
    for index, (marked, clean) in enumerate(zip(generated["marked"], generated["clean"], strict=True)):
        digests = {marked["sha256"], clean["sha256"]}
        qualified = (
            all(marked["checks"].values())
            and all(clean["checks"].values())
            and marked["score"]["matched"]["detected"]
            and not clean["score"]["matched"]["detected"]
            and not marked["score"]["wrong_key"]["detected"]
            and not accepted.intersection(digests)
        )
        rows.append({"index": index, "marked": marked, "clean": clean, "qualified": qualified})
        if qualified:
            accepted.update(digests)
            trials.append(
                {
                    "id": f"threefry-{index + 1}",
                    "detector": detector,
                    "marked": {"source": marked["text"], "outputs": dict.fromkeys(repair.ARMS)},
                    "clean": {"source": clean["text"], "outputs": dict.fromkeys(repair.ARMS)},
                }
            )

    output.mkdir()
    paired._write(
        output / "qualification.json",
        {
            "config": config,
            "config_sha256": config_digest,
            "device": device,
            "runtime": {name: version(name) for name in ("numpy", "scipy", "tokenizers", "torch", "transformers")},
            "worker_sha256": paired._file_digest(Path(__file__)),
            "sampler_sha256": paired._file_digest(Path(sampling.__file__)),
            "receipt_worker_sha256": paired._file_digest(Path(threefry.__file__)),
            "library_sha256": paired._file_digest(Path(threefry.text.__file__)),
            "eos_token_ids": eos,
            "rows": rows,
        },
    )
    paired._write(
        output / "trials.json",
        {
            "arms": list(repair.ARMS),
            "detectors": {detector: config["minimum_positions"]},
            "pairs": trials,
        },
    )
    log.info("Qualified %s of %s pairs.", len(trials), len(rows))


def main() -> None:
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "reference-root", "model-dir", "output-dir"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        generate(args.config, args.reference_root, args.model_dir, args.output_dir, device=args.device)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
