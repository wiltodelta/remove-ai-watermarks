"""Generate matched clean and marked KGW texts with a pinned local model."""

from __future__ import annotations

import argparse
import logging
import sys
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_text_kgw_receipts as kgw  # noqa: E402
from scripts import benchmark_text_pairs as paired  # noqa: E402
from scripts import benchmark_text_source_sampling as sampling  # noqa: E402

log = logging.getLogger(__name__)
ARMS = ("restore", "strict_residual", "document_residual")


def _check_plan(config: dict[str, Any]) -> None:
    sampling.check_plan(config)
    wrong_key = config.get("wrong_key")
    if type(wrong_key) is not int or wrong_key < 0 or wrong_key == config["key"]:
        raise ValueError("The wrong-key control must use a different nonnegative research key.")


def generate(
    config_path: Path,
    reference_root: Path,
    model_dir: Path,
    output: Path,
    *,
    device: str,
    dependency_dir: Path | None = None,
) -> None:
    """Freeze qualified pairs and preserve the traces of all scheduled attempts."""
    config, config_digest = paired._read(config_path)
    if not isinstance(config, dict):
        raise ValueError("Expected a KGW source plan object.")
    detector_id, key, gamma, delta, threshold = kgw._settings(config)
    _check_plan(config)
    paired._verify_file_digests(reference_root, config.get("reference_sha256"), kgw.REFERENCE_FILES)
    sampling.verify_checkpoint(model_dir, config.get("model_sha256"))
    if output.exists():
        raise FileExistsError(output)
    if dependency_dir is not None:
        sys.path.append(str(dependency_dir.resolve()))
    import torch
    from transformers import LogitsProcessor

    module = kgw._reference_module(reference_root)
    tokenizer, model, eos = sampling.load_checkpoint(model_dir, device)
    cpu = torch.device("cpu")
    watermark = {
        "vocab": list(range(model.config.vocab_size)),
        "gamma": gamma,
        "delta": delta,
        "seeding_scheme": "simple_1",
        "select_green_tokens": True,
    }

    class AuthorOnCpu(LogitsProcessor):
        def __init__(self) -> None:
            self.processor = module.WatermarkLogitsProcessor(**watermark, hash_key=key)

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            return self.processor(input_ids.cpu(), scores.float().cpu()).to(device=model.device, dtype=scores.dtype)

    detectors = {
        label: module.WatermarkDetector(
            **watermark,
            hash_key=hash_key,
            device=cpu,
            tokenizer=tokenizer,
            z_threshold=threshold,
            normalizers=[],
            ignore_repeated_bigrams=True,
        )
        for label, hash_key in (("matched", key), ("wrong_key", config["wrong_key"]))
    }

    def sample(tokens: list[int]) -> dict[str, Any]:
        result, recovered = sampling.continuation(tokenizer, tokens, eos)
        positions = len(set(pairwise(recovered)))
        scores = {}
        if positions:
            for label, detector in detectors.items():
                observation = detector.detect(tokenized_text=torch.tensor(recovered))
                scores[label] = {name: x.item() if hasattr(x, "item") else x for name, x in observation.items()}
                if scores[label]["num_tokens_scored"] != positions:
                    raise RuntimeError("Reference positions differ from the source-token trace.")
        result["score"] = scores
        result["checks"]["scored"] = positions >= config["minimum_positions"]
        return result

    generated = {}
    for side in ("clean", "marked"):
        observations = []
        for batch in sampling.batches(
            tokenizer,
            model,
            config,
            eos,
            processor=(lambda: [AuthorOnCpu()]) if side == "marked" else list,
            top_k=config["top_k"],
            temperature=config["temperature"],
        ):
            observations.extend(sample(ids) for ids in batch)
            log.info("Scored %s %s sources.", len(observations), side)
        generated[side] = observations

    rows, trials = [], []
    qualified_sources: set[str] = set()
    for index, (marked, clean) in enumerate(zip(generated["marked"], generated["clean"], strict=True)):
        qualified = (
            all(marked["checks"].values())
            and all(clean["checks"].values())
            and marked["score"]["matched"]["prediction"]
            and not clean["score"]["matched"]["prediction"]
            and not marked["score"]["wrong_key"]["prediction"]
            and not {marked["sha256"], clean["sha256"]} & qualified_sources
        )
        rows.append({"index": index, "marked": marked, "clean": clean, "qualified": qualified})
        if qualified:
            qualified_sources.update((marked["sha256"], clean["sha256"]))
            trials.append(
                {
                    "id": f"kgw-{index + 1}",
                    "detector": detector_id,
                    "marked": {"source": marked["text"], "outputs": dict.fromkeys(ARMS)},
                    "clean": {"source": clean["text"], "outputs": dict.fromkeys(ARMS)},
                }
            )
    output.mkdir()
    paired._write(
        output / "qualification.json",
        {
            "config": config,
            "config_sha256": config_digest,
            "device": device,
            "runtime": {name: version(name) for name in ("torch", "transformers", "scipy", "nltk")},
            "worker_sha256": paired._file_digest(Path(__file__)),
            "sampler_sha256": paired._file_digest(Path(sampling.__file__)),
            "eos_token_ids": eos,
            "rows": rows,
        },
    )
    paired._write(
        output / "trials.json",
        {
            "arms": list(ARMS),
            "detectors": {detector_id: config["minimum_positions"]},
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
    parser.add_argument("--dependency-dir", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        generate(
            args.config,
            args.reference_root,
            args.model_dir,
            args.output_dir,
            device=args.device,
            dependency_dir=args.dependency_dir,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
