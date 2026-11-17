"""Evaluate local NLI support on synthetic fidelity pairs, without truncation.

This research diagnostic is neither semantic proof nor a watermark detector.
It never downloads models or enables a library acceptance policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
log = logging.getLogger(__name__)


class InputWindowExceeded(ValueError):
    """The complete paired input exceeds the explicitly selected token limit."""


def entailment_label_index(labels: Mapping[int | str, str]) -> int:
    """Read the exact label, never a substring of not_entailment or a fixed ID."""
    indices = {int(index) for index in labels}
    if indices != set(range(len(labels))):
        raise ValueError("NLI label IDs must be contiguous and start at zero")
    found = [int(index) for index, name in labels.items() if name.casefold() == "entailment"]
    if len(found) != 1:
        raise ValueError("NLI config needs exactly one explicit entailment label")
    return found[0]


def aggregate_support(source: str, candidate: str, predict: Callable[[str, str], object]) -> dict[str, float]:
    """Keep both directions and every sentence; do not pool away weak claims."""
    if not source.strip() or not candidate.strip():
        raise ValueError("Fidelity pairs need nonempty prose")

    def checked(before: str, after: str) -> float:
        value = predict(before, after)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("NLI support must be a finite probability")
        return float(value)

    whole = [checked(source, candidate), checked(candidate, source)]
    sentences = [
        checked(premise, sentence.strip())
        for premise, hypothesis in ((source, candidate), (candidate, source))
        for sentence in re.split(r"(?<=[.!?])\s+(?=\S)", hypothesis.strip())
        if sentence.strip()
    ]
    if not sentences:
        raise ValueError("Fidelity pairs need nonempty prose")
    return {"whole": min(whole), "sentence": min(sentences)}


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = cast("dict[str, object]", payload).get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Expected a nonempty cases array")
    cases: list[dict[str, Any]] = []
    for item in cast("list[object]", raw_cases):
        if not isinstance(item, dict):
            raise ValueError("Cases need unique nonempty string IDs")
        cases.append(cast("dict[str, Any]", item))
    seen: set[str] = set()
    for row in cases:
        if not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Cases need unique nonempty string IDs")
        seen.add(row["id"])
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("source", "candidate")):
            raise ValueError("Cases need complete nonempty source and candidate text")
        if "equivalent" not in row or (row["equivalent"] is not None and type(row["equivalent"]) is not bool):
            raise ValueError("equivalent must be bool, or null for an unlabeled diagnostic")
    return cases


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(args.cases)
    if args.max_pair_tokens <= 0:
        raise ValueError("max-pair-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False
    )
    original_limit = tokenizer.model_max_length
    tokenizer.model_max_length = args.max_pair_tokens
    model, info = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if info["missing_keys"] or info["unexpected_keys"]:
        raise RuntimeError("NLI checkpoint does not exactly match the loaded model")
    labels = model.config.id2label
    index = entailment_label_index(labels)
    capacity = model.config.max_position_embeddings
    if model.config.model_type in {"roberta", "xlm-roberta"}:
        capacity -= model.config.pad_token_id + 1
    if args.max_pair_tokens > capacity:
        raise ValueError("Selected input limit exceeds the model's position capacity")
    model.train(False)
    model.to(args.device)
    torch.set_num_threads(4)
    report: dict[str, Any] = {
        "scope": "Examined NLI support diagnostic, not semantic proof or watermark removal",
        "completed": False,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers")},
        "cases_sha256": sha(args.cases),
        "script_sha256": sha(Path(__file__)),
        "model_files_sha256": {
            p.name: sha(p)
            for p in sorted(args.model_dir.iterdir())
            if p.is_file() and p.suffix in {".json", ".safetensors", ".model"}
        },
        "model_config": model.config.to_dict(),
        "raw_tokenizer_model_max_length": original_limit,
        "max_pair_tokens": args.max_pair_tokens,
        "truncation": False,
        "device": args.device,
        "dtype": "float32",
        "attention": "eager",
        "calls": [],
        "cases": [],
    }
    cache: dict[tuple[str, str], float] = {}

    def predict(source: str, candidate: str) -> float:
        key = (source, candidate)
        if key in cache:
            return cache[key]
        inputs = tokenizer(source, candidate, truncation=False, return_tensors="pt")
        count = int(inputs["input_ids"].shape[1])
        if count > args.max_pair_tokens:
            raise InputWindowExceeded(f"Complete pair has {count} tokens, limit is {args.max_pair_tokens}")
        with torch.inference_mode():
            probabilities = (
                model(**{k: v.to(args.device) for k, v in inputs.items()}).logits[0].softmax(-1).cpu().tolist()
            )
        if len(probabilities) != len(labels):
            raise RuntimeError("NLI output size differs from the label map")
        value = float(probabilities[index])
        report["calls"].append(
            {
                "source": source,
                "candidate": candidate,
                "input_tokens": inputs["input_ids"][0].tolist(),
                "probabilities": {labels[i]: float(p) for i, p in enumerate(probabilities)},
            }
        )
        cache[key] = value
        return value

    def save() -> None:
        (args.output_dir / "benchmark.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    for case in cases:
        row = dict(case)
        try:
            row.update(status="scored", scores=aggregate_support(case["source"], case["candidate"], predict))
        except InputWindowExceeded as error:
            row.update(status="input_window_exceeded", error=str(error))
        report["cases"].append(row)
        save()
    report["completed"] = True
    save()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/evaluations/text-watermark/fidelity-cases.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-pair-tokens", type=int, default=512)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = evaluate(args)
    unscored = sum(row["status"] != "scored" for row in report["cases"])
    log.info("NLI pairs scored: %s; unscored: %s", len(report["cases"]) - unscored, unscored)
    if unscored:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
