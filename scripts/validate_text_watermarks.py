"""Validate local prose regeneration with research-owned SynthID-Text keys.

Uses synthetic English/Russian samples, never a provider's private detector.
Frozen inputs are preferable for comparing implementations. Requires cached Qwen
weights and the optional torch/transformers runtime; no content is uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
log = logging.getLogger(__name__)

EXAMPLES = (
    (
        "en_explainer",
        "Write about 170 words in two paragraphs explaining why bicycle lanes can help a town. "
        "Discuss benefits and limitations. No headings.",
    ),
    (
        "en_facts",
        "Write a 170-word project update in two paragraphs. These are fictional facts and must stay exact: "
        "the town of Cedarbrook opened 3 reading rooms in 2024, the budget was 12000 dollars, and no branch closed. "
        "Explain the outcome without inventing additional numbers or names. No headings.",
    ),
    (
        "ru_explainer",
        "Напиши около 170 слов на русском языке о том, как библиотеки помогают небольшим городам. "  # noqa: RUF001
        "Два абзаца, без заголовков. Обсуди пользу и ограничения.",
    ),
    (
        "ru_facts",
        "Напиши на русском языке два абзаца, около 170 слов, об учебном проекте. "  # noqa: RUF001
        "Вымышленные факты: команда «Маяк» провела 12 занятий в 2025 году, участвовали 48 человек, "
        "бюджет составил 9600 рублей. Ни одно занятие не отменили. "
        "Не выдумывай другие имена и числа. Без заголовков.",  # noqa: RUF001
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory for this run")
    parser.add_argument("--input-dir", type=Path, help="Reuse hash-bound synthetic inputs from a prior run")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="auto")
    parser.add_argument("--rewriter-model", help="Compare a different local rewriter on the same source scheme")
    args = parser.parse_args()

    import torch
    from benchmark_text_schemes import assess
    from transformers import SynthIDTextWatermarkingConfig
    from transformers import __version__ as transformers_version

    from remove_ai_watermarks import text as lib

    if not Path(lib.__file__).is_relative_to(REPO_ROOT / "src"):
        raise RuntimeError("Validation must import this checkout's text implementation")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    tokenizer, model, device = lib._load_local_rewriter_runtime(lib.DEFAULT_TEXT_REWRITER_MODEL, args.device)
    rewriter_model = args.rewriter_model or lib.DEFAULT_TEXT_REWRITER_MODEL
    _, rewriter_runtime, _ = lib._load_local_rewriter_runtime(rewriter_model, device)
    config = SynthIDTextWatermarkingConfig(ngram_len=5, keys=list(range(101, 131)), sampling_table_seed=42)
    cases: list[dict[str, Any]] = []
    report = {
        "model": lib.DEFAULT_TEXT_REWRITER_MODEL,
        "model_revision": model.config._commit_hash,
        "rewriter_model": rewriter_model,
        "rewriter_revision": rewriter_runtime.config._commit_hash,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "device": device,
        "threshold_z": 4.0,
        "score_kind": "mean-g diagnostic, not the trained Bayesian or a provider detector",
        "key_kind": "research-owned, not a vendor key",
        "watermarking_config": config.to_dict(),
        "generation": {"max_new_tokens": 256, "temperature": 0.8, "top_k": 0, "top_p": 1.0},
        "text_module_sha256": hashlib.sha256(Path(lib.__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    frozen: dict[str, Any] = {}
    if args.input_dir:
        previous = json.loads((args.input_dir / "benchmark.json").read_text())
        frozen = {row["name"]: row for row in previous["cases"]}

    def score(value: str, score_device: str = device) -> dict[str, Any]:
        ids = tokenizer.encode(value, add_special_tokens=False)
        measured = lib.score_synthid_text(
            ids, config.keys, sampling_table_seed=42, device=score_device, threshold_z=4.0
        )
        return {"z": measured.z_score, "detected": measured.detected, "tokens": measured.num_tokens}

    def generate(prompt: str, *, marked: bool, seed: int) -> str:
        torch.manual_seed(seed)
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.8,
                top_k=0,
                top_p=1.0,
                watermarking_config=config if marked else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    for index, (name, prompt) in enumerate(EXAMPLES):
        log.info("Validating %s", name)
        if args.input_dir:
            source = (args.input_dir / f"{name}-source.txt").read_text()
            baseline = (args.input_dir / f"{name}-baseline.txt").read_text()
            if hashlib.sha256(source.encode()).hexdigest() != frozen[name]["source_sha256"]:
                raise ValueError(f"Frozen source hash mismatch: {name}")
            expected_baseline = frozen[name].get("baseline_sha256")
            if expected_baseline and hashlib.sha256(baseline.encode()).hexdigest() != expected_baseline:
                raise ValueError(f"Frozen baseline hash mismatch: {name}")
        else:
            baseline = generate(prompt, marked=False, seed=100 + index)
            source = generate(prompt, marked=True, seed=100 + index)
        row = {
            "name": name,
            "prompt": prompt,
            "baseline": score(baseline),
            "before": score(source),
            "cpu_table_control": score(source, "cpu"),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "baseline_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        }
        (args.output_dir / f"{name}-source.txt").write_text(source)
        (args.output_dir / f"{name}-baseline.txt").write_text(baseline)
        positive = {"is_watermarked": row["before"]["detected"]}
        negative = {"is_watermarked": row["baseline"]["detected"]}
        control_status = assess(positive, negative, None)
        if control_status != "refused":
            row["status"] = control_status
        else:
            try:
                result = lib.remove_text_invisible(source, rewriter_model=rewriter_model, device=device, seed=0)
                cleaned = result.cleaned_text
                (args.output_dir / f"{name}-after.txt").write_text(cleaned)
                measured_after = score(cleaned)
                row.update(
                    status=assess(positive, negative, {"is_watermarked": measured_after["detected"]}),
                    after=measured_after,
                    facts_check=lib.verify_text_rewrite_facts(source, cleaned),
                    changed=source != cleaned,
                    after_sha256=hashlib.sha256(cleaned.encode()).hexdigest(),
                )
            except RuntimeError as exc:
                row.update(status="refused", reason=str(exc))
        cases.append(row)
        (args.output_dir / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        log.info("Result %s", row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
