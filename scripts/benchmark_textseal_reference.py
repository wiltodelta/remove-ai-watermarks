"""Evaluate local rewrites against the author's TextSeal reference implementations.

Research-owned keys only; the textGrain reference is not OpenAI's production
key or detector. Requires an external textseal checkout and optional ML deps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_text_schemes import PROMPTS, assess, cpu_reference_model, sha  # noqa: E402 - project root added above

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=("textseal", "textseal2", "automark", "textgrain"),
        default=["textseal", "textseal2", "automark", "textgrain"],
    )
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.dependency_dir.resolve()))
    sys.path.insert(0, str(args.reference_root.resolve()))
    import torch
    from textseal.watermarking.config import WatermarkConfig
    from textseal.watermarking.detector import build_detector
    from textseal.watermarking.generator import build_generator

    from remove_ai_watermarks import text as lib

    torch.set_num_threads(4)
    tokenizer, model, _ = lib._load_local_rewriter_runtime(args.model, args.device)
    revision = subprocess.check_output(  # noqa: S603 - fixed argv, no shell
        [shutil.which("git") or "/usr/bin/git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPTS["en"]}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    torch.manual_seed(100)
    with torch.no_grad():
        baseline_ids = model.generate(
            **inputs,
            max_new_tokens=args.tokens,
            do_sample=True,
            temperature=0.8,
            top_k=0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )[0][len(prompt_ids) :]
    baseline = str(tokenizer.decode(baseline_ids, skip_special_tokens=True))
    report = {
        "reference_revision": revision,
        "model": args.model,
        "model_revision": model.config._commit_hash,
        "model_device": args.device,
        "reference_device": "cpu",
        "threshold_p": 0.01,
        "generation": {"tokens": args.tokens, "temperature": 0.8, "top_p": 1.0, "seed": 100},
        "text_module_sha256": sha(Path(lib.__file__).read_text()),
        "cases": [],
    }

    def timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("Reference case exceeded its time budget")

    signal.signal(signal.SIGALRM, timeout)
    for scheme in args.schemes:
        row = {"scheme": scheme}
        log.info("Starting %s", scheme)
        signal.alarm(args.timeout)
        try:
            config = WatermarkConfig(
                watermark_type=scheme,
                secret_key=42,
                ngram=3,
                prf="hires" if scheme in {"textseal2", "automark"} else "textseal",
            )
            row["config"] = dataclasses.asdict(config)
            torch.manual_seed(100)
            marked_ids = build_generator(cpu_reference_model(model, args.device), tokenizer, config).generate(
                [prompt_ids], max_gen_len=args.tokens, temperature=0.8, top_p=1.0, return_ids=True
            )[0]
            source = str(tokenizer.decode(marked_ids, skip_special_tokens=True))
            detector = build_detector(tokenizer, config)

            def detect(text: str, detector: Any = detector) -> dict:
                scores = detector.get_scores_by_t([text], scoring_method="v2", seen_windows=set())[0]
                p_value = float(detector.get_pvalue(sum(scores), len(scores), eps=1e-200))
                return {"is_watermarked": p_value < 0.01, "p_value": p_value, "scored_tokens": len(scores)}

            before, control = detect(source), detect(baseline)
            row.update(before=before, baseline=control, source_sha256=sha(source), baseline_sha256=sha(baseline))
            (args.output_dir / f"{scheme}-source.txt").write_text(source)
            (args.output_dir / f"{scheme}-baseline.txt").write_text(baseline)
            after = None
            if before["is_watermarked"] and not control["is_watermarked"]:
                try:
                    cleaned = lib.remove_text_invisible(
                        source, rewriter_model=args.model, device=args.device, seed=0
                    ).cleaned_text
                    after = detect(cleaned)
                    row.update(after=after, after_sha256=sha(cleaned))
                    (args.output_dir / f"{scheme}-after.txt").write_text(cleaned)
                except RuntimeError as exc:
                    row["reason"] = str(exc)
            row["status"] = assess(before, control, after)
        except Exception as exc:
            row.update(status="reference_runtime_error", reason=f"{type(exc).__name__}: {exc}")
            log.exception("Reference case failed: %s", scheme)
        finally:
            signal.alarm(0)
        report["cases"].append(row)
        (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, default=lambda x: x.item()))
        log.info("Result %s", row)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
