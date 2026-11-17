"""Benchmark library rewrites against an external, revision-recorded MarkLLM checkout.

Synthetic local generation only. Each scheme runs in a bounded subprocess.
Unavailable schemes and invalid controls are failures of coverage, never passes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
log = logging.getLogger(__name__)
PROMPTS = {
    "en": "Explain how public libraries help a small town. "
    "Discuss benefits and limitations in two paragraphs. No headings.",
    "ru": "Объясни на русском языке, как библиотеки помогают небольшому городу. "
    "Два абзаца о пользе и ограничениях, без заголовков.",  # noqa: RUF001
}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def assess(
    before: dict[str, Any],
    baseline: dict[str, Any],
    after: dict[str, Any] | None,
    *,
    retained_prose_spans: int = 0,
) -> str:
    """Classify matched diagnostic controls, keeping partial rewrites unverified."""
    if not before["is_watermarked"]:
        return "invalid_positive_control"
    if baseline["is_watermarked"]:
        return "invalid_negative_control"
    if after is None:
        return "refused"
    if after["is_watermarked"]:
        return "retained"
    return "partial_unverified" if retained_prose_spans else "below_reference_threshold"


def cpu_reference_model(model: Any, device: str) -> Any:
    """Keep reference sampling on CPU while forwarding model inference to its device."""

    class Bridge:
        device = "cpu"

        def __getattr__(self, name: str) -> Any:
            return getattr(model, name)

        def __call__(self, token_ids: Any, **kwargs: Any) -> Any:
            for key in ("attention_mask", "position_ids", "token_type_ids"):
                if key in kwargs:
                    kwargs[key] = kwargs[key].to(device)
            output = model(token_ids.to(device), **kwargs)
            output.logits = output.logits.cpu()
            return output

        forward = __call__

    return Bridge()


def guard_reference_memory(scheme: str, vocab_size: int, config: dict[str, Any], budget: int) -> None:
    """Reject known dense reference tables before the external constructor allocates them."""
    if scheme == "EXPGumbel":
        minimum_bytes = 2 * 4 * vocab_size**2 * int(config["prefix_length"])
        if minimum_bytes > budget:
            raise MemoryError(f"EXPGumbel reference tables need at least {minimum_bytes} bytes; budget is {budget}")


def strip_reference_prompt(generated: str, plain: str, tagged: str) -> str:
    """Remove a complete input prefix and reject truncated prompt-only results."""
    for prefix in (plain, tagged):
        if generated.startswith(prefix):
            continuation = generated[len(prefix) :].strip()
            if not continuation:
                raise ValueError("Reference returned the prompt without generated content")
            return continuation
        if generated.strip() and prefix.startswith(generated.strip()):
            raise ValueError("Reference returned a truncated input prompt")
    return generated


def reset_reference_history(reference: Any, *, generation: bool) -> None:
    """Match the DIP/Unbiased public methods' history and phase lifecycle."""
    utils = getattr(reference, "utils", None)
    if utils is not None and hasattr(utils, "cc_history") and hasattr(utils, "state_indicator"):
        utils.cc_history.clear()
        utils.state_indicator = 0 if generation else 1


def worker(args: argparse.Namespace) -> None:
    if args.dependency_dir:
        sys.path.insert(0, str(args.dependency_dir))
    import torch
    from transformers import LogitsProcessor, LogitsProcessorList
    from transformers import __version__ as transformers_version

    from remove_ai_watermarks import text as lib

    sys.path.insert(0, str(args.markllm_root))
    # These development-only modules live in the caller's external checkout.
    TransformersConfig: Any = importlib.import_module("utils.transformers_config").TransformersConfig
    AutoWatermark: Any = importlib.import_module("watermark.auto_watermark").AutoWatermark

    torch.set_num_threads(4)
    # Deliberately share the internal cached runtime for this revision-bound study.
    runtime_loader: Any = cast("Any", lib)._load_local_rewriter_runtime
    tokenizer, model, device = runtime_loader(args.model, args.device)
    config_path = args.markllm_root / "config" / f"{args.worker}.json"
    config = TransformersConfig(
        model,
        tokenizer,
        vocab_size=model.config.vocab_size,
        device="cpu",
        max_new_tokens=args.tokens,
        temperature=0.8,
        top_p=1.0,
    )
    guard_reference_memory(
        args.worker,
        model.config.vocab_size,
        json.loads(config_path.read_text()),
        args.reference_memory_budget_gib * 1024**3,
    )
    if args.worker == "ITSEdit":
        # This reference still uses the older two-argument constructor.
        ITSEdit: Any = importlib.import_module("watermark.its_edit").ITSEdit

        watermark = ITSEdit(str(config_path), config)
    else:
        watermark = AutoWatermark.load(args.worker, algorithm_config=str(config_path), transformers_config=config)
    custom_generation = args.worker in {"EXP", "EXPGumbel", "EXPEdit", "ITSEdit", "PF"}
    if (args.worker in {"SWEET", "EWD", "TextSeal"} or custom_generation) and device != "cpu":
        watermark.config.generation_model = cpu_reference_model(model, device)
    if not custom_generation and not hasattr(watermark, "logits_processor"):
        raise NotImplementedError("Scheme needs a custom generation pipeline, not the logits-processor adapter")

    class CPUReference(LogitsProcessor):
        def __call__(self, input_ids: Any, scores: Any) -> Any:
            return watermark.logits_processor(input_ids.cpu(), scores.cpu()).to(scores.device)

    report: dict[str, Any] = {
        "scheme": args.worker,
        "benchmark_script_sha256": sha(Path(__file__).read_text()),
        "reference_assets": {
            str(path.relative_to(args.markllm_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((args.markllm_root / "watermark").rglob("*"))
            if path.is_file() and path.suffix in {".pt", ".pth", ".so", ".joblib"}
        },
        "model": args.model,
        "model_revision": model.config._commit_hash,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "generation_device": device,
        "rewriter_model": args.rewriter_model or args.model,
        "reference_device": "cpu",
        "prompt_mode": "plain_completion" if args.worker == "PF" else "chat_template",
        "generation": {"max_new_tokens": args.tokens, "temperature": 0.8, "top_k": 0, "top_p": 1.0},
        "adapter": "author_custom_generation" if custom_generation else "cpu_logits_processor",
        "config": json.loads(config_path.read_text()),
        "config_sha256": sha(config_path.read_text()),
        "text_module_sha256": sha(Path(lib.__file__).read_text()),
        "cases": [],
    }
    report_path = args.output_dir / f"{args.worker}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for index, language in enumerate(args.languages):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPTS[language]}], tokenize=False, add_generation_prompt=True
        )
        # PF's EOS trimming scans the entire sequence, so chat EOS tokens in
        # the input would truncate its result to the system prompt.
        if args.worker == "PF":
            prompt = PROMPTS[language] + "\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        def generate(marked: bool, *, inputs: Any = inputs, index: int = index, prompt: str = prompt) -> str:
            cast("Any", torch).manual_seed(100 + index)
            if marked:
                reset_reference_history(watermark, generation=True)
            if marked and custom_generation:
                watermark.config.sequence_length = args.tokens
                generated = watermark.generate_watermarked_text(prompt)
                prompt_prose = tokenizer.decode(tokenizer.encode(prompt), skip_special_tokens=True)
                return strip_reference_prompt(str(generated), prompt_prose, prompt)
            with torch.no_grad():
                tokens = model.generate(
                    **inputs,
                    max_new_tokens=args.tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_k=0,
                    top_p=1.0,
                    logits_processor=LogitsProcessorList([CPUReference()]) if marked else None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return str(tokenizer.decode(tokens[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True))

        prefix = f"{args.worker}-{language}"
        previous_path = args.input_dir / f"{args.worker}.json" if args.input_dir else None
        previous: dict[str, Any] = (
            json.loads(previous_path.read_text()) if previous_path and previous_path.exists() else {}
        )
        frozen: dict[str, Any] | None = next(
            (row for row in previous.get("cases", []) if row["language"] == language), None
        )
        if frozen is not None:
            source = (args.input_dir / f"{prefix}-source.txt").read_text()
            baseline = (args.input_dir / f"{prefix}-baseline.txt").read_text()
            if sha(source) != frozen["source_sha256"]:
                raise ValueError("Frozen source hash mismatch")
            if frozen.get("baseline_sha256") and sha(baseline) != frozen["baseline_sha256"]:
                raise ValueError("Frozen baseline hash mismatch")
            if previous["config_sha256"] != report["config_sha256"]:
                raise ValueError("Frozen reference configuration mismatch")
            if previous["model_revision"] != report["model_revision"]:
                raise ValueError("Frozen source model revision mismatch")
        else:
            baseline, source = generate(False), generate(True)
        reset_reference_history(watermark, generation=False)
        before = watermark.detect_watermark(source)
        reset_reference_history(watermark, generation=False)
        control = watermark.detect_watermark(baseline)
        row = {
            "language": language,
            "before": before,
            "baseline": control,
            "source_sha256": sha(source),
            "baseline_sha256": sha(baseline),
        }
        prefix = f"{args.worker}-{language}"
        (args.output_dir / f"{prefix}-source.txt").write_text(source)
        (args.output_dir / f"{prefix}-baseline.txt").write_text(baseline)
        after = None
        if before["is_watermarked"] and not control["is_watermarked"]:
            try:
                result = lib.remove_text_invisible(
                    source, rewriter_model=args.rewriter_model or args.model, device=device, seed=0
                )
                reset_reference_history(watermark, generation=False)
                after = watermark.detect_watermark(result.cleaned_text)
                row.update(after=after, after_sha256=sha(result.cleaned_text))
                (args.output_dir / f"{prefix}-after.txt").write_text(result.cleaned_text)
            except RuntimeError as exc:
                row["reason"] = str(exc)
        row["status"] = assess(before, control, after)
        report["cases"].append(row)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda x: x.item()))
        log.info("Result %s %s", args.worker, row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markllm-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, help="Replay frozen sources from a previous run")
    parser.add_argument("--dependency-dir", type=Path, help="Optional isolated reference dependencies")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--rewriter-model", help="Use a separate local rewrite model")
    parser.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--languages", nargs="+", choices=tuple(PROMPTS), default=["en", "ru"])
    parser.add_argument("--schemes", nargs="+")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--reference-memory-budget-gib", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.markllm_root = args.markllm_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.input_dir:
        args.input_dir = args.input_dir.resolve()
    if args.dependency_dir:
        args.dependency_dir = args.dependency_dir.resolve()
    if args.worker:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.markllm_root))
    mapping: dict[str, str] = importlib.import_module("watermark.auto_watermark").WATERMARK_MAPPING_NAMES

    schemes: list[str] = args.schemes or list(mapping)
    unknown = set(schemes) - mapping.keys()
    if unknown:
        parser.error(f"Unknown reference schemes: {sorted(unknown)}")
    revision = subprocess.check_output(  # noqa: S603 - fixed argv, no shell
        [shutil.which("git") or "/usr/bin/git", "-C", str(args.markllm_root), "rev-parse", "HEAD"], text=True
    ).strip()
    summary: dict[str, Any] = {"markllm_revision": revision, "model": args.model, "schemes": []}
    for scheme in schemes:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", scheme]
        log.info("Starting %s", scheme)
        with (args.output_dir / f"{scheme}.log").open("w") as output:
            try:
                result = subprocess.run(  # noqa: S603 - this script with argparse-validated scheme, no shell
                    command,
                    cwd=args.markllm_root,
                    env=os.environ.copy(),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                status = "completed" if result.returncode == 0 else "reference_runtime_error"
                summary["schemes"].append({"scheme": scheme, "status": status, "exit_code": result.returncode})
            except subprocess.TimeoutExpired:
                summary["schemes"].append({"scheme": scheme, "status": "timeout"})
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Finished %s: %s", scheme, summary["schemes"][-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
