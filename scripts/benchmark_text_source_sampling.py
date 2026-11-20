"""Shared offline checkpoint sampling for the source benchmark commands."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from scripts import benchmark_text_pairs as paired

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

COMPLETE = re.compile(r"""[.!?][”"']?\s*$""")
TOKENIZER_FILES = {"config.json", "tokenizer.json", "tokenizer_config.json"}


def check_plan(config: dict[str, Any]) -> None:
    bounded = {"minimum_positions": 1, "batch_size": 1, "seed": 0, "top_k": 0, "max_new_tokens": 1}
    if any(type(config.get(key)) is not int or config[key] < floor for key, floor in bounded.items()):
        raise ValueError("Invalid source generation integer settings.")
    temperature, prompts = config.get("temperature"), config.get("prompts")
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Sampling temperature must be positive and finite.")
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(paired._nonempty_string(value) for value in prompts)
        or len(set(prompts)) != len(prompts)
    ):
        raise ValueError("Provide distinct, nonempty source prompts.")


def verify_checkpoint(model_dir: Path, digests: object) -> None:
    weights = {path.name for path in model_dir.glob("*.safetensors")}
    if not weights:
        raise ValueError("The local model directory must have safetensors weights.")
    required = TOKENIZER_FILES | {"generation_config.json"} | weights
    index = "model.safetensors.index.json"
    if (model_dir / index).is_file():
        required.add(index)
    paired._verify_file_digests(model_dir, digests, required)


def load_checkpoint(model_dir: Path, device: str) -> tuple[Any, Any, tuple[int, ...]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("This source plan requires MPS. Choose CPU explicitly for a different runtime.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, padding_side="left", trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, local_files_only=True, use_safetensors=True, trust_remote_code=False
    ).to(torch.device(device))
    model.train(False)
    end_tokens = model.generation_config.eos_token_id
    eos = tuple(sorted([end_tokens] if isinstance(end_tokens, int) else (end_tokens or [])))
    if not eos:
        raise ValueError("The pinned model generation configuration must specify EOS token IDs.")
    return tokenizer, model, eos


def batches(
    tokenizer: Any,
    model: Any,
    config: dict[str, Any],
    eos: tuple[int, ...],
    *,
    processor: Callable[[], list[Any]],
    top_k: int,
    temperature: float,
) -> Iterator[list[list[int]]]:
    import torch
    from transformers import set_seed

    prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": value}], tokenize=False, add_generation_prompt=True)
        for value in config["prompts"]
    ]
    batch_size = config["batch_size"]
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size], padding=True, add_special_tokens=False, return_tensors="pt"
        ).to(model.device)
        prefix = encoded.input_ids.shape[1]
        set_seed(config["seed"] + start)
        with torch.inference_mode():
            tokens = model.generate(
                **encoded,
                do_sample=True,
                max_new_tokens=config["max_new_tokens"],
                top_k=top_k,
                temperature=temperature,
                top_p=1.0,
                min_p=None,
                repetition_penalty=1.0,
                eos_token_id=list(eos),
                pad_token_id=eos[0],
                forced_eos_token_id=None,
                logits_processor=processor(),
                return_dict_in_generate=False,
            )
        if not torch.equal(tokens[:, :prefix], encoded.input_ids):
            raise RuntimeError("The generated trace changed the prompt prefix.")
        yield tokens[:, prefix:].cpu().tolist()


def continuation(tokenizer: Any, tokens: list[int], eos: tuple[int, ...]) -> tuple[dict[str, Any], list[int]]:
    end = next((i for i, token in enumerate(tokens) if token in eos), None)
    ids = tokens if end is None else tokens[:end]
    value = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    recovered = tokenizer.encode(value, add_special_tokens=False)
    return {
        "text": value,
        "sha256": paired.text_digest(value),
        "continuation": ids,
        "eos_token": None if end is None else tokens[end],
        "checks": {
            "round_trip": ids == recovered,
            "no_specials": not set(ids).intersection(tokenizer.all_special_ids),
            "complete": end is not None and bool(COMPLETE.search(value)),
        },
    }, recovered
