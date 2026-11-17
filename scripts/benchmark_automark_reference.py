"""Replay selected AutoMark author detectors on frozen synthetic text.

Requires a separate author checkout, local model assets and isolated research
dependencies. Research-owned keys and a local HF execution bridge only.
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
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
log = logging.getLogger(__name__)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_assets(
    tokenizer: Path, reference_tokenizer: Path, groups_path: Path, canon_path: Path, vocab_size: int
) -> dict[str, Any]:
    """Verify text-token compatibility and forbid every unshared special/padding ID."""
    np: Any = importlib.import_module("numpy")
    candidate = json.loads(tokenizer.read_bytes())
    reference = json.loads(reference_tokenizer.read_bytes())
    if any(candidate[field] != reference[field] for field in ("model", "decoder", "pre_tokenizer")):
        raise ValueError("AutoMark tokenizer text mapping differs from the reference tokenizer")
    left = {token["id"]: token for token in candidate["added_tokens"]}
    right = {token["id"]: token for token in reference["added_tokens"]}
    differing = {index for index in left.keys() | right.keys() if left.get(index) != right.get(index)}
    known = set(candidate["model"]["vocab"].values()) | set(left)
    forbidden = differing | (set(range(vocab_size)) - known)
    with np.load(groups_path, allow_pickle=False) as payload:
        groups = payload["groups"]
        eligible = payload["eligible"]
    with np.load(canon_path, allow_pickle=False) as payload:
        canonical = payload["sense"]
        canonical_size = int(payload["vocab_size"])
    if (
        vocab_size <= 0
        or groups.shape != (vocab_size,)
        or eligible.shape != (vocab_size,)
        or canonical.shape != (vocab_size,)
        or canonical_size != vocab_size
        or groups.dtype.kind not in "iu"
        or canonical.dtype.kind not in "iu"
        or np.any(groups < 0)
        or np.any(groups >= vocab_size)
        or np.any(canonical < 0)
        or np.any(canonical >= vocab_size)
    ):
        raise ValueError("AutoMark asset dimensions or token IDs do not match the model vocabulary")
    if any(index >= vocab_size or eligible[index] or int(canonical[index]) != index for index in differing):
        raise ValueError("AutoMark asset groups a tokenizer-specific special token")
    return {
        "ordinary_mapping_equal": True,
        "vocab_size": vocab_size,
        "differing_added_token_ids": sorted(differing),
        "forbidden_token_ids": sorted(forbidden),
        "tokenizer_sha256": sha(tokenizer),
        "reference_tokenizer_sha256": sha(reference_tokenizer),
        "group_asset_sha256": sha(groups_path),
        "canon_asset_sha256": sha(canon_path),
    }


def assess(source: float, baseline: float, after: float | None, alpha: float) -> str:
    """Keep positive/negative controls, missing output and retained marks distinct."""
    values = [source, baseline, alpha, *([] if after is None else [after])]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values) or not 0 < alpha < 1:
        raise ValueError("AutoMark p-values and alpha must be finite probabilities")
    if source > alpha:
        return "invalid_positive_control"
    if baseline <= alpha:
        return "invalid_negative_control"
    if after is None:
        return "no_after_output"
    return "retained" if after <= alpha else "not_detected_at_reference_alpha"


def validate_auxiliary(tokens: list[int], logprobs: Any, row_ids: Any, row_lps: Any, vocab_size: int) -> None:
    """Refuse absent/invalid weighted detection signals instead of degrading to zeros."""
    np: Any = importlib.import_module("numpy")
    lp, ids, rows = map(np.asarray, (logprobs, row_ids, row_lps))
    if (
        len(tokens) < 2
        or lp.shape != (len(tokens),)
        or ids.shape != (len(tokens), 8)
        or rows.shape != ids.shape
        or not all(np.isfinite(array).all() for array in (lp, ids, rows))
        or np.any(lp > 0)
        or np.any(rows > 0)
        or np.any(ids < 0)
        or np.any(ids >= vocab_size)
        or np.any(ids != np.rint(ids))
        or not np.any(lp[1:] < 0)
        or not np.any(rows[1:] < 0)
    ):
        raise ValueError("AutoMark auxiliary signal is missing, malformed or nonfinite")


def load_reference(args: argparse.Namespace, config_payload: dict[str, Any], vocab_size: int) -> tuple[Any, Any]:
    """Build a fresh author detector with explicit, checked tokenizer-specific assets."""
    for directory in reversed(args.dependency_dir):
        sys.path.insert(0, str(directory.resolve()))
    selected = args.reference_root.resolve() / "lm-wm-library/selected"
    sys.path.insert(0, str(selected / "src"))
    os.environ["NUMBA_CACHE_DIR"] = str(args.output_dir.resolve() / "numba-cache")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    config_module: Any = importlib.import_module("lm_wm_tools.configs.main_config")
    processor = config_payload["logit_processor_config"]
    if processor["watermark_class"] == "RotationFirst":
        processor["group_asset_path"] = str(args.group_asset.resolve())
    elif processor["watermark_class"] == "ConcordMark":
        processor.setdefault("distribution_parameters", {})["canon_asset_path"] = str(args.canon_asset.resolve())
    else:
        raise ValueError("Only selected RotationFirst and ConcordMark replays are supported")
    config_payload["device"] = "cpu"
    config = config_module.WatermarkConfig.from_payload(config_payload)
    sampling = SimpleNamespace(temperature=1.0, top_k=50, top_p=1.0)
    return config.get_detector(sampling, vocab_size), config


def auxiliary_signals(args: argparse.Namespace, batches: list[list[int]]) -> tuple[list[Any], list[Any]]:
    """Use unchanged author HF signal providers with an explicitly supplied local model."""
    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")
    providers: Any = importlib.import_module("lm_wm_tools.watermarks.signal_providers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    model, info = transformers.AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if info["missing_keys"]:
        raise RuntimeError("AutoMark auxiliary model has missing checkpoint weights")
    model.train(False)
    torch.set_num_threads(4)
    device = torch.device(args.model_device)
    model.to(device)
    token_provider = providers.TokenLogprobSignalProvider()
    row_provider = providers._TopKRowCore(width=8)
    for provider in (token_provider, row_provider._logprob_provider):
        provider._model_name = str(args.model_dir)
        provider._device = device
        provider._model = model
        provider._tokenizer = tokenizer
    # Calling these methods directly prevents the outer provider from silently
    # replacing a failed auxiliary computation with all-zero rows.
    token_scores = token_provider._via_local_lm(batches, str(args.model_dir), device)
    rows = row_provider._via_local_lm(batches, str(args.model_dir), device)
    return token_scores, rows


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in cast("dict[str, Any]", value).items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in cast("list[Any]", value)]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# Keep archived research runners compatible with the original helper name.
_json_value = json_value


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.benchmark_alimark_reference import load_inputs

    assess(0.0, 1.0, None, args.alpha)
    texts, manifest = load_inputs(args.input_dir.resolve())
    config_path = args.input_dir / "watermark.json"
    generation = manifest["generation"]
    if sha(config_path) != generation["watermark_config_sha256"]:
        raise ValueError("AutoMark frozen watermark configuration hash mismatch")
    payload = json.loads(config_path.read_bytes())
    if payload["logit_processor_config"]["watermark_class"] != generation["scheme"]:
        raise ValueError("AutoMark generation scheme does not match detector configuration")
    if generation["sampling"] != {"temperature": 1.0, "top_k": 50, "top_p": 1.0}:
        raise ValueError("AutoMark replay requires the declared temperature=1, top_k=50, top_p=1 sampling")
    if generation.get("sampling_after_processor") is not True:
        raise ValueError("AutoMark sampling after the author processor must remain stochastic")
    if payload["seed"] != generation["key"]:
        raise ValueError("AutoMark generation key does not match detector configuration")
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to record the reference revision")
    revision = subprocess.run(  # noqa: S603 - fixed read-only git invocation, no shell
        [git, "-C", str(args.reference_root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != generation["reference_revision"]:
        raise ValueError("AutoMark reference revision does not match the frozen generation")
    for directory in reversed(args.dependency_dir):
        sys.path.insert(0, str(directory.resolve()))
    vocab_size = int(generation["vocab_size"])
    model_config = json.loads((args.model_dir / "config.json").read_bytes())
    actual_vocabulary = model_config.get("text_config", model_config)["vocab_size"]
    if actual_vocabulary != vocab_size:
        raise ValueError("AutoMark generation vocabulary does not match the local model config")
    proof = validate_assets(
        args.model_dir / "tokenizer.json",
        args.reference_tokenizer,
        args.group_asset,
        args.canon_asset,
        vocab_size,
    )
    for name, path in [("group_asset_sha256", args.group_asset), ("canon_asset_sha256", args.canon_asset)]:
        if sha(path) != generation[name]:
            raise ValueError(f"AutoMark {name} differs from frozen generation")
    for name, digest in generation["model_files_sha256"].items():
        if sha(args.model_dir / name) != digest:
            raise ValueError(f"AutoMark local model file hash mismatch: {name}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    detector, _config = load_reference(args, payload, vocab_size)
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.model_dir), local_files_only=True, trust_remote_code=False
    )
    token_rows = [tokenizer.encode(text, add_special_tokens=False) for text in texts.values()]
    forbidden = set(proof["forbidden_token_ids"])
    if any(not row or forbidden.intersection(row) for row in token_rows):
        raise ValueError("AutoMark input contains empty or incompatible tokenizer IDs")
    weighted = payload["logit_processor_config"]["watermark_class"] == "ConcordMark"
    signals, rows = auxiliary_signals(args, token_rows) if weighted else ([], [])
    scores: dict[str, Any] = {}
    for index, (name, tokens) in enumerate(zip(texts, token_rows, strict=True)):
        kwargs = {}
        if weighted:
            ids, lps = rows[index]
            lp = signals[index].numpy()
            validate_auxiliary(tokens, lp, ids, lps, vocab_size)
            kwargs = {"token_logprobs": lp, "row_ids": ids, "row_lps": lps}
        output = detector.detect(tokens, **kwargs)
        if any(
            not math.isfinite(float(score))
            for score, active in zip(output.tokens_scores, output.mask, strict=True)
            if active
        ):
            raise ValueError("AutoMark detector returned a nonfinite scored token")
        pvalue = float(output.pvalue)
        assess(pvalue, 1.0, None, args.alpha)
        scores[name] = {"pvalue": pvalue, "raw": _json_value(output.to_dict())}
    report = {
        "scheme": payload["logit_processor_config"]["watermark_class"],
        "reference_revision": revision,
        "generation_manifest": manifest,
        "asset_proof": proof,
        "reference_files_sha256": {
            str(p.relative_to(args.reference_root)): sha(p)
            for p in sorted((args.reference_root / "lm-wm-library/selected/src").rglob("*.py"))
        },
        "scores": scores,
        "alpha": args.alpha,
        "status": assess(
            scores["source"]["pvalue"], scores["baseline"]["pvalue"], scores.get("after", {}).get("pvalue"), args.alpha
        ),
        "auxiliary_model_device": args.model_device,
        "execution_bridge": (
            "CPU author detectors. ConcordMark uses unchanged author HF signal bodies with the supplied local "
            "image/text model; no vLLM generation, remote APIs or zero-signal fallback."
        ),
        "versions": {
            name: importlib.import_module("importlib.metadata").version(name)
            for name in ("torch", "transformers", "numpy", "numba", "scipy", "msgspec")
        },
        "notice": (
            "Local synthetic reference evidence, not production/provider coverage, calibrated FPR or semantic "
            "equivalence. Raw unscored nonfinite diagnostic values serialize as null."
        ),
    }
    (args.output_dir / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-root",
        "reference-tokenizer",
        "model-dir",
        "group-asset",
        "canon-asset",
        "input-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, action="append", default=[])
    parser.add_argument("--model-device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--alpha", type=float, default=0.01)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(parser.parse_args())
    log.info("AutoMark reference replay: %s", report["status"])


if __name__ == "__main__":
    main()
