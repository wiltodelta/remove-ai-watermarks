"""Replay synthetic inputs with the external author's PostMark word selector and detector.

Local Nomic CPU inference and inspected model code are required. No API clients
are instantiated and no model or data files are downloaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_texts(
    texts: dict[str, str], embedder: Any, compute_presence: Callable[..., Any], cosine_threshold: float
) -> dict[str, Any]:
    """Recompute the response-dependent detection words for every input."""
    scores: dict[str, Any] = {}
    for name, text in texts.items():
        words: Any = embedder.get_words(text)
        if (
            not isinstance(words, list)
            or not words
            or any(not isinstance(word, str) or not word for word in cast("list[object]", words))
        ):
            raise ValueError(f"PostMark {name} selected no valid detection words")
        score = compute_presence(text, words, threshold=cosine_threshold)
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"PostMark {name} returned an invalid presence score")
        scores[name] = {"words": words, "presence": float(score)}
    return scores


def load_reference(args: argparse.Namespace) -> tuple[Any, Callable[..., Any], dict[str, Any]]:
    """Load author code unchanged, redirecting only encoder loading to local CPU files."""
    reference = args.reference_root.resolve()
    encoder = args.embedder.resolve()
    tokenizer_path = args.tokenizer.resolve()
    encoder_code = args.encoder_code.resolve()
    dependency_dirs = [path.resolve() for path in args.dependency_dir]
    nltk_data = args.nltk_data.resolve()
    required = [reference / "postmark/models.py", reference / "postmark/utils.py"]
    required += [reference / name for name in ("paragram_xxl.pkl", "paragram_xxl_words.json", "wikitext_freq.json")]
    required += [reference / "valid_wtmk_words_in_wiki_base-only-f1000.pkl"]
    required += [reference / "filtered_data_100k_unique_250w_sentbound_nomic_embs.pkl"]
    required += [encoder / "config.json", encoder / "model.safetensors"]
    required += [encoder_code / name for name in ("configuration_hf_nomic_bert.py", "modeling_hf_nomic_bert.py")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    reference_hashes = {
        str(path.relative_to(reference)): _hash(path) for path in required if path.is_relative_to(reference)
    }
    local_hashes = {
        label: {str(p.relative_to(directory)): _hash(p) for p in sorted(directory.rglob("*")) if p.is_file()}
        for label, directory in [("embedder", encoder), ("tokenizer", tokenizer_path), ("encoder_code", encoder_code)]
    }
    for path in reversed(dependency_dirs):
        sys.path.insert(0, str(path))
    sys.path.insert(0, str(reference / "postmark"))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    os.environ["NLTK_DATA"] = str(nltk_data)
    os.environ["HF_MODULES_CACHE"] = str(args.output_dir.resolve() / "module-cache")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(args.tiktoken_cache.resolve())
    nltk: Any = importlib.import_module("nltk")
    torch: Any = importlib.import_module("torch")
    transformers: Any = importlib.import_module("transformers")

    torch.set_num_threads(4)
    nltk.data.path.insert(0, str(nltk_data))
    local = args.output_dir.resolve() / "encoder-runtime"
    local.mkdir()
    for path in encoder.iterdir():
        if path.is_file() and path.name != "config.json":
            (local / path.name).symlink_to(path)
    config = json.loads((encoder / "config.json").read_bytes())
    config["auto_map"] = {key: value.split("--")[-1] for key, value in config["auto_map"].items()}
    (local / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for path in encoder_code.glob("*.py"):
        shutil.copy2(path, local / path.name)
    original_tokenizer = transformers.AutoTokenizer.from_pretrained
    original_model = transformers.AutoModel.from_pretrained

    def tokenizer_local(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name != "bert-base-uncased":
            raise ValueError(f"Unexpected author tokenizer request: {name}")
        return original_tokenizer(str(tokenizer_path), local_files_only=True)

    def model_local(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name != "nomic-ai/nomic-embed-text-v1":
            raise ValueError(f"Unexpected author encoder request: {name}")
        return original_model(
            str(local),
            trust_remote_code=True,
            local_files_only=True,
            use_flash_attn=False,
            fused_bias_fc=False,
            fused_dropout_add_ln=False,
            use_xentropy=False,
        )

    transformers.AutoTokenizer.from_pretrained = tokenizer_local
    transformers.AutoModel.from_pretrained = model_local
    previous_cwd = Path.cwd()
    try:
        os.chdir(reference)
        # Fresh CLI process required: the author's generic module names must not
        # silently resolve to a different package already loaded by a caller.
        if "models" in sys.modules or "utils" in sys.modules:
            raise RuntimeError("Run the PostMark replay in a fresh Python process")
        models = importlib.import_module("models")
        utils = importlib.import_module("utils")
        embedder = models.NomicEmbed(ratio=args.ratio)
    finally:
        os.chdir(previous_cwd)
        transformers.AutoTokenizer.from_pretrained = original_tokenizer
        transformers.AutoModel.from_pretrained = original_model
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to record the author revision")
    revision = subprocess.run(  # noqa: S603 - fixed git command, caller-selected local checkout
        [git, "-C", str(reference), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata = {
        "reference_revision": revision,
        "reference_hashes": reference_hashes,
        "local_files_sha256": local_hashes,
        "word_table_sha256": hashlib.sha256(embedder.embedding_table.cpu().numpy().tobytes()).hexdigest(),
        "word_table_shape": list(embedder.embedding_table.shape),
        "encoder_device": str(embedder.device),
        "execution_bridge": (
            "Local Nomic CPU nonfused kernels; config auto_map redirected to supplied inspected code. "
            "Author selector and presence functions unchanged. No paid model clients or generation run."
        ),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "spacy", "nltk", "tiktoken")
        },
    }
    return embedder, utils.compute_presence, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.benchmark_alimark_reference import assess_scores, load_inputs

    for name in ("ratio", "cosine_threshold", "threshold"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1 or (name == "ratio" and value == 0)):
            raise ValueError(f"Invalid PostMark {name}: expected a finite fraction")
    texts, manifest = load_inputs(args.input_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    embedder, compute_presence, metadata = load_reference(args)
    generation = manifest["generation"]
    if "ratio" in generation and generation["ratio"] != args.ratio:
        raise ValueError("PostMark generation ratio does not match the replay")
    if "table_sha256" in generation and generation["table_sha256"] != metadata["word_table_sha256"]:
        raise ValueError("PostMark generation word table does not match the replay")
    scores = score_texts(texts, embedder, compute_presence, args.cosine_threshold)
    report = {
        "scheme": "PostMark author Nomic selector and Paragram presence replay",
        **metadata,
        "generation_manifest": manifest,
        "ratio": args.ratio,
        "cosine_threshold": args.cosine_threshold,
        "threshold": args.threshold,
        "scores": scores,
        "status": assess_scores(
            scores["source"]["presence"],
            scores["baseline"]["presence"],
            scores.get("after", {}).get("presence"),
            threshold=args.threshold,
            retained_prose_spans=manifest.get("retained_prose_spans", 0),
        ),
        "notice": (
            "Numerical author-reference evidence only. Recomputed words per output. "
            "No production/provider verdict, calibrated FPR or semantic certificate."
        ),
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference-root",
        "embedder",
        "encoder-code",
        "tokenizer",
        "nltk-data",
        "tiktoken-cache",
        "input-dir",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--dependency-dir", type=Path, action="append", default=[])
    parser.add_argument("--ratio", type=float, default=0.12)
    parser.add_argument("--cosine-threshold", type=float, default=0.7)
    parser.add_argument(
        "--threshold", type=float, help="Explicit experimental decision threshold; not cosine similarity"
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(parser.parse_args())
    log.info("PostMark reference replay: %s", report["status"])


if __name__ == "__main__":
    main()
