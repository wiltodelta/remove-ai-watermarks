"""Replay one content-addressed AutoMark submission on frozen local text.

Supports explicit candidate-001 through candidate-025 and candidate-037 through
candidate-048 source states, plus the declared AAR and SynthID baseline trees.
This research-key diagnostic does not certify meaning or private-provider removal.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
log = logging.getLogger(__name__)


SUPPORTED_SUBMISSIONS = {
    "candidate-001": "CascadeCopula",
    "candidate-002": "CascadeCopula",
    "candidate-003": "PortfolioRace",
    "candidate-004": "PartitionRace",
    "candidate-005": "SphericalRenewal",
    "candidate-006": "HaarRenewal",
    "candidate-007": "HaarGamma",
    "candidate-008": "HaarPredictive",
    "candidate-009": "HaarGraph",
    "candidate-010": "HaarLexical",
    "candidate-011": "HaarLexicalFirst",
    "candidate-012": "ProjectiveFirst",
    "candidate-013": "AxialFirst",
    "candidate-014": "RotationFirst",
}
LEXICAL_SUBMISSIONS = frozenset(SUPPORTED_SUBMISSIONS) - {"candidate-001"}
FRESHRACE_SUBMISSIONS = {
    "candidate-015": "lm_wm_tools.watermarks.logits_processors.freshrace.FreshRaceWatermark",
    "candidate-016": "lm_wm_tools.watermarks.logits_processors.freshrace.SelectiveFreshRaceWatermark",
    "candidate-017": "lm_wm_tools.watermarks.logits_processors.freshrace.DeploymentFreshRaceWatermark",
    "candidate-018": "lm_wm_tools.watermarks.logits_processors.freshrace.ChannelFreshRaceWatermark",
    "candidate-019": "lm_wm_tools.watermarks.logits_processors.freshrace.AdaptiveChannelFreshRaceWatermark",
    "candidate-020": "lm_wm_tools.watermarks.logits_processors.freshrace_grouped.GroupedFreshRaceWatermark",
    "candidate-021": (
        "lm_wm_tools.watermarks.logits_processors.freshrace_antithetic.AntitheticReservoirFreshRaceWatermark"
    ),
    "candidate-022": "lm_wm_tools.watermarks.logits_processors.freshrace_antithetic.BalancedFreshRaceWatermark",
    "candidate-023": "lm_wm_tools.watermarks.logits_processors.freshrace_lexical.LexicalFreshRaceWatermark",
    "candidate-024": "lm_wm_tools.watermarks.logits_processors.freshrace_lexical.LexicalFreshRaceWatermark",
    "candidate-025": "lm_wm_tools.watermarks.logits_processors.renewalrace.RenewalRaceWatermark",
}
SUPPORTED_SUBMISSIONS.update(FRESHRACE_SUBMISSIONS)
SUPPORTED_SUBMISSIONS.update(
    {
        f"candidate-{i:03}": "lm_wm_tools.watermarks.logits_processors.tallymark.TallyMarkWatermark"
        for i in range(37, 41)
    }
)
BASELINE_SUBMISSIONS = {"baseline-AAR-context3": "AAR", "baseline-SynthID-context3": "SynthID"}
SUPPORTED_SUBMISSIONS.update(BASELINE_SUBMISSIONS)
SUPPORTED_SUBMISSIONS.update(
    {
        "candidate-041": "lm_wm_tools.watermarks.logits_processors.choirmark.ChoirMarkWatermark",
        "candidate-042": "lm_wm_tools.watermarks.logits_processors.canonmark.CanonMarkWatermark",
        "candidate-043": "lm_wm_tools.watermarks.logits_processors.laddermark.LadderMarkWatermark",
        "candidate-044": "lm_wm_tools.watermarks.logits_processors.preludemark.PreludeMarkWatermark",
        "candidate-045": "lm_wm_tools.watermarks.logits_processors.consortmark.ConsortMarkWatermark",
        "candidate-046": "lm_wm_tools.watermarks.logits_processors.sestetmark.SestetMarkWatermark",
        "candidate-047": "lm_wm_tools.watermarks.logits_processors.motetmark.MotetMarkWatermark",
        "candidate-048": "lm_wm_tools.watermarks.logits_processors.codamark.CodaMarkWatermark",
    }
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_files(source: Path) -> dict[str, str]:
    """Hash a materialized tree without following symbolic links."""
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Baseline source root must be a directory without symlinks")
    result: dict[str, str] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError("Baseline source tree must not contain symlinks")
        if path.is_file():
            result[path.relative_to(source).as_posix()] = sha(path)
    return result


def _verify_baseline_tree(store: Path, source: Path, row: dict[str, Any], entry: dict[str, Any]) -> dict[str, str]:
    """Bind a declared baseline to its original project, configuration and tree."""
    if row.get("kind") != "baseline" or entry.get("kind") != "baseline":
        raise ValueError("Baseline source kind differs from the manifest")
    if row["source_project"] != entry["source_project"] or row["configuration_path"] != entry["config"]:
        raise ValueError("Baseline source/configuration declaration differs from the manifest")
    original = store / row["source_project"]
    if not original.resolve().is_relative_to(store.resolve().parent):
        raise ValueError("Baseline source project escapes the pinned reference")
    original_files = _tree_files(original)
    digest = hashlib.sha256(json.dumps(original_files, sort_keys=True).encode()).hexdigest()
    if digest != row["source_tree_sha256"]:
        raise ValueError("Original baseline source-tree hash mismatch")
    if _tree_files(source) != original_files:
        raise ValueError("Materialized baseline source tree has changed, missing or extra files")
    config = row["configuration_path"]
    if config not in original_files or original_files[config] != row["configuration_sha256"]:
        raise ValueError("Baseline configuration is outside the tree or has a hash mismatch")
    return original_files


def verify_submission(store: Path, source: Path, inventory: Path, identifier: str) -> dict[str, Any]:
    """Check the inventory anchor and exact candidate objects or declared baseline tree."""
    catalog = json.loads(inventory.read_bytes())
    if sha(store / "manifest.json") != catalog["manifest_sha256"]:
        raise ValueError("Submission manifest hash mismatch")
    rows = [row for row in catalog["submissions"] if row["id"] == identifier]
    if len(rows) != 1:
        raise ValueError("Submission ID is missing or ambiguous")
    row = rows[0]
    manifest = json.loads((store / "manifest.json").read_bytes())
    entries = [entry for entry in manifest["submissions"] if entry["job_id"] == identifier]
    if len(entries) != 1:
        raise ValueError("Submission manifest entry is missing or ambiguous")
    if row.get("kind") == "baseline" or entries[0].get("kind") == "baseline":
        files = _verify_baseline_tree(store, source, row, entries[0])
        return {
            "submission": row,
            "reference_revision": catalog["reference_revision"],
            "files_sha256": files,
            "inventory_sha256": sha(inventory),
            "source_manifest_sha256": catalog["manifest_sha256"],
        }
    mapping_path = store / "files" / f"{identifier}.json"
    if sha(mapping_path) != row["source_map_sha256"]:
        raise ValueError("Submission source-map hash mismatch")
    mapping = json.loads(mapping_path.read_bytes())
    if len(mapping) != entries[0]["file_count"]:
        raise ValueError("Submission file count mismatch")
    base = source.resolve()
    actual: set[str] = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("Materialized submission must not contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(source).as_posix())
    if actual != set(mapping):
        raise ValueError("Materialized submission has missing or extra files")
    for relative, digest in mapping.items():
        path = source / relative
        if not path.resolve().is_relative_to(base) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Unsafe submission object path or digest")
        if any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Invalid submission object digest")
        if sha(store / "objects" / digest) != digest or sha(path) != digest:
            raise ValueError("Submission object hash mismatch")
    if row["configuration_path"] not in mapping:
        raise ValueError("Submission configuration is outside its source map")
    if sha(source / row["configuration_path"]) != row["configuration_sha256"]:
        raise ValueError("Submission configuration hash mismatch")
    return {
        "submission": row,
        "reference_revision": catalog["reference_revision"],
        "files_sha256": mapping,
        "inventory_sha256": sha(inventory),
        "source_manifest_sha256": catalog["manifest_sha256"],
    }


def canonical_tokenizer(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize old string/new pair BPE serialization without changing merge ranks."""
    result = dict(payload)
    model = dict(payload["model"])
    if model.get("type") != "BPE" or not isinstance(model.get("merges"), list):
        raise ValueError("Lexical replay requires an explicit BPE merge table")
    pairs: list[list[str]] = []
    for row in cast("list[Any]", model["merges"]):
        values = row.split(" ") if isinstance(row, str) else row
        if not isinstance(values, list) or len(cast("list[object]", values)) != 2:
            raise ValueError("Invalid BPE merge pair")
        pair = cast("list[object]", values)
        if not all(isinstance(value, str) and value for value in pair):
            raise ValueError("Invalid BPE merge tokens")
        pairs.append([cast("str", value) for value in pair])
    model["merges"] = pairs
    result["model"] = model
    return result


def validate_reference_tokenizer(tokenizer: Path, reference_path: Path, reference: dict[str, Any]) -> None:
    """Require the pinned complete tokenizer; vocabulary length alone is insufficient."""
    if sha(reference_path) != reference["tokenizer_sha256"]:
        raise ValueError("Reference tokenizer hash mismatch")
    candidate_path = tokenizer / "tokenizer.json"
    if canonical_tokenizer(json.loads(candidate_path.read_bytes())) != canonical_tokenizer(
        json.loads(reference_path.read_bytes())
    ):
        raise ValueError("Lexical tokenizer semantics differ from the pinned reference")


def validate_lexical_assets(
    source: Path, tokenizer: Path, reference_path: Path, reference: dict[str, Any], vocab_size: int
) -> dict[str, Any]:
    """Require the published lexical map and the full pinned tokenizer semantics."""
    validate_reference_tokenizer(tokenizer, reference_path, reference)
    candidate_path = tokenizer / "tokenizer.json"
    assets = source / "src/lm_wm_tools/watermarks/assets"
    metadata_path = assets / "llama3_wordnet_groups.json"
    group_path = assets / "llama3_wordnet_groups.npz"
    metadata = json.loads(metadata_path.read_bytes())
    if metadata["tokenizer"] != reference["model"] or metadata["tokenizer_revision"] != reference["revision"]:
        raise ValueError("Lexical asset tokenizer identity mismatch")
    np: Any = importlib.import_module("numpy")
    with np.load(group_path, allow_pickle=False) as data:
        groups, eligible = data["groups"], data["eligible"]
    if (
        groups.shape != (vocab_size,)
        or eligible.shape != (vocab_size,)
        or groups.dtype.kind not in "iu"
        or eligible.dtype.kind != "b"
        or np.any(groups < 0)
        or np.any(groups >= vocab_size)
        or not np.array_equal(groups[groups], groups)
        or metadata["vocabulary_size"] != vocab_size
    ):
        raise ValueError("Lexical group arrays do not match the vocabulary")
    counts = {
        "groups": len(np.unique(groups)),
        "eligible_tokens": int(np.count_nonzero(eligible)),
        "merged_tokens": int(vocab_size - len(np.unique(groups))),
    }
    if any(metadata[key] != value for key, value in counts.items()):
        raise ValueError("Lexical asset summary differs from its arrays")
    return {
        "reference_tokenizer_sha256": sha(reference_path),
        "tokenizer_sha256": sha(candidate_path),
        "group_asset_sha256": sha(group_path),
        "asset_metadata_sha256": sha(metadata_path),
        "vocab_size": vocab_size,
        "counts": counts,
        "canonical_tokenizer_equal": True,
    }


def validate_freshrace_assets(
    source: Path, tokenizer: Path, reference_path: Path, reference: dict[str, Any], vocab_size: int
) -> dict[str, Any]:
    """Bind unchanged FreshRace assets to the declared reference-tokenizer bridge."""
    validate_reference_tokenizer(tokenizer, reference_path, reference)
    payload = json.loads(reference_path.read_bytes())
    ids = set(payload["model"]["vocab"].values()) | {row["id"] for row in payload["added_tokens"]}
    if ids != set(range(vocab_size)):
        raise ValueError("FreshRace reference vocabulary differs from generation")
    np: Any = importlib.import_module("numpy")
    root = source / "src/lm_wm_tools/watermarks/assets"
    pairs = np.load(root / "freshrace_common_pairs.npy", allow_pickle=False)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or not len(pairs) or pairs.dtype != np.dtype("uint64"):
        raise ValueError("Invalid FreshRace common-pair table")
    if np.any(pairs[:, 1] >= vocab_size):
        raise ValueError("FreshRace common-pair token is outside the vocabulary")
    paths = [root / "freshrace_common_pairs.npy"]
    predictable = root / "freshrace_predictable.npy"
    if predictable.exists():
        values = np.load(predictable, allow_pickle=False)
        if values.ndim != 1 or not len(values) or values.dtype != np.dtype("uint64"):
            raise ValueError("Invalid FreshRace predictable hashes")
        paths.append(predictable)
    for name in ("freshrace_adaptive.npz", "freshrace_lexical.npz"):
        path = root / name
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            expected = {"rare", "common", "groups"} if "lexical" in name else {"rare", "common"}
            if set(data.files) != expected:
                raise ValueError("Unexpected FreshRace asset arrays")
            rare, common = data["rare"], data["common"]
            if rare.shape != (vocab_size,) or rare.dtype.kind != "b":
                raise ValueError("Invalid FreshRace rare-token mask")
            if (
                common.ndim != 1
                or not len(common)
                or common.dtype != np.dtype("uint64")
                or np.any(common[1:] < common[:-1])
            ):
                raise ValueError("Invalid FreshRace common hashes")
            if "groups" in expected:
                groups = data["groups"]
                if (
                    groups.shape != (vocab_size,)
                    or groups.dtype.kind not in "iu"
                    or np.any(groups < 0)
                    or np.any(groups >= vocab_size)
                    or not np.array_equal(groups[groups], groups)
                ):
                    raise ValueError("Invalid FreshRace lexical groups")
        paths.append(path)
    return {
        "reference_tokenizer_sha256": sha(reference_path),
        "tokenizer_sha256": sha(tokenizer / "tokenizer.json"),
        "canonical_tokenizer_equal": True,
        "vocab_size": vocab_size,
        "assets_sha256": {path.name: sha(path) for path in paths},
        "embedded_tokenizer_metadata": None,
        "reference_basis": "Author default evaluation model; asset-build tokenizer metadata is not embedded",
    }


def validate_provenance(generation: dict[str, Any], proof: dict[str, Any]) -> None:
    row = proof["submission"]
    tree_based = row.get("kind") == "baseline"
    identity = "source_tree_sha256" if tree_based else "source_map_sha256"
    other_identity = "source_map_sha256" if tree_based else "source_tree_sha256"
    if other_identity in generation:
        raise ValueError("Frozen generation has an incompatible source proof")
    expected = {
        "submission_id": row["id"],
        "reference_revision": proof["reference_revision"],
        identity: row[identity],
        "configuration_sha256": row["configuration_sha256"],
    }
    if any(generation.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen generation does not match the exact submission")
    if SUPPORTED_SUBMISSIONS.get(row["id"]) != row["watermark_class"]:
        raise ValueError("Unsupported exact submission/class pair")
    if row["id"] in LEXICAL_SUBMISSIONS and not isinstance(generation.get("lexical_assets"), dict):
        raise ValueError("Lexical submission requires explicit asset provenance")
    if row["id"] in FRESHRACE_SUBMISSIONS and not isinstance(generation.get("freshrace_assets"), dict):
        raise ValueError("FreshRace submission requires explicit asset provenance")
    if generation.get("sampling_after_processor") is not True:
        raise ValueError("Sampling after the author processor must remain stochastic")
    sampling = generation.get("sampling", {})
    if set(sampling) != {"temperature", "top_k", "top_p"}:
        raise ValueError("Unsupported generation sampling fields")
    for key, value in {"temperature": 1.0, "top_k": 50, "top_p": 1.0}.items():
        if type(sampling.get(key)) not in {int, float} or sampling[key] != value:
            raise ValueError("Unsupported generation sampling settings")
    if type(generation.get("key")) is not int or not 0 <= generation["key"] < 2**64:
        raise ValueError("Research key must be a 64-bit integer")
    if type(generation.get("vocab_size")) is not int or not 0 < generation["vocab_size"] < 2**20:
        raise ValueError("CascadeCopula vocabulary must be smaller than 2**20")


def generation_is_qualified(generation: dict[str, Any], scores: dict[str, Any], texts: dict[str, str]) -> bool:
    for name in ("source", "baseline"):
        data = generation[name + "_generation"]
        if data.get("text") != texts[name] or data.get("text_ids") != scores[name]["raw"]["tokens"]:
            return False
        if data.get("ended_with_eos") is not True or data.get("token_roundtrip_exact") is not True:
            return False
        if data.get("unsupported_output_ids") != [] or sum(scores[name]["raw"]["mask"]) < 64:
            return False
    return True


def assess_submission(
    scores: dict[str, Any], generation: dict[str, Any], texts: dict[str, str], retained: int, alpha: float
) -> str:
    """Keep generation/control failures and partial rewrites distinct."""
    from scripts.benchmark_automark_reference import assess

    if type(retained) is not int or retained < 0:
        raise ValueError("retained_prose_spans must be a nonnegative integer")
    status = assess(
        scores["source"]["pvalue"], scores["baseline"]["pvalue"], scores.get("after", {}).get("pvalue"), alpha
    )
    if not generation_is_qualified(generation, scores, texts):
        return "unqualified_generation"
    if status == "not_detected_at_reference_alpha" and retained:
        return "partial_unverified"
    return status


def require_fresh_namespace() -> None:
    if any(name == "lm_wm_tools" or name.startswith("lm_wm_tools.") for name in sys.modules):
        raise RuntimeError("Use a fresh process for each content-addressed submission")


def require_plain_detection(config: Any) -> None:
    """Reject methods whose declared auxiliary signals this replay cannot supply."""
    signals = config.required_auxiliary_signals()
    if signals:
        raise ValueError(f"Author method requires auxiliary signals: {signals}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.benchmark_alimark_reference import load_inputs
    from scripts.benchmark_automark_reference import json_value

    texts, manifest = load_inputs(args.input_dir)
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory")
    proof = verify_submission(args.source_store, args.submission_dir, args.inventory, args.submission_id)
    generation = manifest["generation"]
    validate_provenance(generation, proof)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if sha(args.tokenizer_dir / name) != generation["tokenizer_files_sha256"][name]:
            raise ValueError("Frozen tokenizer/model configuration hash mismatch")
    if json.loads((args.tokenizer_dir / "config.json").read_bytes())["vocab_size"] != generation["vocab_size"]:
        raise ValueError("Model configuration vocabulary differs from generation")
    config_path = args.input_dir / "watermark.json"
    if sha(config_path) != generation["watermark_config_sha256"]:
        raise ValueError("Frozen watermark configuration hash mismatch")
    yaml: Any = importlib.import_module("yaml")
    expected = yaml.safe_load((args.submission_dir / proof["submission"]["configuration_path"]).read_bytes())
    expected.update(device="cpu", seed=generation["key"])
    payload = json.loads(config_path.read_bytes())
    if payload != expected:
        raise ValueError("Watermark settings differ from the published submission")
    if args.submission_id in LEXICAL_SUBMISSIONS:
        if args.reference_tokenizer is None:
            raise ValueError("Lexical submission needs a pinned reference tokenizer")
        reference = json.loads(args.inventory.read_bytes())["tokenizer_references"]["llama3_wordnet"]
        assets = validate_lexical_assets(
            args.submission_dir, args.tokenizer_dir, args.reference_tokenizer, reference, generation["vocab_size"]
        )
        if assets != generation["lexical_assets"]:
            raise ValueError("Frozen lexical asset/tokenizer provenance mismatch")
    if args.submission_id in FRESHRACE_SUBMISSIONS:
        if args.reference_tokenizer is None:
            raise ValueError("FreshRace submission needs a pinned reference tokenizer")
        reference = json.loads(args.inventory.read_bytes())["tokenizer_references"]["llama3_wordnet"]
        assets = validate_freshrace_assets(
            args.submission_dir, args.tokenizer_dir, args.reference_tokenizer, reference, generation["vocab_size"]
        )
        if assets != generation["freshrace_assets"]:
            raise ValueError("Frozen FreshRace asset/tokenizer provenance mismatch")
    require_fresh_namespace()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ["NUMBA_CACHE_DIR"] = str(args.output_dir / "numba-cache")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.submission_dir.resolve() / "src"))
    author: Any = importlib.import_module("lm_wm_tools.configs.main_config")
    config = author.WatermarkConfig.from_payload(payload)
    require_plain_detection(config)
    detector = config.get_detector(SimpleNamespace(**generation["sampling"]), generation["vocab_size"])
    if (
        not Path(inspect.getfile(cast("type[Any]", type(detector))))
        .resolve()
        .is_relative_to(args.submission_dir.resolve() / "src")
    ):
        raise RuntimeError("Detector was imported from a different source tree")
    transformers: Any = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_dir, local_files_only=True, trust_remote_code=False
    )
    scores: dict[str, Any] = {}
    for name, text in texts.items():
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if not tokens or any(not 0 <= token < generation["vocab_size"] for token in tokens):
            raise ValueError("Frozen text contains an empty or incompatible token sequence")
        result = detector.detect(tokens)
        if not math.isfinite(float(result.pvalue)) or not 0 <= float(result.pvalue) <= 1:
            raise ValueError("Author detector returned an invalid p-value")
        if len(result.mask) != len(tokens) or len(result.tokens_scores) != len(tokens):
            raise ValueError("Author detector token-score alignment mismatch")
        if any(
            not math.isfinite(float(score))
            for score, active in zip(result.tokens_scores, result.mask, strict=True)
            if active
        ):
            raise ValueError("Author detector returned a nonfinite active score")
        scores[name] = {"pvalue": float(result.pvalue), "raw": json_value(result.to_dict())}
    status = assess_submission(scores, generation, texts, manifest.get("retained_prose_spans", 0), args.alpha)
    report = {
        "submission_proof": proof,
        "manifest": manifest,
        "scores": scores,
        "alpha": args.alpha,
        "status": status,
        "quality": "Not established by detector scores",
        "script_sha256": sha(Path(__file__)),
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-store", "submission-dir", "tokenizer-dir", "input-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reference-tokenizer", type=Path)
    parser.add_argument("--submission-id", required=True)
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "data/evaluations/text-watermark/automark-variants.json"
    )
    parser.add_argument("--alpha", type=float, default=0.01)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run(parser.parse_args())
    log.info("AutoMark exact-submission replay: %s", report["status"])


if __name__ == "__main__":
    main()
