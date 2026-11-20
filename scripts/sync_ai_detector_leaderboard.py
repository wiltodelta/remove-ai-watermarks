#!/usr/bin/env python3
"""Synchronize `wiltodelta/ai-image-detector-leaderboard` and audit external detector arenas.

Keeps the public 20-model Hugging Face Space leaderboard aligned with the tracked
`docs/source-classify-hf/metrics.json` manifest, optionally recomputing the 578-image
open benchmark row from a frozen `.npz` bundle on CPU, exporting predictions for
external evaluation arenas (`AI Detector Arena`), and checking external surfaces
for version or baseline drift.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_SRC = REPO_ROOT / "src"
if (_REPO_SRC / "remove_ai_watermarks").is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from remove_ai_watermarks._internal.utils import staged_sibling  # noqa: E402
from remove_ai_watermarks.source_classify import (  # noqa: E402
    MODEL_SHA256,
    WEIGHTS_REPO,
    WEIGHTS_REVISION,
)

log = logging.getLogger(__name__)

LEADERBOARD_SPACE_REPO = "wiltodelta/ai-image-detector-leaderboard"
HF_BLOG_URL = "https://huggingface.co/blog/wiltodelta/ai-image-detector-false-positives-benchmark"
AIDETECTARENA_HOME_URL = "https://aidetectarena.com"
AIDETECTARENA_VERSIONS_URL = (
    "https://raw.githubusercontent.com/AI-Detect-Arena/benchmark-dataset/main/dataset_versions.md"
)
AIDETECTARENA_ISSUE_API_URL = "https://api.github.com/repos/AI-Detect-Arena/benchmark-dataset/issues/1"

CARD_DIR = REPO_ROOT / "docs" / "source-classify-hf"
LEADERBOARD_DIR = REPO_ROOT / "docs" / "ai-image-detector-leaderboard"
LEADERBOARD_JSON_REL = "data/leaderboard.json"
SPACE_TRACKED_FILES: tuple[str, ...] = (
    "README.md",
    "index.html",
    "app.js",
    "styles.css",
    LEADERBOARD_JSON_REL,
)
SPACE_RETIRED_FILES: tuple[str, ...] = ("style.css",)

EXPECTED_NON_AI_COUNT = 200
EXPECTED_NATIVE_AI_COUNT = 125
EXPECTED_ARENA_AI_COUNT = 253


def _atomic_write_text(path: Path, content: str) -> None:
    """Write UTF-8 text atomically via `staged_sibling` with default permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with staged_sibling(path) as staged:
        staged.write_text(content, encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Refusing empty JSON file: {path}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(data).__name__}")
    return data


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _load_benchmark_module() -> Any:
    bench_path = CARD_DIR / "benchmark.py"
    spec = importlib.util.spec_from_file_location("raiw_open_benchmark", bench_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark module from {bench_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_ast_str(node: ast.AST | None, const_vars: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return const_vars.get(node.id)
    return None


def _extract_pinned_baselines_from_source(
    leaderboard_dir: Path = LEADERBOARD_DIR,
    *,
    leaderboard: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Extract all pinned third-party `(repo_id, revision)` pairs from `leaderboard.json` and `benchmark.py`."""
    pairs: dict[str, str] = {}
    lb = leaderboard
    if lb is None:
        lb_file = leaderboard_dir / LEADERBOARD_JSON_REL
        if lb_file.is_file():
            lb = _load_json(lb_file)
    if lb is not None:
        for row in lb.get("results", []):
            model_id = str(row.get("model_id") or "").split(" + ", 1)[0].strip()
            rev = str(row.get("revision") or "").strip()
            if model_id and model_id != WEIGHTS_REPO and re.fullmatch(r"[0-9a-f]{10,40}", rev):
                pairs[model_id] = rev

    tree = ast.parse((CARD_DIR / "benchmark.py").read_text(encoding="utf-8"))
    const_vars: dict[str, str] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            val = _resolve_ast_str(stmt.value, const_vars)
            if val is not None:
                const_vars[stmt.targets[0].id] = val
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            val = _resolve_ast_str(stmt.value, const_vars)
            if val is not None:
                const_vars[stmt.target.id] = val

    for name, repo_val in const_vars.items():
        if name.endswith("_REPO") and repo_val != WEIGHTS_REPO:
            rev_val = const_vars.get(f"{name[:-5]}_REVISION")
            if rev_val and re.fullmatch(r"[0-9a-f]{10,40}", rev_val):
                pairs[repo_val] = rev_val

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kw_map = {kw.arg: _resolve_ast_str(kw.value, const_vars) for kw in node.keywords if kw.arg}
            repo_val = kw_map.get("repo_id")
            rev_val = kw_map.get("revision")
            if repo_val and repo_val != WEIGHTS_REPO and rev_val and re.fullmatch(r"[0-9a-f]{10,40}", rev_val):
                pairs[repo_val] = rev_val
    return pairs


def _artifact_exists_at_hub_revision(artifact_file: str, revision: str) -> bool:
    """Return True when `artifact_file` already exists on Hub at `revision`."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        log.info(
            "HF API request: file_exists repo_id=%s filename=%s revision=%s",
            WEIGHTS_REPO,
            artifact_file,
            revision[:12],
        )
        exists = bool(
            api.file_exists(
                repo_id=WEIGHTS_REPO,
                filename=artifact_file,
                revision=revision,
                repo_type="model",
            )
        )
        log.info("HF API response: file_exists=%s", exists)
        return exists
    except Exception as exc:
        log.warning("Could not verify %s at %s: %s", artifact_file, revision[:12], type(exc).__name__)
        return False


def resolve_canonical_raiw_revision(
    *,
    weights_path: Path | None = None,
    explicit_revision: str | None = None,
) -> str:
    """Resolve the immutable Hub revision for the active or newly published weights."""
    cleaned = (explicit_revision or "").strip()
    if cleaned and cleaned not in ("(dry-run)", "(unchanged)"):
        if weights_path is not None and weights_path.is_file():
            digest = _sha256_file(weights_path).lower()
            if (
                digest == MODEL_SHA256.lower()
                and WEIGHTS_REVISION != "main"
                and _artifact_exists_at_hub_revision(weights_path.name, WEIGHTS_REVISION)
            ):
                return WEIGHTS_REVISION
        return cleaned
    return WEIGHTS_REVISION


def verify_leaderboard_parity(
    leaderboard: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Verify `results[0]` in `leaderboard.json` matches `metrics.json` `open_benchmark_v2_578`."""
    if leaderboard.get("non_ai_count") != EXPECTED_NON_AI_COUNT:
        raise ValueError(f"Unexpected non_ai_count={leaderboard.get('non_ai_count')!r}")
    if leaderboard.get("native_ai_count") != EXPECTED_NATIVE_AI_COUNT:
        raise ValueError(f"Unexpected native_ai_count={leaderboard.get('native_ai_count')!r}")
    if leaderboard.get("arena_ai_count") != EXPECTED_ARENA_AI_COUNT:
        raise ValueError(f"Unexpected arena_ai_count={leaderboard.get('arena_ai_count')!r}")

    results = leaderboard.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("leaderboard.json has no results array")
    raiw = results[0]
    if raiw.get("model_id") != WEIGHTS_REPO:
        raise ValueError(f"Expected results[0].model_id={WEIGHTS_REPO!r}, got {raiw.get('model_id')!r}")
    revision = str(raiw.get("revision") or "").strip()
    if len(revision) != 40:
        raise ValueError(f"Expected 40-char immutable git SHA in results[0].revision, got {revision!r}")

    ob = metrics.get("open_benchmark_v2_578")
    if not isinstance(ob, dict):
        raise ValueError("metrics.json is missing open_benchmark_v2_578")

    checks: tuple[tuple[str, float, float, float], ...] = (
        ("weights_mb", float(raiw["weights_mb"]), float(metrics["artifact_size_mb"]), 0.05),
        ("non_ai_fp", float(raiw["non_ai_fp"]), float(ob["non_ai_fp"]), 0.0),
        ("ai_recall_pct", float(raiw["ai_recall_pct"]), round(float(ob["overall_ai_recall"]) * 100.0, 1), 0.11),
        (
            "provider_attribution_pct",
            float(raiw["provider_attribution_pct"]),
            round(float(ob["overall_provider_attribution"]) * 100.0, 1),
            0.11,
        ),
        (
            "native_ai.recall_pct",
            float(raiw["native_ai"]["recall_pct"]),
            round(float(ob["native_ai_synthetic_recall"]) * 100.0, 1),
            0.11,
        ),
        (
            "native_ai.provider_attribution_pct",
            float(raiw["native_ai"]["provider_attribution_pct"]),
            round(float(ob["native_ai_provider_attribution"]) * 100.0, 1),
            0.11,
        ),
        (
            "arena_ai.recall_pct",
            float(raiw["arena_ai"]["recall_pct"]),
            round(float(ob["arena_ai_synthetic_recall"]) * 100.0, 1),
            0.11,
        ),
        (
            "arena_ai.provider_attribution_pct",
            float(raiw["arena_ai"]["provider_attribution_pct"]),
            round(float(ob["arena_ai_provider_attribution"]) * 100.0, 1),
            0.11,
        ),
        ("precision_pct", float(raiw["precision_pct"]), round(float(ob["overall_precision"]) * 100.0, 1), 0.11),
    )
    for field_name, actual, expected, tol in checks:
        if abs(actual - expected) > tol:
            raise ValueError(
                f"Leaderboard parity mismatch for {field_name}: leaderboard={actual} vs metrics.json={expected}"
            )


def evaluate_raiw_entry(
    *,
    weights_path: Path | None = None,
    revision: str,
) -> dict[str, Any]:
    """Run the 578-image open benchmark on CPU for `wiltodelta/raiw-source-classify`."""
    from huggingface_hub import HfApi, HfFileSystem

    bm = _load_benchmark_module()
    log.info("HF API request: streaming 578 pinned benchmark slices via HfFileSystem/HfApi")
    samples = bm.load_all_samples(HfFileSystem(), HfApi())
    log.info("HF API response: loaded %d deduplicated benchmark samples", len(samples))
    return bm.evaluate_raiw(samples, weights_path=weights_path, revision=revision)


def query_raiw_hub_timestamps(
    *,
    artifact_file: str,
    revision: str,
) -> tuple[str, str, float] | None:
    """Query Hugging Face Hub commit timestamps for `wiltodelta/raiw-source-classify`."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        log.info("HF API request: list_repo_commits repo_id=%s revision=main", WEIGHTS_REPO)
        head_commits = list(api.list_repo_commits(WEIGHTS_REPO, repo_type="model"))
        if not head_commits:
            return None
        repo_updated_at = head_commits[0].created_at.isoformat()
        weights_commit = next((c for c in head_commits if c.commit_id == revision), head_commits[0])
        weights_updated_at = weights_commit.created_at.isoformat()
        weights_epoch = float(weights_commit.created_at.timestamp())
        log.info(
            "HF API response: repo_id=%s artifact=%s head=%s weights_updated_at=%s repo_updated_at=%s",
            WEIGHTS_REPO,
            artifact_file,
            head_commits[0].commit_id[:12],
            weights_updated_at,
            repo_updated_at,
        )
        return weights_updated_at, repo_updated_at, weights_epoch
    except Exception as exc:
        log.warning("Failed to query Hub commit timestamps for %s: %s", WEIGHTS_REPO, type(exc).__name__)
        return None


def stage_leaderboard_space(
    dest: Path,
    *,
    leaderboard_dir: Path = LEADERBOARD_DIR,
    metrics_path: Path = CARD_DIR / "metrics.json",
    metrics: dict[str, Any] | None = None,
    override_leaderboard: dict[str, Any] | None = None,
) -> None:
    """Validate parity against `metrics.json` and stage Space files into `dest`."""
    missing = [str(leaderboard_dir / rel) for rel in SPACE_TRACKED_FILES if not (leaderboard_dir / rel).is_file()]
    if missing:
        raise SystemExit(f"missing leaderboard Space files: {', '.join(missing)}")

    loaded_metrics = metrics or _load_json(metrics_path)
    lb_data = override_leaderboard or _load_json(leaderboard_dir / LEADERBOARD_JSON_REL)
    verify_leaderboard_parity(lb_data, loaded_metrics)

    for rel in SPACE_TRACKED_FILES:
        out_file = dest / rel
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if rel == LEADERBOARD_JSON_REL and override_leaderboard is not None:
            _atomic_write_text(out_file, json.dumps(override_leaderboard, indent=2) + "\n")
        else:
            shutil.copy2(leaderboard_dir / rel, out_file)


def publish_leaderboard_space(
    stage: Path,
    *,
    token: str,
    message: str,
    dry_run: bool = False,
) -> str:
    """Upload the staged leaderboard Space to Hugging Face Hub."""
    if dry_run:
        staged = sorted(str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file())
        log.info(
            "dry-run: staged leaderboard Space %s files=%s delete=%s",
            LEADERBOARD_SPACE_REPO,
            staged,
            list(SPACE_RETIRED_FILES),
        )
        return "(dry-run)"

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    log.info(
        "HF API request: create_repo repo_id=%s repo_type=space space_sdk=static private=False",
        LEADERBOARD_SPACE_REPO,
    )
    api.create_repo(
        LEADERBOARD_SPACE_REPO,
        repo_type="space",
        space_sdk="static",
        exist_ok=True,
        private=False,
    )
    log.info(
        "HF API request: upload_folder repo_id=%s repo_type=space allow_patterns=%s delete_patterns=%s message=%r",
        LEADERBOARD_SPACE_REPO,
        list(SPACE_TRACKED_FILES),
        list(SPACE_RETIRED_FILES),
        message,
    )
    commit = api.upload_folder(
        folder_path=str(stage),
        repo_id=LEADERBOARD_SPACE_REPO,
        repo_type="space",
        commit_message=message,
        allow_patterns=list(SPACE_TRACKED_FILES),
        delete_patterns=list(SPACE_RETIRED_FILES),
    )
    oid = getattr(commit, "oid", "") or ""
    log.info("HF API response: upload_folder repo_id=%s oid=%s", LEADERBOARD_SPACE_REPO, oid or "(unchanged)")
    return oid


def _http_get_text(url: str, *, accept: str = "text/html,application/json;q=0.9,*/*;q=0.8") -> tuple[int, str]:
    log.info("HTTP GET request: url=%s", url)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "remove-ai-watermarks-benchmark-sync/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        status = int(resp.status)
        body = resp.read().decode("utf-8", errors="replace")
    log.info("HTTP GET response: url=%s status=%d bytes=%d", url, status, len(body))
    return status, body


def check_external_arenas(
    *,
    metrics_path: Path = CARD_DIR / "metrics.json",
    metrics: dict[str, Any] | None = None,
    leaderboard: dict[str, Any] | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Audit live external benchmark, blog, and arena surfaces for version or baseline drift."""
    from huggingface_hub import HfApi

    loaded_metrics = metrics or _load_json(metrics_path)
    expected_artifact = str(loaded_metrics["artifact_file"])
    expected_sha256 = str(loaded_metrics["artifact_sha256"])
    canonical_rev = resolve_canonical_raiw_revision(explicit_revision=expected_revision)

    def _audit_space() -> dict[str, Any]:
        live_url = f"https://huggingface.co/spaces/{LEADERBOARD_SPACE_REPO}/resolve/main/{LEADERBOARD_JSON_REL}"
        try:
            _st, raw_lb = _http_get_text(live_url, accept="application/json")
            live_lb = json.loads(raw_lb)
            live_raiw = live_lb["results"][0]
            live_rev = str(live_raiw.get("revision") or "")
            parity_ok = True
            parity_err = None
            try:
                verify_leaderboard_parity(live_lb, loaded_metrics)
            except ValueError as exc:
                parity_ok = False
                parity_err = str(exc)
            return {
                "url": f"https://huggingface.co/spaces/{LEADERBOARD_SPACE_REPO}",
                "owned": True,
                "up_to_date": bool(live_rev == canonical_rev and parity_ok),
                "live_revision": live_rev,
                "expected_revision": canonical_rev,
                "parity_error": parity_err,
            }
        except Exception as exc:
            log.warning("External check failed for %s: %s", LEADERBOARD_SPACE_REPO, exc)
            return {
                "url": f"https://huggingface.co/spaces/{LEADERBOARD_SPACE_REPO}",
                "owned": True,
                "up_to_date": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _audit_blog() -> dict[str, Any]:
        try:
            _status, blog_html = _http_get_text(HF_BLOG_URL)
            found_artifacts = sorted(set(re.findall(r"multi-provider-source-v\d+\.npz", blog_html)))
            has_current = expected_artifact in blog_html and expected_sha256[:16] in blog_html
            stale_artifacts = [a for a in found_artifacts if a != expected_artifact]
            return {
                "url": HF_BLOG_URL,
                "owned": True,
                "up_to_date": bool(has_current and not stale_artifacts),
                "expected_artifact": expected_artifact,
                "found_artifacts": found_artifacts,
                "has_expected_sha256_prefix": expected_sha256[:16] in blog_html,
            }
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("External check failed for %s: %s", HF_BLOG_URL, exc)
            return {
                "url": HF_BLOG_URL,
                "owned": True,
                "up_to_date": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _audit_arena() -> dict[str, Any]:
        try:
            _st_v, versions_md = _http_get_text(AIDETECTARENA_VERSIONS_URL, accept="text/plain")
            ver_matches = re.findall(r"\*\*(\d+\.\d+)\*\*", versions_md)
            latest_dataset_ver = ver_matches[0] if ver_matches else "unknown"
            _st_i, issue_raw = _http_get_text(AIDETECTARENA_ISSUE_API_URL, accept="application/vnd.github+json")
            issue_obj = json.loads(issue_raw)
            issue_body = str(issue_obj.get("body") or "")
            issue_artifacts = sorted(set(re.findall(r"multi-provider-source-v\d+\.npz", issue_body)))
            _st_h, home_html = _http_get_text(AIDETECTARENA_HOME_URL)
            listed_live = "raiw-source-classify" in home_html.lower() or "remove-ai-watermarks" in home_html.lower()
            return {
                "url": AIDETECTARENA_HOME_URL,
                "issue_url": str(issue_obj.get("html_url") or ""),
                "owned": False,
                "latest_dataset_version": latest_dataset_ver,
                "listed_on_live_leaderboard": listed_live,
                "issue_state": str(issue_obj.get("state") or ""),
                "issue_comments": int(issue_obj.get("comments") or 0),
                "issue_artifacts": issue_artifacts,
                "issue_matches_current_artifact": expected_artifact in issue_artifacts,
            }
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.warning("External check failed for AI Detector Arena: %s", exc)
            return {
                "url": AIDETECTARENA_HOME_URL,
                "owned": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    pinned_baselines = _extract_pinned_baselines_from_source(leaderboard=leaderboard)
    api = HfApi()

    def _check_baseline(item: tuple[str, str]) -> dict[str, str] | None:
        repo_id, pinned_sha = item
        try:
            log.info("HF API request: repo_info repo_id=%s", repo_id)
            info = api.repo_info(repo_id=repo_id, repo_type="model")
            head_sha = str(info.sha or "")
            log.info("HF API response: repo_id=%s head=%s pinned=%s", repo_id, head_sha[:12], pinned_sha[:12])
            if head_sha and not head_sha.startswith(pinned_sha):
                return {"repo_id": repo_id, "pinned": pinned_sha, "head": head_sha}
        except Exception as exc:
            log.warning("Baseline revision check skipped for %s: %s", repo_id, type(exc).__name__)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        fut_space = pool.submit(_audit_space)
        fut_blog = pool.submit(_audit_blog)
        fut_arena = pool.submit(_audit_arena)
        baseline_results = list(pool.map(_check_baseline, sorted(pinned_baselines.items())))
        surfaces: dict[str, dict[str, Any]] = {
            "hf_leaderboard_space": fut_space.result(),
            "hf_blog_article": fut_blog.result(),
            "ai_detector_arena": fut_arena.result(),
        }

    drifted_baselines = [item for item in baseline_results if item is not None]
    surfaces["upstream_hf_baselines"] = {
        "checked_count": len(pinned_baselines),
        "drifted_count": len(drifted_baselines),
        "drifted": drifted_baselines,
    }

    owned_stale = [
        name for name, item in surfaces.items() if item.get("owned") is True and not item.get("up_to_date", False)
    ]
    return {
        "schema": str(loaded_metrics["schema"]),
        "expected_artifact": expected_artifact,
        "expected_sha256": expected_sha256,
        "expected_revision": canonical_rev,
        "owned_surfaces_up_to_date": len(owned_stale) == 0,
        "owned_stale_surfaces": owned_stale,
        "surfaces": surfaces,
    }


def export_aidetectarena_submission(
    dataset_dir: Path,
    out_dir: Path,
    *,
    weights_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate `predictions.csv`, official metrics, and an issue update for AI Detector Arena."""
    from remove_ai_watermarks import __version__
    from remove_ai_watermarks import source_classify as sc

    meta_csv = dataset_dir / "metadata" / "images_metadata.csv"
    if not meta_csv.is_file():
        meta_csv = dataset_dir / "dataset_metadata" / "images_metadata.csv"
    if not meta_csv.is_file():
        raise FileNotFoundError(f"images_metadata.csv not found under {dataset_dir}")

    resolved_weights = sc.resolve_weights_path(weights_path)
    weights_sha = _sha256_file(resolved_weights)
    rev = resolve_canonical_raiw_revision(weights_path=resolved_weights)

    pred_rows: list[tuple[str, str, float]] = []
    tp = tn = fp = fn = 0
    with meta_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            img_path = dataset_dir / row["filename"].strip()
            if not img_path.is_file():
                continue
            image_id = row["id"].strip()
            gt_ai = row["is_ai"].strip().lower() == "true"
            res = sc.classify_source(img_path, weights_path=resolved_weights)
            pred_label = "ai" if res.synthetic_detected else "real"
            prob = float(res.synthetic_probability)
            conf = prob if res.synthetic_detected else (1.0 - prob)
            pred_rows.append((image_id, pred_label, round(conf, 6)))
            if gt_ai and res.synthetic_detected:
                tp += 1
            elif gt_ai and not res.synthetic_detected:
                fn += 1
            elif (not gt_ai) and res.synthetic_detected:
                fp += 1
            else:
                tn += 1

    total = tp + tn + fp + fn
    if total == 0:
        raise ValueError(f"Zero evaluable images found under {dataset_dir}")
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2.0 * prec * rec / max(1e-9, prec + rec)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, tp + fn)
    combined = 100.0 * (0.5 * f1 + 0.3 * (1.0 - fpr) + 0.2 * (1.0 - fnr))

    summary: dict[str, Any] = {
        "package_version": __version__,
        "artifact_file": resolved_weights.name,
        "artifact_sha256": weights_sha,
        "hub_revision": rev,
        "total_images": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy_pct": round(100.0 * (tp + tn) / total, 2),
        "precision_pct": round(100.0 * prec, 2),
        "recall_pct": round(100.0 * rec, 2),
        "f1_pct": round(100.0 * f1, 2),
        "fpr_pct": round(100.0 * fpr, 2),
        "fnr_pct": round(100.0 * fnr, 2),
        "combined_score": round(combined, 2),
    }

    if dry_run:
        log.info("dry-run: evaluated %d AI Detector Arena images (%s); skipping file writes", total, summary)
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf, lineterminator="\n")
    writer.writerow(["image_id", "prediction", "confidence"])
    writer.writerows(pred_rows)
    _atomic_write_text(out_dir / "predictions.csv", csv_buf.getvalue())
    _atomic_write_text(out_dir / "arena_report.json", json.dumps(summary, indent=2) + "\n")

    md_note = (
        f"Updated `{WEIGHTS_REPO}` results for `{resolved_weights.name}` "
        f"(`remove-ai-watermarks=={__version__}`, Hub revision `{rev}`):\n\n"
        f"- Accuracy: {summary['accuracy_pct']:.2f}% (TP {tp}, TN {tn}, FP {fp}, FN {fn})\n"
        f"- Precision: {summary['precision_pct']:.2f}%, Recall: {summary['recall_pct']:.2f}%, "
        f"F1: {summary['f1_pct']:.2f}%\n"
        f"- FPR: {summary['fpr_pct']:.2f}%, FNR: {summary['fnr_pct']:.2f}%, "
        f"Combined Score: {summary['combined_score']:.2f}\n"
    )
    _atomic_write_text(out_dir / "issue_update.md", md_note)
    return summary


def format_arena_markdown_summary(report: dict[str, Any]) -> str:
    """Format `check_external_arenas` output as a compact Markdown status table."""
    lines = [
        f"### Benchmark and external arena sync (`{report['expected_artifact']}`)",
        "",
        "| Surface | Status | Details |",
        "|---|---|---|",
    ]
    sf = report["surfaces"]
    sp = sf.get("hf_leaderboard_space", {})
    sp_ok = "up to date" if sp.get("up_to_date") else "STALE / ERROR"
    sp_det = sp.get("error") or sp.get("parity_error") or f"revision `{str(sp.get('live_revision', ''))[:12]}`"
    lines.append(f"| `{LEADERBOARD_SPACE_REPO}` | {sp_ok} | {sp_det} |")

    bl = sf.get("hf_blog_article", {})
    bl_ok = "up to date" if bl.get("up_to_date") else "NEEDS EDIT"
    bl_det = bl.get("error") or f"artifacts `{bl.get('found_artifacts', [])}`"
    lines.append(f"| HF Blog report | {bl_ok} | {bl_det} |")

    ar = sf.get("ai_detector_arena", {})
    if "error" in ar:
        lines.append(f"| `aidetectarena.com` | check failed | {ar['error']} |")
    else:
        ar_status = "listed" if ar.get("listed_on_live_leaderboard") else f"issue `{ar.get('issue_state')}`"
        ar_det = (
            f"dataset `v{ar.get('latest_dataset_version')}`, "
            f"issue artifacts `{ar.get('issue_artifacts', [])}`, "
            f"comments `{ar.get('issue_comments', 0)}`"
        )
        lines.append(f"| `aidetectarena.com` | {ar_status} | {ar_det} |")

    up = sf.get("upstream_hf_baselines", {})
    lines.append(
        f"| Upstream HF baselines (`benchmark.py`) | checked {up.get('checked_count', 0)} | "
        f"{up.get('drifted_count', 0)} repos have newer `main` commits than pinned SHAs |"
    )
    return "\n".join(lines) + "\n"


def _build_competitor_chart_cards(lb: dict[str, Any]) -> str:
    """Build the 7 competitor comparison rows dynamically from `leaderboard.json`."""
    by_id = {str(r["model_id"]): r for r in lb.get("results", [])}
    card_defs: tuple[tuple[str, str, str, str | None], ...] = (
        (
            "abacusai/Smaug-AI-Image-Detector",
            "abacusai/Smaug-AI-Image-Detector",
            "EmbeddingGemma-2 · Apache-2.0 · Oct 2026 · Binary only",
            None,
        ),
        (
            "TensorFold/detectra-v1",
            "TensorFold/detectra-v1",
            "ViT-S384 · MIT · Detectra v3 · Binary only",
            "0% UI",
        ),
        (
            "bendavidsteel/openfake-swinv2-base + attribution-head",
            "openfake-swinv2 + attribution-head",
            "SwinV2 · CC-BY-NC-4.0 (non-commercial) · 15 families",
            None,
        ),
        (
            "Thermostatic/community-forensics-frontier-detector-2026-08",
            "Thermostatic/commfor-frontier-2026-08",
            "ViT-S384 · MIT · Aug 2026 · Binary only",
            None,
        ),
        (
            "Organika/sdxl-detector",
            "Organika/sdxl-detector",
            "Swin · CC-BY-NC-3.0 · 12.7k dl/mo · Binary only",
            None,
        ),
        (
            "umm-maybe/AI-image-detector",
            "umm-maybe/AI-image-detector",
            "Swin · CC-BY-4.0 · 64.9k dl/mo (#1 HF) · Binary only",
            None,
        ),
        (
            "Ateeqq/ai-vs-human-image-detector",
            "Ateeqq/ai-vs-human-image-detector",
            "SigLIP · Apache-2.0 · 11.5k dl/mo · Binary only",
            None,
        ),
    )
    slice_names = {
        "camera_photo": "camera photos",
        "ui_screenshot": "UI screenshots",
        "science_diagram": "science diagrams",
        "scanned_document": "scanned docs",
    }
    cards: list[str] = []
    for model_id, display_name, meta_tail, worst_suffix in card_defs:
        row = by_id[model_id]
        w_mb = float(row["weights_mb"])
        fpr = float(row["non_ai_fpr_pct"])
        fp_n = int(row["non_ai_fp"])
        tot_n = int(row["non_ai_total"])
        nat_attr = float(row["native_ai"]["provider_attribution_pct"])
        sf = row["slice_fpr_pct"]
        worst_key, worst_val = max(
            ((k, float(sf[k])) for k in slice_names),
            key=lambda kv: kv[1],
        )
        w_a = max(4, round(fpr * 4.4))
        w_r = max(4, round(worst_val * 4.4))
        t_a = f"{fpr:.1f}% overall FPR ({fp_n}/{tot_n})"
        if nat_attr > 0.0:
            t_a += f" · {nat_attr:.1f}% native attr (NC)"
        t_r = f"{worst_val:.1f}% on {slice_names[worst_key]}"
        if worst_suffix:
            t_r += f" ({worst_suffix})"
        cards.append(
            f'    <div class="card"><div><div class="m-name">{display_name}</div>'
            f'<div class="m-meta">{w_mb:.1f} MB · {meta_tail}</div></div><div class="bars">'
            f'<div class="bar-line"><span class="bar bar-amber" style="width:{w_a}px"></span>'
            f'<span class="txt-amber">{t_a}</span></div>'
            f'<div class="bar-line"><span class="bar bar-red" style="width:{w_r}px"></span>'
            f'<span class="txt-red">{t_r}</span></div></div></div>'
        )
    return "\n".join(cards)


def render_blog_chart_png(
    out_png: Path,
    *,
    leaderboard_path: Path = LEADERBOARD_DIR / LEADERBOARD_JSON_REL,
    metrics_path: Path = CARD_DIR / "metrics.json",
    leaderboard: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> Path:
    """Render the 2400x1260 OpenGraph / blog comparison chart from `leaderboard.json`."""
    lb = leaderboard or _load_json(leaderboard_path)
    loaded_metrics = metrics or _load_json(metrics_path)
    raiw = lb["results"][0]
    schema = str(loaded_metrics["schema"])
    ver_match = re.search(r"(v\d+)$", schema)
    ver_tag = ver_match.group(1) if ver_match else schema
    w_mb = float(raiw["weights_mb"])
    nat_rec = float(raiw["native_ai"]["recall_pct"])
    arn_rec = float(raiw["arena_ai"]["recall_pct"])
    nat_attr = float(raiw["native_ai"]["provider_attribution_pct"])
    prec = float(raiw["precision_pct"])
    fp_cnt = int(raiw["non_ai_fp"])
    non_ai_tot = int(raiw["non_ai_total"])
    fpr_pct = float(raiw["non_ai_fpr_pct"])

    font_url = (
        "https://fonts.googleapis.com/css2?"
        "family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap"
    )
    hero_pill = (
        f"{fpr_pct:.1f}% non-AI FPR ({fp_cnt}/{non_ai_tot}) · "
        f"{nat_rec:.1f}% native AI · {arn_rec:.1f}% arena AI · "
        f"{nat_attr:.1f}% attribution · {prec:.1f}% precision"
    )
    comp_cards = _build_competitor_chart_cards(lb)
    eyebrow = "OPEN FORENSIC BENCHMARK · 578 THIRD-PARTY IMAGES (200 NON-AI + 125 NATIVE AI + 253 ARENA AI)"
    subtitle = (
        "Pinned Hugging Face shards (ImageNet, ScreenSpot, AI2D, DocVQA, "
        "Native Exports, Rapidata) · 0 training hash overlaps"
    )
    footer_left = (
        "20 open models tested: OwensLab/commfor-384 (0.0% FPR, MIT) · "
        "detectra-openfake (0.0% FPR, NC) · Bombek1 740M (3.5% FPR, 10% UI)"
    )

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<link href="{font_url}" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    width: 1200px; height: 630px; background: #0b111e; color: #f8fafc;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
    padding: 28px 48px 22px; display: flex; flex-direction: column;
    justify-content: space-between; -webkit-font-smoothing: antialiased;
  }}
  .top-row {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; }}
  .eyebrow {{ font-size: 12px; font-weight: 600; letter-spacing: 0.04em; color: #38bdf8; margin-bottom: 6px; }}
  h1 {{ font-size: 28px; line-height: 1.18; font-weight: 700; color: #f8fafc; letter-spacing: -0.015em; }}
  .sub {{ margin-top: 8px; font-size: 13px; color: #94a3b8; }}
  .legend {{
    background: #101827; border: 1px solid #1e293b; border-radius: 8px;
    padding: 10px 14px; display: flex; flex-direction: column; gap: 6px; min-width: 288px;
  }}
  .leg-item {{ display: flex; align-items: center; gap: 8px; font-size: 11.5px; color: #cbd5e1; }}
  .dot {{ width: 10px; height: 10px; border-radius: 3px; flex: none; }}
  .dot-green {{ background: #10b981; }}
  .dot-amber {{ background: #f59e0b; }}
  .dot-red {{ background: #ef4444; }}
  .rows {{ display: flex; flex-direction: column; gap: 6px; margin-top: 4px; }}
  .card {{
    display: grid; grid-template-columns: 424px 1fr; column-gap: 16px; align-items: center;
    padding: 7px 16px; border-radius: 7px; background: #0f172a; border: 1px solid #1e293b;
  }}
  .card-hero {{ background: #082721; border: 1.5px solid #10b981; padding: 9px 16px; }}
  .m-name {{ font-family: "IBM Plex Mono", monospace; font-size: 15px; font-weight: 500; color: #e2e8f0; }}
  .card-hero .m-name {{ color: #34d399; }}
  .m-meta {{ font-size: 11px; color: #94a3b8; margin-top: 2px; white-space: nowrap; }}
  .card-hero .m-meta {{ color: #a7f3d0; opacity: 0.88; }}
  .hero-pill {{
    background: #062f26; border: 1px solid #10b981; border-radius: 6px;
    padding: 8px 14px; font-size: 12.5px; font-weight: 600; color: #ecfdf5; white-space: nowrap;
  }}
  .bars {{ display: flex; flex-direction: column; gap: 4px; }}
  .bar-line {{ display: flex; align-items: center; gap: 8px; font-size: 11.5px; font-weight: 600; }}
  .bar {{ height: 11px; border-radius: 3px; flex: none; }}
  .bar-amber {{ background: #f59e0b; }}
  .bar-red {{ background: #ef4444; }}
  .txt-amber {{ color: #f8fafc; }}
  .txt-red {{ color: #fca5a5; }}
  .footer {{
    display: flex; justify-content: space-between; align-items: center;
    font-size: 12px; color: #94a3b8; padding-top: 2px;
  }}
  .footer-url {{ font-weight: 600; color: #cbd5e1; }}
</style>
</head>
<body>
  <div class="top-row">
    <div>
      <div class="eyebrow">{eyebrow}</div>
      <h1>2026 open AI image detectors: false positives vs<br>commercial 10-generator attribution</h1>
      <div class="sub">{subtitle}</div>
    </div>
    <div class="legend">
      <div class="leg-item">
        <span class="dot dot-green"></span><span>0.0% non-AI FPR + Apache-2.0 attribution</span>
      </div>
      <div class="leg-item"><span class="dot dot-amber"></span><span>Overall non-AI FPR (200 real images)</span></div>
      <div class="leg-item"><span class="dot dot-red"></span><span>Worst slice FPR (UI / Diagram / Photo)</span></div>
    </div>
  </div>
  <div class="rows">
    <div class="card card-hero">
      <div>
        <div class="m-name">wiltodelta/raiw-source-classify ({ver_tag})</div>
        <div class="m-meta">{w_mb:.1f} MB · NumPy + ONNX CPU · Apache-2.0 · 10 families + 31 subfamilies</div>
      </div>
      <div class="hero-pill">{hero_pill}</div>
    </div>
{comp_cards}
  </div>
  <div class="footer">
    <span>{footer_left}</span>
    <span class="footer-url">huggingface.co/wiltodelta/raiw-source-classify</span>
  </div>
</body>
</html>"""

    if dry_run:
        log.info("dry-run: validated blog chart HTML for %s (%s); skipping PNG write", ver_tag, out_png)
        return out_png

    from playwright.sync_api import sync_playwright

    out_png.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel="chrome")
        ctx = browser.new_context(viewport={"width": 1200, "height": 630}, device_scale_factor=2)
        page = ctx.new_page()
        page.set_content(html, wait_until="load")
        page.evaluate("() => document.fonts.ready")
        page.screenshot(path=str(out_png), full_page=False)
        browser.close()
    log.info("Rendered blog chart PNG to %s (%d bytes)", out_png, out_png.stat().st_size)
    return out_png


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=None, help="Optional frozen `.npz` weights file.")
    parser.add_argument("--raiw-revision", default=None, help="Hub revision OID for the published weights.")
    parser.add_argument(
        "--recompute-raiw",
        action="store_true",
        help="Re-run the 578-image open benchmark on CPU and update `data/leaderboard.json`.",
    )
    parser.add_argument(
        "--refresh-hub-dates",
        action="store_true",
        help="Refresh `weights_updated_at` and `repo_updated_at` on `results[0]` from Hugging Face Hub.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Validate and upload `docs/ai-image-detector-leaderboard/` to Hugging Face Space.",
    )
    parser.add_argument(
        "--message",
        default="Sync Open AI Image Detector Leaderboard",
        help="Commit message when `--publish` uploads to Hugging Face Space.",
    )
    parser.add_argument(
        "--check-external-arenas",
        action="store_true",
        help="Audit live HF leaderboard, HF blog report, AI Detector Arena, and baseline revisions.",
    )
    parser.add_argument(
        "--fail-on-stale",
        action="store_true",
        help="Exit non-zero if any owned public benchmark surface lags `metrics.json`.",
    )
    parser.add_argument("--report-json", type=Path, default=None, help="Optional JSON path for external arena report.")
    parser.add_argument("--summary-md", type=Path, default=None, help="Optional Markdown summary output file.")
    parser.add_argument("--export-aidetectarena", type=Path, default=None, help="Path to AI Detector Arena dataset.")
    parser.add_argument("--arena-out", type=Path, default=None, help="Output directory for AI Detector Arena CSV/MD.")
    parser.add_argument("--render-blog-chart", type=Path, default=None, help="Render 2400x1260 PNG chart for HF blog.")
    parser.add_argument("--dry-run", action="store_true", help="Roll back local edits and perform zero Hub writes.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_path = CARD_DIR / "metrics.json"
    metrics = _load_json(metrics_path)
    lb_json_path = LEADERBOARD_DIR / LEADERBOARD_JSON_REL
    lb_data = _load_json(lb_json_path)

    canonical_rev = resolve_canonical_raiw_revision(
        weights_path=args.weights,
        explicit_revision=args.raiw_revision,
    )

    mutated_lb = False
    if args.recompute_raiw:
        entry = evaluate_raiw_entry(weights_path=args.weights, revision=canonical_rev)
        prev = lb_data["results"][0]
        for ts_key in ("weights_updated_at", "repo_updated_at", "weights_updated_epoch"):
            if ts_key in prev:
                entry[ts_key] = prev[ts_key]
        lb_data["results"][0] = entry
        mutated_lb = True

    if args.refresh_hub_dates or args.recompute_raiw:
        ts_tuple = query_raiw_hub_timestamps(
            artifact_file=str(metrics["artifact_file"]),
            revision=canonical_rev,
        )
        if ts_tuple is not None:
            w_at, r_at, w_ep = ts_tuple
            lb_data["results"][0]["revision"] = canonical_rev
            lb_data["results"][0]["weights_updated_at"] = w_at
            lb_data["results"][0]["repo_updated_at"] = r_at
            lb_data["results"][0]["weights_updated_epoch"] = w_ep
            mutated_lb = True

    verify_leaderboard_parity(lb_data, metrics)

    if mutated_lb and not args.dry_run:
        _atomic_write_text(lb_json_path, json.dumps(lb_data, indent=2) + "\n")
        log.info("Updated %s (revision=%s)", lb_json_path, lb_data["results"][0]["revision"])
    elif mutated_lb:
        log.info("dry-run: verified updated leaderboard payload without modifying %s", lb_json_path)

    if args.publish:
        token = os.environ.get("HF_TOKEN", "").strip()
        if not args.dry_run and not token:
            raise SystemExit("HF_TOKEN is not set (write role required for --publish)")
        with tempfile.TemporaryDirectory(prefix="raiw-leaderboard-hf-") as tmp:
            stage_dir = Path(tmp)
            stage_leaderboard_space(
                stage_dir,
                leaderboard_dir=LEADERBOARD_DIR,
                metrics_path=metrics_path,
                metrics=metrics,
                override_leaderboard=lb_data,
            )
            oid = publish_leaderboard_space(
                stage_dir,
                token=token,
                message=args.message,
                dry_run=args.dry_run,
            )
        log.info("published %s revision=%s", LEADERBOARD_SPACE_REPO, oid or "(unchanged)")

    if args.render_blog_chart is not None:
        render_blog_chart_png(
            args.render_blog_chart,
            leaderboard=lb_data,
            metrics=metrics,
            dry_run=args.dry_run,
        )

    if args.export_aidetectarena is not None:
        if args.arena_out is None and not args.dry_run:
            raise SystemExit("--arena-out is required with --export-aidetectarena unless --dry-run is set")
        arena_out = args.arena_out or Path(tempfile.gettempdir())
        export_aidetectarena_submission(
            args.export_aidetectarena,
            arena_out,
            weights_path=args.weights,
            dry_run=args.dry_run,
        )

    if args.check_external_arenas:
        report = check_external_arenas(
            metrics_path=metrics_path,
            metrics=metrics,
            leaderboard=lb_data,
            expected_revision=canonical_rev,
        )
        md_summary = format_arena_markdown_summary(report)
        sys.stdout.write(md_summary)
        if args.report_json is not None and not args.dry_run:
            _atomic_write_text(args.report_json, json.dumps(report, indent=2) + "\n")
        if args.summary_md is not None and not args.dry_run:
            _atomic_write_text(args.summary_md, md_summary)
        if args.fail_on_stale and not report["owned_surfaces_up_to_date"]:
            log.error("Stale owned benchmark surfaces detected: %s", report["owned_stale_surfaces"])
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
