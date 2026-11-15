#!/usr/bin/env python3
"""Publish the source-pipeline model to Hugging Face.

The derivative weight file is downloaded by the GitHub Action from a model
freeze release. The private corpus, feature caches, and training catalog never
enter this repository or the Hub staging directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

HUB_REPO = "wiltodelta/raiw-source-classify"
RETIRED_HUB_REPOS = (
    "wiltodelta/openai-google-image-source-classifier",
    "wiltodelta/raiw-photo-classify",
)
CONFIG_FILE = "config.json"
MODEL_FILE = "multi-provider-source-v6.npz"
RETIRED_FILES = (
    "source-pipeline-mlp.npz",
    "openai-google-source-v1.npz",
    "multi-provider-source-v3.npz",
    "multi-provider-source-v4.npz",
    "multi-provider-source-v5.npz",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = REPO_ROOT / "docs" / "source-classify-hf"
TRACKED_CARD_FILES = ("README.md", "metrics.json", "infer.py", "benchmark.py", "demo.ipynb")
PUBLIC_FILES = (*TRACKED_CARD_FILES, CONFIG_FILE, MODEL_FILE)


def _build_hub_config(metrics: dict[str, Any]) -> dict[str, Any]:
    """Synthesize Hugging Face Hub ``config.json`` from the tracked metrics manifest."""
    classes = [str(item) for item in metrics["classes"]]
    return {
        "model_type": "raiw-source-classify",
        "schema": metrics["schema"],
        "artifact_file": metrics["artifact_file"],
        "artifact_sha256": metrics["artifact_sha256"],
        "feature_dimension": metrics["feature_dimension"],
        "stacked_stage2_dimension": metrics["stacked_stage2_dimension"],
        "providers": [cls for cls in classes if cls != "unknown"],
        "abstain_label": "unknown",
        "library_name": "numpy",
        "package": "remove-ai-watermarks[source-classify]",
        "license": "apache-2.0",
    }


def stage_release(dest: Path, src: Path) -> None:
    """Stage tracked metadata, synthesized Hub config, and one derivative model artifact."""
    sources = {name: CARD_DIR / name for name in TRACKED_CARD_FILES}
    sources[MODEL_FILE] = src.expanduser().resolve() / MODEL_FILE
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing publication files: {', '.join(missing)}")
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        shutil.copy2(path, dest / name)
    metrics = json.loads((CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    (dest / CONFIG_FILE).write_text(
        json.dumps(_build_hub_config(metrics), indent=2) + "\n",
        encoding="utf-8",
    )


def publish(stage: Path, *, token: str, message: str) -> str:
    """Upload the exact staged allowlist, delete retired weights/repos, and return the Hub revision."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(HUB_REPO, repo_type="model", exist_ok=True, private=False)
    commit = api.upload_folder(
        folder_path=str(stage),
        repo_id=HUB_REPO,
        repo_type="model",
        commit_message=message,
        allow_patterns=list(PUBLIC_FILES),
        delete_patterns=list(RETIRED_FILES),
    )
    for retired_repo in RETIRED_HUB_REPOS:
        api.delete_repo(retired_repo, repo_type="model", missing_ok=True)
    return getattr(commit, "oid", "") or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--message", default="Publish source-pipeline classifier")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set (write role required)")
    with tempfile.TemporaryDirectory(prefix="raiw-source-hf-") as tmp:
        stage = Path(tmp)
        stage_release(stage, args.src)
        revision = publish(stage, token=token, message=args.message)
    log.info("published %s revision=%s", HUB_REPO, revision or "(unknown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
