#!/usr/bin/env python3
"""Publish the source-pipeline model to Hugging Face.

The derivative weight file is downloaded by the GitHub Action from a model
freeze release. The private corpus, feature caches, and training catalog never
enter this repository or the Hub staging directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

from hf_publish import publish_model_folder

log = logging.getLogger(__name__)

HUB_REPO = "wiltodelta/raiw-source-classify"
RETIRED_HUB_REPOS = ("wiltodelta/openai-google-image-source-classifier",)
MODEL_FILE = "multi-provider-source-v4.npz"
RETIRED_FILES = (
    "source-pipeline-mlp.npz",
    "openai-google-source-v1.npz",
    "multi-provider-source-v3.npz",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = REPO_ROOT / "docs" / "source-classify-hf"
PUBLIC_FILES = ("README.md", "metrics.json", MODEL_FILE)


def stage_release(dest: Path, src: Path) -> None:
    """Stage the two tracked documents and one derivative model artifact."""
    sources = {
        "README.md": CARD_DIR / "README.md",
        "metrics.json": CARD_DIR / "metrics.json",
        MODEL_FILE: src.expanduser().resolve() / MODEL_FILE,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing publication files: {', '.join(missing)}")
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        shutil.copy2(path, dest / name)


def publish(stage: Path, *, token: str, message: str) -> str:
    """Upload the exact staged allowlist, delete retired weights/repos, and return the Hub revision."""
    from huggingface_hub import HfApi

    revision = publish_model_folder(
        stage,
        repo_id=HUB_REPO,
        token=token,
        message=message,
        allow_patterns=PUBLIC_FILES,
        delete_patterns=RETIRED_FILES,
    )
    api = HfApi(token=token)
    for retired_repo in RETIRED_HUB_REPOS:
        api.delete_repo(retired_repo, repo_type="model", missing_ok=True)
    return revision


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
