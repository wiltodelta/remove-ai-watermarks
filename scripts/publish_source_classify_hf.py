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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

HUB_REPO = "wiltodelta/raiw-source-classify"
RETIRED_HUB_REPOS = (
    "wiltodelta/openai-google-image-source-classifier",
    "wiltodelta/raiw-photo-classify",
)
CONFIG_FILE = "config.json"
MODEL_FILE = "multi-provider-source-v8.npz"
RETIRED_FILES = (
    "source-pipeline-mlp.npz",
    "openai-google-source-v1.npz",
    "multi-provider-source-v3.npz",
    "multi-provider-source-v4.npz",
    "multi-provider-source-v5.npz",
)
REPO_ROOT = Path(__file__).resolve().parents[1]
CARD_DIR = REPO_ROOT / "docs" / "source-classify-hf"
TRACKED_CARD_FILES = (
    "README.md",
    "metrics.json",
    "website-content.json",
    "infer.py",
    "benchmark.py",
    "demo.ipynb",
)
PUBLIC_FILES = (*TRACKED_CARD_FILES, CONFIG_FILE, MODEL_FILE)
WEBSITE_SECTIONS = ("## Model architecture", "## Evaluation")
FORBIDDEN_VISITOR_COPY = (
    "<",
    ">",
    "`",
    "{",
    "}",
)
FORBIDDEN_VISITOR_PATTERN = (
    r"\b(?:sudo|pip install|brew install|npm install|MCP|CLI|CUDA|API)\b"
    r"|receipt|invoice|raiw-photo-classify"
)


def validate_website_contract(card_dir: Path) -> None:
    """Enforce wiltodelta-app's import contract before staging Hub files."""
    import re

    card = (card_dir / "README.md").read_text(encoding="utf-8")
    for heading in WEBSITE_SECTIONS:
        if heading not in card:
            raise SystemExit(f"README.md missing personal-website section: {heading}")
    value = json.loads((card_dir / "website-content.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or sorted(value) != ["intro", "requirements", "sections"]:
        raise SystemExit("website-content.json: invalid visitor copy shape")
    forbidden_re = re.compile(FORBIDDEN_VISITOR_PATTERN)

    def check_plain(text: Any, limit: int) -> None:
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > limit
            or any(ch in text for ch in FORBIDDEN_VISITOR_COPY)
            or forbidden_re.search(text)
        ):
            raise SystemExit("website-content.json: visitor copy violates plain-text or scope rules")

    check_plain(value["intro"], 400)
    check_plain(value["requirements"], 600)
    sections = value["sections"]
    if not isinstance(sections, list) or not (1 <= len(sections) <= 6):
        raise SystemExit("website-content.json: invalid visitor copy sections")
    for section in sections:
        if not isinstance(section, list) or len(section) != 2:
            raise SystemExit("website-content.json: invalid visitor section")
        check_plain(section[0], 80)
        check_plain(section[1], 650)
    if re.search(r"receipt|invoice|raiw-photo-classify", json.dumps(value), re.IGNORECASE):
        raise SystemExit("website-content.json: prohibited term in visitor copy")


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
        "library_name": "onnx",
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
    validate_website_contract(CARD_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        shutil.copy2(path, dest / name)
    metrics = json.loads((CARD_DIR / "metrics.json").read_text(encoding="utf-8"))
    (dest / CONFIG_FILE).write_text(
        json.dumps(_build_hub_config(metrics), indent=2) + "\n",
        encoding="utf-8",
    )


def publish(stage: Path, *, token: str, message: str, dry_run: bool = False) -> str:
    """Upload the exact staged allowlist, delete retired weights/repos, and return the Hub revision."""
    if dry_run:
        staged_files = sorted(path.name for path in stage.iterdir() if path.is_file())
        log.info(
            "dry-run: staged %s for %s (allow=%s, delete=%s)",
            staged_files,
            HUB_REPO,
            list(PUBLIC_FILES),
            list(RETIRED_FILES),
        )
        return "(dry-run)"

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    log.info("HF API request: create_repo repo_id=%s repo_type=model private=False", HUB_REPO)
    api.create_repo(HUB_REPO, repo_type="model", exist_ok=True, private=False)
    log.info(
        "HF API request: upload_folder repo_id=%s allow_patterns=%s delete_patterns=%s message=%r",
        HUB_REPO,
        list(PUBLIC_FILES),
        list(RETIRED_FILES),
        message,
    )
    commit = api.upload_folder(
        folder_path=str(stage),
        repo_id=HUB_REPO,
        repo_type="model",
        commit_message=message,
        allow_patterns=list(PUBLIC_FILES),
        delete_patterns=list(RETIRED_FILES),
    )
    oid = getattr(commit, "oid", "") or ""
    log.info("HF API response: upload_folder repo_id=%s oid=%s", HUB_REPO, oid or "(unchanged)")
    for retired_repo in RETIRED_HUB_REPOS:
        log.info("HF API request: delete_repo repo_id=%s repo_type=model missing_ok=True", retired_repo)
        api.delete_repo(retired_repo, repo_type="model", missing_ok=True)
    return oid


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--message", default="Publish source-pipeline classifier")
    parser.add_argument("--revision-out", type=Path, default=None, help="Optional file to record the published OID.")
    parser.add_argument("--dry-run", action="store_true", help="Stage and validate in a temp dir without Hub writes.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not args.dry_run and not token:
        raise SystemExit("HF_TOKEN is not set (write role required)")
    with tempfile.TemporaryDirectory(prefix="raiw-source-hf-") as tmp:
        stage = Path(tmp)
        stage_release(stage, args.src)
        revision = publish(stage, token=token, message=args.message, dry_run=args.dry_run)
    if args.revision_out is not None and not args.dry_run:
        args.revision_out.parent.mkdir(parents=True, exist_ok=True)
        args.revision_out.write_text(revision + "\n", encoding="utf-8")
    log.info("published %s revision=%s", HUB_REPO, revision or "(unknown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
