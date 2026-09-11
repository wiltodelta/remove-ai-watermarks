#!/usr/bin/env python3
"""Publish the photo-classify freeze to Hugging Face wiltodelta/raiw-photo-classify.

Manual. Not a library release step. Weights are not in git: pass --src, a
directory that holds clip-l-ft.pt, clip-l-ft-vision-fp32.onnx, and the probe,
with detector.pt and provider.pt either at the root or in run1/.

    uv run python scripts/publish_photo_classify_hf.py --src DIR
    uv run python scripts/publish_photo_classify_hf.py --card-only

Needs HF_TOKEN with write role. The GitHub Action
publish-photo-classify-hf.yml injects secrets.HF_TOKEN.

Card mode uploads the README, operating-point.json, and the receipt-gate
head (staged from the package assets) without touching the freeze weights.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

HUB_REPO = "wiltodelta/raiw-photo-classify"
CLIP_FILE = "clip-l-ft.pt"
CLIP_ONNX_FILE = "clip-l-ft-vision-fp32.onnx"
PROBE_FILE = "probe-weights-clip-l-ft.npz"
DETECTOR_FILE = "detector.pt"
PROVIDER_FILE = "provider.pt"
WEIGHT_FILES = (CLIP_FILE, CLIP_ONNX_FILE, PROBE_FILE, DETECTOR_FILE, PROVIDER_FILE)
GATE_FILE = "receipt-gate-2026-09-02.npz"
GATE_STABLE_FILE = "receipt-gate.npz"
REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SOURCE = REPO_ROOT / "src" / "remove_ai_watermarks" / "assets" / GATE_FILE
CARD_DIR = REPO_ROOT / "docs" / "photo-classify-hf"


def weight_paths(src: Path) -> dict[str, Path]:
    """Map Hub filenames to files under ``src`` or ``src/run1``."""
    src = src.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"missing weights directory: {src}")
    found: dict[str, Path] = {}
    for name in WEIGHT_FILES:
        direct = src / name
        nested = src / "run1" / name
        if direct.is_file():
            found[name] = direct
        elif nested.is_file():
            found[name] = nested
        else:
            raise SystemExit(f"missing {name} in {src} or {src / 'run1'}")
    return found


def stage_card(dest: Path) -> None:
    readme = CARD_DIR / "README.md"
    operating = CARD_DIR / "operating-point.json"
    if not readme.is_file() or not operating.is_file():
        raise SystemExit(f"Hub card missing under {CARD_DIR}")
    if not GATE_SOURCE.is_file():
        raise SystemExit(f"receipt-gate asset missing: {GATE_SOURCE}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(readme, dest / "README.md")
    shutil.copy2(operating, dest / "operating-point.json")
    # The gate head versions with the model: card mode carries it even when
    # the five freeze weight files are not being re-uploaded.
    shutil.copy2(GATE_SOURCE, dest / GATE_FILE)


def _place(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def stage_weights(dest: Path, src: Path) -> None:
    for name, path in weight_paths(src).items():
        _place(path, dest / name)
        log.info("stage %s (%s bytes)", name, path.stat().st_size)
    stable_gate = src / GATE_STABLE_FILE
    if stable_gate.is_file():
        _place(stable_gate, dest / GATE_STABLE_FILE)
        log.info("stage %s (%s bytes)", GATE_STABLE_FILE, stable_gate.stat().st_size)


def publish(stage: Path, *, token: str, message: str) -> str:
    from huggingface_hub import HfApi, create_repo

    create_repo(HUB_REPO, repo_type="model", exist_ok=True, private=False, token=token)
    api = HfApi(token=token)
    commit = api.upload_folder(
        folder_path=str(stage),
        repo_id=HUB_REPO,
        repo_type="model",
        commit_message=message,
        allow_patterns=["README.md", "operating-point.json", *WEIGHT_FILES, GATE_FILE, GATE_STABLE_FILE],
    )
    return getattr(commit, "oid", "") or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="Freeze or weights directory")
    parser.add_argument("--card-only", action="store_true")
    parser.add_argument("--message", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set (write role required)")
    if args.card_only == bool(args.src):
        raise SystemExit("pass exactly one of --src or --card-only")
    message = args.message.strip() or (
        "Update photo-classify card" if args.card_only else "Publish photo-classify freeze"
    )
    with tempfile.TemporaryDirectory(prefix="raiw-hf-") as tmp:
        stage = Path(tmp)
        stage_card(stage)
        if args.src is not None:
            stage_weights(stage, args.src)
        oid = publish(stage, token=token, message=message)
    log.info("published %s revision=%s", HUB_REPO, oid or "(unknown)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
