#!/usr/bin/env python3
"""Publish the photo-classify freeze to Hugging Face wiltodelta/raiw-photo-classify.

Manual. Not a library release step. Weights are not in git: pass --src, a
directory that holds clip-l-ft.pt, clip-l-ft-vision-fp32.onnx,
probe-weights-clip-l-ft.npz, multi-provider-source-v6.npz, and detector.pt
(either at the root or in run1/).

    uv run python scripts/publish_photo_classify_hf.py --src DIR
    uv run python scripts/publish_photo_classify_hf.py --card-only

Needs HF_TOKEN with write role. The GitHub Action
publish-photo-classify-hf.yml injects secrets.HF_TOKEN.

Card mode uploads the README, operating-point.json, and the fallback
receipt-gate head (staged from the package assets) without touching the freeze
weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from hf_publish import publish_model_folder

log = logging.getLogger(__name__)

HUB_REPO = "wiltodelta/raiw-photo-classify"
CLIP_FILE = "clip-l-ft.pt"
CLIP_ONNX_FILE = "clip-l-ft-vision-fp32.onnx"
PROBE_FILE = "probe-weights-clip-l-ft.npz"
DETECTOR_FILE = "detector.pt"
RETIRED_FILES = ("provider.pt",)
GATE_FILE = "receipt-gate-2026-09-02.npz"
GATE_STABLE_FILE = "receipt-gate.npz"
REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SOURCE = REPO_ROOT / "src" / "remove_ai_watermarks" / "assets" / GATE_FILE
CARD_DIR = REPO_ROOT / "docs" / "photo-classify-hf"
_OPERATING_POINT = json.loads((CARD_DIR / "operating-point.json").read_text(encoding="utf-8"))
PROVIDER_FILE: str = _OPERATING_POINT["model2"]["artifact"]
PROVIDER_SHA256: str = _OPERATING_POINT["model2"]["artifact_sha256"]
WEIGHT_FILES = (CLIP_FILE, CLIP_ONNX_FILE, PROBE_FILE, DETECTOR_FILE, PROVIDER_FILE)


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


def _verify_sha256(path: Path, expected: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")


def stage_weights(dest: Path, src: Path) -> None:
    paths = weight_paths(src)
    _verify_sha256(paths[PROVIDER_FILE], PROVIDER_SHA256)
    for name, path in paths.items():
        _place(path, dest / name)
        log.info("stage %s (%s bytes)", name, path.stat().st_size)
    stable_gate = src / GATE_STABLE_FILE
    if stable_gate.is_file():
        _place(stable_gate, dest / GATE_STABLE_FILE)
        log.info("stage %s (%s bytes)", GATE_STABLE_FILE, stable_gate.stat().st_size)


def publish(stage: Path, *, token: str, message: str) -> str:
    return publish_model_folder(
        stage,
        repo_id=HUB_REPO,
        token=token,
        message=message,
        allow_patterns=["README.md", "operating-point.json", *WEIGHT_FILES, GATE_FILE, GATE_STABLE_FILE],
        delete_patterns=RETIRED_FILES,
    )


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
