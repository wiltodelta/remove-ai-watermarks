#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#     "remove-ai-watermarks[source-classify]>=0.64.0",
# ]
# ///
"""Standalone reference runner and pickle-free `.npz` weight inspector for `wiltodelta/raiw-source-classify`.

Run directly without setting up a virtual environment via `uv`:

    uv run infer.py image.jpg
    uv run infer.py ./images --json
    uv run infer.py --inspect-weights

This model identifies the likely AI generation/export pixel pipeline across 10
provider families (`openai`, `google`, `bytedance`, `flux`, `stable_diffusion`,
`meta`, `midjourney`, `ideogram`, `firefly`, `xai`) or abstains with `unknown`.
It is not a provenance verifier and does not detect or verify SynthID or any
invisible watermark (`watermark_truth="unknown"`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if (_REPO_SRC / "remove_ai_watermarks").is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

log = logging.getLogger(__name__)

SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".heic", ".heif", ".avif"}
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _resolve_artifact_path(weights_path: Path | None = None) -> Path:
    from remove_ai_watermarks import source_classify as sc

    resolver = getattr(sc, "resolve_weights_path", None)
    if callable(resolver):
        return Path(resolver(weights_path))
    if weights_path is not None:
        candidate = Path(weights_path).expanduser()
        return candidate / sc.MODEL_FILE if candidate.is_dir() else candidate
    override = os.environ.get(sc.WEIGHTS_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        return candidate / sc.MODEL_FILE if candidate.is_dir() else candidate
    from huggingface_hub import hf_hub_download

    downloaded = Path(
        hf_hub_download(
            repo_id=sc.WEIGHTS_REPO,
            filename=sc.MODEL_FILE,
            revision=sc.WEIGHTS_REVISION,
        )
    )
    if _sha256_file(downloaded).lower() != sc.MODEL_SHA256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {downloaded.name}")
    return downloaded


def inspect_weights(weights_path: Path | None = None) -> dict[str, Any]:
    """Inspect a pickle-free `.npz` source-classifier artifact (`allow_pickle=False`)."""
    import numpy as np

    resolved = _resolve_artifact_path(weights_path)
    size_bytes = resolved.stat().st_size
    sha256 = _sha256_file(resolved)

    with np.load(resolved, allow_pickle=False) as data:
        schema = str(np.asarray(data["schema"]).item())
        feature_dim = int(np.asarray(data["feature_dim"], dtype=np.int64).squeeze())
        cls_key = "provider_classes" if "provider_classes" in data else "all_ai_classes"
        providers = [str(item) for item in np.asarray(data[cls_key]).tolist()]

        thresholds: dict[str, float] = {}
        global_ai_threshold: float | None = None
        if "det_thresholds" in data:
            det_thr = np.asarray(data["det_thresholds"], dtype=np.float64).reshape(-1)
            for idx, prov in enumerate(providers):
                if idx < det_thr.shape[0]:
                    thresholds[prov] = round(float(det_thr[idx]), 6)
            if det_thr.shape[0] > len(providers):
                global_ai_threshold = round(float(det_thr[len(providers)]), 6)
        elif "router_thresholds" in data:
            r_thr = np.asarray(data["router_thresholds"], dtype=np.float64).reshape(-1)
            for prov, val in zip(providers, r_thr.tolist(), strict=False):
                thresholds[prov] = round(float(val), 6)

        version_subfamilies: dict[str, list[str]] = {}
        if "stage2_version_classes" in data:
            for p_cls in (str(x) for x in np.asarray(data["stage2_version_classes"]).tolist()):
                key = f"s2_{p_cls}_subfamilies"
                if key in data:
                    version_subfamilies[p_cls] = [str(x) for x in np.asarray(data[key]).tolist()]

        if "sub_names" in data and "sub_providers" in data:
            sub_names = [str(x) for x in np.asarray(data["sub_names"]).tolist()]
            sub_provs = [str(x) for x in np.asarray(data["sub_providers"]).tolist()]
            for prov in providers:
                if prov not in version_subfamilies:
                    subs = list(
                        dict.fromkeys(
                            s_name for s_name, s_prov in zip(sub_names, sub_provs, strict=False) if s_prov == prov
                        )
                    )
                    if subs:
                        version_subfamilies[prov] = subs

        array_count = len(data.files)
        synthetic_threshold: float | None = None
        vit_onnx_size_bytes: int | None = None
        if "syn_threshold" in data:
            synthetic_threshold = round(float(np.asarray(data["syn_threshold"], dtype=np.float64).squeeze()), 6)
        if "vit_onnx_bytes" in data:
            import numpy.lib.format as npfmt

            zf = getattr(data, "zip", None)
            if zf is not None and "vit_onnx_bytes.npy" in zf.namelist():
                with zf.open("vit_onnx_bytes.npy") as handle:
                    ver = npfmt.read_magic(handle)
                    hdr_fn = npfmt.read_array_header_1_0 if ver == (1, 0) else npfmt.read_array_header_2_0
                    shape, _fortran, _dtype = hdr_fn(handle)
                vit_onnx_size_bytes = int(shape[0]) if len(shape) == 1 else int(np.prod(shape))
            else:
                vit_onnx_size_bytes = int(np.asarray(data["vit_onnx_bytes"], dtype=np.uint8).size)

    summary: dict[str, Any] = {
        "artifact_file": resolved.name,
        "resolved_path": str(resolved),
        "size_bytes": size_bytes,
        "sha256": sha256,
        "pickle_free": True,
        "schema": schema,
        "feature_dim": feature_dim,
        "array_count": array_count,
        "providers": providers,
        "abstain_label": "unknown",
        "watermark_truth": "unknown",
        "thresholds": thresholds,
        "version_subfamilies": version_subfamilies,
    }
    if global_ai_threshold is not None:
        summary["global_ai_threshold"] = global_ai_threshold
    if synthetic_threshold is not None:
        summary["synthetic_threshold"] = synthetic_threshold
    if vit_onnx_size_bytes is not None:
        summary["vit_onnx_size_bytes"] = vit_onnx_size_bytes
    return summary


def _expand_targets(targets: Sequence[Path]) -> list[Path]:
    expanded: list[Path] = []
    for target in targets:
        if target.is_dir():
            expanded.extend(
                sorted(
                    path
                    for path in target.iterdir()
                    if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
                )
            )
        else:
            expanded.append(target)
    return expanded


def classify_targets(
    targets: Sequence[Path],
    *,
    weights_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run `classify_source` over image files or directories and return JSON-ready rows."""
    import remove_ai_watermarks as raiw

    rows: list[dict[str, Any]] = []
    for image_path in _expand_targets(targets):
        res = raiw.classify_source(image_path, weights_path=weights_path)
        rows.append({"path": str(image_path), **res.to_dict()})
    return rows


def _write_human_inspection(summary: dict[str, Any]) -> None:
    lines = [
        f"Artifact: {summary['artifact_file']} ({summary['size_bytes']:,} bytes, allow_pickle=False)",
        f"SHA-256:  {summary['sha256']}",
        f"Schema:   {summary['schema']} (feature_dim={summary['feature_dim']}, arrays={summary['array_count']})",
        f"Classes:  {', '.join(summary['providers'])} + {summary['abstain_label']}",
        "Scope:    pixel source/export pipeline attribution only (watermark_truth='unknown')",
        "Stage-2 version subfamilies:",
    ]
    for prov in summary["providers"]:
        subs = summary["version_subfamilies"].get(prov, [])
        if subs:
            lines.append(f"  - {prov}: {', '.join(subs)}")
    sys.stdout.write("\n".join(lines) + "\n")


def _write_human_predictions(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        sub = row.get("version_subfamily") or "-"
        conf = float(row.get("confidence", 0.0))
        syn_flag = "yes" if row.get("synthetic_detected") else "no"
        syn_prob = float(row.get("synthetic_probability", 0.0))
        sys.stdout.write(
            f"{row['path']}\tlabel={row['label']}\tversion={sub}\t"
            f"synthetic={syn_flag}({syn_prob:.4f})\t"
            f"confidence={conf:.4f}\treason={row['reason']}\twatermark_truth={row['watermark_truth']}\n"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for standalone weight inspection and batch/single inference."""
    parser = argparse.ArgumentParser(
        description="Run or inspect the CPU-only RAIW 10-provider source-pipeline classifier."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        help="Image file(s) or directory(ies) to classify.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional local .npz file or directory override (defaults to Hub or RAIW_SOURCE_CLASSIFY_WEIGHTS).",
    )
    parser.add_argument(
        "--inspect-weights",
        action="store_true",
        help="Inspect the pickle-free .npz artifact schema, thresholds, and Stage-2 version subfamilies.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.inspect_weights and not args.targets:
        parser.print_usage(sys.stderr)
        log.error("Provide one or more image/directory targets, or pass --inspect-weights.")
        return 2

    missing = [str(p) for p in args.targets if not p.exists()]
    if missing:
        log.error("Target path does not exist: %s", ", ".join(missing))
        return 2

    if args.inspect_weights and not args.targets:
        summary = inspect_weights(args.weights)
        if args.json:
            sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        else:
            _write_human_inspection(summary)
        return 0

    rows = classify_targets(args.targets, weights_path=args.weights)
    if args.inspect_weights:
        payload = {"weights": inspect_weights(args.weights), "predictions": rows}
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        return 0

    if args.json:
        out: Any = rows[0] if len(rows) == 1 and args.targets[0].is_file() else rows
        sys.stdout.write(json.dumps(out, indent=2) + "\n")
    else:
        _write_human_predictions(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
