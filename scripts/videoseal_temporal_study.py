#!/usr/bin/env python3
"""Development-only VideoSeal temporal evaluation study.

Three questions the case-level benchmark verdict aggregates away:

- how the decoded bit accuracy decays across H.264 quality (crf 18-32),
  downscaling, and frame-rate halving;
- whether the frame-aggregation choice (avg, squared_avg, l1norm_avg,
  l2norm_avg) changes the outcome on the same artifact;
- how the per-frame bit accuracy moves in time on real provider clips versus
  synthetic carriers.

Carriers are the two synthetic cohort clips plus committed real provider
videos (their bytes already carry the project's public test clearance). No
provider API is contacted; artifacts stay outside the repository under
``.local-eval/``.

    uv run --extra dev python scripts/videoseal_temporal_study.py \
      --output-dir .local-eval/videoseal-temporal-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import videoseal_oracle  # noqa: E402
from audioseal_experiment import ffmpeg_version  # noqa: E402
from videoseal_oracle import read, read_aggregation_matrix  # noqa: E402
from watermark_benchmark import _decode_video, decode_video_artifact, sha256_file  # noqa: E402
from watermark_benchmark_video_cohort import (  # noqa: E402
    FRAME_RATE,
    encode_clip,
    synth_carrier,
)

log = logging.getLogger("videoseal_temporal_study")

CRF_SWEEP: tuple[int, ...] = (18, 23, 28, 32)
ATTACKS: tuple[str, ...] = ("scale_075", "fps_half")
REAL_CARRIERS: tuple[str, ...] = ("veo", "sora")
STUDY_FRAMES = 64


def require_tools() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("videoseal temporal study requires ffmpeg on PATH")
    return ffmpeg


def real_carrier_frames(name: str) -> tuple[np.ndarray, dict[str, object]]:
    """Decode a committed provider clip to a square 256x256 64-frame prefix."""
    import cv2

    source = REPO / "data" / "fixtures" / "visible" / name / "provider-original.mp4"
    if not source.is_file():
        raise SystemExit(f"provider fixture missing: {source}")
    frames = _decode_video(source)
    if frames is None or frames.shape[0] < STUDY_FRAMES:
        raise SystemExit(f"provider fixture {name} did not decode to {STUDY_FRAMES} frames")
    square = frames[:STUDY_FRAMES]
    height, width = square.shape[1:3]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    cropped = square[:, top : top + side, left : left + side, :]
    resized = np.stack([cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA) for frame in cropped]).astype(
        np.float32
    )
    provenance = {
        "source": f"data/fixtures/visible/{name}/provider-original.mp4",
        "source_sha256": sha256_file(source),
        "source_geometry": [int(height), int(width)],
        "recipe": "first-64-frames, center-crop to square, INTER_AREA resize to 256x256",
    }
    return resized, provenance


def apply_crf(ffmpeg: str, marked: Path, crf: int, workdir: Path) -> Path:
    target = workdir / f"{marked.stem}-crf{crf}.mp4"
    subprocess.run(  # noqa: S603 - resolved ffmpeg with fixed arguments
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(marked),
            "-c:v",
            "libx264",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
    )
    return target


def apply_scale(ffmpeg: str, marked: Path, workdir: Path) -> Path:
    target = workdir / f"{marked.stem}-scale_075.mp4"
    subprocess.run(  # noqa: S603 - resolved ffmpeg with fixed arguments
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(marked),
            "-vf",
            "scale=192:192",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
    )
    return target


def apply_fps_half(ffmpeg: str, marked: Path, workdir: Path) -> Path:
    target = workdir / f"{marked.stem}-fps_half.mp4"
    subprocess.run(  # noqa: S603 - resolved ffmpeg with fixed arguments
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(marked),
            "-vf",
            f"fps={FRAME_RATE / 2:.12g}",
            "-c:v",
            "libx264",
            "-crf",
            "8",
            "-pix_fmt",
            "yuv420p",
            str(target),
        ],
        check=True,
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir: Path = args.output_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory {out_dir}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ffmpeg = require_tools()
    out_dir.mkdir(parents=True)
    model = videoseal_oracle.load_model()
    message = videoseal_oracle.message_bits()

    rows: list[dict[str, object]] = []

    def record(stage: str, **fields: object) -> None:
        row = {"stage": stage, **fields}
        rows.append(row)
        log.info("%s", json.dumps(row, sort_keys=True, default=str)[:400])

    import importlib.metadata as md

    record(
        "pins",
        torch=md.version("torch"),
        videoseal_jit_sha256=videoseal_oracle.JIT_SHA256,
        study_source_sha256=sha256_file(Path(__file__)),
        ffmpeg=ffmpeg_version(ffmpeg),
        message_sha256=hashlib.sha256(bytes(message)).hexdigest(),
    )

    carriers: dict[str, tuple[np.ndarray, dict[str, object]]] = {
        name: (synth_carrier(name)[:STUDY_FRAMES], {"recipe": "synthetic cohort carrier"})
        for name in ("moving_gradient", "moving_texture")
    }
    for name in REAL_CARRIERS:
        carriers[f"real_{name}"] = real_carrier_frames(name)

    for name, (clean_frames, provenance) in carriers.items():
        workdir = out_dir / name
        workdir.mkdir()
        clean_path = encode_clip(ffmpeg, workdir / f"{name}-clean.mp4", clean_frames)
        clean_frames, clean_digest = decode_video_artifact(clean_path)
        record(
            "carrier_clean",
            carrier=name,
            sha256=clean_digest,
            measurement_domain="decoded_artifact",
            aggregation_matrix=read_aggregation_matrix(model, clean_frames),
            **provenance,
        )

        marked_frames = np.asarray(videoseal_oracle.embed(model, clean_frames, message), dtype=np.float32)
        marked_path = encode_clip(ffmpeg, workdir / f"{name}-marked.mp4", marked_frames)
        marked_frames, marked_digest = decode_video_artifact(marked_path)
        marked_read = read(model, marked_frames)
        record(
            "carrier_marked",
            carrier=name,
            sha256=marked_digest,
            measurement_domain="decoded_artifact",
            bit_accuracy_avg=marked_read.bit_accuracy,
            aggregation_matrix=read_aggregation_matrix(model, marked_frames),
            per_frame_bit_accuracy=[round(v, 4) for v in marked_read.per_frame_bit_accuracy],
            wm_snr_db=round(
                float(
                    10
                    * np.log10(
                        np.sum(clean_frames.astype(np.float64) ** 2)
                        / max(np.sum((marked_frames - clean_frames).astype(np.float64) ** 2), 1e-12)
                    )
                ),
                3,
            ),
        )

        attacked: list[tuple[str, Path]] = [
            (f"crf{crf}", apply_crf(ffmpeg, marked_path, crf, workdir)) for crf in CRF_SWEEP
        ]
        attacked += [
            ("scale_075", apply_scale(ffmpeg, marked_path, workdir)),
            ("fps_half", apply_fps_half(ffmpeg, marked_path, workdir)),
        ]
        for attack, path in attacked:
            frames, digest = decode_video_artifact(path)
            reading = read(model, frames)
            record(
                "attack",
                carrier=name,
                attack=attack,
                sha256=digest,
                measurement_domain="decoded_artifact",
                frames=frames.shape[0],
                bit_accuracy_avg=reading.bit_accuracy,
                aggregation_matrix=read_aggregation_matrix(model, frames),
                per_frame_bit_accuracy=[round(v, 4) for v in reading.per_frame_bit_accuracy],
            )

    cases_path = out_dir / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    summary_path = out_dir / "matrix.md"
    summary_path.write_text(render_matrix(rows))
    print(f"wrote {cases_path}")
    print(f"wrote {summary_path}")
    return 0


def cast_to_matrix(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TypeError(f"aggregation matrix must be a dict, got {type(value).__name__}")
    return value  # type: ignore[no-any-return]


def render_matrix(rows: list[dict[str, object]]) -> str:
    aggregations = videoseal_oracle.AGGREGATIONS
    lines = [
        "# VideoSeal temporal evaluation study",
        "",
        "Bit accuracy of the aggregated 256-bit decode per carrier, attack, and",
        "frame aggregation. The benchmark threshold (0.9) is the adapter's",
        "verdict rule; this matrix shows the readings underneath it.",
        "",
        "| Carrier | Attack | " + " | ".join(aggregations) + " |",
        "| --- | --- | " + " | ".join(["---"] * len(aggregations)) + " |",
    ]
    clean_rows = {str(r["carrier"]): r for r in rows if r["stage"] == "carrier_clean"}
    for carrier, row in sorted(clean_rows.items()):
        matrix = cast_to_matrix(row["aggregation_matrix"])
        lines.append(f"| {carrier} | (clean) | " + " | ".join(f"{float(matrix[a]):.3f}" for a in aggregations) + " |")
    for r in rows:
        if r["stage"] != "attack":
            continue
        matrix = cast_to_matrix(r["aggregation_matrix"])
        lines.append(
            f"| {r['carrier']} | {r['attack']} | " + " | ".join(f"{float(matrix[a]):.3f}" for a in aggregations) + " |"
        )
    lines += [
        "",
        "Per-frame bit accuracy ranges (avg aggregation) for marked clips:",
        "",
        "| Carrier | min | mean | max |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        if r["stage"] != "carrier_marked":
            continue
        per_frame = [float(v) for v in r["per_frame_bit_accuracy"]]  # type: ignore[arg-type]
        lines.append(
            f"| {r['carrier']} | {min(per_frame):.3f} | {sum(per_frame) / len(per_frame):.3f} | {max(per_frame):.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
