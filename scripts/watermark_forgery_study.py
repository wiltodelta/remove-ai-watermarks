#!/usr/bin/env python3
"""Development-only watermark forgery study (four states plus double embedding).

The benchmark's removed and forged rows assert outcomes; this study measures
the mechanism underneath them, for both pinned oracles:

- ``clean`` baseline; ``marked`` the oracle's message A;
- ``removed`` the transform that measurably destroys A;
- ``forged on clean`` message B embedded into unmarked media (false
  attribution);
- ``forged on marked`` message B embedded OVER an A-marked artifact - the
  double-embedding question: does the decoder see B (forgery wins), A (the
  original survives), or neither (mutual interference)?
- ``forged on removed`` B embedded after the removal attempt.

Everything runs locally against the pinned oracles. Artifacts stay outside
the repository under ``.local-eval/``.

    uv run --extra dev python scripts/watermark_forgery_study.py \
      --output-dir .local-eval/watermark-forgery-v1
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import audioseal_oracle  # noqa: E402
import videoseal_oracle  # noqa: E402
from audioseal_experiment import numpy_attacks  # noqa: E402
from audioseal_experiment import synth_carrier as synth_audio_carrier  # noqa: E402
from videoseal_temporal_study import apply_crf, real_carrier_frames  # noqa: E402
from watermark_benchmark import sha256_file  # noqa: E402
from watermark_benchmark_video_cohort import synth_carrier as synth_video_carrier  # noqa: E402

log = logging.getLogger("watermark_forgery_study")

FORGED_SEED = 8
CLIP_FRAMES = 64


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("forgery study requires ffmpeg on PATH")
    return ffmpeg


def bit_accuracy(decoded: Sequence[int], message: Sequence[int]) -> float:
    return sum(int(a == b) for a, b in zip(decoded, message, strict=True)) / len(message)


def audio_cells(out_dir: Path) -> list[dict[str, object]]:
    import importlib.metadata as md

    from audioseal_experiment import SAMPLE_RATE, message_bits, wav_pcm16_bytes

    generator, detector = audioseal_oracle.load_pinned_models()
    message_a = message_bits()
    message_b = message_bits(FORGED_SEED)
    rows: list[dict[str, object]] = []

    def measure(label: str, samples: np.ndarray, path: Path | None = None) -> dict[str, object]:
        reading = audioseal_oracle.read(detector, samples)
        row: dict[str, object] = {
            "scheme": "audioseal",
            "cell": label,
            "presence_frac": round(reading.frac_above_threshold, 6),
            "bit_accuracy_vs_a": round(bit_accuracy(reading.decoded_bits, message_a), 6),
            "bit_accuracy_vs_b": round(bit_accuracy(reading.decoded_bits, message_b), 6),
        }
        if path is not None:
            row["sha256"] = sha256_file(path)
        rows.append(row)
        log.info("%s", json.dumps(row, sort_keys=True))
        return row

    carrier_name = "tone_stack"
    clean = synth_audio_carrier(carrier_name, 8.0)
    workdir = out_dir / "audioseal"
    workdir.mkdir()
    clean_path = workdir / "clean.wav"
    clean_path.write_bytes(wav_pcm16_bytes(clean, SAMPLE_RATE))
    measure("clean", clean, clean_path)

    marked_a = np.asarray(audioseal_oracle.embed(generator, clean, message_a), dtype=np.float32)
    marked_a_path = workdir / "marked_a.wav"
    marked_a_path.write_bytes(wav_pcm16_bytes(marked_a, SAMPLE_RATE))
    measure("marked_a", marked_a, marked_a_path)

    removed = numpy_attacks(marked_a)["noise_snr10"]
    removed_path = workdir / "removed.wav"
    removed_path.write_bytes(wav_pcm16_bytes(removed, SAMPLE_RATE))
    measure("removed_noise", removed, removed_path)

    forged_on_clean = np.asarray(audioseal_oracle.embed(generator, clean, message_b), dtype=np.float32)
    path = workdir / "forged_on_clean.wav"
    path.write_bytes(wav_pcm16_bytes(forged_on_clean, SAMPLE_RATE))
    measure("forged_b_on_clean", forged_on_clean, path)

    forged_on_marked = np.asarray(audioseal_oracle.embed(generator, marked_a, message_b), dtype=np.float32)
    path = workdir / "forged_b_on_marked.wav"
    path.write_bytes(wav_pcm16_bytes(forged_on_marked, SAMPLE_RATE))
    measure("forged_b_on_marked_a", forged_on_marked, path)

    forged_on_removed = np.asarray(audioseal_oracle.embed(generator, removed, message_b), dtype=np.float32)
    path = workdir / "forged_on_removed.wav"
    path.write_bytes(wav_pcm16_bytes(forged_on_removed, SAMPLE_RATE))
    measure("forged_b_on_removed_a", forged_on_removed, path)

    rows.append(
        {
            "scheme": "audioseal",
            "cell": "pins",
            "audioseal": md.version("audioseal"),
            "huggingface_revision": audioseal_oracle.HF_REVISION,
        }
    )
    return rows


def video_cells(out_dir: Path, ffmpeg: str) -> list[dict[str, object]]:
    from watermark_benchmark import decode_video_artifact
    from watermark_benchmark_video_cohort import encode_clip

    model = videoseal_oracle.load_model()
    message_a = videoseal_oracle.message_bits()
    message_b = videoseal_oracle.message_bits(FORGED_SEED)
    rows: list[dict[str, object]] = []
    workdir = out_dir / "videoseal"
    workdir.mkdir()

    def measure(label: str, path: Path) -> np.ndarray:
        frames, digest = decode_video_artifact(path)
        reading = videoseal_oracle.read(model, frames, message=message_a)
        row: dict[str, object] = {
            "scheme": "videoseal",
            "cell": label,
            "bit_accuracy_vs_a": round(reading.bit_accuracy, 6),
            "bit_accuracy_vs_b": round(bit_accuracy(reading.decoded_bits, message_b), 6),
            "detected_by_matched_rule": reading.detected,
            "sha256": digest,
            "measurement_domain": "decoded_artifact",
        }
        rows.append(row)
        log.info("%s", json.dumps(row, sort_keys=True))
        return frames

    for carrier_name, clean in (
        ("moving_gradient", synth_video_carrier("moving_gradient")[:CLIP_FRAMES]),
        *((f"real_{name}", real_carrier_frames(name)[0]) for name in ("sora",)),
    ):
        prefix = f"{carrier_name}-"
        clean_path = encode_clip(ffmpeg, workdir / f"{prefix}clean.mp4", clean)
        clean = measure(f"{carrier_name}/clean", clean_path)

        marked_a = np.asarray(videoseal_oracle.embed(model, clean, message_a), dtype=np.float32)
        marked_a_path = encode_clip(ffmpeg, workdir / f"{prefix}marked_a.mp4", marked_a)
        marked_a = measure(f"{carrier_name}/marked_a", marked_a_path)

        removed_path = apply_crf(ffmpeg, marked_a_path, 23, workdir)
        removed_frames = measure(f"{carrier_name}/removed_crf23", removed_path)

        forged_on_clean = np.asarray(videoseal_oracle.embed(model, clean, message_b), dtype=np.float32)
        path = encode_clip(ffmpeg, workdir / f"{prefix}forged_b_on_clean.mp4", forged_on_clean)
        measure(f"{carrier_name}/forged_b_on_clean", path)

        forged_on_marked = np.asarray(videoseal_oracle.embed(model, marked_a, message_b), dtype=np.float32)
        path = encode_clip(ffmpeg, workdir / f"{prefix}forged_b_on_marked.mp4", forged_on_marked)
        measure(f"{carrier_name}/forged_b_on_marked_a", path)

        if removed_frames.shape[0] == marked_a.shape[0]:
            forged_on_removed = np.asarray(videoseal_oracle.embed(model, removed_frames, message_b), dtype=np.float32)
            path = encode_clip(ffmpeg, workdir / f"{prefix}forged_b_on_removed.mp4", forged_on_removed)
            measure(f"{carrier_name}/forged_b_on_removed_a", path)

    rows.append(
        {
            "scheme": "videoseal",
            "cell": "pins",
            "videoseal_jit_sha256": videoseal_oracle.JIT_SHA256,
        }
    )
    return rows


def remover_trace_probe(out_dir: Path) -> dict[str, object]:
    """Scoped remover-trace probe: does our noise 'removal' leave a trace?

    Measures the high-to-low frequency band energy ratio of clean versus
    noise-removed audio. This detects the trace of OUR gaussian-noise removal,
    not a general remover detector; it is first evidence, not a product.
    """
    from audioseal_experiment import SAMPLE_RATE, message_bits, numpy_attacks, synth_carrier

    generator, _detector = audioseal_oracle.load_pinned_models()
    message_a = message_bits()
    stats: dict[str, list[float]] = {"clean": [], "removed": []}
    for name in ("tone_stack", "pinkish_noise", "white_noise"):
        clean = synth_carrier(name, 8.0)
        marked = np.asarray(audioseal_oracle.embed(generator, clean, message_a), dtype=np.float32)
        removed = numpy_attacks(marked)["noise_snr10"]
        for label, samples in (("clean", clean), ("removed", removed)):
            spectrum = np.abs(np.fft.rfft(samples.astype(np.float64))) ** 2
            freqs = np.fft.rfftfreq(samples.shape[0], 1.0 / SAMPLE_RATE)
            high = spectrum[freqs > 6000].sum()
            low = spectrum[freqs <= 6000].sum()
            stats[label].append(float(high / max(low, 1e-12)))
    probe = {
        "scope": "traces of this study's gaussian-noise removal only",
        "hf_lf_ratio_clean": [round(v, 6) for v in stats["clean"]],
        "hf_lf_ratio_removed": [round(v, 6) for v in stats["removed"]],
    }
    (out_dir / "remover_trace_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    return probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir: Path = args.output_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory {out_dir}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ffmpeg = require_ffmpeg()
    out_dir.mkdir(parents=True)

    rows = audio_cells(out_dir)
    rows += video_cells(out_dir, ffmpeg)
    probe = remover_trace_probe(out_dir)
    rows.append({"scheme": "probe", "cell": "remover_trace", **probe})

    cases_path = out_dir / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    print(f"wrote {cases_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
