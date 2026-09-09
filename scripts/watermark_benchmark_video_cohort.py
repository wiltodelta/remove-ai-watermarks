#!/usr/bin/env python3
"""Build a local, synthetic video-watermark benchmark cohort.

The builder renders deterministic synthetic clips, embeds the pinned VideoSeal
message through the TorchScript oracle, applies fixed ffmpeg attacks, and
writes every generated artifact once under a new output directory. It emits
the strict JSONL manifest consumed by ``watermark_benchmark.py`` with
``media_type: video`` rows for the ``videoseal`` adapter.

Generated artifacts and reports belong under ``.local-eval/`` and are never
committed. No provider oracle or API is used. The first build downloads the
pinned TorchScript checkpoint into the user cache when it is incomplete.

    uv run --extra dev python scripts/watermark_benchmark_video_cohort.py \
      --output-dir .local-eval/watermark-benchmark-video-cohort-v1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

import videoseal_oracle  # noqa: E402
from audioseal_experiment import ffmpeg_version  # noqa: E402
from watermark_benchmark import SCHEMA_VERSION, load_manifest, sha256_file, write_jsonl  # noqa: E402

log = logging.getLogger("watermark_benchmark_video_cohort")

RECIPE_VERSION = "watermark-benchmark-video-cohort-v2"
FORGED_MESSAGE_SEED = 8
CARRIER_NAMES: tuple[str, ...] = ("moving_gradient", "moving_texture")
HARD_NEGATIVE_CARRIER = "moving_texture_hard_negative"
ATTACKS: tuple[str, ...] = ("h264_crf23", "scale_075", "fps_half")
FRAME_COUNT = 64
FRAME_RATE = 12.0
WIDTH = 256
HEIGHT = 256
CARRIER_SEED = 20260908
# Attacks that keep the artifact geometry comparable with the reference clip.
REFERENCE_BACKED_ATTACKS: tuple[str, ...] = ("h264_crf23",)


def require_tools() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("video cohort builder requires ffmpeg on PATH")
    return ffmpeg


def oracle_dependencies() -> dict[str, str]:
    """Return versions of everything that can affect cohort artifacts."""

    def package(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "numpy": package("numpy"),
        "torch": package("torch"),
        "videoseal-jit-sha256": videoseal_oracle.JIT_SHA256,
        "ffmpeg": ffmpeg_version(require_tools()),
    }


def _source_revision() -> str:
    return f"{RECIPE_VERSION}@sha256:{sha256_file(Path(__file__))}"


def _carrier_seed(name: str) -> int:
    import zlib

    return CARRIER_SEED + zlib.crc32(name.encode()) % 1000


def synth_carrier(name: str, *, seed_offset: int = 0, seed: int | None = None) -> np.ndarray:
    """Render one deterministic clip as float RGB frames in [0, 1]."""
    rng = np.random.default_rng(_carrier_seed(name) + seed_offset if seed is None else seed)
    frames = np.empty((FRAME_COUNT, HEIGHT, WIDTH, 3), dtype=np.float32)
    axis = np.arange(WIDTH, dtype=np.float32)[None, :]
    rows = np.arange(HEIGHT, dtype=np.float32)[:, None]
    for index in range(FRAME_COUNT):
        if name == "moving_gradient":
            luma = 0.35 + 0.25 * np.sin((axis + index * 4) / 23.0) + 0.15 * np.cos((rows - index * 2) / 31.0)
            frame = np.stack((luma, np.clip(luma + 0.08, 0, 1), np.clip(luma - 0.06, 0, 1)), axis=2)
        elif name == "moving_texture":
            noise = rng.normal(0.0, 1.0, (HEIGHT // 4, WIDTH // 4)).astype(np.float32)
            texture = np.asarray(np.kron(noise, np.ones((4, 4), dtype=np.float32)) * 0.12 + 0.5, dtype=np.float32)
            frame = np.stack((texture, texture, texture), axis=2)
            block_x = (index * 5) % (WIDTH - 64)
            frame[:, block_x : block_x + 64, 1] = np.clip(frame[:, block_x : block_x + 64, 1] + 0.25, 0, 1)
        else:
            raise ValueError(f"unknown carrier {name!r}")
        frames[index] = np.clip(frame, 0.0, 1.0)
    return frames


def encode_clip(ffmpeg: str, path: Path, frames: np.ndarray, *, crf: int = 8) -> Path:
    """Encode float frames in [0, 1] to an MP4 clip."""
    import subprocess

    height, width = frames.shape[1:3]
    payload = (frames * 255.0).astype(np.uint8).tobytes()
    subprocess.run(  # noqa: S603 - resolved ffmpeg with fixed arguments
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{FRAME_RATE:.12g}",
            "-i",
            "pipe:0",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        input=payload,
        check=True,
    )
    return path


ATTACK_ARGS: dict[str, list[str]] = {
    "h264_crf23": ["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"],
    "scale_075": ["-vf", "scale=192:192", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"],
    "fps_half": ["-vf", f"fps={FRAME_RATE / 2:.12g}", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p"],
}


def attack_parameters(attack: str, dependencies: dict[str, str]) -> dict[str, object]:
    values: dict[str, object] = dict.fromkeys((), "")
    if attack == "h264_crf23":
        values = {"codec": "libx264", "crf": 23}
    elif attack == "scale_075":
        values = {"scale": [192, 192], "codec": "libx264", "crf": 18}
    elif attack == "fps_half":
        values = {"fps": FRAME_RATE / 2, "codec": "libx264", "crf": 8}
    else:
        raise ValueError(f"unknown attack {attack!r}")
    return {**values, "dependencies": dependencies}


def apply_attack(ffmpeg: str, attack: str, marked: Path, artifacts: Path) -> Path:
    import subprocess

    target = artifacts / f"{marked.stem}-{attack}.mp4"
    subprocess.run(  # noqa: S603 - resolved ffmpeg with fixed arguments
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(marked), *ATTACK_ARGS[attack], str(target)],
        check=True,
    )
    return target


def benchmark_row(
    *,
    root: Path,
    case_id: str,
    pair_id: str,
    arm: str,
    state: str,
    path: Path,
    reference: Path | None,
    transform_name: str,
    transform_revision: str,
    parameters: dict[str, object],
    seed: int | None,
    expected: str,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "pair_id": pair_id,
        "media_type": "video",
        "adapter": "videoseal",
        "arm": arm,
        "state": state,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "reference_path": reference.relative_to(root).as_posix() if reference is not None else None,
        "reference_sha256": sha256_file(reference) if reference is not None else None,
        "source_revision": _source_revision(),
        "transform": {"name": transform_name, "revision": transform_revision, "parameters": parameters},
        "seed": seed,
        "expected": expected,
    }


def build_cohort(
    output_dir: Path,
    *,
    carriers: Sequence[str] = CARRIER_NAMES,
    attacks: Sequence[str] = ATTACKS,
) -> Path:
    """Build and validate a new video cohort directory, returning its manifest."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not videoseal_oracle.available():
        raise SystemExit("video cohort builder requires the dev extra: uv sync --extra dev (torch)")
    unknown = sorted(set(carriers) - set(CARRIER_NAMES))
    if unknown:
        raise ValueError(f"unknown carriers: {', '.join(unknown)}")
    unknown_attacks = sorted(set(attacks) - set(ATTACKS))
    if unknown_attacks:
        raise ValueError(f"unknown attacks: {', '.join(unknown_attacks)}")

    ffmpeg = require_tools()
    model = videoseal_oracle.load_model()
    message = videoseal_oracle.message_bits()
    output_dir.mkdir(parents=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    dependencies = oracle_dependencies()
    rows: list[dict[str, object]] = []

    for name in carriers:
        executed_seed = _carrier_seed(name)
        clean = synth_carrier(name, seed=executed_seed)
        clean_path = encode_clip(ffmpeg, artifacts / f"{name}-clean.mp4", clean)
        marked_frames = np.asarray(videoseal_oracle.embed(model, clean, message), dtype=np.float32)
        marked_path = encode_clip(ffmpeg, artifacts / f"{name}-marked.mp4", marked_frames)
        geometry = {"frames": FRAME_COUNT, "width": WIDTH, "height": HEIGHT, "fps": FRAME_RATE}

        rows.append(
            benchmark_row(
                root=output_dir,
                case_id=f"{name}-clean",
                pair_id=name,
                arm="matched_negative",
                state="clean",
                path=clean_path,
                reference=None,
                transform_name="synthesize-clip",
                transform_revision=RECIPE_VERSION,
                parameters={"carrier": name, **geometry, "dependencies": dependencies},
                seed=executed_seed,
                expected="not_detected",
            )
        )
        message_digest = hashlib.sha256(bytes(message)).hexdigest()
        rows.append(
            benchmark_row(
                root=output_dir,
                case_id=f"{name}-marked",
                pair_id=name,
                arm="positive",
                state="marked",
                path=marked_path,
                reference=clean_path,
                transform_name="videoseal-embed",
                transform_revision=f"torch@{dependencies['torch']}+jit@{videoseal_oracle.JIT_SHA256}",
                parameters={
                    "message_sha256": message_digest,
                    **geometry,
                    "dependencies": dependencies,
                },
                seed=videoseal_oracle.MESSAGE_SEED,
                expected="detected",
            )
        )
        for attack in attacks:
            attacked = apply_attack(ffmpeg, attack, marked_path, artifacts)
            rows.append(
                benchmark_row(
                    root=output_dir,
                    case_id=f"{name}-{attack}",
                    pair_id=name,
                    arm="positive",
                    state="attacked",
                    path=attacked,
                    reference=clean_path if attack in REFERENCE_BACKED_ATTACKS else None,
                    transform_name=f"attack-{attack}",
                    transform_revision=RECIPE_VERSION,
                    parameters=attack_parameters(attack, dependencies),
                    seed=None,
                    expected="unresolved",
                )
            )

        # The removed row reuses the crf 23 re-encode that measurably takes
        # both carriers below the decision rule.
        removed_path = artifacts / f"{name}-marked-h264_crf23.mp4"
        if removed_path.is_file():
            rows.append(
                benchmark_row(
                    root=output_dir,
                    case_id=f"{name}-removed",
                    pair_id=name,
                    arm="positive",
                    state="removed",
                    path=removed_path,
                    reference=clean_path,
                    transform_name="remove-h264-crf23",
                    transform_revision=RECIPE_VERSION,
                    parameters=attack_parameters("h264_crf23", dependencies),
                    seed=None,
                    expected="not_detected",
                )
            )

        # The forged row embeds a different message into the same clean clip.
        forged_message = videoseal_oracle.message_bits(FORGED_MESSAGE_SEED)
        forged_frames = np.asarray(videoseal_oracle.embed(model, clean, forged_message), dtype=np.float32)
        forged_path = encode_clip(ffmpeg, artifacts / f"{name}-forged.mp4", forged_frames)
        rows.append(
            benchmark_row(
                root=output_dir,
                case_id=f"{name}-forged",
                pair_id=name,
                arm="wrong_key",
                state="forged",
                path=forged_path,
                reference=clean_path,
                transform_name="videoseal-embed-foreign-message",
                transform_revision=f"torch@{dependencies['torch']}+jit@{videoseal_oracle.JIT_SHA256}",
                parameters={
                    "message_sha256": hashlib.sha256(bytes(forged_message)).hexdigest(),
                    "message_seed": FORGED_MESSAGE_SEED,
                    **geometry,
                    "dependencies": dependencies,
                },
                seed=FORGED_MESSAGE_SEED,
                expected="not_detected",
            )
        )

    hard_seed = _carrier_seed("moving_texture") + 991
    hard = synth_carrier("moving_texture", seed=hard_seed)
    hard_path = encode_clip(ffmpeg, artifacts / f"{HARD_NEGATIVE_CARRIER}.mp4", hard)
    rows.append(
        benchmark_row(
            root=output_dir,
            case_id=f"{HARD_NEGATIVE_CARRIER}-clean",
            pair_id=HARD_NEGATIVE_CARRIER,
            arm="hard_negative",
            state="clean",
            path=hard_path,
            reference=None,
            transform_name="synthesize-clip",
            transform_revision=RECIPE_VERSION,
            parameters={"carrier": "moving_texture", "seed_offset": 991, "dependencies": dependencies},
            seed=hard_seed,
            expected="not_detected",
        )
    )

    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    load_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        manifest = build_cohort(args.output_dir.resolve())
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    log.info("Wrote validated video benchmark cohort to %s", manifest)


if __name__ == "__main__":
    main()
