#!/usr/bin/env python3
"""Build a local, synthetic audio-watermark benchmark cohort.

The builder creates deterministic synthetic carriers, embeds the pinned
AudioSeal message through the shared local oracle, applies fixed audio attacks,
and writes every generated artifact once under a new output directory. It emits
the strict JSONL manifest consumed by ``watermark_benchmark.py`` with
``media_type: audio`` rows for the ``audioseal`` adapter.

Generated artifacts and reports belong under ``.local-eval/`` and are never
committed. No provider oracle or API is used; the first build may download the
revision-pinned AudioSeal checkpoints when the local cache is incomplete.

    uv run --extra dev python scripts/watermark_benchmark_audio_cohort.py \
      --output-dir .local-eval/watermark-benchmark-audio-cohort-v1
"""

from __future__ import annotations

import argparse
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

import audioseal_oracle  # noqa: E402
from audioseal_experiment import (  # noqa: E402
    FFMPEG_ATTACKS,
    MESSAGE_SEED,
    NOISE_ATTACK_SEED,
    SAMPLE_RATE,
    carrier_seed,
    decode_audio_f32,
    ffmpeg_attack,
    ffmpeg_version,
    message_bits,
    numpy_attacks,
    synth_carrier,
    wav_pcm16_bytes,
)
from watermark_benchmark import SCHEMA_VERSION, load_manifest, sha256_file, write_jsonl  # noqa: E402

log = logging.getLogger("watermark_benchmark_audio_cohort")

RECIPE_VERSION = "watermark-benchmark-audio-cohort-v2"
FORGED_MESSAGE_SEED = 8
HARD_NEGATIVE_SEED = 20260907 + 991
# Carriers the 10 dB additive-noise transform measurably takes below the
# audioseal decision rule; the synthetic noise carriers survive it.
REMOVED_BY_NOISE: tuple[str, ...] = ("tone_stack",)
CARRIER_NAMES: tuple[str, ...] = ("tone_stack", "pinkish_noise", "white_noise")
HARD_NEGATIVE_CARRIER = "white_noise_hard_negative"
ATTACKS: tuple[str, ...] = ("mp3_128k", "aac_128k", "noise_snr10")
DEFAULT_DURATION_S = 8.0


def require_tools() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("audio cohort builder requires ffmpeg on PATH")
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
        "audioseal": package("audioseal"),
        "torch": package("torch"),
        "huggingface-revision": audioseal_oracle.HF_REVISION,
        "ffmpeg": ffmpeg_version(require_tools()),
    }


def _source_revision() -> str:
    return f"{RECIPE_VERSION}@sha256:{sha256_file(Path(__file__))}"


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
    """Build one strict benchmark-manifest row for a generated artifact."""
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "pair_id": pair_id,
        "media_type": "audio",
        "adapter": "audioseal",
        "arm": arm,
        "state": state,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "reference_path": reference.relative_to(root).as_posix() if reference is not None else None,
        "reference_sha256": sha256_file(reference) if reference is not None else None,
        "source_revision": _source_revision(),
        "transform": {
            "name": transform_name,
            "revision": transform_revision,
            "parameters": parameters,
        },
        "seed": seed,
        "expected": expected,
    }


def apply_attack(ffmpeg: str, attack: str, marked: Path, artifacts: Path) -> Path:
    """Apply one named attack to the marked WAV and return its artifact path."""
    if attack in FFMPEG_ATTACKS:
        suffix, ffargs = FFMPEG_ATTACKS[attack]
        return ffmpeg_attack(ffmpeg, attack, suffix, ffargs, marked, artifacts, stem=marked.stem)
    if attack == "noise_snr10":
        samples = numpy_attacks(decode_audio_f32(ffmpeg, marked))[attack]
        target = artifacts / f"{marked.stem}-{attack}.wav"
        target.write_bytes(wav_pcm16_bytes(samples, SAMPLE_RATE))
        return target
    raise ValueError(f"unknown attack {attack!r}")


def build_cohort(
    output_dir: Path,
    *,
    carriers: Sequence[str] = CARRIER_NAMES,
    attacks: Sequence[str] = ATTACKS,
    duration_s: float = DEFAULT_DURATION_S,
    include_speech: bool = False,
) -> Path:
    """Build and validate a new audio cohort directory, returning its manifest."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not audioseal_oracle.available():
        raise SystemExit(
            "audio cohort builder requires the dev extra: uv sync --extra dev (audioseal, torch, huggingface-hub)"
        )
    unknown = sorted(set(carriers) - set(CARRIER_NAMES))
    if unknown:
        raise ValueError(f"unknown carriers: {', '.join(unknown)}")
    unknown_attacks = sorted(set(attacks) - set(ATTACKS))
    if unknown_attacks:
        raise ValueError(f"unknown attacks: {', '.join(unknown_attacks)}")

    from audioseal_oracle import embed, load_pinned_models

    ffmpeg = require_tools()
    output_dir.mkdir(parents=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    dependencies = oracle_dependencies()
    message = message_bits()
    generator, _detector = load_pinned_models()
    rows: list[dict[str, object]] = []

    for name in carriers:
        executed_seed = carrier_seed(name)
        clean = synth_carrier(name, duration_s, seed=executed_seed)
        clean_path = artifacts / f"{name}-clean.wav"
        clean_path.write_bytes(wav_pcm16_bytes(clean, SAMPLE_RATE))

        marked = np.asarray(embed(generator, clean, message), dtype=np.float32)
        marked_path = artifacts / f"{name}-marked.wav"
        marked_path.write_bytes(wav_pcm16_bytes(marked, SAMPLE_RATE))

        rows.append(
            benchmark_row(
                root=output_dir,
                case_id=f"{name}-clean",
                pair_id=name,
                arm="matched_negative",
                state="clean",
                path=clean_path,
                reference=None,
                transform_name="synthesize-carrier",
                transform_revision=RECIPE_VERSION,
                parameters={
                    "carrier": name,
                    "duration_s": duration_s,
                    "sample_rate": SAMPLE_RATE,
                    "dependencies": dependencies,
                },
                seed=executed_seed,
                expected="not_detected",
            )
        )
        rows.append(
            benchmark_row(
                root=output_dir,
                case_id=f"{name}-marked",
                pair_id=name,
                arm="positive",
                state="marked",
                path=marked_path,
                reference=clean_path,
                transform_name="audioseal-embed",
                transform_revision=f"audioseal@{dependencies['audioseal']}+hf@{audioseal_oracle.HF_REVISION}",
                parameters={
                    "message": message,
                    "sample_rate": SAMPLE_RATE,
                    "alpha": 1.0,
                    "dependencies": dependencies,
                },
                seed=MESSAGE_SEED,
                expected="detected",
            )
        )

        for attack in attacks:
            attacked_path = apply_attack(ffmpeg, attack, marked_path, artifacts)
            rows.append(
                benchmark_row(
                    root=output_dir,
                    case_id=f"{name}-{attack}",
                    pair_id=name,
                    arm="positive",
                    state="attacked",
                    path=attacked_path,
                    reference=clean_path,
                    transform_name=f"attack-{attack}",
                    transform_revision=RECIPE_VERSION,
                    parameters=attack_parameters(attack, dependencies),
                    seed=NOISE_ATTACK_SEED if attack == "noise_snr10" else None,
                    expected="unresolved",
                )
            )

        # The removed row asserts a removal outcome; only carriers the
        # transform measurably clears qualify (see REMOVED_BY_NOISE).
        removed_path = artifacts / f"{name}-marked-noise_snr10.wav"
        if removed_path.is_file() and name in REMOVED_BY_NOISE:
            rows.append(
                benchmark_row(
                    root=output_dir,
                    case_id=f"{name}-removed",
                    pair_id=name,
                    arm="positive",
                    state="removed",
                    path=removed_path,
                    reference=clean_path,
                    transform_name="remove-noise-snr10",
                    transform_revision=RECIPE_VERSION,
                    parameters=attack_parameters("noise_snr10", dependencies),
                    seed=NOISE_ATTACK_SEED,
                    expected="not_detected",
                )
            )

        # The forged row embeds a different message into the same clean
        # carrier: a watermark is present, with a foreign decoded payload.
        forged_message = message_bits(FORGED_MESSAGE_SEED)
        forged_samples = np.asarray(audioseal_oracle.embed(generator, clean, forged_message), dtype=np.float32)
        forged_path = artifacts / f"{name}-forged.wav"
        forged_path.write_bytes(wav_pcm16_bytes(forged_samples, SAMPLE_RATE))
        rows.append(
            benchmark_row(
                root=output_dir,
                case_id=f"{name}-forged",
                pair_id=name,
                arm="wrong_key",
                state="forged",
                path=forged_path,
                reference=clean_path,
                transform_name="audioseal-embed-foreign-message",
                transform_revision=f"audioseal@{dependencies['audioseal']}+hf@{audioseal_oracle.HF_REVISION}",
                parameters={
                    "message": list(forged_message),
                    "message_seed": FORGED_MESSAGE_SEED,
                    "sample_rate": SAMPLE_RATE,
                    "alpha": 1.0,
                    "dependencies": dependencies,
                },
                seed=FORGED_MESSAGE_SEED,
                # The audioseal decision rule is watermark PRESENCE, so a
                # foreign payload still reads detected; the decoded label is
                # what distinguishes it from the oracle's message.
                expected="detected",
            )
        )

    hard_negative = synth_carrier_hard_negative(duration_s)
    hard_path = artifacts / f"{HARD_NEGATIVE_CARRIER}.wav"
    hard_path.write_bytes(wav_pcm16_bytes(hard_negative, SAMPLE_RATE))
    rows.append(
        benchmark_row(
            root=output_dir,
            case_id=f"{HARD_NEGATIVE_CARRIER}-clean",
            pair_id=HARD_NEGATIVE_CARRIER,
            arm="hard_negative",
            state="clean",
            path=hard_path,
            reference=None,
            transform_name="synthesize-carrier",
            transform_revision=RECIPE_VERSION,
            parameters={
                "carrier": "white_noise",
                "seed_offset": 991,
                "duration_s": duration_s,
                "sample_rate": SAMPLE_RATE,
                "dependencies": dependencies,
            },
            seed=HARD_NEGATIVE_SEED,
            expected="not_detected",
        )
    )

    if include_speech:
        _append_speech_rows(
            rows,
            root=output_dir,
            artifacts=artifacts,
            ffmpeg=ffmpeg,
            attacks=attacks,
            dependencies=dependencies,
            generator=generator,
            message=message,
        )

    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    load_manifest(manifest)
    return manifest


def _append_speech_rows(
    rows: list[dict[str, object]],
    *,
    root: Path,
    artifacts: Path,
    ffmpeg: str,
    attacks: Sequence[str],
    dependencies: dict[str, str],
    generator: object,
    message: list[int],
) -> None:
    """Add system-voice speech carriers with their own TTS provenance."""
    from audioseal_experiment import (
        SPEECH_VOICES,
        assert_distinct_voices,
        decode_audio_f32,
        speech_provenance,
        synthesize_speech,
    )

    raw = {voice: synthesize_speech(voice, artifacts) for voice in SPEECH_VOICES}
    assert_distinct_voices({voice: path.read_bytes() for voice, path in raw.items()})
    provenance = speech_provenance()
    for voice, aiff in raw.items():
        pair = f"speech_{voice}"
        clean = decode_audio_f32(ffmpeg, aiff)
        clean_path = artifacts / f"{pair}-clean.wav"
        clean_path.write_bytes(wav_pcm16_bytes(clean, SAMPLE_RATE))
        tts = {**provenance, "voice": voice, "duration_s": round(clean.shape[0] / SAMPLE_RATE, 3)}

        marked = np.asarray(audioseal_oracle.embed(generator, clean, message), dtype=np.float32)
        marked_path = artifacts / f"{pair}-marked.wav"
        marked_path.write_bytes(wav_pcm16_bytes(marked, SAMPLE_RATE))

        rows.append(
            benchmark_row(
                root=root,
                case_id=f"{pair}-clean",
                pair_id=pair,
                arm="matched_negative",
                state="clean",
                path=clean_path,
                reference=None,
                transform_name="synthesize-speech",
                transform_revision=RECIPE_VERSION,
                parameters={"tts": tts, "sample_rate": SAMPLE_RATE, "dependencies": dependencies},
                seed=None,
                expected="not_detected",
            )
        )
        rows.append(
            benchmark_row(
                root=root,
                case_id=f"{pair}-marked",
                pair_id=pair,
                arm="positive",
                state="marked",
                path=marked_path,
                reference=clean_path,
                transform_name="audioseal-embed",
                transform_revision=f"audioseal@{dependencies['audioseal']}+hf@{audioseal_oracle.HF_REVISION}",
                parameters={
                    "message": message,
                    "sample_rate": SAMPLE_RATE,
                    "alpha": 1.0,
                    "tts": tts,
                    "dependencies": dependencies,
                },
                seed=MESSAGE_SEED,
                expected="detected",
            )
        )
        for attack in attacks:
            attacked_path = apply_attack(ffmpeg, attack, marked_path, artifacts)
            rows.append(
                benchmark_row(
                    root=root,
                    case_id=f"{pair}-{attack}",
                    pair_id=pair,
                    arm="positive",
                    state="attacked",
                    path=attacked_path,
                    reference=clean_path,
                    transform_name=f"attack-{attack}",
                    transform_revision=RECIPE_VERSION,
                    parameters={**attack_parameters(attack, dependencies), "tts": tts},
                    seed=NOISE_ATTACK_SEED if attack == "noise_snr10" else None,
                    expected="unresolved",
                )
            )


def synth_carrier_hard_negative(duration_s: float) -> np.ndarray:
    """A never-embedded white-noise carrier keyed off a distinct seed."""
    rng = np.random.default_rng(HARD_NEGATIVE_SEED)
    n = int(duration_s * SAMPLE_RATE)
    return (rng.normal(0.0, 0.1, n)).astype(np.float32)


def attack_parameters(attack: str, dependencies: dict[str, str]) -> dict[str, object]:
    """Return the exact parameters represented by an attack name."""
    if attack == "mp3_128k":
        values: dict[str, object] = {"codec": "libmp3lame", "bitrate": "128k"}
    elif attack == "aac_128k":
        values = {"codec": "aac", "bitrate": "128k", "container": "m4a"}
    elif attack == "noise_snr10":
        values = {"additive_gaussian": True, "snr_db": 10.0}
    else:
        raise ValueError(f"unknown attack {attack!r}")
    return {**values, "dependencies": dependencies}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument(
        "--with-speech",
        action="store_true",
        help="add deterministic system-voice speech carriers (requires the macOS say tool)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        manifest = build_cohort(args.output_dir.resolve(), duration_s=args.duration, include_speech=args.with_speech)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    log.info("Wrote validated audio benchmark cohort to %s", manifest)


if __name__ == "__main__":
    main()
