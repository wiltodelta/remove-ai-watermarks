#!/usr/bin/env python3
"""Build and run the local Perth and PixelSeal benchmark study.

The study uses only deterministic synthetic audio and the two publication-
cleared engine-selection carriers named below.  It measures Perth before and
after the production visible-video cleaning path, and PixelSeal before and
after JPEG plus a foreign-message embed.  Invisible image-removal profiles are
recorded as unavailable when the host has no CUDA device; this script never
falls back to a provider oracle or uploads an artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import perth_oracle  # noqa: E402
import pixelseal_oracle  # noqa: E402
import videoseal_oracle  # noqa: E402
from audioseal_experiment import (  # noqa: E402
    FRAME_COUNT,
    SPEECH_VOICES,
    aac_packet_bytes,
    assert_distinct_voices,
    carrier_seed,
    decode_audio_f32,
    encode_video,
    ffmpeg_version,
    require_tools,
    run_ffmpeg,
    sora_like_mark,
    speech_provenance,
    stamped_frame,
    synth_carrier,
    synthesize_speech,
    wav_pcm16_bytes,
)
from watermark_benchmark import SCHEMA_VERSION, run_benchmark, sha256_file, write_jsonl  # noqa: E402
from watermark_benchmark_cohort import attack_image, attack_parameters  # noqa: E402

from remove_ai_watermarks.video import remove_video_visible  # noqa: E402

log = logging.getLogger("perth_pixelseal_study")

RECIPE_VERSION = "perth-pixelseal-study-v2"
IMAGE_SIZE = 512
REMOVAL_PROFILES: tuple[str, ...] = ("qwen-zimage", "sdxl-zimage", "chroma-zimage")
IMAGE_CARRIERS: tuple[Path, ...] = (
    REPO / "data/evaluations/engine-selection/originals/openai/00.png",
    REPO / "data/evaluations/engine-selection/originals/meta/00.png",
)
AUDIO_CARRIERS: tuple[str, ...] = ("tone_stack", "pinkish_noise", "white_noise")
AUDIO_DURATION_S = 8.0


def _row(
    *,
    root: Path,
    case_id: str,
    pair_id: str,
    media_type: str,
    adapter: str,
    arm: str,
    state: str,
    path: Path,
    reference: Path | None,
    source_revision: str,
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
        "media_type": media_type,
        "adapter": adapter,
        "arm": arm,
        "state": state,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "reference_path": reference.relative_to(root).as_posix() if reference is not None else None,
        "reference_sha256": sha256_file(reference) if reference is not None else None,
        "source_revision": source_revision,
        "transform": {
            "name": transform_name,
            "revision": transform_revision,
            "parameters": parameters,
        },
        "seed": seed,
        "expected": expected,
    }


def _save_rgb(path: Path, values: np.ndarray) -> Image.Image:
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    image.save(path)
    return image


def _invisible_engine_available() -> bool:
    from remove_ai_watermarks.invisible_engine import is_available

    return is_available()


def _run_pixelseal_removal_profile(profile: str, source_path: Path, output_path: Path) -> None:
    from remove_ai_watermarks.invisible_engine import InvisibleEngine

    engine = InvisibleEngine(pipeline=profile)
    engine.remove_watermark(image_path=source_path, output_path=output_path)


def _build_image_seal_rows(
    output_dir: Path,
    artifacts: Path,
    *,
    adapter: str,
    stack_label: str,
    oracle: object,
    embed_fn: object,
    pins: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not oracle.available():  # type: ignore[attr-defined]
        failure = oracle.import_failure()  # type: ignore[attr-defined]
        raise SystemExit(f"{stack_label} development stack is unavailable: {failure}")
    model = oracle.load_model()  # type: ignore[attr-defined]
    fixed = oracle.message_bits()  # type: ignore[attr-defined]
    foreign = oracle.message_bits(seed=8)  # type: ignore[attr-defined]
    source_commit = str(oracle.SOURCE_COMMIT)  # type: ignore[attr-defined]
    message_seed = int(oracle.MESSAGE_SEED)  # type: ignore[attr-defined]
    full_pins = {**pins, "message": list(fixed)}
    rows: list[dict[str, object]] = []
    source_manifest = REPO / "data/evaluations/engine-selection/carrier-manifest.csv"
    source_revision = f"engine-selection@sha256:{sha256_file(source_manifest)}"
    marked_pairs: list[tuple[str, Path, Path]] = []
    for source in IMAGE_CARRIERS:
        provider = source.parent.name
        pair_id = f"{adapter}-{provider}-00"
        clean_path = artifacts / f"{pair_id}-clean.png"
        clean = Image.open(source).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        clean.save(clean_path)
        clean_values = np.asarray(clean, dtype=np.float32) / 255.0

        marked_path = artifacts / f"{pair_id}-marked.png"
        marked_image = _save_rgb(marked_path, np.asarray(embed_fn(model, clean_values, fixed)))  # type: ignore[operator]
        attacked_path = artifacts / f"{pair_id}-jpeg-q90.jpg"
        marked_image.save(attacked_path, quality=90, subsampling=0)
        resize_path = artifacts / f"{pair_id}-resize-75.png"
        attack_image(marked_image, "resize-75").save(resize_path)
        crop_path = artifacts / f"{pair_id}-crop-5.png"
        attack_image(marked_image, "crop-5").save(crop_path)
        forged_path = artifacts / f"{pair_id}-forged.png"
        _save_rgb(forged_path, np.asarray(embed_fn(model, clean_values, foreign)))  # type: ignore[operator]
        marked_pairs.append((pair_id, clean_path, marked_path))

        common = {
            "root": output_dir,
            "pair_id": pair_id,
            "media_type": "image",
            "adapter": adapter,
            "source_revision": source_revision,
        }
        rows.extend(
            (
                _row(
                    **common,
                    case_id=f"{pair_id}-clean",
                    arm="matched_negative",
                    state="clean",
                    path=clean_path,
                    reference=None,
                    transform_name="standardize-carrier",
                    transform_revision=RECIPE_VERSION,
                    parameters={"source_sha256": sha256_file(source), "size": IMAGE_SIZE},
                    seed=None,
                    expected="not_detected",
                ),
                _row(
                    **common,
                    case_id=f"{pair_id}-marked",
                    arm="positive",
                    state="marked",
                    path=marked_path,
                    reference=clean_path,
                    transform_name=f"{adapter}-embed",
                    transform_revision=f"videoseal@{source_commit}",
                    parameters=full_pins,
                    seed=message_seed,
                    expected="detected",
                ),
                _row(
                    **common,
                    case_id=f"{pair_id}-jpeg-q90",
                    arm="positive",
                    state="attacked",
                    path=attacked_path,
                    reference=clean_path,
                    transform_name="jpeg",
                    transform_revision="pillow",
                    parameters={**attack_parameters("jpeg-q90"), **full_pins},
                    seed=None,
                    expected="unresolved",
                ),
                _row(
                    **common,
                    case_id=f"{pair_id}-resize-75",
                    arm="positive",
                    state="attacked",
                    path=resize_path,
                    reference=clean_path,
                    transform_name="resize-75",
                    transform_revision="pillow",
                    parameters={**attack_parameters("resize-75"), **full_pins},
                    seed=None,
                    expected="unresolved",
                ),
                _row(
                    **common,
                    case_id=f"{pair_id}-crop-5",
                    arm="positive",
                    state="attacked",
                    path=crop_path,
                    reference=clean_path,
                    transform_name="crop-5",
                    transform_revision="pillow",
                    parameters={**attack_parameters("crop-5"), **full_pins},
                    seed=None,
                    expected="unresolved",
                ),
                _row(
                    **common,
                    case_id=f"{pair_id}-forged",
                    arm="wrong_key",
                    state="forged",
                    path=forged_path,
                    reference=clean_path,
                    transform_name=f"{adapter}-embed-foreign-message",
                    transform_revision=f"videoseal@{source_commit}",
                    parameters={**full_pins, "foreign_message": list(foreign), "foreign_message_seed": 8},
                    seed=8,
                    expected="not_detected",
                ),
            )
        )
    del model
    gc.collect()

    if not _cuda_available():
        removal_summary: dict[str, object] = {
            "status": "not_run",
            "reason": "qwen-zimage, sdxl-zimage, and chroma-zimage require an NVIDIA CUDA device",
            "profiles": [],
        }
    elif not _invisible_engine_available():
        removal_summary = {
            "status": "not_run",
            "reason": "invisible removal dependencies are not installed",
            "profiles": [],
        }
    else:
        for profile in REMOVAL_PROFILES:
            for pair_id, clean_path, marked_path in marked_pairs:
                removed_path = artifacts / f"{pair_id}-removed-{profile}.png"
                _run_pixelseal_removal_profile(profile, marked_path, removed_path)
                rows.append(
                    _row(
                        root=output_dir,
                        pair_id=pair_id,
                        media_type="image",
                        adapter=adapter,
                        source_revision=source_revision,
                        case_id=f"{pair_id}-removed-{profile}",
                        arm="positive",
                        state="removed",
                        path=removed_path,
                        reference=clean_path,
                        transform_name=f"remove-invisible-{profile}",
                        transform_revision=RECIPE_VERSION,
                        parameters={"profile": profile, **full_pins},
                        seed=0,
                        expected="unresolved",
                    )
                )
        removal_summary = {
            "status": "executed",
            "profiles": list(REMOVAL_PROFILES),
        }
    return rows, removal_summary


def _build_pixelseal_rows(output_dir: Path, artifacts: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    return _build_image_seal_rows(
        output_dir,
        artifacts,
        adapter="pixelseal",
        stack_label="PixelSeal",
        oracle=pixelseal_oracle,
        embed_fn=pixelseal_oracle.embed,
        pins={
            "source_commit": pixelseal_oracle.SOURCE_COMMIT,
            "checkpoint_sha256": pixelseal_oracle.CHECKPOINT_SHA256,
            "model_card_sha256": pixelseal_oracle.MODEL_CARD_SHA256,
        },
    )


def _build_videoseal_image_rows(output_dir: Path, artifacts: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    return _build_image_seal_rows(
        output_dir,
        artifacts,
        adapter="videoseal-image",
        stack_label="VideoSeal",
        oracle=videoseal_oracle,
        embed_fn=videoseal_oracle.embed_image,
        pins={
            "source_commit": videoseal_oracle.SOURCE_COMMIT,
            "checkpoint_sha256": videoseal_oracle.JIT_SHA256,
        },
    )


def _extract_audio(ffmpeg: str, source: Path, target: Path) -> None:
    run_ffmpeg(
        ffmpeg,
        [
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-ac",
            "1",
            "-ar",
            str(perth_oracle.SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
    )


def _build_perth_rows(
    output_dir: Path,
    artifacts: Path,
    ffmpeg: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not perth_oracle.available():
        failure = perth_oracle.import_failure()
        raise SystemExit(f"Perth development stack is unavailable: {failure}")
    model = perth_oracle.load_model()
    pins = {
        "source_commit": perth_oracle.SOURCE_COMMIT,
        "checkpoint_sha256": perth_oracle.CHECKPOINT_SHA256,
        "implicit_payload": True,
    }
    source_revision = f"{RECIPE_VERSION}@sha256:{sha256_file(Path(__file__))}"
    rows: list[dict[str, object]] = []
    samantha_marked_path: Path | None = None
    for name in AUDIO_CARRIERS:
        executed_seed = carrier_seed(name)
        clean_samples = synth_carrier(name, AUDIO_DURATION_S, seed=executed_seed)
        clean_path = artifacts / f"perth-{name}-clean.wav"
        clean_path.write_bytes(wav_pcm16_bytes(clean_samples, perth_oracle.SAMPLE_RATE))
        marked_path = artifacts / f"perth-{name}-marked.wav"
        marked = np.asarray(perth_oracle.embed(model, clean_samples), dtype=np.float32)
        marked_path.write_bytes(wav_pcm16_bytes(marked, perth_oracle.SAMPLE_RATE))
        common = {
            "root": output_dir,
            "pair_id": f"perth-{name}",
            "media_type": "audio",
            "adapter": "perth",
            "source_revision": source_revision,
        }
        rows.extend(
            (
                _row(
                    **common,
                    case_id=f"perth-{name}-clean",
                    arm="matched_negative",
                    state="clean",
                    path=clean_path,
                    reference=None,
                    transform_name="synthesize-carrier",
                    transform_revision=RECIPE_VERSION,
                    parameters={"carrier": name, "duration_s": AUDIO_DURATION_S},
                    seed=executed_seed,
                    expected="not_detected",
                ),
                _row(
                    **common,
                    case_id=f"perth-{name}-marked",
                    arm="positive",
                    state="marked",
                    path=marked_path,
                    reference=clean_path,
                    transform_name="perth-embed",
                    transform_revision=f"resemble-ai/Perth@{perth_oracle.SOURCE_COMMIT}",
                    parameters=pins,
                    seed=None,
                    # Perth is speech-trained; these deliberately out-of-domain
                    # carriers measure dependence without asserting detection.
                    expected="unresolved",
                ),
            )
        )

    raw_speech = {voice: synthesize_speech(voice, artifacts) for voice in SPEECH_VOICES}
    assert_distinct_voices({voice: path.read_bytes() for voice, path in raw_speech.items()})
    tts = speech_provenance()
    for voice, aiff in raw_speech.items():
        pair_id = f"perth-speech-{voice.casefold()}"
        clean_samples = decode_audio_f32(ffmpeg, aiff)
        clean_path = artifacts / f"{pair_id}-clean.wav"
        clean_path.write_bytes(wav_pcm16_bytes(clean_samples, perth_oracle.SAMPLE_RATE))
        marked = np.asarray(perth_oracle.embed(model, clean_samples), dtype=np.float32)
        marked_path = artifacts / f"{pair_id}-marked.wav"
        marked_path.write_bytes(wav_pcm16_bytes(marked, perth_oracle.SAMPLE_RATE))
        if voice == "Samantha":
            samantha_marked_path = marked_path
        common = {
            "root": output_dir,
            "pair_id": pair_id,
            "media_type": "audio",
            "adapter": "perth",
            "source_revision": source_revision,
        }
        provenance = {**tts, "voice": voice, "duration_s": round(clean_samples.shape[0] / perth_oracle.SAMPLE_RATE, 6)}
        rows.extend(
            (
                _row(
                    **common,
                    case_id=f"{pair_id}-clean",
                    arm="matched_negative",
                    state="clean",
                    path=clean_path,
                    reference=None,
                    transform_name="synthesize-speech",
                    transform_revision=RECIPE_VERSION,
                    parameters={"tts": provenance},
                    seed=None,
                    expected="not_detected",
                ),
                _row(
                    **common,
                    case_id=f"{pair_id}-marked",
                    arm="positive",
                    state="marked",
                    path=marked_path,
                    reference=clean_path,
                    transform_name="perth-embed",
                    transform_revision=f"resemble-ai/Perth@{perth_oracle.SOURCE_COMMIT}",
                    parameters={**pins, "tts": provenance},
                    seed=None,
                    expected="detected",
                ),
            )
        )

    mark, mark_x, mark_y = sora_like_mark()
    frames = [stamped_frame(index, mark, mark_x, mark_y) for index in range(FRAME_COUNT)]
    if samantha_marked_path is None:
        raise RuntimeError("the configured speech carriers must include Samantha")
    source_video = artifacts / "perth-video-source.mp4"
    encode_video(
        ffmpeg,
        source_video,
        frames,
        samantha_marked_path,
        audio_codec="aac",
        audio_rate=perth_oracle.SAMPLE_RATE,
    )
    cleaned_video = artifacts / "perth-video-cleaned.mp4"
    result = remove_video_visible(source_video, cleaned_video, mark="sora", backend="cv2")
    if result.output is None:
        raise SystemExit("visible-video cleaning did not produce an output")
    source_packets = aac_packet_bytes(ffmpeg, source_video)
    cleaned_packets = aac_packet_bytes(ffmpeg, cleaned_video)
    source_audio = artifacts / "perth-video-source-audio.wav"
    cleaned_audio = artifacts / "perth-video-cleaned-audio.wav"
    _extract_audio(ffmpeg, source_video, source_audio)
    _extract_audio(ffmpeg, cleaned_video, cleaned_audio)
    packet_identity = source_packets == cleaned_packets
    rows.extend(
        (
            _row(
                root=output_dir,
                case_id="perth-video-source-audio",
                pair_id="perth-video-pass-through",
                media_type="audio",
                adapter="perth",
                arm="positive",
                state="marked",
                path=source_audio,
                reference=None,
                source_revision=source_revision,
                transform_name="mux-aac",
                transform_revision=ffmpeg_version(ffmpeg),
                parameters={"codec": "aac", "bitrate": "128k", **pins},
                seed=None,
                expected="detected",
            ),
            _row(
                root=output_dir,
                case_id="perth-video-cleaned-audio",
                pair_id="perth-video-pass-through",
                media_type="audio",
                adapter="perth",
                arm="positive",
                state="attacked",
                path=cleaned_audio,
                reference=source_audio,
                source_revision=source_revision,
                transform_name="remove-video-visible-audio-pass-through",
                transform_revision=source_revision,
                parameters={
                    "audio_packets_identical": packet_identity,
                    "visual_removed_frames": result.removed_frames,
                    **pins,
                },
                seed=None,
                expected="detected",
            ),
        )
    )
    del model
    gc.collect()
    return rows, {
        "audio_packets_identical": packet_identity,
        "source_audio_packet_sha256": hashlib.sha256(source_packets).hexdigest(),
        "cleaned_audio_packet_sha256": hashlib.sha256(cleaned_packets).hexdigest(),
        "visual_removed_frames": result.removed_frames,
    }


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def build_and_run(output_dir: Path) -> Path:
    """Create the study artifacts, manifest, metadata, and kernel results."""
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    ffmpeg, _ffprobe = require_tools()
    output_dir.mkdir(parents=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir()
    rows, removal_summary = _build_pixelseal_rows(output_dir, artifacts)
    videoseal_rows, videoseal_removal = _build_videoseal_image_rows(output_dir, artifacts)
    rows.extend(videoseal_rows)
    perth_rows, pass_through = _build_perth_rows(output_dir, artifacts, ffmpeg)
    rows.extend(perth_rows)
    manifest = output_dir / "manifest.jsonl"
    write_jsonl(manifest, rows)
    metadata = {
        "schema_version": 1,
        "recipe": RECIPE_VERSION,
        "pixelseal": {
            "source_commit": pixelseal_oracle.SOURCE_COMMIT,
            "checkpoint_sha256": pixelseal_oracle.CHECKPOINT_SHA256,
            "removal_profiles": removal_summary,
        },
        "videoseal_image": {
            "source_commit": videoseal_oracle.SOURCE_COMMIT,
            "checkpoint_sha256": videoseal_oracle.JIT_SHA256,
            "removal_profiles": videoseal_removal,
        },
        "perth": {
            "source_commit": perth_oracle.SOURCE_COMMIT,
            "checkpoint_sha256": perth_oracle.CHECKPOINT_SHA256,
            "message": "not_applicable_fixed_implicit_mark",
            "video_visible_path": pass_through,
        },
    }
    (output_dir / "study-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = output_dir / "results.jsonl"
    run_benchmark(manifest, results)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        results = build_and_run(args.output_dir.resolve())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    log.info("Wrote study results to %s", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
