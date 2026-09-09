#!/usr/bin/env python3
"""Local AudioSeal audio-provenance experiment (development-only).

The study answers one question the image benchmark cannot: does a video stay
provenance-positive through its untouched audio track after a visual-only
cleaning pass? AudioSeal (Meta, MIT license) is embedded locally into
synthetic carriers, muxed into videos next to a visible Sora-like mark, and the
production ``remove_video_visible`` path is run on the result. That path
transcodes the video stream and stream-copies audio, so the experiment measures
three separable things and never merges them:

- ``bitstream_identity``: whether the cleaned video's audio packets are
  byte-identical to the source video's audio packets.
- ``cleaning_invariance``: whether the detector verdict on the decoded audio is
  unchanged by the visual cleaning pass.
- ``audio_attack``: how the watermark responds to audio-path processing only
  (codecs, resampling, noise, gain, lowpass) with no video involved.

Everything runs locally. No provider oracle or API is contacted, the package
runtime is untouched, and generated artifacts stay outside the repository under
``.local-eval/``. Weights are pinned to an exact Hugging Face revision and
verified against pinned SHA-256 digests before use.

    uv run python scripts/audioseal_experiment.py \
      --output-dir .local-eval/audioseal-experiment-v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import struct
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import audioseal_oracle as oracle  # noqa: E402
from audioseal_oracle import SAMPLE_RATE  # noqa: E402

log = logging.getLogger("audioseal_experiment")

DURATION_S = 8.0
FRAME_COUNT = 24
FRAME_RATE = 12.0
FRAME_WIDTH = 840
FRAME_HEIGHT = 480
CARRIER_SEED = 20260907
NOISE_ATTACK_SEED = CARRIER_SEED + 1
MESSAGE_SEED = 7


@dataclass(frozen=True)
class Detection:
    """AudioSeal detector readings on one decoded audio tensor."""

    mean_detect_prob: float
    frac_above_threshold: float
    bit_accuracy: float | None
    decoded_bits: list[int]

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_detect_prob": round(self.mean_detect_prob, 6),
            "frac_above_threshold": round(self.frac_above_threshold, 6),
            "bit_accuracy": None if self.bit_accuracy is None else round(self.bit_accuracy, 6),
            "decoded_bits": self.decoded_bits,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- Pure carrier and WAV helpers (unit-tested without torch) --------------


def carrier_seed(name: str) -> int:
    """Stable per-carrier seed; Python's hash() is salted per process."""
    return CARRIER_SEED + zlib.crc32(name.encode()) % 1000


def synth_carrier(name: str, seconds: float, *, seed: int | None = None) -> np.ndarray:
    """Synthesize one deterministic carrier by registered name."""
    factory = CARRIERS.get(name)
    if factory is None:
        raise KeyError(f"unknown carrier {name!r}; expected one of {sorted(CARRIERS)}")
    master = np.random.default_rng(carrier_seed(name) if seed is None else seed)
    return np.asarray(factory(int(seconds * SAMPLE_RATE), master), dtype=np.float32)


def tone_stack_carrier(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 1.1 * t)
    carrier = (
        0.5 * am * np.sin(2 * np.pi * 220.0 * t)
        + 0.3 * am * np.sin(2 * np.pi * 440.0 * t)
        + 0.2 * am * np.sin(2 * np.pi * 880.0 * t)
    )
    return carrier + rng.normal(0.0, 0.005, n)


def pinkish_carrier(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, n)
    kernel = np.ones(9) / 9.0
    smoothed = np.convolve(white, kernel, mode="same")
    return smoothed / np.std(smoothed) * 0.1


def white_carrier(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 0.1, n)


CARRIERS: dict[str, object] = {
    "tone_stack": tone_stack_carrier,
    "pinkish_noise": pinkish_carrier,
    "white_noise": white_carrier,
}


def message_bits(seed: int = MESSAGE_SEED) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(b) for b in rng.integers(0, 2, oracle.N_BITS)]


def wav_pcm16_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Serialize mono float samples in [-1, 1) as a 16-bit PCM WAV file."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    payload = pcm.tobytes()
    return b"".join(
        (
            b"RIFF",
            struct.pack("<I", 36 + len(payload)),
            b"WAVEfmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16),
            b"data",
            struct.pack("<I", len(payload)),
            payload,
        )
    )


def snr_db(signal: np.ndarray, residual: np.ndarray) -> float:
    power = float(np.sum(signal.astype(np.float64) ** 2))
    noise = float(np.sum(residual.astype(np.float64) ** 2))
    return 10.0 * np.log10(power / max(noise, 1e-12))


# --- ffmpeg helpers --------------------------------------------------------


def require_tools() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise SystemExit("audioseal_experiment requires ffmpeg and ffprobe on PATH")
    return ffmpeg, ffprobe


def run_ffmpeg(ffmpeg: str, args: Sequence[str], *, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(  # noqa: S603
        [ffmpeg, "-y", "-loglevel", "error", *args],
        input=stdin,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {result.stderr.decode(errors='replace')[:400]}")
    return result.stdout


def ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run([ffmpeg, "-version"], capture_output=True, check=False)  # noqa: S603
    return result.stdout.decode(errors="replace").splitlines()[0].strip()


def decode_audio_f32(ffmpeg: str, path: Path) -> np.ndarray:
    raw = run_ffmpeg(
        ffmpeg,
        ["-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
    )
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def aac_packet_bytes(ffmpeg: str, path: Path) -> bytes:
    return run_ffmpeg(ffmpeg, ["-i", str(path), "-map", "0:a:0", "-c:a", "copy", "-f", "adts", "pipe:1"])


def pcm_packet_bytes(ffmpeg: str, path: Path) -> bytes:
    return run_ffmpeg(ffmpeg, ["-i", str(path), "-map", "0:a:0", "-c:a", "copy", "-f", "s16le", "pipe:1"])


# --- Synthetic video (independent Sora-like mark, moving background) -------


def moving_background(frame_index: int) -> np.ndarray:
    x = np.arange(FRAME_WIDTH, dtype=np.float32)[None, :]
    y = np.arange(FRAME_HEIGHT, dtype=np.float32)[:, None]
    luma = 42 + 10 * np.sin((x + frame_index * 5) / 38) + 5 * np.cos((y - frame_index * 2) / 54)
    frame = np.stack(
        (
            np.clip(luma - 5, 0, 255),
            np.clip(luma + 1, 0, 255),
            np.clip(luma + 7, 0, 255),
        ),
        axis=2,
    ).astype(np.uint8)
    moving_x = 30 + frame_index * 8
    frame[moving_x : moving_x + 72, 70:132] = (80, 140, 210)
    frame[220 + frame_index : 223 + frame_index, :] = (110, 70, 45)
    return frame


def sora_like_mark() -> tuple[np.ndarray, int, int]:
    """Render a Sora-like mark without reusing the detector's template."""
    mark = Image.new("L", (180, 64), 0)
    draw = ImageDraw.Draw(mark)
    draw.ellipse((1, 14, 32, 54), fill=255)
    draw.ellipse((25, 8, 62, 58), fill=255)
    draw.ellipse((15, 20, 28, 44), fill=0)
    draw.ellipse((37, 18, 50, 43), fill=0)
    from PIL import ImageFont

    try:
        font = ImageFont.load_default(size=49)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((68, 1), "Sora", font=font, fill=255, stroke_width=1)
    resized = mark.resize((124, 44), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.uint8), 620, 398


def stamped_frame(frame_index: int, mark: np.ndarray, mark_x: int, mark_y: int) -> np.ndarray:
    frame = moving_background(frame_index)
    height, width = mark.shape
    crop = frame[mark_y : mark_y + height, mark_x : mark_x + width].astype(np.float32)
    alpha = mark[:, :, None].astype(np.float32) / 255.0 * 0.78
    frame[mark_y : mark_y + height, mark_x : mark_x + width] = np.clip(crop * (1 - alpha) + 255 * alpha, 0, 255).astype(
        np.uint8
    )
    return frame


def encode_video(
    ffmpeg: str,
    path: Path,
    frames: list[np.ndarray],
    audio: Path,
    *,
    audio_codec: str,
    audio_rate: int,
) -> None:
    """Encode raw BGR frames plus a WAV audio input into one container."""
    height, width = frames[0].shape[:2]
    duration = len(frames) / FRAME_RATE
    args = [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{FRAME_RATE:.12g}",
        "-i",
        "pipe:0",
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "8",
        "-pix_fmt",
        "yuv420p",
    ]
    if audio_codec == "pcm_s16le":
        args += ["-c:a", "pcm_s16le"]
    else:
        args += [
            "-c:a",
            audio_codec,
            "-b:a",
            "128k",
            "-ar",
            str(audio_rate),
        ]
    args += ["-t", f"{duration:.12g}", str(path)]
    payload = b"".join(frame.astype(np.uint8).tobytes() for frame in frames)
    run_ffmpeg(ffmpeg, args, stdin=payload)


# --- AudioSeal oracle (shared, revision-pinned) -----------------------------


def load_pinned_models() -> tuple[object, object, dict[str, str]]:
    """Load the shared pinned oracle models and record their provenance."""
    import importlib.metadata as md
    import inspect

    import torch
    from audioseal_oracle import load_pinned_models as shared_load

    generator, detector = shared_load()
    versions = {
        "audioseal": md.version("audioseal"),
        "torch": torch.__version__,
        "huggingface_revision": oracle.HF_REVISION,
        "generator_sha256": oracle.GENERATOR_SHA256,
        "detector_sha256": oracle.DETECTOR_SHA256,
        "oracle_source_sha256": sha256_file(Path(inspect.getfile(shared_load))),
    }
    return generator, detector, versions


class DetectorClock:
    """Records the first (cold) detector call separately from warm calls."""

    def __init__(self) -> None:
        self.cold_ms: float | None = None
        self.warm_ms: list[float] = []

    def measure(self, call: object, wav: object) -> object:
        started = perf_counter_ns()
        result = call(wav)
        elapsed = (perf_counter_ns() - started) / 1e6
        if self.cold_ms is None:
            self.cold_ms = elapsed
        else:
            self.warm_ms.append(elapsed)
        return result


def detect_audio(detector: object, clock: DetectorClock, samples: np.ndarray) -> Detection:
    """Read one detection under the clock and score it against the message."""
    reading = clock.measure(lambda wav: oracle.read(detector, wav), samples)
    expected = message_bits()
    bits = list(reading.decoded_bits)
    accuracy = sum(int(b == e) for b, e in zip(bits, expected, strict=True)) / oracle.N_BITS
    return Detection(
        mean_detect_prob=reading.mean_detect_prob,
        frac_above_threshold=reading.frac_above_threshold,
        bit_accuracy=accuracy,
        decoded_bits=bits,
    )


def embed_audio(generator: object, samples: np.ndarray, message: list[int]) -> np.ndarray:
    """Embed the fixed message through the shared pinned oracle."""
    return np.asarray(oracle.embed(generator, samples, message), dtype=np.float32)


# --- Attack transforms -----------------------------------------------------


def numpy_attacks(samples: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(NOISE_ATTACK_SEED)
    out: dict[str, np.ndarray] = {}
    power = float(np.sum(samples.astype(np.float64) ** 2))
    noise = rng.normal(0.0, 1.0, samples.shape[0]).astype(np.float32)
    noise *= np.sqrt(power / max(np.sum(noise.astype(np.float64) ** 2), 1e-12) / 10.0)
    out["noise_snr10"] = samples + noise
    out["gain_0.5"] = samples * 0.5
    return out


def partial_noise_attack(samples: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Mix 10 dB-SNR gaussian noise into one time window, leaving the rest bit-identical."""
    rng = np.random.default_rng(CARRIER_SEED + 2)
    out = samples.copy()
    start = int(start_s * SAMPLE_RATE)
    end = int(end_s * SAMPLE_RATE)
    window = out[start:end]
    power = float(np.sum(window.astype(np.float64) ** 2))
    noise = rng.normal(0.0, 1.0, window.shape[0]).astype(np.float32)
    noise *= np.sqrt(power / max(np.sum(noise.astype(np.float64) ** 2), 1e-12) / 10.0)
    out[start:end] = window + noise
    return out


# --- Local speech carriers (system TTS, no network) --------------------------

SPEECH_TEXT = (
    "Provenance through audio remains measurable after visual cleaning. "
    "The second sentence extends this carrier past three seconds."
)
SPEECH_VOICES: tuple[str, ...] = ("Samantha", "Daniel", "Milena")


def assert_distinct_voices(artifacts: dict[str, bytes]) -> None:
    """Reject voice aliasing: two names rendering byte-identical audio.

    macOS 26.6.2 aliases ``Alex`` to ``Samantha``; a carrier-diversity claim
    built on aliased voices is one carrier counted twice. Verified by content
    hash, never by voice name.
    """
    seen: dict[str, str] = {}
    for voice, payload in artifacts.items():
        digest = sha256_bytes(payload)
        duplicate = seen.get(digest)
        if duplicate is not None:
            raise SystemExit(
                f"system voices {duplicate!r} and {voice!r} render byte-identical audio; pick genuinely distinct voices"
            )
        seen[digest] = voice


def speech_provenance() -> dict[str, str]:
    """Record everything that pins system-synthesized speech bytes."""
    import platform

    product = ""
    sw_vers = shutil.which("sw_vers")
    if sw_vers is not None:
        result = subprocess.run(  # noqa: S603 - resolved sw_vers with fixed arguments
            [sw_vers, "-productVersion"], capture_output=True, text=True, check=False
        )
        product = result.stdout.strip()
    return {
        "tts": "say",
        "macos": product or platform.mac_ver()[0],
        "text_sha256": sha256_bytes(SPEECH_TEXT.encode()),
    }


def synthesize_speech(voice: str, workdir: Path) -> Path:
    """Render one deterministic utterance through the local system voice."""
    say = shutil.which("say")
    if say is None:
        raise SystemExit("speech carriers require the macOS say tool")
    target = workdir / f"speech-{voice}.aiff"
    result = subprocess.run(  # noqa: S603 - resolved say with fixed arguments
        [say, "-o", str(target), "-v", voice, SPEECH_TEXT],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not target.is_file():
        raise SystemExit(f"system voice {voice!r} is unavailable: {result.stderr.decode(errors='replace')[:200]}")
    return target


def speech_samples(voice: str, workdir: Path, ffmpeg: str) -> np.ndarray:
    """Decode one system-voice utterance to mono 16 kHz float32."""
    return decode_audio_f32(ffmpeg, synthesize_speech(voice, workdir))


def ffmpeg_attack(
    ffmpeg: str, name: str, suffix: str, args: Sequence[str], source: Path, workdir: Path, *, stem: str = "attack"
) -> Path:
    target = workdir / f"{stem}-{name}.{suffix}"
    run_ffmpeg(ffmpeg, ["-i", str(source), *args, str(target)])
    return target


FFMPEG_ATTACKS: dict[str, tuple[str, list[str]]] = {
    # AAC attacks use the MP4/M4A container so the decode honors the encoder's
    # priming edit list. A raw ADTS decode keeps the 1024-sample encoder-delay
    # head in the stream, which measures the no-gapless decode path rather
    # than the codec itself; that path is kept as its own explicit arm below.
    "aac_128k": ("m4a", ["-c:a", "aac", "-b:a", "128k"]),
    "aac_128k_48k": ("m4a", ["-c:a", "aac", "-b:a", "128k", "-ar", "48000"]),
    "aac_128k_adts_raw": ("adts", ["-c:a", "aac", "-b:a", "128k"]),
    "mp3_128k": ("mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
    "mp3_64k": ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
    "resample_44k_wav": ("wav", ["-c:a", "pcm_s16le", "-ar", "44100"]),
    "lowpass_8k_wav": ("wav", ["-c:a", "pcm_s16le", "-af", "lowpass=f=8000"]),
}


# --- Experiment ------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-video", action="store_true", help="audio stages only")
    parser.add_argument(
        "--skip-speech", action="store_true", help="skip system-voice speech carriers and interval readings"
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output directory {out_dir}")
    out_dir.mkdir(parents=True)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ffmpeg, _ffprobe = require_tools()

    generator, detector, versions = load_pinned_models()
    clock = DetectorClock()
    rows: list[dict[str, object]] = []

    def record(stage: str, **fields: object) -> None:
        row = {"stage": stage, **fields}
        rows.append(row)
        log.info("%s", json.dumps(row, sort_keys=True, default=str)[:400])

    def detect_and_record(stage: str, label: str, samples: np.ndarray, path: Path | None, **extra: object) -> Detection:
        det = detect_audio(detector, clock, samples)
        record(
            stage,
            label=label,
            detection=det.as_dict(),
            sha256=None if path is None else sha256_file(path),
            **extra,
        )
        return det

    message = message_bits()
    record("pins", message=message, **versions, ffmpeg=ffmpeg_version(ffmpeg))

    # Stage 1: carriers, clean vs marked.
    marked_wavs: dict[str, Path] = {}
    for name in CARRIERS:
        clean = synth_carrier(name, DURATION_S)
        clean_path = out_dir / f"{name}-clean.wav"
        clean_path.write_bytes(wav_pcm16_bytes(clean, SAMPLE_RATE))
        detect_and_record("carrier_clean", f"{name}/clean", clean, clean_path)

        marked = embed_audio(generator, clean, message)
        marked_path = out_dir / f"{name}-marked.wav"
        marked_path.write_bytes(wav_pcm16_bytes(marked, SAMPLE_RATE))
        detect_and_record(
            "carrier_marked",
            f"{name}/marked",
            marked,
            marked_path,
            watermark_snr_db=round(snr_db(clean, marked - clean), 3),
        )
        marked_wavs[name] = marked_path

    # Stage 2: audio attacks on the tone_stack marked wav only; the codec arms
    # double as the audio tracks used for the video arms below.
    primary = marked_wavs["tone_stack"]
    attack_paths: dict[str, Path] = {}
    for name, (suffix, ffargs) in FFMPEG_ATTACKS.items():
        target = ffmpeg_attack(ffmpeg, name, suffix, ffargs, primary, out_dir)
        attack_paths[name] = target
        detect_and_record("audio_attack", name, decode_audio_f32(ffmpeg, target), target)
    for name, samples in numpy_attacks(decode_audio_f32(ffmpeg, primary)).items():
        target = out_dir / f"attack-{name}.wav"
        target.write_bytes(wav_pcm16_bytes(samples, SAMPLE_RATE))
        attack_paths[name] = target
        detect_and_record("audio_attack", name, samples, target)

    # Stage 2b: system-voice speech carriers with per-second interval readings.
    # Speech is the detector's training domain, and the interval report shows
    # WHERE in time the signal lives - including its localized death under an
    # attack confined to seconds two through four.
    if not args.skip_speech:
        from audioseal_oracle import read_intervals

        provenance = speech_provenance()
        raw_speech: dict[str, Path] = {}
        for voice in SPEECH_VOICES:
            raw_speech[voice] = synthesize_speech(voice, out_dir)
        assert_distinct_voices({voice: path.read_bytes() for voice, path in raw_speech.items()})
        for voice in SPEECH_VOICES:
            label = f"speech_{voice}"
            clean_speech = decode_audio_f32(ffmpeg, raw_speech[voice])
            clean_path = out_dir / f"{label}-clean.wav"
            clean_path.write_bytes(wav_pcm16_bytes(clean_speech, SAMPLE_RATE))
            detect_and_record(
                "speech_carrier",
                f"{label}/clean",
                clean_speech,
                clean_path,
                tts={**provenance, "voice": voice},
                duration_s=round(clean_speech.shape[0] / SAMPLE_RATE, 3),
            )

            marked_speech = embed_audio(generator, clean_speech, message)
            marked_path = out_dir / f"{label}-marked.wav"
            marked_path.write_bytes(wav_pcm16_bytes(marked_speech, SAMPLE_RATE))
            detect_and_record(
                "speech_carrier",
                f"{label}/marked",
                marked_speech,
                marked_path,
                tts={**provenance, "voice": voice},
                watermark_snr_db=round(snr_db(clean_speech, marked_speech - clean_speech), 3),
            )
            partial = partial_noise_attack(marked_speech, 2.0, 4.0)
            partial_path = out_dir / f"{label}-partial_noise_2s_4s.wav"
            partial_path.write_bytes(wav_pcm16_bytes(partial, SAMPLE_RATE))
            for tag, samples, artifact_path in (
                ("marked", marked_speech, marked_path),
                ("partial_noise_2s_4s", partial, partial_path),
            ):
                intervals = read_intervals(detector, samples)
                record(
                    "speech_intervals",
                    label=f"{label}/{tag}",
                    per_second=[
                        {
                            "start_s": index,
                            "mean_detect_prob": round(reading.mean_detect_prob, 6),
                            "frac_above_threshold": round(reading.frac_above_threshold, 6),
                            "detected": reading.detected,
                        }
                        for index, reading in enumerate(intervals)
                    ],
                    sha256=sha256_file(artifact_path),
                )

    # Stage 3: video arms. Each arm muxes the SAME marked wav next to a
    # stamped Sora-like mark, differing only in how audio is carried.
    if not args.skip_video:
        from remove_ai_watermarks.video import remove_video_visible

        mark, mark_x, mark_y = sora_like_mark()
        frames = [stamped_frame(i, mark, mark_x, mark_y) for i in range(FRAME_COUNT)]
        arms: dict[str, tuple[str, int | None]] = {
            "pcm_mov": ("pcm_s16le", None),
            "aac_16k": ("aac", SAMPLE_RATE),
            "aac_48k": ("aac", 48_000),
        }
        for arm, (codec, rate) in arms.items():
            suffix = "mov" if codec == "pcm_s16le" else "mp4"
            source = out_dir / f"video-{arm}-source.{suffix}"
            encode_video(ffmpeg, source, frames, primary, audio_codec=codec, audio_rate=rate or SAMPLE_RATE)
            packetize = pcm_packet_bytes if codec == "pcm_s16le" else aac_packet_bytes
            source_packets = packetize(ffmpeg, source)
            source_digest = sha256_bytes(source_packets)

            cleaned = out_dir / f"video-{arm}-cleaned.{suffix}"
            result = remove_video_visible(source, cleaned, mark="sora", backend="cv2")
            if result.output is None:
                raise SystemExit(f"visual cleaning wrote no output for arm {arm}")
            cleaned_packets = packetize(ffmpeg, cleaned)
            cleaned_digest = sha256_bytes(cleaned_packets)
            identical = source_digest == cleaned_digest

            det_source = detect_and_record(
                "video_audio",
                f"{arm}/source",
                decode_audio_f32(ffmpeg, source),
                source,
                audio_codec=codec,
            )
            det_cleaned = detect_and_record(
                "video_audio_cleaned",
                f"{arm}/cleaned",
                decode_audio_f32(ffmpeg, cleaned),
                cleaned,
                audio_codec=codec,
                bitstream_identical=identical,
                source_packet_sha256=source_digest,
                cleaned_packet_sha256=cleaned_digest,
                visual_removed_frames=result.removed_frames,
            )
            invariance = (
                det_source.mean_detect_prob == det_cleaned.mean_detect_prob
                and det_source.frac_above_threshold == det_cleaned.frac_above_threshold
                and det_source.decoded_bits == det_cleaned.decoded_bits
            )
            record(
                "cleaning_invariance",
                arm=arm,
                bitstream_identical=identical,
                verdict_invariant=invariance,
            )

    # Timing summary and reports.
    warm = sorted(clock.warm_ms)
    timing = {
        "detector_cold_ms": None if clock.cold_ms is None else round(clock.cold_ms, 3),
        "detector_warm_n": len(warm),
        "detector_warm_median_ms": None if not warm else round(warm[len(warm) // 2], 3),
        "detector_warm_max_ms": None if not warm else round(warm[-1], 3),
    }
    cases_path = out_dir / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    summary_path = out_dir / "summary.md"
    summary_path.write_text(render_summary(rows, timing, versions, ffmpeg, out_dir))
    print(f"wrote {cases_path}")
    print(f"wrote {summary_path}")
    return 0


def render_summary(
    rows: list[dict[str, object]],
    timing: dict[str, object],
    versions: dict[str, str],
    ffmpeg: str,
    out_dir: Path,
) -> str:
    def table(stage: str, columns: list[str]) -> list[str]:
        selected = [r for r in rows if r["stage"] == stage]
        lines = [
            f"### {stage}",
            "",
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
        ]
        for r in selected:
            det = r.get("detection", {}) if isinstance(r.get("detection"), dict) else {}
            cells = []
            for col in columns:
                if col == "mean_detect":
                    cells.append(f"{det.get('mean_detect_prob', '')}")
                elif col == "frac>0.5":
                    cells.append(f"{det.get('frac_above_threshold', '')}")
                elif col == "bit_acc":
                    cells.append(f"{det.get('bit_accuracy', '')}")
                elif col == "label":
                    cells.append(str(r.get("label", r.get("arm", ""))))
                else:
                    cells.append(str(r.get(col, "")))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        return lines

    def interval_table() -> list[str]:
        selected = [r for r in rows if r["stage"] == "speech_intervals"]
        lines = [
            "### speech_intervals",
            "",
            "| label | second | mean_detect | frac>0.5 | detected |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in selected:
            for window in r.get("per_second", []):
                if not isinstance(window, dict):
                    continue
                lines.append(
                    "| "
                    + " | ".join(
                        (
                            str(r.get("label", "")),
                            str(window.get("start_s", "")),
                            str(window.get("mean_detect_prob", "")),
                            str(window.get("frac_above_threshold", "")),
                            str(window.get("detected", "")),
                        )
                    )
                    + " |"
                )
        lines.append("")
        return lines

    lines = [
        "# AudioSeal audio-provenance experiment",
        "",
        f"- output: `{out_dir.name}`",
        f"- revision: `{versions['huggingface_revision']}`",
        f"- audioseal {versions['audioseal']}, torch {versions['torch']}",
        f"- ffmpeg: `{ffmpeg_version(ffmpeg)}`",
        f"- detector timing: {json.dumps(timing)}",
        "",
        *table("carrier_clean", ["label", "mean_detect", "frac>0.5"]),
        *table("speech_carrier", ["label", "mean_detect", "frac>0.5", "bit_acc", "watermark_snr_db"]),
        *interval_table(),
        *table("carrier_marked", ["label", "mean_detect", "frac>0.5", "bit_acc", "watermark_snr_db"]),
        *table("audio_attack", ["label", "mean_detect", "frac>0.5", "bit_acc"]),
        *table("video_audio", ["label", "mean_detect", "frac>0.5", "bit_acc"]),
        *table("video_audio_cleaned", ["label", "mean_detect", "frac>0.5", "bit_acc", "bitstream_identical"]),
        *table("cleaning_invariance", ["arm", "bitstream_identical", "verdict_invariant"]),
        "",
        "`bitstream_identical` compares audio packet digests of the source and",
        "visually cleaned videos. `verdict_invariant` additionally requires the",
        "detector readings to match. Rows are measurements on synthetic",
        "carriers, not detector-quality claims.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
