#!/usr/bin/env python3
"""Run reproducible end-to-end audio and video watermark removal benchmarks.

Unlike ``scripts/watermark_benchmark.py`` (which evaluates static pre-saved
artifacts), this maintainer tool exercises the public removal pipelines
(``remove_audio_invisible``, ``remove_video_audio``, ``remove_video_message``)
across clean negative controls and marked positive inputs, binding source and
delivered digests alongside detector verdicts and delivered quality metrics.

    uv run python scripts/media_removal_benchmark.py --builtin-suite \\
        --work-dir /tmp/raiw-bench-work --output /tmp/raiw-bench-report.json \\
        --markdown /tmp/raiw-bench-report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, get_args

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from remove_ai_watermarks import audio, video, video_audio, video_messages  # noqa: E402
from remove_ai_watermarks._internal import audio_io  # noqa: E402
from remove_ai_watermarks._internal.audio_quality import compute_perceptual_metrics, sha256_file  # noqa: E402
from remove_ai_watermarks._internal.schema import require_schema_version  # noqa: E402
from remove_ai_watermarks.video_tools import require_video_tool  # noqa: E402

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
Surface = Literal[
    "audio_invisible",
    "video_audio",
    "video_audio_all_streams",
    "video_message",
    "video_all_4layer",
]
ExpectedOutcome = Literal["skipped_clean", "verified_removed"]


@dataclass(frozen=True)
class RemovalBenchmarkCase:
    """One immutable end-to-end removal benchmark case."""

    case_id: str
    surface: Surface
    scheme: str
    source_path: Path
    source_sha256: str
    output_suffix: str
    expected_outcome: ExpectedOutcome
    method: str = "auto"
    stream: int = 0
    reference_hex: str | None = None
    long_side: int = video_messages.DEFAULT_VIDEO_MESSAGE_LONG_SIDE
    device: str = "cpu"


def _ensure_new_file(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing benchmark output: {path}")


def load_manifest(manifest_path: Path) -> list[RemovalBenchmarkCase]:
    """Parse and validate a JSONL removal benchmark manifest."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Removal benchmark manifest not found: {manifest_path}")
    base_dir = manifest_path.parent
    cases: list[RemovalBenchmarkCase] = []
    seen_ids: set[str] = set()
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError(f"{manifest_path}:{line_no}: expected JSON object")
        require_schema_version(
            payload.get("schema_version"),
            contract=f"{manifest_path}:{line_no}",
            supported={SCHEMA_VERSION},
        )
        case_id = str(payload["case_id"])
        if case_id in seen_ids:
            raise ValueError(f"{manifest_path}:{line_no}: duplicate case_id {case_id!r}")
        seen_ids.add(case_id)
        surface = payload["surface"]
        if surface not in get_args(Surface):
            raise ValueError(f"{manifest_path}:{line_no}: unsupported surface {surface!r}")
        expected = payload["expected_outcome"]
        if expected not in get_args(ExpectedOutcome):
            raise ValueError(f"{manifest_path}:{line_no}: unsupported expected_outcome {expected!r}")
        method = str(payload.get("method", "auto"))
        allowed_methods = video_messages.VIDEO_MESSAGE_METHODS if surface == "video_message" else audio.AUDIO_METHODS
        if method not in allowed_methods:
            raise ValueError(f"{manifest_path}:{line_no}: unsupported method {method!r} for surface {surface!r}")
        src = Path(payload["source_path"])
        if not src.is_absolute():
            src = (base_dir / src).resolve()
        cases.append(
            RemovalBenchmarkCase(
                case_id=case_id,
                surface=surface,
                scheme=str(payload["scheme"]),
                source_path=src,
                source_sha256=str(payload["source_sha256"]),
                output_suffix=str(payload["output_suffix"]),
                expected_outcome=expected,
                method=method,
                stream=int(payload.get("stream", 0)),
                reference_hex=payload.get("reference_hex"),
                long_side=int(payload.get("long_side", video_messages.DEFAULT_VIDEO_MESSAGE_LONG_SIDE)),
                device=str(payload.get("device", "cpu")),
            )
        )
    if not cases:
        raise ValueError(f"Removal benchmark manifest is empty: {manifest_path}")
    return cases


def _synthetic_speechlike_carrier(sample_rate: int, seconds: float, *, seed: int) -> Any:
    import numpy as np

    rng = np.random.default_rng(seed)
    total = round(sample_rate * seconds)
    t = np.arange(total, dtype=np.float64) / sample_rate
    # Harmonic carrier with syllabic amplitude modulation across 150 Hz..3.4 kHz
    env = 0.55 + 0.35 * np.sin(2.0 * np.pi * 3.2 * t) * np.cos(2.0 * np.pi * 1.1 * t)
    f0 = 155.0 + 12.0 * np.sin(2.0 * np.pi * 0.7 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / sample_rate
    sig = np.zeros_like(t)
    for k, amp in ((1, 0.45), (2, 0.28), (3, 0.18), (5, 0.12), (8, 0.10), (11, 0.07), (16, 0.05)):
        if k * 170.0 < 0.45 * sample_rate:
            sig += amp * np.sin(k * phase + float(rng.uniform(0.0, 2.0 * np.pi)))
    sig = env * sig + 0.008 * rng.standard_normal(total)
    peak = float(np.max(np.abs(sig)))
    if peak > 0.0:
        sig = 0.42 * sig / peak
    return sig.astype(np.float32)


STRICT_BUILTIN_REQUIRED_CASE_IDS: tuple[str, ...] = (
    "audioseal-mono-clean",
    "audioseal-mono-marked",
    "auto-scheme-audioseal-marked",
    "audioseal-lossy-m4a-marked",
    "audioseal-lossy-mp3-marked",
    "audioseal-stereo-marked",
    "audioseal-long31s-marked",
    "audioseal-video-mp4-aac192k",
    "video-audio-multistream-marked",
    "wavmark-clean",
    "wavmark-marked",
    "wavmark-stereo-ms-marked",
    "perth-clean",
    "perth-marked-auto",
    "videoseal-clean",
    "videoseal-marked-auto",
    "video-all-4layer-marked",
)


def build_builtin_suite(work_dir: Path, *, strict: bool = False) -> Path:
    """Synthesize deterministic clean/marked audio and video fixtures and write manifest.jsonl."""
    import cv2
    import numpy as np
    import soundfile as sf

    from remove_ai_watermarks._internal import audio_dac, audio_focalcodec, audioseal, perth, wavmark
    from remove_ai_watermarks.video_decode import open_video_capture
    from remove_ai_watermarks.video_encoding import (
        finish_raw_video_encoder,
        probe_video_encode_profile,
        raw_video_command,
        start_raw_video_encoder,
    )
    from scripts import videoseal_oracle

    fixtures_dir = work_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def add_row(
        case_id: str,
        surface: Surface,
        scheme: str,
        src_path: Path,
        output_suffix: str,
        expected: ExpectedOutcome,
        **extra: Any,
    ) -> None:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "surface": surface,
                "scheme": scheme,
                "source_path": str(src_path),
                "source_sha256": sha256_file(src_path),
                "output_suffix": output_suffix,
                "expected_outcome": expected,
                **extra,
            }
        )

    # 1. AudioSeal mono 16 kHz, stereo 16 kHz, long 31 s 16 kHz, and 3-stream MP4 AAC 192k
    gen: Any = None
    if audioseal.available():
        gen, _det = audioseal.load_pinned_models()
        msg_bits = [int(b) for b in "1011001111000101"]
        clean_mono = _synthetic_speechlike_carrier(16000, 3.0, seed=101)
        marked_mono = np.asarray(audioseal.embed(gen, clean_mono, msg_bits), dtype=np.float32)

        as_clean_p = fixtures_dir / "audioseal-mono-clean.wav"
        as_marked_p = fixtures_dir / "audioseal-mono-marked.wav"
        sf.write(as_clean_p, clean_mono, 16000, subtype="FLOAT")
        sf.write(as_marked_p, marked_mono, 16000, subtype="FLOAT")
        add_row("audioseal-mono-clean", "audio_invisible", "audioseal", as_clean_p, ".wav", "skipped_clean")
        add_row("audioseal-mono-marked", "audio_invisible", "audioseal", as_marked_p, ".wav", "verified_removed")
        add_row("auto-scheme-audioseal-marked", "audio_invisible", "auto", as_marked_p, ".wav", "verified_removed")
        add_row("audioseal-lossy-m4a-marked", "audio_invisible", "auto", as_marked_p, ".m4a", "verified_removed")
        add_row("audioseal-lossy-mp3-marked", "audio_invisible", "auto", as_marked_p, ".mp3", "verified_removed")

        clean_r = _synthetic_speechlike_carrier(16000, 3.0, seed=102)
        marked_r = np.asarray(audioseal.embed(gen, clean_r, msg_bits), dtype=np.float32)
        as_stereo_p = fixtures_dir / "audioseal-stereo-marked.wav"
        sf.write(as_stereo_p, np.column_stack((marked_mono, marked_r)), 16000, subtype="FLOAT")
        add_row("audioseal-stereo-marked", "audio_invisible", "audioseal", as_stereo_p, ".wav", "verified_removed")

        # 31 s long clip (> 30 s whole-clip boundary)
        tile_clean = _synthetic_speechlike_carrier(16000, 4.0, seed=103)
        tile_marked = np.asarray(audioseal.embed(gen, tile_clean, msg_bits), dtype=np.float32)
        long_marked = np.tile(tile_marked, 8)[: 16000 * 31]
        as_long_p = fixtures_dir / "audioseal-31s-marked.wav"
        sf.write(as_long_p, long_marked, 16000, subtype="FLOAT")
        add_row("audioseal-long31s-marked", "audio_invisible", "audioseal", as_long_p, ".wav", "verified_removed")

        # 3-stream MOV container -> delivered MP4 AAC 192k on stream=1 (second audio track)
        ffmpeg_bin = require_video_tool("ffmpeg")
        src_mov = fixtures_dir / "audioseal-container-marked.mov"
        subprocess.run(  # noqa: S603
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x1e293b:s=320x180:r=24:d=3.0",
                "-i",
                str(as_clean_p),
                "-i",
                str(as_marked_p),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "2:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-c:a:0",
                "aac",
                "-b:a:0",
                "192k",
                "-c:a:1",
                "pcm_f32le",
                "-metadata:s:a:0",
                "language=eng",
                "-metadata:s:a:1",
                "language=eng",
                str(src_mov),
            ],
            check=True,
        )
        add_row(
            "audioseal-video-mp4-aac192k",
            "video_audio",
            "audioseal",
            src_mov,
            ".mp4",
            "verified_removed",
            stream=1,
        )
        # Multi-stream container where BOTH audio tracks carry watermarks
        src_multi_mov = fixtures_dir / "audioseal-multistream-marked.mov"
        subprocess.run(  # noqa: S603
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=0x0f172a:s=320x180:r=24:d=3.0",
                "-i",
                str(as_marked_p),
                "-i",
                str(as_stereo_p),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-map",
                "2:a:0",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-c:a:0",
                "pcm_f32le",
                "-c:a:1",
                "pcm_f32le",
                "-metadata:s:a:0",
                "language=eng",
                "-metadata:s:a:1",
                "language=spa",
                str(src_multi_mov),
            ],
            check=True,
        )
        add_row(
            "video-audio-multistream-marked",
            "video_audio_all_streams",
            "auto",
            src_multi_mov,
            ".mp4",
            "verified_removed",
        )

    # 2. WavMark 16 kHz (mono + stereo Mid/Side EnCodec)
    if wavmark.available():
        wm_model = wavmark.load_model()
        wm_clean = _synthetic_speechlike_carrier(16000, 3.0, seed=201)
        wm_marked = np.asarray(wavmark.embed(wm_model, wm_clean), dtype=np.float32)
        wm_clean_p = fixtures_dir / "wavmark-clean.wav"
        wm_marked_p = fixtures_dir / "wavmark-marked.wav"
        sf.write(wm_clean_p, wm_clean, 16000, subtype="FLOAT")
        sf.write(wm_marked_p, wm_marked, 16000, subtype="FLOAT")
        add_row("wavmark-clean", "audio_invisible", "wavmark", wm_clean_p, ".wav", "skipped_clean")
        add_row("wavmark-marked", "audio_invisible", "wavmark", wm_marked_p, ".wav", "verified_removed")

        wm_clean_r = _synthetic_speechlike_carrier(16000, 3.0, seed=202)
        wm_marked_r = np.asarray(wavmark.embed(wm_model, wm_clean_r), dtype=np.float32)
        wm_stereo_p = fixtures_dir / "wavmark-stereo-marked.wav"
        sf.write(wm_stereo_p, np.column_stack((wm_marked, wm_marked_r)), 16000, subtype="FLOAT")
        add_row("wavmark-stereo-ms-marked", "audio_invisible", "wavmark", wm_stereo_p, ".wav", "verified_removed")

    # 3. Perth 16 kHz (both auto and conditioned focalcodec when cached)
    if perth.available():
        p_model = perth.load_model()
        p_clean = _synthetic_speechlike_carrier(16000, 3.0, seed=301)
        p_marked = np.asarray(perth.embed(p_model, p_clean), dtype=np.float32)
        p_clean_p = fixtures_dir / "perth-clean.wav"
        p_marked_p = fixtures_dir / "perth-marked.wav"
        sf.write(p_clean_p, p_clean, 16000, subtype="FLOAT")
        sf.write(p_marked_p, p_marked, 16000, subtype="FLOAT")
        add_row("perth-clean", "audio_invisible", "perth", p_clean_p, ".wav", "skipped_clean")
        add_row("perth-marked-auto", "audio_invisible", "perth", p_marked_p, ".wav", "verified_removed")
        if audio_focalcodec.is_available():
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    repo_id=audio_focalcodec.MODEL_REPOSITORY,
                    filename=audio_focalcodec.CHECKPOINT_FILENAME,
                    revision=audio_focalcodec.MODEL_REVISION,
                    local_files_only=True,
                )
                add_row(
                    "perth-marked-focalcodec",
                    "audio_invisible",
                    "perth",
                    p_marked_p,
                    ".wav",
                    "verified_removed",
                    method="focalcodec",
                )
            except Exception:
                log.info("Skipping optional focalcodec builtin case because weights are not cached locally")
        if audio_dac.is_available():
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    repo_id=audio_dac.MODEL_REPOSITORY,
                    filename=audio_dac.CHECKPOINT_FILENAME,
                    revision=audio_dac.MODEL_REVISION,
                    local_files_only=True,
                )
                add_row(
                    "perth-marked-dac",
                    "audio_invisible",
                    "perth",
                    p_marked_p,
                    ".wav",
                    "verified_removed",
                    method="dac",
                )
                p_long_clean = _synthetic_speechlike_carrier(16000, 31.0, seed=303)
                p_long_marked = np.asarray(perth.embed(p_model, p_long_clean), dtype=np.float32)
                p_long_p = fixtures_dir / "perth-long31s-marked.wav"
                sf.write(p_long_p, p_long_marked, 16000, subtype="FLOAT")
                add_row(
                    "perth-long31s-windowed-dac",
                    "audio_invisible",
                    "perth",
                    p_long_p,
                    ".wav",
                    "verified_removed",
                    method="dac",
                )
                if gen is not None:
                    from remove_ai_watermarks._internal import audio_resampling

                    as33_l = np.asarray(
                        audioseal.embed(
                            gen,
                            _synthetic_speechlike_carrier(16000, 33.0, seed=311),
                            [int(b) for b in "1010110011110001"],
                        ),
                        dtype=np.float32,
                    )
                    as33_r = np.asarray(
                        audioseal.embed(
                            gen,
                            _synthetic_speechlike_carrier(16000, 33.0, seed=312),
                            [int(b) for b in "0101001100001110"],
                        ),
                        dtype=np.float32,
                    )
                    as33_16k = np.column_stack((as33_l, as33_r))
                    clock_44k = audio_resampling.RationalClock(len(as33_16k), 16000, 44100)
                    as33_44k = np.asarray(audio_resampling.forward(as33_16k, clock_44k), dtype=np.float32)
                    as33_p = fixtures_dir / "audioseal-long33s-stereo44k-marked.wav"
                    sf.write(as33_p, as33_44k, 44100, subtype="FLOAT")
                    add_row(
                        "audioseal-long33s-stereo44k-windowed-dac",
                        "audio_invisible",
                        "audioseal",
                        as33_p,
                        ".wav",
                        "verified_removed",
                        method="dac",
                    )
            except Exception:
                log.info("Skipping optional dac builtin cases because weights are not cached locally")

    # 4. VideoSeal 36-frame 512x288 @ 12 fps MP4
    ref_fixture = REPO / "data/fixtures/visible/veo/provider-original.mp4"
    if videoseal_oracle.import_failure() is None and ref_fixture.is_file():
        vs_model = videoseal_oracle.load_model()
        msg_bits_256 = list(videoseal_oracle.message_bits(7))
        ref_hex = "".join(f"{sum(msg_bits_256[idx + k] << (7 - k) for k in range(8)):02x}" for idx in range(0, 256, 8))
        cap = open_video_capture(ref_fixture)
        frames_bgr: list[Any] = []
        try:
            while len(frames_bgr) < 36:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames_bgr.append(cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA))
        finally:
            cap.release()
        if len(frames_bgr) == 36:
            src_prof = probe_video_encode_profile(ref_fixture)

            def _write_mp4(bgr_seq: list[Any], dst: Path) -> None:
                proc = start_raw_video_encoder(
                    raw_video_command(dst, width=512, height=288, fps=12.0, crf=12, profile=src_prof)
                )
                for fr in bgr_seq:
                    proc.stdin.write(np.ascontiguousarray(fr).tobytes())
                finish_raw_video_encoder(proc, dst, operation="benchmark fixture encode")

            vs_clean_p = fixtures_dir / "videoseal-clean.mp4"
            vs_marked_p = fixtures_dir / "videoseal-marked.mp4"
            _write_mp4(frames_bgr, vs_clean_p)
            rgb_f32 = np.stack([f[:, :, ::-1] for f in frames_bgr]).astype(np.float32) / 255.0
            marked_rgb = np.clip(np.asarray(videoseal_oracle.embed(vs_model, rgb_f32, msg_bits_256)), 0.0, 1.0)
            marked_bgr = [np.ascontiguousarray(np.round(fr * 255.0).astype(np.uint8)[:, :, ::-1]) for fr in marked_rgb]
            _write_mp4(marked_bgr, vs_marked_p)
            dev = _auto_video_device()
            add_row(
                "videoseal-clean",
                "video_message",
                "videoseal",
                vs_clean_p,
                ".mp4",
                "skipped_clean",
                reference_hex=ref_hex,
                long_side=512,
                device=dev,
            )
            add_row(
                "videoseal-marked-auto",
                "video_message",
                "videoseal",
                vs_marked_p,
                ".mp4",
                "verified_removed",
                reference_hex=ref_hex,
                long_side=512,
                device=dev,
            )

            # 5. Unified 4-layer audiovisual clip (AI metadata + visible Kling + VideoSeal pixels + AudioSeal audio)
            kling_fixture = REPO / "data/fixtures/visible/kling/provider-original.mp4"
            if gen is not None and kling_fixture.is_file():
                k_cap = open_video_capture(kling_fixture)
                k_frames: list[Any] = []
                try:
                    while len(k_frames) < 24:
                        ok, frame = k_cap.read()
                        if not ok or frame is None:
                            break
                        k_frames.append(frame)
                finally:
                    k_cap.release()
                if len(k_frames) == 24:
                    kh, kw = k_frames[0].shape[:2]
                    k_prof = probe_video_encode_profile(kling_fixture)
                    k_rgb = np.stack([f[:, :, ::-1] for f in k_frames]).astype(np.float32) / 255.0
                    k_marked_rgb = np.clip(np.asarray(videoseal_oracle.embed(vs_model, k_rgb, msg_bits_256)), 0.0, 1.0)
                    k_marked_bgr = [
                        np.ascontiguousarray(np.round(fr * 255.0).astype(np.uint8)[:, :, ::-1]) for fr in k_marked_rgb
                    ]
                    raw_4l = fixtures_dir / "four-layer-video-only.mp4"
                    proc_4l = start_raw_video_encoder(
                        raw_video_command(raw_4l, width=kw, height=kh, fps=24.0, crf=12, profile=k_prof)
                    )
                    for fr in k_marked_bgr:
                        proc_4l.stdin.write(fr.tobytes())
                    finish_raw_video_encoder(proc_4l, raw_4l, operation="4-layer benchmark video encode")

                    aud_2s = _synthetic_speechlike_carrier(16000, 2.0, seed=401)
                    aud_2s_marked = np.asarray(
                        audioseal.embed(gen, aud_2s, [int(b) for b in "1011001111000101"]),
                        dtype=np.float32,
                    )
                    aud_4l_wav = fixtures_dir / "four-layer-audio.wav"
                    sf.write(aud_4l_wav, aud_2s_marked, 16000, subtype="FLOAT")

                    four_layer_mp4 = fixtures_dir / "video-all-4layer-marked.mp4"
                    aigc_decl = '{"Label":"1","ContentProducer":"kling"}'
                    subprocess.run(  # noqa: S603
                        [
                            require_video_tool("ffmpeg"),
                            "-v",
                            "error",
                            "-nostdin",
                            "-y",
                            "-i",
                            str(raw_4l),
                            "-i",
                            str(aud_4l_wav),
                            "-map",
                            "0:v:0",
                            "-map",
                            "1:a:0",
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-b:a",
                            "192k",
                            "-shortest",
                            "-movflags",
                            "use_metadata_tags",
                            "-metadata",
                            f"AIGC={aigc_decl}",
                            str(four_layer_mp4),
                        ],
                        check=True,
                    )
                    add_row(
                        "video-all-4layer-marked",
                        "video_all_4layer",
                        "auto",
                        four_layer_mp4,
                        ".mp4",
                        "verified_removed",
                        reference_hex=ref_hex,
                        device=dev,
                    )

    if strict:
        emitted_ids = {str(r["case_id"]) for r in rows}
        missing = [cid for cid in STRICT_BUILTIN_REQUIRED_CASE_IDS if cid not in emitted_ids]
        if missing:
            raise RuntimeError(f"--builtin-suite --strict missing required case(s): {', '.join(missing)}")

    manifest_path = work_dir / "manifest.jsonl"
    manifest_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    return manifest_path


def _auto_video_device() -> str:
    try:
        from remove_ai_watermarks.video_invisible import pick_device

        return pick_device("auto")
    except Exception:
        return "cpu"


def _output_sha256(res: Any) -> str | None:
    after_sha = getattr(getattr(res, "after", None), "sha256", None)
    if isinstance(after_sha, str) and after_sha:
        return after_sha
    out = getattr(res, "output", None)
    return sha256_file(out) if out and Path(out).is_file() else None


def _try_perceptual_metrics(
    source_path: Path,
    out_path: Path,
    *,
    stream: int = 0,
    skipped: bool = False,
    source_info: object = None,
    delivered_info: object = None,
) -> dict[str, float] | None:
    if skipped or not out_path.is_file():
        return None
    try:
        src_info = (
            source_info
            if isinstance(source_info, audio_io.AudioStreamInfo)
            else audio_io.probe_audio(source_path, stream=stream)
        )
        deliv_info = delivered_info if isinstance(delivered_info, audio_io.AudioStreamInfo) else None
        return asdict(
            compute_perceptual_metrics(
                source_path,
                src_info,
                out_path,
                stream=stream,
                delivered_info=deliv_info,
            )
        )
    except Exception:
        return None


def evaluate_removal_case(
    case: RemovalBenchmarkCase,
    output_dir: Path,
    *,
    preverified_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify source digest, execute the public removal API, and score outcome & quality."""
    if preverified_sha256 is not None:
        actual_sha = preverified_sha256
    else:
        if not case.source_path.is_file():
            raise FileNotFoundError(f"Case {case.case_id}: missing source {case.source_path}")
        actual_sha = sha256_file(case.source_path)
        if actual_sha != case.source_sha256:
            raise ValueError(f"Case {case.case_id}: source digest mismatch ({actual_sha} != {case.source_sha256})")

    out_path = output_dir / f"{case.case_id}-out{case.output_suffix}"
    out_path.unlink(missing_ok=True)
    started = time.monotonic()
    base: dict[str, Any] = {
        "case_id": case.case_id,
        "surface": case.surface,
        "scheme": case.scheme,
        "expected_outcome": case.expected_outcome,
        "source_sha256": actual_sha,
    }

    try:
        if case.surface == "audio_invisible":
            res_a = audio.remove_audio_invisible(
                case.source_path,
                out_path,
                stream=case.stream,
                method=case.method,  # type: ignore[arg-type]
                scheme=case.scheme,  # type: ignore[arg-type]
            )
            elapsed = time.monotonic() - started
            passed = _check_outcome(
                case.expected_outcome,
                skipped=res_a.skipped,
                verified_removed=res_a.verified_removed,
                quality_passed=res_a.quality_passed,
                before_status=res_a.before.watermark_status,
            )
            return {
                **base,
                "status": "passed" if passed else "failed",
                "output_sha256": _output_sha256(res_a),
                "selected_method": res_a.method,
                "before_status": res_a.before.watermark_status,
                "after_status": res_a.after.watermark_status if res_a.after else None,
                "skipped": res_a.skipped,
                "verified_removed": res_a.verified_removed,
                "quality_passed": res_a.quality_passed,
                "fidelity": asdict(res_a.fidelity) if res_a.fidelity else None,
                "perceptual": _try_perceptual_metrics(
                    case.source_path,
                    out_path,
                    stream=case.stream,
                    skipped=res_a.skipped,
                    source_info=getattr(res_a.before, "stream", None),
                    delivered_info=getattr(res_a.after, "stream", None) if res_a.after else None,
                ),
                "elapsed_seconds": round(elapsed, 3),
            }

        if case.surface == "video_audio":
            res_va = video_audio.remove_video_audio(
                case.source_path,
                out_path,
                stream=case.stream,
                method=case.method,  # type: ignore[arg-type]
                scheme=case.scheme,  # type: ignore[arg-type]
            )
            elapsed = time.monotonic() - started
            passed = _check_outcome(
                case.expected_outcome,
                skipped=res_va.skipped,
                verified_removed=res_va.verified_removed,
                quality_passed=res_va.quality_passed,
                before_status=res_va.before.watermark_status,
            )
            return {
                **base,
                "status": "passed" if passed else "failed",
                "output_sha256": _output_sha256(res_va),
                "selected_method": res_va.method,
                "before_status": res_va.before.watermark_status,
                "after_status": res_va.after.watermark_status if res_va.after else None,
                "skipped": res_va.skipped,
                "verified_removed": res_va.verified_removed,
                "quality_passed": res_va.quality_passed,
                "unchanged_streams": list(res_va.unchanged_streams),
                "fidelity": asdict(res_va.fidelity) if res_va.fidelity else None,
                "perceptual": _try_perceptual_metrics(
                    case.source_path,
                    out_path,
                    stream=case.stream,
                    skipped=res_va.skipped,
                    source_info=getattr(res_va.before, "stream", None),
                    delivered_info=getattr(res_va.after, "stream", None) if res_va.after else None,
                ),
                "elapsed_seconds": round(elapsed, 3),
            }

        if case.surface == "video_audio_all_streams":
            res_vam = video_audio.remove_video_audio_all_streams(
                case.source_path,
                out_path,
                method=case.method,  # type: ignore[arg-type]
                scheme=case.scheme,  # type: ignore[arg-type]
            )
            elapsed = time.monotonic() - started
            vam_befores = [getattr(getattr(s, "before", None), "watermark_status", None) for s in res_vam.streams]
            vam_before_status = (
                "not_detected"
                if vam_befores and all(st == "not_detected" for st in vam_befores)
                else ("detected" if "detected" in vam_befores else "unverified")
            )
            passed = _check_outcome(
                case.expected_outcome,
                skipped=res_vam.skipped,
                verified_removed=res_vam.verified_removed,
                quality_passed=res_vam.quality_passed,
                before_status=vam_before_status,
            )
            first_fid = next((s.fidelity for s in res_vam.streams if s.fidelity is not None), None)
            methods = sorted({s.method for i, s in enumerate(res_vam.streams) if i in res_vam.processed_ordinals})
            last_idx = res_vam.processed_ordinals[-1] if res_vam.processed_ordinals else -1
            last_stream = res_vam.streams[last_idx] if 0 <= last_idx < len(res_vam.streams) else res_vam
            first_ord = res_vam.processed_ordinals[0] if res_vam.processed_ordinals else 0
            first_stream = res_vam.streams[first_ord] if 0 <= first_ord < len(res_vam.streams) else None
            return {
                **base,
                "status": "passed" if passed else "failed",
                "output_sha256": _output_sha256(last_stream),
                "selected_method": "+".join(methods) if methods else "-",
                "before_status": vam_before_status,
                "processed_ordinals": list(res_vam.processed_ordinals),
                "unchanged_streams": list(res_vam.unchanged_streams),
                "skipped": res_vam.skipped,
                "verified_removed": res_vam.verified_removed,
                "quality_passed": res_vam.quality_passed,
                "fidelity": asdict(first_fid) if first_fid else None,
                "perceptual": _try_perceptual_metrics(
                    case.source_path,
                    out_path,
                    stream=first_ord,
                    skipped=res_vam.skipped,
                    source_info=getattr(getattr(first_stream, "before", None), "stream", None),
                    delivered_info=getattr(getattr(first_stream, "after", None), "stream", None),
                ),
                "elapsed_seconds": round(elapsed, 3),
            }

        if case.surface == "video_all_4layer":
            if not case.reference_hex:
                raise ValueError(f"Case {case.case_id}: video_all_4layer requires reference_hex")
            msg_before = video_messages.identify_video_message(
                case.source_path, reference_hex=case.reference_hex, scheme="videoseal"
            )
            res_4l = video.remove_video_all(
                case.source_path,
                out_path,
                include_invisible=True,
                invisible_profile="auto",
                long_side=case.long_side,
                device=case.device,
                temporal_stabilize=True,
                include_audio_invisible=True,
                audio_scheme=case.scheme,  # type: ignore[arg-type]
                audio_method=case.method,  # type: ignore[arg-type]
            )
            elapsed = time.monotonic() - started
            vid_after = video.identify_video(out_path)
            msg_after = video_messages.identify_video_message(
                out_path, reference_hex=case.reference_hex, scheme="videoseal"
            )
            audio_after = audio.identify_audio(out_path, scheme=case.scheme)  # type: ignore[arg-type]
            audio_after_status = audio_after.watermark_status
            layers_ok = (
                msg_before.status == "matched"
                and bool(res_4l.detected_metadata)
                and not res_4l.remaining_metadata
                and res_4l.visible_mark is not None
                and res_4l.visible_removed_frames > 0
                and vid_after.visible_mark is None
                and not vid_after.has_ai_metadata
                and msg_after.status == "not_matched"
                and res_4l.audio.stream_action == "processed"
                and res_4l.audio.watermark_status == "not_detected"
                and audio_after_status == "not_detected"
            )
            q_passed = getattr(res_4l, "audio_quality_passed", None) is True
            passed = case.expected_outcome == "verified_removed" and layers_ok and q_passed
            visual_label = getattr(res_4l, "invisible_profile", None) or "vae"
            return {
                **base,
                "status": "passed" if passed else "failed",
                "output_sha256": getattr(msg_after, "sha256", None) or _output_sha256(res_4l),
                "selected_method": f"4-layer(meta+vis+{visual_label}+audio)",
                "visible_mark_removed": res_4l.visible_mark,
                "metadata_cleared": not res_4l.remaining_metadata,
                "pixel_message_before": msg_before.status,
                "pixel_message_after": msg_after.status,
                "audio_after": audio_after_status,
                "skipped": False,
                "verified_removed": layers_ok,
                "quality_passed": q_passed,
                "elapsed_seconds": round(elapsed, 3),
            }

        if case.surface == "video_message":
            if not case.reference_hex:
                raise ValueError(f"Case {case.case_id}: video_message requires reference_hex")
            res_vm = video_messages.remove_video_message(
                case.source_path,
                out_path,
                reference_hex=case.reference_hex,
                scheme=case.scheme,  # type: ignore[arg-type]
                method=case.method,  # type: ignore[arg-type]
                long_side=case.long_side,
                device=case.device,  # type: ignore[arg-type]
            )
            elapsed = time.monotonic() - started
            passed = _check_outcome(
                case.expected_outcome,
                skipped=res_vm.skipped,
                verified_removed=res_vm.verified_removed,
                quality_passed=res_vm.quality_passed,
                before_status=res_vm.before.status,
            )
            return {
                **base,
                "status": "passed" if passed else "failed",
                "output_sha256": _output_sha256(res_vm),
                "selected_method": res_vm.method,
                "noise_std": res_vm.noise_std,
                "before_status": res_vm.before.status,
                "after_status": res_vm.after.status if res_vm.after else None,
                "skipped": res_vm.skipped,
                "verified_removed": res_vm.verified_removed,
                "quality_passed": res_vm.quality_passed,
                "quality": dict(res_vm.quality) if res_vm.quality else None,
                "elapsed_seconds": round(elapsed, 3),
            }
    except ImportError as exc:
        return {
            **base,
            "status": "unavailable",
            "reason": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    except (audio.AudioVerificationIncomplete, video_messages.VideoMessageVerificationIncomplete) as exc:
        return {
            **base,
            "status": "failed",
            "reason": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    raise ValueError(f"Unsupported benchmark surface: {case.surface}")


def _check_outcome(
    expected: ExpectedOutcome,
    *,
    skipped: bool,
    verified_removed: bool | None,
    quality_passed: bool | None,
    before_status: str | None = None,
) -> bool:
    if expected == "skipped_clean":
        return (
            skipped is True
            and verified_removed is None
            and (before_status is None or before_status in {"not_detected", "not_matched"})
        )
    if expected == "verified_removed":
        return skipped is False and verified_removed is True and quality_passed is True
    return False


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Render a compact Markdown table summarizing all benchmark cases."""
    lines = [
        "# Audio and Video Removal Benchmark Summary",
        "",
        f"- Manifest SHA-256: `{report['manifest_sha256']}`",
        f"- Total cases: `{report['counts']['total']}` (`passed={report['counts']['passed']}`, "
        f"`failed={report['counts']['failed']}`, `unavailable={report['counts']['unavailable']}`)",
        "",
        "| Case | Surface | Scheme | Expected | Status | Method | Quality metric | Elapsed (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["results"]:
        metric = "-"
        if row.get("fidelity"):
            fid = row["fidelity"]
            snr = f"{fid['snr_db']:.2f} dB" if fid.get("snr_db") is not None else "n/a"
            rms_err = f"{fid['rms_error']:.4f}" if fid.get("rms_error") is not None else "n/a"
            metric = f"SNR {snr}, RMS err {rms_err}"
            perc = row.get("perceptual")
            if isinstance(perc, dict) and perc.get("mr_stft_loss") is not None:
                metric += f", MR-STFT {perc['mr_stft_loss']:.3f}, Mel {perc['mel_distance_db']:.2f} dB"
        elif row.get("quality"):
            q = row["quality"]
            psnr = f"{q['source_psnr_db']:.2f} dB" if q.get("source_psnr_db") is not None else "lossless"
            ssim = f"{q['source_ssim']:.4f}" if q.get("source_ssim") is not None else "n/a"
            metric = f"PSNR {psnr}, SSIM {ssim}"
        lines.append(
            f"| `{row['case_id']}` | `{row['surface']}` | `{row['scheme']}` | "
            f"`{row['expected_outcome']}` | `{row['status']}` | "
            f"`{row.get('selected_method') or '-'}` | {metric} | {row['elapsed_seconds']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def run_removal_benchmark(
    manifest_path: Path,
    *,
    work_dir: Path,
    output_json: Path,
    output_markdown: Path | None = None,
) -> dict[str, Any]:
    """Validate all input digests up front, run removal cases, and save reports."""
    _ensure_new_file(output_json)
    if output_markdown is not None:
        _ensure_new_file(output_markdown)

    cases = load_manifest(manifest_path)
    if not cases:
        raise ValueError(f"Benchmark manifest {manifest_path} contains zero cases")

    # Pre-verify all source hashes before running any removal model
    checked_digests: dict[Path, str] = {}
    for case in cases:
        if not case.source_path.is_file():
            raise FileNotFoundError(f"Case {case.case_id}: missing source {case.source_path}")
        actual = checked_digests.setdefault(case.source_path, sha256_file(case.source_path))
        if actual != case.source_sha256:
            raise ValueError(f"Case {case.case_id}: source SHA-256 mismatch before benchmark start")

    outputs_dir = work_dir / "delivered"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results = [evaluate_removal_case(case, outputs_dir, preverified_sha256=case.source_sha256) for case in cases]

    counts = {
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "passed"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "unavailable": sum(1 for r in results if r["status"] == "unavailable"),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "counts": counts,
        "results": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_markdown is not None:
        output_markdown.parent.mkdir(parents=True, exist_ok=True)
        output_markdown.write_text(format_markdown_summary(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="JSONL manifest of removal benchmark cases")
    group.add_argument(
        "--builtin-suite",
        action="store_true",
        help="synthesize deterministic local suite in --work-dir",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="working directory for synthesized/delivered files",
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSON report path (never overwritten)")
    parser.add_argument("--markdown", type=Path, default=None, help="optional new Markdown summary path")
    parser.add_argument("--strict", action="store_true", help="exit nonzero if any case fails or is unavailable")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_builtin_suite(args.work_dir, strict=args.strict) if args.builtin_suite else args.manifest
    if manifest is None:
        raise ValueError("A benchmark manifest path is required")

    report = run_removal_benchmark(
        manifest,
        work_dir=args.work_dir,
        output_json=args.output,
        output_markdown=args.markdown,
    )
    log.info(
        "Removal benchmark complete: total=%s passed=%s failed=%s unavailable=%s -> %s",
        report["counts"]["total"],
        report["counts"]["passed"],
        report["counts"]["failed"],
        report["counts"]["unavailable"],
        args.output,
    )
    if report["counts"]["failed"] > 0 or (args.strict and report["counts"]["unavailable"] > 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
