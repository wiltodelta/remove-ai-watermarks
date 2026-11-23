"""Unit tests for the end-to-end audio and video removal benchmark script."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from remove_ai_watermarks._internal.audio_quality import AudioFidelity, sha256_file
from scripts import media_removal_benchmark as bench

if TYPE_CHECKING:
    from pathlib import Path


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_load_manifest_validates_schema_duplicates_and_relative_paths(tmp_path: Path) -> None:
    src = tmp_path / "clip.wav"
    src.write_bytes(b"RIFFfake")
    digest = sha256_file(src)
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {
                "schema_version": 1,
                "case_id": "case-a",
                "surface": "audio_invisible",
                "scheme": "audioseal",
                "source_path": "clip.wav",
                "source_sha256": digest,
                "output_suffix": ".wav",
                "expected_outcome": "skipped_clean",
            }
        ],
    )
    cases = bench.load_manifest(manifest)
    assert len(cases) == 1
    assert cases[0].source_path == src.resolve()
    assert cases[0].long_side == 512

    dup_manifest = _write_manifest(
        tmp_path / "dup.jsonl",
        [
            {
                "schema_version": 1,
                "case_id": "dup",
                "surface": "audio_invisible",
                "scheme": "audioseal",
                "source_path": "clip.wav",
                "source_sha256": digest,
                "output_suffix": ".wav",
                "expected_outcome": "skipped_clean",
            },
            {
                "schema_version": 1,
                "case_id": "dup",
                "surface": "audio_invisible",
                "scheme": "audioseal",
                "source_path": "clip.wav",
                "source_sha256": digest,
                "output_suffix": ".wav",
                "expected_outcome": "skipped_clean",
            },
        ],
    )
    with pytest.raises(ValueError, match="duplicate case_id"):
        bench.load_manifest(dup_manifest)

    bad_schema = _write_manifest(
        tmp_path / "bad_schema.jsonl",
        [
            {
                "schema_version": 99,
                "case_id": "bad",
                "surface": "audio_invisible",
                "scheme": "audioseal",
                "source_path": "clip.wav",
                "source_sha256": digest,
                "output_suffix": ".wav",
                "expected_outcome": "skipped_clean",
            }
        ],
    )
    with pytest.raises(ValueError, match=r"Unsupported .* schema"):
        bench.load_manifest(bad_schema)


def test_run_removal_benchmark_refuses_overwrite_and_source_digest_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    src.write_bytes(b"audio-bytes")
    manifest = _write_manifest(
        tmp_path / "manifest.jsonl",
        [
            {
                "schema_version": 1,
                "case_id": "c1",
                "surface": "audio_invisible",
                "scheme": "audioseal",
                "source_path": str(src),
                "source_sha256": "0" * 64,
                "output_suffix": ".wav",
                "expected_outcome": "skipped_clean",
            }
        ],
    )
    existing_out = tmp_path / "existing.json"
    existing_out.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        bench.run_removal_benchmark(manifest, work_dir=tmp_path / "w", output_json=existing_out)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        bench.run_removal_benchmark(manifest, work_dir=tmp_path / "w", output_json=tmp_path / "new.json")


def test_evaluate_removal_case_enforces_quality_and_verified_removal_across_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src_a = tmp_path / "a.wav"
    src_a.write_bytes(b"audio")
    src_v = tmp_path / "v.mp4"
    src_v.write_bytes(b"video")
    sha_a = sha256_file(src_a)
    sha_v = sha256_file(src_v)

    def fake_audio_remove(_src: Path, out: Path, **_kw: object) -> object:
        out.write_bytes(b"cleaned-audio")
        return SimpleNamespace(
            output=out,
            method="spectral",
            before=SimpleNamespace(watermark_status="detected"),
            after=SimpleNamespace(watermark_status="not_detected"),
            skipped=False,
            verified_removed=True,
            quality_passed=False,
            fidelity=AudioFidelity(
                frames=16000,
                rms_error=0.05,
                max_abs_error=0.2,
                snr_db=11.0,
                peak_amplitude=0.5,
                samples_above_full_scale=0,
                source_rms=0.2,
                output_rms=0.04,
            ),
        )

    def fake_video_audio_remove(_src: Path, out: Path, **_kw: object) -> object:
        out.write_bytes(b"cleaned-mux")
        return SimpleNamespace(
            output=out,
            method="combined",
            before=SimpleNamespace(watermark_status="detected"),
            after=SimpleNamespace(watermark_status="not_detected"),
            skipped=False,
            verified_removed=True,
            quality_passed=True,
            unchanged_streams=(0, 1),
            fidelity=AudioFidelity(
                frames=16000,
                rms_error=0.004,
                max_abs_error=0.04,
                snr_db=28.0,
                peak_amplitude=0.45,
                samples_above_full_scale=0,
                source_rms=0.2,
                output_rms=0.198,
            ),
        )

    def fake_video_message_remove(_src: Path, out: Path, **_kw: object) -> object:
        out.write_bytes(b"cleaned-video")
        return SimpleNamespace(
            output=out,
            method="vae",
            noise_std=0.15,
            before=SimpleNamespace(status="matched"),
            after=SimpleNamespace(status="not_matched"),
            skipped=False,
            verified_removed=True,
            quality_passed=True,
            quality={"source_psnr_db": 31.4, "source_ssim": 0.9612},
        )

    monkeypatch.setattr(bench.audio, "remove_audio_invisible", fake_audio_remove)
    monkeypatch.setattr(bench.video_audio, "remove_video_audio", fake_video_audio_remove)
    monkeypatch.setattr(bench.video_messages, "remove_video_message", fake_video_message_remove)

    manifest = _write_manifest(
        tmp_path / "cases.jsonl",
        [
            {
                "schema_version": 1,
                "case_id": "audio-low-quality",
                "surface": "audio_invisible",
                "scheme": "audioseal",
                "source_path": str(src_a),
                "source_sha256": sha_a,
                "output_suffix": ".wav",
                "expected_outcome": "verified_removed",
            },
            {
                "schema_version": 1,
                "case_id": "mux-pass",
                "surface": "video_audio",
                "scheme": "audioseal",
                "source_path": str(src_v),
                "source_sha256": sha_v,
                "output_suffix": ".mp4",
                "expected_outcome": "verified_removed",
                "stream": 1,
            },
            {
                "schema_version": 1,
                "case_id": "videoseal-pass",
                "surface": "video_message",
                "scheme": "videoseal",
                "source_path": str(src_v),
                "source_sha256": sha_v,
                "output_suffix": ".mp4",
                "expected_outcome": "verified_removed",
                "reference_hex": "ab" * 32,
            },
        ],
    )
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    report = bench.run_removal_benchmark(
        manifest,
        work_dir=tmp_path / "work",
        output_json=out_json,
        output_markdown=out_md,
    )
    assert report["counts"] == {"total": 3, "passed": 2, "failed": 1, "unavailable": 0}
    by_id = {row["case_id"]: row for row in report["results"]}
    assert by_id["audio-low-quality"]["status"] == "failed"
    assert by_id["mux-pass"]["status"] == "passed"
    assert by_id["videoseal-pass"]["status"] == "passed"
    md_text = out_md.read_text(encoding="utf-8")
    assert "`audio-low-quality`" in md_text
    assert "`videoseal-pass`" in md_text


def test_evaluate_multistream_and_four_layer_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src_v = tmp_path / "four.mp4"
    src_v.write_bytes(b"video-4l")
    sha_v = sha256_file(src_v)

    def fake_multistream(_src: Path, out: Path, **_kw: object) -> object:
        out.write_bytes(b"multi-cleaned")
        fid = AudioFidelity(
            frames=16000,
            rms_error=0.002,
            max_abs_error=0.02,
            snr_db=31.0,
            peak_amplitude=0.4,
            samples_above_full_scale=0,
            source_rms=0.2,
            output_rms=0.2,
        )
        return SimpleNamespace(
            output=out,
            skipped=False,
            verified_removed=True,
            quality_passed=True,
            processed_ordinals=(0, 1),
            unchanged_streams=(0,),
            streams=(
                SimpleNamespace(
                    method="spectral",
                    quality_passed=True,
                    before=SimpleNamespace(watermark_status="detected"),
                    fidelity=fid,
                ),
                SimpleNamespace(
                    method="encodec",
                    quality_passed=True,
                    before=SimpleNamespace(watermark_status="detected"),
                    fidelity=fid,
                ),
            ),
        )

    captured_video_all_kwargs: dict[str, object] = {}

    def fake_video_all(_src: Path, out: Path, **kw: object) -> object:
        captured_video_all_kwargs.update(kw)
        out.write_bytes(b"four-layer-cleaned")
        return SimpleNamespace(
            output=out,
            invisible_profile="synthid",
            detected_metadata={"audio_creation_tool": "Suno"},
            remaining_metadata={},
            visible_mark="kling",
            visible_removed_frames=24,
            audio=SimpleNamespace(stream_action="processed", watermark_status="not_detected"),
            audio_quality_passed=True,
        )

    monkeypatch.setattr(bench.video_audio, "remove_video_audio_all_streams", fake_multistream)
    monkeypatch.setattr(bench.video, "remove_video_all", fake_video_all)
    monkeypatch.setattr(
        bench.video,
        "identify_video",
        lambda _p: SimpleNamespace(visible_mark=None, has_ai_metadata=False),
    )
    monkeypatch.setattr(
        bench.video_messages,
        "identify_video_message",
        lambda p, **_kw: SimpleNamespace(status="matched" if str(p) == str(src_v) else "not_matched"),
    )
    monkeypatch.setattr(
        bench.audio,
        "identify_audio",
        lambda _p, **_kw: SimpleNamespace(watermark_status="not_detected"),
    )
    monkeypatch.setattr(bench.video_quality, "delivered_geometry", lambda _p: (64, 48, 12.0))
    monkeypatch.setattr(
        bench.video_quality,
        "measure_video_fidelity",
        lambda *_a, **_kw: {
            "source_psnr_db": 31.0,
            "source_ssim": 0.96,
            "source_color_ssim": 0.95,
            "min_frame_psnr_db": 29.0,
            "min_frame_ssim": 0.93,
            "worst_window_ssim": 0.95,
            "temporal_residual_ratio": 1.02,
        },
    )

    # Clean-control predicate must reject unverified before_status
    assert (
        bench._check_outcome(
            "skipped_clean",
            skipped=True,
            verified_removed=None,
            quality_passed=None,
            before_status="unverified",
        )
        is False
    )
    assert (
        bench._check_outcome(
            "skipped_clean",
            skipped=True,
            verified_removed=None,
            quality_passed=None,
            before_status="not_detected",
        )
        is True
    )

    manifest = _write_manifest(
        tmp_path / "cases_v2.jsonl",
        [
            {
                "schema_version": 1,
                "case_id": "multi-audio",
                "surface": "video_audio_all_streams",
                "scheme": "auto",
                "source_path": str(src_v),
                "source_sha256": sha_v,
                "output_suffix": ".mp4",
                "expected_outcome": "verified_removed",
            },
            {
                "schema_version": 1,
                "case_id": "four-layer",
                "surface": "video_all_4layer",
                "scheme": "auto",
                "source_path": str(src_v),
                "source_sha256": sha_v,
                "output_suffix": ".mp4",
                "expected_outcome": "verified_removed",
                "reference_hex": "ab" * 32,
            },
        ],
    )
    report = bench.run_removal_benchmark(
        manifest,
        work_dir=tmp_path / "work2",
        output_json=tmp_path / "report2.json",
    )
    assert report["counts"] == {"total": 2, "passed": 2, "failed": 0, "unavailable": 0}
    assert captured_video_all_kwargs["invisible_profile"] == "auto"


def test_load_manifest_accepts_dac_method(tmp_path: Path) -> None:
    src = tmp_path / "clip.wav"
    src.write_bytes(b"RIFFdac")
    digest = sha256_file(src)
    manifest = _write_manifest(
        tmp_path / "dac_manifest.jsonl",
        [
            {
                "schema_version": 1,
                "case_id": "perth-dac",
                "surface": "audio_invisible",
                "scheme": "perth",
                "method": "dac",
                "source_path": "clip.wav",
                "source_sha256": digest,
                "output_suffix": ".wav",
                "expected_outcome": "verified_removed",
            }
        ],
    )
    cases = bench.load_manifest(manifest)
    assert cases[0].method == "dac"

    bad_method = _write_manifest(
        tmp_path / "bad_method.jsonl",
        [
            {
                "schema_version": 1,
                "case_id": "bad-method",
                "surface": "audio_invisible",
                "scheme": "perth",
                "method": "unknown_codec",
                "source_path": "clip.wav",
                "source_sha256": digest,
                "output_suffix": ".wav",
                "expected_outcome": "verified_removed",
            }
        ],
    )
    with pytest.raises(ValueError, match="unsupported method"):
        bench.load_manifest(bad_method)


def test_perceptual_audio_metrics_phase_invariance_and_markdown_rendering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil
    import wave

    np = pytest.importorskip("numpy")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")

    from remove_ai_watermarks._internal import audio_io, audio_quality

    sr = 24000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    carrier = (
        0.30 * np.sin(2 * np.pi * 220.0 * t)
        + 0.18 * np.sin(2 * np.pi * 660.0 * t)
        + 0.10 * np.sin(2 * np.pi * 1980.0 * t)
    ).astype(np.float32)

    def _write_pcm16_wav(path: Path, mono: object) -> Path:
        pcm = np.clip(np.rint(np.asarray(mono) * 32767.0), -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        return path

    src_wav = _write_pcm16_wav(tmp_path / "src.wav", carrier)
    inv_wav = _write_pcm16_wav(tmp_path / "inverted.wav", -carrier)
    rng = np.random.default_rng(7)
    noisy_wav = _write_pcm16_wav(tmp_path / "noisy.wav", carrier * 0.4 + rng.normal(0.0, 0.18, size=carrier.shape))

    info = audio_io.probe_audio(src_wav)
    fid_inv = audio_quality.measure_fidelity(src_wav, info, inv_wav)
    assert fid_inv.snr_db is not None
    assert fid_inv.snr_db < -5.0

    perc_same = audio_quality.compute_perceptual_metrics(src_wav, info, src_wav)
    assert perc_same.mr_stft_loss == pytest.approx(0.0, abs=1e-6)
    assert perc_same.mel_distance_db == pytest.approx(0.0, abs=1e-6)

    perc_inv = audio_quality.compute_perceptual_metrics(src_wav, info, inv_wav)
    assert perc_inv.mr_stft_loss < 0.01
    assert perc_inv.mel_distance_db < 0.05

    perc_noisy = audio_quality.compute_perceptual_metrics(src_wav, info, noisy_wav)
    assert perc_noisy.mr_stft_loss > 0.4
    assert perc_noisy.mel_distance_db > 4.0

    def fake_audio_remove(_src: Path, out: Path, **_kw: object) -> object:
        shutil.copyfile(inv_wav, out)
        return SimpleNamespace(
            output=out,
            method="dac",
            before=SimpleNamespace(watermark_status="detected"),
            after=SimpleNamespace(watermark_status="not_detected"),
            skipped=False,
            verified_removed=True,
            quality_passed=True,
            fidelity=fid_inv,
        )

    monkeypatch.setattr(bench.audio, "remove_audio_invisible", fake_audio_remove)
    case = bench.RemovalBenchmarkCase(
        case_id="phase-free-vocoder",
        surface="audio_invisible",
        scheme="perth",
        source_path=src_wav,
        source_sha256=sha256_file(src_wav),
        output_suffix=".wav",
        expected_outcome="verified_removed",
        method="dac",
    )
    delivered_dir = tmp_path / "delivered"
    delivered_dir.mkdir()
    row = bench.evaluate_removal_case(case, delivered_dir)
    assert row["status"] == "passed"
    assert row["perceptual"] is not None
    assert row["perceptual"]["mr_stft_loss"] < 0.01
    assert row["perceptual"]["mel_distance_db"] < 0.05

    md = bench.format_markdown_summary(
        {
            "manifest_sha256": "0" * 64,
            "counts": {"total": 1, "passed": 1, "failed": 0, "unavailable": 0},
            "results": [row],
        }
    )
    assert "MR-STFT" in md
    assert "Mel" in md


def test_schema_v2_gates_four_layer_video_quality_multistream_perceptual_and_external_comparator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shutil
    import wave
    from pathlib import Path

    np = pytest.importorskip("numpy")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")

    # 1. skipped_clean must reject before_status=None
    assert (
        bench._check_outcome(
            "skipped_clean",
            skipped=True,
            verified_removed=None,
            quality_passed=None,
            before_status=None,
        )
        is False
    )
    assert (
        bench._check_outcome(
            "control_stays_positive",
            skipped=True,
            verified_removed=None,
            quality_passed=None,
            before_status="detected",
        )
        is True
    )

    # 2. video_all_4layer must fail when visual quality (e.g., color SSIM) collapses despite clean audio
    src_v = tmp_path / "four_bad_color.mp4"
    src_v.write_bytes(b"video-4l")
    sha_v = sha256_file(src_v)

    def fake_video_all(_src: Path, out: Path, **_kw: object) -> object:
        out.write_bytes(b"four-layer-chroma-damaged")
        return SimpleNamespace(
            output=out,
            invisible_profile="synthid",
            detected_metadata={"audio_creation_tool": "Suno"},
            remaining_metadata={},
            visible_mark="kling",
            visible_removed_frames=24,
            audio=SimpleNamespace(stream_action="processed", watermark_status="not_detected"),
            audio_quality_passed=True,
        )

    monkeypatch.setattr(bench.video, "remove_video_all", fake_video_all)
    monkeypatch.setattr(
        bench.video,
        "identify_video",
        lambda _p: SimpleNamespace(visible_mark=None, has_ai_metadata=False),
    )
    monkeypatch.setattr(
        bench.video_messages,
        "identify_video_message",
        lambda p, **_kw: SimpleNamespace(status="matched" if str(p) == str(src_v) else "not_matched"),
    )
    monkeypatch.setattr(
        bench.audio,
        "identify_audio",
        lambda _p, **_kw: SimpleNamespace(watermark_status="not_detected"),
    )
    monkeypatch.setattr(bench.video_quality, "delivered_geometry", lambda _p: (64, 48, 12.0))
    monkeypatch.setattr(
        bench.video_quality,
        "measure_video_fidelity",
        lambda *_a, **_kw: {
            "source_psnr_db": 26.0,
            "source_ssim": 0.92,
            "source_color_ssim": 0.22,
            "min_frame_psnr_db": 24.0,
            "min_frame_ssim": 0.88,
            "worst_window_ssim": 0.90,
            "temporal_residual_ratio": 1.05,
        },
    )

    case_4l = bench.RemovalBenchmarkCase(
        case_id="four-layer-chroma-fail",
        surface="video_all_4layer",
        scheme="auto",
        source_path=src_v,
        source_sha256=sha_v,
        output_suffix=".mp4",
        expected_outcome="verified_removed",
        reference_hex="ab" * 32,
    )
    out_dir = tmp_path / "deliv"
    out_dir.mkdir()
    row_4l = bench.evaluate_removal_case(case_4l, out_dir)
    assert row_4l["status"] == "failed"
    assert row_4l["video_quality"]["source_color_ssim"] == 0.22

    # 3. video_audio_all_streams scores all processed ordinals and fails strict mode when track 1 errors
    def fake_multistream(_src: Path, out: Path, **_kw: object) -> object:
        out.write_bytes(b"multi")
        fid = AudioFidelity(16000, 0.002, 0.02, 30.0, 0.4, 0, 0.2, 0.2)
        return SimpleNamespace(
            output=out,
            skipped=False,
            verified_removed=True,
            quality_passed=True,
            processed_ordinals=(0, 1),
            unchanged_streams=(0,),
            streams=(
                SimpleNamespace(
                    method="spectral",
                    quality_passed=True,
                    fidelity=fid,
                    before=SimpleNamespace(watermark_status="detected", stream=None),
                    after=SimpleNamespace(watermark_status="not_detected", stream=None),
                ),
                SimpleNamespace(
                    method="spectral",
                    quality_passed=True,
                    fidelity=fid,
                    before=SimpleNamespace(watermark_status="detected", stream=None),
                    after=SimpleNamespace(watermark_status="not_detected", stream=None),
                ),
            ),
        )

    monkeypatch.setattr(bench.video_audio, "remove_video_audio_all_streams", fake_multistream)
    monkeypatch.setattr(
        bench,
        "_try_perceptual_metrics",
        lambda _s, _o, *, stream=0, **_k: (
            ({"mr_stft_loss": 0.1, "mel_distance_db": 0.5}, None)
            if stream == 0
            else (None, "corrupt second audio stream")
        ),
    )
    case_ms = bench.RemovalBenchmarkCase(
        case_id="multi-track1-perc-fail",
        surface="video_audio_all_streams",
        scheme="auto",
        source_path=src_v,
        source_sha256=sha_v,
        output_suffix=".mp4",
        expected_outcome="verified_removed",
    )
    row_ms_strict = bench.evaluate_removal_case(case_ms, out_dir, strict=True)
    assert row_ms_strict["status"] == "failed"
    assert "1" in row_ms_strict["perceptual_errors"]

    # 4. External competitor output scoring (--external-outputs-dir) on real WAV
    monkeypatch.undo()
    sr = 16000
    t = np.arange(sr * 2, dtype=np.float32) / sr
    tone = (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    pcm = np.clip(np.rint(tone * 32767.0), -32768, 32767).astype("<i2")
    src_wav = tmp_path / "ext_src.wav"
    ext_dir = tmp_path / "external_tool"
    ext_dir.mkdir()
    ext_wav = ext_dir / "ext-competitor-case.wav"
    for p in (src_wav, ext_wav):
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    monkeypatch.setattr(
        bench.audio,
        "remove_audio_invisible",
        lambda *_a, **_kw: pytest.fail("remove_audio_invisible must not run in external_outputs_dir mode"),
    )
    monkeypatch.setattr(
        bench.audio,
        "identify_audio",
        lambda p, **_kw: SimpleNamespace(
            watermark_status="detected" if Path(p).resolve() == src_wav.resolve() else "not_detected",
            stream=bench.audio_io.probe_audio(p),
        ),
    )
    ext_manifest = _write_manifest(
        tmp_path / "ext_manifest.jsonl",
        [
            {
                "schema_version": 2,
                "case_id": "ext-competitor-case",
                "generation_id": "gen-001",
                "role": "candidate",
                "surface": "audio_invisible",
                "scheme": "audioseal",
                "source_path": str(src_wav),
                "source_sha256": sha256_file(src_wav),
                "output_suffix": ".wav",
                "expected_outcome": "verified_removed",
            }
        ],
    )
    ext_report = bench.run_removal_benchmark(
        ext_manifest,
        work_dir=tmp_path / "ext_work",
        output_json=tmp_path / "ext_report.json",
        external_outputs_dir=ext_dir,
        strict=True,
    )
    assert ext_report["counts"] == {"total": 1, "passed": 1, "failed": 0, "unavailable": 0}
    assert ext_report["results"][0]["selected_method"] == "external"
    assert ext_report["results"][0]["perceptual"]["mr_stft_loss"] == pytest.approx(0.0, abs=1e-5)


def test_benchmark_v3_parity_and_runtime_safety_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pathlib
    import shutil
    import wave

    import numpy as np

    from remove_ai_watermarks import audio, video_invisible, video_quality
    from remove_ai_watermarks._internal.audio_quality import AudioFidelity, audio_quality_passed
    from remove_ai_watermarks.video_encoding import VideoEncodeProfile

    # 1. _check_outcome parity: byte-identical external clean passes skipped_clean;
    #    clean-to-positive false acquisition fails control_stays_positive.
    assert bench._check_outcome(
        "skipped_clean",
        skipped=False,
        verified_removed=None,
        quality_passed=True,
        before_status="not_detected",
        byte_identical=True,
    )
    assert not bench._check_outcome(
        "skipped_clean",
        skipped=False,
        verified_removed=None,
        quality_passed=True,
        before_status="not_detected",
        byte_identical=False,
    )
    assert not bench._check_outcome(
        "control_stays_positive",
        skipped=False,
        verified_removed=False,
        quality_passed=True,
        before_status="not_detected",
    )
    assert bench._check_outcome(
        "control_stays_positive",
        skipped=False,
        verified_removed=False,
        quality_passed=True,
        before_status="detected",
    )

    # 2. _normalize_video_quality sets lossless_pixels=True when source_psnr_db is inf
    norm_vq = bench._normalize_video_quality({"source_psnr_db": float("inf"), "source_ssim": 1.0})
    assert norm_vq["source_psnr_db"] is None
    assert norm_vq["lossless_pixels"] is True
    assert bench.video_messages._quality_passes("codec", norm_vq) is True

    # 3. Runtime audio Side/Mid drift gate + video worst_window_ssim gate
    base_fid = AudioFidelity(
        frames=16000,
        rms_error=0.01,
        max_abs_error=0.05,
        snr_db=25.0,
        peak_amplitude=0.8,
        samples_above_full_scale=0,
        source_rms=0.2,
        output_rms=0.2,
        source_peak=0.8,
        max_side_mid_drift_db=7.5,
    )
    assert audio_quality_passed(base_fid, lossy=False) is False
    assert audio_quality_passed(base_fid, lossy=True) is True

    good_vq = {
        "source_psnr_db": 28.0,
        "source_ssim": 0.90,
        "source_color_ssim": 0.88,
        "min_frame_ssim": 0.72,
        "worst_window_ssim": 0.82,
        "temporal_residual_ratio": 1.1,
    }
    assert video_quality.video_quality_passed(good_vq) is True
    assert video_quality.video_quality_passed({**good_vq, "worst_window_ssim": 0.55}) is False

    # 4. 10-bit / PQ HDR fail-safe guard in video_invisible.regenerate_video_candidate
    hdr_src = tmp_path / "hdr10.mp4"
    hdr_src.write_bytes(b"hdr")
    monkeypatch.setattr(video_invisible, "_probe_video", lambda _p: (1280, 720, 24.0))
    monkeypatch.setattr(
        video_invisible,
        "load_video_vae_runtime",
        lambda **_kw: SimpleNamespace(model="m", requested_device="cpu", resolved_device="cpu", vae=object()),
    )
    monkeypatch.setattr(
        video_invisible,
        "probe_video_encode_profile",
        lambda _p: VideoEncodeProfile(
            pixel_format="yuv420p",
            color_transfer="smpte2084",
            source_pixel_format="yuv420p10le",
            component_depth=10,
        ),
    )
    with pytest.raises(RuntimeError, match="refusing to silently reduce"):
        video_invisible.regenerate_video_candidate(hdr_src, tmp_path / "hdr_out.mp4", device="cpu", model="m")

    # 5. Mastered -0.05 dBFS source with filter overshoot > 1.0 writes FLAC cleanly without ValueError
    monkeypatch.undo()
    sr = 16000
    t = np.arange(sr, dtype=np.float32) / sr
    # Square-ish harmonic wave near 0.985 whose 1.0-1.5 kHz band-stop Gibbs ring overshoots > 1.01
    loud = np.clip(0.95 * np.sin(2 * np.pi * 950.0 * t) + 0.35 * np.sin(2 * np.pi * 1250.0 * t), -0.99, 0.99).astype(
        np.float32
    )
    pcm_loud = np.clip(np.rint(loud * 32767.0), -32768, 32767).astype("<i2")
    loud_wav = tmp_path / "loud_master.wav"
    with wave.open(str(loud_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_loud.tobytes())
    out_flac = tmp_path / "loud_clean.flac"
    res_flac = audio.remove_audio_invisible(
        loud_wav,
        out_flac,
        method="spectral",
        scheme="audioseal",
        force=True,
        allow_unverified=True,
        verify=False,
    )
    assert res_flac.output == out_flac
    assert res_flac.quality_passed is True
    assert res_flac.fidelity is not None
    assert res_flac.fidelity.peak_amplitude <= 1.0

    # 6. Manifest bandwidth_kbps parsing + external video_all_4layer and video_audio_all_streams scoring
    ext_dir2 = tmp_path / "ext_v3"
    ext_dir2.mkdir()
    ext_4l = ext_dir2 / "ext-4l.mp4"
    ext_ms = ext_dir2 / "ext-ms.mp4"
    src_4l = tmp_path / "src_4l.mp4"
    for p in (src_4l, ext_4l, ext_ms):
        shutil.copyfile(loud_wav, p)  # valid 1-stream audio container for probe_audio / measure_fidelity

    monkeypatch.setattr(
        bench.video_audio,
        "stream_inventory",
        lambda _p: [{"index": 0, "codec_type": "audio"}],
    )
    monkeypatch.setattr(
        bench.audio,
        "identify_audio",
        lambda p, **_kw: SimpleNamespace(
            watermark_status="detected" if pathlib.Path(p).resolve() == src_4l.resolve() else "not_detected",
            detections=(),
            stream=bench.audio_io.probe_audio(p),
        ),
    )
    monkeypatch.setattr(
        bench.audio,
        "verification_outcome",
        lambda b, a: b.watermark_status == "detected" and a.watermark_status == "not_detected",
    )
    monkeypatch.setattr(
        bench.video,
        "identify_video",
        lambda p: SimpleNamespace(
            has_ai_metadata=pathlib.Path(p).resolve() == src_4l.resolve(),
            visible_mark="kling" if pathlib.Path(p).resolve() == src_4l.resolve() else None,
        ),
    )
    monkeypatch.setattr(
        bench.video_messages,
        "identify_video_message",
        lambda p, **_kw: SimpleNamespace(
            status="matched" if pathlib.Path(p).resolve() == src_4l.resolve() else "not_matched",
            sha256=sha256_file(p),
        ),
    )
    monkeypatch.setattr(bench.video_quality, "delivered_geometry", lambda _p: (320, 240, 24.0))
    monkeypatch.setattr(bench.video_quality, "measure_video_fidelity", lambda *_a, **_kw: dict(good_vq))

    v3_manifest = _write_manifest(
        tmp_path / "v3_manifest.jsonl",
        [
            {
                "schema_version": 2,
                "case_id": "ext-ms",
                "surface": "video_audio_all_streams",
                "scheme": "audioseal",
                "method": "encodec",
                "bandwidth_kbps": 6,
                "source_path": str(src_4l),
                "source_sha256": sha256_file(src_4l),
                "output_suffix": ".mp4",
                "expected_outcome": "verified_removed",
            },
            {
                "schema_version": 2,
                "case_id": "ext-4l",
                "surface": "video_all_4layer",
                "scheme": "audioseal",
                "reference_hex": "aa" * 32,
                "source_path": str(src_4l),
                "source_sha256": sha256_file(src_4l),
                "output_suffix": ".mp4",
                "expected_outcome": "verified_removed",
            },
        ],
    )
    loaded_cases = bench.load_manifest(v3_manifest)
    assert loaded_cases[0].bandwidth_kbps == 6.0
    v3_report = bench.run_removal_benchmark(
        v3_manifest,
        work_dir=tmp_path / "v3_work",
        output_json=tmp_path / "v3_report.json",
        external_outputs_dir=ext_dir2,
        strict=True,
    )
    assert v3_report["counts"] == {"total": 2, "passed": 2, "failed": 0, "unavailable": 0}
