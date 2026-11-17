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
                    fidelity=AudioFidelity(
                        frames=16000,
                        rms_error=0.002,
                        max_abs_error=0.02,
                        snr_db=31.0,
                        peak_amplitude=0.4,
                        samples_above_full_scale=0,
                        source_rms=0.2,
                        output_rms=0.2,
                    ),
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
