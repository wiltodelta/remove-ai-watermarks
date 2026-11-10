"""Guard experiment evidence against false removal claims and temporal damage."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import seedance_video_study as study
from click.testing import CliRunner

requires_ffmpeg = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="ffmpeg and ffprobe are required for video integration tests",
)


def captured_receipt(row, verdict):
    """Synthetic transport fixture, never a real provider observation."""
    detected = verdict == "detected"
    filename = row["id"] + ".mp4"
    final = {
        "status": "completed",
        "signal_detected": detected,
        "signals": [],
        "file_info": {"filename": filename, "size_bytes": row["size_bytes"], "media_type": "video"},
        "display": {"state": "detected" if detected else "not_detected"},
    }
    text = json.dumps(final)
    selected = {"filename": filename, "size": row["size_bytes"], "type": "video/mp4", "sha256": row["sha256"]}
    return {
        "filename": filename,
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "media_type": "video",
        "verdict": verdict,
        "http_status": 200,
        "response": final,
        "exchange": {"job_id": "job-1", "final": {"status": 200, "text": text}},
        "capture": {
            "fileUploads": 1,
            "errors": [],
            "requests": [
                {
                    "method": "POST",
                    "url": "/open/v1/detect",
                    "body_kind": "multipart",
                    "selected_file_sha256": row["sha256"],
                    "fields": {"file": selected, "media_type": "video"},
                },
                {"method": "POST", "url": "/open/v1/detect", "body_kind": "job-only-poll", "body_text": "job_id=job-1"},
            ],
            "responses": [
                {"http_status": 202, "response_text": json.dumps({"status": "pending", "job_id": "job-1"})},
                {"http_status": 200, "response_text": text},
            ],
        },
    }


def evidence():
    rows = [
        {"id": name, "sha256": hashlib.sha256(name.encode()).hexdigest(), "size_bytes": 1234}
        for name in ("source", "metadata", "control", "candidate")
    ]
    candidate = rows[-1]
    candidate["control_id"] = "control"
    observations = []
    for artifact_id, verdict in (
        ("control", "detected"),
        ("candidate", "not_detected"),
    ):
        row = next(item for item in rows if item["id"] == artifact_id)
        observations.append(
            {
                "session": "one-pass",
                "surface": "https://ai-generated-detect.capcut.com/",
                "artifact_id": row["id"],
                "receipt": captured_receipt(row, verdict),
            }
        )
    return candidate, rows, observations


def test_complete_receipts_are_scoped_supported_signal_suppression():
    candidate, rows, observations = evidence()
    assert study.suppression_status(candidate, rows, observations) == "observed_suppression"


@pytest.mark.parametrize("missing", range(2))
def test_every_control_and_candidate_is_required(missing):
    candidate, rows, observations = evidence()
    observations.pop(missing)
    assert study.suppression_status(candidate, rows, observations) == "unverified"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("session", "different-pass"),
        ("receipt", None),
        ("receipt", {"verdict": "not_detected", "raw_response": "an unsupported manual label"}),
        ("surface", "https://example.com"),
    ],
)
def test_unrelated_or_uncertain_candidate_evidence_never_passes(field, value):
    candidate, rows, observations = evidence()
    observations[1][field] = value
    assert study.suppression_status(candidate, rows, observations) == "unverified"


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "selected_hash",
        "size",
        "poll_job",
        "reupload",
        "origin",
        "verdict",
        "boolean",
        "file_echo",
        "error",
        "state",
    ],
)
def test_corrupt_receipts_never_pass(fault):
    candidate, rows, observations = evidence()
    receipt = observations[1]["receipt"]
    if fault == "hash":
        receipt["sha256"] = "0" * 64
    elif fault == "selected_hash":
        receipt["capture"]["requests"][0]["selected_file_sha256"] = "0" * 64
    elif fault == "size":
        receipt["capture"]["requests"][0]["fields"]["file"]["size"] += 1
    elif fault == "poll_job":
        receipt["capture"]["requests"][1]["body_text"] = "job_id=another-job"
    elif fault == "reupload":
        receipt["capture"]["requests"][1] = copy.deepcopy(receipt["capture"]["requests"][0])
    elif fault == "origin":
        receipt["capture"]["requests"][0]["url"] = "https://example.com/open/v1/detect"
    elif fault == "verdict":
        receipt["verdict"] = "detected"
    else:
        body = receipt["response"]
        if fault == "boolean":
            body["signal_detected"] = "false"
        elif fault == "file_echo":
            body["file_info"]["filename"] = "somebody-else.mp4"
        elif fault == "error":
            body["error_code"] = "quota"
        elif fault == "state":
            body["display"]["state"] = "detected"
        text = json.dumps(body)
        receipt["exchange"]["final"]["text"] = text
        receipt["capture"]["responses"][-1]["response_text"] = text
    assert study.suppression_status(candidate, rows, observations) == "unverified"


def test_temporal_filter_reduces_independent_noise_without_recursive_feedback():
    rng = np.random.default_rng(41)
    frames = [np.clip(100 + rng.normal(0, 4, (64, 64, 3)), 0, 255).astype(np.uint8) for _ in range(3)]
    before = [frame.copy() for frame in frames]
    output = list(study.temporal_frames(frames))
    assert len(output) == len(frames)
    assert np.std(output[1].astype(float)) < np.std(frames[1].astype(float)) * 0.85
    for source, saved in zip(frames, before, strict=True):
        np.testing.assert_array_equal(source, saved)


def test_scene_cut_does_not_blend_unrelated_frames():
    dark = np.full((32, 32, 3), 30, np.uint8)
    bright = np.full_like(dark, 220)
    np.testing.assert_array_equal(study.blend_neighbors(dark, [bright]), dark)
    np.testing.assert_array_equal(study.blend_neighbors(bright, [dark]), bright)


def test_empty_and_single_frame_are_not_duplicated():
    frame = np.full((32, 32, 3), 50, np.uint8)
    assert list(study.temporal_frames([])) == []
    output = list(study.temporal_frames([frame]))
    assert len(output) == 1
    np.testing.assert_array_equal(output[0], frame)


def test_saved_artifact_change_invalidates_evidence(tmp_path):
    path = tmp_path / "artifact.mp4"
    path.write_bytes(b"original")
    report = {"artifacts": [{"id": "candidate", "path": str(path), "sha256": study.sha256(path)}]}
    path.write_bytes(b"changed")
    with pytest.raises(Exception, match="Artifact changed"):
        study.verify_artifacts(report)


@requires_ffmpeg
def test_local_full_clip_run_and_assessment(tmp_path):
    source = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
    for index in range(6):
        frame = np.full((48, 64, 3), 60, np.uint8)
        frame[10:25, 10 + index : 25 + index] = 170
        writer.write(frame)
    writer.release()
    original = study.sha256(source)
    output = tmp_path / "run"
    runner = CliRunner()
    args = ["run", str(source), "--output-dir", str(output), "--origin", "synthetic", "--long-side", "64", "--skip-vae"]
    args.extend(["--codec-crf", "32"])
    result = runner.invoke(study.main, args)
    assert result.exit_code == 0, result.output + str(result.exception)
    report = json.loads((output / "results.json").read_text())
    assert report["complete"] is True
    assert report["certifies_erasure"] is False
    assert study.sha256(source) == original
    assert {row["id"] for row in report["artifacts"]} == {
        "source",
        "metadata",
        "reencode-64",
        "temporal-64",
        "codec-64-crf32",
    }
    for row in report["artifacts"]:
        assert row["watermark_status"] == "unverified"
        if row["fidelity"]:
            assert row["fidelity"]["frames"] == 6
    artifacts = {row["id"]: row for row in report["artifacts"]}
    assert artifacts["codec-64-crf32"]["sha256"] != artifacts["reencode-64"]["sha256"]
    plan = json.loads((output / "plan.json").read_text())
    for relative, digest in plan["code_sha256"].items():
        assert study.sha256(output / "code" / relative) == digest
    assert runner.invoke(study.main, args).exit_code != 0
    observations = tmp_path / "observations.json"
    observations.write_text("[]")
    assessed = tmp_path / "assessed.json"
    result = runner.invoke(
        study.main, ["assess", str(output / "results.json"), str(observations), "--output", str(assessed)]
    )
    assert result.exit_code == 0, result.output
    assert all(row["watermark_status"] == "unverified" for row in json.loads(assessed.read_text())["artifacts"])
    snapshot = output / "code" / "scripts" / "seedance_video_study.py"
    original_snapshot = snapshot.read_bytes()
    snapshot.write_bytes(b"changed source")
    result = runner.invoke(
        study.main,
        ["assess", str(output / "results.json"), str(observations), "--output", str(tmp_path / "bad-code.json")],
    )
    assert result.exit_code != 0
    assert "Code snapshot changed" in result.output
    snapshot.write_bytes(original_snapshot)
    (output / "plan.json").write_text("{}")
    result = runner.invoke(
        study.main, ["assess", str(output / "results.json"), str(observations), "--output", str(tmp_path / "bad.json")]
    )
    assert result.exit_code != 0
    assert "Plan changed" in result.output


def test_conflicting_detector_responses_do_not_pass():
    candidate, rows, observations = evidence()
    observations.append({**observations[1], "receipt": captured_receipt(candidate, "detected")})
    assert study.suppression_status(candidate, rows, observations) == "unverified"


@requires_ffmpeg
def test_opt_in_lowpass_suppresses_texture_and_preserves_audio_and_timing(tmp_path):
    silent, source = tmp_path / "silent.mp4", tmp_path / "synthetic.mp4"
    yy, xx = np.indices((144, 256))
    texture = np.repeat((((xx // 2 + yy // 2) % 2) * 200 + 25).astype(np.uint8)[..., None], 3, axis=2)
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 24, (256, 144))
    for index in range(6):
        frame = texture.copy()
        frame[112:140, index * 16 : index * 16 + 16] = 100
        writer.write(frame)
    writer.release()
    subprocess.run(  # noqa: S603
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-i",
            str(silent),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=0.25",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    original = study.sha256(source)
    output = tmp_path / "run"
    runner = CliRunner()
    args = [
        "run",
        str(source),
        "--output-dir",
        str(output),
        "--origin",
        "synthetic",
        "--long-side",
        "256",
        "--skip-vae",
        "--include-lowpass64",
        "--preserve-timestamps",
    ]
    result = runner.invoke(study.main, args)
    assert result.exit_code == 0, result.output + str(result.exception)
    report = json.loads((output / "results.json").read_text())
    rows = {row["id"]: row for row in report["artifacts"]}
    control, candidate = rows["reencode-native"], rows["lowpass64-native"]
    assert candidate["control_id"] == control["id"]
    assert study.sha256(source) == original
    for row in [control, candidate]:
        assert row["fidelity"]["frames"] == 6
        assert row["fidelity"]["max_timestamp_error_seconds"] == 0
        assert row["fidelity"]["video_duration_error_seconds"] == 0
        assert study._probe_video(Path(row["path"]))[:2] == (256, 144)

        def audio(path):
            return subprocess.check_output(  # noqa: S603
                [
                    shutil.which("ffmpeg"),
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-c:a",
                    "copy",
                    "-f",
                    "adts",
                    "pipe:1",
                ]
            )

        assert audio(row["path"]) == audio(source)
    first_control = next(study._iter_frames(Path(control["path"])))
    first_candidate = next(study._iter_frames(Path(candidate["path"])))
    assert first_candidate[:100].var() < first_control[:100].var() * 0.1
    plan = json.loads((output / "plan.json").read_text())
    assert plan["include_lowpass64"] is True
    # Synthetic receipts exercise annotation only, not actual removal efficacy.
    observations = tmp_path / "receipts.json"
    observations.write_text(
        json.dumps(
            [
                {
                    "session": "synthetic-pass",
                    "surface": "https://ai-generated-detect.capcut.com/",
                    "artifact_id": row["id"],
                    "receipt": captured_receipt(row, verdict),
                }
                for row, verdict in [(control, "detected"), (candidate, "not_detected")]
            ]
        )
    )
    assessed = tmp_path / "assessed.json"
    result = runner.invoke(
        study.main, ["assess", str(output / "results.json"), str(observations), "--output", str(assessed)]
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    marked = next(row for row in json.loads(assessed.read_text())["artifacts"] if row["id"] == candidate["id"])
    assert marked["capcut_signal_status"] == "observed_suppression"
    assert marked["watermark_status"] == "unverified"


@pytest.mark.parametrize("lowpass", [False, True])
@requires_ffmpeg
def test_native_lowpass_keeps_variable_frame_clock_and_offset(tmp_path, lowpass):
    source, target = tmp_path / "source.mp4", tmp_path / "target.mp4"
    subprocess.run(  # noqa: S603
        [
            shutil.which("ffmpeg"),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=12:duration=1",
            "-vf",
            "settb=1/1000,setpts=N*80+mod(N\\,2)*20+500",
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            "1/1000",
            str(source),
        ],
        check=True,
    )
    study.encode_lowpass64(source, target, lowpass=lowpass)
    assert study.probe_video_timestamps(target) == pytest.approx(study.probe_video_timestamps(source), abs=0.000002)
    metrics = study._measure(
        source, target, source_geometry=study._probe_video(source), duration=None, preserve_timestamps=True
    )
    assert metrics["frames"] == 12
    assert metrics["video_duration_error_seconds"] == 0


@requires_ffmpeg
def test_reconstruction_mixture_preserves_source_detail_and_uses_native_control(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    yy, xx = np.indices((72, 128))
    pattern = np.repeat((((xx // 2 + yy // 2) % 2) * 160 + 45).astype(np.uint8)[..., None], 3, axis=2)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24, (128, 72))
    for _ in range(6):
        writer.write(pattern)
    writer.release()
    runtime = SimpleNamespace(vae=SimpleNamespace(state_dict=dict), resolved_device="cpu", scaling_factor=0.18215)
    monkeypatch.setattr(study, "load_video_vae_runtime", lambda **_kwargs: runtime)
    monkeypatch.setattr(
        study, "regenerated_frames", lambda frames, *_args: (np.full_like(frame, 128) for frame in frames)
    )
    original_version = study.importlib.metadata.version
    monkeypatch.setattr(
        study.importlib.metadata,
        "version",
        lambda name: "test" if name in {"torch", "diffusers"} else original_version(name),
    )
    output = tmp_path / "run"
    args = [
        "run",
        str(source),
        "--output-dir",
        str(output),
        "--origin",
        "synthetic",
        "--long-side",
        "64",
        "--noise",
        "0",
        "--preserve-timestamps",
        "--reconstruction-weight",
        "0.25",
    ]
    result = CliRunner().invoke(study.main, args)
    assert result.exit_code == 0, result.output + str(result.exception)
    report = json.loads((output / "results.json").read_text())
    rows = {row["id"]: row for row in report["artifacts"]}
    mixed = rows["vae-64-0-mix0.25"]
    assert mixed["guide_id"] == "vae-64-0"
    assert mixed["control_id"] == "reencode-blend-crf0"
    assert mixed["reconstruction_weight"] == 0.25
    control = Path(rows[mixed["control_id"]]["path"])
    target = Path(mixed["path"])
    assert study._probe_video(target)[:2] == (128, 72)
    assert study.probe_video_timestamps(target) == study.probe_video_timestamps(source)
    contrast = next(study._iter_frames(target)).var() / next(study._iter_frames(control)).var()
    assert 0.4 < contrast < 0.7  # Partial reconstruction must retain most source contrast, not replace it.
    assert mixed["watermark_status"] == mixed["capcut_signal_status"] == "unverified"


def test_reconstruction_mixture_rejects_same_count_with_wrong_frame_times(monkeypatch):
    source, reconstruction = Path("source.mp4"), Path("guide.mp4")
    monkeypatch.setattr(study, "probe_video_timestamps", lambda path: (0.0, 0.04) if path == source else (0.01, 0.05))
    monkeypatch.setattr(study, "_iter_frames", lambda _path: iter([np.full((8, 8, 3), 128, np.uint8)] * 2))
    with pytest.raises(ValueError, match="frame timestamps differ"):
        next(study.blend_reconstruction_frames(source, reconstruction, 0.25))


def test_cached_flow_matches_independent_blending(monkeypatch):
    rng = np.random.default_rng(12)
    frames = [rng.integers(70, 90, (32, 32, 3), dtype=np.uint8) for _ in range(4)]
    expected = [
        study.blend_neighbors(frame, frames[max(0, index - 1) : index] + frames[index + 1 : index + 2])
        for index, frame in enumerate(frames)
    ]
    original = study._backward_map
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(study, "_backward_map", counted)
    actual = list(study.temporal_frames(frames))
    assert len(calls) == 2 * (len(frames) - 1)
    for first, second in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(first, second)


@requires_ffmpeg
def test_variable_frame_timing_is_rejected(tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg
    assert ffprobe
    source = tmp_path / "vfr.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=12:duration=1",
            "-vf",
            "settb=1/1000,setpts=N*80+mod(N\\,2)*20",
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            "1/1000",
            str(source),
        ],
        check=True,
    )
    with pytest.raises(Exception, match="Variable frame timing"):
        study.validate_input(source, ffprobe, study._probe_video(source)[2])


@pytest.mark.parametrize(("offset", "short_tail"), [(0, False), (500, False), (0, True)])
@requires_ffmpeg
def test_timestamped_encoding_and_fidelity_keep_vfr_and_start_offset(tmp_path, monkeypatch, offset, short_tail):
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg
    source = tmp_path / "timed.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=12:duration=1",
            "-vf",
            f"settb=1/1000,setpts=N*80+mod(N\\,2)*20+{offset}",
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            "1/1000",
            str(source),
        ],
        check=True,
    )
    if short_tail:
        shortened = tmp_path / "short-tail.mp4"
        subprocess.run(  # noqa: S603
            [
                ffmpeg,
                "-v",
                "error",
                "-copyts",
                "-i",
                str(source),
                "-c",
                "copy",
                "-bsf:v",
                "setts=pts=PTS:dts=DTS:duration='if(eq(N,11),128,DURATION)'",
                "-avoid_negative_ts",
                "disabled",
                str(shortened),
            ],
            check=True,
        )
        source = shortened
    geometry = study._probe_video(source)
    pts = study.probe_video_timestamps(source)
    target = tmp_path / "output.mp4"
    study.encode_frames(study._iter_frames(source), source, target, geometry[2], timestamps=pts)
    assert study.probe_video_timestamps(target) == pytest.approx(pts, abs=0.000002)
    metrics = study._measure(source, target, source_geometry=geometry, duration=None, preserve_timestamps=True)
    assert metrics["frames"] == len(pts)
    assert metrics["max_timestamp_error_seconds"] == 0
    assert metrics["video_duration_error_seconds"] == 0
    if short_tail:
        broken = tmp_path / "wrong-tail.mp4"
        with monkeypatch.context() as patch:
            patch.setattr(study, "match_video_end", lambda *_args: None)
            study.encode_frames(study._iter_frames(source), source, broken, geometry[2], timestamps=pts)
        with pytest.raises(Exception, match="duration changed"):
            study._measure(source, broken, source_geometry=geometry, duration=None, preserve_timestamps=True)
    # A CFR output of the same frame count must not pass time-aware fidelity.
    retimed = tmp_path / "retimed.mp4"
    study.encode_frames(study._iter_frames(source), source, retimed, geometry[2])
    with pytest.raises(Exception, match="timestamps changed"):
        study._measure(source, retimed, source_geometry=geometry, duration=None, preserve_timestamps=True)


def test_study_imports_and_reports_missing_peak_rss_without_unix_resource(monkeypatch):
    import builtins
    import runpy

    original_import = builtins.__import__

    def without_resource(name, *args, **kwargs):
        if name == "resource":
            raise ModuleNotFoundError("No module named 'resource'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_resource)
    namespace = runpy.run_path(study.__file__, run_name="study_without_resource")
    assert namespace["peak_process_rss_bytes"]() is None
