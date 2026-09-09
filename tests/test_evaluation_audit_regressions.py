"""Regressions for evaluation evidence and process outcome handling."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def test_archive_imports():
    import remove_ai_watermarks

    assert Path(remove_ai_watermarks.__file__).is_relative_to(ROOT)


def test_rectangular_video_decode_preserves_pixel_layout(monkeypatch, tmp_path):
    import watermark_benchmark as benchmark

    pixels = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(1, 2, 4, 3)
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: name)
    responses = iter(
        [SimpleNamespace(returncode=0, stdout="4,2"), SimpleNamespace(returncode=0, stdout=pixels.tobytes())]
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: next(responses))
    decoded = benchmark._decode_video(tmp_path / "clip.mp4")
    assert decoded.shape == pixels.shape
    assert np.array_equal(decoded, pixels.astype(np.float32) / 255)


@pytest.mark.parametrize("code", [-11, 127, 137, 1])
def test_smoke_rejects_unexpected_process_outcomes(monkeypatch, tmp_path, code):
    import smoke_matrix as smoke

    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=code, stdout="", stderr=""))
    assert smoke.Runner(tmp_path).run("visible", ["visible", "input.png"], expect_exit=(0, 2)).status == "FAIL"


@pytest.mark.parametrize("code", [-11, 127, 137])
def test_robustness_rejects_abnormal_exit_without_text(monkeypatch, code):
    import robustness_suite as robustness

    monkeypatch.setattr(robustness, "run", lambda *a, **k: (code, "", False))
    result = robustness.Results()
    robustness.graceful(result, "malformed", "identify", [])
    assert result.rows[0][2] is False


def test_video_psnr_is_reported_with_explicit_semantics():
    import watermark_benchmark_report as report

    record = SimpleNamespace(
        data={
            "adapter": "videoseal",
            "state": "attacked",
            "artifact": {"sha256": "a"},
            "reference": {"sha256": "b"},
            "fidelity": {"status": "measured", "mean_psnr_db": 40.0},
        }
    )
    group = report._fidelity_groups([record])[0]
    assert group["finite_mean_psnr"] == 1
    assert group["p50_mean_psnr_db"] == 40.0
    assert group["finite_psnr"] == 0


def test_web_results_survive_later_browser_failure(monkeypatch, tmp_path):
    import provider_oracle_web as web

    manifest = {
        "surface": "meta-web",
        "slot": {},
        "rows": [{"artifact_id": "first", "upload_path": "one"}, {"artifact_id": "second", "upload_path": "two"}],
    }
    results = {"rows": [{"watermark_result": None}, {"watermark_result": None}]}
    saved = []
    monkeypatch.setattr(web.oracles, "verify_batch", lambda *a, **k: {})
    monkeypatch.setattr(web.oracles, "load_batch", lambda *a: (manifest, results))
    monkeypatch.setattr(web, "proxy_settings", lambda *a: None)
    monkeypatch.setattr(web.oracles, "record_result", lambda *a, **k: saved.append(k))

    def submit(*args, **kwargs):
        yield web.WebVerdict("detected", "unavailable", "first settled")
        assert len(saved) == 1
        raise RuntimeError("second page failed")

    monkeypatch.setattr(web, "_submit_batch", submit)
    with pytest.raises(RuntimeError, match="second page failed"):
        web.run_web_batch(tmp_path / "manifest.json", acknowledge_uploads=True)
    assert saved[0]["artifact_id"] == "first"
    assert saved[0]["checked_at"].endswith("Z")


def test_normalized_edit_distance_retains_historical_value():
    import fidelity_metrics as fidelity

    assert fidelity._text_ned("a", "aaaa") == 0.75
    assert fidelity._text_ned("A", "a") == 1.0


def test_recall_deduplicates_bytes_independently_of_detector(tmp_path):
    import visible_recall_sample as recall

    paths = [tmp_path / name for name in ("a", "b", "copy")]
    for path, content in zip(paths, (b"a", b"b", b"a"), strict=True):
        path.write_bytes(content)
    rows = [{"path": str(path), "shape": [4, 4], "marks": {"x": {"conf": 0.5}}} for path in paths]
    assert recall.unique_content_rows(rows) == rows[:2]


def test_temporal_study_measures_only_saved_decodes(monkeypatch, tmp_path):
    import videoseal_temporal_study as study

    raw = np.zeros((2, 4, 4, 3), dtype=np.float32)
    decoded = {}
    measured = []

    def encode(_ffmpeg, path, frames):
        path.write_bytes(path.name.encode())
        decoded[path] = np.full_like(raw, len(decoded) / 20 + 0.1)
        return path

    def read(_model, frames):
        assert any(frames is value for value in decoded.values())
        measured.append(frames)
        return SimpleNamespace(bit_accuracy=0.75, per_frame_bit_accuracy=[0.75, 0.75])

    def matrix(_model, frames):
        assert any(frames is value for value in decoded.values())
        measured.append(frames)
        return {"avg": 0.75}

    monkeypatch.setattr(study, "require_tools", lambda: "fake")
    monkeypatch.setattr(study, "ffmpeg_version", lambda _: "fake")
    monkeypatch.setattr(study.videoseal_oracle, "load_model", lambda: object())
    monkeypatch.setattr(study.videoseal_oracle, "embed", lambda *_: raw)
    monkeypatch.setattr(study, "synth_carrier", lambda _: raw)
    monkeypatch.setattr(study, "REAL_CARRIERS", ())
    monkeypatch.setattr(study, "CRF_SWEEP", ())
    monkeypatch.setattr(study, "encode_clip", encode)
    monkeypatch.setattr(study, "apply_scale", lambda _f, _p, d: encode(_f, d / "scale.mp4", raw))
    monkeypatch.setattr(study, "apply_fps_half", lambda _f, _p, d: encode(_f, d / "fps.mp4", raw))
    import watermark_benchmark as benchmark

    monkeypatch.setattr(benchmark, "_decode_video", lambda path: decoded[path])
    monkeypatch.setattr(study, "_decode_video", lambda path: decoded[path])
    monkeypatch.setattr(study, "read", read)
    monkeypatch.setattr(study, "read_aggregation_matrix", matrix)
    monkeypatch.setattr(study, "render_matrix", lambda rows: "")
    monkeypatch.setattr(sys, "argv", ["study", "--output-dir", str(tmp_path / "out")])
    assert study.main() == 0
    assert len(measured) == 14


def test_forgery_study_measures_only_saved_decodes(monkeypatch, tmp_path):
    import watermark_benchmark as benchmark
    import watermark_benchmark_video_cohort as cohort
    import watermark_forgery_study as study

    raw = np.zeros((2, 4, 4, 3), dtype=np.float32)
    decoded = {}
    measured = []

    def encode(_ffmpeg, path, frames):
        path.write_bytes(path.name.encode())
        decoded[path] = np.full_like(raw, len(decoded) / 30 + 0.1)
        return path

    def read(_model, frames, **kwargs):
        assert any(frames is value for value in decoded.values())
        measured.append(frames)
        return SimpleNamespace(bit_accuracy=0.75, decoded_bits=study.videoseal_oracle.message_bits(), detected=True)

    monkeypatch.setattr(study.videoseal_oracle, "load_model", lambda: object())
    monkeypatch.setattr(study.videoseal_oracle, "embed", lambda *_: raw)
    monkeypatch.setattr(study.videoseal_oracle, "read", read)
    monkeypatch.setattr(study, "synth_video_carrier", lambda _: raw)
    monkeypatch.setattr(study, "real_carrier_frames", lambda _: (raw, {}))
    monkeypatch.setattr(cohort, "encode_clip", encode)
    monkeypatch.setattr(study, "apply_crf", lambda _f, _p, _c, d: encode(_f, d / "removed.mp4", raw))
    monkeypatch.setattr(benchmark, "_decode_video", lambda path: decoded[path])
    rows = study.video_cells(tmp_path, "fake")
    assert len(measured) == 12
    assert len(rows) == 13


def test_isolated_workers_contain_process_exit(tmp_path):
    import _isolated_image_workers as workers

    script = tmp_path / "worker.py"
    script.write_text(
        "import os\ndef _one(path):\n    if path == 'bad':\n        os._exit(7)\n"
        "    return {'path': path, 'status': 'ok'}\n"
    )
    rows = workers.run_batch(script, ["good", "bad", "last"], jobs=2, timeout=10)
    assert [row["status"] for row in rows] == ["ok", "crashed", "ok"]
    assert rows[1]["returncode"] == 7


def test_isolated_worker_timeout_has_no_parent_fallback(monkeypatch, tmp_path):
    import _isolated_image_workers as workers

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(workers.subprocess, "run", run)
    rows = workers.run_batch(tmp_path / "never-import-this.py", ["one", "two"], jobs=2, timeout=3)
    assert [row["status"] for row in rows] == ["timeout", "timeout"]
    assert len(calls) == 2
    assert all(kwargs["timeout"] == 3 for _, kwargs in calls)


@pytest.mark.parametrize(("name", "function"), [("visible_positives", "_run_batch"), ("pill_gate_audit", "_batch")])
def test_audit_uses_isolated_runner(monkeypatch, name, function):
    module = importlib.import_module(name)
    calls = []

    def run(script, paths, *, jobs, timeout):
        calls.append((script, paths, jobs, timeout))
        return [{"path": "one", "status": "timeout"}]

    monkeypatch.setattr(module, "run_batch", run)
    assert getattr(module, function)(["one"], 2, 3)[0]["status"] == "timeout"
    assert calls == [(Path(module.__file__), ["one"], 2, 3)]


def test_audio_manifest_records_executed_carrier_seeds(monkeypatch, tmp_path):
    import json

    import audioseal_experiment as experiment
    import watermark_benchmark_audio_cohort as cohort

    observed = []
    original = np.random.default_rng

    def rng(seed):
        observed.append(seed)
        return original(seed)

    monkeypatch.setattr(experiment.np.random, "default_rng", rng)
    monkeypatch.setattr(cohort.audioseal_oracle, "available", lambda: True)
    monkeypatch.setattr(cohort.audioseal_oracle, "load_pinned_models", lambda: (None, None))
    monkeypatch.setattr(cohort.audioseal_oracle, "embed", lambda _m, data, _msg: data + 0.01)
    monkeypatch.setattr(cohort, "require_tools", lambda: "fake")
    monkeypatch.setattr(cohort, "oracle_dependencies", lambda: {"audioseal": "test"})
    manifest = cohort.build_cohort(tmp_path / "audio", carriers=("white_noise",), attacks=(), duration_s=0.01)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    clean = next(row for row in rows if row["case_id"] == "white_noise-clean")
    hard = next(row for row in rows if row["arm"] == "hard_negative")
    assert clean["seed"] == experiment.carrier_seed("white_noise")
    assert clean["seed"] in observed
    assert hard["seed"] in observed


def test_video_hard_negative_manifest_records_executed_seed(monkeypatch, tmp_path):
    import json

    import watermark_benchmark_video_cohort as cohort

    observed = []
    original = np.random.default_rng

    def rng(seed):
        observed.append(seed)
        return original(seed)

    monkeypatch.setattr(cohort.np.random, "default_rng", rng)
    monkeypatch.setattr(cohort, "FRAME_COUNT", 1)
    monkeypatch.setattr(cohort, "HEIGHT", 8)
    monkeypatch.setattr(cohort, "WIDTH", 128)
    monkeypatch.setattr(cohort.videoseal_oracle, "load_model", lambda: None)
    monkeypatch.setattr(cohort, "require_tools", lambda: "fake")
    monkeypatch.setattr(cohort, "oracle_dependencies", dict)

    def encode(_f, path, frames):
        path.write_bytes(frames.tobytes())
        return path

    monkeypatch.setattr(cohort, "encode_clip", encode)
    manifest = cohort.build_cohort(tmp_path / "video", carriers=(), attacks=())
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert rows[0]["seed"] == observed[-1]


def test_video_artifact_rejects_a_change_during_decode(monkeypatch, tmp_path):
    import watermark_benchmark as benchmark

    path = tmp_path / "clip"
    path.write_bytes(b"initial")

    def decode(source):
        source.write_bytes(b"changed")
        return np.zeros((1, 2, 2, 3))

    monkeypatch.setattr(benchmark, "_decode_video", decode)
    with pytest.raises(ValueError, match="changed during decoding"):
        benchmark.decode_video_artifact(path)


@pytest.mark.parametrize("code", [0, 2])
def test_smoke_accepts_only_declared_outcomes(monkeypatch, tmp_path, code):
    import smoke_matrix as smoke

    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=code, stdout="", stderr=""))
    assert smoke.Runner(tmp_path).run("visible", ["visible", "input.png"], expect_exit=(0, 2)).status == "pass"


def test_web_real_iterator_persists_before_late_page_failure(monkeypatch, tmp_path):
    import json
    from contextlib import nullcontext

    import playwright.sync_api as playwright
    import provider_oracle_web as web
    import provider_oracles as oracles
    from PIL import Image

    sources = []
    for name, color in [("one", "red"), ("two", "blue")]:
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (8, 8), color).save(path)
        sources.append(path)
    slot = {
        "name": "meta-direct",
        "surface": "meta-web",
        "account_label": None,
        "browser_profile": None,
        "google_account_index": None,
        "network_label": "direct",
        "proxy_url_env": None,
    }
    manifest = oracles.prepare_batch(
        "meta-web", sources, output_dir=tmp_path / "batch", repository_root=ROOT, slot=slot
    )
    pages = []

    def new_page():
        if pages:
            saved = json.loads((manifest.parent / "results.json").read_text())
            assert saved["rows"][0]["watermark_result"] == "detected"
            raise RuntimeError("later page failed")
        page = SimpleNamespace(close=lambda: None)
        pages.append(page)
        return page

    context = SimpleNamespace(new_page=new_page, close=lambda: None)
    browser = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(playwright, "sync_playwright", lambda: nullcontext(SimpleNamespace(chromium=None)))
    monkeypatch.setattr(web, "_launch_browser", lambda *a, **k: browser)
    monkeypatch.setattr(web, "_new_context", lambda *a, **k: context)
    monkeypatch.setattr(web, "_check_page", lambda *a: web.WebVerdict("detected", "unavailable", "settled response"))
    with pytest.raises(RuntimeError, match="later page failed"):
        web.run_web_batch(manifest, acknowledge_uploads=True, environ={})
    saved = json.loads((manifest.parent / "results.json").read_text())
    assert saved["rows"][0]["watermark_result"] == "detected"
    assert saved["rows"][1]["watermark_result"] is None
    submitted = []

    def remaining(surface, uploads, *a, **k):
        submitted.extend(uploads)
        return [web.WebVerdict("not_detected", "unavailable", "second settled")]

    monkeypatch.setattr(web, "_submit_batch", remaining)
    web.run_web_batch(manifest, acknowledge_uploads=True, environ={})
    assert len(submitted) == 1


def test_recall_main_keeps_distinct_content_with_identical_observations(monkeypatch, tmp_path):
    import csv
    import json

    import visible_recall_sample as recall
    from PIL import Image

    rows = []
    for index, color in enumerate(("red", "blue", "red")):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (32, 32), color).save(path)
        rows.append(
            {
                "path": str(path),
                "shape": [32, 32],
                "uid": str(index),
                "cls": "tc260",
                "marks": {"x": {"conf": 0.5, "strict": False}},
            }
        )
    scan = tmp_path / "scan.jsonl"
    scan.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["sample", str(scan), str(out), "--tc260", "10"])
    recall.main()
    with (out / "MANIFEST_DO_NOT_OPEN.csv").open() as source:
        sampled = list(csv.DictReader(source))
    assert {row["uid"] for row in sampled} == {"0", "1"}


def test_real_rectangular_video_decode_preserves_pixels(tmp_path):
    import shutil

    import watermark_benchmark as benchmark

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg and ffprobe required")
    pixels = np.arange(8 * 16 * 3, dtype=np.uint8).reshape(1, 8, 16, 3)
    path = tmp_path / "rectangular.mkv"
    subprocess.run(  # noqa: S603 -- resolved ffmpeg and generated input
        [
            ffmpeg,
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "16x8",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            "-c:v",
            "ffv1",
            str(path),
        ],
        input=pixels.tobytes(),
        capture_output=True,
        check=True,
    )
    decoded = benchmark._decode_video(path)
    assert decoded.shape == pixels.shape
    assert np.array_equal(decoded, pixels.astype(np.float32) / 255)


@pytest.mark.parametrize(("name", "function"), [("visible_positives", "_run_batch"), ("pill_gate_audit", "_batch")])
def test_real_audit_worker_entry_points(name, function, tmp_path):
    module = importlib.import_module(name)
    # A missing synthetic path still reaches real imread inside the isolated child.
    path = str(tmp_path / "missing.png")
    rows = getattr(module, function)([path], 1, 10)
    assert rows[0]["status"] == "unreadable"
