"""Regression coverage for the opt-in Seedance pixel recipe and output contract."""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks import video_seedance as seedance
from remove_ai_watermarks.video_encoding import VideoEncodeProfile, probe_video_timestamps, video_end_repair
from remove_ai_watermarks.video_invisible import RegenerationMetrics

if TYPE_CHECKING:
    from pathlib import Path


def _metrics() -> RegenerationMetrics:
    return RegenerationMetrics(frames=12, fps=12, width=64, height=48, psnr_db=30, temporal_residual_ratio=1)


def test_texture_uses_guide_low_frequencies_and_source_fine_detail():
    source = np.full((64, 64, 3), 160, np.uint8)
    guide = np.full((32, 32, 3), 80, np.uint8)
    assert np.all(seedance._restore_texture(source, guide) == 80)
    source[::2] = 64
    source[1::2] = 192
    output = seedance._restore_texture(source, guide)
    assert output.shape == source.shape
    assert float(output[10::2].mean()) < 25
    assert float(output[11::2].mean()) > 135
    assert float(output[10:-10].mean()) == pytest.approx(80, abs=1)


@pytest.mark.parametrize(("value", "expected"), [(64.5, 64), (65.5, 66)])
def test_texture_rounds_half_to_even(value, expected):
    source = np.zeros((16, 16, 3), np.uint8)
    guide = np.full((16, 16, 3), value, np.float64)
    assert np.all(seedance._restore_texture(source, guide) == expected)


def test_high_level_api_is_explicit_and_reports_experimental_status(tmp_path, monkeypatch):
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    calls = []

    def generate(actual_source, output, **kwargs):
        calls.append((actual_source, kwargs))
        output.write_bytes(source.read_bytes())
        return _metrics()

    monkeypatch.setattr(seedance, "regenerate_seedance_experimental", generate)
    result = raiw.remove_video_seedance_experimental(source, quality="master", device="cpu")
    assert result.output == tmp_path / "owned_seedance_experimental.mp4"
    assert calls == [(source, {"quality": "master", "device": "cpu", "runtime": None})]
    assert result.profile == "seedance-texture-v1"
    assert result.experimental is True
    assert result.quality == "master"
    assert result.total_frames == 12
    assert result.visual_invisible_action == "regenerated"
    assert result.remaining_metadata == {}
    assert result.audio.stream_action == "copied_if_present"
    assert result.audio.watermark_status == "unverified"


def test_invalid_quality_preserves_existing_output(tmp_path):
    source, output = tmp_path / "owned.mp4", tmp_path / "existing.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    output.write_bytes(b"existing")
    with pytest.raises(ValueError, match="quality"):
        raiw.remove_video_seedance_experimental(source, output, quality="unknown")
    assert output.read_bytes() == b"existing"


def test_rejects_in_place_processing_and_other_containers(tmp_path):
    source = tmp_path / "owned.mp4"
    payload = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    source.write_bytes(payload)
    with pytest.raises(ValueError, match="distinct output"):
        raiw.remove_video_seedance_experimental(source, source)
    assert source.read_bytes() == payload
    other = tmp_path / "owned.mov"
    other.write_bytes(payload)
    with pytest.raises(ValueError, match="MP4 input"):
        raiw.remove_video_seedance_experimental(other)


def test_hdr_is_rejected_before_model_loading(tmp_path, monkeypatch):
    output = tmp_path / "existing.mp4"
    output.write_bytes(b"existing")
    monkeypatch.setattr(seedance, "probe_video_encode_profile", lambda _: VideoEncodeProfile(component_depth=10))
    monkeypatch.setattr(seedance, "load_video_vae_runtime", lambda **_: pytest.fail("Model must not load"))
    with pytest.raises(ValueError, match="SDR 8-bit"):
        seedance.regenerate_seedance_experimental(tmp_path / "hdr.mp4", output)
    assert output.read_bytes() == b"existing"


def _source(tmp_path: Path, *, short_tail: bool = False) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "source.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=12:duration=1",
            "-itsoffset",
            "2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=32000:duration=1.5",
            "-vf",
            "settb=1/1000,setpts=N*80+mod(N\\,2)*20+2000",
            "-fps_mode",
            "vfr",
            "-enc_time_base",
            "1/1000",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    if short_tail:
        retimed = tmp_path / "short-tail.mp4"
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
                str(retimed),
            ],
            check=True,
        )
        return retimed
    return source


def _fake_model(monkeypatch):
    monkeypatch.setattr(seedance, "is_available", lambda: True)
    monkeypatch.setattr(seedance, "load_video_vae_runtime", lambda **_: SimpleNamespace())
    monkeypatch.setattr(
        seedance,
        "_reconstruct",
        lambda source, _runtime, size: (cv2.resize(frame, size) for frame in seedance._frames(source)),
    )


def _audio(path: Path) -> tuple[bytes, list[dict]]:
    payload = subprocess.check_output(  # noqa: S603
        [shutil.which("ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "adts", "pipe:1"]
    )
    raw = subprocess.check_output(  # noqa: S603
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_entries",
            "packet=pts_time,dts_time,duration_time,size",
            "-of",
            "json",
            str(path),
        ]
    )
    return payload, json.loads(raw)["packets"]


@pytest.mark.parametrize("quality", ["master", "delivery"])
@pytest.mark.parametrize("short_tail", [False, True])
def test_complete_pixel_pipeline_preserves_vfr_offset_audio_and_geometry(tmp_path, monkeypatch, quality, short_tail):
    source = _source(tmp_path, short_tail=short_tail)
    timestamps = probe_video_timestamps(source)
    assert timestamps[0] == pytest.approx(2)
    assert float(np.ptp(np.diff(timestamps))) > 0.01
    _fake_model(monkeypatch)
    output = tmp_path / "out.mp4"
    if quality == "delivery" and not short_tail:
        runtime = SimpleNamespace(model="stabilityai/sd-vae-ft-mse", requested_device="cpu")
        monkeypatch.setattr(seedance, "load_video_vae_runtime", lambda **_: pytest.fail("Cached VAE must be reused"))
        result = raiw.remove_video_invisible(
            source, output, profile="seedance-experimental", device="cpu", _runtime=runtime
        )
    else:
        result = raiw.remove_video_seedance_experimental(source, output, quality=quality, device="cpu")
    assert probe_video_timestamps(output) == timestamps
    assert result.total_frames == len(timestamps) == 12
    assert (result.width, result.height) == (64, 48)
    assert _audio(output) == _audio(source)
    assert float(seedance._clock(output)["streams"][0]["duration"]) == pytest.approx(
        float(seedance._clock(source)["streams"][0]["duration"]), abs=0.000002
    )
    assert result.remaining_metadata == {}
    assert not list(tmp_path.glob("seedance-experimental-*"))


def test_clock_mismatch_prevents_publication_and_cleans_intermediates(tmp_path, monkeypatch):
    source = _source(tmp_path)
    _fake_model(monkeypatch)
    original_probe = seedance.probe_video_timestamps
    monkeypatch.setattr(
        seedance,
        "probe_video_timestamps",
        lambda path: original_probe(path) if path == source else (99.0,),
    )
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous output")
    with pytest.raises(RuntimeError, match="timestamps changed"):
        seedance.regenerate_seedance_experimental(source, output, quality="master", device="cpu")
    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("seedance-experimental-*"))


def test_encoding_failure_keeps_existing_output(tmp_path, monkeypatch):
    source = _source(tmp_path)
    _fake_model(monkeypatch)

    def fail(*_):
        raise RuntimeError("Synthetic delivery failure")

    monkeypatch.setattr(seedance, "_encode_delivery", fail)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous output")
    with pytest.raises(RuntimeError, match="Synthetic delivery failure"):
        seedance.regenerate_seedance_experimental(source, output, device="cpu")
    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("seedance-experimental-*"))


def test_truncated_reconstruction_cannot_publish_partial_video(tmp_path, monkeypatch):
    source = _source(tmp_path)
    _fake_model(monkeypatch)
    monkeypatch.setattr(seedance, "_reconstruct", lambda *_: iter([np.zeros((48, 64, 3), np.uint8)]))
    output = tmp_path / "out.mp4"
    output.write_bytes(b"previous output")
    with pytest.raises(ValueError, match="zip"):
        seedance.regenerate_seedance_experimental(source, output, device="cpu")
    assert output.read_bytes() == b"previous output"
    assert not list(tmp_path.glob("seedance-experimental-*"))


def test_common_invisible_api_routes_seedance_and_keeps_output_convention(tmp_path, monkeypatch):
    from remove_ai_watermarks import video

    source = tmp_path / "owned.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    calls = []

    def generate(actual_source, output, **kwargs):
        calls.append((actual_source, output, kwargs))
        output.write_bytes(source.read_bytes())
        return _metrics()

    monkeypatch.setattr(seedance, "regenerate_seedance_experimental", generate)
    result = video.remove_video_invisible(source, profile="seedance-experimental", quality="master", device="cpu")
    assert result.output == tmp_path / "owned_clean.mp4"
    assert calls == [(source, result.output, {"quality": "master", "device": "cpu", "runtime": None})]
    assert result.experimental
    assert result.quality == "master"


def test_all_runs_visible_before_seedance_and_reports_selected_profile(tmp_path, monkeypatch):
    from remove_ai_watermarks import video

    source, output = tmp_path / "owned.mp4", tmp_path / "clean.mp4"
    payload = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    source.write_bytes(payload)
    calls = []

    def visible(actual_source, target, **kwargs):
        assert actual_source == source
        target.write_bytes(payload + b"\x00\x00\x00\x08free")
        return video.VideoVisibleResult(source, target, "seedance", 12, 12, 12, {})

    def generate(actual_source, target, **kwargs):
        assert actual_source != source
        assert actual_source.read_bytes() != payload
        calls.append((actual_source, kwargs))
        target.write_bytes(payload)
        return _metrics()

    monkeypatch.setattr(video, "remove_video_visible", visible)
    monkeypatch.setattr(seedance, "regenerate_seedance_experimental", generate)
    result = video.remove_video_all(
        source,
        output,
        include_invisible=True,
        invisible_profile="seedance-experimental",
        invisible_quality="master",
        device="cpu",
    )
    assert result.invisible_profile == "seedance-experimental"
    assert result.invisible_removed
    assert result.visible_mark == "seedance"
    assert calls[0][1] == {"quality": "master", "device": "cpu", "runtime": None}
    assert not calls[0][0].exists()
    assert output.read_bytes() == payload


@pytest.mark.parametrize(
    "options",
    [
        {"invisible_profile": "unknown", "include_invisible": True},
        {"invisible_profile": "seedance-experimental"},
        {"include_invisible": True, "invisible_quality": "master"},
        {"include_invisible": True, "invisible_profile": "seedance-experimental", "noise_std": 0.5},
    ],
)
def test_invalid_pipeline_profile_options_fail_before_visible_stage(tmp_path, monkeypatch, options):
    from remove_ai_watermarks import video

    source, output = tmp_path / "owned.mp4", tmp_path / "clean.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    output.write_bytes(b"previous output")
    monkeypatch.setattr(video, "remove_video_visible", lambda *a, **k: pytest.fail("Visible stage must not run"))
    with pytest.raises(ValueError, match=r"profile|recipe"):
        video.remove_video_all(source, output, **options)
    assert output.read_bytes() == b"previous output"


def test_seedance_batch_reuses_runtime_and_rejects_other_containers(tmp_path, monkeypatch):
    from remove_ai_watermarks import video, video_invisible

    directory = tmp_path / "videos"
    directory.mkdir()
    payload = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    for name in ("one.mp4", "two.mp4", "unsupported.mov"):
        (directory / name).write_bytes(payload)
    runtime = SimpleNamespace(model="stabilityai/sd-vae-ft-mse", requested_device="cpu")
    loaded, used = [], []

    def load(**kwargs):
        loaded.append(kwargs)
        return runtime

    def visible(source, target, **kwargs):
        return video.VideoVisibleResult(source, None, None, 12, 0, 0, {})

    def generate(source, target, **kwargs):
        used.append(kwargs)
        target.write_bytes(payload)
        return _metrics()

    monkeypatch.setattr(video_invisible, "load_video_vae_runtime", load)
    monkeypatch.setattr(video, "remove_video_visible", visible)
    monkeypatch.setattr(seedance, "regenerate_seedance_experimental", generate)
    result = video.remove_video_batch(
        directory,
        include_invisible=True,
        invisible_profile="seedance-experimental",
        device="cpu",
    )
    assert result.processed == 2
    assert result.failed == 1
    assert loaded == [{"model": "stabilityai/sd-vae-ft-mse", "device": "cpu"}]
    assert used == [{"device": "cpu", "quality": "delivery", "runtime": runtime}] * 2
    assert "MP4 input" in result.items[-1].error
    assert not (result.output_directory / "unsupported.mov").exists()


@pytest.mark.parametrize("command", ["invisible", "all", "batch"])
def test_cli_forwards_profile_quality_and_names_experiment(tmp_path, monkeypatch, command):
    from click.testing import CliRunner

    from remove_ai_watermarks import video
    from remove_ai_watermarks.cli import main

    source = tmp_path / "owned.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    output = tmp_path / "clean.mp4"
    calls = []

    def process(*args, **kwargs):
        calls.append(kwargs)
        if command == "invisible":
            return video.VideoSeedanceExperimentalResult(source, output, 0, _metrics(), {}, quality="master")
        if command == "all":
            return video.VideoAllResult(
                source, output, None, 12, 0, 0, {}, {}, True, invisible_profile="seedance-experimental"
            )
        return video.VideoBatchResult(tmp_path, tmp_path / "clean", ())

    monkeypatch.setattr(video, f"remove_video_{command}", process)
    args = [
        "video",
        command,
        str(tmp_path if command == "batch" else source),
        "--profile",
        "seedance-experimental",
        "--quality",
        "master",
        "--device",
        "cpu",
    ]
    if command != "invisible":
        args.append("--invisible")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == (2 if command == "batch" else 0), result.output
    prefix = "" if command == "invisible" else "invisible_"
    assert calls[0][prefix + "profile"] == "seedance-experimental"
    assert calls[0][prefix + "quality"] == "master"
    assert calls[0]["device"] == "cpu"
    if command != "batch":
        assert "experimental Seedance profile" in result.output
        assert "calibrated SynthID" not in result.output


@pytest.mark.parametrize(
    ("packet_pts", "decode_duration", "final_pts", "presentation_duration"),
    [([2000, 3900, 3600], 200, 3900, 100), ([2000, 3600, 3700], 300, 3700, 300)],
)
def test_end_repair_separates_decode_and_presentation_clocks(
    packet_pts, decode_duration, final_pts, presentation_duration
):
    source = {"streams": [{"time_base": "1/1000", "start_pts": 1000, "duration_ts": 1000}]}
    encoded = {
        "streams": [{"time_base": "1/2000"}],
        "packets": [
            {"dts": 1600, "pts": packet_pts[0]},
            {"dts": 3200, "pts": packet_pts[1]},
            {"dts": 3400, "pts": packet_pts[2]},
        ],
    }
    repair = video_end_repair(source, encoded)
    assert repair.last_packet_index == 2
    assert repair.last_packet_duration == decode_duration
    assert repair.final_frame_pts == final_pts
    assert repair.final_frame_duration == presentation_duration
    assert repair.movie_timescale == 2000
