"""Unit coverage for experimental CUDA 3D Wan 2.1 video diffusion profiles."""

from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks import video
from remove_ai_watermarks import video_wan as wan
from remove_ai_watermarks.video_encoding import VideoEncodeProfile, probe_video_timestamps
from remove_ai_watermarks.video_invisible import RegenerationMetrics

if TYPE_CHECKING:
    from pathlib import Path


def _metrics() -> RegenerationMetrics:
    return RegenerationMetrics(frames=6, fps=12.0, width=70, height=50, psnr_db=31.0, temporal_residual_ratio=0.98)


def _source_mp4(tmp_path: Path, *, width: int = 70, height: int = 50, frames: int = 6) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "source.mp4"
    duration = frames / 12.0
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate=12:duration={duration:.3f}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=32000:duration={duration:.3f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
    )
    return source


def test_cuda_device_guard_rejects_non_cuda_and_missing_gpu(monkeypatch):
    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="device='auto' or device='cuda'"):
        wan.resolve_wan_device("cpu")
    with pytest.raises(ValueError, match="device='auto' or device='cuda'"):
        wan.resolve_wan_device("mps")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA CUDA GPU"):
        wan.resolve_wan_device("auto")


def test_pad_and_run_wan_enforces_16px_spatial_and_4k_plus_1_temporal_grid():
    pytest.importorskip("torch")
    calls = []

    class FakePipe:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            video_frames = kwargs["video"]
            # Must satisfy 16-px spatial multiples and (T - 1) % 4 == 0
            assert kwargs["height"] % 16 == 0
            assert kwargs["width"] % 16 == 0
            assert (len(video_frames) - 1) % 4 == 0
            stacked = np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in video_frames], axis=0)
            return SimpleNamespace(frames=[stacked])

    runtime = wan.VideoWanRuntime(
        model=wan.WAN21_MODEL_ID,
        revision=wan.WAN21_MODEL_REVISION,
        resolved_device="cpu",
        pipe=FakePipe(),
    )
    # 6 frames (not 4k+1 -> pads to 9), 50x70 BGR (pads to 64x80)
    rng = np.random.default_rng(11)
    frames_bgr = [rng.integers(10, 240, size=(50, 70, 3), dtype=np.uint8) for _ in range(6)]
    out_bgr = wan._pad_and_run_wan(frames_bgr, runtime, strength=0.12, seed=42)

    assert len(calls) == 1
    assert calls[0]["height"] == 64
    assert calls[0]["width"] == 80
    assert len(calls[0]["video"]) == 9
    assert calls[0]["strength"] == pytest.approx(0.12)
    assert out_bgr.shape == (6, 50, 70, 3)
    np.testing.assert_array_equal(out_bgr, np.stack(frames_bgr, axis=0))


@pytest.mark.parametrize("profile", ["wan21-synthid-experimental", "wan21-seedance-experimental"])
@pytest.mark.parametrize("quality", ["delivery", "master"])
def test_complete_wan_pipeline_preserves_timestamps_audio_and_geometry(tmp_path, monkeypatch, profile, quality):
    pytest.importorskip("torch")
    source = _source_mp4(tmp_path, width=70, height=50, frames=6)
    timestamps = probe_video_timestamps(source)
    assert len(timestamps) == 6

    class IdentityPipe:
        def __call__(self, **kwargs):
            stacked = np.stack([np.asarray(img, dtype=np.uint8) for img in kwargs["video"]], axis=0)
            return SimpleNamespace(frames=[stacked])

    runtime = wan.VideoWanRuntime(
        model=wan.WAN21_MODEL_ID,
        revision=wan.WAN21_MODEL_REVISION,
        resolved_device="cpu",
        pipe=IdentityPipe(),
    )
    monkeypatch.setattr(wan, "load_video_wan_runtime", lambda **_: pytest.fail("Injected runtime must be reused"))
    output = tmp_path / f"out_{profile}_{quality}.mp4"
    result = raiw.remove_video_invisible(
        source,
        output,
        profile=profile,  # type: ignore[arg-type]
        quality=quality,  # type: ignore[arg-type]
        seed=7,
        device="cuda",
        _runtime=runtime,
    )
    assert isinstance(result, raiw.VideoWanExperimentalResult)
    assert result.profile == profile
    assert result.experimental is True
    assert result.quality == quality
    assert result.noise_std == pytest.approx(0.38 if profile == "wan21-synthid-experimental" else 0.12)
    assert probe_video_timestamps(output) == timestamps
    assert (result.width, result.height) == (70, 50)
    assert result.total_frames == 6
    assert result.remaining_metadata == {}
    assert not list(tmp_path.glob("wan21-experimental-*"))


def test_validation_rejects_hdr_odd_dims_wrong_runtime_and_2d_vae_flags(tmp_path, monkeypatch):
    source = tmp_path / "owned.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    output = tmp_path / "out.mp4"

    with pytest.raises(ValueError, match="2D VAE SynthID tuning options"):
        video.remove_video_all(
            source,
            output,
            include_invisible=True,
            invisible_profile="wan21-synthid-experimental",
            long_side=768,
        )
    other = tmp_path / "owned.mov"
    other.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="MP4 input"):
        raiw.remove_video_wan_experimental(other)

    bad_runtime = wan.VideoWanRuntime(
        model="other/model",
        revision="deadbeef",
        resolved_device="cpu",
        pipe=None,
    )
    with pytest.raises(ValueError, match="pinned model and revision"):
        wan.regenerate_wan_experimental(source, output, runtime=bad_runtime)

    monkeypatch.setattr(wan, "probe_video_encode_profile", lambda _: VideoEncodeProfile(component_depth=10))
    with pytest.raises(ValueError, match="SDR 8-bit"):
        wan.regenerate_wan_experimental(source, output)


@pytest.mark.parametrize("profile", ["wan21-synthid-experimental", "wan21-seedance-experimental"])
def test_cli_routes_wan21_profiles_and_prints_experimental_label(tmp_path, monkeypatch, profile):
    from click.testing import CliRunner

    from remove_ai_watermarks.cli import main

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    output = tmp_path / "clean.mp4"
    calls = []

    def fake_invisible(*args, **kwargs):
        calls.append(kwargs)
        return raiw.VideoWanExperimentalResult(
            source=source,
            output=output,
            noise_std=0.38,
            metrics=_metrics(),
            remaining_metadata={},
            quality=kwargs["quality"],
            profile=profile,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(video, "remove_video_invisible", fake_invisible)
    res = CliRunner().invoke(
        main,
        [
            "video",
            "invisible",
            str(source),
            "-o",
            str(output),
            "--profile",
            profile,
            "--quality",
            "master",
            "--seed",
            "9",
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls[0]["profile"] == profile
    assert calls[0]["quality"] == "master"
    assert calls[0]["seed"] == 9
    expected_label = (
        "experimental CUDA 3D Wan 2.1 SynthID profile"
        if profile == "wan21-synthid-experimental"
        else "experimental CUDA 3D Wan 2.1 Seedance profile"
    )
    assert expected_label in res.output
    assert "calibrated SynthID" not in res.output


def test_wan_batch_reuses_wan_runtime_across_mp4_items(tmp_path, monkeypatch):
    directory = tmp_path / "clips"
    directory.mkdir()
    payload = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    for name in ("a.mp4", "b.mp4", "c.mov"):
        (directory / name).write_bytes(payload)

    runtime = wan.VideoWanRuntime(
        model=wan.WAN21_MODEL_ID,
        revision=wan.WAN21_MODEL_REVISION,
        resolved_device="cuda",
        pipe="sentinel",
    )
    loaded, used = [], []

    monkeypatch.setattr(wan, "load_video_wan_runtime", lambda **kw: (loaded.append(kw), runtime)[1])
    monkeypatch.setattr(
        video,
        "remove_video_visible",
        lambda src, dst, **kw: video.VideoVisibleResult(src, None, None, 6, 0, 0, {}),
    )

    def fake_regen(src, dst, **kw):
        used.append(kw)
        dst.write_bytes(payload)
        return _metrics()

    monkeypatch.setattr(wan, "regenerate_wan_experimental", fake_regen)
    batch_res = video.remove_video_batch(
        directory,
        include_invisible=True,
        invisible_profile="wan21-seedance-experimental",
        invisible_quality="master",
        seed=5,
        device="cuda",
    )
    assert batch_res.processed == 2
    assert batch_res.failed == 1
    assert loaded == [{"device": "cuda"}]
    assert (
        used
        == [
            {
                "profile": "wan21-seedance-experimental",
                "device": "cuda",
                "quality": "master",
                "seed": 5,
                "chunk_frames": 193,
                "overlap_frames": 17,
                "runtime": runtime,
            }
        ]
        * 2
    )
    assert "MP4 input" in (batch_res.items[-1].error or "")


def test_pad_and_run_wan_sliding_windows_crossfade_long_clips_smoothly():
    calls: list[tuple[int, int, int]] = []

    class WindowPipe:
        def __call__(self, **kwargs):
            n_frames = len(kwargs["video"])
            assert (n_frames - 1) % 4 == 0
            val = 100.0 if not calls else 200.0
            calls.append((n_frames, kwargs["height"], kwargs["width"]))
            h, w = kwargs["height"], kwargs["width"]
            out = np.full((1, n_frames, h, w, 3), val / 255.0, dtype=np.float32)
            return SimpleNamespace(frames=out)

    runtime = wan.VideoWanRuntime(
        model=wan.WAN21_MODEL_ID,
        revision=wan.WAN21_MODEL_REVISION,
        resolved_device="cpu",
        pipe=WindowPipe(),
    )
    frames = [np.zeros((30, 46, 3), dtype=np.uint8) for _ in range(30)]
    out = wan._pad_and_run_wan(
        frames,
        runtime,
        strength=0.38,
        seed=4,
        chunk_frames=21,
        overlap_frames=5,
    )
    assert out.shape == (30, 30, 46, 3)
    assert calls == [(21, 32, 48), (17, 32, 48)]  # second window has 14 frames -> padded to 17 (4*4+1)
    assert np.all(out[:16] == 100)
    overlap_vals = out[16:21, 0, 0, 0].astype(int)
    assert np.all((overlap_vals > 100) & (overlap_vals < 200))
    assert np.all(np.diff(overlap_vals) > 0)
    assert np.all(out[21:] == 200)

    with pytest.raises(ValueError, match="4k\\+1"):
        wan._pad_and_run_wan(frames, runtime, strength=0.38, seed=0, chunk_frames=20, overlap_frames=4)
    with pytest.raises(ValueError, match="overlap_frames"):
        wan._pad_and_run_wan(frames, runtime, strength=0.38, seed=0, chunk_frames=21, overlap_frames=21)


def test_auto_video_profile_routes_bytedance_and_google_across_cuda_and_cpu(tmp_path, monkeypatch):
    source = _source_mp4(tmp_path, width=64, height=48, frames=6)

    # 1. On CPU (no CUDA): ByteDance -> seedance-experimental; Google -> synthid
    monkeypatch.setattr(wan, "is_available", lambda: True)
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    defaults = {
        "device": "auto",
        "quality": "delivery",
        "noise_std": video.DEFAULT_VIDEO_SYNTHID_NOISE_STD,
        "long_side": video.DEFAULT_VIDEO_SYNTHID_LONG_SIDE,
        "fps": video.DEFAULT_VIDEO_SYNTHID_FPS,
        "batch_size": 4,
        "seed": 0,
        "model": video.DEFAULT_VIDEO_SYNTHID_VAE,
        "temporal_stabilize": True,
    }
    assert (
        video._resolve_video_invisible_profile(
            "auto",
            source,
            **defaults,
            detected_metadata={"issuer": "ByteDance", "ai_tool": "Dreamina"},
        )
        == "seedance-experimental"
    )
    assert (
        video._resolve_video_invisible_profile(
            "auto",
            source,
            **defaults,
            detected_metadata={"issuer": "Google LLC", "synthid_watermark": "present"},
        )
        == "synthid"
    )

    # 2. On CUDA: ByteDance -> wan21-seedance-experimental; Google -> wan21-synthid-experimental
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert (
        video._resolve_video_invisible_profile(
            "auto",
            source,
            **defaults,
            detected_metadata={"issuer": "BytePlus (ByteDance)"},
        )
        == "wan21-seedance-experimental"
    )
    assert (
        video._resolve_video_invisible_profile(
            "auto",
            source,
            **defaults,
            detected_metadata={"issuer": "Google LLC"},
        )
        == "wan21-synthid-experimental"
    )


def test_wan_preflight_budget_bf16_guard_and_strength_field(tmp_path, monkeypatch):
    import torch

    source = _source_mp4(tmp_path, width=64, height=48, frames=6)
    output = tmp_path / "out.mp4"

    # 1. BF16 capability check
    monkeypatch.setattr(wan, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="BF16"):
        wan.resolve_wan_device("auto")
    assert wan.can_run_wan_on_source(source, device="auto") is False

    # 2. Frame-pixel / host-RAM budget rejects oversized input BEFORE decoding frames
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    frames_decoded = []
    monkeypatch.setattr(wan, "_frames", lambda p: (frames_decoded.append(p), iter(()))[1])
    monkeypatch.setattr(
        wan,
        "validate_experimental_mp4_source",
        lambda *_a, **_k: (3840, 2160, 24.0, tuple(range(10))),
    )
    with pytest.raises(ValueError, match="frame area"):
        wan.regenerate_wan_experimental(source, output)
    assert frames_decoded == []
    assert wan.can_run_wan_on_source(source, device="auto") is False

    monkeypatch.setattr(
        wan,
        "validate_experimental_mp4_source",
        lambda *_a, **_k: (1280, 720, 24.0, tuple(range(1000))),
    )
    with pytest.raises(ValueError, match="host-memory budget"):
        wan.regenerate_wan_experimental(source, output)
    assert frames_decoded == []
    assert wan.can_run_wan_on_source(source, device="auto") is False

    # 3. VideoWanExperimentalResult exposes explicit strength alongside compatibility noise_std
    res = raiw.VideoWanExperimentalResult(
        source=source,
        output=output,
        noise_std=0.12,
        metrics=_metrics(),
        remaining_metadata={},
        quality="delivery",
        profile="wan21-seedance-experimental",
    )
    assert res.strength == pytest.approx(0.12)
    assert res.noise_std == pytest.approx(0.12)
