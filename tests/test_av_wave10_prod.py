"""Production readiness tests for Wave 10 (v0.71.0): 4K Wan 2.1, worker guardrails, progress callbacks, clean APIs."""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import remove_ai_watermarks as raiw
from remove_ai_watermarks import audio, video, video_invisible, video_seedance, video_wan

if TYPE_CHECKING:
    from pathlib import Path


def _make_mp4(path: Path, *, width: int = 64, height: int = 48, fps: int = 12, duration: float = 0.5) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_wan21_4k_guided_window_preserves_4k_resolution_and_detail() -> None:
    pytest.importorskip("torch")
    assert video_wan._wan_guide_size(1920, 1080) is None
    guide_size = video_wan._wan_guide_size(3840, 2160)
    assert guide_size is not None
    assert guide_size[0] * guide_size[1] <= video_wan.WAN21_NATIVE_MAX_PIXELS
    assert guide_size[0] % 16 == 0
    assert guide_size[1] % 16 == 0
    guide_8k = video_wan._wan_guide_size(7680, 4320)
    assert guide_8k is not None
    assert guide_8k[0] * guide_8k[1] <= video_wan.WAN21_NATIVE_MAX_PIXELS

    seen_sizes: list[tuple[int, int]] = []

    def fake_pipe(**kwargs: Any) -> Any:
        frames = kwargs["video"]
        arr = np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in frames], axis=0)
        seen_sizes.append((arr.shape[2], arr.shape[1]))
        return SimpleNamespace(frames=[arr])

    runtime = video_wan.VideoWanRuntime(
        model=video_wan.WAN21_MODEL_ID,
        revision=video_wan.WAN21_MODEL_REVISION,
        resolved_device="cpu",
        pipe=fake_pipe,
        prompt_embeds=None,
    )
    # Use a compact >1080p aspect via monkeypatched native cap to keep unit test fast
    orig_cap = video_wan.WAN21_NATIVE_MAX_PIXELS
    video_wan.WAN21_NATIVE_MAX_PIXELS = 64 * 64
    try:
        frame = np.full((128, 128, 3), 120, dtype=np.uint8)
        frame[::2, ::2] = 180
        out = video_wan._run_wan_window(
            [frame, frame, frame, frame, frame],
            runtime,
            strength=0.12,
            seed=0,
            orig_h=128,
            orig_w=128,
        )
    finally:
        video_wan.WAN21_NATIVE_MAX_PIXELS = orig_cap

    assert seen_sizes == [(64, 64)]
    assert out.shape == (5, 128, 128, 3)
    assert out.dtype == np.uint8


def test_video_all_guardrails_reject_oversized_duration_pixels_and_cpu_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _make_mp4(tmp_path / "clip.mp4", width=64, height=48, fps=12, duration=1.0)
    with pytest.raises(ValueError, match="max_frame_pixels"):
        video.remove_video_all(src, tmp_path / "out1.mp4", max_frame_pixels=1000)
    with pytest.raises(ValueError, match="max_duration_seconds"):
        video.remove_video_all(src, tmp_path / "out2.mp4", max_duration_seconds=0.2)

    # require_gpu=True refuses 512p 2D CPU fallback
    monkeypatch.setattr(video_wan, "can_run_wan_on_source", lambda *a, **k: False)
    with pytest.raises(RuntimeError, match="require_gpu"):
        video.remove_video_all(
            src,
            tmp_path / "out3.mp4",
            include_invisible=True,
            invisible_profile="synthid",
            require_gpu=True,
            device="cpu",
        )


def test_video_all_and_audio_emit_progress_callbacks_and_enforce_audio_duration_limit(tmp_path: Path) -> None:
    src = _make_mp4(tmp_path / "clip.mp4", width=64, height=48, fps=12, duration=0.5)
    events: list[tuple[str, float]] = []
    res = video.remove_video_all(
        src,
        tmp_path / "clean.mp4",
        progress_callback=lambda stage, pct: events.append((stage, pct)),
    )
    assert res.output.is_file()
    assert events[0][0] == "preflight"
    assert events[-1] == ("complete", 1.0)

    pytest.importorskip("scipy")
    wav_path = tmp_path / "tone.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000)
    with pytest.raises(ValueError, match="max_duration_seconds"):
        audio.remove_audio_invisible(
            wav_path,
            tmp_path / "tone_out.wav",
            verify=False,
            allow_unverified=True,
            max_duration_seconds=0.2,
        )

    audio_events: list[tuple[str, float]] = []
    audio.remove_audio_invisible(
        wav_path,
        tmp_path / "tone_clean.wav",
        verify=False,
        allow_unverified=True,
        method="spectral",
        progress_callback=lambda stage, pct: audio_events.append((stage, pct)),
    )
    assert audio_events[0][0] == "identify"
    assert audio_events[-1] == ("complete", 1.0)


def test_production_aliases_remove_video_seedance_and_wan_clear_experimental_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "owned.mp4"
    src.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")
    metrics = video_invisible.RegenerationMetrics(
        frames=12, fps=24.0, width=64, height=48, psnr_db=33.0, temporal_residual_ratio=1.1
    )

    def _fake_regen(_s: Any, o: Path, **_k: Any) -> Any:
        o.write_bytes(src.read_bytes())
        return metrics

    monkeypatch.setattr(video_seedance, "regenerate_seedance_experimental", _fake_regen)
    monkeypatch.setattr(video_wan, "regenerate_wan_experimental", _fake_regen)

    res_sd = raiw.remove_video_seedance(src, device="cpu")
    assert res_sd.output == tmp_path / "owned_clean.mp4"
    assert res_sd.experimental is False

    res_wan = raiw.remove_video_wan(src, tmp_path / "wan_out.mp4", profile="wan21-synthid", device="cuda")
    assert res_wan.output == tmp_path / "wan_out.mp4"
    assert res_wan.experimental is False

    res_inv = video.remove_video_invisible(src, tmp_path / "inv_sd.mp4", profile="seedance", device="cpu")
    assert res_inv.experimental is False


def test_2d_synthid_logs_warning_when_downscaling(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = _make_mp4(tmp_path / "hd.mp4", width=640, height=360, fps=24, duration=0.25)
    caplog.set_level(logging.WARNING, logger="remove_ai_watermarks.video_invisible")
    monkeypatch.setattr(
        video_invisible,
        "load_video_vae_runtime",
        lambda **_: (_ for _ in ()).throw(RuntimeError("stop_after_warning")),
    )
    with pytest.raises(RuntimeError, match="stop_after_warning"):
        video_invisible.regenerate_video_candidate(src, tmp_path / "out.mp4", fps=12.0, device="cpu")
    assert any("2D SynthID VAE fallback downscaling" in rec.message for rec in caplog.records)

    caplog.clear()
    with pytest.raises(RuntimeError, match="stop_after_warning"):
        video_invisible.regenerate_video_candidate(src, tmp_path / "native.mp4", device="cpu")
    assert not any("2D SynthID VAE fallback downscaling" in rec.message for rec in caplog.records)
